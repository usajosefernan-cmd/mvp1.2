"""
handler.py — RunPod Serverless Handler: Video Restorer MVP 1.2
==============================================================
API que recibe un video, detecta escenas, separa clips,
genera masters 4K, restaura cada clip, y devuelve el video final.

Modo A: SeedVR2 7B directo (batch=25)
Modo B: Qwen Reference Latent + masters → SeedVR2 3B deflicker
"""

import os
import sys
import json
import time
import shutil
import subprocess
import tempfile
import urllib.request
import urllib.error
import base64
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# RunPod SDK
try:
    import runpod
except ImportError:
    runpod = None  # Para testing local

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════

WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
MODELS_DIR = WORKSPACE / "models"
COMFYUI_DIR = WORKSPACE / "ComfyUI"

# ComfyUI API (para TODO: SeedVR2 + Qwen)
COMFYUI_API_URL = os.environ.get("COMFYUI_API_URL", "http://127.0.0.1:8188")
COMFYUI_PROCESS = None  # Referencia global al proceso de ComfyUI

# ═══════════════════════════════════════════════════════════════
# NanoBananaPro (gemini-3-pro-image-preview)
# Proveedores: LaoZhang + Apiyi (protocolo Google Nativo)
# Docs: endpoints.md
# ═══════════════════════════════════════════════════════════════
NANOBANANA_PROVIDERS = [
    {"name": "laozhang-1", "base_url": "https://api.laozhang.ai",
     "key": os.environ.get("LAOZHANG_KEY_1", "sk-aduYr9zcGnV39Vpj238041B0Af384432BeFf37C5E8F8Bf24")},
    {"name": "apiyi-1", "base_url": "https://api.apiyi.com",
     "key": os.environ.get("APIYI_KEY_1", "sk-6cV1gKNgef3AZWD5887429E3D5Cf4d67B7A4624bF786AfD1")},
    {"name": "laozhang-2", "base_url": "https://api.laozhang.ai",
     "key": os.environ.get("LAOZHANG_KEY_2", "sk-k04XRL6BJ8g5OTAo4c834a2e1b80439388D51c3a7c1cFdFa")},
    {"name": "apiyi-2", "base_url": "https://api.apiyi.com",
     "key": os.environ.get("APIYI_KEY_2", "sk-DqGDeoOHsR4B0XGsCe5e0118F97a40789bEcF48d1e654eD7")},
]
NANOBANANA_MODEL = "gemini-3-pro-image-preview"
_nanobanana_call_counter = 0  # Para rotación round-robin

# ═══════════════════════════════════════════════════════════════
# Gemini Vision (análisis de frames para prompts)
# Usa protocolo OpenAI Compatible (ambos proveedores)
# ═══════════════════════════════════════════════════════════════
GEMINI_VISION_MODEL = "gemini-2.5-flash"

# Configuración por defecto
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 25
DEFAULT_OVERLAP = 5
DEFAULT_SCENE_THRESHOLD = 27.0

# Modelos SeedVR2 (ComfyUI)
SEEDVR2_7B_MODEL = "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
SEEDVR2_3B_MODEL = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"


# ═══════════════════════════════════════════════════════════════
# PASO 1: DESCARGA Y ANÁLISIS DE ESCENAS
# ═══════════════════════════════════════════════════════════════

def download_video(url: str, dest: Path) -> Path:
    """Descarga video desde URL."""
    output_path = dest / "input.mp4"
    print(f"[DOWNLOAD] {url} → {output_path}")

    req = urllib.request.Request(url, headers={"User-Agent": "VideoRestorer/1.2"})
    with urllib.request.urlopen(req, timeout=300) as response:
        with open(output_path, "wb") as f:
            shutil.copyfileobj(response, f)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[DOWNLOAD] OK — {size_mb:.1f} MB")
    return output_path


def get_video_info(video_path: Path) -> Dict:
    """Obtiene info del video con ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-print_format", "json",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    stream = data["streams"][0]
    fps_parts = stream["r_frame_rate"].split("/")
    fps = int(fps_parts[0]) / int(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "nb_frames": int(stream.get("nb_frames", 0)),
        "duration": float(stream.get("duration", data["format"]["duration"]))
    }


def detect_scenes(video_path: Path, threshold: float = 27.0) -> List[Tuple[float, float]]:
    """
    Detecta escenas usando PySceneDetect.
    Devuelve lista de (start_time, end_time) en segundos.
    """
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except ImportError:
        print("[SCENES] PySceneDetect no disponible — tratando como clip único")
        info = get_video_info(video_path)
        return [(0.0, info["duration"])]

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        info = get_video_info(video_path)
        return [(0.0, info["duration"])]

    scenes = []
    for scene in scene_list:
        start = scene[0].get_seconds()
        end = scene[1].get_seconds()
        scenes.append((start, end))

    print(f"[SCENES] Detectadas {len(scenes)} escenas")
    for i, (s, e) in enumerate(scenes):
        print(f"  Clip {i+1}: {s:.2f}s — {e:.2f}s ({e-s:.2f}s)")

    return scenes


def split_video_into_clips(video_path: Path, scenes: List[Tuple[float, float]],
                           output_dir: Path) -> List[Path]:
    """Separa el video en clips usando ffmpeg."""
    clips = []
    for i, (start, end) in enumerate(scenes):
        clip_path = output_dir / f"clip_{i+1:03d}.mp4"
        duration = end - start

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-c", "copy",  # Sin recodificar para velocidad
            str(clip_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        clips.append(clip_path)
        print(f"[SPLIT] Clip {i+1}: {clip_path.name} ({duration:.2f}s)")

    return clips


# ═══════════════════════════════════════════════════════════════
# PASO 2: EXTRACCIÓN DE FRAMES
# ═══════════════════════════════════════════════════════════════

def extract_frames(video_path: Path, output_dir: Path) -> List[Path]:
    """Extrae todos los frames de un clip como PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "frame_%04d.png")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-qscale:v", "2",
        pattern
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    frames = sorted(output_dir.glob("frame_*.png"))
    print(f"[EXTRACT] {len(frames)} frames extraídos de {video_path.name}")
    return frames


# ═══════════════════════════════════════════════════════════════
# PASO 3: GENERACIÓN DE MASTERS (Gemini Vision + NanoBananaPro)
# ═══════════════════════════════════════════════════════════════
# Lógica extraída de la app ForensicRestore Lab:
# 1. Gemini 2.5 Flash analiza el frame (visión) → genera prompt "style-only"
# 2. El prompt se envía a NanoBananaPro junto con el frame → genera master 4K
# Modos: "fidelity" (preservar estética histórica) / "enhancement" (recrear HD)
# Opción: colorize (colorizar B&W o preservar paleta original)
# ═══════════════════════════════════════════════════════════════

def generate_prompt_gemini_vision(frame_path: Path, restoration_style: str = "fidelity",
                                   colorize: bool = False,
                                   historical_context: str = "") -> str:
    """
    Usa Gemini 2.5 Flash Vision para analizar un frame y generar un prompt
    detallado de restauración para NanoBananaPro.

    Protocolo v3 "Style-Only": describe SOLO apariencia (color, textura,
    materiales, óptica). NUNCA describe poses, geometría ni composición.
    Esto evita que la IA mueva cosas al regenerar.

    Args:
        frame_path: Ruta al frame a analizar
        restoration_style: "fidelity" (preservar estética) o "enhancement" (recrear HD)
        colorize: Si True, colorizar B&W. Si False, preservar paleta original.
        historical_context: Contexto adicional del usuario (ej: "El cartel dice HOTEL")

    Returns:
        Prompt optimizado para NanoBananaPro
    """
    if not NANOBANANA_PROVIDERS or not NANOBANANA_PROVIDERS[0].get("key"):
        print("[GEMINI] Sin API keys configuradas — usando prompt genérico")
        return _fallback_prompt(restoration_style, colorize)

    # Leer frame como base64
    with open(frame_path, "rb") as f:
        frame_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Instrucciones de modo (extraídas de la app React)
    if restoration_style == "fidelity":
        mode_instruction = (
            "RESTORATION MODE: HISTORICAL FIDELITY. "
            "Preserve film grain, softness, and analog texture. "
            "Do NOT over-sharpen. Target: period-accurate enhancement."
        )
    else:
        mode_instruction = (
            "RESTORATION MODE: TECHNICAL ENHANCEMENT. "
            "Maximize sharpness. Hallucinate realistic skin pores and fabric weave "
            "where originals are blurry. Target: 8K Cinema quality."
        )

    # Instrucciones de color
    if colorize:
        color_instruction = (
            "COLOR MODE: FULL COLORIZATION. "
            "If input is B&W/Sepia -> Apply full realistic colorization (Kodachrome target). "
            "If input is color -> Enhance. "
            "MANDATORY: Assign a specific HEX color to EVERY element described."
        )
    else:
        color_instruction = (
            "COLOR MODE: STRICT PRESERVATION. "
            "If input is B&W -> Output MUST be B&W. "
            "If input is color -> Preserve original palette. "
            "DO NOT ADD COLOR to B&W footage."
        )

    # System prompt (Protocolo v3: Style-Only)
    system_prompt = f"""ROLE: Forensic Image Analyst & Prompt Engineer.
TASK: Generate a Master Restoration Prompt for an EDIT-MODE AI (NanoBananaPro).

**CRITICAL PROTOCOL v3.0: THE "STYLE ONLY" RULE**
The AI is in EDIT MODE. It preserves what you do NOT describe.
- IF YOU DESCRIBE A POSE (e.g., "man raising hand"), the AI will RE-GENERATE the hand, often MOVING it.
- IF YOU IGNORE THE POSE, the AI will LOCK the original geometry.

YOUR JOB:
Describe ONLY the APPEARANCE (Color, Texture, Material, Lighting, Optics).
NEVER describe the POSITION, POSE, GESTURE, or SPATIAL LAYOUT.

**TEXT & WRITING:** If you see text, DO NOT say "preserve text". Instead, READ IT and say: "Render perfectly sharp text reading exactly: 'XYZ'".
**BLURRY FACES:** Do not say "fix face". Say: "Skin texture is porous, weathered, with distinct stubble. Eyes are sharp."

{mode_instruction}
CONTEXT: "{historical_context}"
{color_instruction}

**OUTPUT FORMAT (Strict, NO PREAMBLE, just the prompt):**

## RESTORATION IDENTITY
[One paragraph: Era, location, camera stock, mood.]

## EXACT TEXT TRANSCRIPTION
[If text exists: "Sign reads exactly '...'" / If none: "No text visible."]

## OPTICAL CHARACTER
[Film grain structure, lens characteristics, depth of field.]

## COLOR PROTOCOL
[EXHAUSTIVE Hex codes. Assign a color to EVERY element. Skin #hex, Hair #hex, Wall #hex.]

## SUBJECT APPEARANCE (MATERIALS ONLY - NO POSES)
[ONLY materials and biological textures.
 - Correct: "Skin is pale #F2C1AE with visible pores. Jacket is rough wool."
 - INCORRECT: "Man standing on left looking right." (DO NOT WRITE THIS)]

## SURFACE & MATERIAL CATALOG
[Material type, color, and texture of background elements.]

## DISAMBIGUATION TABLE
[Resolve visual ambiguities. e.g. "Dark spot on cheek -> Mole"]

## REPAIR INSTRUCTIONS
[Remove scratches, dust, noise.]

## CONSTRAINTS (NEGATIVE PROMPT)
"Do NOT: move subjects, change poses, morph faces, change expressions, alter spatial relationships, cartoonish look, plastic skin."
"""

    # Llamar a Gemini Vision via protocolo OpenAI Compatible
    # (LaoZhang/Apiyi — ambos idénticos, usa el primer proveedor)
    provider = NANOBANANA_PROVIDERS[0]
    vision_url = f"{provider['base_url']}/v1/chat/completions"

    payload = {
        "model": GEMINI_VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame_b64}"}}
            ]
        }],
        "max_tokens": 4000,
        "temperature": 0.2
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {provider['key']}"
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(vision_url, data=data, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))

        # Extraer texto (formato OpenAI: choices[0].message.content)
        choices = result.get("choices", [])
        if choices:
            raw_prompt = choices[0].get("message", {}).get("content", "").strip()
        else:
            print("[GEMINI] Respuesta sin choices, usando fallback")
            return _fallback_prompt(restoration_style, colorize)

        # Filtrar secciones peligrosas (protocolo v3: eliminar geometría)
        clean_prompt = _filter_safe_sections(raw_prompt)

        print(f"[GEMINI] Prompt generado ({len(clean_prompt)} chars) para {frame_path.name}")
        return clean_prompt

    except Exception as e:
        print(f"[GEMINI] Error: {e} — usando prompt genérico")
        return _fallback_prompt(restoration_style, colorize)


def _filter_safe_sections(master_prompt: str) -> str:
    """
    Filtro de secciones seguras (Protocolo v3).
    Elimina secciones que causan movimiento en la imagen.
    """
    SAFE_MARKERS = [
        "RESTORATION IDENTITY", "OPTICAL CHARACTER", "COLOR",
        "SUBJECT APPEARANCE", "SURFACE & MATERIAL CATALOG",
        "DISAMBIGUATION", "REPAIR", "CONSTRAINTS", "NEGATIVE PROMPT",
        "EXACT TEXT"
    ]
    DANGEROUS_MARKERS = [
        "STRUCTURAL FIDELITY", "SCENE MAP", "SUBJECT CATALOG",
        "DYNAMIC STATES", "GEOMETRY", "POSES", "COMPOSITION",
        "SPATIAL RELATIONSHIPS"
    ]

    lines = master_prompt.split("\n")
    safe_lines = []
    current_safe = True

    for line in lines:
        trimmed = line.strip()
        if trimmed.startswith("##"):
            upper = trimmed.upper()
            if any(m in upper for m in DANGEROUS_MARKERS):
                current_safe = False
            elif any(m in upper for m in SAFE_MARKERS):
                current_safe = True
        if current_safe:
            safe_lines.append(line)

    result = "\n".join(safe_lines).strip()
    # Si el filtro eliminó demasiado, devolver original
    return result if len(result) >= 200 else master_prompt


def _fallback_prompt(restoration_style: str, colorize: bool) -> str:
    """Prompt genérico cuando Gemini no está disponible."""
    base = (
        "Photorealistic 4K enhancement of this exact frame. "
        "Preserve the exact pose, expression, clothing, lighting, and composition. "
    )
    if restoration_style == "fidelity":
        base += (
            "Maintain period-accurate film grain and analog texture. "
            "Add subtle skin micro-detail and fabric texture. "
            "Do not over-sharpen or modernize the look."
        )
    else:
        base += (
            "Add skin pores, fabric weave, hair strands, and environmental detail. "
            "Maximize sharpness. Target 8K cinema quality."
        )

    if colorize:
        base += " Apply full realistic Kodachrome colorization."
    else:
        base += " Preserve original color palette exactly."

    return base


def generate_master_nanobanana(frame_path: Path, output_path: Path,
                                aspect_ratio: str = "16:9",
                                restoration_style: str = "fidelity",
                                colorize: bool = False,
                                historical_context: str = "") -> Path:
    """
    Genera un master 4K desde un frame:
    1. Gemini 2.5 Flash Vision analiza el frame → prompt detallado
    2. NanoBananaPro (gemini-3-pro-image-preview) genera master 4K real

    Proveedores: LaoZhang + Apiyi (protocolo Google Nativo /v1beta/)
    Auth: x-goog-api-key / Bearer
    Docs: endpoints.md
    """
    global _nanobanana_call_counter
    print(f"[MASTER] Generando master desde {frame_path.name}...")

    # PASO 1: Generar prompt con Gemini Vision
    prompt = generate_prompt_gemini_vision(
        frame_path, restoration_style, colorize, historical_context
    )
    print(f"[MASTER] Prompt Gemini: {prompt[:120]}...")

    # PASO 2: Enviar a NanoBananaPro via Google Nativo
    with open(frame_path, "rb") as f:
        frame_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Rotación round-robin entre proveedores
    provider = NANOBANANA_PROVIDERS[_nanobanana_call_counter % len(NANOBANANA_PROVIDERS)]
    _nanobanana_call_counter += 1

    # Endpoint Google Nativo (4K real)
    url = f"{provider['base_url']}/v1beta/models/{NANOBANANA_MODEL}:generateContent"

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": frame_b64}}
            ]
        }],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": "4K"
            }
        }
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": provider["key"]
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)

    # Retry con backoff (2 intentos, luego siguiente provider)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))

            # Respuesta Google Nativo: imagen en inlineData
            candidates = result.get("candidates", [])
            if not candidates:
                raise ValueError(f"Sin candidates en respuesta: {list(result.keys())}")

            for part in candidates[0]["content"]["parts"]:
                if "inlineData" in part:
                    img_data = base64.b64decode(part["inlineData"]["data"])
                    with open(output_path, "wb") as f:
                        f.write(img_data)
                    size_kb = len(img_data) / 1024
                    print(f"[MASTER] ✅ {output_path.name} ({size_kb:.0f}KB) via {provider['name']}")
                    return output_path

            raise ValueError("No se encontró inlineData en la respuesta")

        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            error_code = getattr(e, 'code', 0)
            print(f"[MASTER] Intento {attempt+1}/{max_retries} falló ({error_code}): {e}")

            if error_code in (429, 503) and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)  # 2s, 4s
                print(f"[MASTER] Esperando {wait}s (rate limit)...")
                time.sleep(wait)
                # Rotar a siguiente provider
                provider = NANOBANANA_PROVIDERS[_nanobanana_call_counter % len(NANOBANANA_PROVIDERS)]
                _nanobanana_call_counter += 1
                url = f"{provider['base_url']}/v1beta/models/{NANOBANANA_MODEL}:generateContent"
                headers["x-goog-api-key"] = provider["key"]
                req = urllib.request.Request(url, data=data, headers=headers)
                continue

            # Fallback: usar frame original como master
            print(f"[MASTER] ❌ NanoBananaPro falló. Usando frame original como master")
            shutil.copy2(frame_path, output_path)
            return output_path

    shutil.copy2(frame_path, output_path)
    return output_path


def generate_masters_for_clip(frames: List[Path], masters_dir: Path,
                               restoration_style: str = "fidelity",
                               colorize: bool = False,
                               historical_context: str = "") -> Dict[str, Path]:
    """
    Genera masters del primer y último frame de un clip.
    Usa Gemini Vision para analizar cada frame y generar prompts específicos.
    """
    masters_dir.mkdir(parents=True, exist_ok=True)

    first_frame = frames[0]
    last_frame = frames[-1]

    master_first = generate_master_nanobanana(
        first_frame, masters_dir / "master_first.png",
        restoration_style=restoration_style,
        colorize=colorize,
        historical_context=historical_context
    )
    master_last = generate_master_nanobanana(
        last_frame, masters_dir / "master_last.png",
        restoration_style=restoration_style,
        colorize=colorize,
        historical_context=historical_context
    )

    return {"first": master_first, "last": master_last}


# ═══════════════════════════════════════════════════════════════
# PASO 4A: RESTAURACIÓN — SeedVR2 DIRECTO (ComfyUI Workflow)
# ═══════════════════════════════════════════════════════════════
# Todo pasa por ComfyUI — más fácil añadir pasos futuros.
# Nodos: SeedVR2DiTModelLoader + SeedVR2VAELoader + SeedVR2VideoUpscaler
# ═══════════════════════════════════════════════════════════════

def _ensure_valid_batch_size(batch_size: int) -> int:
    """batch_size debe ser 4n+1 (1,5,9,13,17,21,25,29,33...). Ajusta si no lo es."""
    if (batch_size - 1) % 4 == 0:
        return batch_size
    adjusted = ((batch_size - 1) // 4) * 4 + 1
    if adjusted < 1:
        adjusted = 5
    print(f"[SEEDVR2] batch_size {batch_size} → {adjusted} (debe ser 4n+1)")
    return adjusted


def build_seedvr2_workflow(input_frames_dir: str, output_dir: str,
                           model: str = "7b", batch_size: int = 25,
                           temporal_overlap: int = 5, seed: int = 42) -> dict:
    """
    Construye workflow ComfyUI API JSON para SeedVR2VideoUpscaler.
    Nodos del custom node numz/ComfyUI-SeedVR2_VideoUpscaler:
    - SeedVR2DiTModelLoader → carga modelo DiT (7B/3B)
    - SeedVR2VAELoader → carga VAE
    - LoadImageBatch → carga directorio de frames
    - SeedVR2VideoUpscaler → procesa batch con consistencia temporal
    - SaveImage → guarda frames restaurados
    """
    batch_size = _ensure_valid_batch_size(batch_size)
    model_file = SEEDVR2_7B_MODEL if model == "7b" else SEEDVR2_3B_MODEL

    workflow_api = {
        # Nodo 1: Cargar modelo DiT
        "1": {
            "class_type": "SeedVR2DiTModelLoader",
            "inputs": {
                "dit_model": model_file,
                "device": "cuda:0",
                "offload_device": "none",
                "cache_model": True,
                "blocks_to_swap": 0,
                "swap_io_components": False,
                "attention_mode": "sdpa"
            }
        },
        # Nodo 2: Cargar VAE
        "2": {
            "class_type": "SeedVR2VAELoader",
            "inputs": {
                "device": "cuda:0",
                "offload_device": "none"
            }
        },
        # Nodo 3: Cargar batch de frames (directorio)
        "3": {
            "class_type": "LoadImageBatch",
            "inputs": {
                "path": input_frames_dir,
                "pattern": "*.png"
            }
        },
        # Nodo 4: SeedVR2 Video Upscaler
        "4": {
            "class_type": "SeedVR2VideoUpscaler",
            "inputs": {
                "dit_model": ["1", 0],
                "vae": ["2", 0],
                "images": ["3", 0],
                "resolution": 1080,
                "max_resolution": 0,
                "batch_size": batch_size,
                "uniform_batch_size": True,
                "temporal_overlap": temporal_overlap,
                "seed": seed,
                "color_correction": "lab",
                "input_noise_scale": 0.0,
                "latent_noise_scale": 0.0
            }
        },
        # Nodo 5: Guardar frames restaurados
        "5": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["4", 0],
                "filename_prefix": Path(output_dir).name
            }
        }
    }

    return workflow_api


def restore_seedvr2_direct(frames: List[Path], output_dir: Path,
                           seed: int = 42, batch_size: int = 25,
                           overlap: int = 5) -> List[Path]:
    """
    Opción A: SeedVR2 7B directo sobre los frames.
    Todo via ComfyUI workflow API.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = frames[0].parent
    print(f"[SEEDVR2] Restaurando {len(frames)} frames (batch={batch_size}, overlap={overlap})")

    # Asegurar ComfyUI corriendo
    start_comfyui_server()

    # Construir y enviar workflow
    workflow = build_seedvr2_workflow(
        input_frames_dir=str(frames_dir),
        output_dir=str(output_dir),
        model="7b",
        batch_size=batch_size,
        temporal_overlap=overlap,
        seed=seed
    )
    run_comfyui_prompt(workflow, timeout=1200)  # SeedVR2 7B puede tardar

    restored = sorted(output_dir.glob("*.png"))
    print(f"[SEEDVR2] {len(restored)} frames restaurados")
    return restored


# ═══════════════════════════════════════════════════════════════
# PASO 4B: RESTAURACIÓN — QWEN + SEEDVR2
# ═══════════════════════════════════════════════════════════════

def restore_qwen_plus_seedvr2(frames: List[Path], masters: Dict[str, Path],
                               output_dir: Path, seed: int = 42,
                               batch_size: int = 25, overlap: int = 5) -> List[Path]:
    """
    Opción B: Qwen Reference Latent + masters → SeedVR2 3B deflicker.
    """
    qwen_dir = output_dir / "qwen_frames"
    deflicker_dir = output_dir / "deflickered"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    deflicker_dir.mkdir(parents=True, exist_ok=True)

    total = len(frames)
    print(f"[QWEN] Fase 1: Restaurando {total} frames con Reference Latent Chaining...")

    # FASE 1: Qwen frame a frame con masters como referencia
    for i, frame_path in enumerate(frames):
        # Selección inteligente: el master más cercano va como Ref2
        progress_ratio = i / max(total - 1, 1)

        if progress_ratio <= 0.5:
            ref2 = masters["first"]   # Más cercano
            ref3 = masters["last"]    # Más lejano
        else:
            ref2 = masters["last"]    # Más cercano
            ref3 = masters["first"]   # Más lejano

        output_frame = qwen_dir / frame_path.name

        restore_single_frame_qwen(
            target_frame=frame_path,
            ref_texture=ref2,
            ref_identity=ref3,
            output_path=output_frame,
            seed=seed
        )

        if (i + 1) % 10 == 0:
            print(f"[QWEN]   {i+1}/{total} frames procesados")

    # FASE 2: SeedVR2 3B como deflicker (via ComfyUI workflow)
    print(f"[QWEN] Fase 2: Deflicker con SeedVR2 3B...")

    workflow = build_seedvr2_workflow(
        input_frames_dir=str(qwen_dir),
        output_dir=str(deflicker_dir),
        model="3b",  # 3B para deflicker (frames ya tienen detalle de Qwen)
        batch_size=batch_size,
        temporal_overlap=overlap,
        seed=seed
    )
    run_comfyui_prompt(workflow, timeout=900)

    restored = sorted(deflicker_dir.glob("*.png"))
    print(f"[QWEN+SEEDVR2] {len(restored)} frames finales")
    return restored


def restore_single_frame_qwen(target_frame: Path, ref_texture: Path,
                               ref_identity: Path, output_path: Path,
                               seed: int = 42):
    """
    Restaura un frame individual con Qwen Reference Latent Chaining.
    Usa ComfyUI HTTP API con los nodos validados.
    """
    # Asegurar que ComfyUI está corriendo
    start_comfyui_server()

    workflow_api = build_qwen_workflow(
        target_frame=str(target_frame),
        ref_texture=str(ref_texture),
        ref_identity=str(ref_identity),
        output_path=str(output_path),
        seed=seed,
        steps=40,            # Oficial HuggingFace (sin LoRA)
        true_cfg=4.0,        # Oficial HuggingFace
        guidance_scale=1.0   # Oficial HuggingFace
    )

    # Enviar a ComfyUI y esperar resultado
    run_comfyui_prompt(workflow_api, timeout=300)


# ═══════════════════════════════════════════════════════════════
# COMFYUI SERVER & API (Motor central: SeedVR2 + Qwen)
# ═══════════════════════════════════════════════════════════════
# Todo pasa por ComfyUI — workflows para SeedVR2 y Qwen.
# ═══════════════════════════════════════════════════════════════

def start_comfyui_server():
    """Arranca ComfyUI como daemon si no está corriendo."""
    global COMFYUI_PROCESS

    # Verificar si ya está corriendo
    try:
        req = urllib.request.Request(f"{COMFYUI_API_URL}/system_stats")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                print("[COMFYUI] Ya está corriendo")
                return
    except Exception:
        pass

    # Arrancar ComfyUI
    print("[COMFYUI] Arrancando servidor...")
    COMFYUI_PROCESS = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", "8188",
         "--disable-auto-launch", "--disable-metadata"],
        cwd=str(COMFYUI_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    # Esperar a que esté listo (max 120s)
    for i in range(120):
        time.sleep(1)
        try:
            req = urllib.request.Request(f"{COMFYUI_API_URL}/system_stats")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    print(f"[COMFYUI] Listo (tardó {i+1}s)")
                    return
        except Exception:
            pass

    raise RuntimeError("ComfyUI no arrancó en 120s")


def run_comfyui_prompt(workflow_api: dict, timeout: int = 600) -> dict:
    """
    Envía un workflow (formato API) a ComfyUI y espera resultado.
    1. POST /prompt → obtiene prompt_id
    2. Polling GET /history/{prompt_id} hasta completado
    """
    # Enviar prompt
    payload = json.dumps({"prompt": workflow_api}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_API_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        prompt_id = result["prompt_id"]

    print(f"[COMFYUI] Prompt enviado: {prompt_id}")

    # Polling hasta completado
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(2)
        try:
            req = urllib.request.Request(f"{COMFYUI_API_URL}/history/{prompt_id}")
            with urllib.request.urlopen(req, timeout=10) as resp:
                history = json.loads(resp.read().decode("utf-8"))

            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("completed", False):
                    print(f"[COMFYUI] Completado en {time.time()-start:.1f}s")
                    return history[prompt_id].get("outputs", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI error: {msgs}")
        except urllib.error.URLError:
            pass

    raise TimeoutError(f"ComfyUI timeout ({timeout}s) para prompt {prompt_id}")


def build_qwen_workflow(target_frame: str, ref_texture: str,
                         ref_identity: str, output_path: str,
                         seed: int = 42, steps: int = 40,
                         true_cfg: float = 4.0,
                         guidance_scale: float = 1.0) -> dict:
    """
    Construye workflow JSON (formato API) de ComfyUI para Qwen.
    Nodos validados (archivo 04: prompting_qwen.md):
    - LoadImage → frame degradado + refs
    - TextEncodeQwenEditPlus → prompt + imágenes ref
    - FluxKontextMultiReferenceLatentMethod → method: to_timestep
    - KSampler → er_sde + bong_tangent
    - VAEDecode + SaveImage
    """
    prompt_text = (
        "The reference images show the same scene at high quality. "
        "Restore this degraded frame to match the reference quality. "
        "Preserve the EXACT pose, expression, and composition. "
        "Transfer skin texture, hair detail, fabric weave, and color grading. "
        "Photorealistic, sharp focus, natural tones."
    )

    # ComfyUI API format: dict de nodos con class_type e inputs
    workflow_api = {
        # Nodo 1: Cargar frame degradado
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": target_frame}
        },
        # Nodo 2: Cargar ref textura (master cercano)
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": ref_texture}
        },
        # Nodo 3: Cargar ref identidad (master lejano)
        "3": {
            "class_type": "LoadImage",
            "inputs": {"image": ref_identity}
        },
        # Nodo 4: Cargar modelo GGUF
        "4": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": "Qwen-Image-Edit-Q4_K_M.gguf"
            }
        },
        # Nodo 5: Cargar VAE
        "5": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"}
        },
        # Nodo 6: Cargar Text Encoder
        "6": {
            "class_type": "DualCLIPLoaderGGUF",
            "inputs": {
                "clip_name1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "clip_name2": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen_image"
            }
        },
        # Nodo 7: Encode prompt con TextEncodeQwenEditPlus
        "7": {
            "class_type": "TextEncodeQwenEditPlus",
            "inputs": {
                "text": prompt_text,
                "clip": ["6", 0],
                "image": ["1", 0],         # Frame degradado
                "ref_image1": ["2", 0],    # Master textura
                "ref_image2": ["3", 0],    # Master identidad
            }
        },
        # Nodo 8: Negative prompt (espacio — oficial HuggingFace)
        "8": {
            "class_type": "TextEncodeQwenEditPlus",
            "inputs": {
                "text": " ",
                "clip": ["6", 0],
                "image": ["1", 0],
            }
        },
        # Nodo 9: VAE Encode del frame degradado → latent
        "9": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["1", 0],
                "vae": ["5", 0]
            }
        },
        # Nodo 10: Reference Latent (inject frame como referencia)
        "10": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "latent": ["9", 0]
            }
        },
        # Nodo 11: Multi Reference Method
        "11": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "method": "to_timestep",
                "reference_latent": ["10", 0]
            }
        },
        # Nodo 12: Empty Latent (target)
        "12": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1
            }
        },
        # Nodo 13: KSampler
        "13": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["7", 0],
                "negative": ["8", 0],
                "latent_image": ["12", 0],
                "seed": seed,
                "steps": steps,
                "cfg": guidance_scale,
                "sampler_name": "er_sde",
                "scheduler": "bong_tangent",
                "denoise": 1.0
            }
        },
        # Nodo 14: VAE Decode
        "14": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["13", 0],
                "vae": ["5", 0]
            }
        },
        # Nodo 15: Save Image
        "15": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["14", 0],
                "filename_prefix": Path(output_path).stem
            }
        }
    }

    return workflow_api


# ═══════════════════════════════════════════════════════════════
# PASO 5: REENSAMBLAJE
# ═══════════════════════════════════════════════════════════════

def frames_to_video(frames_dir: Path, output_path: Path,
                    fps: float, audio_source: Optional[Path] = None):
    """Reensambla frames en video con ffmpeg."""
    pattern = str(frames_dir / "frame_%04d.png")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-crf", "18",           # Calidad alta
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    print(f"[ASSEMBLE] Video sin audio: {output_path}")

    # Añadir audio original si existe
    if audio_source and audio_source.exists():
        final_path = output_path.with_name("restored_final.mp4")
        cmd_audio = [
            "ffmpeg", "-y",
            "-i", str(output_path),
            "-i", str(audio_source),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(final_path)
        ]
        subprocess.run(cmd_audio, capture_output=True, check=True)
        output_path.unlink()
        final_path.rename(output_path)
        print(f"[ASSEMBLE] Audio reintegrado")


def concat_clips(clip_paths: List[Path], output_path: Path, audio_source: Path = None):
    """Concatena múltiples clips restaurados en un video final."""
    if len(clip_paths) == 1:
        shutil.copy2(clip_paths[0], output_path)
        return

    # Crear archivo de lista para ffmpeg concat
    list_file = output_path.parent / "concat_list.txt"
    with open(list_file, "w") as f:
        for clip in clip_paths:
            f.write(f"file '{clip}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p"
    ]

    # Reintegrar audio original
    if audio_source and audio_source.exists():
        cmd.extend(["-i", str(audio_source), "-c:a", "aac", "-shortest"])

    cmd.append(str(output_path))
    subprocess.run(cmd, capture_output=True, check=True)
    list_file.unlink()
    print(f"[CONCAT] {len(clip_paths)} clips → {output_path.name}")


# ═══════════════════════════════════════════════════════════════
# HANDLER PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def handler(job):
    """
    RunPod Serverless Handler.
    Recibe un video, lo restaura, y devuelve la URL del resultado.
    """
    job_input = job["input"]
    job_id = job.get("id", "local_test")

    # Parámetros
    video_url = job_input["video_url"]
    mode = job_input.get("mode", "A").upper()
    seed = job_input.get("seed", DEFAULT_SEED)
    batch_size = job_input.get("batch_size", DEFAULT_BATCH_SIZE)
    overlap = job_input.get("overlap", DEFAULT_OVERLAP)
    scene_threshold = job_input.get("scene_threshold", DEFAULT_SCENE_THRESHOLD)

    # Opciones de restauración del usuario
    restoration_style = job_input.get("restoration_style", "fidelity")  # "fidelity" o "enhancement"
    colorize = job_input.get("colorize", False)  # True/False
    historical_context = job_input.get("context", "")  # Texto libre del usuario

    # Directorio de trabajo
    work_dir = WORKSPACE / f"job_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # ── PASO 1: Descarga ──
        print(f"\n{'='*60}")
        print(f"[JOB {job_id}] Modo: {mode} | Seed: {seed}")
        print(f"{'='*60}")

        video_path = download_video(video_url, work_dir)
        video_info = get_video_info(video_path)
        print(f"[INFO] {video_info['width']}x{video_info['height']}, "
              f"{video_info['fps']}fps, {video_info['nb_frames']} frames")

        # ── PASO 2: Detectar escenas ──
        scenes = detect_scenes(video_path, threshold=scene_threshold)

        # ── PASO 3: Separar clips ──
        clips_dir = work_dir / "clips"
        clips_dir.mkdir(exist_ok=True)

        if len(scenes) == 1:
            # Video es un solo clip — no separar
            clip_paths = [video_path]
        else:
            clip_paths = split_video_into_clips(video_path, scenes, clips_dir)

        # ── PASO 4: Procesar cada clip ──
        restored_clips = []

        for clip_idx, clip_path in enumerate(clip_paths):
            clip_name = f"clip_{clip_idx+1:03d}"
            clip_work = work_dir / clip_name
            clip_work.mkdir(exist_ok=True)

            print(f"\n[CLIP {clip_idx+1}/{len(clip_paths)}] Procesando {clip_path.name}...")

            # Extraer frames del clip
            frames_dir = clip_work / "frames"
            frames = extract_frames(clip_path, frames_dir)

            if not frames:
                print(f"[CLIP {clip_idx+1}] Sin frames, saltando")
                continue

            # Generar masters (Gemini Vision analiza → NanoBananaPro genera 4K)
            masters_dir = clip_work / "masters"
            masters = generate_masters_for_clip(
                frames, masters_dir,
                restoration_style=restoration_style,
                colorize=colorize,
                historical_context=historical_context
            )

            # Restaurar según modo
            output_dir = clip_work / "output"

            if mode == "A":
                restored_frames = restore_seedvr2_direct(
                    frames, output_dir, seed=seed,
                    batch_size=batch_size, overlap=overlap
                )
            elif mode == "B":
                restored_frames = restore_qwen_plus_seedvr2(
                    frames, masters, output_dir, seed=seed,
                    batch_size=batch_size, overlap=overlap
                )
            else:
                raise ValueError(f"Modo inválido: {mode}. Usar 'A' o 'B'")

            # Reensamblar clip restaurado
            clip_output = clip_work / f"restored_{clip_name}.mp4"
            frames_to_video(output_dir, clip_output, fps=video_info["fps"])
            restored_clips.append(clip_output)

        # ── PASO 5: Concatenar clips → video final ──
        final_video = work_dir / "restored_final.mp4"
        concat_clips(restored_clips, final_video, audio_source=video_path)

        elapsed = time.time() - start_time

        # ── Resultado (todo local en RunPod) ──
        result = {
            "video_path": str(final_video),
            "clips_detected": len(scenes),
            "frames_total": video_info["nb_frames"],
            "mode": mode,
            "restoration_style": restoration_style,
            "colorize": colorize,
            "processing_time_s": round(elapsed, 1),
            "video_info": video_info
        }

        print(f"\n{'='*60}")
        print(f"[JOB {job_id}] COMPLETADO en {elapsed:.1f}s")
        print(f"{'='*60}")

        return result

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Error en job {job_id}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return {"error": error_msg, "processing_time_s": round(elapsed, 1)}

    finally:
        # Limpieza parcial (mantener resultado, borrar intermedios)
        for subdir in ["clips"]:
            d = work_dir / subdir
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if runpod:
        print("[STARTUP] Iniciando RunPod Serverless Worker...")
        runpod.serverless.start({"handler": handler})
    else:
        # Test local
        print("[TEST LOCAL] Ejecutando con video de prueba...")
        test_job = {
            "id": "test_001",
            "input": {
                "video_url": "file:///workspace/nosfe.mp4",
                "mode": "A",
                "seed": 42
            }
        }
        result = handler(test_job)
        print(json.dumps(result, indent=2, default=str))
