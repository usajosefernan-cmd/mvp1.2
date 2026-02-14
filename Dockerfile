# ═══════════════════════════════════════════════════════════════
# Dockerfile — VideoRestorer MVP 1.2 (RunPod Serverless)
# ═══════════════════════════════════════════════════════════════
# TODO pasa por ComfyUI:
#   Modo A: SeedVR2 7B directo (workflow ComfyUI)
#   Modo B: Qwen Edit + SeedVR2 3B deflicker (workflows ComfyUI)
# NanoBananaPro: LaoZhang/Apiyi (protocolo Google Nativo)
# Base: RunPod PyTorch (CUDA 12.1)
# ═══════════════════════════════════════════════════════════════

FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE_DIR=/workspace

WORKDIR /workspace

# ── Sistema: ffmpeg + dependencias ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# ── Python: dependencias base ──
RUN pip install --no-cache-dir \
    runpod \
    scenedetect[opencv] \
    Pillow \
    numpy

# ── ComfyUI (motor central: SeedVR2 + Qwen) ──
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI && \
    cd /workspace/ComfyUI && \
    pip install --no-cache-dir -r requirements.txt

# ── Nodos custom ComfyUI ──
RUN cd /workspace/ComfyUI/custom_nodes && \
    # GGUF para modelos cuantizados Qwen
    git clone https://github.com/city96/ComfyUI-GGUF.git && \
    cd ComfyUI-GGUF && pip install --no-cache-dir -r requirements.txt && cd .. && \
    # QwenEditUtils (TextEncodeQwenEditPlus, pad/crop)
    git clone https://github.com/lrzjason/ComfyUI-QwenEditUtils.git && \
    # SeedVR2 Video Upscaler (nodos ComfyUI: SeedVR2DiTModelLoader + SeedVR2VideoUpscaler)
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git seedvr2_videoupscaler && \
    cd seedvr2_videoupscaler && pip install --no-cache-dir -r requirements.txt && cd ..

# ═══════════════════════════════════════════════════════════════
# MODELOS — descarga durante build (todo queda en el container)
# ═══════════════════════════════════════════════════════════════

# ── SeedVR2: 7B FP8 (Modo A — calidad principal) ──
RUN mkdir -p /workspace/ComfyUI/models/SEEDVR2 && \
    wget -q --show-progress -O /workspace/ComfyUI/models/SEEDVR2/seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors \
    "https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"

# ── SeedVR2: 3B FP8 (Modo B — deflicker) ──
RUN wget -q --show-progress -O /workspace/ComfyUI/models/SEEDVR2/seedvr2_ema_3b_fp8_e4m3fn.safetensors \
    "https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_3b_fp8_e4m3fn.safetensors"

# ── SeedVR2: VAE (compartida 3B/7B) ──
RUN wget -q --show-progress -O /workspace/ComfyUI/models/SEEDVR2/seedvr2_vae.safetensors \
    "https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_vae.safetensors"

# ── Qwen: Diffusion Model GGUF Q4_K_M (solo Modo B) ──
RUN mkdir -p /workspace/ComfyUI/models/unet && \
    wget -q --show-progress -O /workspace/ComfyUI/models/unet/Qwen-Image-Edit-Q4_K_M.gguf \
    "https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF/resolve/main/Qwen-Image-Edit-Q4_K_M.gguf"

# ── Qwen: Text Encoder (solo Modo B) ──
RUN mkdir -p /workspace/ComfyUI/models/text_encoders && \
    wget -q --show-progress -O /workspace/ComfyUI/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"

# ── Qwen: VAE (solo Modo B) ──
RUN mkdir -p /workspace/ComfyUI/models/vae && \
    wget -q --show-progress -O /workspace/ComfyUI/models/vae/qwen_image_vae.safetensors \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"

# ── Handler ──
COPY handler.py /workspace/handler.py

# ── Entrypoint ──
CMD ["python", "/workspace/handler.py"]
