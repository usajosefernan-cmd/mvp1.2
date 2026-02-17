# ═══════════════════════════════════════════════════════════════
# Dockerfile — VideoRestorer MVP 1.2 (RunPod Serverless)
# ═══════════════════════════════════════════════════════════════
# BUILD LIGERO: solo instala sistema + ComfyUI + nodos custom
# Los MODELOS (~24GB) se descargan al PRIMER ARRANQUE.
# Flashboot cachea el estado → arranques subsiguientes son rápidos.
# ═══════════════════════════════════════════════════════════════

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

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
# NATIVOS de ComfyUI (comfy_extras/): ReferenceLatent, FluxKontextMultiReferenceLatentMethod,
# ModelSamplingAuraFlow, CFGNorm, TextEncodeQwenImageEditPlus.
# CUSTOM: GGUF loaders, SeedVR2, WAS Node Suite (LoadImageBatch), QwenEditUtils.
RUN cd /workspace/ComfyUI/custom_nodes && \
    # GGUF para modelos cuantizados (UnetLoaderGGUF)
    git clone https://github.com/city96/ComfyUI-GGUF.git && \
    cd ComfyUI-GGUF && pip install --no-cache-dir -r requirements.txt && cd .. && \
    # SeedVR2 Video Upscaler
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git seedvr2_videoupscaler && \
    cd seedvr2_videoupscaler && pip install --no-cache-dir -r requirements.txt && cd .. && \
    # WAS Node Suite (LoadImageBatch — carga batch de frames desde directorio)
    git clone https://github.com/WASasquatch/was-node-suite-comfyui.git && \
    cd was-node-suite-comfyui && pip install --no-cache-dir -r requirements.txt && cd .. && \
    # QwenEditUtils (TextEncodeQwenImageEditPlus_lrzjason — encoding con refs)
    git clone https://github.com/lrzjason/Comfyui-QwenEditUtils.git

# ── Handler (modelos se descargan en ensure_models() al arrancar) ──
COPY handler.py /workspace/handler.py

# ── Entrypoint ──
CMD ["python", "/workspace/handler.py"]
