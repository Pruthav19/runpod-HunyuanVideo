# ══════════════════════════════════════════════════════════════════════════════
# HunyuanVideo-Avatar — RunPod Serverless Worker
# GPU target : NVIDIA RTX 5090 (Blackwell, sm_120, 32 GB VRAM)
# CUDA       : 12.8  ← required for Blackwell / sm_120
# PyTorch    : 2.7.0 ← first stable release with official sm_120 support
# ══════════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# ---------- System packages ---------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip python3.10-venv \
        git git-lfs wget curl ffmpeg \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
        libgomp1 \
        ninja-build build-essential \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3    /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ---------- PyTorch 2.7.0 + CUDA 12.8 (Blackwell sm_120 support) -------------
RUN pip install --upgrade pip && \
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128

# ---------- Clone HunyuanVideo-Avatar + CodeFormer ---------------------------
WORKDIR /app
RUN git lfs install && \
    git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git hunyuan

# CodeFormer provides basicsr.archs.codeformer_arch (not present in XPixelGroup/BasicSR)
RUN git clone --depth=1 https://github.com/sczhou/CodeFormer.git /app/codeformer

# ---------- Flash Attention 2 (compiled for sm_120) ---------------------------
# REQUIRED — HunyuanVideo-Avatar models_audio.py imports flash_attn_varlen_func
# unconditionally at startup. Build MUST succeed or inference will not start.
# --no-build-isolation is mandatory: flash-attn's setup.py imports torch to
# detect the CUDA version. Without this flag pip creates an isolated subprocess
# that has no torch and the build fails with "No module named 'torch'".
RUN pip install ninja packaging && \
    MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="12.0" \
    pip install --no-build-isolation \
        git+https://github.com/Dao-AILab/flash-attention.git@v2.7.0

# ---------- HunyuanVideo-Avatar Python deps ----------------------------------
RUN pip install -r /app/hunyuan/requirements.txt

# diffusers older than 0.32.0 imports FLAX_WEIGHTS_NAME from transformers.utils
# which was removed in transformers>=4.52. Force-upgrade diffusers AFTER hunyuan
# requirements so we always get a version that no longer has this import.
RUN pip install --upgrade "diffusers>=0.32.0"

# ---------- Worker deps -------------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Force-reinstall basicsr from git LAST so realesrgan/facexlib cannot overwrite
# it with the torchvision-incompatible PyPI version, then inject CodeFormer arch.
RUN pip install --force-reinstall --no-deps \
        git+https://github.com/XPixelGroup/BasicSR.git && \
    BASICSR_ARCHS="$(python3 -c 'import basicsr,os; print(os.path.join(os.path.dirname(basicsr.__file__),"archs"))')" && \
    cp /app/codeformer/basicsr/archs/codeformer_arch.py "${BASICSR_ARCHS}/"

# ---------- Copy worker files -------------------------------------------------
COPY handler.py            /app/handler.py
COPY download_models.py    /app/download_models.py
COPY codeformer_worker.py  /app/codeformer_worker.py
COPY realesrgan_worker.py  /app/realesrgan_worker.py
COPY start.sh              /app/start.sh

RUN chmod +x /app/start.sh

# ---------- Temp workspace ----------------------------------------------------
RUN mkdir -p /tmp/workspace

CMD ["/app/start.sh"]
