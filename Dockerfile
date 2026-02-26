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
        ninja-build build-essential \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3    /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ---------- PyTorch 2.7.0 + CUDA 12.8 (Blackwell sm_120 support) -------------
RUN pip install --upgrade pip && \
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128

# ---------- Clone HunyuanVideo-Avatar ----------------------------------------
WORKDIR /app
RUN git lfs install && \
    git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git hunyuan

# ---------- Flash Attention 2 (compiled for sm_120) ---------------------------
# Must be built AFTER PyTorch so it links against the correct torch headers.
# Non-fatal: HunyuanVideo-Avatar falls back to PyTorch SDPA if FA2 is absent.
RUN pip install ninja && \
    MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="12.0" \
    pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.3 \
    || echo "⚠️  Flash Attention build failed — PyTorch SDPA fallback will be used"

# ---------- HunyuanVideo-Avatar Python deps ----------------------------------
RUN pip install -r /app/hunyuan/requirements.txt

# ---------- Worker deps -------------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

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
