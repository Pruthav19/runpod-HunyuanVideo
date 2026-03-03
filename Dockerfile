# ══════════════════════════════════════════════════════════════════════════════
# HunyuanVideo-Avatar — RunPod Serverless Worker
# GPU target : NVIDIA H100 80 GB (Hopper, sm_90) — also works on A100 (sm_80)
# CUDA       : 12.8
# PyTorch    : 2.7.0
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

# ---------- PyTorch 2.7.0 + CUDA 12.8 ----------------------------------------
RUN pip install --upgrade pip && \
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128

# ---------- Clone HunyuanVideo-Avatar + CodeFormer ---------------------------
WORKDIR /app
RUN git lfs install && \
    git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git hunyuan

# CodeFormer provides basicsr.archs.codeformer_arch (not present in XPixelGroup/BasicSR)
RUN git clone --depth=1 https://github.com/sczhou/CodeFormer.git /app/codeformer

# ---------- Flash Attention 2 ------------------------------------------------
# REQUIRED — models_audio.py hard-imports flash_attn_varlen_func at startup.
#
# Strategy:
#   • If FLASH_ATTN_WHEEL_URL build-arg is set (GitHub Actions secret), the pre-built
#     wheel is downloaded in ~5 seconds — no compilation, no OOM risk.
#   • If not set (local docker build etc.), falls back to source compilation with
#     MAX_JOBS=1 NVCC_THREADS=1 (≈ 90 min, ≈1.5 GB RAM — safe on any machine).
#
# To generate the pre-built wheel, run build_flash_attn_wheel.sh once on a
# RunPod GPU pod and upload the resulting .whl to GitHub Releases.
ARG FLASH_ATTN_WHEEL_URL=""
RUN pip install ninja packaging && \
    if [ -n "${FLASH_ATTN_WHEEL_URL}" ]; then \
        echo "📦  Installing pre-built flash-attn wheel from: ${FLASH_ATTN_WHEEL_URL}" && \
        pip install --no-build-isolation "${FLASH_ATTN_WHEEL_URL}"; \
    else \
        echo "🔨  Compiling flash-attn from source for Ampere + Hopper (SM 8.0, 9.0)..." && \
        TORCH_CUDA_ARCH_LIST="8.0;9.0" MAX_JOBS=2 \
        pip install --no-build-isolation \
            git+https://github.com/Dao-AILab/flash-attention.git@v2.7.0; \
    fi

# ---------- HunyuanVideo-Avatar Python deps ----------------------------------
RUN pip install -r /app/hunyuan/requirements.txt

# diffusers older than 0.32.0 imports FLAX_WEIGHTS_NAME from transformers.utils
# which was removed in transformers>=4.52. Upgrade diffusers, then pin transformers
# back to 4.46.x: new enough for diffusers 0.32 (requires >=4.41), old enough that
# LlavaForConditionalGeneration still exposes `.language_model` (removed in 4.48+).
RUN pip install --upgrade "diffusers>=0.32.0" && \
    pip install "transformers>=4.44.0,<4.48.0"

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
COPY run_inference_wrapper.py /app/run_inference_wrapper.py
COPY start.sh              /app/start.sh

RUN chmod +x /app/start.sh

# ---------- Temp workspace ----------------------------------------------------
RUN mkdir -p /tmp/workspace

CMD ["/app/start.sh"]
