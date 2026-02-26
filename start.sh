#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# HunyuanVideo-Avatar  —  RunPod Serverless Entrypoint
# GPU  : RTX 5090 (Blackwell sm_120, 32 GB VRAM)
# CUDA : 12.8
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

echo "🚀  Starting HunyuanVideo-Avatar RunPod Worker..."
echo "    GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR="${MODEL_DIR:-/runpod-volume/pretrained_models}"
HUNYUAN_WEIGHTS="${MODEL_DIR}/hunyuan_avatar"
FP8_CKPT="${HUNYUAN_WEIGHTS}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
DONE_FLAG="${MODEL_DIR}/.hunyuan_avatar_download_complete"

# ── Symlink so the repo's relative weight paths resolve correctly ───────────────
# hymm_sp/sample_gpu_poor.py expects weights at ./weights/ relative to /app/hunyuan
ln -sfn "${HUNYUAN_WEIGHTS}" /app/hunyuan/weights
echo "🔗  Symlinked /app/hunyuan/weights → ${HUNYUAN_WEIGHTS}"

# ── Flash Attention 2 (built on first boot on the actual GPU) ─────────────────
# flash-attn must be compiled with CUDA — cannot be done on the GitHub Actions
# runner (2 CPU / 7 GB RAM). We build it here on the RTX 5090, cache the wheel
# to the network volume, and on every subsequent boot reinstall from the cache.
FLASH_ATTN_WHEEL_DIR="${MODEL_DIR}/.flash_attn_wheel"
FLASH_ATTN_TAG="v2.7.0"

if python3 -c "import flash_attn" 2>/dev/null; then
    echo "✅  flash-attn already installed — skipping build"
else
    CACHED_WHEEL=$(find "${FLASH_ATTN_WHEEL_DIR}" -name 'flash_attn*.whl' 2>/dev/null | head -1 || true)
    if [ -n "${CACHED_WHEEL}" ]; then
        echo "📦  Installing cached flash-attn wheel: ${CACHED_WHEEL}"
        pip install --no-build-isolation "${CACHED_WHEEL}"
    else
        echo ""
        echo "🔨  Building flash-attn ${FLASH_ATTN_TAG} from source on RTX 5090..."
        echo "    (one-time ~8 min build, wheel cached to network volume afterwards)"
        mkdir -p "${FLASH_ATTN_WHEEL_DIR}"
        MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="12.0" \
        pip wheel --no-build-isolation \
            --wheel-dir "${FLASH_ATTN_WHEEL_DIR}" \
            "git+https://github.com/Dao-AILab/flash-attention.git@${FLASH_ATTN_TAG}"
        BUILT_WHEEL=$(find "${FLASH_ATTN_WHEEL_DIR}" -name 'flash_attn*.whl' | head -1)
        pip install --no-build-isolation "${BUILT_WHEEL}"
        echo "✅  flash-attn built and cached at ${BUILT_WHEEL}"
    fi
fi

# ── Download models on first boot ─────────────────────────────────────────────
if [ ! -f "${DONE_FLAG}" ] || [ ! -f "${FP8_CKPT}" ]; then
    echo ""
    echo "📥  First boot — downloading model weights (~35 GB total)"
    echo "    Breakdown:"
    echo "      HunyuanVideo-Avatar FP8 transformer  ~18 GB"
    echo "      VAE + text encoders (LLaVA / CLIP)   ~14 GB"
    echo "      CodeFormer                            ~0.3 GB"
    echo "      Real-ESRGAN x2plus                   ~0.1 GB"
    echo "    Estimated time on RunPod 10 Gbps: ~4–8 min"
    echo ""

    python3 /app/download_models.py

    touch "${DONE_FLAG}"
    echo "✅  Models downloaded and cached at ${MODEL_DIR}"
else
    echo "✅  Model weights already cached — skipping download"
fi

# ── Verify critical checkpoint exists ─────────────────────────────────────────
if [ ! -f "${FP8_CKPT}" ]; then
    echo "❌  FP8 checkpoint missing: ${FP8_CKPT}" >&2
    echo "    Re-running download..." >&2
    python3 /app/download_models.py
fi

# ── Environment ───────────────────────────────────────────────────────────────
export PYTHONPATH="/app/hunyuan:${PYTHONPATH:-}"
export MODEL_DIR="${MODEL_DIR}"
export MODEL_BASE="${HUNYUAN_WEIGHTS}"
export INSIGHTFACE_HOME="${HUNYUAN_WEIGHTS}/ckpts"

# Blackwell sm_120 CUDA tuning
export CUDA_VISIBLE_DEVICES="0"
export TORCH_CUDA_ARCH_LIST="12.0"         # compile CUDA kernels only for sm_120
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # PyTorch 2.4+ allocator — reduces fragmentation on large VRAM GPUs

echo ""
echo "🎬  Starting serverless handler..."
exec python3 -u /app/handler.py
