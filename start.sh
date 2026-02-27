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
# facexlib auto-downloads detection/parsing models here so they persist
# across cold starts on the network volume instead of re-downloading every time.
export FACEXLIB_CACHE="${MODEL_DIR}/facexlib"

# Blackwell sm_120 CUDA tuning
export CUDA_VISIBLE_DEVICES="0"
export TORCH_CUDA_ARCH_LIST="12.0"         # compile CUDA kernels only for sm_120
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # PyTorch 2.4+ allocator — reduces fragmentation on large VRAM GPUs

echo ""
echo "🎬  Starting serverless handler..."
exec python3 -u /app/handler.py
