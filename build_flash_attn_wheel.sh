#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# ONE-TIME SCRIPT — run this on a RunPod GPU pod (NOT serverless) to build the
# flash-attn wheel and upload it as a GitHub Release asset.
#
# Requirements (RunPod pod setup):
#   • Image : runpod/pytorch:2.7.0-py3.10-cuda12.8.1-devel-ubuntu22.04
#             (or any image matching the Dockerfile base)
#   • GPU   : Any CUDA-capable GPU (RTX 5090 preferred, any Ampere+ works)
#   • Disk  : 10 GB container disk (wheel is ~200 MB)
#
# How to use:
#   1. Start a RunPod on-demand GPU pod with the image above
#   2. Open a terminal and run:
#        bash <(curl -fsSL https://raw.githubusercontent.com/Pruthav19/runpod-HunyuanVideo/main/build_flash_attn_wheel.sh)
#   3. At the end, the script prints a GitHub CLI command — run it to upload
#      the wheel as a Release asset.
#   4. Copy the printed download URL into FLASH_ATTN_WHEEL_URL in your
#      GitHub Actions secrets (or update the ARG default in the Dockerfile).
#   5. You never need to run this again unless you change torch/CUDA versions.
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

FLASH_ATTN_TAG="v2.7.0"
WHEEL_DIR="/tmp/flash_attn_wheels"
REPO="Pruthav19/runpod-HunyuanVideo"
RELEASE_TAG="flash-attn-wheel"

echo "================================================================"
echo " Building flash-attn ${FLASH_ATTN_TAG} wheel"
echo " torch   : $(python3 -c 'import torch; print(torch.__version__)')"
echo " CUDA    : $(python3 -c 'import torch; print(torch.version.cuda)')"
echo " Python  : $(python3 --version)"
echo "================================================================"
echo ""

# Ensure build deps are present
pip install ninja packaging --quiet

mkdir -p "${WHEEL_DIR}"

echo "🔨 Compiling (uses all CPU cores — ~8-15 min on most pods)..."
MAX_JOBS="$(nproc)" \
TORCH_CUDA_ARCH_LIST="$(python3 -c "
import torch
caps = set()
for i in range(torch.cuda.device_count()):
    cc = torch.cuda.get_device_capability(i)
    caps.add(f'{cc[0]}.{cc[1]}')
print(';'.join(sorted(caps)))
")" \
pip wheel --no-build-isolation \
    --wheel-dir "${WHEEL_DIR}" \
    "git+https://github.com/Dao-AILab/flash-attention.git@${FLASH_ATTN_TAG}"

WHEEL_FILE=$(find "${WHEEL_DIR}" -name 'flash_attn*.whl' | head -1)
echo ""
echo "✅ Wheel built: ${WHEEL_FILE}"
echo "   Size: $(du -sh "${WHEEL_FILE}" | cut -f1)"

# Quick smoke test
pip install --no-build-isolation "${WHEEL_FILE}" --quiet
python3 -c "import flash_attn; print(f'✅ flash_attn {flash_attn.__version__} imports OK')"

echo ""
echo "================================================================"
echo " NEXT STEPS — upload the wheel to GitHub Releases:"
echo "================================================================"
echo ""
echo " Option A: GitHub CLI (easiest if gh is installed on the pod)"
echo "   pip install gh || apt-get install -y gh"
echo "   gh auth login"
echo "   gh release create ${RELEASE_TAG} \\"
echo "       --repo ${REPO} \\"
echo "       --title 'flash-attn pre-built wheel (torch2.7+cu128+sm120)' \\"
echo "       --notes 'Pre-built for torch==2.7.0 CUDA 12.8 sm_120 cp310' \\"
echo "       '${WHEEL_FILE}'"
echo ""
echo " Option B: Copy the wheel locally then upload via GitHub UI"
echo "   scp pod:${WHEEL_FILE} ."
echo "   Then go to: https://github.com/${REPO}/releases/new"
echo "   Tag: ${RELEASE_TAG}"
echo "   Attach the .whl file"
echo ""
echo " After uploading, the download URL will be:"
WHEEL_BASENAME=$(basename "${WHEEL_FILE}")
echo "   https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${WHEEL_BASENAME}"
echo ""
echo " Set this URL as the FLASH_ATTN_WHEEL_URL secret in GitHub Actions,"
echo " or update the ARG default in the Dockerfile."
