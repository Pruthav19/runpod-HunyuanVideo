"""
Download all required model weights for HunyuanVideo-Avatar.
Designed to run on first boot on RunPod (fast 10 Gbps network).
Models are saved to the network volume so they persist across restarts.

Weight map (approximate sizes):
  HunyuanVideo-Avatar full snapshot  ~35 GB  (FP8 transformer + VAE + encoders)
  CodeFormer                         ~0.3 GB
  Real-ESRGAN x2plus                 ~0.07 GB
"""
import os
import sys
import subprocess

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR   = os.environ.get("MODEL_DIR", "/runpod-volume/pretrained_models")
HUNYUAN_DIR = "/app/hunyuan"

# The HunyuanVideo-Avatar checkpoint lives here inside the volume:
#   {MODEL_DIR}/hunyuan_avatar/
# And the repo expects its weights at /app/hunyuan/weights/  — we symlink on boot.
HUNYUAN_WEIGHTS_DIR = os.path.join(MODEL_DIR, "hunyuan_avatar")


# ── Helpers ────────────────────────────────────────────────────────────────────
def run(cmd: list, **kwargs):
    """Run a subprocess, raise on failure."""
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def wget(url: str, dest: str):
    run(["wget", "-q", "--timeout=300", "--tries=5", url, "-O", dest])


# ── 1. HunyuanVideo-Avatar weights (HuggingFace snapshot) ─────────────────────
def download_hunyuan_avatar():
    print("📥 Downloading HunyuanVideo-Avatar weights from HuggingFace...")
    print(f"   Target: {HUNYUAN_WEIGHTS_DIR}")

    # The repo provides two checkpoints:
    #   mp_rank_00_model_states.pt       — full precision (~55 GB), needs 80 GB GPU
    #   mp_rank_00_model_states_fp8.pt   — FP8 quantised (~18 GB), fits on 32 GB RTX 5090
    # We download the full snapshot (VAE, text encoders, etc.) but then only the
    # FP8 transformer, saving ~37 GB of download.

    if os.path.isdir(HUNYUAN_WEIGHTS_DIR) and os.path.exists(
        os.path.join(
            HUNYUAN_WEIGHTS_DIR,
            "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt",
        )
    ):
        print("   HunyuanVideo-Avatar weights already present, skipping.")
        return

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="tencent/HunyuanVideo-Avatar",
        local_dir=HUNYUAN_WEIGHTS_DIR,
        # Exclude the full-precision transformer to save ~37 GB — we only need FP8.
        ignore_patterns=[
            "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
        ],
    )
    print("✅ HunyuanVideo-Avatar weights downloaded!")


# ── 2. CodeFormer (face restoration, more identity-preserving than GFPGAN) ────
def download_codeformer():
    print("📥 Downloading CodeFormer weights...")
    cf_dir = os.path.join(MODEL_DIR, "codeformer")
    os.makedirs(cf_dir, exist_ok=True)

    cf_weights = os.path.join(cf_dir, "codeformer.pth")
    if os.path.exists(cf_weights):
        print("   CodeFormer weights already present, skipping.")
        return

    wget(
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        cf_weights,
    )

    # CodeFormer also needs the face-parsing and detection helper models from facexlib
    # These are auto-downloaded by facexlib on first use, but we trigger it here
    # so first inference doesn't stall.
    detection_dir = os.path.join(MODEL_DIR, "facexlib")
    os.makedirs(detection_dir, exist_ok=True)
    try:
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        # Dry-run to trigger facexlib auto-download into our custom path
        os.environ["FACEXLIB_CACHE"] = detection_dir
        FaceRestoreHelper(upscale_factor=1, use_parse=True, device="cpu")
    except Exception as e:
        print(f"   facexlib prefetch warning (non-fatal): {e}")

    print("✅ CodeFormer weights ready!")


# ── 3. Real-ESRGAN x2plus (background upscaler) ───────────────────────────────
def download_realesrgan():
    print("📥 Downloading Real-ESRGAN x2plus weights...")
    esrgan_dir = os.path.join(MODEL_DIR, "realesrgan")
    os.makedirs(esrgan_dir, exist_ok=True)

    esrgan_weights = os.path.join(esrgan_dir, "RealESRGAN_x2plus.pth")
    if os.path.exists(esrgan_weights):
        print("   Real-ESRGAN weights already present, skipping.")
        return

    wget(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        esrgan_weights,
    )
    print("✅ Real-ESRGAN weights ready!")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    steps = [
        ("HunyuanVideo-Avatar weights", download_hunyuan_avatar),
        ("CodeFormer",                  download_codeformer),
        ("Real-ESRGAN x2plus",          download_realesrgan),
    ]

    for name, fn in steps:
        try:
            fn()
        except Exception as exc:
            print(f"\n❌ Failed to download {name}: {exc}", file=sys.stderr)
            sys.exit(1)

    print("\n🎉 All models downloaded and ready!")
