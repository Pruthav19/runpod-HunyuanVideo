"""
Real-ESRGAN x2plus worker — run as isolated subprocess to avoid basicsr registry conflicts.

Usage (called by handler.py's postprocess()):
    python3 /app/realesrgan_worker.py
        --input_path  <dir of PNG frames>
        --output_path <dir for upscaled PNGs>
        --model_path  <path to RealESRGAN_x2plus.pth>
        --outscale    2
        [--fp32]
"""
import argparse
import glob
import os
import sys

import cv2
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path",  required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--model_path",  required=True)
    p.add_argument("--outscale",    type=float, default=2.0)
    p.add_argument("--fp32",        action="store_true",
                   help="Use FP32 instead of FP16 (safer on new GPU architectures like Blackwell)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # On Blackwell (sm_120) FP16 may produce NaNs in early driver versions — default to FP32
    half = not args.fp32 and torch.cuda.is_available()

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32, scale=2,
    )

    upsampler = RealESRGANer(
        scale=2,
        model_path=args.model_path,
        model=model,
        tile=256,       # tile to avoid OOM on high-res portrait frames
        tile_pad=10,
        pre_pad=0,
        half=half,
        device=device,
    )

    frame_files = sorted(glob.glob(os.path.join(args.input_path, "*.png")))
    print(f"Upscaling {len(frame_files)} frames with Real-ESRGAN x2plus…")

    errors = 0
    for fpath in frame_files:
        fname  = os.path.basename(fpath)
        out_fp = os.path.join(args.output_path, fname)

        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  Skipped unreadable frame: {fname}", file=sys.stderr)
            errors += 1
            continue

        try:
            out, _ = upsampler.enhance(img, outscale=args.outscale)
            cv2.imwrite(out_fp, out)
        except Exception as exc:
            print(f"  Error enhancing {fname}: {exc}", file=sys.stderr)
            cv2.imwrite(out_fp, img)   # write original on failure so video can still be assembled
            errors += 1

    print(f"✅ Real-ESRGAN done — {len(frame_files)} frames, {errors} errors")
    if errors == len(frame_files) and len(frame_files) > 0:
        sys.exit(1)   # all frames failed — signal caller to fall back


if __name__ == "__main__":
    main()
