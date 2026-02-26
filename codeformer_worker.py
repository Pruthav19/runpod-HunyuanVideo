"""
CodeFormer worker — run as isolated subprocess to avoid basicsr registry conflicts.

Usage (called by handler.py's postprocess()):
    python3 /app/codeformer_worker.py
        --input_path  <dir of PNG frames>
        --output_path <dir for enhanced PNGs>
        --model_path  <path to codeformer.pth>
        --fidelity_weight 0.7
        [--face_upsample]
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path",      required=True)
    p.add_argument("--output_path",     required=True)
    p.add_argument("--model_path",      required=True)
    p.add_argument("--fidelity_weight", type=float, default=0.7)
    p.add_argument("--face_upsample",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load CodeFormer ────────────────────────────────────────────────────────
    from basicsr.utils.download_util import load_file_from_url
    from basicsr.utils import img2tensor, tensor2img
    from basicsr.archs.codeformer_arch import CodeFormer

    net = CodeFormer(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)

    ckpt = torch.load(args.model_path, map_location=device)
    net.load_state_dict(ckpt["params_ema"] if "params_ema" in ckpt else ckpt)
    net.eval()

    # ── Load face helper (facexlib) ────────────────────────────────────────────
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper

    face_helper = FaceRestoreHelper(
        upscale_factor=1,       # upscaling handled separately by Real-ESRGAN
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        use_parse=True,
        device=device,
    )

    frame_files = sorted(glob.glob(os.path.join(args.input_path, "*.png")))
    skipped = 0

    for fpath in frame_files:
        fname  = os.path.basename(fpath)
        out_fp = os.path.join(args.output_path, fname)

        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            cv2.imwrite(out_fp, img_bgr)
            continue

        face_helper.clean_all()
        face_helper.read_image(img_bgr)
        face_helper.get_face_landmarks_5(only_center_face=True, resize=640)
        face_helper.align_warp_face()

        if not face_helper.cropped_faces:
            # No face found — copy frame unchanged
            cv2.imwrite(out_fp, img_bgr)
            skipped += 1
            continue

        restored_faces = []
        for cropped in face_helper.cropped_faces:
            t = img2tensor(cropped / 255.0, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
            with torch.no_grad():
                out = net(t, w=args.fidelity_weight, adain=True)[0]
            restored = tensor2img(out, rgb2bgr=True, min_max=(-1, 1))
            restored_faces.append(restored.astype("uint8"))

        face_helper.add_restored_face(restored_faces[0])
        face_helper.get_inverse_affine(None)
        result = face_helper.paste_faces_to_input_image(upsample_img=None)
        cv2.imwrite(out_fp, result)

    print(
        f"CodeFormer: processed {len(frame_files)} frames "
        f"({skipped} skipped — no face detected)",
        flush=True,
    )


if __name__ == "__main__":
    main()
