"""
RunPod Serverless Handler — HunyuanVideo-Avatar
Accepts: avatar image + (text OR audio)  →  Returns: video URL

Key differences from the Hallo2 handler:
  • Inference script : hymm_sp/sample_gpu_poor.py  (single-GPU FP8 mode)
  • Input format     : CSV file (ref_image_path, audio_path, [ref_emotion_img_path])
  • No per-job YAML config — all settings are CLI flags
  • Resolution       : 704×1280 (portrait) — matches training distribution
  • Face restoration : CodeFormer (fidelity_weight=0.7) instead of GFPGAN
  • Upscaling        : Real-ESRGAN x2plus  instead of lanczos
"""

import os
import csv
import glob
import uuid
import asyncio
import subprocess
import logging

import boto3
import requests
import runpod

# ── Configuration ──────────────────────────────────────────────────────────────
WORKSPACE    = "/tmp/workspace"
MODEL_DIR    = os.environ.get("MODEL_DIR", "/runpod-volume/pretrained_models")
HUNYUAN_DIR  = "/app/hunyuan"
S3_BUCKET    = os.environ.get("S3_BUCKET",    "your-bucket-name")
S3_REGION    = os.environ.get("S3_REGION",    "us-east-1")
S3_ACCESS_KEY= os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY= os.environ.get("S3_SECRET_KEY", "")
S3_ENDPOINT  = os.environ.get("S3_ENDPOINT",   None)   # Cloudflare R2 / MinIO

# FP8 checkpoint path (fits in 32 GB VRAM of RTX 5090)
CKPT_PATH = os.path.join(
    MODEL_DIR,
    "hunyuan_avatar",
    "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── S3 / R2 ───────────────────────────────────────────────────────────────────
def get_s3():
    kw = dict(
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
    )
    if S3_ENDPOINT:
        kw["endpoint_url"] = S3_ENDPOINT
    return boto3.client("s3", **kw)


def upload_to_s3(local_path: str, s3_key: str) -> str:
    s3 = get_s3()
    s3.upload_file(local_path, S3_BUCKET, s3_key,
                   ExtraArgs={"ContentType": "video/mp4"})
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=3600,
    )
    log.info(f"Uploaded → {s3_key}")
    return url


# ── File helpers ───────────────────────────────────────────────────────────────
def download_file(url: str, dest: str) -> str:
    log.info(f"Downloading: {url}")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return dest


def generate_tts(text: str, voice: str, out: str) -> str:
    import edge_tts

    async def _run():
        await edge_tts.Communicate(text, voice).save(out)

    asyncio.run(_run())
    return out


def convert_to_wav(src: str, dst: str) -> str:
    """Re-encode any audio to 16 kHz mono WAV (required by HunyuanVideo-Avatar)."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", src,
         "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", dst],
        check=True, capture_output=True,
    )
    return dst


def get_audio_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def get_video_fps(path: str) -> float:
    """Return the frame-rate of a video as a float."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    raw = r.stdout.strip() or "25"
    from fractions import Fraction
    return float(Fraction(raw))


# ── Avatar image pre-processing ───────────────────────────────────────────────
def preprocess_avatar(src: str, dst: str, target_w: int = 704, target_h: int = 1280) -> str:
    """
    Prepare the avatar image for HunyuanVideo-Avatar:
      1. Fix EXIF rotation
      2. Convert to RGB
      3. Face-detect with InsightFace RetinaFace (more robust than Haar)
      4. Crop with generous torso/neck padding (portrait mode)
      5. Resize to target_w × target_h with LANCZOS
      6. Save as lossless PNG

    HunyuanVideo-Avatar is trained at 704×1280 (portrait 9:16).
    Feeding a well-cropped portrait of that ratio gives the best results.
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    img_pil = Image.open(src)
    img_pil = ImageOps.exif_transpose(img_pil).convert("RGB")
    img_np  = np.array(img_pil)
    h, w    = img_np.shape[:2]

    # ── Face detection with InsightFace RetinaFace ─────────────────────────────
    face_detected = False
    try:
        from insightface.app import FaceAnalysis
        face_dir = os.path.join(MODEL_DIR, "hunyuan_avatar", "ckpts", "face_analysis")
        fa = FaceAnalysis(name="buffalo_l", root=face_dir, providers=["CPUExecutionProvider"])
        fa.prepare(ctx_id=-1, det_size=(640, 640))
        faces = fa.get(img_np)
        if faces:
            # Pick the largest face
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            fw, fh = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            face_detected = True
            log.info(f"InsightFace found face at ({x1},{y1},{x2},{y2})")
    except Exception as e:
        log.warning(f"InsightFace unavailable during preprocess ({e}), using Haar fallback")

    if not face_detected:
        # Haar cascade fallback
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        dets   = cascade.detectMultiScale(gray, 1.1, 4, minSize=(64, 64))
        if len(dets) > 0:
            fx, fy, fw_, fh_ = max(dets, key=lambda r: r[2]*r[3])
            fw, fh = fw_, fh_
            cx, cy = fx + fw//2, fy + fh//2
            face_detected = True
            log.info(f"Haar found face at ({fx},{fy},{fw},{fh})")

    # ── Crop to 9:16 portrait centred on face ─────────────────────────────────
    aspect = target_w / target_h   # 704/1280 ≈ 0.55

    if face_detected:
        # Crop height: face should occupy roughly 20-25% of the frame height
        # (HunyuanVideo-Avatar is trained on upper-body/portrait shots)
        crop_h = min(int(fh * 5.0), h)
        crop_w = int(crop_h * aspect)

        # Centre horizontally on face, put face in upper-third vertically
        left  = max(cx - crop_w // 2, 0)
        top   = max(cy - crop_h // 4, 0)
        left  = min(left, w - crop_w)
        top   = min(top,  h - crop_h)
        right = left + crop_w
        bot   = top  + crop_h
    else:
        log.warning("No face detected — centre-crop fallback")
        if w / h > aspect:
            crop_w = int(h * aspect)
            crop_h = h
        else:
            crop_w = w
            crop_h = int(w / aspect)
        left  = (w - crop_w) // 2
        top   = (h - crop_h) // 2
        right = left + crop_w
        bot   = top  + crop_h

    cropped = img_pil.crop((left, top, right, bot))
    out     = cropped.resize((target_w, target_h), Image.LANCZOS)
    out.save(dst, format="PNG", optimize=False)
    log.info(f"Avatar preprocessed → {dst}  ({target_w}×{target_h})")
    return dst


# ── CSV input file ────────────────────────────────────────────────────────────
def write_input_csv(image_path: str, audio_path: str, csv_path: str,
                    job_id: str = "job",
                    prompt: str = "",
                    emotion_image_path: str | None = None) -> str:
    """
    HunyuanVideo-Avatar's VideoAudioTextLoaderVal (audio_dataset.py) reads a CSV
    with exactly these columns:
        videoid  — unique ID for the job (used in output filename)
        image    — absolute path to the reference portrait image
        audio    — absolute path to the 16 kHz mono WAV
        prompt   — text description (dataset prepends fixed quality tokens)
        fps      — output frame rate (25.0 matches the model's training FPS)

    NOTE: `ref_image_path`, `audio_path`, `ref_emotion_img_path` were the OLD
    (wrong) column names — the actual dataset code never read those fields.
    """
    if not prompt:
        prompt = "A person talking naturally"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["videoid", "image", "audio", "prompt", "fps"])
        writer.writerow([job_id, image_path, audio_path, prompt, "25.0"])
    return csv_path


# ── Core inference ────────────────────────────────────────────────────────────
def run_inference(
    image_path: str,
    audio_path: str,
    output_dir: str,
    job_id: str = "job",
    prompt: str = "",
    emotion_image_path: str | None = None,
    *,
    # Quality / speed knobs
    infer_steps: int   = 50,
    cfg_scale: float   = 7.5,
    image_size: int    = 704,    # width — height is fixed at 1280 internally
    n_frames: int      = 129,    # 129 frames ≈ 5.16 s at 25 fps (max per chunk)
    flow_shift: float  = 5.0,
    use_deepcache: int = 1,      # DeepCache acceleration (1=on, 0=off)
    seed: int          = 42,
) -> str:
    """
    Invoke hymm_sp/sample_gpu_poor.py (single-GPU FP8 mode).
    Returns the path to the generated mp4.

    Long audio (> 5 s) is handled automatically by the model's
    Time-aware Position Shift Fusion — it generates overlapping 129-frame
    chunks and stitches them together without seam artefacts.
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "input.csv")
    write_input_csv(image_path, audio_path, csv_path, job_id=job_id, prompt=prompt)

    cmd = [
        "python3", "/app/run_inference_wrapper.py",
        "--input",              csv_path,
        "--ckpt",               CKPT_PATH,
        "--sample-n-frames",    str(n_frames),
        "--seed",               str(seed),
        "--image-size",         str(image_size),
        "--cfg-scale",          str(cfg_scale),
        "--infer-steps",        str(infer_steps),
        "--use-deepcache",      str(use_deepcache),
        "--flow-shift-eval-video", str(flow_shift),
        "--save-path",          output_dir,
        "--use-fp8",            # FP8 quantised transformer — fits 32 GB VRAM
        # NOTE: --cpu-offload intentionally REMOVED — RTX 5090 has 32 GB VRAM;
        # the FP8 model is ~18 GB and fits without block-level CPU offloading.
        # With --cpu-offload, apply_group_offloading moves every transformer block
        # to CPU one at a time → GPU stays at 7%, CPU pins at 100%, 10–50× slower.
        #
        # NOTE: --infer-min intentionally REMOVED — it hardcodes audio_len=129
        # frames (5.16 s at 25 fps), truncating all longer audio.  n_frames is
        # now computed dynamically from the actual audio duration in handler().
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = HUNYUAN_DIR
    env["MODEL_BASE"]  = os.path.join(MODEL_DIR, "hunyuan_avatar")
    # Tell InsightFace where to look for buffalo_l
    env["INSIGHTFACE_HOME"] = os.path.join(MODEL_DIR, "hunyuan_avatar", "ckpts")
    # Required by torchrun even for single-GPU — parallel_states.py calls get_rank()
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29500")

    log.info(f"Running HunyuanVideo-Avatar: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd=HUNYUAN_DIR,
        env=env, capture_output=True, text=True,
    )

    if result.returncode != 0:
        log.error(f"Inference STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"HunyuanVideo-Avatar inference failed:\n{result.stderr[-500:]}")

    log.info("Inference complete. Searching for output mp4...")

    mp4s = glob.glob(os.path.join(output_dir, "**", "*.mp4"), recursive=True)
    if not mp4s:
        raise RuntimeError("Inference succeeded but no mp4 found in output directory.")

    # Sort by modification time — newest file is the result
    mp4s.sort(key=os.path.getmtime, reverse=True)
    log.info(f"Generated video: {mp4s[0]}")
    return mp4s[0]


# ── Post-processing ────────────────────────────────────────────────────────────
def postprocess(
    src: str,
    dst: str,
    *,
    apply_codeformer: bool   = True,
    codeformer_fidelity: float = 0.7,   # 0 = max enhancement, 1 = max fidelity
    apply_realesrgan: bool   = True,
    fade_in_sec: float       = 0.25,
) -> str:
    """
    Post-processing pipeline:
      1. Extract frames
      2. CodeFormer per-frame face restoration (fidelity_weight=0.7)
      3. Real-ESRGAN x2plus full-frame upscale (704×1280 → 1408×2560)
      4. Reassemble video with fade-in + high-quality encode (CRF 15)
      5. Add back original audio

    This replaces the Hallo2 pipeline's GFPGAN + lanczos approach.
    CodeFormer is measurably more identity-preserving than GFPGANv1.4 at
    the same blend strength.
    """
    job_dir       = os.path.dirname(dst)
    frames_raw    = os.path.join(job_dir, "frames_raw")
    frames_cf     = os.path.join(job_dir, "frames_cf")
    frames_esr    = os.path.join(job_dir, "frames_esr")
    os.makedirs(frames_raw, exist_ok=True)
    os.makedirs(frames_cf,  exist_ok=True)
    os.makedirs(frames_esr, exist_ok=True)

    fps = get_video_fps(src)
    fade_frames = max(4, round(fps * fade_in_sec))

    # 1. Extract frames ─────────────────────────────────────────────────────────
    log.info("Extracting frames...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", src,
         os.path.join(frames_raw, "frame_%05d.png")],
        check=True, capture_output=True,
    )

    frame_files = sorted(glob.glob(os.path.join(frames_raw, "*.png")))
    if not frame_files:
        log.warning("No frames extracted — skipping post-processing")
        import shutil; shutil.copy2(src, dst)
        return dst

    # 2. CodeFormer face restoration ────────────────────────────────────────────
    if apply_codeformer:
        log.info(f"Running CodeFormer (fidelity={codeformer_fidelity})...")
        cf_weights = os.path.join(MODEL_DIR, "codeformer", "codeformer.pth")
        cf_script  = os.path.join(HUNYUAN_DIR, "..", "codeformer_enhance.py")

        # CodeFormer is called as a subprocess to avoid import conflicts
        # (same pattern used for GFPGAN in the Hallo2 worker)
        cf_cmd = [
            "python3", "/app/codeformer_worker.py",
            "--input_path",   frames_raw,
            "--output_path",  frames_cf,
            "--model_path",   cf_weights,
            "--fidelity_weight", str(codeformer_fidelity),
            "--face_upsample",          # upsample faces within frame
        ]
        cf_result = subprocess.run(
            cf_cmd, capture_output=True, text=True,
        )
        if cf_result.returncode != 0:
            log.warning(f"CodeFormer failed ({cf_result.stderr[-300:]}), using raw frames")
            frames_after_cf = frames_raw
        else:
            log.info(cf_result.stdout.strip())
            frames_after_cf = frames_cf
    else:
        frames_after_cf = frames_raw

    # 3. Real-ESRGAN x2plus upscale ─────────────────────────────────────────────
    if apply_realesrgan:
        log.info("Running Real-ESRGAN x2plus upscale...")
        esr_weights = os.path.join(MODEL_DIR, "realesrgan", "RealESRGAN_x2plus.pth")
        esr_cmd = [
            "python3", "/app/realesrgan_worker.py",
            "--input_path",  frames_after_cf,
            "--output_path", frames_esr,
            "--model_path",  esr_weights,
            "--outscale",    "2",
            "--fp32",        # safer on Blackwell until fp16 rounding issues ironed out
        ]
        esr_result = subprocess.run(esr_cmd, capture_output=True, text=True)
        if esr_result.returncode != 0:
            log.warning(f"Real-ESRGAN failed ({esr_result.stderr[-300:]}), using frames without upscale")
            frames_final = frames_after_cf
        else:
            frames_final = frames_esr
    else:
        frames_final = frames_after_cf

    # 4 + 5. Reassemble + add audio ─────────────────────────────────────────────
    log.info("Reassembling video with audio...")
    vf = f"fade=type=in:start_frame=0:nb_frames={fade_frames},deflicker=mode=pm:size=5"

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_final, "frame_%05d.png"),
            "-i", src,                   # source audio track
            "-map", "0:v", "-map", "1:a",
            "-vf", vf,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "15", "-preset", "slow",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            dst,
        ],
        check=True, capture_output=True,
    )

    log.info(f"Post-processed video → {dst}")
    return dst


# ── RunPod handler ─────────────────────────────────────────────────────────────
def handler(event: dict) -> dict:
    """
    RunPod Serverless Handler — HunyuanVideo-Avatar

    Input schema:
    {
      "input": {
        "avatar_image_url"   : "https://...",  # REQUIRED
        "audio_url"          : "https://...",  # provide audio_url OR text
        "text"               : "Hello ...",    # provide audio_url OR text
        "voice"              : "en-US-JennyNeural",  # TTS voice (default)

        "emotion_image_url"  : "https://...",  # OPTIONAL – emotion reference image
                                               # drives Audio Emotion Module (AEM)

        # ── Inference knobs ──────────────────────────────────────────────────
        "infer_steps"     : 50,     # diffusion steps (default 50, max ~60)
        "cfg_scale"       : 7.5,    # guidance scale (7.5 recommended, 6–9 range)
        "image_size"      : 704,    # portrait width in px (height auto = 1280)
        "flow_shift"      : 5.0,    # flow-matching shift (5.0 for stable output)
        "use_deepcache"   : 1,      # DeepCache speed-up (1=on, 0=off for max quality)

        # ── Post-processing ──────────────────────────────────────────────────
        "enhance"         : true,   # CodeFormer + Real-ESRGAN (default: true)
        "codeformer_fidelity": 0.7  # 0=max enhancement, 1=max identity (default 0.7)
      }
    }
    """
    try:
        inp    = event["input"]
        job_id = str(uuid.uuid4())[:8]

        if not inp.get("avatar_image_url"):
            return {"error": "avatar_image_url is required", "status": "failed"}
        if not inp.get("audio_url") and not inp.get("text"):
            return {"error": "Provide either audio_url or text", "status": "failed"}

        job_dir = os.path.join(WORKSPACE, job_id)
        os.makedirs(job_dir, exist_ok=True)
        log.info(f"Job {job_id} started")

        # ── Step 1: Avatar image ─────────────────────────────────────────────
        ext = inp["avatar_image_url"].split(".")[-1].split("?")[0] or "jpg"
        raw_img = os.path.join(job_dir, f"avatar_raw.{ext}")
        download_file(inp["avatar_image_url"], raw_img)

        img_path = os.path.join(job_dir, "avatar.png")
        preprocess_avatar(raw_img, img_path,
                          target_w=inp.get("image_size", 704),
                          target_h=1280)

        # ── Step 1b: Optional emotion reference image ────────────────────────
        emotion_img_path = None
        if inp.get("emotion_image_url"):
            e_ext = inp["emotion_image_url"].split(".")[-1].split("?")[0] or "jpg"
            raw_emotion = os.path.join(job_dir, f"emotion_raw.{e_ext}")
            download_file(inp["emotion_image_url"], raw_emotion)
            emotion_img_path = os.path.join(job_dir, "emotion.png")
            preprocess_avatar(raw_emotion, emotion_img_path,
                              target_w=inp.get("image_size", 704),
                              target_h=1280)
            log.info("Emotion reference image ready — AEM will be used")

        # ── Step 2: Audio ────────────────────────────────────────────────────
        if inp.get("audio_url"):
            a_ext    = inp["audio_url"].split(".")[-1].split("?")[0] or "mp3"
            raw_audio= os.path.join(job_dir, f"audio_raw.{a_ext}")
            download_file(inp["audio_url"], raw_audio)
        else:
            voice     = inp.get("voice", "en-US-JennyNeural")
            raw_audio = os.path.join(job_dir, "audio_raw.mp3")
            generate_tts(inp["text"], voice, raw_audio)

        wav_path = os.path.join(job_dir, "audio.wav")
        convert_to_wav(raw_audio, wav_path)
        duration = get_audio_duration(wav_path)
        log.info(f"Audio duration: {duration:.1f}s")

        # Compute n_frames from audio duration so the generated video covers
        # the full audio clip.  Model generates at 25 fps; minimum 129 frames
        # (one TPSF chunk); cap at 257 frames (~10 s) for VRAM headroom.
        _FPS = 25.0
        n_frames = max(129, min(int(duration * _FPS + 0.5), 257))
        log.info(f"n_frames = {n_frames} ({n_frames / _FPS:.1f} s at {_FPS} fps)")

        # ── Step 3: Inference ────────────────────────────────────────────────
        infer_out_dir = os.path.join(job_dir, "infer_out")
        raw_video = run_inference(
            image_path        = img_path,
            audio_path        = wav_path,
            output_dir        = infer_out_dir,
            job_id            = job_id,
            prompt            = inp.get("prompt", ""),
            emotion_image_path= emotion_img_path,
            infer_steps       = int(inp.get("infer_steps",   50)),
            cfg_scale         = float(inp.get("cfg_scale",   7.5)),
            image_size        = int(inp.get("image_size",    704)),
            flow_shift        = float(inp.get("flow_shift",  5.0)),
            use_deepcache     = int(inp.get("use_deepcache", 1)),
            n_frames          = n_frames,
        )

        # ── Step 4: Upload raw video (safety net) ────────────────────────────
        raw_key = f"generated_videos/{job_id}_raw.mp4"
        raw_url = upload_to_s3(raw_video, raw_key)
        log.info(f"Raw video uploaded: {raw_key}")

        # ── Step 5: Post-processing ──────────────────────────────────────────
        final_url = raw_url
        enhanced_url = None

        if inp.get("enhance", True):
            try:
                final_video = os.path.join(job_dir, "output_final.mp4")
                postprocess(
                    src=raw_video,
                    dst=final_video,
                    apply_codeformer=True,
                    codeformer_fidelity=float(inp.get("codeformer_fidelity", 0.7)),
                    apply_realesrgan=True,
                )
                enh_key = f"generated_videos/{job_id}_enhanced.mp4"
                enhanced_url = upload_to_s3(final_video, enh_key)
                final_url    = enhanced_url
                log.info(f"Enhanced video uploaded: {enh_key}")
            except Exception as err:
                log.warning(f"Post-processing failed, returning raw video: {err}")

        # ── Step 6: Cleanup ──────────────────────────────────────────────────
        subprocess.run(["rm", "-rf", job_dir], capture_output=True)

        result = {
            "status":           "success",
            "job_id":           job_id,
            "video_url":        final_url,
            "raw_video_url":    raw_url,
            "duration_seconds": round(duration, 2),
        }
        if enhanced_url:
            result["enhanced_video_url"] = enhanced_url
        return result

    except Exception:
        log.exception("Handler error")
        import traceback
        return {"status": "failed", "error": traceback.format_exc()[-1000:]}


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
