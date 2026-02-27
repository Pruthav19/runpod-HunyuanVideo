"""
Wrapper that boots the PyTorch distributed process group (single-GPU, NCCL/GLOO)
BEFORE sample_gpu_poor.py is imported, so parallel_attention's
torch.distributed.get_rank() call doesn't raise 'Default process group has
not been initialized'.

sample_gpu_poor.py itself never calls init_process_group anywhere — it assumes
the caller (normally a distributed launch script) has already done it.

Usage (called by handler.py):
    python3 /app/run_inference_wrapper.py <all sample_gpu_poor.py args>
"""
import os
import sys
import torch
import torch.distributed as dist


def _init_dist():
    """Initialise single-process distributed group if not already done."""
    if dist.is_initialized():
        return

    # Prefer NCCL (GPU-native), fall back to GLOO (CPU, always available)
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # These env vars are required by the 'env://' rendezvous backend.
    # Set defaults so the wrapper works even without torchrun.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=1,
        rank=0,
    )


if __name__ == "__main__":
    _init_dist()

    # Patch sys.argv so sample_gpu_poor.py sees exactly the args it expects
    # (strip this script's own name, leave everything else intact)
    script = os.path.join(os.path.dirname(__file__), "hunyuan", "hymm_sp", "sample_gpu_poor.py")
    sys.argv = [script] + sys.argv[1:]

    # Run the target script in __main__ scope so its  `if __name__ == "__main__":`
    # guard fires normally.
    import runpy
    runpy.run_path(script, run_name="__main__")
