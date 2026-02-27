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


def _enable_flash_attn_fallback_if_needed():
    """
    Some flash-attn builds (e.g. older wheels) reject Blackwell devices at
    runtime with: "FlashAttention only supports Ampere GPUs or newer".
    For that case, patch flash_attn_varlen_func with a safe SDPA fallback.
    """
    if not torch.cuda.is_available():
        return

    force_flag = os.environ.get("FORCE_SDPA_FALLBACK", "0") == "1"
    major, _minor = torch.cuda.get_device_capability(0)

    # FlashAttention 2.x wheels frequently only recognize SM8x/SM9x.
    # Blackwell (SM12x) may fail even though hardware is newer.
    needs_patch = force_flag or major >= 10
    if not needs_patch:
        return

    try:
        import torch.nn.functional as F
        import flash_attn.flash_attn_interface as fai

        def _sdpa_varlen_fallback(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            *args,
            **kwargs,
        ):
            # Expected q/k/v shape: [total_tokens, num_heads, head_dim]
            total_q, num_heads, head_dim = q.shape
            total_k = k.shape[0]

            # Hunyuan uses a packed layout equivalent to [B * max_seqlen, H, D]
            # for these tensors in inference. Recover batch size from totals.
            bq = max(1, total_q // max_seqlen_q)
            bk = max(1, total_k // max_seqlen_k)
            batch = min(bq, bk)

            q4 = q.view(batch, max_seqlen_q, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
            k4 = k.view(batch, max_seqlen_k, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
            v4 = v.view(batch, max_seqlen_k, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

            out = F.scaled_dot_product_attention(
                q4,
                k4,
                v4,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            return out.permute(0, 2, 1, 3).contiguous().view(total_q, num_heads, head_dim)

        fai.flash_attn_varlen_func = _sdpa_varlen_fallback
        print("[wrapper] Patched flash_attn_varlen_func -> SDPA fallback")
    except Exception as exc:
        print(f"[wrapper] FlashAttention fallback patch skipped: {exc}")


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
    _enable_flash_attn_fallback_if_needed()
    _init_dist()

    # Patch sys.argv so sample_gpu_poor.py sees exactly the args it expects
    # (strip this script's own name, leave everything else intact)
    script = os.path.join(os.path.dirname(__file__), "hunyuan", "hymm_sp", "sample_gpu_poor.py")
    sys.argv = [script] + sys.argv[1:]

    # Run the target script in __main__ scope so its  `if __name__ == "__main__":`
    # guard fires normally.
    import runpy
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
