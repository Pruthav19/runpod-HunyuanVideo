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
    FlashAttention 2.x CUDA kernels only ship for Ampere (SM8x), Ada (SM8.9)
    and Hopper (SM9x).  Blackwell GPUs (SM12.0 — RTX 5090, B200, etc.) crash
    with: "FlashAttention only supports Ampere GPUs or newer".

    This function detects that situation and replaces both
    ``flash_attn_varlen_func`` and ``flash_attn_func`` with pure-PyTorch
    ``F.scaled_dot_product_attention`` (SDPA) equivalents.  SDPA dispatches to
    the best available backend (cuDNN / memory-efficient / math) automatically,
    so it should work on any GPU CUDA supports.

    The patches are applied *before* any downstream model code is imported
    (via ``runpy.run_path``), so the ``from flash_attn ... import ...``
    statements in parallel_states.py / models_audio.py etc. will pick up the
    replacement functions.
    """
    if not torch.cuda.is_available():
        return

    force_flag = os.environ.get("FORCE_SDPA_FALLBACK", "0") == "1"
    major, _minor = torch.cuda.get_device_capability(0)
    print(f"[wrapper] GPU compute capability: SM {major}.{_minor}")

    # FlashAttention 2.x wheels frequently only recognise SM8x/SM9x.
    # Blackwell (SM12x) may fail even though hardware is NEWER.
    needs_patch = force_flag or major >= 10
    if not needs_patch:
        print("[wrapper] FlashAttention should work natively — no SDPA patch needed")
        return

    try:
        import math
        import torch.nn.functional as F
        import flash_attn.flash_attn_interface as fai

        # ── varlen (packed) fallback ──────────────────────────────────────
        def _sdpa_varlen_fallback(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
            *args,
            **kwargs,
        ):
            """
            Drop-in replacement for ``flash_attn_varlen_func``.

            Handles the packed variable-length layout used by
            HunyuanVideo-Avatar's ``parallel_attention`` in inference.
            q/k/v come in as ``[total_tokens, num_heads, head_dim]``.
            """
            total_q, num_heads, head_dim = q.shape

            scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(head_dim))

            batch = cu_seqlens_q.numel() - 1

            if batch == 1:
                # Fast path: single sequence — just reshape, no padding needed.
                seq_q = int((cu_seqlens_q[1] - cu_seqlens_q[0]).item())
                seq_k = int((cu_seqlens_k[1] - cu_seqlens_k[0]).item())
                q4 = q[:seq_q].view(1, seq_q, num_heads, head_dim).transpose(1, 2)
                k4 = k[:seq_k].view(1, seq_k, num_heads, head_dim).transpose(1, 2)
                v4 = v[:seq_k].view(1, seq_k, num_heads, head_dim).transpose(1, 2)

                out = F.scaled_dot_product_attention(
                    q4, k4, v4,
                    attn_mask=None,
                    dropout_p=dropout_p if q.requires_grad else 0.0,
                    is_causal=causal,
                    scale=scale,
                )
                return out.transpose(1, 2).reshape(total_q, num_heads, head_dim)

            # General path: multiple variable-length sequences.
            # Process each sequence independently to avoid allocating a giant
            # dense [B, Sq, Sk] attention mask (which can OOM on long clips).
            parts = []
            for i in range(batch):
                q_start = int(cu_seqlens_q[i].item())
                q_end = int(cu_seqlens_q[i + 1].item())
                k_start = int(cu_seqlens_k[i].item())
                k_end = int(cu_seqlens_k[i + 1].item())

                q_i = q[q_start:q_end]  # [Sq, H, D]
                k_i = k[k_start:k_end]  # [Sk, H, D]
                v_i = v[k_start:k_end]  # [Sk, H, D]

                q4 = q_i.unsqueeze(0).transpose(1, 2)  # [1, H, Sq, D]
                k4 = k_i.unsqueeze(0).transpose(1, 2)  # [1, H, Sk, D]
                v4 = v_i.unsqueeze(0).transpose(1, 2)  # [1, H, Sk, D]

                out_i = F.scaled_dot_product_attention(
                    q4,
                    k4,
                    v4,
                    attn_mask=None,
                    dropout_p=dropout_p if q.requires_grad else 0.0,
                    is_causal=causal,
                    scale=scale,
                )
                parts.append(out_i.transpose(1, 2).squeeze(0))  # [Sq, H, D]
            return torch.cat(parts, dim=0)  # [total_q, H, D]

        # ── non-varlen (standard batched) fallback ────────────────────────
        def _sdpa_func_fallback(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
            *args,
            **kwargs,
        ):
            """
            Drop-in replacement for ``flash_attn_func``.

            q/k/v shape: ``[batch, seqlen, num_heads, head_dim]``.
            Returns the same shape.
            """
            head_dim = q.shape[-1]
            scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(head_dim))

            # SDPA expects [B, H, S, D]
            q4 = q.transpose(1, 2)
            k4 = k.transpose(1, 2)
            v4 = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q4, k4, v4,
                attn_mask=None,
                dropout_p=dropout_p if q.requires_grad else 0.0,
                is_causal=causal,
                scale=scale,
            )
            return out.transpose(1, 2)  # back to [B, S, H, D]

        fai.flash_attn_varlen_func = _sdpa_varlen_fallback
        fai.flash_attn_func = _sdpa_func_fallback
        print(
            f"[wrapper] Patched flash_attn_varlen_func + flash_attn_func -> SDPA fallback "
            f"(SM {major}.{_minor}, force={force_flag})"
        )
    except Exception as exc:
        print(f"[wrapper] FlashAttention fallback patch FAILED: {exc}")
        import traceback
        traceback.print_exc()


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
