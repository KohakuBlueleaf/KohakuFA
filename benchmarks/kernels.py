"""The attention implementations the benchmarks compare, behind one signature:

    fn(q, k, v, mode, block) -> out     q / k / v / out [B, H, S, D] fp16 or bf16

``mode`` is ``"dense"``, ``"causal"`` or ``"block"`` (block-causal with frame size
``block``). An implementation returns ``None`` from ``supports`` for modes it lacks.
FlashAttention-4 is used when ``flash_attn.cute`` is importable.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from kohakufa import attention

try:
    import cutlass
    from cutlass import cute
    from flash_attn.cute import utils as fa4_utils
    from flash_attn.cute.compute_block_sparsity import compute_block_sparsity
    from flash_attn.cute.interface import flash_attn_func as fa4_func
except ImportError:
    fa4_func = None


def block_mask(s_q, s_kv, block, device):
    """Bool [Sq, Skv], True = visible: key frame <= query frame (top-left aligned)."""
    r = torch.arange(s_q, device=device)[:, None] // block
    c = torch.arange(s_kv, device=device)[None, :] // block
    return c <= r


def kohakufa(q, k, v, mode, block):
    if mode == "causal":
        return attention(q, k, v, causal=True)
    return attention(q, k, v, block_causal=block if mode == "block" else 0)


def _sdpa(backend):
    def fn(q, k, v, mode, block):
        mask = block_mask(q.shape[-2], k.shape[-2], block, q.device) if mode == "block" else None
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=mode == "causal"
            )

    return fn


if fa4_func is not None:

    @cute.jit
    def _frame_causal(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        frame = aux_tensors[0]
        m_frame = fa4_utils.scalar_to_ssa(frame[m_idx[0]], cutlass.Int32)
        n_frame = fa4_utils.scalar_to_ssa(frame[n_idx[0]], cutlass.Int32)
        return n_frame <= m_frame


def fa4(q, k, v, mode, block):
    """FlashAttention-4 on the [B, S, H, D] storage behind the [B, H, S, D] views."""
    batch, heads, s_q, _ = q.shape
    s_kv = k.shape[-2]
    kwargs = {}
    if mode == "causal":
        kwargs["causal"] = True
    elif mode == "block":
        frame = torch.arange(max(s_q, s_kv), device=q.device, dtype=torch.int32) // block
        kwargs = {
            "mask_mod": _frame_causal,
            "aux_tensors": [frame.contiguous()],
            "block_sparse_tensors": compute_block_sparsity(
                256, 128, batch, heads, s_q, s_kv, _frame_causal, [frame], "cuda"
            ),
            "block_sparse_tensors_bwd": compute_block_sparsity(
                128, 128, batch, heads, s_q, s_kv, _frame_causal, [frame], "cuda"
            ),
        }
    out = fa4_func(*(t.transpose(1, 2) for t in (q, k, v)), **kwargs)
    out = out[0] if isinstance(out, tuple) else out
    return out.transpose(1, 2)


torch._dynamo.config.recompile_limit = 64  # one flex graph per shape and grad mode
_flex = torch.compile(flex_attention, dynamic=False)


def flex(q, k, v, mode, block):
    s_q, s_kv = q.shape[-2], k.shape[-2]
    if mode == "dense":
        return _flex(q, k, v)
    size = 1 if mode == "causal" else block
    mask = create_block_mask(
        lambda b, h, qi, ki: ki // size <= qi // size, None, None, s_q, s_kv, device=q.device
    )
    return _flex(q, k, v, block_mask=mask)


# name -> (fn, modes)
KERNELS = {
    "KohakuFA": (kohakufa, ("dense", "causal", "block")),
    "FA2 (SDPA flash)": (_sdpa(SDPBackend.FLASH_ATTENTION), ("dense", "causal")),
    "cuDNN (SDPA)": (_sdpa(SDPBackend.CUDNN_ATTENTION), ("dense", "causal", "block")),
    "mem-efficient (SDPA)": (_sdpa(SDPBackend.EFFICIENT_ATTENTION), ("dense", "causal", "block")),
    "flex": (flex, ("dense", "causal", "block")),
}
if fa4_func is not None:
    # block-causal through mask_mod + block sparsity: forward correct, backward wrong
    # (dq / dk / dv relative error 1.4 - 3.0 against fp64), so it is not compared there
    KERNELS["FA4"] = (fa4, ("dense", "causal"))
