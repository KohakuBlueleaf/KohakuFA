"""The sm_100 kernels as torch custom ops with autograd: ``attention(q, k, v, scale,
block_causal)``.

    q [B, H, Sq, D], k / v [B, H, Skv, D] fp16 (views of [B, S, H, D] projections are
    fine: any strides with a contiguous last dim and 16-byte aligned strides), D = 64.
    scale: softmax scale (None -> D ** -0.5). block_causal: frame size P (query i sees
    key j iff j // P <= i // P), 0 for dense attention.
    Returns out [B, H, Sq, D] fp16, a view of [B, Sq, H, D] storage (so
    ``out.transpose(1, 2).flatten(2)`` is free).

Forward and backward are Gluon kernels (fwd.py, bwd.py) registered as torch custom ops
(``kohakufa::sm100_forward`` / ``_backward``) with autograd, so
torch.compile keeps them in the graph (no graph break) and CUDA-graph capture works:
no host synchronization, all buffers allocated by the caching allocator.

Extra memory: forward lse (row max and log2 sum, fp32 [B, H, 2, Sq]); backward
(-m, -log2 l, -delta) per query (fp32,
padded to 128 rows) and an fp32 dQ accumulator [B, H, Sq, D] while it runs.
"""

import torch
from torch import Tensor

from kohakufa.sm100.bwd import attention_backward
from kohakufa.sm100.fwd import attention_forward


def _check(q: Tensor, k: Tensor, v: Tensor) -> None:
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("attention: q, k, v must be fp16 or bf16 (all the same)")
    if q.shape[-1] != 64:
        raise ValueError("attention: head_dim must be 64")
    if k.shape != v.shape or k.shape[0] != q.shape[0] or k.shape[-1] != q.shape[-1]:
        raise ValueError(
            f"attention: k {tuple(k.shape)} / v {tuple(v.shape)} vs q {tuple(q.shape)}"
        )
    if q.shape[1] % k.shape[1]:
        raise ValueError("attention: query heads must be a multiple of K / V heads (GQA)")
    for t in (q, k, v):
        if t.stride(-1) != 1 or any((s * 2) % 16 for s in t.stride()[:-1]):
            raise ValueError("attention: inputs need a contiguous last dim and 16-byte strides")


@torch.library.custom_op("kohakufa::sm100_forward", mutates_args=())
def attention_fwd_op(
    q: Tensor, k: Tensor, v: Tensor, scale: float, block: int
) -> tuple[Tensor, Tensor]:
    return attention_forward(q, k, v, scale, block)


@attention_fwd_op.register_fake
def _(q, k, v, scale, block):
    batch, heads, seq_q, dim = q.shape
    out = q.new_empty(batch, seq_q, heads, dim).transpose(1, 2)
    lse = q.new_empty(batch, heads, 2, seq_q, dtype=torch.float32)
    return out, lse


@torch.library.custom_op("kohakufa::sm100_backward", mutates_args=())
def attention_bwd_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    dout: Tensor,
    lse: Tensor,
    scale: float,
    block: int,
) -> tuple[Tensor, Tensor, Tensor]:
    return attention_backward(q, k, v, out, dout, lse, scale, block)


@attention_bwd_op.register_fake
def _(q, k, v, out, dout, lse, scale, block):
    def like(t):
        batch, heads, seq, dim = t.shape
        return t.new_empty(batch, seq, heads, dim).transpose(1, 2)

    return like(q), like(k), like(v)


def _setup_context(ctx, inputs, output):
    q, k, v, scale, block = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.scale = scale
    ctx.block = block
    ctx.mark_non_differentiable(lse)


def _backward(ctx, dout, dlse):
    q, k, v, out, lse = ctx.saved_tensors
    if dout.stride(-1) != 1 or any((s * 2) % 16 for s in dout.stride()[:-1]):
        dout = dout.contiguous()
    dq, dk, dv = attention_bwd_op(q, k, v, out, dout, lse, ctx.scale, ctx.block)
    return dq, dk, dv, None, None


attention_fwd_op.register_autograd(_backward, setup_context=_setup_context)


def attention(
    q: Tensor, k: Tensor, v: Tensor, scale: float | None = None, block_causal: int = 0
) -> Tensor:
    """``softmax(scale q k^T) v`` over ``[B, H, S, D]``; ``block_causal > 0``: tokens of
    block ``i`` (``block_causal`` consecutive tokens) see blocks ``0..i`` only."""
    _check(q, k, v)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    out, _ = attention_fwd_op(q, k, v, float(scale), int(block_causal))
    return out
