"""Public entry point: argument handling and dispatch to the kernels."""

import torch
from torch import Tensor

from kohakufa.sm100.op import attention as _sm100_attention
from kohakufa.sm100.op import attention_varlen as _sm100_attention_varlen
from kohakufa.sm100.op import varlen_plan  # noqa: F401 (re-exported)

COMPUTE_DTYPES = (torch.float16, torch.bfloat16)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool = False,
    block_causal: int = 0,
    scale: float | None = None,
    mask: Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
) -> Tensor:
    """``softmax(scale q k^T) v``.

    Args:
        q: ``[B, H, Sq, D]``; k, v: ``[B, Hkv, Skv, D]`` with ``H`` a multiple of ``Hkv``
            (grouped-query attention: query head ``h`` uses K / V head ``h // (H / Hkv)``;
            ``Hkv = 1`` is multi-query). Views of ``[B, S, H, D]``
            projections are fine (any strides, contiguous last dim, 16-byte aligned).
        causal: query ``r`` sees keys ``0..r`` (top-left aligned when ``Sq != Skv``).
        block_causal: frame size ``P > 0``: query ``r`` sees keys of frames up to its
            own, ``c // P <= r // P``. ``causal`` is ``block_causal=1``.
        scale: softmax scale, ``D ** -0.5`` by default.
        mask: bool, broadcastable to ``[B, H, Sq, Skv]``, True = visible, or a
            ``pack_mask(...)`` result to reuse one packing across calls (exclusive with
            ``causal`` / ``block_causal``: fold those into the mask). A query row that
            sees no key outputs 0 and gets zero gradients.
        compute_dtype: tensor-core operand dtype; inputs of another dtype (e.g. fp32)
            are cast. Defaults to the inputs' dtype if it is one, else fp16.

    Returns ``[B, H, Sq, D]`` in the inputs' dtype (a view of ``[B, Sq, H, D]``
    storage, so ``out.transpose(1, 2).flatten(2)`` is free).
    """
    if causal and block_causal not in (0, 1):
        raise ValueError("causal is block_causal=1: give one of them")
    block = 1 if causal else int(block_causal)
    if compute_dtype is None:
        compute_dtype = q.dtype if q.dtype in COMPUTE_DTYPES else torch.float16
    if compute_dtype not in COMPUTE_DTYPES:
        raise TypeError(f"compute_dtype {compute_dtype}: one of {COMPUTE_DTYPES}")
    in_dtype = q.dtype
    q, k, v = (t.to(compute_dtype) for t in (q, k, v))
    if mask is not None and block:
        raise ValueError("give a mask or causal / block_causal, not both (fold them in)")
    out = _sm100_attention(q, k, v, scale, block, mask)
    return out if out.dtype == in_dtype else out.to(in_dtype)


def attention_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    *,
    causal: bool = False,
    block_causal: int = 0,
    scale: float | None = None,
    plan: tuple | None = None,
) -> Tensor:
    """Variable-length attention over packed sequences, each its own problem (no padding
    computed): ``q [Tq, H, D]``, ``k / v [Tk, Hkv, D]`` (fp16 / bf16), sequence ``b`` at
    rows ``cu_seqlens[b] : cu_seqlens[b + 1]`` (int32 [B + 1]). ``causal`` /
    ``block_causal`` apply within each sequence. ``plan``: ``varlen_plan(cu_seqlens_q,
    cu_seqlens_k, H, Hkv, block)``, built once per batch of lengths (it reads the lengths
    on the host), e.g. to keep the call free of host syncs inside a CUDA graph."""
    if causal and block_causal not in (0, 1):
        raise ValueError("causal is block_causal=1: give one of them")
    block = 1 if causal else int(block_causal)
    return _sm100_attention_varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_k, scale=scale, block_causal=block, plan=plan
    )
