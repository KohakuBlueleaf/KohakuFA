"""Public entry point: argument handling and dispatch to the kernels."""

import torch
from torch import Tensor

from kohakufa.sm100.op import attention as _sm100_attention

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
        mask: bool, broadcastable to ``[B, H, Sq, Skv]``, True = visible (exclusive with
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
