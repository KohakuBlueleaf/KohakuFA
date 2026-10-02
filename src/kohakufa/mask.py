"""Boolean attention masks packed for the kernels: 32 keys per int32 word along each
query row (forward), and 32 queries per word along each key row (backward), padded to
whole 128-wide tiles so a tile's words never run past a row."""

import torch
from torch import Tensor

TILE = 128


def normalize(mask: Tensor, batch: int, heads: int, s_q: int, s_kv: int) -> Tensor:
    """A bool mask broadcastable to ``[B, H, Sq, Skv]`` (True = visible) as 4D with
    batch / head dimensions of size 1 or B / H."""
    if mask.dtype != torch.bool:
        raise TypeError("mask: a boolean tensor (True = the key is visible)")
    while mask.dim() < 4:
        mask = mask.unsqueeze(0)
    if (
        mask.shape[-2:] != (s_q, s_kv)
        or mask.shape[0] not in (1, batch)
        or mask.shape[1] not in (1, heads)
    ):
        raise ValueError(
            f"mask {tuple(mask.shape)} does not broadcast to {(batch, heads, s_q, s_kv)}"
        )
    return mask


def pack_rows(mask: Tensor) -> Tensor:
    """``[..., R, C]`` bool -> ``[..., R, W]`` int32, bit ``c % 32`` of word ``c // 32``
    = ``mask[..., r, c]``; ``W`` covers ``C`` rounded up to a multiple of TILE."""
    cols = mask.shape[-1]
    padded = -(-cols // TILE) * TILE
    bits = torch.zeros(*mask.shape[:-1], padded, dtype=torch.int64, device=mask.device)
    bits[..., :cols] = mask
    bits = bits.unflatten(-1, (padded // 32, 32)) << torch.arange(32, device=mask.device)
    words = bits.sum(-1)  # 0 .. 2^32 - 1
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32).contiguous()


def pack(mask: Tensor, batch: int, heads: int, s_q: int, s_kv: int) -> tuple[Tensor, Tensor]:
    """(forward words over keys ``[Bm, Hm, Sq, W]``, backward words over queries
    ``[Bm, Hm, Skv, Wq]``)."""
    mask = normalize(mask, batch, heads, s_q, s_kv)
    return pack_rows(mask), pack_rows(mask.transpose(-1, -2))
