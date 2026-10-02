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


def tile_lists(mask: Tensor, batch: int, heads: int, rows: int, cols: int):
    """Tile skipping for a normalized mask ``[Bm, Hm, Sq, Skv]``: for every (batch * head,
    query tile of ``rows``) the key tiles of ``cols`` it must visit, as CSR (start int32
    [B * H * nq], count, entries int32) plus per entry a class (bit h: the h-th 128-row
    part of the query tile sees every key of it, so no mask is needed). Hidden tiles are
    left out; a query tile that sees nothing keeps one (fully masked) entry, so every
    tile runs at least one step."""
    bm, hm, s_q, s_kv = mask.shape
    nq, nk = -(-s_q // rows), -(-s_kv // cols)
    parts = rows // TILE
    visible = torch.zeros(bm, hm, nq * rows, nk * cols, dtype=torch.bool, device=mask.device)
    visible[:, :, :s_q, :s_kv] = mask
    full = torch.ones_like(visible)  # rows past Sq do not spoil a full tile; keys past Skv do
    full[:, :, :, s_kv:] = False
    full[:, :, :s_q, :s_kv] = mask
    tiles = visible.view(bm, hm, nq, rows, nk, cols)
    any_ = tiles.any(dim=5).any(dim=3)  # [Bm, Hm, nq, nk]
    empty = ~any_.any(dim=-1, keepdim=True)
    first = torch.zeros_like(any_)
    first[..., 0] = True
    listed = any_ | (empty & first)
    whole = (
        full.view(bm, hm, nq, parts, TILE, nk, cols).all(dim=6).all(dim=4)
    )  # [.., nq, parts, nk]
    cls = sum(whole[:, :, :, p].to(torch.int32) << p for p in range(parts))  # [Bm, Hm, nq, nk]
    # expand the broadcast slices to every (batch, head), then CSR over the listed tiles
    listed = listed.expand(batch, heads, nq, nk).reshape(batch * heads * nq, nk)
    cls = cls.expand(batch, heads, nq, nk).reshape(batch * heads * nq, nk)
    count = listed.sum(dim=1).to(torch.int32)
    start = (count.cumsum(0) - count).to(torch.int32)
    entries = listed.nonzero()[:, 1].to(torch.int32).contiguous()
    classes = cls[listed].to(torch.int32).contiguous()
    return start.contiguous(), count.contiguous(), entries, classes, nq


class PackedMask:
    """A boolean mask packed once for the kernels (words, transposed words, tile lists),
    reusable across calls with the same mask and shape (e.g. every layer of a model):
    packing a large mask costs more than the attention it gates."""

    def __init__(self, mask: Tensor, batch: int, heads: int, s_q: int, s_kv: int) -> None:
        dense = normalize(mask, batch, heads, s_q, s_kv)
        self.shape = (batch, heads, s_q, s_kv)
        self.words = pack_rows(dense)
        self.words_t = pack_rows(dense.transpose(-1, -2))
        self.lists = tile_lists(dense, batch, heads, 256, TILE)[:4]
        # the backward walks query tiles per key tile: the transposed lists
        self.lists_t = tile_lists(dense.transpose(-1, -2), batch, heads, TILE, TILE)[:4]


def pack_mask(mask: Tensor, batch: int, heads: int, s_q: int, s_kv: int) -> PackedMask:
    """Pack ``mask`` (bool, broadcastable to ``[B, H, Sq, Skv]``, True = visible) once;
    pass the result as ``attention(..., mask=packed)``."""
    return PackedMask(mask, batch, heads, s_q, s_kv)
