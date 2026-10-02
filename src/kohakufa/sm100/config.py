"""Kernel configurations for sm_100 and their resource legality.

A configuration fixes the compile-time shape of a kernel: the head-dim chunk, the S ring,
the Q / K / V buffering. It is legal when it fits the SM: 512 tensor-memory columns and
the shared memory one CTA may use. ``forward_configs`` enumerates the legal ones for a
head dim (the tuner scores and times them); ``forward_config`` is the default pick.
"""

import dataclasses

TMEM_COLUMNS = 512  # per SM (fp32 columns of 128 lanes)
SMEM_BYTES = 227 * 1024  # per CTA, dynamic shared memory
SMEM_RESERVED = 6 * 1024  # mbarriers, softmax statistics, alignment
HALF = 128  # query rows per softmax half (two halves per CTA)
CHUNKS = (64, 32, 16)  # head-dim chunks: a 16-bit TMA row of 128 B at most (128 B swizzle)


@dataclasses.dataclass(frozen=True)
class ForwardConfig:
    """``dc`` x ``nch`` = head dim; ``block_n`` keys per K / V tile; ``s_slots`` S tiles in
    tensor memory (how far QK^T runs ahead of P V); ``q_bufs`` Q buffers per half;
    ``kv_buffers`` K / V ring depth."""

    dc: int
    nch: int
    block_n: int
    s_slots: int
    q_bufs: int
    kv_buffers: int

    @property
    def head_dim(self) -> int:
        return self.dc * self.nch

    def tmem_columns(self) -> int:
        return self.s_slots * self.block_n + 2 * self.head_dim  # S ring + O of two halves

    def smem_bytes(self, elem_bytes: int = 2) -> int:
        q = 2 * self.q_bufs * HALF * self.head_dim
        kv = self.kv_buffers * self.block_n * self.head_dim
        staging = 2 * HALF * self.dc
        return (q + kv + staging) * elem_bytes + SMEM_RESERVED

    def legal(self, elem_bytes: int = 2) -> bool:
        return (
            self.tmem_columns() <= TMEM_COLUMNS
            and self.smem_bytes(elem_bytes) <= SMEM_BYTES
            and self.s_slots >= 1
            and self.kv_buffers >= 2
        )


def chunk_of(head_dim: int) -> int:
    """The largest supported chunk dividing ``head_dim`` (a multiple of 16)."""
    if head_dim % 16:
        raise ValueError(f"kernel head dim {head_dim}: a multiple of 16")
    return next(c for c in CHUNKS if head_dim % c == 0)


def forward_configs(head_dim: int, elem_bytes: int = 2, block_n: int = 128):
    """Every legal forward configuration for ``head_dim``."""
    dc = chunk_of(head_dim)
    for s_slots in (3, 2, 1):
        for q_bufs in (2, 1):
            for kv_buffers in range(8, 1, -1):
                cfg = ForwardConfig(dc, head_dim // dc, block_n, s_slots, q_bufs, kv_buffers)
                if cfg.legal(elem_bytes):
                    yield cfg


def forward_config(head_dim: int, elem_bytes: int = 2) -> ForwardConfig:
    """Default: the deepest S ring that fits (QK^T ahead of P V), then Q double
    buffering, then the deepest K / V ring (capped at 6, which hides TMA latency)."""
    for cfg in forward_configs(head_dim, elem_bytes):
        if cfg.kv_buffers <= 6:
            return cfg
    raise ValueError(f"head dim {head_dim}: no forward configuration fits one SM")


TILE = 128  # backward: keys per CTA tile = query rows per step
REDUCE_COLS = 32  # backward: fp32 dQ columns per TMA reduce-add (staging [TILE, 32])


@dataclasses.dataclass(frozen=True)
class BackwardConfig:
    """``dc`` x ``nch`` = head dim. ``alias``: P^T shares the S^T columns and dQ the dP^T
    columns (FA4's layout; needed above D = 64, at the price of dP^T(i + 1) waiting for
    dQ(i) to be read out). ``kv_bufs`` K / V tiles (2: the next tile loads during this
    one); ``q_slots`` / ``do_slots`` Q (+ statistics) and dO buffers (the loads run that
    many steps ahead)."""

    dc: int
    nch: int
    alias: bool
    kv_bufs: int
    q_slots: int
    do_slots: int

    @property
    def head_dim(self) -> int:
        return self.dc * self.nch

    def tmem_columns(self) -> int:
        scores = 2 * TILE  # S^T, dP^T
        if self.alias:
            return scores + 2 * self.head_dim  # dV, dK (P^T in S^T, dQ in dP^T)
        return scores + TILE // 2 + 3 * self.head_dim  # P^T (16-bit), dV, dK, dQ

    def smem_bytes(self, elem_bytes: int = 2) -> int:
        tiles = (2 * self.kv_bufs + self.q_slots + self.do_slots) * TILE * self.head_dim
        ds = TILE * TILE  # dS^T
        staging = TILE * min(self.dc, REDUCE_COLS) * 4
        stats = 3 * self.q_slots * TILE * 4
        return (tiles + ds) * elem_bytes + staging + stats + SMEM_RESERVED

    def legal(self, elem_bytes: int = 2) -> bool:
        # the 3D aliased dQ view needs D <= 128 (it lives in the 128 dP^T columns)
        if self.alias and self.head_dim > TILE:
            return False
        return (
            self.tmem_columns() <= TMEM_COLUMNS
            and self.smem_bytes(elem_bytes) <= SMEM_BYTES
            and 1 <= self.do_slots <= self.q_slots
        )


def backward_configs(head_dim: int, elem_bytes: int = 2):
    """Every legal backward configuration for ``head_dim``, in default preference order:
    no aliasing, then Q at least double-buffered (the loads run ahead), double-buffered
    K / V, deeper Q, then dO buffering."""
    dc = chunk_of(head_dim)
    buffering = ((2, 3), (1, 3), (2, 2), (1, 2), (2, 1), (1, 1))  # (kv_bufs, q_slots)
    for alias in (False, True):
        for kv_bufs, q_slots in buffering:
            for do_slots in range(q_slots, 0, -1):
                cfg = BackwardConfig(dc, head_dim // dc, alias, kv_bufs, q_slots, do_slots)
                if cfg.legal(elem_bytes):
                    yield cfg


def backward_config(head_dim: int, elem_bytes: int = 2) -> BackwardConfig:
    """Default backward configuration (the first legal one, see ``backward_configs``)."""
    for cfg in backward_configs(head_dim, elem_bytes):
        return cfg
    raise ValueError(f"head dim {head_dim}: no backward configuration fits one SM")
