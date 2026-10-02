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
