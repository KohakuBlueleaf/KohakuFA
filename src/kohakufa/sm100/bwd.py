"""Attention backward as one warp-specialized Gluon kernel (Blackwell, tcgen05 + TMEM).

Given q, k, v, out, dout [B, H, S, D] fp16 and the forward's softmax statistics (row
max m of the raw scores and log2 l), with P = softmax(scale q k^T) = exp2(c (q k^T - m)
- log2 l), c = scale log2(e), and delta = rowsum(dout * out):

    dV = P^T dO,   dS = P * (dO V^T - delta),   dK = scale dS^T Q,   dQ = scale dS K

Work decomposition. A CTA tile is 128 keys of one (batch, head); it walks the query
tiles (128 rows, "steps") that see any of its keys and accumulates dK and dV in TMEM.
Everything is computed transposed (keys on the TMEM lanes): S^T = K Q^T, dP^T = V dO^T,
so P^T is the A operand of dV += P^T dO straight from TMEM, and dS^T (written once to
smem) is the A operand of dK += dS^T Q and, transposed, of dQ_i = dS K. dQ_i is a
128 x D fp32 partial per (key tile, query tile), added to an fp32 dQ accumulator in
global memory with TMA bulk reduce-add (cp.reduce.async.bulk.tensor .add).

Block-causal masking (block size P) by block sparsity: a key tile only visits query
tiles at or after the first frame that sees it (earlier ones are fully hidden: never
loaded or computed); query tiles whose first row sees the whole key tile run unmasked;
only the few query tiles on the frame boundary apply an elementwise mask (key c is
visible to query r iff r >= (c // P) * P, one division per key row and tile). Query
rows past S_q get -m = -inf (P = 0); keys past S_kv are masked (P = 0) in the last key
tile: their score is 0, and exp2(-c m - log2 l) overflows when a row's scores are all
negative (dQ += dS K would then be inf * 0).

Partitions:

    load       1 warp   TMA: K, V of the tile (two buffers: the next tile's load
                        overlaps); Q_i, dO_i and the step's (-m, -log2 l, -delta) into three
                        rotating slots (loads run two steps ahead)
    mma        1 warp   per step i, in issue order:
                          [P(i) in TMEM]        dV += P^T(i) dO_i, S^T(i+1) = K Q^T
                          [dP(i) read out]      dP^T(i+1) = V dO^T
                          [dS(i) in smem]       dK += dS^T(i) Q_i, dQ(i) = dS(i) K
    p          4 warps  P^T(i) = exp2(c (S^T - m) - log2 l) for all 128 queries ->
                        TMEM (fp16); runs a step ahead of the dS warpgroup
    ds         4 warps  dP^T(i) -> registers (frees its TMEM slot for dP^T(i+1) at
                        once), dS^T = P^T (dP^T - delta) -> smem (fp16)
    reduce     4 warps  dQ(i) TMEM -> smem -> TMA reduce-add; at the end of a tile
                        dK (scaled), dV -> fp16 -> TMA store

TMEM (512 columns): S^T 128 | dP^T 128 | P^T 64 (fp16 x 128) | dV 64 | dK 64 | dQ 64.
smem: K / V 2 x 32 KB, Q / dO 3 x 32 KB, dS^T 32 KB, staging 16 KB, stats 4.5 KB.
"""

import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._core import builtin
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    float2,
    get_tmem_reg_layout,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
    tensor_memory_descriptor,
    tma,
)
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.language.core import _aggregate as aggregate

from kohakufa.sm100.fwd import LOG2E, _join_columns, sm_count, tma_descriptor

TILE_ROWS = 128
TILE = gl.constexpr(TILE_ROWS)  # keys per CTA tile = query rows per step
QDO_SLOTS = gl.constexpr(3)  # Q / dO / stats buffers (the loads run two steps ahead)
STATS = gl.constexpr(3)  # per query row: -m, -log2 l, -delta
CHUNK = gl.constexpr(32)  # score columns a compute thread holds at once


@aggregate
class Problem:
    qk_scale: gl.tensor  # softmax scale * log2(e)
    heads: gl.tensor  # query heads
    group: gl.tensor  # query heads per K / V head (GQA; 1 = multi-head)
    seq_q: gl.tensor
    seq_kv: gl.tensor
    block: gl.tensor  # block-causal block size (unused when not CAUSAL)
    num_kv_tiles: gl.tensor
    num_tiles: gl.tensor
    HEAD_DIM: gl.constexpr
    CAUSAL: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self, qk_scale, heads, group, seq_q, seq_kv, block, num_kv_tiles, num_tiles,
        HEAD_DIM, CAUSAL,
    ):  # fmt: skip
        self.qk_scale = qk_scale
        self.heads = heads
        self.group = group
        self.seq_q = seq_q
        self.seq_kv = seq_kv
        self.block = block
        self.num_kv_tiles = num_kv_tiles
        self.num_tiles = num_tiles
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.CAUSAL = gl.constexpr(CAUSAL)

    @gluon.jit
    def key_limit(self, row):
        """Keys visible to query ``row`` (scalar or tensor): [0, key_limit)."""
        if self.CAUSAL:
            return gl.minimum((row // self.block + 1) * self.block, self.seq_kv)
        else:
            return row * 0 + self.seq_kv

    @gluon.jit
    def tile(self, tile_id):
        """CTA tile ``tile_id``: its keys and the query tiles it visits."""
        # a head's key tiles are consecutive: CTAs running together share its Q / dO
        # in L2 (block-causal: within a head, the longest key tiles come first)
        kv_tile = tile_id % self.num_kv_tiles
        batch_kv_head = tile_id // self.num_kv_tiles  # batch * kv_heads + kv_head
        kv_start = kv_tile * TILE
        num_q_tiles = gl.cdiv(self.seq_q, TILE)
        if self.CAUSAL:
            # first query row that sees key kv_start: the start of its frame
            first_q = (kv_start // self.block) * self.block // TILE
            # first query tile whose first row sees every key of the tile
            kv_end = gl.minimum(kv_start + TILE, self.seq_kv)
            full_row = (gl.cdiv(kv_end, self.block) - 1) * self.block
            first_full = gl.minimum(gl.cdiv(full_row, TILE), num_q_tiles)
            first_full = gl.maximum(first_full, first_q)
        else:
            first_q = kv_start * 0
            first_full = first_q
        kv_heads = self.heads // self.group
        return TileInfo(batch_kv_head // kv_heads, batch_kv_head % kv_heads,
                        kv_start, first_q, first_full, num_q_tiles)  # fmt: skip

    @gluon.jit
    def q_head(self, info, g):
        """Query head ``g`` (0 .. group - 1) of the tile's K / V head."""
        return info.head * self.group + g


@aggregate
class TileInfo:
    batch: gl.tensor
    head: gl.tensor  # the K / V head (query heads head * group .. + group - 1)
    kv_start: gl.tensor
    first_q: gl.tensor  # first query tile visited
    first_full: gl.tensor  # first query tile computed without a mask
    end_q: gl.tensor  # one past the last query tile

    @gluon.constexpr_function
    def __init__(self, batch, head, kv_start, first_q, first_full, end_q):
        self.batch = batch
        self.head = head
        self.kv_start = kv_start
        self.first_q = first_q
        self.first_full = first_full
        self.end_q = end_q


@aggregate
class Smem:
    k: gl.shared_memory_descriptor  # [2, 1, 1, TILE, D]: alternate tiles
    v: gl.shared_memory_descriptor  # [2, 1, 1, TILE, D]
    q: gl.shared_memory_descriptor  # [QDO_SLOTS, 1, 1, TILE, D]
    do: gl.shared_memory_descriptor  # [QDO_SLOTS, 1, 1, TILE, D]
    ds: gl.shared_memory_descriptor  # [TILE, TILE]: dS^T (keys x queries), fp16
    out: gl.shared_memory_descriptor  # [1, 1, TILE, D / 2] fp32: dQ / dK / dV staging
    stats: gl.shared_memory_descriptor  # [STATS QDO_SLOTS, TILE] fp32: -m, -log2 l, -delta

    @gluon.constexpr_function
    def __init__(self, k, v, q, do, ds, out, stats):
        self.k = k
        self.v = v
        self.q = q
        self.do = do
        self.ds = ds
        self.out = out
        self.stats = stats


@aggregate
class Bars:
    kv_ready: gl.shared_memory_descriptor  # [2] K and V of the tile loaded
    kv_free: gl.shared_memory_descriptor  # [2] all MMAs of the tile done
    qdo_ready: gl.shared_memory_descriptor  # [QDO_SLOTS] Q_i, dO_i, stats loaded
    qdo_free: gl.shared_memory_descriptor  # [QDO_SLOTS]
    s_ready: gl.shared_memory_descriptor  # [1] S^T(i) computed
    p_ready: gl.shared_memory_descriptor  # [1] P^T(i) in TMEM, S^T(i) read
    p_read: gl.shared_memory_descriptor  # [1] dS warpgroup done reading P^T(i)
    dp_ready: gl.shared_memory_descriptor  # [1] dP^T(i) computed
    dp_read: gl.shared_memory_descriptor  # [1] dS warpgroup holds dP^T(i) in registers
    ds_ready: gl.shared_memory_descriptor  # [1] dS^T(i) in smem
    ds_free: gl.shared_memory_descriptor  # [1] dK(i), dQ(i) done reading dS^T(i)
    dq_ready: gl.shared_memory_descriptor  # [1] dQ(i) in TMEM
    dq_free: gl.shared_memory_descriptor  # [1] reduce partition read dQ(i)
    dkv_ready: gl.shared_memory_descriptor  # [1] dK, dV of the tile final
    dkv_free: gl.shared_memory_descriptor  # [1] reduce partition read dK, dV

    @gluon.constexpr_function
    def __init__(
        self, kv_ready, kv_free, qdo_ready, qdo_free, s_ready, p_ready, p_read, dp_ready,
        dp_read,
        ds_ready, ds_free, dq_ready, dq_free, dkv_ready, dkv_free,
    ):  # fmt: skip
        self.kv_ready = kv_ready
        self.kv_free = kv_free
        self.qdo_ready = qdo_ready
        self.qdo_free = qdo_free
        self.s_ready = s_ready
        self.p_ready = p_ready
        self.p_read = p_read
        self.dp_ready = dp_ready
        self.dp_read = dp_read
        self.ds_ready = ds_ready
        self.ds_free = ds_free
        self.dq_ready = dq_ready
        self.dq_free = dq_free
        self.dkv_ready = dkv_ready
        self.dkv_free = dkv_free


@aggregate
class Tmem:
    s: tensor_memory_descriptor  # [TILE, TILE] fp32: S^T
    dp: tensor_memory_descriptor  # [TILE, TILE] fp32: dP^T
    p: tensor_memory_descriptor  # [TILE, TILE] fp16: P^T
    dv: tensor_memory_descriptor  # [TILE, D] fp32
    dk: tensor_memory_descriptor  # [TILE, D] fp32
    dq: tensor_memory_descriptor  # [TILE, D] fp32

    @gluon.constexpr_function
    def __init__(self, s, dp, p, dv, dk, dq):
        self.s = s
        self.dp = dp
        self.p = p
        self.dv = dv
        self.dk = dk
        self.dq = dq


@aggregate
class Kernel:
    problem: Problem
    q_desc: tma.tensor_descriptor
    k_desc: tma.tensor_descriptor
    v_desc: tma.tensor_descriptor
    do_desc: tma.tensor_descriptor
    dk_desc: tma.tensor_descriptor
    dv_desc: tma.tensor_descriptor
    stats_desc: tma.tensor_descriptor  # [B H num_q_tiles STATS, TILE]
    dq_desc: tma.tensor_descriptor  # fp32 accumulator [B, H, S_q, D], [TILE, D / 2] box
    smem: Smem
    bars: Bars
    tmem: Tmem

    @gluon.constexpr_function
    def __init__(
        self, problem, q_desc, k_desc, v_desc, do_desc, dk_desc, dv_desc, stats_desc,
        dq_desc, smem, bars, tmem,
    ):  # fmt: skip
        self.problem = problem
        self.q_desc = q_desc
        self.k_desc = k_desc
        self.v_desc = v_desc
        self.do_desc = do_desc
        self.dk_desc = dk_desc
        self.dv_desc = dv_desc
        self.stats_desc = stats_desc
        self.dq_desc = dq_desc
        self.smem = smem
        self.bars = bars
        self.tmem = tmem


@gluon.jit
def _wait_ready(bar, count):
    """Wait for the ``count``-th (0-based) completion of ``bar``."""
    mbarrier.wait(bar, count & 1)


@gluon.jit
def _wait_free(bar, count):
    """Producer side: wait until the consumer released the buffer for use ``count``
    (passes immediately for the first use)."""
    mbarrier.wait(bar, (count & 1) ^ 1)


@gluon.jit
def _matrix(buffer, rows: gl.constexpr, cols: gl.constexpr):
    """A [1, 1, rows, cols] TMA box in smem as the [rows, cols] MMA operand."""
    return buffer.reshape([rows, cols])


# --------------------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------------------


@gluon.jit
def _load_partition(k):
    """K, V once per tile; Q_i, dO_i per step into two alternating slots."""
    p, sm, bars = k.problem, k.smem, k.bars
    kv_bytes: gl.constexpr = 2 * k.k_desc.block_type.nbytes
    qdo_bytes: gl.constexpr = (
        2 * k.q_desc.block_type.nbytes + STATS * k.stats_desc.block_type.nbytes
    )
    num_q_tiles = gl.cdiv(p.seq_q, TILE)
    tiles = 0
    steps = 0
    for tile_id in range(gl.program_id(0), p.num_tiles, gl.num_programs(0)):
        info = p.tile(tile_id)
        kv_slot = tiles & 1
        _wait_free(bars.kv_free.index(kv_slot), tiles // 2)
        ready = bars.kv_ready.index(kv_slot)
        mbarrier.expect(ready, kv_bytes)
        coords = [info.batch, info.head, info.kv_start, 0]
        tma.async_copy_global_to_shared(k.k_desc, coords, ready, sm.k.index(kv_slot))
        tma.async_copy_global_to_shared(k.v_desc, coords, ready, sm.v.index(kv_slot))
        tiles += 1  # noqa: SIM113 (Gluon has no enumerate)
        for g in range(p.group):
            q_head = p.q_head(info, g)
            for i in range(info.first_q, info.end_q):
                slot = steps % QDO_SLOTS
                _wait_free(bars.qdo_free.index(slot), steps // QDO_SLOTS)
                ready = bars.qdo_ready.index(slot)
                mbarrier.expect(ready, qdo_bytes)
                coords = [info.batch, q_head, i * TILE, 0]
                tma.async_copy_global_to_shared(k.q_desc, coords, ready, sm.q.index(slot))
                tma.async_copy_global_to_shared(k.do_desc, coords, ready, sm.do.index(slot))
                stats_row = (info.batch * p.heads + q_head) * num_q_tiles + i
                stats_at = stats_row * STATS * TILE
                for which in gl.static_range(STATS):
                    tma.async_copy_global_to_shared(
                        k.stats_desc, [stats_at + which * TILE], ready,
                        sm.stats.index(STATS * slot + which),
                    )  # fmt: skip
                steps += 1


# --------------------------------------------------------------------------------------
# MMA
# --------------------------------------------------------------------------------------


@gluon.jit
def _issue_s(k, kv_slot, step):
    """S^T(step) = K Q^T into TMEM."""
    sm, bars = k.smem, k.bars
    D: gl.constexpr = k.problem.HEAD_DIM
    slot = step % QDO_SLOTS
    _wait_ready(bars.qdo_ready.index(slot), step // QDO_SLOTS)
    q_tile = _matrix(sm.q.index(slot), TILE, D).permute((1, 0))
    tcgen05_mma(
        _matrix(sm.k.index(kv_slot), TILE, D), q_tile, k.tmem.s, use_acc=False,
        mbarriers=[bars.s_ready.index(0)],
    )  # fmt: skip


@gluon.jit
def _issue_dp(k, kv_slot, step):
    """dP^T(step) = V dO^T into TMEM (Q / dO of the step already waited for)."""
    sm = k.smem
    D: gl.constexpr = k.problem.HEAD_DIM
    do_tile = _matrix(sm.do.index(step % QDO_SLOTS), TILE, D).permute((1, 0))
    tcgen05_mma(
        _matrix(sm.v.index(kv_slot), TILE, D), do_tile, k.tmem.dp, use_acc=False,
        mbarriers=[k.bars.dp_ready.index(0)],
    )  # fmt: skip


@gluon.jit
def _mma_partition(k):
    """Per tile: S^T(0), dP^T(0); then per step i (see module docstring) dV(i),
    S^T(i+1) once P(i) is in TMEM, and dP^T(i+1), dK(i), dQ(i) once dS(i) is in smem."""
    p, sm, bars, tm = k.problem, k.smem, k.bars, k.tmem
    D: gl.constexpr = p.HEAD_DIM
    tiles = 0
    steps = 0  # query steps of all tiles so far
    dq_count = 0
    for tile_id in range(gl.program_id(0), p.num_tiles, gl.num_programs(0)):
        info = p.tile(tile_id)
        n = (info.end_q - info.first_q) * p.group  # every query head of the K / V head
        kv_slot = tiles & 1
        _wait_ready(bars.kv_ready.index(kv_slot), tiles // 2)
        _issue_s(k, kv_slot, steps)
        _issue_dp(k, kv_slot, steps)
        for i in range(n):
            step = steps + i
            slot = step % QDO_SLOTS
            q_tile = _matrix(sm.q.index(slot), TILE, D)
            do_tile = _matrix(sm.do.index(slot), TILE, D)
            # P^T(i) ready: dV += P^T dO; the S^T slot is free for S^T(i + 1)
            _wait_ready(bars.p_ready.index(0), step)
            if i == 0:
                _wait_free(bars.dkv_free.index(0), tiles)  # last tile's dK, dV read
            tcgen05_mma(tm.p, do_tile, tm.dv, use_acc=i > 0)
            if i + 1 < n:
                _issue_s(k, kv_slot, step + 1)
            # dP^T(i) read out: dP^T(i + 1) into the freed slot
            if i + 1 < n:
                _wait_ready(bars.dp_read.index(0), step)
                _issue_dp(k, kv_slot, step + 1)
            # dS^T(i) ready in smem: dK += dS^T Q, dQ(i) = dS K
            ds = sm.ds
            _wait_ready(bars.ds_ready.index(0), step)
            if i + 1 == n:
                _wait_ready(bars.dp_read.index(0), step)  # keep the barrier count
            tcgen05_mma(ds, q_tile, tm.dk, use_acc=i > 0)
            _wait_free(bars.dq_free.index(0), dq_count)
            tcgen05_mma(
                ds.permute((1, 0)), _matrix(sm.k.index(kv_slot), TILE, D), tm.dq,
                use_acc=False,
                mbarriers=[bars.dq_ready.index(0), bars.ds_free.index(0),
                           bars.qdo_free.index(slot)],
            )  # fmt: skip
            dq_count += 1
        tcgen05_commit(bars.dkv_ready.index(0))
        tcgen05_commit(bars.kv_free.index(kv_slot))
        steps += n
        tiles += 1  # noqa: SIM113 (Gluon has no enumerate)


# --------------------------------------------------------------------------------------
# Compute: P^T and dS^T
# --------------------------------------------------------------------------------------


@gluon.jit
def _p_partition(k):
    """P warpgroup: per step, P^T = exp2(qk_scale (S^T - m) - log2 l) for all 128 query
    columns -> TMEM (fp16): the forward's shift-then-scale (S - m exact near the row
    max, whatever the scores' magnitude), then log2 l in the same fma. Thread t owns key row t. It runs a step ahead of the dS
    warpgroup: P^T(i + 1) is written once the dS warpgroup has read P^T(i)."""
    p = k.problem
    chunk_tmem: gl.constexpr = TensorMemoryLayout([TILE, CHUNK], col_stride=1)
    regs: gl.constexpr = get_tmem_reg_layout(gl.float32, [TILE, CHUNK], chunk_tmem, gl.num_warps())
    row_layout: gl.constexpr = gl.SliceLayout(1, regs)
    steps = 0
    for tile_id in range(gl.program_id(0), p.num_tiles, gl.num_programs(0)):
        info = p.tile(tile_id)
        # key c is visible to query r iff r >= first query of c's frame
        keys = info.kv_start + gl.arange(0, TILE, layout=row_layout)
        if p.CAUSAL:
            first_query = (keys // p.block) * p.block
        else:
            first_query = keys * 0
        # padded keys (past S_kv, zero K / V rows) are visible to no query: their score
        # is 0, so P = exp2(-c m - log2 l) overflows for rows whose scores are all
        # negative, and dQ += dS K would be inf * 0. The key tile holding them runs
        # every step masked; full tiles keep the unmasked path.
        first_query = gl.where(keys < p.seq_kv, first_query, 2**30)
        masked_end = info.first_full
        if info.kv_start + TILE > p.seq_kv:
            masked_end = info.end_q
        for g in range(p.group):  # the same keys for every query head of the group
            steps = _p_steps(k, first_query, info.first_q, masked_end, steps, regs, True)
            steps = _p_steps(k, first_query, masked_end, info.end_q, steps, regs, False)


@gluon.jit
def _p_steps(k, first_query, start, end, steps, regs: gl.constexpr, MASKED: gl.constexpr):
    """Query tiles ``start .. end - 1``; ``MASKED`` applies the block-causal mask."""
    p, sm, bars, tm = k.problem, k.smem, k.bars, k.tmem
    NUM_CHUNKS: gl.constexpr = TILE // CHUNK
    col_layout: gl.constexpr = gl.SliceLayout(0, regs)
    offsets = gl.arange(0, CHUNK, layout=col_layout)
    for i in range(start, end):
        neg_m = sm.stats.index(STATS * (steps % QDO_SLOTS))
        neg_log_l = sm.stats.index(STATS * (steps % QDO_SLOTS) + 1)
        _wait_ready(bars.s_ready.index(0), steps)
        probs = ()
        for c in gl.static_range(NUM_CHUNKS):
            s2 = float2.pack(tm.s.slice(c * CHUNK, CHUNK).load(regs), axis=1)
            m = neg_m.slice(c * CHUNK, CHUNK).load(col_layout)
            m2 = float2.pack(m[None, :].broadcast_to([TILE, CHUNK]), axis=1)
            log_l = neg_log_l.slice(c * CHUNK, CHUNK).load(col_layout)
            log_l2 = float2.pack(log_l[None, :].broadcast_to([TILE, CHUNK]), axis=1)
            scale2 = float2.full_like(s2, p.qk_scale)
            x2 = float2.fma(s2 + m2, scale2, log_l2)
            prob = gl.exp2(float2.unpack(x2, axis=1))
            if MASKED:
                queries = i * TILE + c * CHUNK + offsets
                visible = queries[None, :] >= first_query[:, None]
                prob = gl.where(visible, prob, 0.0)
            probs = probs + (prob.to(k.q_desc.block_type.element_ty),)
        _wait_free(bars.p_read.index(0), steps)  # the dS warpgroup read P^T(i - 1)
        tm.p.store(_join_columns(probs))
        mbarrier.arrive(bars.p_ready.index(0), count=1)
        steps += 1
    return steps


@gluon.jit
def _ds_partition(k):
    """dS warpgroup: per step, dS^T = P^T (dP^T - delta) for all 128 query columns ->
    smem (fp16, the A operand of the dK and dQ MMAs)."""
    p, sm, bars, tm = k.problem, k.smem, k.bars, k.tmem
    NUM_CHUNKS: gl.constexpr = TILE // CHUNK
    chunk_tmem: gl.constexpr = TensorMemoryLayout([TILE, CHUNK], col_stride=1)
    regs: gl.constexpr = get_tmem_reg_layout(gl.float32, [TILE, CHUNK], chunk_tmem, gl.num_warps())
    p_regs: gl.constexpr = get_tmem_reg_layout(
        k.q_desc.block_type.element_ty, [TILE, CHUNK], chunk_tmem, gl.num_warps()
    )
    col_layout: gl.constexpr = gl.SliceLayout(0, regs)
    steps = 0
    for tile_id in range(gl.program_id(0), p.num_tiles, gl.num_programs(0)):
        info = p.tile(tile_id)
        for _ in range((info.end_q - info.first_q) * p.group):  # every query head's steps
            neg_delta = sm.stats.index(STATS * (steps % QDO_SLOTS) + 2)
            # all of dP^T(i) to registers first: its TMEM slot is free for dP^T(i + 1)
            _wait_ready(bars.dp_ready.index(0), steps)
            dps = ()
            for c in gl.static_range(NUM_CHUNKS):
                dps = dps + (tm.dp.slice(c * CHUNK, CHUNK).load(regs),)
            mbarrier.arrive(bars.dp_read.index(0), count=1)

            _wait_ready(bars.p_ready.index(0), steps)
            _wait_free(bars.ds_free.index(0), steps)
            for c in gl.static_range(NUM_CHUNKS):
                prob = tm.p.slice(c * CHUNK, CHUNK).load(p_regs)
                prob = gl.convert_layout(prob.to(gl.float32), regs)
                delta = neg_delta.slice(c * CHUNK, CHUNK).load(col_layout)
                delta2 = float2.pack(delta[None, :].broadcast_to([TILE, CHUNK]), axis=1)
                ds2 = (float2.pack(dps[c], axis=1) + delta2) * float2.pack(prob, axis=1)
                ds = float2.unpack(ds2, axis=1).to(k.q_desc.block_type.element_ty)
                sm.ds.slice(c * CHUNK, CHUNK, dim=1).store(ds)
            mbarrier.arrive(bars.p_read.index(0), count=1)
            fence_async_shared()
            mbarrier.arrive(bars.ds_ready.index(0), count=1)
            steps += 1


# --------------------------------------------------------------------------------------
# Reduce: dQ atomics, dK / dV epilogue
# --------------------------------------------------------------------------------------


@gluon.jit
def _reduce_partition(k):
    """Default partition. Per step: dQ(i) TMEM -> registers -> smem (fp32) -> TMA
    reduce-add into the dQ accumulator (two [TILE, D / 2] boxes). Per tile: dV and
    dK (scaled) -> fp16 -> TMA store, staged in the same smem."""
    p, sm, bars, tm = k.problem, k.smem, k.bars, k.tmem
    D: gl.constexpr = p.HEAD_DIM
    HALF_D: gl.constexpr = D // 2
    half_tmem: gl.constexpr = TensorMemoryLayout([TILE, HALF_D], col_stride=1)
    half_regs: gl.constexpr = get_tmem_reg_layout(
        gl.float32, [TILE, HALF_D], half_tmem, gl.num_warps()
    )
    acc_tmem: gl.constexpr = TensorMemoryLayout([TILE, D], col_stride=1)
    regs: gl.constexpr = get_tmem_reg_layout(gl.float32, [TILE, D], acc_tmem, gl.num_warps())
    # the staging buffer seen as one fp16 [TILE, D] box for the dK / dV stores
    out16 = sm.out._reinterpret(k.dk_desc.block_type.element_ty, [1, 1, TILE, D], k.dk_desc.layout)
    tiles = 0
    dq_count = 0
    scale = p.qk_scale / 1.4426950408889634
    for tile_id in range(gl.program_id(0), p.num_tiles, gl.num_programs(0)):
        info = p.tile(tile_id)
        for g in range(p.group):
            q_head = p.q_head(info, g)
            for i in range(info.first_q, info.end_q):
                _wait_ready(bars.dq_ready.index(0), dq_count)
                dq_lo = tm.dq.slice(0, HALF_D).load(half_regs)
                dq_hi = tm.dq.slice(HALF_D, HALF_D).load(half_regs)
                mbarrier.arrive(bars.dq_free.index(0), count=1)
                dq_count += 1
                coords = [info.batch, q_head, i * TILE, 0]
                _stage_reduce(k, dq_lo, coords)
                coords = [info.batch, q_head, i * TILE, HALF_D]
                _stage_reduce(k, dq_hi, coords)

        _wait_ready(bars.dkv_ready.index(0), tiles)
        dv = tm.dv.load(regs)
        dk = tm.dk.load(regs)
        mbarrier.arrive(bars.dkv_free.index(0), count=1)
        tiles += 1  # noqa: SIM113 (Gluon has no enumerate)
        coords = [info.batch, info.head, info.kv_start, 0]
        tma.store_wait(pendings=0)
        _matrix(out16, TILE, D).store(dv.to(k.dk_desc.block_type.element_ty))
        fence_async_shared()
        tma.async_copy_shared_to_global(k.dv_desc, coords, out16)
        tma.store_wait(pendings=0)
        _matrix(out16, TILE, D).store((dk * scale).to(k.dk_desc.block_type.element_ty))
        fence_async_shared()
        tma.async_copy_shared_to_global(k.dk_desc, coords, out16)
    tma.store_wait(pendings=0)


@gluon.jit
def _stage_reduce(k, values, coords):
    """global[coords box] += values through the staging buffer (TMA reduce-add)."""
    tma.store_wait(pendings=0)  # the previous TMA op finished reading the staging
    _matrix(k.smem.out, TILE, k.problem.HEAD_DIM // 2).store(values)
    fence_async_shared()
    _tma_reduce_add(k.dq_desc, coords, k.smem.out)


@builtin
def _tma_reduce_add(tensor_desc, coord, src, _semantic=None):
    """TMA bulk reduce: global[box at coord] += smem ``src`` (cp.reduce.async.bulk
    .tensor .add). Completion is tracked like a TMA store (``tma.store_wait``)."""
    coord = _semantic._convert_to_ir_values(coord, require_i64=False)
    _semantic.builder.create_async_tma_reduce(
        ir.DESCRIPTOR_REDUCE_KIND.ADD, tensor_desc.handle, coord, src.handle
    )


# --------------------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------------------


@gluon.jit
def _barriers(n: gl.constexpr, count: gl.constexpr = 1):
    bars = gl.allocate_shared_memory(gl.int64, [n, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(n):
        mbarrier.init(bars.index(i), count=count)
    return bars


@gluon.jit
def attention_bwd_kernel(
    q_desc, k_desc, v_desc, do_desc, dk_desc, dv_desc, stats_desc, dq_desc, qk_scale, heads, group, seq_q, seq_kv, block, num_kv_tiles, num_tiles,
    HEAD_DIM: gl.constexpr, CAUSAL: gl.constexpr,
):  # fmt: skip
    problem = Problem(
        gl.to_tensor(qk_scale), gl.to_tensor(heads), gl.to_tensor(group),
        gl.to_tensor(seq_q),
        gl.to_tensor(seq_kv), gl.to_tensor(block), gl.to_tensor(num_kv_tiles),
        gl.to_tensor(num_tiles), HEAD_DIM, CAUSAL,
    )  # fmt: skip
    box: gl.constexpr = [1, 1, TILE, HEAD_DIM]
    # 64-byte swizzle: compute threads store dS^T in 32-column slices
    ds_layout: gl.constexpr = gl.NVMMASharedLayout(64, 16, rank=2)
    dtype: gl.constexpr = q_desc.block_type.element_ty  # fp16 or bf16 operands
    smem = Smem(
        gl.allocate_shared_memory(dtype, [2] + box, k_desc.layout),
        gl.allocate_shared_memory(dtype, [2] + box, v_desc.layout),
        gl.allocate_shared_memory(dtype, [QDO_SLOTS] + box, q_desc.layout),
        gl.allocate_shared_memory(dtype, [QDO_SLOTS] + box, do_desc.layout),
        gl.allocate_shared_memory(dtype, [TILE, TILE], ds_layout),
        gl.allocate_shared_memory(gl.float32, [1, 1, TILE, HEAD_DIM // 2], dq_desc.layout),
        gl.allocate_shared_memory(gl.float32, [STATS * QDO_SLOTS, TILE], stats_desc.layout),
    )
    bars = Bars(
        _barriers(2), _barriers(2), _barriers(QDO_SLOTS), _barriers(QDO_SLOTS),
        _barriers(1), _barriers(1), _barriers(1), _barriers(1), _barriers(1),
        _barriers(1), _barriers(1), _barriers(1), _barriers(1), _barriers(1),
        _barriers(1),
    )  # fmt: skip
    fence_async_shared()
    scores: gl.constexpr = TensorMemoryLayout([TILE, TILE], col_stride=1)
    acc: gl.constexpr = TensorMemoryLayout([TILE, HEAD_DIM], col_stride=1)
    s_tmem = allocate_tensor_memory(gl.float32, [TILE, TILE], scores)
    dp_tmem = allocate_tensor_memory(gl.float32, [TILE, TILE], scores)
    p_tmem = allocate_tensor_memory(dtype, [TILE, TILE], scores)
    tmem = Tmem(
        s_tmem, dp_tmem, p_tmem,
        allocate_tensor_memory(gl.float32, [TILE, HEAD_DIM], acc),
        allocate_tensor_memory(gl.float32, [TILE, HEAD_DIM], acc),
        allocate_tensor_memory(gl.float32, [TILE, HEAD_DIM], acc),
    )  # fmt: skip
    k = Kernel(
        problem, q_desc, k_desc, v_desc, do_desc, dk_desc, dv_desc, stats_desc, dq_desc,
        smem, bars, tmem,
    )  # fmt: skip
    gl.warp_specialize(
        [
            (_reduce_partition, (k,)),  # default partition: 4 warps
            (_p_partition, (k,)),
            (_ds_partition, (k,)),
            (_mma_partition, (k,)),
            (_load_partition, (k,)),
        ],
        [4, 4, 1, 1],
        [152, 208, 32, 32],
    )


# --------------------------------------------------------------------------------------
# Small kernels around the main one
# --------------------------------------------------------------------------------------


@triton.jit
def _prepare_kernel(
    out_ptr, dout_ptr, lse_ptr, stats_ptr, dq_ptr, seq, heads, num_q_tiles,
    stride_ob, stride_oh, stride_os, stride_db, stride_dh, stride_ds,
    D: tl.constexpr, TILE_Q: tl.constexpr,
):  # fmt: skip
    """One query tile of one (batch, head): stats[bh, tile] = (-m, -log2 l, -delta) with
    delta = rowsum(dout * out) (rows past seq: -m = -inf, so P = 0); zeroes the tile's
    rows of the fp32 dQ accumulator."""
    bh = tl.program_id(0) // num_q_tiles
    tile = tl.program_id(0) % num_q_tiles
    b = bh // heads
    h = bh % heads
    s = tile * TILE_Q + tl.arange(0, TILE_Q)
    valid = s < seq
    cols = tl.arange(0, D)
    o_off = b * stride_ob + h * stride_oh + s * stride_os
    d_off = b * stride_db + h * stride_dh + s * stride_ds
    o = tl.load(out_ptr + o_off[:, None] + cols[None, :], mask=valid[:, None])
    do = tl.load(dout_ptr + d_off[:, None] + cols[None, :], mask=valid[:, None])
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    row_stats = lse_ptr + bh.to(tl.int64) * 2 * seq + s
    m = tl.load(row_stats, mask=valid, other=float("inf"))
    log_l = tl.load(row_stats + seq, mask=valid, other=0.0)
    stats = stats_ptr + (bh.to(tl.int64) * num_q_tiles + tile) * 3 * TILE_Q
    tl.store(stats + tl.arange(0, TILE_Q), -m)
    tl.store(stats + TILE_Q + tl.arange(0, TILE_Q), -log_l)
    tl.store(stats + 2 * TILE_Q + tl.arange(0, TILE_Q), -delta)
    dq_rows = dq_ptr + (bh.to(tl.int64) * seq + s[:, None]) * D + cols[None, :]
    tl.store(dq_rows, tl.zeros([TILE_Q, D], dtype=tl.float32), mask=valid[:, None])


@triton.jit
def _dq_convert_kernel(
    acc_ptr, dq_ptr, rows, seq, heads, scale,
    stride_b, stride_h, stride_s, D: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """dq = scale * accumulator, fp16, into the [B, H, S, D] view (any strides)."""
    pid = tl.program_id(0)
    row = pid * BLOCK + tl.arange(0, BLOCK)
    valid = row < rows
    s = row % seq
    bh = row // seq
    b = bh // heads
    h = bh % heads
    cols = tl.arange(0, D)
    acc = tl.load(acc_ptr + row[:, None].to(tl.int64) * D + cols[None, :], mask=valid[:, None])
    off = b * stride_b + h * stride_h + s * stride_s
    tl.store(
        dq_ptr + off[:, None] + cols[None, :],
        (acc * scale).to(dq_ptr.dtype.element_ty),
        mask=valid[:, None],
    )


def attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    block: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients of ``attention_forward``: dq, dk, dv as [B, H, S, D] views of
    [B, S, H, D] storage. Extra memory: (-m, -log2 l, -delta) per row (fp32) and the fp32 dQ
    accumulator ([B, H, Sq, D])."""
    batch, heads, seq_q, dim = q.shape
    seq_kv = k.shape[2]
    device = q.device
    if dout.stride(-1) != 1 or (dout.stride(2) * 2) % 16:
        dout = dout.contiguous()

    rows = batch * heads * seq_q
    num_q_tiles = triton.cdiv(seq_q, TILE_ROWS)
    stats = torch.empty(
        batch * heads * num_q_tiles * 3 * TILE_ROWS, device=device, dtype=torch.float32
    )
    dq_acc = torch.empty(batch, heads, seq_q, dim, device=device, dtype=torch.float32)
    _prepare_kernel[(batch * heads * num_q_tiles,)](
        out, dout, lse, stats, dq_acc, seq_q, heads, num_q_tiles,
        out.stride(0), out.stride(1), out.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(2),
        D=dim, TILE_Q=TILE_ROWS,
    )  # fmt: skip
    stats_layout = gl.NVMMASharedLayout(0, 32, rank=1)
    stats_desc = TensorDescriptor.from_tensor(stats, [TILE_ROWS], stats_layout)
    dq_box = [1, 1, TILE_ROWS, dim // 2]
    dq_layout = gl.NVMMASharedLayout.get_default_for(dq_box, gl.float32)
    dq_desc = TensorDescriptor(dq_acc, list(dq_acc.shape), list(dq_acc.stride()), dq_box, dq_layout)

    # Keys no query sees (block-causal with Sq < Skv: key c is first seen by query
    # (c // P) * P) get zero dK / dV; only key tiles some query sees are launched (a
    # tile with no query step would never signal its dK / dV).
    seen_kv = seq_kv if block == 0 else min(seq_kv, ((seq_q - 1) // block + 1) * block)
    alloc = torch.empty if seen_kv == seq_kv else torch.zeros
    kv_heads = k.shape[1]  # GQA: dK / dV per K / V head, summed over its query heads
    dk = alloc(batch, seq_kv, kv_heads, dim, device=device, dtype=q.dtype)
    dv = alloc(batch, seq_kv, kv_heads, dim, device=device, dtype=q.dtype)
    dk, dv = dk.transpose(1, 2), dv.transpose(1, 2)
    num_kv_tiles = triton.cdiv(seen_kv, TILE_ROWS)
    num_tiles = batch * kv_heads * num_kv_tiles
    grid = (min(num_tiles, sm_count(device)),)
    attention_bwd_kernel[grid](
        tma_descriptor(q, TILE_ROWS), tma_descriptor(k, TILE_ROWS),
        tma_descriptor(v, TILE_ROWS), tma_descriptor(dout, TILE_ROWS),
        tma_descriptor(dk, TILE_ROWS), tma_descriptor(dv, TILE_ROWS),
        stats_desc, dq_desc, scale * LOG2E, heads, heads // kv_heads, seq_q, seq_kv,
        max(block, 1),
        num_kv_tiles, num_tiles, HEAD_DIM=dim, CAUSAL=block > 0, num_warps=4,
    )  # fmt: skip

    dq = torch.empty(batch, seq_q, heads, dim, device=device, dtype=q.dtype)
    dq = dq.transpose(1, 2)
    _dq_convert_kernel[(triton.cdiv(rows, 64),)](
        dq_acc, dq, rows, seq_q, heads, scale,
        dq.stride(0), dq.stride(1), dq.stride(2), D=dim, BLOCK=64,
    )  # fmt: skip
    return dq, dk, dv
