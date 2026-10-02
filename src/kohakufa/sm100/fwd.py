"""Attention forward as one warp-specialized Gluon kernel (Blackwell, tcgen05 + TMEM).

    out = softmax(scale * q k^T) v,   q/k/v [B, H, S, D] fp16 (any strides, D contiguous)
    lse = the softmax statistics per query row, kept for the backward: the row max m of
          the raw scores and log2 l (l = sum exp2(scale log2(e) (scores - m))) as two
          fp32 planes [B, H, 2, Sq], never summed (see attention_forward)

Block-causal masking (block size P > 0): query row i sees key j iff j // P <= i // P. It
is applied by block sparsity: for every (query tile, key tile) pair the kernel knows
from the tile coordinates alone whether the pair is fully visible (computed as a plain
dense tile), fully hidden (never loaded or computed: the key loop stops before it) or
on the frame boundary (computed with an elementwise mask). The same mask machinery
handles the key tail (keys past S_kv) of every shape; on short sequences (at most four
key tiles) the last key tile is computed only TAIL_N columns wide.

Work decomposition. A CTA tile is two 128-row query halves. Normally both are rows of
one (batch, head) (256 consecutive rows) and share every K / V tile. When a sequence
ends with a short remainder (seq_q % 256 in (0, 128]), the remainders of two heads are
paired into one tile, each half with its own K / V, so both softmax partitions always
work (e.g. 269 = 256 + 13 rows). The grid is persistent; block-causal tiles are ordered
longest first.

The S-tile stream. Every (tile, key tile j, half h) is one S tile, S = Q_h K_j^T
(128 x 128). The CTA walks one stream of S tiles across all its CTA tiles, ordered
(j, h) within a tile. S tiles live in a ring of three TMEM slots; P (fp16) is written
over its S. The MMA warp issues QK^T three S tiles ahead of P V, so a softmax partition
finds its next S already computed when it finishes the current one, and the next CTA
tile's first S tiles are computed while the current tile drains.

Partitions (warp specialization):

    load        1 warp   TMA: Q (double-buffered per half), K_j and V_j into a ring of
                         smem slots in the order the MMA warp first uses them
    mma         1 warp   QK(t + 3) after PV(t): S = Q_h K_j^T into the slot ring;
                         O_h += P V_j with P read from TMEM
    softmax h   4 warps  one per half: S -> row max m, alpha = exp2(m_old - m) to the
                         correction partition, P = exp2(S - m) as fp16 over S, l += ...
    correction  4 warps  O_h *= alpha between consecutive P V MMAs of a half; at the end
                         of a tile O_h / l -> fp16 -> smem -> TMA store, lse = (m, log2 l)

TMEM (all 512 columns): three S slots (128 columns each), O_0 and O_1 (64 each).
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    float2,
    get_tmem_reg_layout,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
    tma,
)
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.language.core import _aggregate as aggregate

HALF_ROWS = 128  # query rows per softmax partition (the tcgen05 MMA M)
LOG2E = 1.4426950408889634

HALF = gl.constexpr(HALF_ROWS)
CHUNK = gl.constexpr(32)  # score columns a softmax thread handles at once
S_SLOTS = gl.constexpr(3)  # TMEM ring of S tiles = how far QK^T runs ahead of P V


# --------------------------------------------------------------------------------------
# Problem description and the tile schedule
# --------------------------------------------------------------------------------------


@aggregate
class Problem:
    """Sizes and the tile schedule. Runtime scalars except the constexpr tile shape.

    Tiles: ``num_pairs`` paired tails first, then ``num_full`` 256-row tiles per
    (batch, head). When seq_q % 256 is in (0, 128], every head ends with a short
    one-half tail; two heads' tails form one two-half tile (each half with its own
    K / V), so both softmax partitions always have work."""

    qk_scale: gl.tensor  # softmax scale * log2(e): scores in the exp2 domain
    heads: gl.tensor  # query heads
    group: gl.tensor  # query heads per K / V head (GQA; 1 = multi-head)
    batch_heads: gl.tensor
    seq_q: gl.tensor
    seq_kv: gl.tensor
    block: gl.tensor  # block-causal block size P (unused when not CAUSAL)
    num_full: gl.tensor  # 256-row tiles per (batch, head)
    num_pairs: gl.tensor  # paired-tail tiles
    num_tiles: gl.tensor
    BLOCK_N: gl.constexpr  # keys per K / V tile
    TAIL_N: gl.constexpr  # columns computed for the last key tile (< BLOCK_N: narrow)
    HEAD_DIM: gl.constexpr
    CAUSAL: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self, qk_scale, heads, group, batch_heads, seq_q, seq_kv, block, num_full,
        num_pairs, num_tiles, BLOCK_N, TAIL_N, HEAD_DIM, CAUSAL,
    ):  # fmt: skip
        self.qk_scale = qk_scale
        self.heads = heads
        self.group = group
        self.batch_heads = batch_heads
        self.seq_q = seq_q
        self.seq_kv = seq_kv
        self.block = block
        self.num_full = num_full
        self.num_pairs = num_pairs
        self.num_tiles = num_tiles
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.TAIL_N = gl.constexpr(TAIL_N)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.CAUSAL = gl.constexpr(CAUSAL)

    @gluon.jit
    def narrow(self, j):
        """Key tile ``j`` is the sequence's last one and computed TAIL_N wide (a
        compile-time False when TAIL_N == BLOCK_N: the narrow code is not built)."""
        if self.TAIL_N < self.BLOCK_N:
            return (j + 1) * self.BLOCK_N > self.seq_kv
        else:
            return gl.constexpr(False)

    @gluon.jit
    def key_limit(self, row):
        """Keys visible to query ``row`` (scalar or tensor): [0, key_limit)."""
        if self.CAUSAL:
            return gl.minimum((row // self.block + 1) * self.block, self.seq_kv)
        else:
            return row * 0 + self.seq_kv

    @gluon.jit
    def tile(self, tile_id):
        """Coordinates and key-loop bounds of CTA tile ``tile_id``."""
        # paired tails: heads 2 p and 2 p + 1, rows from tail_start (last rows: for
        # block-causal they see the most keys, so they come first)
        pair = tile_id < self.num_pairs
        tail_start = self.num_full * (2 * HALF)
        bh_tail = 2 * tile_id
        # full tiles
        full_id = tile_id - self.num_pairs
        if self.CAUSAL:
            # longest first over all heads: the last query tiles see the most keys
            # (query-tile work varies too much for a head-major order to balance)
            q_tile = self.num_full - 1 - full_id // self.batch_heads
            bh_full = full_id % self.batch_heads
        else:
            # consecutive tiles share a head: CTAs running together share its K / V
            q_tile = full_id % self.num_full
            bh_full = full_id // self.num_full
        q_start = q_tile * (2 * HALF)

        bh0 = gl.where(pair, bh_tail, bh_full)
        bh1 = gl.where(pair, gl.minimum(bh_tail + 1, self.batch_heads - 1), bh_full)
        row0 = gl.where(pair, tail_start, q_start)
        row1 = gl.where(pair, tail_start, q_start + HALF)
        last_row = gl.where(pair, self.seq_q, gl.minimum(q_start + 2 * HALF, self.seq_q))
        num_kv = gl.cdiv(self.key_limit(last_row - 1), self.BLOCK_N)
        halves = gl.where(pair, gl.where(bh_tail + 1 < self.batch_heads, 2, 1),
                          gl.where(q_start + HALF < self.seq_q, 2, 1))  # fmt: skip
        return TileInfo(bh0, row0, bh1, row1, num_kv, halves, gl.where(pair, 0, 1))

    @gluon.jit
    def unmasked_tiles(self, info, half: gl.constexpr):
        """Leading key tiles every row of ``half`` sees in full (no mask needed)."""
        first_row = info.row(half)
        return gl.minimum(self.key_limit(first_row) // self.BLOCK_N, info.num_kv)

    @gluon.jit
    def coords(self, info, half):
        """TMA coordinates [batch, head, first row, 0] of ``half``'s query rows."""
        bh = info.batch_head(half)
        return [bh // self.heads, bh % self.heads, info.row(half), 0]


@aggregate
class TileInfo:
    """A CTA tile: two 128-row halves, each with its (batch * heads + head) and first
    query row; ``shared``: both halves read the same K / V (same head)."""

    bh0: gl.tensor
    row0: gl.tensor
    bh1: gl.tensor
    row1: gl.tensor
    num_kv: gl.tensor  # key tiles the tile visits (later ones are fully hidden)
    halves: gl.tensor  # 2, or 1 when there is no second half
    shared: gl.tensor

    @gluon.constexpr_function
    def __init__(self, bh0, row0, bh1, row1, num_kv, halves, shared):
        self.bh0 = bh0
        self.row0 = row0
        self.bh1 = bh1
        self.row1 = row1
        self.num_kv = num_kv
        self.halves = halves
        self.shared = shared

    @gluon.jit
    def batch_head(self, half):
        return gl.where(half == 0, self.bh0, self.bh1)

    @gluon.jit
    def row(self, half):
        return gl.where(half == 0, self.row0, self.row1)


@aggregate
class Cursor:
    """A position in this CTA's stream of S tiles: CTA tile (``tile_id``, ``info``),
    key tile ``j``, ``half``, and ``index``: the S tile's position in the stream (its
    TMEM slot is index % 3). ``uses0`` / ``uses1``: earlier tiles of this CTA with a
    first / second half (they pick the double-buffered Q slots)."""

    tile_id: gl.tensor
    info: TileInfo
    j: gl.tensor
    half: gl.tensor
    index: gl.tensor
    uses0: gl.tensor
    uses1: gl.tensor

    @gluon.constexpr_function
    def __init__(self, tile_id, info, j, half, index, uses0, uses1):
        self.tile_id = tile_id
        self.info = info
        self.j = j
        self.half = half
        self.index = index
        self.uses0 = uses0
        self.uses1 = uses1

    @gluon.jit
    def start(problem):
        tile_id = gl.program_id(0)
        zero = gl.to_tensor(0)
        return Cursor(tile_id, problem.tile(tile_id), zero, zero, zero, zero, zero)

    @gluon.jit
    def q_slot(self):
        """Q smem slot (and barrier) of this S tile's half and the use count of it:
        slots 0, 1 alternate for half 0, slots 2, 3 for half 1."""
        uses = gl.where(self.half == 0, self.uses0, self.uses1)
        return 2 * self.half + (uses & 1), uses // 2

    @gluon.jit
    def valid(self, problem):
        return self.tile_id < problem.num_tiles

    @gluon.jit
    def next(self, problem):
        """The following S tile: other half, then next key tile, then next CTA tile."""
        half = self.half + 1
        j = self.j
        tile_id = self.tile_id
        info = self.info
        uses0 = self.uses0
        uses1 = self.uses1
        if half == info.halves:
            half = 0
            j = j + 1
            if j == info.num_kv:
                j = 0
                uses0 = uses0 + 1
                uses1 = uses1 + info.halves - 1
                tile_id = tile_id + gl.num_programs(0)
                info = problem.tile(tile_id)
        return Cursor(tile_id, info, j, half, self.index + 1, uses0, uses1)

    @gluon.jit
    def first_use(self):
        """K / V tile j is acquired here (shared by the halves: at the first one)."""
        return (self.half == 0) | (self.info.shared == 0)

    @gluon.jit
    def last_use(self):
        """K / V tile j is released here (shared: after the last half)."""
        return (self.half == self.info.halves - 1) | (self.info.shared == 0)


# --------------------------------------------------------------------------------------
# Kernel state shared by every partition
# --------------------------------------------------------------------------------------


@aggregate
class Smem:
    q: gl.shared_memory_descriptor  # [4, 1, 1, HALF, D]: slot 2 * half + buffer
    kv: gl.shared_memory_descriptor  # [KV_BUFFERS, 1, 1, BLOCK_N, D]: K / V ring
    out: gl.shared_memory_descriptor  # [2, 1, 1, HALF, D]: output staging
    alpha: gl.shared_memory_descriptor  # [4, HALF] fp32: [2 * half + buffer]
    stats: gl.shared_memory_descriptor  # [4, HALF] fp32: (m, l) of half 0, of half 1

    @gluon.constexpr_function
    def __init__(self, q, kv, out, alpha, stats):
        self.q = q
        self.kv = kv
        self.out = out
        self.alpha = alpha
        self.stats = stats


@aggregate
class Bars:
    """mbarriers. ``*_ready``: producer done; ``*_free``: the consumer released the
    buffer."""

    q_ready: gl.shared_memory_descriptor  # [4] per Q slot
    q_free: gl.shared_memory_descriptor  # [4]
    kv_ready: gl.shared_memory_descriptor  # [KV_BUFFERS]
    kv_free: gl.shared_memory_descriptor  # [KV_BUFFERS]
    # [6] mma -> softmax: S computed in slot s for half h (barrier 2 s + h). Split by
    # half: a softmax that skipped a one-half tile must never wait on a barrier phase
    # two completions ahead (the parity would alias).
    s_ready: gl.shared_memory_descriptor
    p_ready: gl.shared_memory_descriptor  # [3] softmax -> mma: P written over it
    alpha_ready: gl.shared_memory_descriptor  # [4] softmax -> correction
    alpha_free: gl.shared_memory_descriptor  # [4] correction -> softmax
    o_ready: gl.shared_memory_descriptor  # [2] mma -> correction: P V accumulated
    o_free: gl.shared_memory_descriptor  # [2] correction -> mma: O rescaled / read out
    stats_ready: gl.shared_memory_descriptor  # [2] softmax -> correction: final (m, l)
    stats_free: gl.shared_memory_descriptor  # [2]

    @gluon.constexpr_function
    def __init__(
        self, q_ready, q_free, kv_ready, kv_free, s_ready, p_ready, alpha_ready,
        alpha_free, o_ready, o_free, stats_ready, stats_free,
    ):  # fmt: skip
        self.q_ready = q_ready
        self.q_free = q_free
        self.kv_ready = kv_ready
        self.kv_free = kv_free
        self.s_ready = s_ready
        self.p_ready = p_ready
        self.alpha_ready = alpha_ready
        self.alpha_free = alpha_free
        self.o_ready = o_ready
        self.o_free = o_free
        self.stats_ready = stats_ready
        self.stats_free = stats_free


@aggregate
class Kernel:
    problem: Problem
    q_desc: tma.tensor_descriptor
    k_desc: tma.tensor_descriptor
    v_desc: tma.tensor_descriptor
    o_desc: tma.tensor_descriptor
    lse_ptr: gl.tensor
    smem: Smem
    bars: Bars
    KV_BUFFERS: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, problem, q_desc, k_desc, v_desc, o_desc, lse_ptr, smem, bars, KV_BUFFERS):
        self.problem = problem
        self.q_desc = q_desc
        self.k_desc = k_desc
        self.v_desc = v_desc
        self.o_desc = o_desc
        self.lse_ptr = lse_ptr
        self.smem = smem
        self.bars = bars
        self.KV_BUFFERS = gl.constexpr(KV_BUFFERS)


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
def _wait_bit(bar, phases, bit):
    """Wait for the next completion of one of several barriers whose waited
    completions (mod 2) are kept as bits of ``phases``; returns the updated bits."""
    mbarrier.wait(bar, (phases >> bit) & 1)
    return phases ^ (1 << bit)


@gluon.jit
def _matrix(buffer, rows: gl.constexpr, cols: gl.constexpr):
    """A [1, 1, rows, cols] TMA box in smem as the [rows, cols] MMA operand."""
    return buffer.reshape([rows, cols])


@gluon.constexpr_function
def _joined_layout(layout):
    """Layout of two column-adjacent copies of ``layout`` joined into one tensor (the
    new column bit becomes the highest register bit)."""
    shape = list(layout.shape)
    reg_bases = layout.reg_bases + [[0, shape[1]]]
    shape[1] *= 2
    return gl.DistributedLinearLayout(
        reg_bases, layout.lane_bases, layout.warp_bases, layout.block_bases, shape
    )


@gluon.jit
def _join_columns(parts):
    """Concatenate equally shaped [M, N] register tensors along the columns (in
    registers, no data movement)."""
    if len(parts) == 1:
        return parts[0]
    else:
        left = _join_columns(parts[: len(parts) // 2])
        right = _join_columns(parts[len(parts) // 2 :])
        joined = gl.join(left, right).permute(0, 2, 1)
        joined = joined.reshape([left.shape[0], left.shape[1] * 2])
        layout: gl.constexpr = _joined_layout(left.type.layout)
        return gl.convert_layout(joined, layout, assert_trivial=True)


@gluon.jit
def _p_view(s_slot, BN: gl.constexpr, dtype: gl.constexpr):
    """P (16-bit ``dtype``, [HALF, BN]) aliasing the first BN / 2 columns of an S slot."""
    layout: gl.constexpr = TensorMemoryLayout([HALF, BN], col_stride=1)
    return s_slot.slice(0, BN // 2)._reinterpret(dtype, [HALF, BN], layout)


# --------------------------------------------------------------------------------------
# Load and MMA: two cursors over the S-tile stream, QK^T running S_SLOTS tiles ahead
# --------------------------------------------------------------------------------------


@gluon.jit
def _load_partition(k, s_tmem, o_tmem):
    """TMA producer. Walks the S-tile stream exactly as the MMA warp does and loads
    every operand at the point the MMA warp first needs it: Q_h at a tile's first S
    tile of that half, K_j at QK(t), V_j at PV(t). K and V share one ring of slots."""
    p = k.problem
    qk = Cursor.start(p)
    pv = Cursor.start(p)
    kv_count = 0
    for _ in gl.static_range(S_SLOTS):
        if qk.valid(p):
            kv_count = _load_for_qk(k, qk, kv_count)
            qk = qk.next(p)
    while pv.valid(p):
        if pv.first_use():
            kv_count = _load_kv(k, k.v_desc, pv, kv_count)
        if qk.valid(p):
            kv_count = _load_for_qk(k, qk, kv_count)
            qk = qk.next(p)
        pv = pv.next(p)


@gluon.jit
def _load_for_qk(k, cur, kv_count):
    """Operands of QK(cur): Q_h at the tile's first key tile, K_j at its first use."""
    if cur.j == 0:
        slot, uses = cur.q_slot()
        _wait_free(k.bars.q_free.index(slot), uses)
        ready = k.bars.q_ready.index(slot)
        mbarrier.expect(ready, k.q_desc.block_type.nbytes)
        coords = k.problem.coords(cur.info, cur.half)
        tma.async_copy_global_to_shared(k.q_desc, coords, ready, k.smem.q.index(slot))
    if cur.first_use():
        kv_count = _load_kv(k, k.k_desc, cur, kv_count)
    return kv_count


@gluon.jit
def _load_kv(k, desc, cur, kv_count):
    """Load key tile ``cur.j`` of ``desc`` (K or V) into the next ring slot."""
    slot = kv_count % k.KV_BUFFERS
    _wait_free(k.bars.kv_free.index(slot), kv_count // k.KV_BUFFERS)
    ready = k.bars.kv_ready.index(slot)
    mbarrier.expect(ready, desc.block_type.nbytes)
    bh = cur.info.batch_head(cur.half)
    kv_head = (bh % k.problem.heads) // k.problem.group
    coords = [bh // k.problem.heads, kv_head, cur.j * k.problem.BLOCK_N, 0]
    tma.async_copy_global_to_shared(desc, coords, ready, k.smem.kv.index(slot))
    return kv_count + 1


@aggregate
class MmaState:
    """The MMA warp's ring positions: K / V ring count, the ring slots holding the
    current K and V tiles, and o_free waits per half."""

    kv_count: gl.tensor
    k_slot: gl.tensor
    v_slot: gl.tensor
    o_count0: gl.tensor
    o_count1: gl.tensor

    @gluon.constexpr_function
    def __init__(self, kv_count, k_slot, v_slot, o_count0, o_count1):
        self.kv_count = kv_count
        self.k_slot = k_slot
        self.v_slot = v_slot
        self.o_count0 = o_count0
        self.o_count1 = o_count1


@gluon.jit
def _acquire_kv(k, kv_count):
    """Wait for the next ring slot to be loaded; returns (slot, new count)."""
    slot = kv_count % k.KV_BUFFERS
    _wait_ready(k.bars.kv_ready.index(slot), kv_count // k.KV_BUFFERS)
    return slot, kv_count + 1


@gluon.jit
def _issue_qk(k, s_tmem, cur, st):
    """S(cur) = Q_h K_j^T into slot index % 3. The slot's previous P was consumed by a
    P V MMA issued before this one (tcgen05 MMAs run in issue order)."""
    p, sm, bars = k.problem, k.smem, k.bars
    D: gl.constexpr = p.HEAD_DIM
    BN: gl.constexpr = p.BLOCK_N
    q_slot, q_uses = cur.q_slot()
    if cur.j == 0:
        _wait_ready(bars.q_ready.index(q_slot), q_uses)
    k_slot = st.k_slot
    kv_count = st.kv_count
    if cur.first_use():
        k_slot, kv_count = _acquire_kv(k, kv_count)
    s_slot = cur.index % S_SLOTS
    q_tile = _matrix(sm.q.index(q_slot), HALF, D)
    k_tile = _matrix(sm.kv.index(k_slot), BN, D)
    ready = bars.s_ready.index(2 * s_slot + cur.half)
    if p.narrow(cur.j):  # the sequence's last key tile: TAIL_N keys
        tcgen05_mma(
            q_tile, k_tile.slice(0, p.TAIL_N, dim=0).permute((1, 0)),
            s_tmem.index(s_slot).slice(0, p.TAIL_N), use_acc=False, mbarriers=[ready],
        )  # fmt: skip
    else:
        tcgen05_mma(
            q_tile, k_tile.permute((1, 0)), s_tmem.index(s_slot), use_acc=False,
            mbarriers=[ready],
        )  # fmt: skip
    if cur.last_use():
        tcgen05_commit(bars.kv_free.index(k_slot))
    if cur.j == cur.info.num_kv - 1:
        tcgen05_commit(bars.q_free.index(q_slot))
    return MmaState(kv_count, k_slot, st.v_slot, st.o_count0, st.o_count1)


@gluon.jit
def _issue_pv(k, s_tmem, o_tmem, cur, st):
    """O_h += P(cur) V_j once softmax wrote P and correction rescaled O_h."""
    p, sm, bars = k.problem, k.smem, k.bars
    D: gl.constexpr = p.HEAD_DIM
    BN: gl.constexpr = p.BLOCK_N
    v_slot = st.v_slot
    kv_count = st.kv_count
    if cur.first_use():
        v_slot, kv_count = _acquire_kv(k, kv_count)
    s_slot = cur.index % S_SLOTS
    _wait_ready(bars.p_ready.index(s_slot), cur.index // S_SLOTS)
    o_count0 = st.o_count0
    o_count1 = st.o_count1
    if cur.half == 0:
        _wait_free(bars.o_free.index(0), o_count0)
        o_count0 += 1
    else:
        _wait_free(bars.o_free.index(1), o_count1)
        o_count1 += 1
    p_tile = _p_view(s_tmem.index(s_slot), BN, k.q_desc.block_type.element_ty)
    v_tile = _matrix(sm.kv.index(v_slot), BN, D)
    o_h = o_tmem.index(cur.half)
    done = bars.o_ready.index(cur.half)
    if p.narrow(cur.j):
        tcgen05_mma(
            p_tile.slice(0, p.TAIL_N), v_tile.slice(0, p.TAIL_N, dim=0), o_h,
            use_acc=cur.j > 0, mbarriers=[done],
        )  # fmt: skip
    else:
        tcgen05_mma(p_tile, v_tile, o_h, use_acc=cur.j > 0, mbarriers=[done])
    if cur.last_use():
        tcgen05_commit(bars.kv_free.index(v_slot))
    return MmaState(kv_count, st.k_slot, v_slot, o_count0, o_count1)


@gluon.jit
def _mma_partition(k, s_tmem, o_tmem):
    """tcgen05 issuer: QK(0), QK(1), QK(2), then PV(t), QK(t + 3) for t = 0, 1, ..."""
    p = k.problem
    zero = gl.to_tensor(0)
    st = MmaState(zero, zero, zero, zero, zero)
    qk = Cursor.start(p)
    pv = Cursor.start(p)
    for _ in gl.static_range(S_SLOTS):
        if qk.valid(p):
            st = _issue_qk(k, s_tmem, qk, st)
            qk = qk.next(p)
    while pv.valid(p):
        st = _issue_pv(k, s_tmem, o_tmem, pv, st)
        if qk.valid(p):
            st = _issue_qk(k, s_tmem, qk, st)
            qk = qk.next(p)
        pv = pv.next(p)


# --------------------------------------------------------------------------------------
# Softmax
# --------------------------------------------------------------------------------------


@gluon.jit
def _softmax_partition(k, s_tmem, o_tmem, HALF_INDEX: gl.constexpr):
    """Online softmax of one 128-row half. Thread t of the 4 warps owns query row t.

    Per key tile j: S -> registers in CHUNK-column slices (row max from the TMEM load
    itself on unmasked tiles), mask (edge / tail tiles only), m = max(m, rowmax),
    alpha = exp2(qk_scale (m_old - m)) to the correction partition, P = exp2(qk_scale
    (S - m)) as fp16 over S in TMEM, l = alpha l + rowsum(P). After the last key tile
    (m, l) go to the correction partition for the epilogue.

    m is the max of the raw scores and the shift comes before the scale: S - m is exact
    near the row max (Sterbenz), so the row's largest P is exactly 1 however large the
    scores. With fma(S, qk_scale, -qk_scale m) it would be 2^(rounding residual of
    qk_scale m) -- up to 2^+-4 at |qk_scale S| ~ 1e8 -- and its fp16 rounding would put a
    relative 5e-4 error on the dominant term of O, which the backward's delta =
    rowsum(dO O) cannot tell from signal (near-one-hot rows: dq, dk off)."""
    p, sm, bars = k.problem, k.smem, k.bars
    chunk_tmem: gl.constexpr = TensorMemoryLayout([HALF, CHUNK], col_stride=1)
    s_regs: gl.constexpr = get_tmem_reg_layout(
        gl.float32, [HALF, CHUNK], chunk_tmem, gl.num_warps()
    )
    rows_layout: gl.constexpr = gl.SliceLayout(1, s_regs)

    stream_base = 0  # index of the current tile's first S tile in the stream
    alpha_base = 0  # alphas this half sent before the current tile
    s_phases = 0  # per S slot (bit), completions waited on this half's s_ready (mod 2)
    local = 0
    for tile_id in range(gl.program_id(0), p.num_tiles, gl.num_programs(0)):
        info = p.tile(tile_id)
        if HALF_INDEX < info.halves:
            rows = info.row(HALF_INDEX) + gl.arange(0, HALF, layout=rows_layout)
            limit = p.key_limit(rows)
            unmasked = p.unmasked_tiles(info, HALF_INDEX)
            m_i = gl.full([HALF], -float("inf"), gl.float32, rows_layout)
            l_i = gl.full([HALF], 0.0, gl.float32, rows_layout)
            m_i, l_i, s_phases = _softmax_steps(
                k, s_tmem, info, stream_base, alpha_base, s_phases, HALF_INDEX, 0,
                unmasked, m_i, l_i, limit, s_regs, False,
            )  # fmt: skip
            m_i, l_i, s_phases = _softmax_steps(
                k, s_tmem, info, stream_base, alpha_base, s_phases, HALF_INDEX,
                unmasked, info.num_kv, m_i, l_i, limit, s_regs, True,
            )  # fmt: skip
            _wait_free(bars.stats_free.index(HALF_INDEX), local)
            sm.stats.index(2 * HALF_INDEX).store(m_i)
            sm.stats.index(2 * HALF_INDEX + 1).store(l_i)
            mbarrier.arrive(bars.stats_ready.index(HALF_INDEX), count=1)
            local += 1
            alpha_base += info.num_kv - 1
        stream_base += info.num_kv * info.halves


@gluon.jit
def _softmax_steps(
    k, s_tmem, info, stream_base, alpha_base, s_phases, HALF_INDEX: gl.constexpr,
    start, end, m_i, l_i, limit, s_regs: gl.constexpr, MASKED: gl.constexpr,
):  # fmt: skip
    """Key tiles ``start .. end - 1`` of the online softmax; ``MASKED`` applies the
    per-row key limit (frame boundary and key tail). The sequence's last key tile is
    TAIL_N columns wide when TAIL_N < BLOCK_N."""
    p, bars = k.problem, k.bars
    for j in range(start, end):
        index = stream_base + j * info.halves + HALF_INDEX
        slot = index % S_SLOTS
        s_phases = _wait_bit(bars.s_ready.index(2 * slot + HALF_INDEX), s_phases, slot)
        if MASKED and p.narrow(j):
            m_i, l_i = _softmax_step(
                k, s_tmem, slot, j, alpha_base, HALF_INDEX, m_i, l_i, limit, s_regs,
                True, p.TAIL_N // CHUNK,
            )  # fmt: skip
        else:
            m_i, l_i = _softmax_step(
                k, s_tmem, slot, j, alpha_base, HALF_INDEX, m_i, l_i, limit, s_regs,
                MASKED, p.BLOCK_N // CHUNK,
            )  # fmt: skip
    return m_i, l_i, s_phases


@gluon.jit
def _softmax_step(
    k, s_tmem, slot, j, alpha_base, HALF_INDEX: gl.constexpr, m_i, l_i, limit,
    s_regs: gl.constexpr, MASKED: gl.constexpr, NUM_CHUNKS: gl.constexpr,
):  # fmt: skip
    """One key tile, its first NUM_CHUNKS * CHUNK columns."""
    p, bars = k.problem, k.bars
    BN: gl.constexpr = p.BLOCK_N
    rows_layout: gl.constexpr = gl.SliceLayout(1, s_regs)
    cols = gl.arange(0, CHUNK, layout=gl.SliceLayout(0, s_regs))
    s_slot = s_tmem.index(slot)
    chunks = ()
    row_max = gl.full_like(m_i, -float("inf"))
    for c in gl.static_range(NUM_CHUNKS):
        if MASKED:
            s = s_slot.slice(c * CHUNK, CHUNK).load(s_regs)
            col_limit = limit - (j * BN + c * CHUNK)
            visible = cols[None, :] < col_limit[:, None]
            s = gl.where(visible, s, -float("inf"))
            chunk_max = gl.max(s, axis=1)
        else:
            s, chunk_max = s_slot.slice(c * CHUNK, CHUNK).load_max(s_regs)
            chunk_max = gl.convert_layout(chunk_max, rows_layout)
        row_max = gl.maximum(row_max, chunk_max)
        chunks = chunks + (s,)
    m_new = gl.maximum(m_i, row_max)  # raw scores (see _softmax_partition)
    if j > 0:
        # double-buffered: this softmax may run a key tile ahead of correction
        alpha = gl.exp2((m_i - m_new) * p.qk_scale)
        alpha_index = alpha_base + j - 1
        buffer = 2 * HALF_INDEX + (alpha_index & 1)
        _wait_free(bars.alpha_free.index(buffer), alpha_index // 2)
        k.smem.alpha.index(buffer).store(alpha)
        mbarrier.arrive(bars.alpha_ready.index(buffer), count=1)
        l_i = l_i * alpha

    scale2 = float2.full_like(float2.pack(chunks[0], axis=1), p.qk_scale)
    neg_m = -m_new[:, None].broadcast_to([HALF, CHUNK])
    neg_m2 = float2.pack(neg_m, axis=1)
    sum2 = float2.full_like(neg_m2, gl.to_tensor(0.0))
    probs = ()
    for c in gl.static_range(NUM_CHUNKS):
        x2 = (float2.pack(chunks[c], axis=1) + neg_m2) * scale2
        prob = gl.exp2(float2.unpack(x2, axis=1))
        sum2 = sum2 + float2.pack(prob, axis=1)
        probs = probs + (prob.to(k.q_desc.block_type.element_ty),)
    # one TMEM store for the whole P tile: P aliases S, so the compiler syncs the
    # warpgroup before every store into the slot
    p_slot = _p_view(s_slot, BN, k.q_desc.block_type.element_ty).slice(0, NUM_CHUNKS * CHUNK)
    p_slot.store(_join_columns(probs))
    mbarrier.arrive(bars.p_ready.index(slot), count=1)
    sum_a, sum_b = float2.unpack2(sum2.sum(axis=1))
    l_i += gl.convert_layout(sum_a + sum_b, rows_layout)
    return m_new, l_i


# --------------------------------------------------------------------------------------
# Correction and epilogue
# --------------------------------------------------------------------------------------


@gluon.jit
def _correction_partition(k, s_tmem, o_tmem):
    """Default partition. Between consecutive P V MMAs of a half, O_h *= alpha; after
    the last one, O_h / l -> fp16 -> smem -> TMA store, and the row's lse."""
    p = k.problem
    D: gl.constexpr = p.HEAD_DIM
    o_layout: gl.constexpr = TensorMemoryLayout([HALF, D], col_stride=1)
    o_regs: gl.constexpr = get_tmem_reg_layout(gl.float32, [HALF, D], o_layout, gl.num_warps())
    # per half: (tiles finished, alpha_ready waits, o_ready waits)
    state0 = (0, 0, 0)
    state1 = (0, 0, 0)
    stores = 0  # TMA stores issued (the staging buffer alternates)
    for tile_id in range(gl.program_id(0), p.num_tiles, gl.num_programs(0)):
        info = p.tile(tile_id)
        second = info.halves == 2
        for j in range(1, info.num_kv):
            state0 = _rescale(k, o_tmem, 0, state0, o_regs)
            if second:
                state1 = _rescale(k, o_tmem, 1, state1, o_regs)
        state0 = _epilogue(k, o_tmem, info, 0, state0, stores, o_regs)
        stores += 1
        if second:
            state1 = _epilogue(k, o_tmem, info, 1, state1, stores, o_regs)
            stores += 1
    tma.store_wait(pendings=0)


@gluon.jit
def _rescale(k, o_tmem, HALF_INDEX: gl.constexpr, state, o_regs: gl.constexpr):
    """O_h *= alpha (per row) before the next P V MMA accumulates into it."""
    bars = k.bars
    tiles, alpha_count, o_count = state
    buffer = 2 * HALF_INDEX + (alpha_count & 1)
    _wait_ready(bars.alpha_ready.index(buffer), alpha_count // 2)
    alpha = k.smem.alpha.index(buffer).load(gl.SliceLayout(1, o_regs))
    mbarrier.arrive(bars.alpha_free.index(buffer), count=1)
    _wait_ready(bars.o_ready.index(HALF_INDEX), o_count)
    o_h = o_tmem.index(HALF_INDEX)
    o_h.store(o_h.load(o_regs) * alpha[:, None])
    mbarrier.arrive(bars.o_free.index(HALF_INDEX), count=1)
    return tiles, alpha_count + 1, o_count + 1


@gluon.jit
def _epilogue(k, o_tmem, info, HALF_INDEX: gl.constexpr, state, stores, o_regs: gl.constexpr):
    """O_h / l -> fp16 -> TMA store of the half's rows; lse = (m, log2 l)."""
    p, sm, bars = k.problem, k.smem, k.bars
    tiles, alpha_count, o_count = state
    rows_layout: gl.constexpr = gl.SliceLayout(1, o_regs)
    _wait_ready(bars.o_ready.index(HALF_INDEX), o_count)
    out = o_tmem.index(HALF_INDEX).load(o_regs)
    mbarrier.arrive(bars.o_free.index(HALF_INDEX), count=1)

    _wait_ready(bars.stats_ready.index(HALF_INDEX), tiles)
    m_i = sm.stats.index(2 * HALF_INDEX).load(rows_layout)
    l_i = sm.stats.index(2 * HALF_INDEX + 1).load(rows_layout)
    mbarrier.arrive(bars.stats_free.index(HALF_INDEX), count=1)
    out = out * (1.0 / l_i)[:, None]

    row0 = info.row(HALF_INDEX)
    staging = sm.out.index(stores & 1)
    tma.store_wait(pendings=1)  # the previous store from this staging buffer landed
    _matrix(staging, HALF, p.HEAD_DIM).store(out.to(k.o_desc.block_type.element_ty))
    fence_async_shared()
    coords = p.coords(info, HALF_INDEX)
    tma.async_copy_shared_to_global(k.o_desc, coords, staging)

    rows = row0 + gl.arange(0, HALF, layout=rows_layout)
    lse_ptrs = k.lse_ptr + info.batch_head(HALF_INDEX).to(gl.int64) * 2 * p.seq_q + rows
    gl.store(lse_ptrs, m_i, mask=rows < p.seq_q)
    gl.store(lse_ptrs + p.seq_q, gl.log2(l_i), mask=rows < p.seq_q)
    return tiles + 1, alpha_count, o_count + 1


# --------------------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------------------


@gluon.jit
def _barriers(n: gl.constexpr):
    bars = gl.allocate_shared_memory(gl.int64, [n, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(n):
        mbarrier.init(bars.index(i), count=1)
    return bars


@gluon.jit
def _softmax_half0(k, s_tmem, o_tmem):
    _softmax_partition(k, s_tmem, o_tmem, 0)


@gluon.jit
def _softmax_half1(k, s_tmem, o_tmem):
    _softmax_partition(k, s_tmem, o_tmem, 1)


@gluon.jit
def attention_fwd_kernel(
    q_desc, k_desc, v_desc, o_desc, lse_ptr, qk_scale, heads, group, batch_heads, seq_q,
    seq_kv, block, num_full, num_pairs, num_tiles,
    HEAD_DIM: gl.constexpr, BLOCK_N: gl.constexpr, TAIL_N: gl.constexpr,
    KV_BUFFERS: gl.constexpr, CAUSAL: gl.constexpr,
):  # fmt: skip
    problem = Problem(
        gl.to_tensor(qk_scale), gl.to_tensor(heads), gl.to_tensor(group),
        gl.to_tensor(batch_heads),
        gl.to_tensor(seq_q), gl.to_tensor(seq_kv), gl.to_tensor(block),
        gl.to_tensor(num_full), gl.to_tensor(num_pairs), gl.to_tensor(num_tiles),
        BLOCK_N, TAIL_N, HEAD_DIM, CAUSAL,
    )  # fmt: skip
    vector_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    q_shape: gl.constexpr = [4, 1, 1, HALF, HEAD_DIM]
    kv_shape: gl.constexpr = [KV_BUFFERS, 1, 1, BLOCK_N, HEAD_DIM]
    dtype: gl.constexpr = q_desc.block_type.element_ty  # fp16 or bf16 operands
    smem = Smem(
        gl.allocate_shared_memory(dtype, q_shape, q_desc.layout),
        gl.allocate_shared_memory(dtype, kv_shape, k_desc.layout),
        gl.allocate_shared_memory(dtype, [2, 1, 1, HALF, HEAD_DIM], o_desc.layout),
        gl.allocate_shared_memory(gl.float32, [4, HALF], vector_layout),
        gl.allocate_shared_memory(gl.float32, [4, HALF], vector_layout),
    )
    bars = Bars(
        _barriers(4), _barriers(4), _barriers(KV_BUFFERS), _barriers(KV_BUFFERS),
        _barriers(2 * S_SLOTS), _barriers(S_SLOTS), _barriers(4), _barriers(4),
        _barriers(2), _barriers(2), _barriers(2), _barriers(2),
    )  # fmt: skip
    fence_async_shared()
    s_tmem = allocate_tensor_memory(
        gl.float32, [S_SLOTS, HALF, BLOCK_N], TensorMemoryLayout([HALF, BLOCK_N], 1)
    )
    o_tmem = allocate_tensor_memory(
        gl.float32, [2, HALF, HEAD_DIM], TensorMemoryLayout([HALF, HEAD_DIM], 1)
    )
    k = Kernel(problem, q_desc, k_desc, v_desc, o_desc, lse_ptr, smem, bars, KV_BUFFERS)
    args = (k, s_tmem, o_tmem)
    gl.warp_specialize(
        [
            (_correction_partition, args),  # default partition: 4 warps
            (_softmax_half0, args),
            (_softmax_half1, args),
            (_mma_partition, args),
            (_load_partition, args),
        ],
        [4, 4, 1, 1],
        [192, 192, 32, 32],
    )


# --------------------------------------------------------------------------------------
# Host side
# --------------------------------------------------------------------------------------

BLOCK_N_DEFAULT = 128
KV_BUFFERS_DEFAULT = 6


def tail_width(seq_kv: int) -> int:
    """Columns computed for the last key tile: the next power of two >= the keys it
    holds (at least one CHUNK). Only for short sequences (at most 4 key tiles): on
    long ones the extra code path costs more than the narrow tile saves."""
    tail = seq_kv % BLOCK_N_DEFAULT
    if tail == 0 or seq_kv > 4 * BLOCK_N_DEFAULT:
        return BLOCK_N_DEFAULT
    return max(32, triton.next_power_of_2(tail))


def tma_descriptor(x: torch.Tensor, rows: int) -> TensorDescriptor:
    """4D descriptor over a [B, H, S, D] view (any strides) loading [rows, D] boxes;
    rows past S read as zeros and are dropped on store."""
    block = [1, 1, rows, x.shape[-1]]
    layout = gl.NVMMASharedLayout.get_default_for(block, GL_DTYPES[x.dtype])
    return TensorDescriptor(x, list(x.shape), list(x.stride()), block, layout)


GL_DTYPES = {torch.float16: gl.float16, torch.bfloat16: gl.bfloat16}

_SM_COUNT = {}
_GRID_LIMIT = 0  # debugging: cap the persistent grid


def sm_count(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SM_COUNT:
        _SM_COUNT[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _SM_COUNT[index]


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``q [B, H, Sq, D]``, ``k / v [B, H, Skv, D]`` fp16 -> ``out [B, H, Sq, D]`` (a
    view of ``[B, Sq, H, D]`` storage) and ``lse [B, H, 2, Sq]`` fp32: per row the max
    m of the raw scores ``q k^T`` and log2 l, l = sum exp2(scale log2(e) (q k^T - m)).

    The two are kept apart: the backward rebuilds P = exp2(c (s - m) - log2 l), and with
    large scores (a near-one-hot attention layer reaches |c s| ~ 1e8, where fp32 steps
    by 8) a single ``c m + log2 l`` would round log2 l (0 .. log2 Skv) away: P would no
    longer sum to one, dP - delta would stop cancelling and dS would blow up."""
    batch, heads, seq_q, dim = q.shape
    seq_kv = k.shape[2]
    out = torch.empty(batch, seq_q, heads, dim, device=q.device, dtype=q.dtype)
    out = out.transpose(1, 2)
    lse = torch.empty(batch, heads, 2, seq_q, device=q.device, dtype=torch.float32)
    # 256-row tiles per head; a remainder of at most 128 rows becomes a paired tail
    remainder = seq_q % (2 * HALF_ROWS)
    pair_tails = 0 < remainder <= HALF_ROWS
    num_full = seq_q // (2 * HALF_ROWS) + (0 if pair_tails or remainder == 0 else 1)
    num_pairs = triton.cdiv(batch * heads, 2) if pair_tails else 0
    num_tiles = batch * heads * num_full + num_pairs
    grid = (min(num_tiles, _GRID_LIMIT or sm_count(q.device)),)
    attention_fwd_kernel[grid](
        tma_descriptor(q, HALF_ROWS), tma_descriptor(k, BLOCK_N_DEFAULT),
        tma_descriptor(v, BLOCK_N_DEFAULT), tma_descriptor(out, HALF_ROWS),
        lse, scale * LOG2E, heads, heads // k.shape[1], batch * heads, seq_q, seq_kv,
        max(block, 1), num_full,
        num_pairs, num_tiles,
        HEAD_DIM=dim, BLOCK_N=BLOCK_N_DEFAULT, TAIL_N=tail_width(seq_kv),
        KV_BUFFERS=KV_BUFFERS_DEFAULT,
        CAUSAL=block > 0, num_warps=4,
    )  # fmt: skip
    return out, lse
