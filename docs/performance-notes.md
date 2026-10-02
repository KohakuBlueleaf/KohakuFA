# Performance notes

Measured findings that shaped (or ruled out) kernel changes. B300, fp16, head dim 64.

## exp2 on the FMA pipe does not help the forward (2026-10-02)

FlashAttention-4 computes part of its softmax exponentials with a degree-3 polynomial on
the FMA pipe to offload the MUFU unit. Tried in KohakuFA's forward (FA4's coefficients,
range reduction by an add rounding toward -inf, one inline-PTX block per element) on 1
or 2 of the 4 score chunks of each key tile:

| emulated chunks | dense 4k | dense 16k | causal 16k |
|---|---|---|---|
| 0 (all MUFU) | 1060 TFLOPS | 966 | 897 |
| 1 | 710 | 726 | 717 |
| 2 | 374 | 377 | 360 |

Slower in every case, so the forward is not MUFU-bound at these sizes; reverted. (FA4
emulates about 1 in 10 fragments with packed f32x2 ops; a finer-grained, packed version
might break even, but it cannot be the source of the ~10% gap to FA4 at long sequences.)

## B300 rates that bound attention (2026-10-02)

Microbenchmarks (`benchmarks/out`-style, all 148 SMs):

* exp2 (MUFU): ~30 per clock per SM (B200: 16; sm_103 doubled it). FMA: ~89 of 128.
* L2 read bandwidth: ~15 TB/s with an L2-resident working set (9 to 12 TB/s from HBM).
* tcgen05 MMAs with both operands in shared memory need ~137 B/clk at N = 128 and ~205
  at N = 64, against ~128 B/clk of shared-memory bandwidth: narrow (N = 64) MMAs fed from
  smem are smem-bound. P (forward) and P^T (backward) come from tensor memory for that
  reason.

## What bounds each path

In-kernel `%clock` traces (one CTA, every partition's waits stamped) located these:

* **Forward, D = 128.** The softmax warpgroup is the critical path: ~1480 cycles per
  128 x 128 tile, of which only ~750 is the exponent loop; the rest is the TMEM load and
  row max, the alpha hand-off, the P store and the wait for the next S. With two query
  halves per CTA, tensor memory (2 S slots + 2 O of 128 columns) leaves each half one S
  slot, so softmax -> O rescale -> P V -> next QK^T is serial; FlashAttention-4 has the
  same chain but skips most O rescales by letting the row max lag, which is exactly what
  makes its dominant P differ from 1 (precision bug 2). KohakuFA rescales exactly every
  tile; one half per CTA with a 3-deep S ring hides the rescale and is ~1.2x FA4's time.
* **Backward, D = 128.** The dQ partials' TMA tensor reduce-adds cost ~2 ms of ~11 at 16k
  (skipping them: 9.1 ms, FA4 9.5). FlashAttention-4 uses 1D `cp.reduce.async.bulk`,
  which Gluon 3.7 does not expose (no shared-memory address for inline PTX). Tried and
  measured no gain: a deeper staging ring, rotated query-tile order (L2 contention),
  contiguous [16, 256] unswizzled boxes, `red.global.add.v4.f32` from registers (slower:
  15.9 ms), register-split tuning beyond removing spills.
* **Backward, D = 128, P / dS.** With P^T aliased in the S^T columns (needed above D =
  64), a separate P and dS warpgroup serialise every step; both compute warpgroups now
  produce P^T and dS^T for half the query columns each.

## Timing methodology

Node-to-node and run-to-run variance on the cluster is ~10%, larger than most effects
being measured, and a GPU heating up during a sweep slows later candidates. Compare
configurations interleaved in one process (`tune.py` does: every candidate once per
round, best of rounds), and time short calls (10 to 50 us) from CUDA graphs: eager
launch overhead swamps the differences and picked worse configurations.
