# Softmax precision in flash attention

Flash attention never stores the attention matrix `P`. The forward keeps a few numbers
per query row, and the backward **recomputes** `P` from the scores and those numbers. When
attention logits are ordinary (|logit| up to ~1e3) any reasonable bookkeeping works. When
a layer learns near-one-hot attention, logits reach 1e5 to 1e8, and fp32 itself starts to
run out of resolution: at |x| ~ 6e7 consecutive fp32 values are 4 apart. Three things then
go wrong in common implementations, each silently, in the **backward** only (the forward
output stays correct, so the loss looks fine).

Notation: `S = q k^T` (raw scores), `c = scale * log2(e)`, `m` the row max, `l` the row
sum, `P = exp2(c (S - m)) / l`, `delta = rowsum(dO * O)`, `dS = P * (dP - delta)`.

## 1. One fp32 log-sum-exp per row

The usual forward saves `lse = c*m + log2(l)` as **one** fp32 number, and the backward
rebuilds `P = exp2(c*S - lse)`.

With `c*m ~ 6e7`, fp32 steps by 4, and `log2(l)` (between 0 and log2 of the key count,
so at most ~8 here) is mostly rounded away. The rebuilt `P` no longer sums to one: it is
off by up to `2^+-2` per row. `dV = P^T dO` is wrong directly, and `dS = P (dP - delta)`
stops cancelling, so `dQ` and `dK` blow up.

**Fix:** keep the two apart. The forward stores `m` (raw score units) and `log2(l)` as two
fp32 planes (`lse` is `[B, H, 2, Sq]`), and the backward rebuilds

```python
# sm100/bwd.py, P warpgroup: P^T = exp2(c (S^T - m) - log2 l)
x2 = float2.fma(s2 + m2, scale2, log_l2)  # m2 = -m, log_l2 = -log2(l)
prob = gl.exp2(float2.unpack(x2, axis=1))
```

Seen in: FlashAttention-4 (`flash_fwd_sm100.py`: `lse = (row_max*scale_log2 +
log2(row_sum)) * LN2`, then `flash_bwd_preprocess.py`: `lse_log2 = lse * LOG2_E`, then
`flash_bwd_sm100.py`: `exp2(fma(S, scale_log2, -lse_log2))`; two extra roundings), and
measured in PyTorch SDPA's cuDNN and memory-efficient backends (the latter even with
fp32 inputs).

## 2. Scaling before shifting

The usual forward computes the exponent as one fused multiply-add,
`fma(S, c, -c*m)`. `c*m` is rounded to fp32, so for the row's largest score the exponent
is not 0 but the rounding residual (up to +-2 at these magnitudes): the dominant `P` is
`2^residual`, not 1.

That alone is harmless for the forward (it cancels in `O / l`). But `P` goes through the
P.V tensor-core MMA in fp16 / bf16, and rounding a `P` of ~1.3 puts a 5e-4 relative
error on the dominant term of `O`. The backward's `delta = rowsum(dO * O)` is then off by
5e-4 of `|dO||V|`, while in a near-one-hot row the true `dP_j - delta` is tiny (it is
`sum_i P_i (dP_j - dP_i)` over the leftover probability mass). The rounding error
dominates and `dQ`, `dK` are wrong.

**Fix:** shift, then scale. `S - m` is exact near the row max (Sterbenz), so the dominant
`P` is exactly 1 and fp16 rounding only touches the small entries:

```python
# sm100/fwd.py, softmax step: m is the max of the raw scores
m_new = gl.maximum(m_i, row_max)
alpha = gl.exp2((m_i - m_new) * p.qk_scale)          # rescale of O and l
x2 = (float2.pack(chunks[c], axis=1) + neg_m2) * scale2  # (S - m) * c, not fma(S, c, -c m)
prob = gl.exp2(float2.unpack(x2, axis=1))
```

Cost: one extra packed f32x2 op per score in the forward and in the backward (about 5%
of attention time).

FlashAttention-4 uses `fma(S, scale_log2, max_offset - row_max*scale_log2)`
(`softmax.py`, `scale_subtract_rowmax`) and additionally lets the row max lag behind the
true max to skip rescales (`rescale_threshold`), so the dominant `P` is generally not 1.
Its preprocess has an opt-in fix for `delta` (`o_lo`, the rounding residual of `O`), but
it is only enabled on the sparse-MLA path. On its own it would not be enough: with the
scale-before-shift exponent, a more precise `O` is *inconsistent* with the `P` the backward
rebuilds, and in our emulation it made `dQ` / `dK` worse (cosine 0.35).

## 3. Padded keys in the backward

When `Skv` is not a multiple of the key tile, the last tile has zero-filled `K` / `V`
rows. Their score is 0, and many kernels skip masking them in the backward ("zero `K`
rows add nothing to `dQ`"). But their rebuilt `P` is `exp2(-c*m - log2 l)`, which
overflows to `inf` for a query row whose real scores are all negative, and then
`dQ += dS K` computes `inf * 0 = NaN`. `dK` / `dV` of those rows are dropped at the store,
so only `dQ` shows it.

**Fix:** the key tile that holds padded keys runs the masked path, with the padded keys
invisible to every query; full tiles keep the unmasked fast path.

## How it was found

A video model fine-tuning a pretrained DINOv3 ViT in fp16 showed a creeping gradient
norm and a falling fp16 loss scale in the second half of training, while the loss looked
normal. Comparing fp16 gradients with fp64 on the same weights and batch, module by
module in backward order, located the divergence inside the first trunk layer's
attention backward: every layer above it matched to cosine 0.999. That layer's attention
is near one-hot (pretrained DINOv3 already has logits ~1e5 there), and during training its
logits grew to 5e7. An emulation of the kernel's arithmetic in torch, toggling one
rounding at a time, isolated the three causes above. See the README's production case
for the plots.

## Reproducing

* `benchmarks/precision.py`: synthetic sweep of logit magnitude, every kernel against
  fp64 on the same fp16 inputs (relative error, max absolute error, cosine, per-row
  error distributions).
* `tests/test_attention.py::test_huge_logits` and `::test_padded_keys_with_all_negative_scores`:
  regression tests for bugs 1, 2 and 3.
