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
