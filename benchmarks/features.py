"""Speed beyond fp16 dense / causal: bf16, grouped-query attention, a key-padding mask,
and variable-length packed sequences, against the kernels that support each (B300,
head dim 64, CUDA graphs). Effective TFLOPS counts visible FLOPs only.

    python benchmarks/features.py --out benchmarks/results
Writes features.csv: bench, seq, kernel, pass, ms, tflops, graph.
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask

import kernels as K
from kohakufa import attention, attention_varlen, varlen_plan
from speed import time_ms

HEADS, DIM, TOKENS = 16, 64, 32768
FIELDS = ["bench", "seq", "kernel", "pass", "ms", "tflops", "graph"]


def bshd(batch, seq, heads, dtype, gen):
    x = torch.randn(batch, seq, heads, DIM, device="cuda", generator=gen, dtype=dtype)
    return x.transpose(1, 2)


def measure(writer, bench, seq, name, fwd, bwd, flops):
    try:
        with torch.no_grad():
            t1, g1 = time_ms(fwd)
        t2, g2 = time_ms(bwd)
    except Exception as error:  # noqa: BLE001
        print(f"{bench} {seq} {name}: failed ({str(error).splitlines()[0][:90]})")
        return
    for label, ms, graph, work in (("fwd", t1, g1, flops), ("fwd+bwd", t2, g2, 3.5 * flops)):
        writer.writerow({"bench": bench, "seq": seq, "kernel": name, "pass": label, "ms": ms,
                         "tflops": work / ms / 1e9, "graph": graph})  # fmt: skip
    print(f"{bench:10s} {seq:6d} {name:22s} fwd {t1:8.3f} ms  fwd+bwd {t2:8.3f} ms", flush=True)


def leaves(*ts):
    return [t.detach().clone().requires_grad_() for t in ts]


def dense_benches(writer, seq, gen):
    batch = max(1, TOKENS // seq)
    full = 4 * batch * HEADS * seq * seq * DIM
    # bf16, dense and causal
    q, k, v = (bshd(batch, seq, HEADS, torch.bfloat16, gen) for _ in range(3))
    dout = torch.randn_like(q)
    for mode, frac in (("dense", 1.0), ("causal", 0.5)):
        for name, (fn, modes) in K.KERNELS.items():
            if mode not in modes or name in ("flex", "mem-efficient (SDPA)"):
                continue
            a, b, c = leaves(q, k, v)
            measure(writer, f"bf16 {mode}", seq, name,
                    lambda fn=fn, mode=mode: fn(q, k, v, mode, 0),
                    lambda fn=fn, mode=mode, a=a, b=b, c=c: fn(a, b, c, mode, 0).backward(dout),
                    full * frac)  # fmt: skip
    # GQA: 16 query heads, 4 K / V heads (fp16)
    q = bshd(batch, seq, HEADS, torch.half, gen)
    k, v = (bshd(batch, seq, HEADS // 4, torch.half, gen) for _ in range(2))
    dout = torch.randn_like(q)

    def sdpa_gqa(a, b, c):
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
            return F.scaled_dot_product_attention(a, b, c, enable_gqa=True)

    gqa = {"KohakuFA": lambda a, b, c: attention(a, b, c), "SDPA (enable_gqa)": sdpa_gqa}
    if K.fa4_func is not None:
        gqa["FA4"] = lambda a, b, c: K.fa4(a, b, c, "dense", 0)
    for name, fn in gqa.items():
        a, b, c = leaves(q, k, v)
        measure(writer, "gqa 16/4", seq, name, lambda fn=fn: fn(q, k, v),
                lambda fn=fn, a=a, b=b, c=c: fn(a, b, c).backward(dout), full)  # fmt: skip


def padding_benches(writer, seq, gen):
    """Key padding: each sequence keeps a random 50 - 100% of its keys (fp16); and the
    same lengths packed (varlen)."""
    batch = max(1, TOKENS // seq)
    lengths = torch.randint(
        seq // 2, seq + 1, (batch,), generator=torch.Generator().manual_seed(seq)
    )
    lengths = lengths.cuda()
    q, k, v = (bshd(batch, seq, HEADS, torch.half, gen) for _ in range(3))
    dout = torch.randn_like(q)
    keep = torch.arange(seq, device="cuda")[None, :] < lengths[:, None]  # [B, Skv]
    mask = keep[:, None, None, :].expand(batch, 1, seq, seq)
    flops = 4 * HEADS * DIM * seq * float(lengths.sum())  # every query row, kept keys

    def sdpa(backend):
        def fn(a, b, c):
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(a, b, c, attn_mask=mask)

        return fn

    lens = lengths

    def flex_fn(
        a,
        b,
        c,
        block_mask=create_block_mask(
            lambda b_, h, qi, ki: ki < lens[b_], batch, None, seq, seq, device="cuda"
        ),
    ):
        return K._flex(a, b, c, block_mask=block_mask)

    benches = {"KohakuFA (mask)": lambda a, b, c: attention(a, b, c, mask=mask),
               "cuDNN (SDPA)": sdpa(SDPBackend.CUDNN_ATTENTION),
               "mem-efficient (SDPA)": sdpa(SDPBackend.EFFICIENT_ATTENTION),
               "flex": flex_fn}  # fmt: skip
    for name, fn in benches.items():
        a, b, c = leaves(q, k, v)
        measure(writer, "key padding", seq, name, lambda fn=fn: fn(q, k, v),
                lambda fn=fn, a=a, b=b, c=c: fn(a, b, c).backward(dout), flops)  # fmt: skip


def varlen_benches(writer, seq, gen):
    """Packed sequences of random lengths in [seq / 4, seq], 32k tokens in total (fp16):
    per-pair FLOPs 4 H D len^2."""
    rng = torch.Generator().manual_seed(seq + 1)
    lengths = []
    while sum(lengths) < TOKENS:
        lengths.append(int(torch.randint(max(1, seq // 4), seq + 1, (1,), generator=rng)))
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32, device="cuda")
    total = int(cu[-1])
    q, k, v = (torch.randn(total, HEADS, DIM, device="cuda", generator=gen, dtype=torch.half)
               for _ in range(3))  # fmt: skip
    dout = torch.randn_like(q)
    flops = sum(4 * HEADS * DIM * n * n for n in lengths)
    plan = varlen_plan(cu, cu, HEADS, HEADS, 0)
    benches = {"KohakuFA (varlen)": lambda a, b, c: attention_varlen(a, b, c, cu, cu, plan=plan)}
    if K.fa4_func is not None:
        from flash_attn.cute.interface import flash_attn_varlen_func

        def fa4_varlen(a, b, c):
            out = flash_attn_varlen_func(
                a,
                b,
                c,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=max(lengths),
                max_seqlen_k=max(lengths),
            )
            return out[0] if isinstance(out, tuple) else out

        benches["FA4 (varlen)"] = fa4_varlen
    for name, fn in benches.items():
        a, b, c = leaves(q, k, v)
        measure(writer, "varlen", seq, name, lambda fn=fn: fn(q, k, v),
                lambda fn=fn, a=a, b=b, c=c: fn(a, b, c).backward(dout), flops)  # fmt: skip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmarks/results")
    parser.add_argument("--seqs", default="512,1024,2048,4096,8192,16384")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "features.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for seq in (int(s) for s in args.seqs.split(",")):
            gen = torch.Generator(device="cuda").manual_seed(seq)
            for bench in (dense_benches, padding_benches, varlen_benches):
                bench(writer, seq, gen)
                file.flush()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
