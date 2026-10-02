"""Gradient precision at large attention logits: every kernel against fp64 math on the
same fp16 inputs, so any difference is the kernel's own error.

Inputs are synthetic and seeded. Queries and keys sit around a few shared directions,
scaled so the largest logit reaches ``scale``; most rows are near one-hot (as in
attention layers that learned a lookup), and the small jitter leaves near-ties between
keys of one cluster. Sweeping ``scale`` from 1e1 to 1e8 walks from ordinary attention
into the regime where fp32 resolves a logit only to a few units.

    python benchmarks/precision.py --out benchmarks/results
Writes precision.csv (one row per scale / mode / kernel / tensor) and rows.pt (the
per-row relative errors at a few scales, for the violin plots).
"""

import argparse
import csv
import math
from pathlib import Path

import torch

from kernels import KERNELS

TENSORS = ("out", "dq", "dk", "dv")


def make_inputs(scale, batch=2, heads=8, seq=1024, dim=64, clusters=16, jitter=1e-3, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    centers = torch.randn(batch, heads, clusters, dim, device="cuda", generator=gen)
    centers = centers / centers.norm(dim=-1, keepdim=True)
    pick = torch.randint(0, clusters, (batch, heads, seq), device="cuda", generator=gen)
    base = torch.gather(centers, 2, pick[..., None].expand(-1, -1, -1, dim))
    radius = math.sqrt(scale * math.sqrt(dim))  # logit = |q||k| / sqrt(D) ~ scale

    def around(t):
        noisy = t + jitter * torch.randn(t.shape, device="cuda", generator=gen)
        return (radius * noisy).half()

    q, k = around(base), around(base)
    v = torch.randn(base.shape, device="cuda", generator=gen).half()
    dout = torch.randn(base.shape, device="cuda", generator=gen).half()
    return q, k, v, dout


def reference(q, k, v, dout, mode):
    q, k, v = (t.double().requires_grad_() for t in (q, k, v))
    s = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    if mode == "causal":
        n = s.shape[-1]
        s = s.masked_fill(torch.ones(n, n, device=s.device, dtype=torch.bool).triu(1), -math.inf)
    out = s.softmax(-1) @ v
    out.backward(dout.double())
    return out.detach(), q.grad, k.grad, v.grad


def run(fn, q, k, v, dout, mode):
    q, k, v = (t.clone().requires_grad_() for t in (q, k, v))
    out = fn(q, k, v, mode, 0)
    out.backward(dout)
    return out.detach(), q.grad, k.grad, v.grad


def metrics(got, want):
    g, w = got.double().flatten(), want.double().flatten()
    diff = g - w
    finite = bool(torch.isfinite(g).all())
    return {
        "cos": float(g @ w / (g.norm() * w.norm()).clamp_min(1e-300)) if finite else math.nan,
        "norm_ratio": float(g.norm() / w.norm().clamp_min(1e-300)) if finite else math.nan,
        "rel_err": float(diff.norm() / w.norm().clamp_min(1e-300)) if finite else math.inf,
        "max_abs_err": float(diff.abs().max()) if finite else math.inf,
        "max_rel_err": (
            float(diff.abs().max() / w.abs().max().clamp_min(1e-300)) if finite else math.inf
        ),
    }


def row_errors(got, want):
    """Relative error of each row (query rows of out / dq, key rows of dk / dv)."""
    g, w = got.double().flatten(0, -2), want.double().flatten(0, -2)
    norm = w.norm(dim=-1)
    keep = norm > norm.max() * 1e-6
    return ((g - w).norm(dim=-1)[keep] / norm[keep]).float().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmarks/results")
    parser.add_argument("--scales", default="1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8")
    parser.add_argument("--violin-scales", default="1e2,1e5,1e7")
    parser.add_argument("--jitter", type=float, default=1e-3, help="key spread within a cluster")
    parser.add_argument("--tag", default="", help="suffix of the output files")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = [float(s) for s in args.scales.split(",")]
    violin = {float(s) for s in args.violin_scales.split(",")}

    rows, per_row = [], {}
    for mode in ("dense", "causal"):
        for scale in scales:
            q, k, v, dout = make_inputs(scale, jitter=args.jitter)
            want = reference(q, k, v, dout, mode)
            for name, (fn, modes) in KERNELS.items():
                if mode not in modes or name == "flex":
                    continue
                try:
                    got = run(fn, q, k, v, dout, mode)
                except Exception as error:  # noqa: BLE001
                    print(f"{mode} {scale:.0e} {name}: failed ({str(error)[:80]})")
                    continue
                for tensor, g, w in zip(TENSORS, got, want):
                    m = metrics(g, w)
                    rows.append(
                        {"mode": mode, "scale": scale, "kernel": name, "tensor": tensor, **m}
                    )
                    if scale in violin:
                        per_row[(mode, scale, name, tensor)] = row_errors(g, w)
                print(
                    f"{mode:6s} {scale:7.0e} {name:22s} "
                    + " ".join(f"{t} {r['rel_err']:.1e}" for t, r in zip(TENSORS, rows[-4:])),
                    flush=True,
                )
    with open(out_dir / f"precision{args.tag}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(per_row, out_dir / f"rows{args.tag}.pt")


if __name__ == "__main__":
    main()
