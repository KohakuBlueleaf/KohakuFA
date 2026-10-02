"""Effective TFLOPS over head dim: forward and forward + backward of every kernel that
runs the head dim (fp16, dense and causal, 16 heads, a fixed 16k tokens per call).

    python benchmarks/head_dims.py --out benchmarks/results

Writes head_dims.csv: mode, dim, kernel, pass, ms, tflops. A kernel that rejects a head
dim is recorded with an empty time. flex attention runs in a child process: at head
dims it cannot handle it faults the CUDA context, which would poison every later
measurement in the process.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

from kernels import KERNELS
from speed import time_ms

HEADS, SEQ, TOKENS = 16, 4096, 16384


def measure(name, dim, mode):
    """(fwd ms, fwd+bwd ms) of kernel ``name``, or None if it cannot run."""
    fn, modes = KERNELS[name]
    if mode not in modes:
        return None
    batch = TOKENS // SEQ
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn(
            batch, SEQ, HEADS, dim, device="cuda", dtype=torch.half, generator=gen
        ).transpose(1, 2)
        for _ in range(3)
    )
    dout = torch.randn_like(q)
    leaves = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    try:
        with torch.no_grad():
            fwd, _ = time_ms(lambda: fn(q, k, v, mode, 256))
        both, _ = time_ms(lambda: fn(*leaves, mode, 256).backward(dout))
    except Exception:  # noqa: BLE001 -- the kernel does not support this head dim
        return None
    return fwd, both


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmarks/results")
    parser.add_argument("--dims", default="64,128,192,256,320,384,448,512")
    parser.add_argument("--child", default="")  # internal: "name,dim,mode"
    args = parser.parse_args()
    if args.child:
        name, dim, mode = args.child.split(",")
        print(json.dumps(measure(name, int(dim), mode)))
        return
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in ("dense", "causal"):
        for dim in (int(d) for d in args.dims.split(",")):
            flops = 4 * (TOKENS // SEQ) * HEADS * SEQ * SEQ * dim * (0.5 if mode == "causal" else 1)
            for name in KERNELS:
                if name == "flex":
                    proc = subprocess.run(
                        [sys.executable, __file__, "--child", f"{name},{dim},{mode}"],
                        capture_output=True, text=True,
                    )  # fmt: skip
                    lines = proc.stdout.strip().splitlines()
                    result = json.loads(lines[-1]) if lines else None
                else:
                    result = measure(name, dim, mode)
                for label, index, work in (("fwd", 0, flops), ("fwd+bwd", 1, 3.5 * flops)):
                    ms = result[index] if result else None
                    rows.append({
                        "mode": mode, "dim": dim, "kernel": name, "pass": label,
                        "ms": ms if ms else "", "tflops": work / ms / 1e9 if ms else "",
                    })  # fmt: skip
                text = (
                    "unsupported"
                    if not result
                    else f"fwd {result[0]:.3f} fwd+bwd {result[1]:.3f} ms"
                )
                print(f"{mode:6s} D={dim:3d} {name:22s} {text}", flush=True)
                torch.cuda.empty_cache()
    with open(out_dir / "head_dims.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
