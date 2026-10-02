"""Forward and forward + backward time of every kernel over sequence length, for dense,
token-causal and block-causal attention (fp16, head_dim 64 or --dim, 16 heads, a fixed number of
tokens per call so every length does comparable work). Timed inside CUDA graphs (no
launch overhead; eager timing if a kernel cannot be captured, flagged in the CSV).

    python benchmarks/speed.py --out benchmarks/results
Writes speed.csv (speed_d{D}.csv for --dim D != 64): mode, seq, kernel, pass, ms, tflops, graph.
"""

import argparse
import csv
from pathlib import Path

import torch

from kernels import KERNELS

HEADS, DIM, TOKENS, FRAME = 16, 64, 32768, 256


def visible_fraction(seq, mode):
    if mode == "dense":
        return 1.0
    size = 1 if mode == "causal" else FRAME
    frames = torch.arange(seq) // size
    return float((frames[None, :] <= frames[:, None]).float().mean())


def time_ms(fn, iters=20):
    """Mean ms per call (best of 5 rounds), captured in one CUDA graph when possible."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            for _ in range(iters):
                fn()
        graph.replay()
        torch.cuda.synchronize()
        captured = True
    except Exception:  # noqa: BLE001 -- eager fallback
        torch.cuda.synchronize()
        captured = False
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(5):
        start.record()
        if captured:
            graph.replay()
        else:
            for _ in range(iters):
                fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters)
    return best, captured


def inputs(seq, dim):
    batch = max(1, TOKENS // seq)
    gen = torch.Generator(device="cuda").manual_seed(0)

    def projection():  # [B, S, H, D] storage viewed as [B, H, S, D], as models make them
        x = torch.randn(batch, seq, HEADS, dim, device="cuda", generator=gen, dtype=torch.half)
        return x.transpose(1, 2)

    return projection(), projection(), projection(), batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmarks/results")
    parser.add_argument("--seqs", default="256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--modes", default="dense,causal,block")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "speed.csv" if args.dim == DIM else f"speed_d{args.dim}.csv"
    file = open(out_dir / name, "w", newline="")
    writer = csv.DictWriter(
        file, fieldnames=["mode", "seq", "kernel", "pass", "ms", "tflops", "graph"]
    )
    writer.writeheader()
    for mode in args.modes.split(","):
        for seq in (int(s) for s in args.seqs.split(",")):
            q, k, v, batch = inputs(seq, args.dim)
            flops = 4 * batch * HEADS * seq * seq * args.dim * visible_fraction(seq, mode)
            dout = torch.randn_like(q)
            for name, (fn, modes) in KERNELS.items():
                if mode not in modes:
                    continue
                leaves = [t.detach().clone().requires_grad_() for t in (q, k, v)]
                try:
                    with torch.no_grad():
                        fwd, g1 = time_ms(lambda: fn(q, k, v, mode, FRAME))
                    both, g2 = time_ms(lambda: fn(*leaves, mode, FRAME).backward(dout))
                except Exception as error:  # noqa: BLE001
                    print(f"{mode} {seq} {name}: failed ({str(error).splitlines()[0][:80]})")
                    continue
                for label, ms, graph, work in (
                    ("fwd", fwd, g1, flops),
                    ("fwd+bwd", both, g2, 3.5 * flops),
                ):
                    writer.writerow(
                        {
                            "mode": mode,
                            "seq": seq,
                            "kernel": name,
                            "pass": label,
                            "ms": ms,
                            "tflops": work / ms / 1e9,
                            "graph": graph,
                        }
                    )
                file.flush()
                print(
                    f"{mode:6s} {seq:6d} {name:22s} fwd {fwd:8.3f} ms  fwd+bwd {both:8.3f} ms"
                    f"{'' if g1 and g2 else '  (eager)'}",
                    flush=True,
                )
            del q, k, v, dout
            torch.cuda.empty_cache()
    file.close()


if __name__ == "__main__":
    main()
