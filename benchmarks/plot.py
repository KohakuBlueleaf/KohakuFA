"""Figures for the README from precision.py / speed.py results.

python benchmarks/plot.py --results benchmarks/results --out docs/images
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.size"] = 13
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

STYLE = {  # kernel -> (color, marker); KohakuFA drawn on top
    "KohakuFA": ("#d62728", "o"),
    "KohakuFA (mask)": ("#d62728", "o"),
    "KohakuFA (varlen)": ("#8b0000", "*"),
    "FA4 (varlen)": ("#9467bd", "D"),
    "SDPA (enable_gqa)": ("#17becf", "s"),
    "FA2 (SDPA flash)": ("#1f77b4", "s"),
    "FA4": ("#9467bd", "D"),
    "cuDNN (SDPA)": ("#2ca02c", "^"),
    "mem-efficient (SDPA)": ("#ff7f0e", "v"),
    "flex": ("#7f7f7f", "x"),
}
TENSORS = ("out", "dq", "dk", "dv")
METRICS = {
    "rel_err": "relative error  ||x - x64|| / ||x64||",
    "max_abs_err": "max absolute error  max|x - x64|",
    "one_minus_cos": "1 - cosine(x, x64)",
}


def read(path):
    with open(path) as file:
        return list(csv.DictReader(file))


def line(ax, xs, ys, kernel):
    color, marker = STYLE.get(kernel, ("black", "."))
    ax.plot(
        xs,
        ys,
        marker=marker,
        color=color,
        label=kernel,
        ms=4,
        lw=2.4 if kernel == "KohakuFA" else 1.3,
        zorder=3 if kernel == "KohakuFA" else 2,
    )


def precision_figures(rows, out, tag):
    for metric, label in METRICS.items():
        fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
        for r, mode in enumerate(("dense", "causal")):
            for c, tensor in enumerate(TENSORS):
                ax = axes[r, c]
                series = defaultdict(list)
                for row in rows:
                    if row["mode"] == mode and row["tensor"] == tensor:
                        if metric == "one_minus_cos":
                            value = (
                                max(1 - float(row["cos"]), 1e-12)
                                if row["cos"] != "nan"
                                else float("nan")
                            )
                        else:
                            value = float(row[metric])
                        series[row["kernel"]].append((float(row["scale"]), min(value, 1e6)))
                for kernel in sorted(series, key=lambda k: k == "KohakuFA"):
                    xs, ys = zip(*sorted(series[kernel]))
                    line(ax, xs, ys, kernel)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(True, which="major", alpha=0.3)
                ax.set_title(f"{mode}: {tensor}")
                if r == 1:
                    ax.set_xlabel("largest |logit|")
                if c == 0:
                    ax.set_ylabel(label)
        axes[0, 0].legend(fontsize=12)
        fig.suptitle(
            f"Error against fp64 on the same fp16 inputs ({label.split('  ')[0]}); lower is better"
        )
        fig.tight_layout()
        fig.savefig(out / f"precision_{metric}{tag}.png", dpi=130)
        plt.close(fig)


def violin_figure(per_row, out, tag):
    keys = sorted({(m, s) for (m, s, _, _) in per_row if m == "dense"}, key=lambda x: x[1])
    kernels = [k for k in STYLE if any(key[2] == k for key in per_row)]
    fig, axes = plt.subplots(len(keys), 3, figsize=(16, 3.6 * len(keys)), squeeze=False)
    for r, (mode, scale) in enumerate(keys):
        for c, tensor in enumerate(("dq", "dk", "dv")):
            ax = axes[r, c]
            data = [
                per_row[(mode, scale, k, tensor)].clamp(1e-8, 1e6).log10().numpy()
                for k in kernels
                if (mode, scale, k, tensor) in per_row
            ]
            names = [k for k in kernels if (mode, scale, k, tensor) in per_row]
            parts = ax.violinplot(data, showmedians=True, widths=0.85)
            for body, name in zip(parts["bodies"], names):
                body.set_facecolor(STYLE[name][0])
                body.set_alpha(0.55)
            ax.set_xticks(
                range(1, len(names) + 1),
                [n.replace(" (SDPA", "\n(SDPA") for n in names],
                fontsize=10,
            )
            ax.set_ylabel("log10 per-row relative error")
            ax.set_title(f"{mode}, |logit| ~ {scale:.0e}: {tensor} rows")
            ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Per-row relative error against fp64 (query rows for dq, key rows for dk / dv)")
    fig.tight_layout()
    fig.savefig(out / f"precision_rows{tag}.png", dpi=130)
    plt.close(fig)


def speed_figures(rows, out):
    by = {(r["mode"], int(r["seq"]), r["kernel"], r["pass"]): float(r["ms"]) for r in rows}
    tf = {(r["mode"], int(r["seq"]), r["kernel"], r["pass"]): float(r["tflops"]) for r in rows}
    eager = {
        (r["mode"], int(r["seq"]), r["kernel"], r["pass"]) for r in rows if r["graph"] == "False"
    }
    modes = ("dense", "causal", "block")
    titles = {"dense": "dense", "causal": "token causal", "block": "block causal (frame 256)"}
    for kind in ("speedup", "tflops", "time"):
        fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True)
        for r, pass_ in enumerate(("fwd", "fwd+bwd")):
            for c, mode in enumerate(modes):
                ax = axes[r, c]
                # FA2 has no block-causal mask: its dense time is the block baseline
                base_mode = "dense" if mode == "block" else mode
                kernels = sorted(
                    {k for (m, _, k, p) in by if m == mode and p == pass_},
                    key=lambda k: k == "KohakuFA",
                )
                for kernel in kernels:
                    pts = []
                    for (m, seq, k, p), ms in sorted(by.items()):
                        if m != mode or k != kernel or p != pass_ or (m, seq, k, p) in eager:
                            continue
                        base = by.get((base_mode, seq, "FA2 (SDPA flash)", p))
                        if kind == "speedup" and base:
                            pts.append((seq, base / ms))
                        elif kind == "time":
                            pts.append((seq, ms))
                        elif kind == "tflops":
                            pts.append((seq, tf[(m, seq, k, p)]))
                    if pts:
                        xs, ys = zip(*pts)
                        line(ax, xs, ys, kernel)
                ax.set_xscale("log", base=2)
                ax.grid(True, which="major", alpha=0.3)
                if kind == "speedup":
                    ax.axhline(1.0, color="black", lw=0.8, ls="--")
                    ax.set_ylabel(f"{pass_}: speedup over FA2 (x)" if c == 0 else "")
                elif kind == "tflops":
                    ax.set_ylabel(f"{pass_}: effective TFLOPS" if c == 0 else "")
                else:
                    ax.set_yscale("log")
                    ax.set_ylabel(f"{pass_}: ms per call" if c == 0 else "")
                suffix = " (vs FA2 dense)" if mode == "block" and kind == "speedup" else ""
                ax.set_title(f"{titles[mode]}: {pass_}{suffix}")
                if r == 1:
                    ax.set_xlabel("sequence length")
        axes[0, 0].legend(fontsize=12)
        what = {
            "speedup": "speedup over PyTorch SDPA flash (FA2), higher is better",
            "tflops": "effective TFLOPS (visible FLOPs only), higher is better",
            "time": "time per call, lower is better",
        }[kind]
        fig.suptitle(f"B300, fp16, D = 64, 16 heads, 32k tokens per call: {what}")
        fig.tight_layout()
        fig.savefig(out / f"speed_{kind}.png", dpi=130)
        plt.close(fig)


def features_figure(rows, out):
    benches = list(dict.fromkeys(r["bench"] for r in rows))
    titles = {"bf16 dense": "bf16 dense", "bf16 causal": "bf16 causal",
              "gqa 16/4": "GQA 16 / 4 heads", "key padding": "key-padding mask",
              "varlen": "varlen"}  # fmt: skip
    fig, axes = plt.subplots(2, len(benches), figsize=(5.0 * len(benches), 8.2), sharex=True)
    for c, bench in enumerate(benches):
        for r, pass_ in enumerate(("fwd", "fwd+bwd")):
            ax = axes[r, c]
            kernels = sorted({x["kernel"] for x in rows if x["bench"] == bench},
                             key=lambda k: k.startswith("KohakuFA"))  # fmt: skip
            for kernel in kernels:
                pts = sorted((int(x["seq"]), float(x["tflops"])) for x in rows
                             if x["bench"] == bench and x["kernel"] == kernel
                             and x["pass"] == pass_ and x["graph"] == "True")  # fmt: skip
                if pts:
                    xs, ys = zip(*pts)
                    line(ax, xs, ys, kernel)
            ax.set_xscale("log", base=2)
            ax.grid(True, alpha=0.3)
            ax.set_title(titles.get(bench, bench) if r == 0 else "", fontsize=14)
            if c == 0:
                ax.set_ylabel(f"{pass_}: effective TFLOPS")
            if r == 1:
                ax.set_xlabel("sequence length S")
            ax.legend(fontsize=10)
    fig.suptitle("B300, D = 64, 16 heads (fp16 unless noted): effective TFLOPS, higher is better")
    fig.tight_layout()
    fig.savefig(out / "features_tflops.png", dpi=130)
    plt.close(fig)


def head_dims_figure(rows, out):
    """Effective TFLOPS over head dim: 2 x 2 small multiples (dense / causal x fwd /
    fwd+bwd), one line per kernel over the head dims it runs."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), sharex=True)
    for c, mode in enumerate(("dense", "causal")):
        for r, pass_ in enumerate(("fwd", "fwd+bwd")):
            ax = axes[r, c]
            kernels = sorted({x["kernel"] for x in rows} - {"mem-efficient (SDPA)"},
                             key=lambda k: k.startswith("KohakuFA"))  # fmt: skip
            for kernel in kernels:
                pts = sorted((int(x["dim"]), float(x["tflops"])) for x in rows
                             if x["mode"] == mode and x["kernel"] == kernel
                             and x["pass"] == pass_ and x["tflops"])  # fmt: skip
                if pts:
                    xs, ys = zip(*pts)
                    line(ax, xs, ys, kernel)
            ax.set_xticks([64 * i for i in range(1, 9)])
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{mode}: {pass_}", fontsize=14)
            if c == 0:
                ax.set_ylabel(f"{pass_}: effective TFLOPS")
            if r == 1:
                ax.set_xlabel("head dim D")
            ax.legend(fontsize=10)
    fig.suptitle("B300, fp16, S = 4096: effective TFLOPS by head dim")
    fig.tight_layout()
    fig.savefig(out / "head_dims_tflops.png", dpi=130)
    plt.close(fig)


def tuner_figure(rows, out):
    """Per shape: time of the planner's top pick, the tuned pick (shortlist timed) and
    the best of a wide sweep, relative to the default configuration (lower is better)."""
    labels = [f"{'fwd' if r['kind'] == 'forward' else 'bwd'} D={r['dim']}\nS={r['seq']}{' causal' if r['causal'] == '1' else ''}"
              for r in rows]  # fmt: skip
    xs = range(len(rows))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.axhline(1.0, color="gray", lw=1.5, ls="--", label="default configuration")
    series = (("planner_top1_ms", "planner's top pick (no timing)", "#9467bd", "s"),
              ("tuned_ms", "tuned (planner shortlist, timed)", "#d62728", "o"),
              ("sweep_best_ms", "best of a wide sweep (50-110 timed)", "black", "x"))  # fmt: skip
    for key, label, color, marker in series:
        ys = [float(r[key]) / float(r["default_ms"]) for r in rows]
        ax.plot(xs, ys, marker=marker, color=color, ls="none", ms=11 if marker == "o" else 9,
                mew=2, label=label)  # fmt: skip
    ax.set_xticks(list(xs), labels, fontsize=11)
    ax.set_ylabel("time relative to the default")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=11, loc="lower left")
    fig.suptitle("B300, fp16, 32k tokens per call: block-size tuning (lower is better)")
    fig.tight_layout()
    fig.savefig(out / "tuner.png", dpi=130)
    plt.close(fig)


def _main_kernels(rows):
    """Speed figures leave out the memory-efficient SDPA kernel (far behind everywhere)."""
    return [r for r in rows if r["kernel"] != "mem-efficient (SDPA)"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="benchmarks/results")
    parser.add_argument("--out", default="docs/images")
    args = parser.parse_args()
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(results.glob("precision*.csv")):
        tag = path.stem.removeprefix("precision")
        precision_figures(read(path), out, tag)
        rows_path = results / f"rows{tag}.pt"
        if rows_path.exists():
            violin_figure(torch.load(rows_path), out, tag)
    if (results / "speed.csv").exists():
        speed_figures(_main_kernels(read(results / "speed.csv")), out)
    if (results / "features.csv").exists():
        features_figure(_main_kernels(read(results / "features.csv")), out)
    if (results / "tuner.csv").exists():
        tuner_figure(read(results / "tuner.csv"), out)
    if (results / "head_dims.csv").exists():
        head_dims_figure(read(results / "head_dims.csv"), out)


if __name__ == "__main__":
    main()
