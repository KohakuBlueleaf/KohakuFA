"""Figures and tables for the DINOv3 fine-tuning precision study (dinov3_backward.py).

    python benchmarks/plot_dinov3.py --runs <dinov3_backward.py --out dir> \\
        --results benchmarks/results/dinov3 --out docs/images

Reads <runs>/<model>_<kernel>_<dtype>/{probe,train}.jsonl. Writes the reduced results
(summary.csv: per run / probe step / dtype / kernel / tensor the median and worst
relative L2 error over layers; layers.csv: every layer at the first and last probe of
each model's KohakuFA fp16 run; train.csv) and, per model, three figures:

- dinov3_<model>_layers.png    per layer, the pretrained model before any training step
- dinov3_<model>_training.png  worst layer over training (probed on the KohakuFA fp16 run)
- dinov3_<model>_runs.png      loss and gradient norm of every training run

It also prints, per model, the markdown table of kernels against the emulated bugs.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.size"] = 13
import matplotlib.pyplot as plt

KERNELS = {  # probe name -> (label, color, marker)
    "cudnn": ("cuDNN (SDPA)", "#2ca02c", "^"),
    "fa4": ("FA4", "#9467bd", "D"),
    "efficient": ("mem-efficient (SDPA)", "#ff7f0e", "v"),
    "flex": ("flex attention", "#7f7f7f", "x"),
    "kohakufa": ("KohakuFA", "#d62728", "o"),
}
FLOOR = "floor"
EMULATED = ("floor", "scale*dS", "single lse", "scale-shift", "all three")
DTYPES = ("fp16", "bf16")
MODELS = {"b": "ViT-B/16", "l": "ViT-L/16"}
TAG = [""]  # figure file name suffix (--tag)


def reference_run(model: str) -> str:
    """The run whose weights the per-layer and over-training figures are probed on."""
    return f"{model}_kohakufa_fp16"


def read_jsonl(path):
    with open(path) as file:
        return [json.loads(line) for line in file]


def write_csv(path, rows):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reduce(runs: Path, results: Path):
    """The committed results: per-layer rows only for the reference run's first and last
    probe, everything else reduced over layers."""
    results.mkdir(parents=True, exist_ok=True)
    summary, layers, train = [], [], []
    for run in sorted(p for p in runs.iterdir() if (p / "probe.jsonl").exists()):
        probes = read_jsonl(run / "probe.jsonl")
        groups = defaultdict(list)
        for row in probes:
            groups[(row["step"], row["dtype"], row["kernel"], row["tensor"])].append(row)
        for (step, dtype, kernel, tensor), rows in sorted(groups.items()):
            errors = sorted(r["rel_l2"] for r in rows)
            worst = max(rows, key=lambda r: r["rel_l2"])
            summary.append(
                {
                    "run": run.name,
                    "step": step,
                    "dtype": dtype,
                    "loss_scale": rows[0]["loss_scale"],
                    "kernel": kernel,
                    "tensor": tensor,
                    "median_rel_l2": errors[len(errors) // 2],
                    "worst_rel_l2": errors[-1],
                    "worst_layer": worst["layer"],
                    "worst_max_rel": worst["max_rel"],
                }
            )
        if run.name == reference_run(run.name.split("_")[0]):
            steps = sorted({r["step"] for r in probes})
            layers += [{"run": run.name, **r} for r in probes if r["step"] in (steps[0], steps[-1])]
        train += [{"run": run.name, **r} for r in read_jsonl(run / "train.jsonl")]
    write_csv(results / "summary.csv", summary)
    write_csv(results / "layers.csv", layers)
    write_csv(results / "train.csv", train)
    return summary, layers, train


def kernel_line(ax, xs, ys, kernel):
    label, color, marker = KERNELS[kernel]
    top = kernel == "kohakufa"
    width, order = (2.4, 3) if top else (1.4, 2)
    ax.plot(xs, ys, marker=marker, color=color, label=label, ms=4, lw=width, zorder=order)


def floor_line(ax, xs, ys):
    ax.plot(xs, ys, "k--", lw=1.2, label="correct-kernel floor (fp64 emulation)", zorder=1)


def layers_figure(layers, model, out):
    layers = [r for r in layers if r["run"] == reference_run(model)]
    first = min(r["step"] for r in layers)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
    for row, dtype in enumerate(DTYPES):
        for col, tensor in enumerate(("dq", "dk")):
            ax = axes[row, col]
            picked = [
                r
                for r in layers
                if r["step"] == first and r["dtype"] == dtype and r["tensor"] == tensor
            ]
            for kernel in (*KERNELS, FLOOR):
                rows = sorted(
                    (r for r in picked if r["kernel"] == kernel), key=lambda r: r["layer"]
                )
                xs, ys = [r["layer"] for r in rows], [r["rel_l2"] for r in rows]
                if kernel == FLOOR:
                    floor_line(ax, xs, ys)
                elif rows:
                    kernel_line(ax, xs, ys, kernel)
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            scale = picked[0]["loss_scale"]
            note = f"loss scale 2^{round(math.log2(scale))}" if dtype == "fp16" else "no loss scale"
            ax.set_title(f"{tensor}, {dtype} ({note})")
            if row == 1:
                ax.set_xlabel("layer")
            if col == 0:
                ax.set_ylabel("relative L2 error vs fp64")
    axes[0, 0].legend(fontsize=10, loc="upper right")
    fig.suptitle(
        f"Pretrained DINOv3 {MODELS[model]}, before any training step: "
        "attention backward error per layer"
    )
    fig.tight_layout()
    fig.savefig(out / f"dinov3_{model}_layers{TAG[0]}.png", dpi=130)
    plt.close(fig)


def training_figure(summary, model, out):
    rows = [r for r in summary if r["run"] == reference_run(model)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
    for row, dtype in enumerate(DTYPES):
        for col, tensor in enumerate(("dq", "dk")):
            ax = axes[row, col]
            for kernel in (*KERNELS, FLOOR):
                picked = sorted(
                    (
                        r
                        for r in rows
                        if r["dtype"] == dtype and r["tensor"] == tensor and r["kernel"] == kernel
                    ),
                    key=lambda r: r["step"],
                )
                xs, ys = [r["step"] for r in picked], [r["worst_rel_l2"] for r in picked]
                if kernel == FLOOR:
                    floor_line(ax, xs, ys)
                elif picked:
                    kernel_line(ax, xs, ys, kernel)
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            scales = {r["loss_scale"] for r in rows if r["dtype"] == "fp16"}
            fixed = f"loss scale 2^{round(math.log2(scales.pop()))}" if len(scales) == 1 else ""
            note = (fixed or "live loss scale") if dtype == "fp16" else "no loss scale"
            ax.set_title(f"{tensor}, {dtype} ({note})")
            if row == 1:
                ax.set_xlabel("training step")
            if col == 0:
                ax.set_ylabel("worst layer: relative L2 error")
    axes[0, 0].legend(fontsize=10, loc="upper right")
    fig.suptitle(f"DINOv3 {MODELS[model]} fine-tuning: attention backward error over training")
    fig.tight_layout()
    fig.savefig(out / f"dinov3_{model}_training{TAG[0]}.png", dpi=130)
    plt.close(fig)


def runs_figure(train, model, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharex=True)
    for run in sorted({r["run"] for r in train if r["run"].startswith(f"{model}_")}):
        _, kernel, dtype = run.split("_")
        label, color, _ = KERNELS[kernel]
        rows = [r for r in train if r["run"] == run]
        style = "-" if dtype == "fp16" else ":"
        steps = [r["step"] for r in rows]
        for ax, key in zip(axes, ("loss", "grad_norm")):
            ax.plot(
                steps, [r[key] for r in rows], style, color=color, lw=1.4, label=f"{label} {dtype}"
            )
    for ax, title in zip(axes, ("training loss", "gradient norm")):
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=10)
    fig.suptitle(f"DINOv3 {MODELS[model]} fine-tuning runs")
    fig.tight_layout()
    fig.savefig(out / f"dinov3_{model}_runs{TAG[0]}.png", dpi=130)
    plt.close(fig)


def attribution_table(summary, model):
    """Markdown: the reference run's first probe, worst layer, every kernel and emulation."""
    rows = [r for r in summary if r["run"] == reference_run(model)]
    print(f"\n{MODELS[model]}, pretrained (step 0), worst layer, relative L2 error vs fp64:")
    first = min(r["step"] for r in rows)
    names = [*KERNELS, *EMULATED]
    print(
        "| dtype | tensor | "
        + " | ".join(KERNELS[n][0] if n in KERNELS else f"emul: {n}" for n in names)
        + " |"
    )
    print("|" + "---|" * (2 + len(names)))
    for dtype in DTYPES:
        for tensor in ("dq", "dk"):
            cells = []
            for name in names:
                match = [
                    r
                    for r in rows
                    if r["step"] == first
                    and r["dtype"] == dtype
                    and r["tensor"] == tensor
                    and r["kernel"] == name
                ]
                cells.append(f"{match[0]['worst_rel_l2']:.1e}" if match else "-")
            print(f"| {dtype} | {tensor} | " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--results", default="benchmarks/results/dinov3")
    parser.add_argument("--out", default="docs/images")
    parser.add_argument("--tag", default="", help="figure file name suffix")
    args = parser.parse_args()
    TAG[0] = args.tag
    summary, layers, train = reduce(Path(args.runs), Path(args.results))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        if any(r["run"] == reference_run(model) for r in summary):
            layers_figure(layers, model, out)
            training_figure(summary, model, out)
            runs_figure(train, model, out)
            attribution_table(summary, model)


if __name__ == "__main__":
    main()
