"""Figures for the README's production case: the training run that exposed the bugs.

Reads summary CSVs in benchmarks/results/production (exported from the internal run; no
weights or activations): the run's grad norm / loss scale, the first trunk layer's logit
growth, that layer's attention backward through every kernel against fp64, whole-model
gradients against fp64, and kernel time at the run's attention call sites.

    python benchmarks/plot_production.py --results benchmarks/results/production --out docs/images
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from plot import STYLE  # noqa: E402

STYLE = {**STYLE, "KohakuFA (pre-fix)": ("#e377c2", "P")}


def read(path):
    with open(path) as file:
        return list(csv.DictReader(file))


def smooth(xs, ys, window):
    out = []
    for i in range(len(ys)):
        chunk = [y for y in ys[max(0, i - window) : i + 1] if y == y]
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return xs, out


def training_figure(results, out):
    run = read(results / "training_run.csv")
    logits = read(results / "layer0_logits.csv")
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    ax = axes[0]
    steps = [int(r["step"]) for r in run]
    gnorm = [float(r["grad_norm"]) for r in run]
    scale = [float(r["loss_scale"]) for r in run]
    ax.plot(steps, gnorm, color="#1f77b4", alpha=0.15, lw=0.6)
    ax.plot(*smooth(steps, gnorm, 200), color="#1f77b4", lw=2, label="grad norm (smoothed)")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("gradient norm (before clipping)", color="#1f77b4")
    twin = ax.twinx()
    twin.plot(steps, scale, color="#d62728", lw=1.4, label="fp16 loss scale")
    twin.set_yscale("log", base=2)
    twin.set_ylabel("fp16 loss scale", color="#d62728")
    ax.set_title("Symptom: the production run (fp16, cuDNN attention)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    s = [int(r["step"]) for r in logits]
    ax.plot(
        s, [float(r["max_logit"]) for r in logits], "o-", color="black", lw=2, label="max |logit|"
    )
    ax.plot(s, [float(r["max_q"]) for r in logits], "s--", color="#9467bd", label="max |q|")
    ax.plot(s, [float(r["max_k"]) for r in logits], "^--", color="#8c564b", label="max |k|")
    ax.set_yscale("log")
    ax.set_xlabel("training step (0 = pretrained DINOv3)")
    ax.set_title("Cause: first-layer attention logits grow 1000x (no QK-norm)")
    for r in logits:
        ax.annotate(
            f"{float(r['one_hot_rows']):.0%} rows\none-hot",
            (int(r["step"]), float(r["max_logit"])),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
        )
    ax.legend(fontsize=8, loc="center left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "production_training.png", dpi=130)
    plt.close(fig)


def precision_figure(results, out):
    rows = read(results / "precision_stage0.csv")
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), sharex=True)
    logit_at = {int(r["step"]): float(r["max_logit"]) for r in rows}
    for c, tensor in enumerate(("dq", "dk", "dv")):
        for r, (metric, label) in enumerate(
            (("rel_err", "relative error vs fp64"), ("max_abs_err", "max absolute error vs fp64"))
        ):
            ax = axes[r, c]
            series = defaultdict(list)
            for row in rows:
                if row["tensor"] == tensor:
                    series[row["kernel"]].append((int(row["step"]), min(float(row[metric]), 1e6)))
            for kernel in sorted(series, key=lambda k: k == "KohakuFA"):
                xs, ys = zip(*sorted(series[kernel]))
                color, marker = STYLE.get(kernel, ("black", "."))
                ax.plot(
                    xs,
                    ys,
                    marker=marker,
                    color=color,
                    label=kernel,
                    ms=5,
                    lw=2.6 if kernel == "KohakuFA" else 1.3,
                    ls="--" if "pre-fix" in kernel else "-",
                )
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{tensor}: {label}")
            if r == 1:
                ax.set_xlabel("checkpoint step (max |logit| above)")
                ax.set_xticks(
                    sorted(logit_at),
                    [f"{s // 1000}k\n{logit_at[s]:.0e}" for s in sorted(logit_at)],
                    fontsize=8,
                )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "The production layer's attention backward (real activations, fp16, loss-scaled dO) "
        "against fp64 on the same inputs"
    )
    fig.tight_layout()
    fig.savefig(out / "production_precision.png", dpi=130)
    plt.close(fig)


def model_figure(results, out):
    path = results / "model_gradients.csv"
    if not path.exists():
        return
    rows = read(path)
    groups = [
        g
        for g in ("enc.other", "enc.trunk.norm+ls", "enc.trunk.attn", "dec.self_attn")
        if any(r["group"] == g for r in rows)
    ]
    names = {
        "enc.other": "embeddings + CLS / register tokens",
        "enc.trunk.norm+ls": "trunk norms + layer scales",
        "enc.trunk.attn": "trunk attention",
        "dec.self_attn": "decoder self-attention",
    }
    fig, axes = plt.subplots(2, len(groups), figsize=(4.4 * len(groups), 7.5), sharex=True)
    for c, group in enumerate(groups):
        for r, (metric, label) in enumerate(
            (("cos", "cosine with fp64"), ("rel_err", "relative error vs fp64"))
        ):
            ax = axes[r, c]
            for attention, color in (
                ("cuDNN (SDPA)", STYLE["cuDNN (SDPA)"][0]),
                ("KohakuFA", STYLE["KohakuFA"][0]),
            ):
                pts = sorted(
                    (int(x["step"]), float(x[metric]))
                    for x in rows
                    if x["group"] == group and x["attention"] == attention
                )
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(
                        xs,
                        ys,
                        "o-",
                        color=color,
                        lw=2.4 if attention == "KohakuFA" else 1.4,
                        label=f"fp16 + {attention}",
                    )
            if metric == "rel_err":
                ax.set_yscale("log")
            ax.set_title(f"{names[group]}: {label}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if r == 1:
                ax.set_xlabel("checkpoint step")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Whole-model gradient of one training step (fp16 autocast) against fp64, same weights and batch"
    )
    fig.tight_layout()
    fig.savefig(out / "production_model_gradients.png", dpi=130)
    plt.close(fig)


def visible_fraction(s_q, s_kv, block):
    if block == 0:
        return 1.0
    q = torch.arange(s_q)[:, None] // block
    k = torch.arange(s_kv)[None, :] // block
    return float((k <= q).float().mean())


def speed_figure(results, out):
    """Small multiples: one panel per call site and pass, horizontal bars of effective
    TFLOPS (visible FLOPs only: block-causal is not credited for skipped tiles), sorted,
    every bar labeled with its kernel and value."""
    path = results / "speed_callsites.csv"
    if not path.exists():
        return
    # FA4 block-causal (mask_mod) has a wrong backward; FA2 has no block-causal mode
    rows = [
        r
        for r in read(path)
        if r["graph"] == "True"
        and r["mode"] != "dense"
        or (r["graph"] == "True" and r["block"] == "0")
    ]
    rows = [
        r for r in rows if not (r["block"] != "0" and r["kernel"] in ("FA4", "FA2 (SDPA flash)"))
    ]
    sites = list(dict.fromkeys(r["site"] for r in rows))
    passes = ("fwd", "fwd+bwd")
    fig, axes = plt.subplots(len(passes), len(sites), figsize=(3.1 * len(sites), 7.2))
    for r, pass_ in enumerate(passes):
        for c, site in enumerate(sites):
            ax = axes[r, c]
            entries = []
            for row in rows:
                if row["site"] != site or row["pass"] != pass_:
                    continue
                b, h, s_q, s_kv = (int(x) for x in row["shape"].split("x"))
                flops = 4 * b * h * s_q * s_kv * 64 * visible_fraction(s_q, s_kv, int(row["block"]))
                flops *= 1.0 if pass_ == "fwd" else 3.5
                entries.append((flops / float(row["ms"]) / 1e9, row["kernel"]))
            entries.sort()
            names = [e[1] for e in entries]
            values = [e[0] for e in entries]
            ax.barh(
                range(len(entries)),
                values,
                height=0.6,
                color=[STYLE.get(n, ("gray", ""))[0] for n in names],
                edgecolor=["black" if n == "KohakuFA" else "none" for n in names],
                linewidth=1.5,
            )
            for i, (v, n) in enumerate(entries):
                ax.text(v, i, f" {v:.0f}", va="center", fontsize=7.5)
            ax.set_yticks(range(len(entries)), names, fontsize=7.5)
            ax.set_xlim(0, max(values) * 1.3)
            ax.grid(True, axis="x", alpha=0.3)
            ax.set_title(f"{site}\n{pass_}", fontsize=9)
            if r == len(passes) - 1:
                ax.set_xlabel("effective TFLOPS", fontsize=8)
    fig.suptitle("B300, fp16, production attention call sites: effective TFLOPS (higher is better)")
    fig.tight_layout()
    fig.savefig(out / "production_speed.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="benchmarks/results/production")
    parser.add_argument("--out", default="docs/images")
    args = parser.parse_args()
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    training_figure(results, out)
    precision_figure(results, out)
    model_figure(results, out)
    speed_figure(results, out)


if __name__ == "__main__":
    main()
