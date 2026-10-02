"""Tune the kernels' configurations on this card and write the table the library ships.

    python benchmarks/tune.py --dims 64,128,256 [--out src/kohakufa/sm100/tuned]

For every head dim: forward (dense, causal) and backward (dense, causal), each the
planner's shortlist timed interleaved; writes ``<card>_d<D>.json`` (one file per head
dim, so dims can be tuned in parallel). Prints every candidate's time.

``--short``: the short-sequence buckets instead (<= 512 and <= 2048 tokens), tuned on the
call-site shapes of a real model (Foliation: ViT encoder, diffusion decoder, temporal
mixers), dense and masked summed separately; writes ``<card>_d<D>_short.json``.
"""

import argparse
import os
import re

import torch

from kohakufa.device import Device
from kohakufa.sm100 import tune

# (batch, heads, seq_q, seq_kv, block): Foliation's attention call sites, per bucket
SHORT_SHAPES = {
    "s512": {
        "dense": [
            (128, 12, 269, 269, 0),  # ViT-B encoder, spatial
            (128, 16, 269, 269, 0),  # ViT-L encoder, spatial
            (112, 12, 257, 256, 0),  # decoder, cross
        ],
        "masked": [
            (16, 12, 192, 192, 24),  # TT3D temporal mixer (B)
            (16, 16, 192, 192, 24),  # TT3D temporal mixer (L)
        ],
    },
    "s2048": {
        "dense": [(16, 12, 1799, 1799, 0)],  # decoder, joint
        "masked": [(16, 12, 1799, 1799, 257)],  # decoder, block-causal frames
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", default="64,128,192,256,320,384,448,512")
    parser.add_argument("--out", default=tune.TABLES)
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--kinds", default="forward,backward")
    parser.add_argument("--short", action="store_true")
    args = parser.parse_args()
    dev = Device.query()
    card = re.sub(r"[^a-z0-9]+", "_", dev.name.lower()).strip("_")
    os.makedirs(args.out, exist_ok=True)
    for dim in (int(d) for d in args.dims.split(",")):
        tune.clear()
        buckets = list(SHORT_SHAPES.values()) if args.short else [None]
        for kind in args.kinds.split(","):
            for causal in (False, True):
                for lists in buckets:
                    print(f"== {kind} D={dim} {'causal' if causal else 'dense'}", flush=True)
                    shapes = lists["masked" if causal else "dense"] if lists else None
                    tune.tune(kind, dim, causal=causal, seq=args.seq, dev=dev, verbose=True,
                              shapes=shapes)  # fmt: skip
                    torch.cuda.empty_cache()
        suffix = "_short" if args.short else ""
        path = os.path.join(args.out, f"{card}_d{dim}{suffix}.json")
        tune.save(path)
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
