"""Attention backward error while fine-tuning a pretrained DINOv3 ViT, in fp16 AND bf16:
PyTorch SDPA (cuDNN, memory-efficient), FlashAttention-4 and KohakuFA on the attention
calls of a real training run, with each error attributed to a named kernel bug by fp64
emulation.

Task: decode every patch token back to its 16x16 RGB patch (one linear head, MSE),
trained with ``--backend`` in ``--dtype`` (fp16 with a dynamic loss scale, or bf16).
Every ``--probe-every`` steps, on a fixed batch, each layer's attention inputs are
captured in fp32 (q, k, v and the incoming gradient dO), then for each probe dtype the
backward is recomputed by every kernel on the same rounded inputs:

- fp16: dO carries the loss scale (the run's live GradScaler scale, or ``--fp16-scale``)
- bf16: dO unscaled (bf16 training needs no loss scale)

Errors are against fp64 on those same rounded inputs, so they are the kernel's own. Next to
the kernels, fp64 emulations of a correct kernel and of each bug:

- floor        dS = P (dP - delta) rounded to the dtype before the dQ / dK matmuls, P.V
               operand and O rounded too: what any tensor-core kernel pays
- scale*dS     the softmax scale folded into dS before that rounding: fp16 loses
               log2(1 / scale) bits of range to underflow (bf16 has fp32's range)
- single lse   the backward rebuilds P from ONE fp32 c*m + log2(l): at large logits fp32
               rounds log2(l) away and P no longer sums to one (fp16 and bf16 alike)
- scale-shift  the forward exponent is fma(S, c, -c*m) instead of (S - m) * c: the row's
               largest P is 2^(rounding residual), its rounding puts error on O, and
               delta = rowsum(dO O) no longer cancels dP in near-one-hot rows (both dtypes)
- all three    the three bugs together

    python benchmarks/dinov3_backward.py --images <folder or uint8 [N, 3, H, W] .pt> \\
        --model l --backend kohakufa --dtype fp16 --steps 20000

Writes <out>/<model>_<backend>_<dtype>/probe.jsonl (one row per probe step / probe dtype /
layer / kernel / tensor) and train.jsonl (loss, grad norm, loss scale). Needs torch,
transformers, kohakufa and access to the (gated) DINOv3 weights on the Hugging Face Hub;
FlashAttention-4 is probed when ``flash_attn.cute`` is importable.
"""

import argparse
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

REPOS = {
    "b": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "l": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}
SDPA = {
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}
try:
    from flash_attn.cute.interface import flash_attn_func as fa4_func
except ImportError:
    fa4_func = None
KERNELS = (*SDPA, "kohakufa", *(("fa4",) if fa4_func else ()))
PROBED = ("cudnn", "efficient", *(("fa4",) if fa4_func else ()), "kohakufa")
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
EMULATED = ("floor", "scale*dS", "single lse", "scale-shift", "all three")
FP16_MIN_NORMAL = 2.0**-14
LOG2E = math.log2(math.e)
PATCH, SIZE = 16, 256
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# -- data and model --------------------------------------------------------------------------


def load_images(path: str) -> torch.Tensor:
    """uint8 [N, 3, SIZE, SIZE]: a saved tensor, or every image in a folder (resized and
    center-cropped)."""
    if path.endswith(".pt"):
        images = torch.load(path)
        return F.interpolate(images.float(), size=SIZE, mode="area").round().to(torch.uint8)
    from PIL import Image

    suffixes = (".jpg", ".jpeg", ".png")
    files = sorted(p for p in Path(path).rglob("*") if p.suffix.lower() in suffixes)
    out = []
    for file in files:
        image = Image.open(file).convert("RGB")
        scale = SIZE / min(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)))
        left, top = (image.width - SIZE) // 2, (image.height - SIZE) // 2
        image = image.crop((left, top, left + SIZE, top + SIZE))
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        out.append(pixels.view(SIZE, SIZE, 3))
    return torch.stack(out).permute(0, 3, 1, 2).contiguous()


def batches(images: torch.Tensor, batch: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    while True:
        x = images[torch.randint(0, images.shape[0], (batch,), generator=gen)].float() / 255
        flip = torch.rand(batch, generator=gen) < 0.5
        x[flip] = x[flip].flip(-1)
        yield ((x - MEAN) / STD).cuda()


class PatchDecoder(torch.nn.Module):
    """DINOv3 trunk (transformers, SDPA attention) + a linear head: patch token -> RGB patch."""

    def __init__(self, repo: str):
        super().__init__()
        from transformers import AutoModel

        self.trunk = AutoModel.from_pretrained(
            repo, attn_implementation="sdpa", dtype=torch.float32
        )
        self.prefix = 1 + self.trunk.config.num_register_tokens  # CLS + registers
        width = self.trunk.config.hidden_size
        self.head = torch.nn.Linear(width, PATCH * PATCH * 3)
        torch.nn.init.normal_(self.head.weight, std=width**-0.5)
        torch.nn.init.zeros_(self.head.bias)

    def loss(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.trunk(pixel_values=images).last_hidden_state[:, self.prefix :]
        target = F.unfold(images.float(), PATCH, stride=PATCH).transpose(1, 2)
        return F.mse_loss(self.head(tokens).float(), target)


# -- kernels ---------------------------------------------------------------------------------


@contextmanager
def use_kernel(name: str):
    """Route every unmasked SDPA call to ``name``: an SDPA backend, FlashAttention-4 or
    KohakuFA."""
    if name in SDPA:
        with sdpa_kernel([SDPA[name]]):
            yield
        return
    import kohakufa

    sdpa = F.scaled_dot_product_attention

    def swapped(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        if attn_mask is not None or dropout_p or is_causal or kwargs:
            return sdpa(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, **kwargs)
        autocast = torch.is_autocast_enabled("cuda")
        dtype = torch.get_autocast_dtype("cuda") if autocast else None
        if name == "kohakufa":
            return kohakufa.attention(q, k, v, scale=scale, compute_dtype=dtype)
        dtype = dtype or (q.dtype if q.dtype in DTYPES.values() else torch.float16)
        q, k, v = (t.to(dtype).transpose(1, 2) for t in (q, k, v))  # FA4: [B, S, H, D]
        out = fa4_func(q, k, v, softmax_scale=scale)
        return (out[0] if isinstance(out, tuple) else out).transpose(1, 2)

    torch.nn.functional.scaled_dot_product_attention = swapped
    try:
        yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = sdpa


# -- probe -----------------------------------------------------------------------------------


def capture(model: PatchDecoder, images: torch.Tensor) -> list[dict]:
    """Every attention call of one fp32 step (math SDPA): q, k, v and dO, in fp32."""
    calls = []
    sdpa = F.scaled_dot_product_attention

    def spy(q, k, v, *args, **kwargs):
        out = sdpa(q, k, v, *args, **kwargs)
        call = {
            "q": q.detach().float(),
            "k": k.detach().float(),
            "v": v.detach().float(),
            "scale": kwargs.get("scale") or q.shape[-1] ** -0.5,
        }
        out.register_hook(lambda g, call=call: call.setdefault("dout", g.detach().float()))
        calls.append(call)
        return out

    torch.nn.functional.scaled_dot_product_attention = spy
    try:
        with use_kernel("math"):
            model.loss(images).backward()
    finally:
        torch.nn.functional.scaled_dot_product_attention = sdpa
    model.zero_grad(set_to_none=True)
    return calls


def rounded(call: dict, dtype: torch.dtype, loss_scale: float) -> dict:
    """The call's inputs as a ``dtype`` kernel gets them; dO carries ``loss_scale``."""
    out = {n: call[n].to(dtype) for n in "qkv"}
    out["dout"] = (call["dout"] * loss_scale).to(dtype)
    out["scale"] = call["scale"]
    return out


def kernel_grads(inputs: dict, kernel: str) -> list[torch.Tensor]:
    """dq, dk, dv through ``kernel`` in the inputs' dtype, or "fp64" (math SDPA in fp64 on
    the same values)."""
    dtype = torch.float64 if kernel == "fp64" else inputs["q"].dtype
    q, k, v = (inputs[n].to(dtype).clone().requires_grad_() for n in "qkv")
    with use_kernel("math" if kernel == "fp64" else kernel):
        out = F.scaled_dot_product_attention(q, k, v, scale=inputs["scale"])
    out.backward(inputs["dout"].to(dtype))
    return [t.grad.double() for t in (q, k, v)]


def emulated_grads(inputs: dict, bugs: set[str]) -> list[torch.Tensor]:
    """dq, dk, dv of a flash-attention backward in fp64, except: P (as the P.V operand), O
    and dS rounded to the inputs' dtype (a correct kernel), plus the named ``bugs``
    ("scale*dS", "single lse", "scale-shift"). fp32 steps of the kernel are rounded to fp32."""
    dtype = inputs["q"].dtype

    def r(x):
        return x.to(dtype).double()

    def f32(x):
        return x.float().double()

    q, k, v, dout = (inputs[n].double() for n in ("q", "k", "v", "dout"))
    scale = inputs["scale"]
    c = f32(torch.tensor(scale * LOG2E, dtype=torch.float64))
    s = f32(q @ k.transpose(-1, -2))  # raw scores, fp32 accumulation
    m = s.amax(-1, keepdim=True)
    if "scale-shift" in bugs:  # fma(S, c, -c m): the row's largest exponent is a residual
        exponent = f32(s * c - f32(c * m))
    else:  # (S - m) * c: exact near the row max, the largest P is exactly 1
        exponent = f32(f32(s - m) * c)
    p_unnorm = torch.exp2(exponent)
    l = p_unnorm.sum(-1, keepdim=True)
    out = r((r(p_unnorm) @ v) / l)  # P.V operand and O in the dtype
    delta = (dout * out).sum(-1, keepdim=True)
    if "single lse" in bugs:  # P = exp2(S c - lse), lse = c m + log2 l as ONE fp32 number
        lse = f32(c * m + torch.log2(l))
        p = torch.exp2(f32(s * c - lse))
    else:  # m and log2 l kept apart
        p = torch.exp2(f32(f32(s - m) * c - torch.log2(l)))
    ds = p * (dout @ v.transpose(-1, -2) - delta)
    dv = r(p).transpose(-1, -2) @ dout
    if "scale*dS" in bugs:
        ds = r(ds * scale)
        return [ds @ k, ds.transpose(-1, -2) @ q, dv]
    ds = r(ds)
    return [ds @ k * scale, ds.transpose(-1, -2) @ q * scale, dv]


EMULATIONS = {
    "floor": set(),
    "scale*dS": {"scale*dS"},
    "single lse": {"single lse"},
    "scale-shift": {"scale-shift"},
    "all three": {"scale*dS", "single lse", "scale-shift"},
}


def errors(got: torch.Tensor, ref: torch.Tensor) -> dict:
    diff = got - ref
    return {
        "rel_l2": float(diff.norm() / ref.norm()),
        "cos": float((got * ref).sum() / (got.norm() * ref.norm()).clamp_min(1e-300)),
        "max_rel": float(diff.abs().max() / ref.abs().max()),
    }


def stats(inputs: dict) -> dict:
    """The layer's sharpness, and how much of dS (and scale * dS) is nonzero but below the
    fp16 normal range at this loss scale."""
    q, k, v, dout = (inputs[n].double() for n in ("q", "k", "v", "dout"))
    logits = (q @ k.transpose(-1, -2)) * inputs["scale"]
    p = logits.softmax(-1)
    ds = p * (dout @ v.transpose(-1, -2) - (dout * (p @ v)).sum(-1, keepdim=True))
    nonzero = ds != 0

    def below(x):
        return float(((x.abs() < FP16_MIN_NORMAL) & nonzero).sum() / nonzero.sum())

    return {
        "max_logit": float(logits.max()),
        "row_max_p": float(p.amax(-1).mean()),
        "ds_subnormal": below(ds),
        "scaled_ds_subnormal": below(ds * inputs["scale"]),
    }


def probe(model, images, fp16_scale, step, sink) -> None:
    """One probe: rows to ``sink``; per probe dtype and kernel / emulation, the worst dq and
    dk relative L2 error over layers (and the layer it is in) to stdout."""
    calls = capture(model, images)
    for name, dtype in DTYPES.items():
        loss_scale = fp16_scale if dtype == torch.float16 else 1.0
        worst = {}
        for layer, call in enumerate(calls):
            inputs = rounded(call, dtype, loss_scale)
            ref = kernel_grads(inputs, "fp64")
            info = {"step": step, "dtype": name, "loss_scale": loss_scale, "layer": layer}
            info.update(stats(inputs))
            results = {kernel: kernel_grads(inputs, kernel) for kernel in PROBED}
            results.update({e: emulated_grads(inputs, bugs) for e, bugs in EMULATIONS.items()})
            for kernel, grads in results.items():
                for tensor, got, want in zip(("dq", "dk", "dv"), grads, ref):
                    row = {**info, "kernel": kernel, "tensor": tensor, **errors(got, want)}
                    sink.write(json.dumps(row) + "\n")
                    key = (kernel, tensor)
                    if tensor != "dv" and row["rel_l2"] > worst.get(key, (-1.0, 0))[0]:
                        worst[key] = (row["rel_l2"], layer)
        sink.flush()
        cells = [
            f"{kernel} {worst[(kernel, 'dq')][0]:.1e}/{worst[(kernel, 'dk')][0]:.1e}"
            f"(L{worst[(kernel, 'dk')][1]})"
            for kernel in (*PROBED, *EMULATIONS)
        ]
        print(
            f"PROBE step {step} {name} scale 2^{math.log2(loss_scale):.0f}  worst dq/dk rel L2: "
            + "  ".join(cells),
            flush=True,
        )


# -- main ------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", required=True, help="image folder or uint8 [N, 3, H, W] .pt")
    parser.add_argument("--model", default="l", choices=tuple(REPOS))
    parser.add_argument("--backend", default="kohakufa", choices=KERNELS, help="trains with it")
    parser.add_argument("--dtype", default="fp16", choices=tuple(DTYPES), help="training dtype")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--probe-every", type=int, default=1000)
    parser.add_argument("--probe-batch", type=int, default=8)
    parser.add_argument(
        "--fp16-scale", default="live", help='fp16 probe loss scale: "live" or a number'
    )
    parser.add_argument("--out", default="benchmarks/results/dinov3")
    args = parser.parse_args()

    torch.manual_seed(0)
    out = Path(args.out) / f"{args.model}_{args.backend}_{args.dtype}"
    out.mkdir(parents=True, exist_ok=True)
    images = load_images(args.images)
    model = PatchDecoder(REPOS[args.model]).cuda()
    data = batches(images, args.batch, seed=0)
    fixed = next(batches(images, args.probe_batch, seed=1))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    dtype = DTYPES[args.dtype]
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    print(
        f"torch {torch.__version__}  cuDNN {torch.backends.cudnn.version()}  "
        f"{torch.cuda.get_device_name()}  {REPOS[args.model]}  {images.shape[0]} images  "
        f"training: {args.backend} {args.dtype}",
        flush=True,
    )

    with open(out / "probe.jsonl", "w") as probes, open(out / "train.jsonl", "w") as log:
        start = time.time()
        for step in range(args.steps + 1):
            if step % args.probe_every == 0:
                live = scaler.get_scale() if scaler.is_enabled() and step else 2.0**16
                fp16_scale = live if args.fp16_scale == "live" else float(args.fp16_scale)
                probe(model, fixed, fp16_scale, step, probes)
            if step == args.steps:
                break
            for group in opt.param_groups:
                group["lr"] = args.lr * min(1.0, (step + 1) / 500)
            with use_kernel(args.backend), torch.autocast("cuda", dtype):
                loss = model.loss(next(data))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
            scaler.step(opt)
            scaler.update()
            if (step + 1) % 50 == 0:
                row = {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "grad_norm": norm,
                    "loss_scale": scaler.get_scale(),
                }
                log.write(json.dumps(row) + "\n")
                if (step + 1) % 500 == 0:
                    ms = (time.time() - start) / (step + 1) * 1e3
                    print(
                        f"step {step + 1:6d}  loss {row['loss']:.5f}  grad norm {norm:.4f}  "
                        f"scale 2^{math.log2(row['loss_scale']):.0f}  {ms:.0f} ms/step",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
