"""fp64 reference attention and error metrics shared by the tests and benchmarks."""

import torch


def visible(s_q, s_kv, block, device):
    """Bool [Sq, Skv]: key c visible to query r (block-causal, top-left aligned)."""
    if block == 0:
        return None
    r = torch.arange(s_q, device=device)[:, None] // block
    c = torch.arange(s_kv, device=device)[None, :] // block
    return c <= r


def reference(q, k, v, dout, scale, block=0, groups=1):
    """out, dq, dk, dv in fp64 on the same (rounded) inputs: any difference from a
    kernel is the kernel's own error."""
    q, k, v = (t.detach().double().requires_grad_() for t in (q, k, v))
    kk, vv = k, v
    if groups > 1:  # GQA: each K / V head serves ``groups`` query heads
        kk, vv = (t.repeat_interleave(groups, dim=1) for t in (k, v))
    s = (q @ kk.transpose(-1, -2)) * scale
    mask = visible(q.shape[-2], k.shape[-2], block, q.device)
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    out = s.softmax(-1) @ vv
    out.backward(dout.double())
    return out.detach(), q.grad, k.grad, v.grad


def metrics(got, want):
    """Error of ``got`` against the fp64 ``want``."""
    g, w = got.detach().double().flatten(), want.detach().double().flatten()
    diff = g - w
    return {
        "cos": float(g @ w / (g.norm() * w.norm()).clamp_min(1e-300)),
        "norm_ratio": float(g.norm() / w.norm().clamp_min(1e-300)),
        "rel_err": float(diff.norm() / w.norm().clamp_min(1e-300)),
        "max_abs_err": float(diff.abs().max()),
        "max_rel_err": float(diff.abs().max() / w.abs().max().clamp_min(1e-300)),
    }
