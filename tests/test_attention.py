"""Correctness against fp64 on the same rounded inputs: dense, token-level causal,
block-causal and cross attention, odd and tile-aligned lengths, eager / compiled /
CUDA graph, and the large-logit regime where single-LSE kernels lose precision."""

import pytest
import torch

from kohakufa import attention

from .reference import metrics, reference

TOL = 5e-3  # relative error (||got - fp64|| / ||fp64||) of out, dq, dk, dv in fp16


def inputs(batch, heads, s_q, s_kv, dim=64, seed=0, dtype=torch.float16, kv_heads=None):
    """q, k, v as [B, S, H, D] projections viewed as [B, H, S, D]; dout."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    kv_heads = kv_heads or heads

    def projection(seq, h):
        x = torch.randn(batch, seq, h, dim, device="cuda", generator=gen)
        return x.to(dtype).transpose(1, 2).requires_grad_()

    dout = torch.randn(batch, heads, s_q, dim, device="cuda", generator=gen).to(dtype)
    return (
        projection(s_q, heads),
        projection(s_kv, kv_heads),
        projection(s_kv, kv_heads),
        dout,
    )


def run(q, k, v, dout, **kwargs):
    for t in (q, k, v):
        t.grad = None
    out = attention(q, k, v, **kwargs)
    out.backward(dout)
    return out, q.grad, k.grad, v.grad


def check(got, want, tol=TOL):
    for name, g, w in zip(("out", "dq", "dk", "dv"), got, want):
        m = metrics(g, w)
        assert m["rel_err"] < tol, f"{name}: {m}"


CAUSAL = [  # (batch, heads, s_q, s_kv)
    (1, 1, 1, 1),
    (2, 3, 7, 7),
    (2, 3, 127, 127),
    (1, 4, 128, 128),
    (2, 2, 129, 129),
    (1, 2, 255, 255),
    (1, 2, 256, 256),
    (1, 2, 269, 269),
    (1, 2, 1000, 1000),
    (1, 2, 2048, 2048),
    (1, 2, 300, 517),  # more keys than queries: top-left aligned
    (1, 2, 517, 300),  # more queries: rows past Skv see every key
]


@pytest.mark.parametrize("shape", CAUSAL, ids=str)
def test_causal(shape):
    q, k, v, dout = inputs(*shape)
    check(run(q, k, v, dout, causal=True), reference(q, k, v, dout, 64**-0.5, block=1))


@pytest.mark.parametrize("block", [1, 7, 24, 128, 129, 257])
def test_block_causal(block):
    q, k, v, dout = inputs(2, 3, 600, 600)
    got = run(q, k, v, dout, block_causal=block)
    check(got, reference(q, k, v, dout, 64**-0.5, block=block))


DENSE = [(1, 1, 5, 7), (1, 3, 269, 269), (2, 2, 131, 517), (2, 12, 1799, 1799)]


@pytest.mark.parametrize("shape", DENSE, ids=str)
def test_dense(shape):
    q, k, v, dout = inputs(*shape)
    check(run(q, k, v, dout), reference(q, k, v, dout, 64**-0.5))


def test_causal_compiled():
    q, k, v, dout = inputs(2, 3, 300, 300)
    want = reference(q, k, v, dout, 64**-0.5, block=1)
    compiled = torch.compile(lambda a, b, c: attention(a, b, c, causal=True), fullgraph=True)
    out = compiled(q, k, v)
    out.backward(dout)
    check((out, q.grad, k.grad, v.grad), want)


def test_causal_cuda_graph():
    """Forward + backward captured in one CUDA graph (warmup and capture on one side
    stream, as a training step graph does), then replayed."""
    q, k, v, dout = inputs(2, 3, 300, 300)
    want = reference(q, k, v, dout, 64**-0.5, block=1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            run(q, k, v, dout, causal=True)
        for t in (q, k, v):
            t.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = attention(q, k, v, causal=True)
            out.backward(dout)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    check((out, q.grad, k.grad, v.grad), want)


def test_fp32_inputs_cast():
    q, k, v, dout = inputs(1, 2, 269, 269, dtype=torch.float32)
    out = attention(q, k, v, causal=True)
    assert out.dtype == torch.float32
    out.backward(dout)
    want = reference(*(t.half() for t in (q, k, v)), dout, 64**-0.5, block=1)
    for name, g, w in zip(("out", "dq", "dk", "dv"), (out, q.grad, k.grad, v.grad), want):
        assert metrics(g, w)["rel_err"] < 1e-2, name


def test_padded_keys_with_all_negative_scores():
    """Every score negative and Skv not a tile multiple: padded keys must not overflow
    in the backward (dQ += dS K would be inf * 0)."""
    gen = torch.Generator(device="cuda").manual_seed(1)
    shape = (2, 269, 4, 64)
    q = (4 * torch.randn(shape, device="cuda", generator=gen).abs()).half()
    k = (-4 * torch.randn(shape, device="cuda", generator=gen).abs()).half()
    v = torch.randn(shape, device="cuda", generator=gen).half()
    q, k, v = (t.transpose(1, 2).requires_grad_() for t in (q, k, v))
    dout = torch.randn(q.shape, device="cuda", generator=gen).half()
    got = run(q, k, v, dout)
    assert all(bool(torch.isfinite(g).all()) for g in got)
    check(got, reference(q, k, v, dout, 64**-0.5))


@pytest.mark.parametrize("causal", [False, True])
def test_huge_logits(causal):
    """Near-one-hot rows with |scores| ~ 1e7: the backward's P must match the
    forward's (row max and log2 l kept apart, shift before scale), so dV is exact and
    dK close. dQ is not checked: with clustered keys of norm ~1e3, dQ = sum_j dS_j k_j
    cancels to far below its terms, and the fp16 dS operand bounds it for any fp16
    kernel (FA2 does worse here)."""
    gen = torch.Generator(device="cuda").manual_seed(2)
    centers = torch.randn(1, 2, 8, 64, device="cuda", generator=gen)
    pick = torch.randint(0, 8, (1, 2, 269), device="cuda", generator=gen)
    base = torch.gather(centers, 2, pick[..., None].expand(-1, -1, -1, 64))
    q = 400 * (base + 1e-3 * torch.randn(base.shape, device="cuda", generator=gen))
    k = 400 * (base + 1e-3 * torch.randn(base.shape, device="cuda", generator=gen))
    v = torch.randn(base.shape, device="cuda", generator=gen)
    q, k, v = (t.half().requires_grad_() for t in (q, k, v))
    dout = torch.randn(base.shape, device="cuda", generator=gen).half()
    got = run(q, k, v, dout, causal=causal)
    want = reference(q, k, v, dout, 64**-0.5, block=1 if causal else 0)
    for name, g, w, bound in zip(("dk", "dv"), got[2:], want[2:], (0.995, 0.9999)):
        assert metrics(g, w)["cos"] > bound, name
