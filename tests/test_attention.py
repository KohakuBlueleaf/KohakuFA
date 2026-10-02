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
        if float(w.abs().max()) == 0.0:  # an exactly-zero gradient (e.g. one visible key)
            assert m["max_abs_err"] < 1e-4, f"{name}: {m}"
        else:
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


BF16_TOL = 2e-2  # 8-bit mantissa: P, dS and the outputs round ~16x coarser than fp16


@pytest.mark.parametrize(
    "shape,mode",
    [
        ((2, 3, 269, 269), "dense"),
        ((1, 2, 1000, 1000), "causal"),
        ((2, 3, 600, 600), "block"),
        ((2, 2, 131, 517), "dense"),
        ((1, 2, 300, 517), "causal"),
    ],
    ids=str,
)
def test_bf16(shape, mode):
    q, k, v, dout = inputs(*shape, dtype=torch.bfloat16)
    kwargs = {"dense": {}, "causal": {"causal": True}, "block": {"block_causal": 24}}[mode]
    block = {"dense": 0, "causal": 1, "block": 24}[mode]
    got = run(q, k, v, dout, **kwargs)
    assert got[0].dtype == torch.bfloat16
    check(got, reference(q, k, v, dout, 64**-0.5, block=block), tol=BF16_TOL)


@pytest.mark.parametrize(
    "heads,kv_heads,shape,mode,dtype",
    [
        (8, 4, (2, 269), "dense", torch.float16),
        (8, 2, (1, 1000), "causal", torch.float16),
        (8, 1, (2, 600), "block", torch.float16),
        (16, 4, (2, 1799), "dense", torch.float16),
        (6, 2, (1, 300), "causal", torch.bfloat16),
        (8, 1, (2, 131), "dense", torch.bfloat16),
    ],
    ids=str,
)
def test_gqa(heads, kv_heads, shape, mode, dtype):
    """Grouped-query (and multi-query, kv_heads = 1) attention: dK / dV are summed over
    each K / V head's query heads."""
    batch, seq = shape
    q, k, v, dout = inputs(batch, heads, seq, seq, dtype=dtype, kv_heads=kv_heads)
    kwargs = {"dense": {}, "causal": {"causal": True}, "block": {"block_causal": 24}}[mode]
    block = {"dense": 0, "causal": 1, "block": 24}[mode]
    got = run(q, k, v, dout, **kwargs)
    assert got[2].shape == k.shape and got[3].shape == v.shape
    want = reference(q, k, v, dout, 64**-0.5, block=block, groups=heads // kv_heads)
    check(got, want, tol=TOL if dtype == torch.float16 else BF16_TOL)


def random_mask(shape, density, seed=3):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand(shape, device="cuda", generator=gen) < density


@pytest.mark.parametrize(
    "shape,mask_shape,density",
    [
        ((2, 3, 269, 269), "full", 0.5),
        ((2, 3, 269, 269), "broadcast", 0.3),
        ((2, 2, 131, 517), "full", 0.7),
        ((1, 2, 1000, 1000), "head", 0.05),
    ],
    ids=str,
)
def test_mask(shape, mask_shape, density):
    batch, heads, s_q, s_kv = shape
    m_shape = {
        "full": (batch, heads, s_q, s_kv),
        "broadcast": (1, 1, s_q, s_kv),
        "head": (1, heads, s_q, s_kv),
    }[mask_shape]
    mask = random_mask(m_shape, density)
    q, k, v, dout = inputs(*shape)
    check(run(q, k, v, dout, mask=mask), reference(q, k, v, dout, 64**-0.5, mask=mask))


def test_mask_key_padding_and_empty_rows():
    """[B, 1, 1, Skv] key padding, plus rows that see no key at all (output 0, finite
    gradients)."""
    q, k, v, dout = inputs(2, 4, 300, 300)
    lengths = torch.tensor([300, 157], device="cuda")
    mask = (torch.arange(300, device="cuda")[None, :] < lengths[:, None])[:, None, None, :]
    mask = mask.expand(2, 4, 300, 300).clone()
    mask[:, :, 10:20] = False  # empty rows
    got = run(q, k, v, dout, mask=mask)
    assert all(bool(torch.isfinite(g).all()) for g in got)
    assert bool((got[0][:, :, 10:20] == 0).all())
    check(got, reference(q, k, v, dout, 64**-0.5, mask=mask))


def test_mask_equals_causal():
    q, k, v, dout = inputs(1, 2, 600, 600)
    causal = torch.ones(600, 600, dtype=torch.bool, device="cuda").tril()
    masked = run(q, k, v, dout, mask=causal)
    native = run(q, k, v, dout, causal=True)
    for name, a, b in zip(("out", "dq", "dk", "dv"), masked, native):
        assert metrics(a, b)["rel_err"] < 1e-3, name


def test_mask_gqa_bf16():
    q, k, v, dout = inputs(2, 8, 269, 269, dtype=torch.bfloat16, kv_heads=2)
    mask = random_mask((2, 8, 269, 269), 0.5)
    got = run(q, k, v, dout, mask=mask)
    check(got, reference(q, k, v, dout, 64**-0.5, groups=4, mask=mask), tol=BF16_TOL)


def packed(lengths, heads, kv_heads, dtype=torch.float16, seed=4):
    """Packed q [T, H, D], k / v [T, Hkv, D], dout, and cu_seqlens."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(lengths)
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32, device="cuda")

    def make(h):
        return torch.randn(total, h, 64, device="cuda", generator=gen).to(dtype).requires_grad_()

    dout = torch.randn(total, heads, 64, device="cuda", generator=gen).to(dtype)
    return make(heads), make(kv_heads), make(kv_heads), dout, cu


def check_varlen(lengths, heads, kv_heads, block, dtype=torch.float16, tol=TOL):
    from kohakufa import attention_varlen

    q, k, v, dout, cu = packed(lengths, heads, kv_heads, dtype)
    out = attention_varlen(q, k, v, cu, cu, block_causal=block)
    out.backward(dout)
    for b in range(len(lengths)):
        s, e = int(cu[b]), int(cu[b + 1])

        def view(t, s=s, e=e):
            return t[s:e].unsqueeze(0).transpose(1, 2)

        want = reference(view(q), view(k), view(v), view(dout), 64**-0.5, block=block,
                         groups=heads // kv_heads)  # fmt: skip
        got = (view(out), view(q.grad), view(k.grad), view(v.grad))
        check(got, want, tol=tol)


@pytest.mark.parametrize(
    "lengths,heads,kv_heads,block",
    [
        ([5, 300, 129, 1000, 256], 4, 4, 0),
        ([1, 77, 512, 3, 1300], 8, 2, 1),
        ([600, 257, 24], 3, 3, 24),
        ([128, 256, 384], 2, 1, 0),
    ],
    ids=str,
)
def test_varlen(lengths, heads, kv_heads, block):
    check_varlen(lengths, heads, kv_heads, block)


def test_varlen_bf16():
    check_varlen([300, 129, 700], 4, 2, 1, dtype=torch.bfloat16, tol=BF16_TOL)


def test_varlen_cuda_graph():
    from kohakufa import attention_varlen, varlen_plan

    q, k, v, dout, cu = packed([300, 129, 700], 4, 4)
    plan = varlen_plan(cu, cu, 4, 4, 1)  # built outside capture: no host sync inside
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            attention_varlen(q, k, v, cu, cu, causal=True, plan=plan).backward(dout)
        for t in (q, k, v):
            t.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = attention_varlen(q, k, v, cu, cu, causal=True, plan=plan)
            out.backward(dout)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    eager = attention_varlen(*(t.detach() for t in (q, k, v)), cu, cu, causal=True)
    assert metrics(out, eager)["rel_err"] < 1e-6


def test_packed_mask_reuse():
    """pack_mask once, reuse across calls: identical to passing the bool mask."""
    from kohakufa import pack_mask

    q, k, v, dout = inputs(2, 3, 600, 600)
    mask = random_mask((2, 1, 600, 600), 0.4)
    packed = pack_mask(mask, 2, 3, 600, 600)
    a = run(q, k, v, dout, mask=packed)
    b = run(q, k, v, dout, mask=mask)
    for name, x, y in zip(("out", "dq", "dk", "dv"), a, b):
        if name == "dq":  # dQ is reduced in arbitrary order (TMA reduce-add)
            assert metrics(x, y)["rel_err"] < 1e-3, name
        else:
            assert metrics(x, y)["max_abs_err"] == 0.0, name
