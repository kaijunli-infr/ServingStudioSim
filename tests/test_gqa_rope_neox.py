import math

import pytest
import torch

from profiling.db.registry import find_kernel_profiler_spec
from profiling.kernels.gqa_rope_neox import GqaRopeNeoxArgs
from profiling.runners.attention import gqa_rope_neox_vllm_inductor as runner
from profiling.runners.attention.gqa_rope_neox_reference import (
    gqa_rope_neox_reference,
    llama3_cos_sin_cache,
)
from profiling.runners.exceptions import ProfilerNotImplemented

# Llama 3.1 8B per GPU at TP1, scaled down in heads where only the indexing matters.
HQ, HKV, D = 4, 2, 16


def _inputs(tokens=5, hq=HQ, hkv=HKV, d=D, rotary=D, max_position=64, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(0)
    positions = torch.randint(0, max_position, (tokens,), generator=g)
    qkv = torch.randn(tokens, (hq + 2 * hkv) * d, generator=g).to(dtype)
    cache = llama3_cos_sin_cache(torch, rotary, max_position)
    return positions, qkv, cache


def test_kind_registers_the_inductor_backend_on_vllm_env():
    # Defect caught: the kind missing from the barrel or run outside the pinned
    # vLLM environment whose Inductor generates the traced kernel.
    spec = find_kernel_profiler_spec("gqa_rope_neox", "vllm_inductor")
    assert spec.subprocess_env == "vllm_env"
    assert {str(d) for d in spec.supports.compute} == {"bf16"}
    names = list(GqaRopeNeoxArgs.__dataclass_fields__)
    assert names == [
        "num_tokens",
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "rotary_dim",
        "max_position",
        "dtype",
    ]


def _complex_rope(positions, qkv, cache, hq, hkv, d, rotary):
    """The same rotation written as a complex product, head by head."""
    half = rotary // 2
    tokens = qkv.shape[0]
    x = qkv.double()
    cos = cache.double()[positions][:, :half]
    sin = cache.double()[positions][:, half:rotary]
    rot = torch.complex(cos, sin)
    outs = []
    for first, count in ((0, hq), (hq, hkv)):
        heads = []
        for h in range(first, first + count):
            head = x[:, h * d : (h + 1) * d]
            z = torch.complex(head[:, :half], head[:, half:rotary]) * rot
            heads.append(torch.cat([z.real, z.imag, head[:, rotary:]], dim=1))
        outs.append(torch.cat(heads, dim=1).reshape(tokens, count * d))
    return outs


@pytest.mark.parametrize("rotary", [D, D // 2])
def test_reference_equals_the_complex_rotation(rotary):
    # Defect caught: GPT-J interleaved pairing in place of NeoX halves, the
    # sine sign flipped, the wrong cache half used as cosine, or a partial
    # rotary_dim rotating the pass-through tail.
    positions, qkv, _ = _inputs(dtype=torch.float32)
    cache = llama3_cos_sin_cache(torch, rotary, 64)
    q, k = gqa_rope_neox_reference(torch, positions, qkv, cache, HQ, HKV, D, rotary)
    want_q, want_k = _complex_rope(positions, qkv, cache, HQ, HKV, D, rotary)
    torch.testing.assert_close(q.double(), want_q, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(k.double(), want_k, rtol=1e-6, atol=1e-6)
    if rotary < D:
        torch.testing.assert_close(
            q.reshape(-1, HQ, D)[..., rotary:], qkv[:, : HQ * D].reshape(-1, HQ, D)[..., rotary:]
        )


def _inductor_port(positions, qkv_flat, cache_flat, hq, hkv, d):
    """Element-wise port of the traced `triton_poi_fused` kernel's indexing.

    Each output element reads its own input and its NeoX partner from the QKV
    row (stride (Hq + 2 Hkv) * D; key heads after Hq * D), and the cos/sin pair
    at row `positions[t]` of the cache (cos in columns [0, D/2), sin after).
    """
    stride, half = (hq + 2 * hkv) * d, d // 2
    tokens = positions.shape[0]

    def out(first_col, heads):
        res = torch.empty(tokens * heads * d, dtype=torch.float64)
        for idx in range(res.numel()):
            t, c = divmod(idx, heads * d)
            j = c % d
            src = t * stride + first_col + c
            partner = src + half if j < half else src - half
            row = int(positions[t]) * d
            cos, sin = cache_flat[row + j % half], cache_flat[row + half + j % half]
            x, y = float(qkv_flat[src]), float(qkv_flat[partner])
            res[idx] = x * cos - y * sin if j < half else x * cos + y * sin
        return res.reshape(tokens, heads * d)

    return out(0, hq), out(hq * d, hkv)


def test_reference_matches_the_traced_kernel_indexing():
    # Defect caught: the reference reading key heads at the value offset, a
    # query/key swap, or a cache row taken from the token index, not position.
    positions, qkv, cache = _inputs(dtype=torch.float32)
    q, k = gqa_rope_neox_reference(torch, positions, qkv, cache, HQ, HKV, D, D)
    port_q, port_k = _inductor_port(
        positions, qkv.reshape(-1).double(), cache.reshape(-1).double(), HQ, HKV, D
    )
    torch.testing.assert_close(q.double(), port_q, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(k.double(), port_k, rtol=1e-6, atol=1e-6)


def test_reference_keeps_the_input_dtype_and_contiguous_layout():
    positions, qkv, cache = _inputs()
    q, k = gqa_rope_neox_reference(torch, positions, qkv, cache, HQ, HKV, D, D)
    assert q.dtype == k.dtype == torch.bfloat16
    assert q.shape == (5, HQ * D) and k.shape == (5, HKV * D)
    assert q.is_contiguous() and k.is_contiguous()


def test_llama3_frequencies_follow_the_published_scaling():
    # Defect caught: a cache without Llama 3's wavelength scaling (plain
    # theta=500000), or the scaled and unscaled bands swapped.
    cache = llama3_cos_sin_cache(torch, 128, 2)
    angle = torch.atan2(cache[1, 64:], cache[1, :64]).double()  # position 1: the frequency
    inv = [500000.0 ** (-2 * i / 128) for i in range(64)]
    wavelen = [2 * math.pi / f for f in inv]
    assert angle[0].item() == pytest.approx(1.0)  # high frequency: unscaled
    last = 63
    assert wavelen[last] > 8192
    assert angle[last].item() == pytest.approx(inv[last] / 8, rel=1e-5)  # low: divided by 8
    mid = [i for i, w in enumerate(wavelen) if 2048 <= w <= 8192]
    assert mid, "Llama 3.1 128-dim has a smoothed band"
    for i in mid:
        smooth = (8192 / wavelen[i] - 1) / 3
        want = (1 - smooth) * inv[i] / 8 + smooth * inv[i]
        assert angle[i].item() == pytest.approx(want, rel=1e-5)


@pytest.mark.parametrize(
    "overrides, error",
    [
        (dict(num_tokens=0), ValueError),
        (dict(num_qo_heads=5), ValueError),
        (dict(rotary_dim=130), ValueError),
        (dict(rotary_dim=7), ValueError),
        (dict(dtype="fp8_e4m3"), ProfilerNotImplemented),
    ],
)
def test_runner_rejects_unprofilable_args(overrides, error):
    args = dict(
        num_tokens=8,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        rotary_dim=128,
        max_position=131072,
        dtype="bf16",
    )
    args.update(overrides)
    with pytest.raises(error):
        runner.validate_args(**args)


def test_bytes_moved_counts_query_and_key_once_each_way():
    # 8B decode token: 40 heads × 128 × 2 B read and written, a 256 B cache row
    # and an 8 B position.
    assert runner.bytes_moved(1, 32, 8, 128, 128) == 2 * 40 * 128 * 2 + 256 + 8
