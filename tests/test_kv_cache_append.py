"""CPU-only contract tests for the ``kv_cache_append`` L1 kind."""

from __future__ import annotations

import subprocess
import sys

import pytest

from profiling.db.args import DType, KvCacheAppendArgs
from profiling.db.registry import find_kernel_profiler_spec, known_backends
from profiling.kernels.kv_cache_append import KIND


def test_args_field_order_matches_runner_kwargs():
    assert list(KvCacheAppendArgs.__dataclass_fields__) == [
        "num_kv_heads",
        "head_dim",
        "block_size",
        "input_dtype",
        "kv_dtype",
        "cache_layout",
        "scale_granularity",
        "num_tokens",
        "head_dim_v",
        "value_row_stride",
    ]


def test_kind_and_backends_are_registered():
    assert KIND == "kv_cache_append"
    assert set(known_backends(KIND)) == {"torch", "vllm_cuda", "vllm_triton_diffkv"}
    torch_spec = find_kernel_profiler_spec(KIND, "torch")
    vllm_spec = find_kernel_profiler_spec(KIND, "vllm_cuda")
    diffkv_spec = find_kernel_profiler_spec(KIND, "vllm_triton_diffkv")
    assert torch_spec.args_schema is KvCacheAppendArgs
    assert vllm_spec.args_schema is KvCacheAppendArgs
    assert diffkv_spec.args_schema is KvCacheAppendArgs
    assert torch_spec.subprocess_env is None
    assert vllm_spec.subprocess_env == "vllm_env"
    assert diffkv_spec.subprocess_env == "vllm_env"
    assert vllm_spec.supports.allows(DType.BF16, DType.FP8_E4M3)
    assert not diffkv_spec.supports.allows(DType.BF16, DType.FP8_E4M3)


def test_kernel_import_does_not_import_runner_or_vllm():
    command = [
        sys.executable,
        "-c",
        (
            "import sys; import profiling.kernels.kv_cache_append; "
            "print('profiling.runners.attention.kv_cache_append' in sys.modules, "
            "'vllm' in sys.modules)"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False False"


def _spec(**overrides):
    spec = {
        "num_kv_heads": 1,
        "head_dim": 192,
        "head_dim_v": 128,
        "block_size": 32,
        "input_dtype": "bf16",
        "kv_dtype": "bf16",
        "cache_layout": "NHD",
        "scale_granularity": "tensor",
        "num_tokens": 4,
    }
    return {**spec, **overrides}


def test_vllm_cuda_refuses_asymmetric_heads_before_touching_a_gpu():
    """reshape_and_cache_flash sizes V by K's head size; an asymmetric spec
    would otherwise time a write of the wrong shape."""
    from profiling.runners.attention.kv_cache_append import profile_kv_cache_append_vllm_cuda

    with pytest.raises(ValueError, match="vllm_triton_diffkv"):
        profile_kv_cache_append_vllm_cuda(**_spec())


def test_diffkv_refuses_per_head_scales():
    """The diffkv kernel loads one scalar k_scale / v_scale."""
    from profiling.runners.attention.kv_cache_append import (
        profile_kv_cache_append_vllm_triton_diffkv,
    )

    with pytest.raises(ValueError, match="tensor scale"):
        profile_kv_cache_append_vllm_triton_diffkv(**_spec(scale_granularity="head"))


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_packed_oracle_reads_k_then_v_from_each_slot(layout):
    """The oracle the diffkv runner checks before timing: a packed slot is
    K followed by V. A swapped half or a wrong page offset must fail it."""
    import torch

    from profiling.runners.attention.kv_cache_append import KernelLaunchFailed, check_packed_append

    blocks, page, heads, k_dim, v_dim = 4, 4, 2, 6, 3
    key = torch.randn(3, heads, k_dim)
    value = torch.randn(3, heads, v_dim)
    slots = torch.tensor([5, 0, 14])
    shape = (blocks, page, heads, k_dim + v_dim)
    if layout == "NHD":
        cache = torch.zeros(shape)
    else:
        cache = torch.zeros(blocks, heads, page, k_dim + v_dim).permute(0, 2, 1, 3)
    for token, slot in enumerate(slots.tolist()):
        cache[slot // page, slot % page] = torch.cat([key[token], value[token]], dim=-1)
    check_packed_append(
        torch, key=key, value=value, kv_cache=cache, slot_mapping=slots, block_size=page
    )

    cache[0, 0, :, :v_dim] = value[1]
    with pytest.raises(KernelLaunchFailed):
        check_packed_append(
            torch, key=key, value=value, kv_cache=cache, slot_mapping=slots, block_size=page
        )


# ---------------------------------------------------------------- serving layout

_SERVED = dict(
    num_kv_heads=8,
    head_dim=128,
    block_size=16,
    input_dtype="bf16",
    kv_dtype="bf16",
    cache_layout="HND_PACKED",
    scale_granularity="tensor",
    num_tokens=8,
    value_row_stride=(32 + 2 * 8) * 128,
)


def _reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, block_size):
    """What vllm::reshape_and_cache_flash_kernel does, written with indexing."""
    import torch

    blocks = torch.div(slot_mapping, block_size, rounding_mode="floor")
    offsets = slot_mapping % block_size
    key_cache[blocks, offsets] = key
    value_cache[blocks, offsets] = value


def _served_tensors(tokens=8, layout="HND_PACKED", stride=(32 + 2 * 8) * 128):
    import torch

    from profiling.runners.attention.kv_cache_append import serving_kv_tensors

    return serving_kv_tensors(
        torch,
        num_tokens=tokens,
        num_kv_heads=8,
        head_dim=128,
        block_size=16,
        dtype=torch.bfloat16,
        cache_layout=layout,
        value_row_stride=stride,
        device="cpu",
    )


def test_serving_tensors_have_the_traced_strides():
    # Strides of the GB300 TP1 serving trace at T = 8 (PHASE3B_PILOT_RESULTS.md
    # §3): each packed cache view (32768, 256, 4096, 1); key contiguous; value
    # a slice of the 6144-wide QKV row.
    raw, key_cache, value_cache, key, value = _served_tensors()
    assert key_cache.stride() == value_cache.stride() == (32768, 256, 4096, 1)
    assert value_cache.storage_offset() - key_cache.storage_offset() == 128
    assert key.is_contiguous() and key.stride() == (1024, 128, 1)
    assert value.stride() == (6144, 128, 1)
    assert raw.shape == (256, 8, 16, 256)
    # The existing HND layout: separate planes, (32768, 128, 2048, 1).
    _, key_cache, _, _, _ = _served_tensors(layout="HND", stride=0)
    assert key_cache.stride() == (32768, 128, 2048, 1)


def test_exact_oracle_accepts_a_correct_append_and_catches_stray_writes():
    import torch

    from profiling.runners.attention.kv_cache_append import check_append_exact
    from profiling.runners.exceptions import KernelLaunchFailed

    raw, key_cache, value_cache, key, value = _served_tensors()
    slots = torch.randperm(256 * 16)[:8]
    before = raw.clone()
    _reshape_and_cache_flash(key, value, key_cache, value_cache, slots, 16)
    kwargs = dict(
        raw_before=before,
        key_cache=key_cache,
        value_cache=value_cache,
        key=key,
        value=value,
        slot_mapping=slots,
        block_size=16,
    )
    check_append_exact(torch, raw_after=raw, **kwargs)
    # A write one slot off (a stride or offset bug) or anywhere else fails.
    raw[0, 0, 0, 0] += 1
    with pytest.raises(KernelLaunchFailed):
        check_append_exact(torch, raw_after=raw, **kwargs)
    # Values written into the K half (a packed-layout offset bug) fail too.
    raw.copy_(before)
    _reshape_and_cache_flash(key, value, key_cache, key_cache, slots, 16)
    with pytest.raises(KernelLaunchFailed):
        check_append_exact(torch, raw_after=raw, **kwargs)


def test_serving_layout_keys_never_collide_with_existing_layouts():
    from profiling.db.batch import coerce_args
    from profiling.db.table import Table

    spec = find_kernel_profiler_spec(KIND, "vllm_cuda")
    table = Table(spec, "unused.db")
    keys = {}
    for layout in ("NHD", "HND", "HND_PACKED"):
        for stride in (None, 0, 6144):
            args = coerce_args(
                spec.args_schema, {**_SERVED, "cache_layout": layout, "value_row_stride": stride}
            )
            keys.setdefault(table.args_hash(table.db_key(args)), set()).add((layout, stride or 0))
    # None and 0 are one key (contiguous values); every other pair its own.
    assert sorted(len(v) for v in keys.values()) == [1] * 6
    assert len(keys) == 6
    legacy = coerce_args(
        spec.args_schema, {**_SERVED, "cache_layout": "HND", "value_row_stride": 0}
    )
    assert legacy.value_row_stride is None


@pytest.mark.parametrize(
    ("backend", "overrides", "match"),
    [
        ("torch", {"value_row_stride": 6144}, "contiguous values only"),
        ("vllm_triton_diffkv", {"value_row_stride": 6144}, "contiguous values only"),
        ("torch", {"value_row_stride": 0}, "cache_layout"),
        ("vllm_triton_diffkv", {"value_row_stride": 0}, "cache_layout"),
        ("vllm_cuda", {"value_row_stride": 1024}, "must exceed"),
    ],
)
def test_unsupported_serving_configs_fail_instead_of_falling_back(backend, overrides, match):
    # Defect caught: a backend that cannot read strided values or the packed
    # cache timing its own contiguous write under the serving key.
    from profiling.runners.attention import kv_cache_append as runner

    profile = getattr(runner, f"profile_kv_cache_append_{backend}")
    with pytest.raises(ValueError, match=match):
        profile(**{**_SERVED, **overrides})


def test_packed_layout_with_an_fp8_cache_is_not_claimed():
    from profiling.runners.attention.kv_cache_append import profile_kv_cache_append_vllm_cuda
    from profiling.runners.exceptions import ProfilerNotImplemented

    with pytest.raises(ProfilerNotImplemented, match="input dtype only"):
        profile_kv_cache_append_vllm_cuda(**{**_SERVED, "kv_dtype": "fp8_e4m3"})
