import dataclasses

import pytest
import torch

from profiling.db.registry import find_kernel_profiler_spec
from profiling.runners.attention import paged_attn_batch_trtllm_gen as runner
from profiling.runners.attention.fa4_varlen_attention_reference import varlen_attention_reference
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented

_GEOMETRY = dict(
    num_qo_heads=6,
    num_kv_heads=2,
    head_dim=64,
    q_dtype="fp8_e4m3",
    kv_dtype="fp8_e4m3",
    o_dtype="bf16",
)


def _decode(**overrides):
    args = dict(_GEOMETRY, batch_size=4, max_kv_len=100, total_kv_len=250)
    args.update(overrides)
    return runner.validate_decode_args(**args)


def _prefill(**overrides):
    args = dict(_GEOMETRY, requests=((40, 4), (0, 3), (7, 3)))
    args.update(overrides)
    return runner.validate_prefill_args(**args)


def test_both_kinds_register_the_trtllm_gen_backend_on_vllm_env():
    # Defect caught: a kind missing from the barrel, or run on an env without
    # FlashInfer's trtllm-gen cubins.
    for kind in ("paged_attn_batch_decode", "paged_attn_batch_prefill"):
        spec = find_kernel_profiler_spec(kind, "flashinfer_trtllm_gen")
        assert spec.backend == "flashinfer_trtllm_gen"
        assert spec.subprocess_env == "vllm_env"
        assert {str(d) for d in spec.supports.compute} == {"fp8_e4m3", "bf16"}
        assert {str(d) for d in spec.supports.kv} == {"fp8_e4m3", "bf16"}


def test_canonical_decode_batch_holds_the_key():
    # Defect caught: a canonical batch whose longest or total context drifts
    # from the row's key.
    for batch, longest, total in [(1, 7, 7), (4, 100, 250), (5, 1000, 1004), (3, 9, 27)]:
        lengths = _decode(batch_size=batch, max_kv_len=longest, total_kv_len=total).lengths
        assert len(lengths) == batch
        assert lengths[0] == longest == max(lengths)
        assert sum(lengths) == total
        assert max(lengths[1:], default=0) - min(lengths[1:], default=0) <= 1


def test_prefill_batch_reads_each_request_after_its_prefix():
    # Defect caught: a context length or query-key count that ignores one
    # request's own prefix.
    shape = _prefill()
    assert shape.appends == [4, 3, 3]
    assert shape.lengths == [44, 3, 10]
    assert shape.interactions == (40 * 4 + 10) + 6 + (7 * 3 + 6)


@pytest.mark.parametrize(
    "overrides, error",
    [
        (dict(total_kv_len=102), ValueError),
        (dict(total_kv_len=401), ValueError),
        (dict(q_dtype="bf16"), ProfilerNotImplemented),
        (dict(kv_dtype="bf16"), ProfilerNotImplemented),
        (dict(q_dtype="bf16", kv_dtype="bf16", o_dtype="fp8_e4m3"), ProfilerNotImplemented),
        (dict(num_qo_heads=5), ValueError),
        (
            dict(batch_size=4, max_kv_len=1 << 30, total_kv_len=(1 << 30) + 3),
            ProfilerNotImplemented,
        ),
    ],
)
def test_decode_validation_rejects_unprofilable_args(overrides, error):
    with pytest.raises(error):
        _decode(**overrides)


@pytest.mark.parametrize(
    "requests, error",
    [((), ValueError), (((3, 0),), ValueError), (((-1, 4),), ValueError), (([0, 4],), TypeError)],
)
def test_prefill_validation_needs_new_tokens_per_request(requests, error):
    with pytest.raises(error):
        _prefill(requests=requests)


_BF16 = dict(q_dtype="bf16", kv_dtype="bf16", o_dtype="bf16")


def test_fp8_geometry_is_unchanged_and_bf16_doubles_the_element():
    # Defect caught: the bf16 extension changing the FP8 row's byte accounting
    # (its metrics are already in the database), or bf16 counted at FP8 width.
    fp8 = _decode().geometry
    assert (fp8.element_bytes, fp8.fp8, fp8.kv_bytes_per_token) == (1, True, 2 * 2 * 64)
    bf16 = _decode(**_BF16).geometry
    assert (bf16.element_bytes, bf16.fp8, bf16.kv_bytes_per_token) == (2, False, 2 * 2 * 64 * 2)
    # FP8 decode of 4 tokens over 250 cached tokens: query 1 B + output 2 B.
    metrics = runner._metrics(fp8, 1000, 4, runner._kv_bytes(fp8, [250]), 1.0, 0.0)
    assert metrics.memory_bandwidth_gbps == pytest.approx((4 * 6 * 64 * 3 + 250 * 256) / 1e-3 / 1e9)
    metrics = runner._metrics(bf16, 1000, 4, runner._kv_bytes(bf16, [250]), 1.0, 0.0)
    assert metrics.memory_bandwidth_gbps == pytest.approx((4 * 6 * 64 * 4 + 250 * 512) / 1e-3 / 1e9)


def test_bf16_and_fp8_rows_never_share_a_key():
    # Defect caught: a bf16 row overwriting (or being served as) the FP8 row of
    # the same shape. The dtypes are fields of the key on both kinds.
    from profiling.kernels.paged_attn_batch_decode import PagedAttnBatchDecodeArgs
    from profiling.kernels.paged_attn_batch_prefill import PagedAttnBatchPrefillArgs

    for args in (PagedAttnBatchDecodeArgs, PagedAttnBatchPrefillArgs):
        names = {f.name for f in dataclasses.fields(args)}
        assert {"q_dtype", "kv_dtype", "o_dtype"} <= names


def _workload(lengths, tokens, **dtypes):
    geometry = _decode(**dtypes).geometry
    return geometry, runner._build_workload(torch, geometry, lengths, tokens, torch.device("cpu"))


def _reference_out(geometry, work, lengths, positions_per_request, row_starts):
    """Fill ``work.out`` from a direct read of the HND cache through the page table."""
    page = runner._PAGE_SIZE
    for request, (positions, start) in enumerate(zip(positions_per_request, row_starts)):
        context = lengths[request]
        pages = work.block_table[request].long()[torch.arange(context) // page]
        offsets = torch.arange(context) % page
        dim = geometry.head_dim
        keys = work.raw_cache[pages, :, offsets, :dim].float()
        values = work.raw_cache[pages, :, offsets, dim:].float()
        rows = torch.arange(start, start + len(positions))
        out = varlen_attention_reference(
            work.q[rows].float(),
            torch.tensor(positions),
            keys,
            values,
            softmax_scale=dim**-0.5,
            window=0,
        )
        work.out[rows] = out.to(work.out.dtype)


@pytest.mark.parametrize("dtypes", [{}, _BF16], ids=["fp8", "bf16"])
def test_decode_check_reads_the_hnd_cache_through_the_page_table(dtypes):
    # Defect caught: the check reading K/V with the wrong layout (HND pages
    # viewed as NHD) or the wrong page, which would pass a broken kernel or
    # fail a correct one; for bf16, a workload still built in FP8.
    shape = _decode(batch_size=3, max_kv_len=70, total_kv_len=110, **dtypes)
    geometry, work = _workload(shape.lengths, shape.batch_size, **dtypes)
    expected = torch.float8_e4m3fn if geometry.fp8 else torch.bfloat16
    assert work.q.dtype == work.raw_cache.dtype == expected
    positions = [[length - 1] for length in shape.lengths]
    _reference_out(geometry, work, shape.lengths, positions, range(shape.batch_size))
    report = runner.check_decode(torch, shape, work)
    assert report["rows_checked"] == 3 and report["max_rel_l2"] < 0.01
    work.out[1] = work.out[1].flip(-1)
    with pytest.raises(KernelLaunchFailed):
        runner.check_decode(torch, shape, work)


@pytest.mark.parametrize("dtypes", [{}, _BF16], ids=["fp8", "bf16"])
def test_prefill_check_places_rows_after_the_prefix(dtypes):
    # Defect caught: query rows checked at positions that ignore the cached
    # prefix or the request's start in the packed query.
    shape = _prefill(requests=((33, 5), (2, 4)), **dtypes)
    geometry, work = _workload(shape.lengths, shape.query_tokens, **dtypes)
    positions = [list(range(p, p + a)) for p, a in shape.requests]
    _reference_out(geometry, work, shape.lengths, positions, [0, shape.appends[0]])
    report = runner.check_prefill(torch, shape, work)
    assert report["rows_checked"] == 9 and report["max_rel_l2"] < 0.01
    work.out[shape.appends[0] + 1] = 0
    with pytest.raises(KernelLaunchFailed):
        runner.check_prefill(torch, shape, work)


def test_page_size_defaults_to_the_legacy_32_and_accepts_the_served_16():
    # Defect caught: rows measured before page_size existed (32-token pages)
    # changing meaning, or a page size trtllm-gen ships no cubin for being
    # timed as another one.
    assert _decode().geometry.page_size == runner._PAGE_SIZE == 32
    assert _decode(page_size=16).geometry.page_size == 16
    assert _prefill(page_size=16).geometry.page_size == 16
    with pytest.raises(ProfilerNotImplemented, match="page_size"):
        _decode(page_size=48)
    with pytest.raises(ProfilerNotImplemented, match="page_size"):
        _prefill(page_size=8)


def test_page_size_is_a_key_field_whose_legacy_value_keeps_old_keys():
    # Defect caught: P16 rows overwriting (or served as) the P32 rows of the
    # same shape, or a spelled-out 32 moving every existing row's key.
    from profiling.db.batch import coerce_args
    from profiling.db.table import Table

    for kind, extra in (
        ("paged_attn_batch_decode", dict(batch_size=4, max_kv_len=100, total_kv_len=250)),
        ("paged_attn_batch_prefill", dict(requests=[[40, 4], [0, 3]])),
    ):
        spec = find_kernel_profiler_spec(kind, "flashinfer_trtllm_gen")
        table = Table(spec, "unused.db")
        base = dict(_GEOMETRY, **extra)
        unset = coerce_args(spec.args_schema, base)
        legacy = coerce_args(spec.args_schema, {**base, "page_size": 32})
        served = coerce_args(spec.args_schema, {**base, "page_size": 16})
        assert legacy.page_size is None
        key = table.args_hash(table.db_key(unset))
        assert table.args_hash(table.db_key(legacy)) == key
        assert table.args_hash(table.db_key(served)) != key


@pytest.mark.parametrize("page_size", [16, 32])
def test_decode_workload_pages_follow_the_page_size(page_size):
    # Defect caught: the cache or page table laid out for 32-token pages while
    # the call (and the cubin) runs 16; the check reads the cache through the
    # page table, so a mismatch fails it.
    shape = _decode(batch_size=3, max_kv_len=70, total_kv_len=110, page_size=page_size, **_BF16)
    geometry = shape.geometry
    work = runner._build_workload(torch, geometry, shape.lengths, 3, torch.device("cpu"))
    assert work.raw_cache.shape[2] == page_size
    assert work.page_counts == [-(-length // page_size) for length in shape.lengths]
    positions = [[length - 1] for length in shape.lengths]
    page = page_size
    for request, position in enumerate(positions):
        context = shape.lengths[request]
        pages = work.block_table[request].long()[torch.arange(context) // page]
        offsets = torch.arange(context) % page
        dim = geometry.head_dim
        out = varlen_attention_reference(
            work.q[[request]].float(),
            torch.tensor(position),
            work.raw_cache[pages, :, offsets, :dim].float(),
            work.raw_cache[pages, :, offsets, dim:].float(),
            softmax_scale=dim**-0.5,
            window=0,
        )
        work.out[[request]] = out.to(work.out.dtype)
    assert runner.check_decode(torch, shape, work)["rows_checked"] == 3


def test_decode_split_bound_is_fixed_by_the_key_and_the_gpu():
    # The split is not a free knob: max_seq_len saturates at 512 tokens per SM
    # (19 splits on GB300's 152 SMs for one request; serving's graph captured
    # at max_model_len 8192 gives 16), so (key, GPU) determines it.
    assert runner.decode_max_seq_len(568, 152) == 512 * 152
    assert runner.decode_max_seq_len(100_000, 152) == 100_000
    assert -(-runner.decode_max_seq_len(568, 152) // 512) == 152
