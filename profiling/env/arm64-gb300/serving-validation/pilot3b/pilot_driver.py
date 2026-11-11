#!/usr/bin/env python
"""Phase 3B-0 GPU correctness pilot (P1-P4) on one GB300 GPU. No database writes.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=<profiling worktree> <serving venv>/bin/python \\
        pilot_driver.py --mode timing   --out DIR --expect-uuid GPU-...
    nsys profile ... <same> pilot_driver.py --mode identity --out DIR --expect-uuid GPU-...

Every case is built with the reviewed runner code (its validators, block
builders, workload builders and checks), not a re-implementation, except the
diagnostic variants named `*_served*` / `embed_norm_exact`, which reproduce a
serving detail the runner does not (documented per case).

timing mode, per case (P4):
  cold_eager   Timer.cupti(fn) exactly as the runner calls it (L2 flushed)
  warm_eager   Timer.cupti(fn, clear_l2=False)
  graph_warm   fn captured once in a CUDA graph; Timer.cupti(graph.replay, clear_l2=False)
  plus per-kernel CUPTI records: 30 cold single launches, and bursts of 50 warm
  eager launches and 50 graph replays, for per-kernel durations, the
  operation's summed / union / span time per launch, and their variability.
  The case's oracle runs before timing and again after graph replay.
identity mode: per case 3 eager launches and 3 graph replays inside NVTX ranges
  `<case>:eager` / `<case>:graph` for Nsight Systems (grid, block, registers).

Stops at the first failed oracle (exit 3); any other error exits 4; a device
other than the expected UUID exits 9. Results are rewritten after every case.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics as st
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

HIDDEN, INTER, HQ, HKV, HEAD, VOCAB = 4096, 14336, 32, 8, 128, 128256
BF16 = torch.bfloat16
COLD_SINGLE = 30
BURST = 50
BURST_WARMUP = 5


class OracleFailure(Exception):
    pass


@dataclass
class Case:
    name: str
    part: str  # P1 / P2 / P3 / P4-gemm
    fn: Callable[[], Any]
    check: Callable[[Any], dict]  # takes fn's return (or the graph's static return)
    timer_kw: dict  # kernel_name / interval_union / warmup, as the runner calls Timer.cupti
    meta: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)  # trace class and roles to compare against


def _strides(**tensors: Any) -> dict:
    return {k: {"shape": list(v.shape), "stride": list(v.stride()), "dtype": str(v.dtype)}
            for k, v in tensors.items()}


# ---------------------------------------------------------------- P1 builders
def norm_case(dev, tokens, addends, store_sum, store_normed, role, label) -> Case:
    from profiling.runners.norm import multi_add_rms_norm_vllm_inductor as R
    from profiling.runners.norm.multi_add_rms_norm_reference import multi_add_rms_norm_reference
    from profiling.runners.vllm_inductor import compile_like_vllm

    R._validate_args(tokens, HIDDEN, addends, store_sum, store_normed, False, "bf16")
    block = R._build_block(store_sum, store_normed, False, addends)
    gen = torch.Generator(device=dev).manual_seed(0)
    weight = 1.0 + 0.1 * torch.randn(HIDDEN, dtype=BF16, device=dev, generator=gen)
    scale = torch.tensor(R._FP8_SCALE, dtype=torch.float32, device=dev)

    def make_inputs(t):
        return [*(torch.randn((t, HIDDEN), dtype=BF16, device=dev, generator=gen)
                  for _ in range(addends)), weight, scale]

    compiled = compile_like_vllm(torch, block, make_inputs, {i: 0 for i in range(addends)}, tokens)
    inputs = make_inputs(tokens)

    def check(outputs):
        expected = multi_add_rms_norm_reference(torch, inputs[:addends], weight, scale, R._EPS)
        it, rep = iter(outputs), {}
        for flag, ref, tol, key in zip((store_sum, store_normed, False), expected,
                                       ((1e-2, 1e-2), (2e-2, 2e-2), (0.13, 2e-2)),
                                       ("sum", "normed", "fp8")):
            if flag:
                got = next(it).float()
                rep[f"{key}_max_abs"] = float((got - ref.float()).abs().max())
                try:
                    torch.testing.assert_close(got, ref.float(), rtol=tol[0], atol=tol[1])
                except AssertionError as exc:
                    raise OracleFailure(f"{label}: {key} {str(exc).splitlines()[0]}") from exc
        return rep

    return Case(f"{label}_T{tokens}", "P1", lambda: compiled(*inputs), check,
                {"kernel_name": None}, {"tokens": tokens, "addends": addends,
                                        "store_sum": store_sum, "store_normed": store_normed,
                                        **_strides(addend0=inputs[0])},
                {"role": role})


def embed_norm_case(dev, tokens) -> Case:
    """Diagnostic: vLLM's first piece, embedding gather of int32 ids + RMSNorm, compiled like vLLM."""
    from vllm import ir
    from profiling.runners.vllm_inductor import compile_like_vllm

    rms_norm = ir.ops.rms_norm.impls["native"].impl_fn
    eps = 1e-5
    gen = torch.Generator(device=dev).manual_seed(1)
    table = 0.02 * torch.randn((VOCAB, HIDDEN), dtype=BF16, device=dev, generator=gen)
    weight = 1.0 + 0.1 * torch.randn(HIDDEN, dtype=BF16, device=dev, generator=gen)

    def block(ids, table, weight):
        # VocabParallelEmbedding at TP1: F.embedding(input.long(), weight); then the
        # first decoder layer's input_layernorm(hidden) with residual = hidden.
        hidden = torch.nn.functional.embedding(ids.long(), table)
        return hidden, rms_norm(hidden, weight, eps)

    def make_inputs(t):
        ids = torch.randint(0, VOCAB, (t,), dtype=torch.int32, device=dev, generator=gen)
        return [ids, table, weight]

    compiled = compile_like_vllm(torch, block, make_inputs, {0: 0}, tokens)
    inputs = make_inputs(tokens)

    def check(outputs):
        hidden, normed = outputs
        rows = table[inputs[0].long()]
        if not torch.equal(hidden, rows):
            raise OracleFailure("embed_norm_exact: gathered rows differ")
        x = rows.float()
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
        try:
            torch.testing.assert_close(normed.float(), ref, rtol=2e-2, atol=2e-2)
        except AssertionError as exc:
            raise OracleFailure(f"embed_norm_exact: {str(exc).splitlines()[0]}") from exc
        return {"normed_max_abs": float((normed.float() - ref).abs().max())}

    return Case(f"embed_norm_exact_T{tokens}", "P1", lambda: compiled(*inputs), check,
                {"kernel_name": None}, {"tokens": tokens, **_strides(ids=inputs[0], table=table)},
                {"role": "embed_input_norm"})


def silu_case(dev, tokens) -> Case:
    from vllm.model_executor.layers.activation import SiluAndMul
    from profiling.runners.elementwise.silu_and_mul_reference import silu_and_mul_reference
    from profiling.runners.vllm_inductor import compile_like_vllm

    gen = torch.Generator(device=dev).manual_seed(0)

    def make_inputs(t):
        return [torch.randn((t, 2 * INTER), dtype=BF16, device=dev, generator=gen)]

    compiled = compile_like_vllm(torch, SiluAndMul.forward_native, make_inputs, {0: 0}, tokens)
    inputs = make_inputs(tokens)

    def check(out):
        ref = silu_and_mul_reference(torch, inputs[0])
        try:
            torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
        except AssertionError as exc:
            raise OracleFailure(f"silu: {str(exc).splitlines()[0]}") from exc
        return {"max_abs": float((out.float() - ref.float()).abs().max())}

    return Case(f"silu_T{tokens}", "P1", lambda: compiled(*inputs), check, {"kernel_name": None},
                {"tokens": tokens, **_strides(x=inputs[0])}, {"role": "activation"})


def rope_case(dev, tokens) -> Case:
    """The gqa_rope_neox runner's block and oracle (profile_gqa_rope_neox_vllm_inductor)."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from profiling.runners.attention import gqa_rope_neox_vllm_inductor as R
    from profiling.runners.attention.gqa_rope_neox_reference import gqa_rope_neox_reference
    from profiling.runners.vllm_inductor import compile_like_vllm

    max_position = 131072
    q_size, kv_size = HQ * HEAD, HKV * HEAD
    R.validate_args(tokens, HQ, HKV, HEAD, HEAD, max_position, "bf16")
    with set_current_vllm_config(VllmConfig()):
        rope = get_rope(HEAD, max_position=max_position, is_neox_style=True,
                        rope_parameters={**R.LLAMA3_ROPE_PARAMETERS, "partial_rotary_factor": 1.0},
                        dtype=BF16).to(dev)
    gen = torch.Generator(device=dev).manual_seed(0)

    def block(positions, qkv):
        q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
        return rope.forward_native(positions, q, k)

    def make_inputs(t):
        positions = torch.randint(0, max_position, (t,), device=dev, generator=gen, dtype=torch.int64)
        qkv = torch.randn((t, q_size + 2 * kv_size), dtype=BF16, device=dev, generator=gen)
        return [positions, qkv]

    compiled = compile_like_vllm(torch, block, make_inputs, {0: 0, 1: 0}, tokens)
    inputs = make_inputs(tokens)

    def check(out):
        q, k = out
        want_q, want_k = gqa_rope_neox_reference(torch, inputs[0], inputs[1], rope.cos_sin_cache,
                                                 HQ, HKV, HEAD, HEAD)
        rep = {}
        for key, got, want in (("q", q, want_q), ("k", k, want_k)):
            got = got.reshape(want.shape)
            rep[f"{key}_max_abs"] = float((got.float() - want.float()).abs().max())
            try:
                torch.testing.assert_close(got, want, rtol=R._RTOL, atol=R._ATOL)
            except AssertionError as exc:
                raise OracleFailure(f"rope {key}: {str(exc).splitlines()[0]}") from exc
        return rep

    return Case(f"rope_T{tokens}", "P1", lambda: compiled(*inputs), check, {"kernel_name": None},
                {"tokens": tokens, **_strides(positions=inputs[0], qkv=inputs[1],
                                               cos_sin_cache=rope.cos_sin_cache)},
                {"role": "rope"})


def _kv_check(label, key, value, k_cache, v_cache, slots):
    blocks, offsets = slots // 16, slots % 16
    ok_k = torch.equal(k_cache[blocks, offsets], key)
    ok_v = torch.equal(v_cache[blocks, offsets], value)
    if not (ok_k and ok_v):
        raise OracleFailure(f"{label}: appended rows differ (K {ok_k}, V {ok_v})")
    return {"bitwise_equal_rows": int(slots.numel())}


def kv_runner_case(dev, tokens) -> Case:
    """profile_kv_cache_append_vllm_cuda's workload and call, HND; the pilot adds the oracle."""
    from vllm import _custom_ops as ops
    from vllm.utils.torch_utils import create_kv_caches_with_random_flash
    from profiling.db.args import DType
    from profiling.runners.attention import kv_cache_append as R

    key, value = R._new_rows(torch, num_tokens=tokens, num_kv_heads=HKV, head_dim=HEAD,
                             head_dim_v=HEAD, input_dtype=DType.BF16)
    slots = R._slot_mapping(torch, num_tokens=tokens, block_size=16)
    kcs, vcs = create_kv_caches_with_random_flash(R._num_blocks(tokens, 16), 16, 1, HKV, HEAD,
                                                  "auto", BF16, seed=42, device="cuda",
                                                  cache_layout="HND")
    k_cache, v_cache = kcs[0], vcs[0]
    k_scale, v_scale = R._scales(torch, num_kv_heads=HKV, scale_granularity="tensor")

    def fn():
        ops.reshape_and_cache_flash(key, value, k_cache, v_cache, slots, "auto", k_scale, v_scale)

    return Case(f"kv_append_runner_T{tokens}", "P1", fn,
                lambda _: _kv_check("kv_append_runner", key, value, k_cache, v_cache, slots),
                {"kernel_name": R._VLLM_KERNEL_NAME, "warmup": 5},
                {"tokens": tokens, **_strides(key=key, value=value, key_cache=k_cache,
                                               value_cache=v_cache)},
                {"role": "kv_cache_append"})


def kv_served_case(dev, tokens) -> Case:
    """Diagnostic: the FlashInfer backend's packed HND cache (B, H, page, 2D) split into K/V
    views, K the contiguous RoPE output, V a strided slice of the QKV GEMM output."""
    from vllm import _custom_ops as ops
    from profiling.runners.attention import kv_cache_append as R

    gen = torch.Generator(device=dev).manual_seed(2)
    num_blocks = R._num_blocks(tokens, 16)
    raw = torch.randn((num_blocks, HKV, 16, 2 * HEAD), dtype=BF16, device=dev, generator=gen)
    k_cache, v_cache = raw.transpose(1, 2).split(HEAD, dim=-1)
    qkv = torch.randn((tokens, (HQ + 2 * HKV) * HEAD), dtype=BF16, device=dev, generator=gen)
    key = torch.randn((tokens, HKV * HEAD), dtype=BF16, device=dev, generator=gen).view(tokens, HKV, HEAD)
    value = qkv[:, (HQ + HKV) * HEAD:].view(tokens, HKV, HEAD)
    slots = R._slot_mapping(torch, num_tokens=tokens, block_size=16)
    scale = torch.ones((1,), dtype=torch.float32, device=dev)

    def fn():
        ops.reshape_and_cache_flash(key, value, k_cache, v_cache, slots, "auto", scale, scale)

    return Case(f"kv_append_served_T{tokens}", "P1", fn,
                lambda _: _kv_check("kv_append_served", key, value, k_cache, v_cache, slots),
                {"kernel_name": R._VLLM_KERNEL_NAME, "warmup": 5},
                {"tokens": tokens, **_strides(key=key, value=value, key_cache=k_cache,
                                               value_cache=v_cache)},
                {"role": "kv_cache_append"})


# ---------------------------------------------------------------- P2 / P3
GEOM = dict(num_qo_heads=HQ, num_kv_heads=HKV, head_dim=HEAD, q_dtype="bf16", kv_dtype="bf16",
            o_dtype="bf16")


def decode_case(dev, label, batch, max_kv, total, trace_class, max_seq_len=None) -> Case:
    """The runner's decode workload and call; `max_seq_len` set only for the serving diagnostic."""
    from profiling.runners.attention import paged_attn_batch_trtllm_gen as R

    shape = R.validate_decode_args(**GEOM, batch_size=batch, max_kv_len=max_kv, total_kv_len=total)
    work, run = R.build_decode(torch, shape, dev)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    runner_msl = max(shape.max_kv_len, R._SPLIT_TOKENS * sms)
    if max_seq_len is not None:
        decode, _ = R._import_flashinfer(R._DECODE)
        ws = R._workspace(torch, dev)

        def run():  # noqa: F811 - identical to build_decode's closure but for max_seq_len
            return decode.trtllm_batch_decode_with_kv_cache(
                query=work.q, kv_cache=work.kv_cache, workspace_buffer=ws,
                block_tables=work.block_table, seq_lens=work.seq_lens, max_seq_len=max_seq_len,
                bmm1_scale=HEAD**-0.5, bmm2_scale=1.0, window_left=-1, sinks=None,
                o_sf_scale=None, out=work.out, kv_layout="HND", backend="trtllm-gen",
                q_len_per_req=1, max_q_len=None, cum_seq_lens_q=None, kv_cache_sf=None,
                lse=None, return_lse=False)

    def check(_):
        try:
            return R.check_decode(torch, shape, work)
        except Exception as exc:
            raise OracleFailure(f"{label}: {exc}") from exc

    return Case(label, "P2", run, check, {"kernel_name": None, "interval_union": True},
                {"batch": batch, "lengths": shape.lengths,
                 "max_seq_len": max_seq_len if max_seq_len is not None else runner_msl,
                 "max_seq_len_source": "serving" if max_seq_len is not None else "runner",
                 **_strides(q=work.q, raw_cache=work.raw_cache, k_cache=work.kv_cache[0],
                            out=work.out)},
                {"class": trace_class, "role": "attention"})


def prefill_case(dev) -> Case:
    from profiling.runners.attention import paged_attn_batch_trtllm_gen as R

    shape = R.validate_prefill_args(**GEOM, requests=((0, 256),))
    work, run = R.build_prefill(torch, shape, dev)

    def check(_):
        try:
            return R.check_prefill(torch, shape, work)
        except Exception as exc:
            raise OracleFailure(f"prefill256: {exc}") from exc

    return Case("prefill256", "P3", run, check, {"kernel_name": None, "interval_union": True},
                {"requests": [[0, 256]], **_strides(q=work.q, raw_cache=work.raw_cache)},
                {"class": "prefill256", "role": "attention"})


# ---------------------------------------------------------------- P4 GEMMs
def gemm_case(dev, role, n, k) -> Case:
    """profile_single_gemm_linear's call (F.linear, weight [n, k]) at m = 8."""
    gen = torch.Generator(device=dev).manual_seed(3)
    x = torch.randn((8, k), dtype=BF16, device=dev, generator=gen)
    w = torch.randn((n, k), dtype=BF16, device=dev, generator=gen)

    def check(out):
        ref = x.float() @ w.float().T
        err = float((out.float() - ref).abs().max() / ref.abs().max())
        if err > 1e-2:
            raise OracleFailure(f"gemm {role}: relative error {err}")
        return {"max_rel_to_max": err}

    return Case(f"gemm_{role}_m8", "P4-gemm", lambda: torch.nn.functional.linear(x, w), check,
                {"kernel_name": None, "interval_union": True},
                {"m": 8, "n": n, "k": k, **_strides(x=x, weight=w)},
                {"class": "decode8", "role": role})


def build_cases(dev, only: set[str] | None) -> list[tuple[str, Callable[[], Case]]]:
    builders: list[tuple[str, Callable[[], Case]]] = []
    for t in (8, 256):
        builders += [
            (f"norm_first_1add_T{t}", lambda t=t: norm_case(dev, t, 1, False, True, "embed_input_norm", "norm_first_1add")),
            (f"embed_norm_exact_T{t}", lambda t=t: embed_norm_case(dev, t)),
            (f"norm_post_T{t}", lambda t=t: norm_case(dev, t, 2, False, True, "post_norm", "norm_post")),
            (f"norm_next_T{t}", lambda t=t: norm_case(dev, t, 3, True, True, "input_norm", "norm_next")),
            (f"norm_final_T{t}", lambda t=t: norm_case(dev, t, 3, False, True, "final_norm", "norm_final")),
            (f"silu_T{t}", lambda t=t: silu_case(dev, t)),
            (f"rope_T{t}", lambda t=t: rope_case(dev, t)),
            (f"kv_append_runner_T{t}", lambda t=t: kv_runner_case(dev, t)),
            (f"kv_append_served_T{t}", lambda t=t: kv_served_case(dev, t)),
        ]
    builders += [
        ("decode8", lambda: decode_case(dev, "decode8", 8, 600, 4789, "decode8")),
        ("decode8_served_graph_msl", lambda: decode_case(dev, "decode8_served_graph_msl", 8, 600, 4789, "decode8", 8192)),
        ("decode4", lambda: decode_case(dev, "decode4", 4, 630, 2520, "decode4")),
        ("decode1", lambda: decode_case(dev, "decode1", 1, 568, 568, "mixed_decode1")),
        ("decode1_served_graph_msl", lambda: decode_case(dev, "decode1_served_graph_msl", 1, 568, 568, "mixed_decode1", 8192)),
        ("decode1_served_eager_msl", lambda: decode_case(dev, "decode1_served_eager_msl", 1, 568, 568, "mixed_decode1", 568)),
        ("prefill256", lambda: prefill_case(dev)),
    ]
    for role, n, k in (("qkv_proj", 6144, 4096), ("o_proj", 4096, 4096), ("up_gate_proj", 28672, 4096),
                       ("down_proj", 4096, 14336), ("lm_head", 128256, 4096)):
        builders.append((f"gemm_{role}_m8", lambda role=role, n=n, k=k: gemm_case(dev, role, n, k)))
    return [(name, b) for name, b in builders if only is None or name in only]


# ---------------------------------------------------------------- measurement
def summary(values: list[float]) -> dict | None:
    if not values:
        return None
    mean = st.fmean(values)
    sd = st.pstdev(values) if len(values) > 1 else 0.0
    s = sorted(values)
    return {"n": len(values), "mean": mean, "median": st.median(values), "min": s[0],
            "p90": s[min(len(s) - 1, int(0.9 * (len(s) - 1) + 0.5))], "max": s[-1],
            "cv": sd / mean if mean else None}


def _union(intervals):
    total, cur = 0, None
    for s, e in sorted(intervals):
        if cur is None or s > cur[1]:
            if cur:
                total += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    return total + (cur[1] - cur[0] if cur else 0)


def launch_stats(launches: list[list]) -> dict:
    """Per-kernel and per-launch statistics (microseconds) from grouped CUPTI records."""
    per_name: dict[str, list[float]] = {}
    sums, unions, spans, seqs = [], [], [], {}
    for recs in launches:
        recs = sorted(recs, key=lambda r: r.start_ns)
        seq = tuple(r.name for r in recs)
        seqs[seq] = seqs.get(seq, 0) + 1
        for r in recs:
            per_name.setdefault(r.name, []).append(r.duration_ns / 1e3)
        sums.append(sum(r.duration_ns for r in recs) / 1e3)
        unions.append(_union([(r.start_ns, r.end_ns) for r in recs]) / 1e3)
        spans.append((recs[-1].end_ns - recs[0].start_ns) / 1e3 if recs else 0.0)
    return {"sequences": [{"kernels": list(k), "count": v} for k, v in seqs.items()],
            "per_kernel_us": {k: summary(v) for k, v in per_name.items()},
            "launch_sum_us": summary(sums), "launch_union_us": summary(unions),
            "launch_span_us": summary(spans)}


def _filter(records, kernel_name):
    return [r for r in records if kernel_name is None or kernel_name in r.name]


def burst(prof, fn, n, kernel_name, cold) -> dict:
    if cold:
        launches = [_filter(prof.capture_once(fn, launches_per_run=1, clear_l2_before_run=True,
                                              clear_l2_between_launches=False), kernel_name)
                    for _ in range(n)]
    else:
        for _ in range(BURST_WARMUP):
            fn()
        torch.cuda.synchronize()
        recs = sorted(_filter(prof.capture_once(fn, launches_per_run=n, clear_l2_before_run=False,
                                                clear_l2_between_launches=False), kernel_name),
                      key=lambda r: r.start_ns)
        if not recs or len(recs) % n:
            return {"irregular_record_count": len(recs), "launches": n,
                    "names": sorted({r.name for r in recs})}
        k = len(recs) // n
        launches = [recs[i * k:(i + 1) * k] for i in range(n)]
    out = launch_stats(launches)
    out["launches"] = n
    return out


def capture_graph(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = fn()
    torch.cuda.synchronize()
    return graph, static_out


def time_case(case: Case, prof) -> dict:
    from profiling.profilers.timer import Timer

    kw = dict(case.timer_kw)
    warmup = kw.pop("warmup", 0)
    name = kw.get("kernel_name")
    res: dict = {"timer_kw": {**case.timer_kw}, "timer_warmup": warmup}
    t0 = time.perf_counter()
    res["cold_eager_ms"] = Timer.cupti(case.fn, warmup=warmup, clear_l2=True, **kw)
    res["warm_eager_ms"] = Timer.cupti(case.fn, warmup=warmup, clear_l2=False, **kw)
    res["cold_eager"] = burst(prof, case.fn, COLD_SINGLE, name, cold=True)
    res["warm_eager"] = burst(prof, case.fn, BURST, name, cold=False)
    graph, static_out = capture_graph(case.fn)
    graph.replay()
    torch.cuda.synchronize()
    res["oracle_after_first_replay"] = case.check(static_out)
    try:
        res["graph_warm_ms"] = Timer.cupti(graph.replay, warmup=warmup, clear_l2=False, **kw)
        res["graph_warm"] = burst(prof, graph.replay, BURST, name, cold=False)
    except Exception as exc:  # noqa: BLE001 - CUPTI may not see graph kernels; fall back
        res["graph_cupti_error"] = repr(exc)
    # Device-time interval of N back-to-back replays (includes inter-kernel gaps).
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(BURST_WARMUP):
        graph.replay()
    start.record()
    for _ in range(BURST):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    res["graph_event_ms_per_replay"] = start.elapsed_time(end) / BURST
    res["oracle_after_timed_replays"] = case.check(static_out)
    res["timing_wall_s"] = time.perf_counter() - t0
    del graph
    return res


def identity_case(case: Case) -> dict:
    nvtx = torch.cuda.nvtx
    torch.cuda.synchronize()
    nvtx.range_push(f"{case.name}:eager")
    for _ in range(3):
        case.fn()
    torch.cuda.synchronize()
    nvtx.range_pop()
    graph, static_out = capture_graph(case.fn)
    nvtx.range_push(f"{case.name}:graph")
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    nvtx.range_pop()
    rep = case.check(static_out)
    del graph
    return {"oracle_after_replay": rep}


def environment(dev, expect_uuid: str) -> dict:
    import importlib.metadata as md

    props = torch.cuda.get_device_properties(dev)
    uuid = f"GPU-{props.uuid}"
    env = {
        "device_count": torch.cuda.device_count(),
        "device_uuid": uuid, "device_name": props.name, "sms": props.multi_processor_count,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": sys.executable,
    }
    for pkg in ("vllm", "flashinfer-python", "triton", "nvidia-cublas", "nvidia-cuda-cupti"):
        try:
            env[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            env[pkg] = None
    try:
        lib = ctypes.CDLL("libcublasLt.so.13")
        lib.cublasLtGetVersion.restype = ctypes.c_size_t
        env["cublasLt_version"] = int(lib.cublasLtGetVersion())
    except OSError as exc:
        env["cublasLt_version"] = f"unavailable: {exc}"
    if uuid != expect_uuid or torch.cuda.device_count() != 1:
        raise SystemExit(9)
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("timing", "identity"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect-uuid", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"{a.mode}.json"
    started = time.time()
    dev = torch.device("cuda", 0)
    doc: dict = {"mode": a.mode, "started_unix": started, "cases": {}, "env": None}

    def flush(status=None):
        doc["elapsed_s"] = time.time() - started
        doc["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(dev)
        doc["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(dev)
        if status:
            doc["status"] = status
        path.write_text(json.dumps(doc, indent=1, default=str) + "\n")

    try:
        doc["env"] = environment(dev, a.expect_uuid)
    except SystemExit:
        doc["status"] = "device mismatch"
        path.write_text(json.dumps(doc, indent=1, default=str) + "\n")
        return 9
    prof = None
    if a.mode == "timing":
        from profiling.profilers.cupti_kernel_profiler import CuptiKernelProfiler

        prof = CuptiKernelProfiler(device=dev)
    flush("running")
    for case_name, build in build_cases(dev, set(a.only) if a.only else None):
        t0 = time.perf_counter()
        entry: dict = {}
        try:
            case = build()
            entry.update(part=case.part, meta=case.meta, trace=case.trace, timer_kw=case.timer_kw)
            entry["build_s"] = time.perf_counter() - t0
            first = case.fn()
            torch.cuda.synchronize()
            entry["oracle"] = case.check(first)
            entry.update(time_case(case, prof) if a.mode == "timing" else identity_case(case))
            entry["status"] = "pass"
            doc["cases"][case_name] = entry
            flush()
            del case, first
            torch.cuda.empty_cache()
        except OracleFailure as exc:
            entry.update(status="oracle_failure", error=str(exc))
            doc["cases"][case_name] = entry
            flush(f"stopped: oracle failure in {case_name}: {exc}")
            return 3
        except Exception as exc:  # noqa: BLE001
            entry.update(status="error", error=repr(exc), traceback=traceback.format_exc())
            doc["cases"][case_name] = entry
            flush(f"stopped: error in {case_name}: {exc!r}")
            return 4
    flush("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
