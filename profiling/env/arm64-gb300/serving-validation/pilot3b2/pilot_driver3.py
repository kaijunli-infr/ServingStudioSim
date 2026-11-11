#!/usr/bin/env python
"""Phase 3B measurement-validation pilot, continuation (P5 controls, P7-P10). No database writes.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=<profiling worktree>:<vllm fork> <serving venv>/bin/python \\
        pilot_driver3.py --mode timing --parts p7 p8 p9 --out DIR --expect-uuid GPU-...
        pilot_driver3.py --mode control --tune 2048|8192 --out DIR --expect-uuid GPU-...
    nsys profile ... <same> pilot_driver3.py --mode identity --out DIR --expect-uuid GPU-...
    pilot_driver3.py --list                       (CPU: print the case plan and exit)
    pilot_driver3.py --selftest N --out DIR ...   (CPU: stand-in cases through the same loop)

pilot_driver2.py (as run on 2026-11-11, stopped) plus:

Durability (Gate 1). Each completed case is persisted at once to
``<out>/cases/<seq>-<case>.json``: written to a temporary file, flushed,
fsync'd, renamed over the final name, and the directory fsync'd. A case is
complete only when that rename has happened, which is after its correctness
checks and measurements. ``<out>/journal.jsonl`` records start / done /
not_run / failed / stopped events, each line flushed and fsync'd. SIGINT and
SIGTERM are blocked while a case file or journal line is written and delivered
right after, so a signal never leaves a half-written record. On either signal
the driver records the stop reason, the last completed case and the case in
progress (journal and ``<out>/stop.json``), writes the summary
``<out>/<mode>.json`` from the persisted cases and exits 128 + signal. The
guard accounts the occupancy and cleans up its own processes.

Modes.
  timing    the parts named by ``--parts`` (p5 .. p9) of the plan below.
  control   the P5 tuning controls at ``--tune`` (next-input norm at 8 and 2,048
            tokens, post and final at 8), named ``p5c_<role>_tune<N>_T<t>``. The
            guard runs each tuning size in its own process with its own,
            initially empty, Inductor and Triton caches.
  identity  IDENTITY_CASES under Nsight Systems.
FlashInfer. The trtllm-gen FMHA module is prebuilt (Gate 0). Around the first
P8 case the driver snapshots the module's build directory (every file's size and
mtime, build.ninja and .so hashes) and records whether anything changed.

  P5  multi_add_rms_norm at tune_tokens 2048, rms_eps 1e-5: next-input
      (3 addends, sum + normed) at 1, 8, 64, 256, 568, 1024, 2048 tokens; post
      (2, normed) and final (3, normed) at 8, 256, 2048.
  P6  embedding_rms_norm at 1, 8, 64, 256, 568, 2048 tokens; at 8, 256 and
      2048 also the graph-placement diagnostic (``placement``).
  P7  kv_cache_append vllm_cuda, HND_PACKED + value_row_stride 6144, at 1, 8,
      64, 256, 568, 1024, 2048 tokens, whole-cache bit-exact oracle.
  P8  paged attention P16 and P32 on identical logical workloads (the P32
      cache and queries are copied token by token from the P16 ones) at the
      runner's split policy: the page-size effect. Batch-1 decode at P16 under
      serving's FULL-graph (max_model_len 8192) and eager (own context) split
      policies (``SPLIT_CASES``): the split effect.
  P9  silu_and_mul and gqa_rope_neox at 8, 256, 568, 1024, 2048, 4096, 8192
      tokens: the method comparison, then three independent calls of the
      vllm_inductor_graph runner's two steps (``_compiled_case`` +
      ``time_graph_replay``), each recompiling.
  P10 is analysis of the P8 cases (per-kernel cold and graph durations, union,
      span, gap and overlap per launch); nothing extra runs for it.

timing / control mode, per case:
  cold_eager   Timer.cupti(fn) as the runner calls it (L2 flushed before each launch)
  warm_eager   Timer.cupti(fn, clear_l2=False)
  graph_warm   fn captured once in a CUDA graph; Timer.cupti(graph.replay, clear_l2=False)
  bursts       CUPTI records of 30 cold single launches, 50 warm eager launches
               and 50 graph replays: per-kernel durations and, per launch, the
               kernels' sum, union, span, gap (span - union) and overlap (sum - union).
  oracle       before timing, after the first graph replay, after the timed replays.
identity mode: 3 eager launches and 3 graph replays inside NVTX ranges
  ``<case>:eager`` / ``<case>:graph`` for Nsight Systems.

A soft deadline (``--soft-deadline-s``) stops starting new cases and marks the
rest ``not_run``; the guard's hard timeout stays the stop condition.
Exit: 0 done, 3 oracle failure, 4 error, 9 device mismatch, 128 + signal on a stop.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import signal
import statistics as st
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HIDDEN, INTER, HQ, HKV, HEAD, VOCAB = 4096, 14336, 32, 8, 128, 128256
QKV_ROW = (HQ + 2 * HKV) * HEAD
MAX_POSITION = 131072
TUNE, EPS = 2048, 1e-5
COLD_SINGLE = 30
BURST = 50
BURST_WARMUP = 5
INDEPENDENT_CALLS = 3

GEOM = dict(
    num_qo_heads=HQ,
    num_kv_heads=HKV,
    head_dim=HEAD,
    q_dtype="bf16",
    kv_dtype="bf16",
    o_dtype="bf16",
)
# (label, kind, args, trace class). Decode args are the runner's key; the
# traced ones reproduce the trace's batches exactly (canonical batch = traced).
ATTENTION_SHAPES = [
    ("dec_b1_568", "decode", dict(batch_size=1, max_kv_len=568, total_kv_len=568), "mixed2048"),
    ("dec_b4_630", "decode", dict(batch_size=4, max_kv_len=630, total_kv_len=2520), "decode4"),
    ("dec_b4_569", "decode", dict(batch_size=4, max_kv_len=569, total_kv_len=2273), "mixed1933"),
    ("dec_b7_630", "decode", dict(batch_size=7, max_kv_len=630, total_kv_len=4407), "decode7"),
    ("dec_b8_600", "decode", dict(batch_size=8, max_kv_len=600, total_kv_len=4789), "decode8"),
    ("dec_b1_4096", "decode", dict(batch_size=1, max_kv_len=4096, total_kv_len=4096), None),
    ("dec_b1_8192", "decode", dict(batch_size=1, max_kv_len=8192, total_kv_len=8192), None),
    ("dec_b1_32768", "decode", dict(batch_size=1, max_kv_len=32768, total_kv_len=32768), None),
    ("dec_b32_4096", "decode", dict(batch_size=32, max_kv_len=4096, total_kv_len=32 * 4096), None),
    (
        "dec_b128_4096",
        "decode",
        dict(batch_size=128, max_kv_len=4096, total_kv_len=128 * 4096),
        None,
    ),
    ("pre_256", "prefill", dict(requests=((0, 256),)), "prefill256"),
    ("pre_568", "prefill", dict(requests=((0, 568),)), "prefill568"),
    ("pre_mixed2048", "prefill", dict(requests=((0, 568),) * 3 + ((0, 343),)), "mixed2048"),
    ("pre_mixed1933", "prefill", dict(requests=((0, 568),) * 3 + ((343, 225),)), "mixed1933"),
    ("pre_2048", "prefill", dict(requests=((0, 2048),)), None),
]
# Batch-1 decode under serving's policies (P16): max_seq_len = max_model_len
# (FULL graph, 8192 in the traced deployment) or the step's own context (eager).
SPLIT_CASES = [
    ("dec_b1_568_eager_msl", 568, 568, "mixed2048"),
    ("dec_b1_4096_graph_msl", 4096, 8192, None),
    ("dec_b1_4096_eager_msl", 4096, 4096, None),
    ("dec_b1_8192_graph_msl", 8192, 8192, None),
]
TOKEN_CLASS = {8: "decode8", 256: "prefill256", 568: "prefill568", 2048: "mixed2048"}
IDENTITY_CASES = {
    "p5_next_T8",
    "p5_next_T2048",
    "p5_post_T8",
    "p5_final_T8",
    "p6_embed_T8",
    "p6_embed_T2048",
    *(f"p7_kv_T{t}" for t in (8, 256, 2048)),
    *(
        f"p8_{s}_P{p}"
        for s in (
            "dec_b1_568",
            "dec_b4_630",
            "dec_b8_600",
            "dec_b128_4096",
            "pre_256",
            "pre_568",
            "pre_2048",
            "pre_mixed2048",
        )
        for p in (16, 32)
    ),
    *(f"p8_{c[0]}" for c in SPLIT_CASES),
    *(f"p9_{k}_T{t}" for k in ("silu", "rope") for t in (8, 2048, 8192)),
}


class OracleFailure(Exception):
    pass


@dataclass
class Case:
    name: str
    part: str
    fn: Callable[[], Any]
    check: Callable[[Any], dict]  # takes fn's return (or the graph's static output)
    timer_kw: dict
    meta: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)
    extra: Callable[[Any], dict] | None = None  # diagnostic run after the timing


def _torch():
    import torch

    return torch


def _strides(**tensors: Any) -> dict:
    return {
        k: {"shape": list(v.shape), "stride": list(v.stride()), "dtype": str(v.dtype)}
        for k, v in tensors.items()
    }


def _oracle(label: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except (AssertionError, Exception) as exc:  # noqa: BLE001 - every oracle error is a failure
        raise OracleFailure(f"{label}: {str(exc).splitlines()[0] if str(exc) else exc!r}") from exc


# ---------------------------------------------------------------- P5 norms
def norm_case(dev, tokens, addends, store_sum, label, role, tune=TUNE) -> Case:
    torch = _torch()
    from profiling.runners.norm import multi_add_rms_norm_vllm_inductor as R
    from profiling.runners.norm.multi_add_rms_norm_reference import multi_add_rms_norm_reference
    from profiling.runners.vllm_inductor import compile_like_vllm

    R._validate_args(tokens, HIDDEN, addends, store_sum, True, False, "bf16", tune, EPS)
    block = R._build_block(store_sum, True, False, addends, EPS)
    gen = torch.Generator(device=dev).manual_seed(0)
    weight = 1.0 + 0.1 * torch.randn(HIDDEN, dtype=torch.bfloat16, device=dev, generator=gen)
    scale = torch.tensor(R._FP8_SCALE, dtype=torch.float32, device=dev)

    def make_inputs(t):
        return [
            *(
                torch.randn((t, HIDDEN), dtype=torch.bfloat16, device=dev, generator=gen)
                for _ in range(addends)
            ),
            weight,
            scale,
        ]

    compiled = compile_like_vllm(
        torch, block, make_inputs, {i: 0 for i in range(addends)}, tokens, tune_tokens=tune
    )
    inputs = make_inputs(tokens)

    def check(outputs):
        total, normed, _ = multi_add_rms_norm_reference(torch, inputs[:addends], weight, scale, EPS)
        want = ([("sum", total, 1e-2)] if store_sum else []) + [("normed", normed, 2e-2)]
        if len(outputs) != len(want):
            raise OracleFailure(f"{label}: {len(outputs)} stored planes, expected {len(want)}")
        rep = {"planes": [w[0] for w in want]}
        for got, (key, ref, tol) in zip(outputs, want):
            rep[f"{key}_max_abs"] = float((got.float() - ref.float()).abs().max())
            _oracle(
                f"{label} {key}",
                lambda: torch.testing.assert_close(got.float(), ref.float(), rtol=tol, atol=tol),
            )
        return rep

    return Case(
        label,
        "P5",
        lambda: compiled(*inputs),
        check,
        {"kernel_name": None},
        {
            "tokens": tokens,
            "addends": addends,
            "store_sum": store_sum,
            "store_normed": True,
            "tune_tokens": tune,
            "rms_eps": EPS,
            **_strides(addend0=inputs[0]),
        },
        {"class": TOKEN_CLASS.get(tokens), "role": role},
    )


# ---------------------------------------------------------------- P6 embedding + norm
def embed_case(dev, tokens) -> Case:
    torch = _torch()
    from vllm import ir

    from profiling.runners.norm import embedding_rms_norm_vllm_inductor as E
    from profiling.runners.vllm_inductor import compile_like_vllm

    E.validate_args(tokens, HIDDEN, VOCAB, "bf16", EPS, TUNE)
    gen = torch.Generator(device=dev).manual_seed(0)
    table = E._TABLE_STD * torch.randn(
        (VOCAB, HIDDEN), dtype=torch.bfloat16, device=dev, generator=gen
    )
    weight = 1.0 + 0.1 * torch.randn(HIDDEN, dtype=torch.bfloat16, device=dev, generator=gen)
    block = E.build_block(
        ir.ops.rms_norm.impls["native"].impl_fn, torch.nn.functional.embedding, EPS
    )

    def make_inputs(t):
        ids = torch.randint(0, VOCAB, (t,), dtype=torch.int32, device=dev, generator=gen)
        return [ids, table, weight]

    compiled = compile_like_vllm(torch, block, make_inputs, {0: 0}, tokens, tune_tokens=TUNE)
    inputs = make_inputs(tokens)
    label = f"p6_embed_T{tokens}"

    def check(outputs):
        residual, normed = outputs
        _oracle(
            label, lambda: E.check_outputs(torch, inputs[0], table, weight, EPS, residual, normed)
        )
        return {
            "residual_bitwise": True,
            "normed_max_abs": float(
                (
                    normed.float()
                    - table[inputs[0].long()].float()
                    * torch.rsqrt(
                        table[inputs[0].long()].float().square().mean(-1, keepdim=True) + EPS
                    )
                    * weight.float()
                )
                .abs()
                .max()
            ),
        }

    case = Case(
        label,
        "P6",
        lambda: compiled(*inputs),
        check,
        {"kernel_name": None},
        {
            "tokens": tokens,
            "vocab": VOCAB,
            "tune_tokens": TUNE,
            "rms_eps": EPS,
            **_strides(ids=inputs[0], table=table),
        },
        {"class": TOKEN_CLASS.get(tokens), "role": "embed_input_norm"},
    )
    if tokens in (8, 256, 2048):
        case.extra = lambda prof: placement(case, prof)
    return case


def placement(case: Case, prof) -> dict:
    """First-node vs later-node graph placement of the fused kernel.

    Graph A holds the block alone (its kernel is the replay's first node, as in
    the served decode graph). Graph B holds a tiny diagnostic kernel first (an
    in-place add on a 1,024-element buffer), then the block. Each is replayed
    warm (50 back-to-back replays) and cold (30 replays, each after the 64 MiB
    L2 displacement, issued outside the graph), in the order A, B, A to expose
    drift. The block's kernel and the diagnostic kernel are reported separately;
    both graphs' outputs are re-checked afterwards.
    """
    torch = _torch()
    scratch = torch.zeros(1024, dtype=torch.float32, device="cuda")

    def later():
        scratch.add_(1.0)
        return case.fn()

    graphs = {"first": capture_graph(case.fn), "later": capture_graph(later)}
    out: dict = {
        "order": ["first", "later", "first_again"],
        "diagnostic_kernel": "in-place add, 1,024 fp32",
    }
    for label in ("first", "later", "first_again"):
        g, static = graphs["first" if label == "first_again" else label]
        g.replay()
        torch.cuda.synchronize()
        warm = burst(prof, g.replay, BURST, None, cold=False)
        cold = burst(prof, g.replay, COLD_SINGLE, None, cold=True)
        out[label] = {"warm": warm, "cold": cold, "oracle": case.check(static)}
    del graphs
    return out


# ---------------------------------------------------------------- P7 KV append (packed)
def kv_case(dev, tokens) -> Case:
    torch = _torch()
    from vllm import _custom_ops as ops

    from profiling.runners.attention import kv_cache_append as R

    raw, key_cache, value_cache, key, value = R.serving_kv_tensors(
        torch,
        num_tokens=tokens,
        num_kv_heads=HKV,
        head_dim=HEAD,
        block_size=16,
        dtype=torch.bfloat16,
        cache_layout=R.PACKED_LAYOUT,
        value_row_stride=QKV_ROW,
        device=dev,
    )
    slots = R._slot_mapping(torch, num_tokens=tokens, block_size=16)
    k_scale, v_scale = R._scales(torch, num_kv_heads=HKV, scale_granularity="tensor")
    before = raw.clone()
    label = f"p7_kv_T{tokens}"

    def fn():
        ops.reshape_and_cache_flash(
            key, value, key_cache, value_cache, slots, "auto", k_scale, v_scale
        )

    def check(_):
        _oracle(
            label,
            lambda: R.check_append_exact(
                torch,
                raw_before=before,
                raw_after=raw,
                key_cache=key_cache,
                value_cache=value_cache,
                key=key,
                value=value,
                slot_mapping=slots,
                block_size=16,
            ),
        )
        return {"whole_cache_bitwise": True, "cache_elements": int(raw.numel())}

    return Case(
        label,
        "P7",
        fn,
        check,
        {"kernel_name": R._VLLM_KERNEL_NAME, "warmup": 5},
        {
            "tokens": tokens,
            "cache_layout": R.PACKED_LAYOUT,
            "value_row_stride": QKV_ROW,
            **_strides(key=key, value=value, key_cache=key_cache, value_cache=value_cache),
        },
        {"class": TOKEN_CLASS.get(tokens), "role": "kv_cache_append"},
    )


# ---------------------------------------------------------------- P8 attention
_PENDING: dict[str, Case] = {}


def _align(torch, src, dst, lengths, ps_src, ps_dst) -> None:
    """Copy every request's logical K/V tokens and the queries from src to dst."""
    dev = src.raw_cache.device
    for r, length in enumerate(lengths):
        t = torch.arange(length, device=dev)
        sp = src.block_table[r].long()[t // ps_src]
        dp = dst.block_table[r].long()[t // ps_dst]
        dst.raw_cache[dp, :, t % ps_dst, :] = src.raw_cache[sp, :, t % ps_src, :]
    dst.q.copy_(src.q)


def _attn_case(torch, R, label, kind, shape, work, run, trace_class, extra_meta) -> Case:
    checker = R.check_decode if kind == "decode" else R.check_prefill

    def check(_):
        return _oracle(label, lambda: checker(torch, shape, work))

    return Case(
        label,
        "P8",
        run,
        check,
        {"kernel_name": None, "interval_union": True},
        {
            "kind": kind,
            "page_size": shape.geometry.page_size,
            "lengths": shape.lengths,
            **extra_meta,
            **_strides(q=work.q, raw_cache=work.raw_cache, k_cache=work.kv_cache[0]),
        },
        {"class": trace_class, "role": "attention", "attention_kind": kind},
    )


def attention_pair(dev, short, kind, args, trace_class) -> Case:
    """Build the P16 and P32 cases on one logical workload; return P16, park P32."""
    torch = _torch()
    from profiling.runners.attention import paged_attn_batch_trtllm_gen as R

    validate = R.validate_decode_args if kind == "decode" else R.validate_prefill_args
    build = R.build_decode if kind == "decode" else R.build_prefill
    shapes = {p: validate(**GEOM, **args, page_size=p) for p in (16, 32)}
    built = {p: build(torch, shapes[p], dev) for p in (16, 32)}
    _align(torch, built[16][0], built[32][0], shapes[16].lengths, 16, 32)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    meta = (
        {"max_seq_len": R.decode_max_seq_len(args["max_kv_len"], sms)} if kind == "decode" else {}
    )
    meta["max_seq_len_source"] = "runner" if kind == "decode" else None
    # Cross-check: both page sizes compute the same attention on the same tokens.
    for p in (16, 32):
        built[p][1]()
    torch.cuda.synchronize()
    diff = float((built[16][0].out.float() - built[32][0].out.float()).abs().max())
    meta["p16_vs_p32_out_max_abs"] = diff
    if diff > 0.05:
        raise OracleFailure(f"{short}: P16 and P32 outputs differ by {diff}")
    c16 = _attn_case(torch, R, f"p8_{short}_P16", kind, shapes[16], *built[16], trace_class, meta)
    _PENDING[f"p8_{short}_P32"] = _attn_case(
        torch, R, f"p8_{short}_P32", kind, shapes[32], *built[32], trace_class, meta
    )
    return c16


def split_case(dev, label, ctx, msl, trace_class) -> Case:
    """P16 batch-1 decode with an explicit max_seq_len (serving's FULL-graph or eager policy)."""
    torch = _torch()
    from profiling.runners.attention import paged_attn_batch_trtllm_gen as R

    shape = R.validate_decode_args(
        **GEOM, batch_size=1, max_kv_len=ctx, total_kv_len=ctx, page_size=16
    )
    work, _ = R.build_decode(torch, shape, dev)
    decode, _ = R._import_flashinfer(R._DECODE)
    ws = R._workspace(torch, dev)

    def run():  # build_decode's call with max_seq_len set by the serving policy
        return decode.trtllm_batch_decode_with_kv_cache(
            query=work.q,
            kv_cache=work.kv_cache,
            workspace_buffer=ws,
            block_tables=work.block_table,
            seq_lens=work.seq_lens,
            max_seq_len=msl,
            bmm1_scale=HEAD**-0.5,
            bmm2_scale=1.0,
            window_left=-1,
            sinks=None,
            o_sf_scale=None,
            out=work.out,
            kv_layout="HND",
            backend="trtllm-gen",
            q_len_per_req=1,
            max_q_len=None,
            cum_seq_lens_q=None,
            kv_cache_sf=None,
            lse=None,
            return_lse=False,
        )

    return _attn_case(
        torch,
        R,
        f"p8_{label}",
        "decode",
        shape,
        work,
        run,
        trace_class,
        {"max_seq_len": msl, "max_seq_len_source": "serving"},
    )


# ---------------------------------------------------------------- P9 SiLU / RoPE
def _p9_compiled(kind, tokens):
    torch = _torch()
    if kind == "silu":
        from profiling.runners.elementwise import silu_and_mul_vllm_inductor as M

        return M._compiled_case(torch, tokens, INTER)
    from profiling.runners.attention import gqa_rope_neox_vllm_inductor as M

    return M._compiled_case(torch, tokens, HQ, HKV, HEAD, HEAD, MAX_POSITION, M._GRAPH_BACKEND)


def p9_case(dev, kind, tokens) -> Case:
    label = f"p9_{kind}_T{tokens}"
    launch, runner_check = _p9_compiled(kind, tokens)

    def check(out):
        _oracle(label, lambda: runner_check(out))
        return {"oracle": "pass"}

    case = Case(
        label,
        "P9",
        launch,
        check,
        {"kernel_name": None},
        {"tokens": tokens},
        {"class": TOKEN_CLASS.get(tokens), "role": "activation" if kind == "silu" else "rope"},
    )
    case.extra = lambda prof: independent_calls(kind, tokens)
    return case


def independent_calls(kind, tokens) -> dict:
    """The vllm_inductor_graph runner's two steps, three times, each recompiling."""
    from profiling.profilers.graph_replay import MEASUREMENT_METHOD, time_graph_replay

    torch = _torch()
    calls = []
    for _ in range(INDEPENDENT_CALLS):
        t0 = time.perf_counter()
        launch, check = _p9_compiled(kind, tokens)
        timing, _replay = time_graph_replay(launch, check, torch=torch)
        calls.append(
            {
                "time_ms": timing.time_ms,
                "batch_ms": list(timing.batch_ms),
                "kernel_names": list(timing.kernel_names),
                "wall_s": time.perf_counter() - t0,
            }
        )
        del _replay
        torch.cuda.empty_cache()
    times = [c["time_ms"] for c in calls]
    within = [(max(c["batch_ms"]) - min(c["batch_ms"])) / st.median(c["batch_ms"]) for c in calls]
    return {
        "method": MEASUREMENT_METHOD,
        "calls": calls,
        "between_call_range_rel": (max(times) - min(times)) / st.median(times),
        "within_call_batch_range_rel_max": max(within),
        "kernel_sequences": sorted({json.dumps(c["kernel_names"]) for c in calls}),
    }


# ---------------------------------------------------------------- plan
def build_cases(dev) -> list[tuple[str, Callable[[], Case]]]:
    b: list[tuple[str, Callable[[], Case]]] = []
    for t in (1, 8, 64, 256, 568, 1024, 2048):
        b.append(
            (f"p5_next_T{t}", lambda t=t: norm_case(dev, t, 3, True, f"p5_next_T{t}", "input_norm"))
        )
    for t in (8, 256, 2048):
        b.append(
            (f"p5_post_T{t}", lambda t=t: norm_case(dev, t, 2, False, f"p5_post_T{t}", "post_norm"))
        )
        b.append(
            (
                f"p5_final_T{t}",
                lambda t=t: norm_case(dev, t, 3, False, f"p5_final_T{t}", "final_norm"),
            )
        )
    for t in (1, 8, 64, 256, 568, 2048):
        b.append((f"p6_embed_T{t}", lambda t=t: embed_case(dev, t)))
    for t in (1, 8, 64, 256, 568, 1024, 2048):
        b.append((f"p7_kv_T{t}", lambda t=t: kv_case(dev, t)))
    for short, kind, args, cls in ATTENTION_SHAPES:
        b.append(
            (
                f"p8_{short}_P16",
                lambda s=short, k=kind, a=args, c=cls: attention_pair(dev, s, k, a, c),
            )
        )
        b.append((f"p8_{short}_P32", lambda s=short: _PENDING.pop(f"p8_{s}_P32")))
    for label, ctx, msl, cls in SPLIT_CASES:
        b.append((f"p8_{label}", lambda n=label, c=ctx, m=msl, k=cls: split_case(dev, n, c, m, k)))
    for kind in ("silu", "rope"):
        for t in (8, 256, 568, 1024, 2048, 4096, 8192):
            b.append((f"p9_{kind}_T{t}", lambda k=kind, t=t: p9_case(dev, k, t)))
    return b


def control_cases(dev, tune: int) -> list[tuple[str, Callable[[], Case]]]:
    """P5 tuning controls at one tuning size (run in a process with its own caches)."""
    spec = [
        ("next", 3, True, "input_norm", 8),
        ("next", 3, True, "input_norm", 2048),
        ("post", 2, False, "post_norm", 8),
        ("final", 3, False, "final_norm", 8),
    ]
    return [
        (
            f"p5c_{role}_tune{tune}_T{t}",
            lambda r=role, a=add, s=ss, ro=trole, t=t: norm_case(
                dev, t, a, s, f"p5c_{r}_tune{tune}_T{t}", ro, tune=tune
            ),
        )
        for role, add, ss, trole, t in spec
    ]


# ---------------------------------------------------------------- measurement
def summary(values: list[float]) -> dict | None:
    if not values:
        return None
    mean = st.fmean(values)
    sd = st.pstdev(values) if len(values) > 1 else 0.0
    s = sorted(values)
    return {
        "n": len(values),
        "mean": mean,
        "median": st.median(values),
        "min": s[0],
        "p90": s[min(len(s) - 1, int(0.9 * (len(s) - 1) + 0.5))],
        "max": s[-1],
        "cv": sd / mean if mean else None,
    }


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
    sums, unions, spans, gaps, overlaps, seqs = [], [], [], [], [], {}
    for recs in launches:
        recs = sorted(recs, key=lambda r: r.start_ns)
        seq = tuple(r.name for r in recs)
        seqs[seq] = seqs.get(seq, 0) + 1
        for r in recs:
            per_name.setdefault(r.name, []).append(r.duration_ns / 1e3)
        total = sum(r.duration_ns for r in recs) / 1e3
        union = _union([(r.start_ns, r.end_ns) for r in recs]) / 1e3
        span = (max(r.end_ns for r in recs) - recs[0].start_ns) / 1e3 if recs else 0.0
        sums.append(total)
        unions.append(union)
        spans.append(span)
        gaps.append(span - union)
        overlaps.append(total - union)
    return {
        "sequences": [{"kernels": list(k), "count": v} for k, v in seqs.items()],
        "per_kernel_us": {k: summary(v) for k, v in per_name.items()},
        "launch_sum_us": summary(sums),
        "launch_union_us": summary(unions),
        "launch_span_us": summary(spans),
        "launch_gap_us": summary(gaps),
        "launch_overlap_us": summary(overlaps),
    }


def _filter(records, kernel_name):
    return [r for r in records if kernel_name is None or kernel_name in r.name]


def burst(prof, fn, n, kernel_name, cold) -> dict:
    torch = _torch()
    if cold:
        launches = [
            _filter(
                prof.capture_once(
                    fn,
                    launches_per_run=1,
                    clear_l2_before_run=True,
                    clear_l2_between_launches=False,
                ),
                kernel_name,
            )
            for _ in range(n)
        ]
    else:
        for _ in range(BURST_WARMUP):
            fn()
        torch.cuda.synchronize()
        recs = sorted(
            _filter(
                prof.capture_once(
                    fn,
                    launches_per_run=n,
                    clear_l2_before_run=False,
                    clear_l2_between_launches=False,
                ),
                kernel_name,
            ),
            key=lambda r: r.start_ns,
        )
        if not recs or len(recs) % n:
            return {
                "irregular_record_count": len(recs),
                "launches": n,
                "names": sorted({r.name for r in recs}),
            }
        k = len(recs) // n
        launches = [recs[i * k : (i + 1) * k] for i in range(n)]
    out = launch_stats(launches)
    out["launches"] = n
    return out


def capture_graph(fn):
    torch = _torch()
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
    torch = _torch()
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
    res["graph_warm_ms"] = Timer.cupti(graph.replay, warmup=warmup, clear_l2=False, **kw)
    res["graph_warm"] = burst(prof, graph.replay, BURST, name, cold=False)
    res["oracle_after_timed_replays"] = case.check(static_out)
    res["timing_wall_s"] = time.perf_counter() - t0
    del graph
    if case.extra is not None:
        t1 = time.perf_counter()
        res["extra"] = case.extra(prof)
        res["extra_wall_s"] = time.perf_counter() - t1
    return res


def identity_case(case: Case) -> dict:
    torch = _torch()
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

    torch = _torch()
    props = torch.cuda.get_device_properties(dev)
    uuid = f"GPU-{props.uuid}"
    env = {
        "device_count": torch.cuda.device_count(),
        "device_uuid": uuid,
        "device_name": props.name,
        "sms": props.multi_processor_count,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": sys.executable,
        "energy": os.environ.get("VIBESIM_PROFILE_ENERGY"),
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


# ---------------------------------------------------------------- durability (Gate 1)
_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class StopRequested(BaseException):
    """Raised by the SIGINT/SIGTERM handler; not an Exception, so no case code swallows it."""

    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


def _on_stop_signal(signum, _frame):
    raise StopRequested(signum)


@contextlib.contextmanager
def _signals_deferred():
    """Block SIGINT/SIGTERM for the duration; a pending one is delivered on exit."""
    old = signal.pthread_sigmask(signal.SIG_BLOCK, _STOP_SIGNALS)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(obj, indent=1, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


class Recorder:
    """Durable per-case results and an event journal (see the module docstring)."""

    def __init__(self, out: Path, persist_delay_s: float = 0.0):
        self.out = out
        self.cases_dir = out / "cases"
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.journal = open(out / "journal.jsonl", "a")  # noqa: SIM115 - held for the run
        self.completed: list[str] = []
        self.in_progress: str | None = None
        self.seq = 0
        self.persist_delay_s = persist_delay_s  # self-test only: widens the write window

    def event(self, **fields: Any) -> None:
        with _signals_deferred():
            self.journal.write(json.dumps({"t": time.time(), **fields}, default=str) + "\n")
            self.journal.flush()
            os.fsync(self.journal.fileno())

    def begin(self, name: str) -> None:
        self.in_progress = name
        self.event(event="start", case=name)

    def persist(self, name: str, entry: dict) -> Path:
        """Durably write a completed case; only then is it counted as complete."""
        with _signals_deferred():
            self.seq += 1
            path = self.cases_dir / f"{self.seq:03d}-{name}.json"
            if self.persist_delay_s:
                time.sleep(self.persist_delay_s)
            atomic_write_json(path, {"case": name, **entry})
            self.completed.append(name)
            self.in_progress = None
            self.journal.write(
                json.dumps({"t": time.time(), "event": "done", "case": name, "file": path.name})
                + "\n"
            )
            self.journal.flush()
            os.fsync(self.journal.fileno())
        return path

    def failed(self, name: str, entry: dict) -> None:
        with _signals_deferred():
            atomic_write_json(self.cases_dir / f"failed-{name}.json", {"case": name, **entry})
        self.event(event="failed", case=name, status=entry.get("status"))
        self.in_progress = None

    def stopped(self, reason: str) -> dict:
        rec = {
            "reason": reason,
            "last_completed": self.completed[-1] if self.completed else None,
            "in_progress": self.in_progress,
            "completed": len(self.completed),
        }
        with _signals_deferred():
            atomic_write_json(self.out / "stop.json", rec)
        self.event(event="stopped", **rec)
        return rec

    def cases(self) -> dict:
        out = {}
        for path in sorted(self.cases_dir.glob("[0-9][0-9][0-9]-*.json")):
            entry = json.loads(path.read_text())
            out[entry.pop("case")] = entry
        return out


def run_plan(rec, plan, execute, *, wanted, soft_deadline_s, started, gap_s=0.0) -> None:
    """Run ``plan`` through ``execute``; persist each completed case before the next."""
    for case_name, build in plan:
        p32_partner = case_name.endswith("_P32") and case_name in _PENDING
        if wanted is not None and case_name not in wanted:
            if p32_partner:
                _PENDING.pop(case_name)
            if not (case_name.endswith("_P16") and case_name.replace("_P16", "_P32") in wanted):
                continue
        if time.time() - started > soft_deadline_s:
            _PENDING.pop(case_name, None)
            rec.event(event="not_run", case=case_name, reason="soft deadline")
            continue
        rec.begin(case_name)
        entry = execute(case_name, build, wanted)
        if entry is None:  # built only to park its P32 partner
            rec.in_progress = None
            continue
        rec.persist(case_name, entry)
        if gap_s:
            time.sleep(gap_s)


class CaseFailed(Exception):
    def __init__(self, entry: dict, code: int):
        super().__init__(entry.get("error"))
        self.entry, self.code = entry, code


def fi_snapshot() -> dict | None:
    """The prebuilt FlashInfer FMHA module's build directory (Gate 0), or None."""
    base = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    if not base:
        return None
    dirs = list(Path(base).glob(".cache/flashinfer/*/*/cached_ops/fmha_gen"))
    if len(dirs) != 1:
        return {"error": f"{len(dirs)} fmha_gen directories under {base}"}
    d, files = dirs[0], {}
    for p in sorted(d.rglob("*")):
        if p.is_file():
            st = p.stat()
            files[str(p.relative_to(d))] = [st.st_size, st.st_mtime_ns]
    snap = {"dir": str(d), "files": files}
    for name in ("build.ninja", "fmha_gen.so"):
        if (d / name).exists():
            snap[f"{name}_sha256"] = hashlib.sha256((d / name).read_bytes()).hexdigest()
    return snap


def gpu_executor(dev, prof, mode, fi_state):
    torch = _torch()

    def execute(case_name, build, wanted):
        t0 = time.perf_counter()
        entry: dict = {}
        watch_fi = case_name.startswith("p8_") and fi_state["before"] is None
        if watch_fi:
            fi_state["before"] = fi_snapshot()
        try:
            case = build()
            entry.update(part=case.part, meta=case.meta, trace=case.trace, timer_kw=case.timer_kw)
            entry["build_s"] = time.perf_counter() - t0
            if wanted is not None and case_name not in wanted:
                del case
                return None
            first = case.fn()
            torch.cuda.synchronize()
            entry["oracle"] = case.check(first)
            if watch_fi:
                after = fi_snapshot()
                before = fi_state["before"] or {}
                changed = sorted(
                    k
                    for k in set(before.get("files", {})) | set((after or {}).get("files", {}))
                    if before.get("files", {}).get(k) != (after or {}).get("files", {}).get(k)
                )
                fi_state["check"] = {
                    "case": case_name,
                    "before": before,
                    "after": after,
                    "changed": changed,
                    "rebuilt": bool(changed),
                }
                entry["flashinfer_rebuild_check"] = {"changed": changed, "rebuilt": bool(changed)}
            entry.update(
                time_case(case, prof) if mode in ("timing", "control") else identity_case(case)
            )
            entry["status"] = "pass"
            entry["case_wall_s"] = time.perf_counter() - t0
            del case, first
            if not _PENDING:
                torch.cuda.empty_cache()
            return entry
        except OracleFailure as exc:
            raise CaseFailed({**entry, "status": "oracle_failure", "error": str(exc)}, 3) from exc
        except Exception as exc:  # noqa: BLE001
            raise CaseFailed(
                {
                    **entry,
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
                4,
            ) from exc

    return execute


def selftest_executor(case_s: float):
    def execute(case_name, build, wanted):
        t0 = time.perf_counter()
        time.sleep(case_s)  # the "measurement"
        return {"status": "pass", "value": case_name, "case_wall_s": time.perf_counter() - t0}

    return execute


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("timing", "identity", "control"))
    ap.add_argument("--parts", nargs="*", default=["p7", "p8", "p9"])
    ap.add_argument("--tune", type=int)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--expect-uuid")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--soft-deadline-s", type=float, default=float("inf"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", type=int, help="CPU: run N stand-in cases")
    ap.add_argument("--selftest-case-s", type=float, default=1.0)
    ap.add_argument("--selftest-gap-s", type=float, default=0.0)
    ap.add_argument("--selftest-persist-s", type=float, default=0.0)
    a = ap.parse_args()
    if a.list:
        names = [n for n, _ in build_cases(None)]
        unknown = sorted(IDENTITY_CASES - set(names))
        controls = [n for t in (2048, 8192) for n, _ in control_cases(None, t)]
        print(
            json.dumps(
                {
                    "cases": len(names),
                    "identity_cases": len(IDENTITY_CASES),
                    "identity_unknown": unknown,
                    "controls": controls,
                    "by_part": {
                        p: sum(n.startswith(p + "_") for n in names)
                        for p in ("p5", "p6", "p7", "p8", "p9")
                    },
                },
                indent=1,
            )
        )
        return 1 if unknown else 0

    for sig in _STOP_SIGNALS:
        signal.signal(sig, _on_stop_signal)
    a.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    mode = "selftest" if a.selftest else a.mode
    rec = Recorder(a.out, persist_delay_s=a.selftest_persist_s if a.selftest else 0.0)
    doc: dict = {
        "mode": mode,
        "started_unix": started,
        "soft_deadline_s": a.soft_deadline_s,
        "pid": os.getpid(),
        "env": None,
    }
    rec.event(event="begin", mode=mode, pid=os.getpid())

    def summary_doc(status: str) -> None:
        doc.update(
            status=status,
            elapsed_s=time.time() - started,
            completed=len(rec.completed),
            cases=rec.cases(),
        )
        if mode not in ("selftest",):
            torch = _torch()
            doc["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            doc["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        with _signals_deferred():
            atomic_write_json(a.out / f"{mode}.json", doc)

    fi_state: dict = {"before": None, "check": None}
    try:
        if a.selftest:
            plan = [(f"st_{i:02d}", None) for i in range(a.selftest)]
            execute, wanted = selftest_executor(a.selftest_case_s), None
        else:
            torch = _torch()
            dev = torch.device("cuda", 0)
            try:
                doc["env"] = environment(dev, a.expect_uuid)
            except SystemExit:
                summary_doc("device mismatch")
                return 9
            prof = None
            if a.mode in ("timing", "control"):
                from profiling.profilers.cupti_kernel_profiler import CuptiKernelProfiler

                prof = CuptiKernelProfiler(device=dev)
            if a.mode == "control":
                plan, wanted = control_cases(dev, a.tune), None
            else:
                plan = build_cases(dev)
                if a.only:
                    wanted = set(a.only)
                elif a.mode == "identity":
                    wanted = IDENTITY_CASES
                else:
                    wanted = {n for n, _ in plan if n.split("_")[0] in set(a.parts)}
            execute = gpu_executor(dev, prof, a.mode, fi_state)
        run_plan(
            rec,
            plan,
            execute,
            wanted=wanted,
            soft_deadline_s=a.soft_deadline_s,
            started=started,
            gap_s=a.selftest_gap_s if a.selftest else 0.0,
        )
    except StopRequested as stop:
        for sig in _STOP_SIGNALS:  # a repeated signal must not interrupt the stop record
            signal.signal(sig, signal.SIG_IGN)
        doc["stop"] = rec.stopped(signal.Signals(stop.signum).name)
        doc["flashinfer_rebuild_check"] = fi_state["check"]
        summary_doc("stopped")
        return 128 + stop.signum
    except CaseFailed as fail:
        for sig in _STOP_SIGNALS:
            signal.signal(sig, signal.SIG_IGN)
        name = rec.in_progress or "?"
        rec.failed(name, fail.entry)
        doc["flashinfer_rebuild_check"] = fi_state["check"]
        summary_doc(f"stopped: {fail.entry['status']} in {name}: {fail.entry.get('error')}")
        return fail.code
    doc["flashinfer_rebuild_check"] = fi_state["check"]
    rec.event(event="end")
    summary_doc("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
