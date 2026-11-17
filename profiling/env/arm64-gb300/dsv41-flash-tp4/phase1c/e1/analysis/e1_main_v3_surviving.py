#!/usr/bin/env python3
"""E1 e1_main_v3 surviving-artifact analysis (CPU only; reads the frozen incident archive, writes a report).

    uv run --no-sync python analysis/e1_main_v3_surviving.py --archive <archive>/run/e1_main_v3 \
        --out <e1>/runs/e1_main_v3/surviving --scratch <scratch dir> [--caches label=path ...]

Inputs (all host side; no Nsight trace exists for this run): trace.csv (frozen plan), load_generator/replay.jsonl +
summary.json (client outcomes), routed_experts/*.npy (450 x (1024, 40, 6) uint16), the vLLM server log (VibeSimAlignmentIteration
and VibeSimAlignmentRequestTiming records, sliced at the persisted measurement_log_offset), *_drive_summary.json (measured window),
autotune_configs.*.json (FlashInfer tactic cache of this launch). Nothing is fitted; no partition is written.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import glob
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ID_RE = re.compile(r"^e1_b(\d+)_kv(\d+)k_w(\d+)_r(\d+)$")
NUM_EXPERTS = 384
TOPK = 6
LAYERS = 40
OUTPUT_TOKENS = 1024
BOOT = 400
RNG = np.random.default_rng(20261117)


def parse_id(rid: str) -> tuple[int, int, int, int]:
    m = ID_RE.match(rid.removeprefix("independent_"))
    if not m:
        raise ValueError(f"unexpected request id {rid!r}")
    return tuple(int(x) for x in m.groups())  # b, kv(k), wave, r


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def summary(a) -> dict:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "p10": pct(a, 10), "p50": pct(a, 50), "p90": pct(a, 90), "min": float(a.min()), "max": float(a.max())}


def entropy_bits(p) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def gini(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * (np.arange(1, n + 1) * x).sum()) / (n * x.sum()) - (n + 1) / n)


def jsd_bits(p, q) -> float:
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def bootstrap_ci(values_per_request: list[np.ndarray], stat, b: int = BOOT) -> tuple[float, float, float]:
    """Resample requests (the independent unit) with replacement; stat() takes the pooled array."""
    n = len(values_per_request)
    point = stat(np.concatenate(values_per_request))
    if n < 2:
        return point, float("nan"), float("nan")
    draws = []
    for _ in range(b):
        idx = RNG.integers(0, n, n)
        draws.append(stat(np.concatenate([values_per_request[i] for i in idx])))
    return point, pct(draws, 2.5), pct(draws, 97.5)


# ----------------------------------------------------------------------------------------------------------------------
def load_plan(trace_csv: Path) -> dict[str, dict]:
    plan = {}
    with trace_csv.open() as fh:
        for row in csv.DictReader(fh):
            b, kv, w, r = parse_id(row["id"])
            plan[row["id"]] = {"b": b, "kv": kv, "w": w, "r": r, "input_len": int(row["input_len"]),
                               "output_len": int(row["output_len"]), "arrival_ms": float(row["arrival_time"])}
    return plan


def load_replay(replay_jsonl: Path) -> dict[str, dict]:
    out = {}
    for line in replay_jsonl.open():
        d = json.loads(line)
        rid = d["source"]["data"]["id"]
        out[rid] = {"source": d["source"]["data"], **d["outcome"]}
    return out


def routing_analysis(routes_dir: Path, plan: dict) -> dict:
    files = sorted(routes_dir.glob("*.npy"))
    per_req = {}
    shape_ok = 0
    bad = []
    for f in files:
        rid = f.stem.removeprefix("independent_")
        a = np.load(f, allow_pickle=False)
        if a.shape == (OUTPUT_TOKENS, LAYERS, TOPK) and a.dtype == np.uint16 and a.max() < NUM_EXPERTS and a.min() >= 0:
            shape_ok += 1
        else:
            bad.append({"file": f.name, "shape": list(a.shape), "dtype": str(a.dtype), "max": int(a.max())})
        per_req[rid] = a
    distinct_ok = all(len(np.unique(a[t, l])) == TOPK for a in list(per_req.values())[:20] for t in range(0, OUTPUT_TOKENS, 128) for l in range(LAYERS))
    cells = collections.defaultdict(list)
    for rid, a in per_req.items():
        b, kv, w, r = parse_id(rid)
        cells[(b, kv, w)].append(rid)
    # pooled per-layer marginals
    def marginal(ids) -> np.ndarray:
        counts = np.zeros((LAYERS, NUM_EXPERTS), dtype=np.int64)
        for rid in ids:
            a = per_req[rid]
            for l in range(LAYERS):
                counts[l] += np.bincount(a[:, l].ravel(), minlength=NUM_EXPERTS)
        return counts
    all_counts = marginal(per_req.keys())
    pooled_p = all_counts / all_counts.sum(axis=1, keepdims=True)
    layer_stats = [{"layer": l, "entropy_bits": entropy_bits(pooled_p[l]), "max_over_mean_load": float(pooled_p[l].max() * NUM_EXPERTS),
                    "gini": gini(all_counts[l]), "experts_never_routed": int((all_counts[l] == 0).sum())} for l in range(LAYERS)]
    # per cell: entropy / max-load ratio with bootstrap over requests (pooled across layers = mean over layers)
    def cell_stat_entropy(tokens_layers):   # tokens_layers: (N, LAYERS, TOPK) flattened by concatenate → (N*?,) no; use per-request arrays
        return float("nan")
    cell_rows = []
    for (b, kv, w), ids in sorted(cells.items()):
        per_request_layer_counts = []
        for rid in ids:
            a = per_req[rid]
            c = np.stack([np.bincount(a[:, l].ravel(), minlength=NUM_EXPERTS) for l in range(LAYERS)])  # (LAYERS, E)
            per_request_layer_counts.append(c[None])
        arr = [x for x in per_request_layer_counts]
        def ent(pooled):
            c = pooled.sum(axis=0)
            p = c / c.sum(axis=1, keepdims=True)
            return float(np.mean([entropy_bits(p[l]) for l in range(LAYERS)]))
        def mxl(pooled):
            c = pooled.sum(axis=0)
            p = c / c.sum(axis=1, keepdims=True)
            return float(np.mean(p.max(axis=1) * NUM_EXPERTS))
        e_pt, e_lo, e_hi = bootstrap_ci(arr, ent)
        m_pt, m_lo, m_hi = bootstrap_ci(arr, mxl)
        cell_counts = np.concatenate(arr).sum(axis=0)
        cell_p = cell_counts / cell_counts.sum(axis=1, keepdims=True)
        jsd_vs_pooled = float(np.mean([jsd_bits(cell_p[l], pooled_p[l]) for l in range(LAYERS)]))
        cell_rows.append({"b": b, "kv_k": kv, "wave": w, "requests": len(ids), "tokens": len(ids) * OUTPUT_TOKENS,
                          "mean_layer_entropy_bits": e_pt, "entropy_ci95": [e_lo, e_hi],
                          "mean_max_over_mean_load": m_pt, "max_load_ci95": [m_lo, m_hi],
                          "mean_layer_jsd_vs_pooled_bits": jsd_vs_pooled})
    # wave repeatability: w0 vs w1 for b in 1,2,4,8
    repeat_rows = []
    for b in (1, 2, 4, 8):
        for kv in (4, 32, 96):
            if (b, kv, 0) in cells and (b, kv, 1) in cells:
                c0, c1 = marginal(cells[(b, kv, 0)]), marginal(cells[(b, kv, 1)])
                p0, p1 = c0 / c0.sum(1, keepdims=True), c1 / c1.sum(1, keepdims=True)
                jsd = float(np.mean([jsd_bits(p0[l], p1[l]) for l in range(LAYERS)]))
                # null: split the pooled two-wave requests at random into two halves of the same sizes, B times
                ids = cells[(b, kv, 0)] + cells[(b, kv, 1)]
                n0 = len(cells[(b, kv, 0)])
                null = []
                for _ in range(min(BOOT, 200)):
                    perm = RNG.permutation(len(ids))
                    a0, a1 = marginal([ids[i] for i in perm[:n0]]), marginal([ids[i] for i in perm[n0:]])
                    q0, q1 = a0 / a0.sum(1, keepdims=True), a1 / a1.sum(1, keepdims=True)
                    null.append(float(np.mean([jsd_bits(q0[l], q1[l]) for l in range(LAYERS)])))
                repeat_rows.append({"b": b, "kv_k": kv, "jsd_w0_w1_bits": jsd, "null_p50": pct(null, 50), "null_p97_5": pct(null, 97.5),
                                    "requests_per_wave": n0})
    # temporal locality: fraction of consecutive tokens sharing >= 1 expert per layer (sampled)
    share = []
    for rid in list(per_req)[::9]:
        a = per_req[rid]
        same = [(len(np.intersect1d(a[t, l], a[t + 1, l])) > 0) for t in range(0, OUTPUT_TOKENS - 1, 37) for l in range(0, LAYERS, 5)]
        share.append(float(np.mean(same)))
    return {"files": len(files), "shape_dtype_range_ok": shape_ok, "bad_files": bad, "topk_distinct_sampled_ok": bool(distinct_ok),
            "requests_per_cell": {f"b{b}_kv{kv}k_w{w}": len(ids) for (b, kv, w), ids in sorted(cells.items())},
            "pooled_layer_stats": layer_stats,
            "pooled_mean_layer_entropy_bits": float(np.mean([s["entropy_bits"] for s in layer_stats])),
            "pooled_mean_max_over_mean_load": float(np.mean([s["max_over_mean_load"] for s in layer_stats])),
            "experts_never_routed_any_layer": int((all_counts == 0).sum()),
            "per_cell": cell_rows, "wave_repeatability": repeat_rows,
            "consecutive_token_expert_overlap_fraction": summary(share)}


def slice_measurement_log(server_log: Path, offset: int, out: Path) -> None:
    with server_log.open("rb") as src, out.open("wb") as dst:
        src.seek(offset)
        shutil.copyfileobj(src, dst)


def iteration_analysis(metrics_jsonl: Path, window: tuple[int, int], plan: dict) -> dict:
    rows = [json.loads(l) for l in metrics_jsonl.open()]
    rows.sort(key=lambda r: r["iteration_index"])
    idx = [r["iteration_index"] for r in rows]
    gaps = [(a, b) for a, b in zip(idx, idx[1:]) if b != a + 1]
    w0, w1 = window
    inwin = [r for r in rows if w0 <= r["observed_start_monotonic_ns"] <= w1]
    # classify
    pure = [r for r in inwin if r["prefill_tokens"] == 0 and r["decode_requests"] > 0]
    prefill_only = [r for r in inwin if r["prefill_tokens"] > 0 and r["decode_requests"] == 0]
    mixed = [r for r in inwin if r["prefill_tokens"] > 0 and r["decode_requests"] > 0]
    idle = [r for r in inwin if r["prefill_tokens"] == 0 and r["decode_requests"] == 0]
    plan_b = sorted({v["b"] for v in plan.values()})  # noqa: F841 (kept for reference)
    plan_kv = sorted({v["kv"] for v in plan.values()})

    def kv_bucket(kv_len: int) -> int | None:
        for kv in plan_kv:
            if kv * 1024 <= kv_len <= kv * 1024 + OUTPUT_TOKENS + 8:
                return kv
        return None

    # assign pure-decode iterations to (b, kv, wave) through the request ids the engine recorded in each step
    cell_iters = collections.defaultdict(list)
    partial = collections.Counter()
    unassigned = []
    for r in pure:
        ids = {p["request_id"].removeprefix("independent_") for p in r["decode_request_progress"]}
        cells_of = {(plan[i]["b"], plan[i]["kv"], plan[i]["w"]) for i in ids if i in plan}
        if len(cells_of) == 1 and len(ids) == len(ids & set(plan)):
            cell = next(iter(cells_of))
            if r["decode_requests"] == cell[0]:
                cell_iters[cell].append(r)
            else:
                partial[cell] += 1  # fewer requests than the wave's batch size (wave head/tail)
        else:
            unassigned.append(r)
    cell_rows = []
    for (b, kv, w), wv in sorted(cell_iters.items()):
        wv.sort(key=lambda r: r["observed_start_monotonic_ns"])
        el = np.array([r["observed_elapsed_ms"] for r in wv])
        toks = sum(r["decode_tokens_scheduled"] for r in wv)
        ii = [r["iteration_index"] for r in wv]
        contiguous = all(y == x + 1 for x, y in zip(ii, ii[1:]))
        kvs = np.array([k for r in wv for k in r["decode_kv_lens"]])
        cell_rows.append({"b": b, "kv_k": kv, "wave_observed": w, "pure_decode_iterations": len(wv), "expected_per_wave": OUTPUT_TOKENS - 1,
                          "partial_batch_iterations": int(partial.get((b, kv, w), 0)),
                          "decode_tokens_scheduled": int(toks), "iteration_index_first": ii[0], "iteration_index_last": ii[-1],
                          "iterations_contiguous": contiguous, "elapsed_ms": summary(el),
                          "kv_len_min": int(kvs.min()), "kv_len_max": int(kvs.max()),
                          "elapsed_ms_first_quarter_p50": pct(el[: len(el) // 4], 50) if len(el) >= 8 else float("nan"),
                          "elapsed_ms_last_quarter_p50": pct(el[-(len(el) // 4):], 50) if len(el) >= 8 else float("nan"),
                          "wall_s": (wv[-1]["observed_end_monotonic_ns"] - wv[0]["observed_start_monotonic_ns"]) / 1e9,
                          "clean_ge_400": len(wv) >= 400})
    for (b, kv, w) in sorted({(v["b"], v["kv"], v["w"]) for v in plan.values()} - set(cell_iters)):
        cell_rows.append({"b": b, "kv_k": kv, "note": f"no full-batch pure-decode iteration found for wave {w}"})
    total_decode_tokens = sum(r["decode_tokens_scheduled"] for r in inwin)
    total_prefill_tokens = sum(r["prefill_tokens"] for r in inwin)
    per_b_pure = collections.Counter(r["decode_requests"] for r in pure)
    return {"records_total": len(rows), "iteration_index_gaps": gaps[:20], "iteration_index_gap_count": len(gaps),
            "records_in_measured_window": len(inwin), "window_first_index": inwin[0]["iteration_index"] if inwin else None,
            "window_last_index": inwin[-1]["iteration_index"] if inwin else None,
            "pure_decode": len(pure), "prefill_only": len(prefill_only), "mixed": len(mixed), "idle_or_empty": len(idle),
            "pure_decode_unassigned_to_a_plan_cell": len(unassigned),
            "unassigned_examples": [{"iteration_index": r["iteration_index"], "decode_requests": r["decode_requests"],
                                     "kv_lens_minmax": [min(r["decode_kv_lens"]), max(r["decode_kv_lens"])]} for r in unassigned[:10]],
            "pure_decode_iterations_by_batch": {str(k): v for k, v in sorted(per_b_pure.items())},
            "decode_tokens_scheduled_in_window": int(total_decode_tokens), "prefill_tokens_in_window": int(total_prefill_tokens),
            "expected_decode_tokens": len(plan) * (OUTPUT_TOKENS - 1), "expected_prefill_tokens": sum(v["input_len"] for v in plan.values()),
            "window_wall_s": (inwin[-1]["observed_end_monotonic_ns"] - inwin[0]["observed_start_monotonic_ns"]) / 1e9 if inwin else None,
            "cells": cell_rows}


def request_timing_analysis(timings_jsonl: Path, replay: dict, plan: dict) -> dict:
    rows = {}
    for l in timings_jsonl.open():
        d = json.loads(l)
        rows[d["request_id"].removeprefix("independent_")] = d
    missing = sorted(set(plan) - set(rows))
    extra = sorted(set(rows) - set(plan))
    cells = collections.defaultdict(list)
    for rid, d in rows.items():
        b, kv, w, r = parse_id(rid)
        cells[(b, kv, w)].append((d, replay.get(rid)))
    out = []
    for (b, kv, w), lst in sorted(cells.items()):
        ttft = np.array([d["engine_core_ttft_ms"] for d, _ in lst])
        tpot = np.array([d["engine_core_tpot_ms"] for d, _ in lst])
        dec = np.array([d["engine_core_decode_ms"] for d, _ in lst])
        client_total = np.array([o["total_duration_ms"] for _, o in lst if o])
        ntok = sorted({d["num_output_tokens"] for d, _ in lst})
        out.append({"b": b, "kv_k": kv, "wave": w, "requests": len(lst), "num_output_tokens": ntok,
                    "engine_core_ttft_ms": summary(ttft), "engine_core_tpot_ms": summary(tpot), "engine_core_decode_ms": summary(dec),
                    "client_total_duration_ms": summary(client_total),
                    "client_minus_engine_ms_p50": pct(client_total - (ttft + dec), 50) if len(client_total) == len(lst) else None})
    return {"records": len(rows), "missing_request_timings": missing, "unexpected_request_ids": extra, "cells": out}


def plan_deviation(plan: dict, replay: dict) -> dict:
    subs = {rid: o["submit_timestamp"] for rid, o in replay.items()}
    t0 = min(subs.values())
    dev = []
    for rid, p in plan.items():
        if rid in subs:
            dev.append((subs[rid] - t0) * 1000.0 - p["arrival_ms"])
    dev = np.array(dev)
    waves = collections.defaultdict(list)
    for rid, p in plan.items():
        if rid in replay:
            o = replay[rid]
            waves[(p["b"], p["kv"], p["w"])].append(o)
    wave_rows = []
    for (b, kv, w), lst in sorted(waves.items()):
        st = np.array([o["submit_timestamp"] for o in lst]); ct = np.array([o["complete_timestamp"] for o in lst])
        wave_rows.append({"b": b, "kv_k": kv, "wave": w, "requests": len(lst), "planned": b, "submit_spread_ms": float((st.max() - st.min()) * 1000),
                          "complete_spread_ms": float((ct.max() - ct.min()) * 1000), "wall_s": float(ct.max() - st.min()),
                          "statuses": sorted({o["status"] for o in lst}), "finish_reasons": sorted({o["finish_reason"] for o in lst}),
                          "output_len_actual": sorted({o["output_len_actual"] for o in lst}),
                          "cached_prompt_tokens_max": max(o["server_usage"]["cached_prompt_tokens"] for o in lst)})
    # overlap between consecutive waves in time order
    ordered = sorted(wave_rows, key=lambda r: min(o["submit_timestamp"] for o in waves[(r["b"], r["kv_k"], r["wave"])]))
    overlaps = 0
    for a, b_ in zip(ordered, ordered[1:]):
        end_a = max(o["complete_timestamp"] for o in waves[(a["b"], a["kv_k"], a["wave"])])
        start_b = min(o["submit_timestamp"] for o in waves[(b_["b"], b_["kv_k"], b_["wave"])])
        if start_b < end_a:
            overlaps += 1
    return {"requests_planned": len(plan), "requests_replayed": len(replay), "arrival_deviation_ms": summary(dev),
            "waves": wave_rows, "consecutive_wave_time_overlaps": overlaps,
            "all_success": all(o["status"] == "SUCCESS" for o in replay.values()),
            "all_length_finish": all(o["finish_reason"] == "length" for o in replay.values()),
            "all_1024_output": all(o["output_len_actual"] == OUTPUT_TOKENS for o in replay.values()),
            "prefix_cache_hits": sum(o["server_usage"]["cached_prompt_tokens"] for o in replay.values())}


def tactic_agreement(caches: dict[str, Path]) -> dict:
    def load_cache(path):
        d = json.load(open(path))
        d.pop("_metadata", None)
        out = {}
        for k, v in d.items():
            try:
                key = ast.literal_eval(k)
            except Exception:
                key = k
            out[str(key)] = json.dumps(v, sort_keys=True)
        return out
    loaded = {k: load_cache(p) for k, p in caches.items()}
    rows = []
    labels = list(loaded)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            common = set(loaded[a]) & set(loaded[b])
            diff = sum(1 for k in common if loaded[a][k] != loaded[b][k])
            rows.append({"a": a, "b": b, "common": len(common), "differing": diff, "fraction": diff / len(common) if common else None,
                         "only_a": len(set(loaded[a]) - common), "only_b": len(set(loaded[b]) - common)})
    return {"caches": {k: str(p) for k, p in caches.items()}, "pairs": rows}


NOT_RECONSTRUCTABLE = [
    "per-rank kernel timings and the pure-decode iteration decomposition into kernel families (needs CUPTI kernel rows)",
    "f_ind (fraction of iteration time that is rank-independent) by b / KV / phase, and its frozen + block-bootstrap CIs",
    "four-rank NVTX / collective alignment, collective end spread, modal-K rank alignment per pure-decode cell",
    "critical-path ranks, wait vs intrinsic decomposition, straggler attribution",
    "autotune tactic-dependent GPU time per kernel family (tactic *choices* survive in the autotune cache; their GPU time does not)",
    "the K=163 decomposition proxy and every pre-registered contrast that is defined on kernel-level quantities",
    "JIT-affected iteration detection from kernel names (only host-side elapsed outliers can hint at it)",
    "routing-vs-timing correlations at kernel granularity (routing survives; per-iteration GPU time does not — only host-observed elapsed)",
    "the Stage 1.5 comparisons that require identical instrumentation (host-side elapsed here includes nsys capture overhead of unknown size)",
]


def write_markdown(out_md: Path, R: dict) -> None:
    L = []
    L.append("# e1_main_v3 — surviving-artifact analysis (host-side only; no Nsight trace exists)\n")
    L.append(f"Archive: `{R['archive']}` (frozen, SHA256SUMS verified). Measured window (CLOCK_MONOTONIC ns): {R['window']}.\n")
    L.append("All latency numbers below are **host-observed under an active Nsight capture and with routing return enabled**; they are routing-instrumented, "
             "profiler-perturbed host timings, not clean GPU timings, and must not be treated as validation data for any model.\n")
    L.append("## 1. Completeness by cell and wave\n")
    L.append("| b | KV | wave | planned | replayed | routing files | request timings | pure-decode iterations (host) | contiguous | ≥ 400 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    rt = {(c["b"], c["kv_k"], c["wave"]): c for c in R["request_timings"]["cells"]}
    it = {(c["b"], c["kv_k"], c.get("wave_observed")): c for c in R["iterations"]["cells"] if "pure_decode_iterations" in c}
    for w in R["plan_deviation"]["waves"]:
        k = (w["b"], w["kv_k"], w["wave"])
        rf = R["routing"]["requests_per_cell"].get(f"b{w['b']}_kv{w['kv_k']}k_w{w['wave']}", 0)
        ic = it.get(k, {})
        L.append(f"| {w['b']} | {w['kv_k']}k | {w['wave']} | {w['planned']} | {w['requests']} | {rf} | {rt.get(k, {}).get('requests', 0)} | "
                 f"{ic.get('pure_decode_iterations', '—')} | {ic.get('iterations_contiguous', '—')} | {ic.get('clean_ge_400', '—')} |")
    L.append("")
    L.append("## 2. Request / token / iteration reconciliation\n")
    I = R["iterations"]; P = R["plan_deviation"]; T = R["request_timings"]
    L.append(f"* requests: planned {P['requests_planned']}, replayed {P['requests_replayed']}, all SUCCESS {P['all_success']}, all finish=length {P['all_length_finish']}, "
             f"all 1024 output tokens {P['all_1024_output']}, prefix-cache hits {P['prefix_cache_hits']}.")
    L.append(f"* routing files: {R['routing']['files']} ({R['routing']['shape_dtype_range_ok']} with shape (1024, 40, 6) uint16 and ids < 384; bad {len(R['routing']['bad_files'])}).")
    L.append(f"* request timing records in the measured window: {T['records']} (missing {len(T['missing_request_timings'])}, unexpected {len(T['unexpected_request_ids'])}).")
    L.append(f"* iteration records: {I['records_total']} in the server log, {I['records_in_measured_window']} in the measured window "
             f"(indices {I['window_first_index']}–{I['window_last_index']}, {I['iteration_index_gap_count']} index gaps overall); "
             f"pure-decode {I['pure_decode']}, prefill-only {I['prefill_only']}, mixed {I['mixed']}, empty {I['idle_or_empty']}; "
             f"pure-decode iterations not attributable to a plan cell: {I['pure_decode_unassigned_to_a_plan_cell']}.")
    L.append(f"* decode tokens scheduled in the window: {I['decode_tokens_scheduled_in_window']} vs expected 450 × 1023 = {I['expected_decode_tokens']}; "
             f"prefill tokens {I['prefill_tokens_in_window']} vs planned {I['expected_prefill_tokens']}; window wall {I['window_wall_s']:.1f} s.")
    L.append(f"* pure-decode iterations by batch size: {I['pure_decode_iterations_by_batch']}.\n")
    L.append("## 3. Routing statistics (per-token top-6 of 384 routed experts, 40 layers)\n")
    Rg = R["routing"]
    L.append(f"* pooled over all 460 800 tokens: mean per-layer entropy {Rg['pooled_mean_layer_entropy_bits']:.3f} bits (uniform = {math.log2(NUM_EXPERTS):.3f}), "
             f"mean max/mean expert load {Rg['pooled_mean_max_over_mean_load']:.2f}, (layer, expert) pairs never routed {Rg['experts_never_routed_any_layer']}; "
             f"consecutive-token expert overlap (sampled) p50 {Rg['consecutive_token_expert_overlap_fraction'].get('p50', float('nan')):.3f}.")
    L.append("\n| b | KV | wave | tokens | mean layer entropy (bits) [95 % CI, request bootstrap] | mean max/mean load [95 % CI] | mean JSD vs pooled (bits) |")
    L.append("|---|---|---|---|---|---|---|")
    for c in Rg["per_cell"]:
        L.append(f"| {c['b']} | {c['kv_k']}k | {c['wave']} | {c['tokens']} | {c['mean_layer_entropy_bits']:.3f} [{c['entropy_ci95'][0]:.3f}, {c['entropy_ci95'][1]:.3f}] | "
                 f"{c['mean_max_over_mean_load']:.2f} [{c['max_load_ci95'][0]:.2f}, {c['max_load_ci95'][1]:.2f}] | {c['mean_layer_jsd_vs_pooled_bits']:.4f} |")
    L.append("\nWave repeatability (w0 vs w1 of the same cell; different prompts): mean per-layer JSD against a permutation null of the same split sizes.\n")
    L.append("| b | KV | JSD w0 vs w1 (bits) | null p50 | null p97.5 |")
    L.append("|---|---|---|---|---|")
    for r in Rg["wave_repeatability"]:
        L.append(f"| {r['b']} | {r['kv_k']}k | {r['jsd_w0_w1_bits']:.4f} | {r['null_p50']:.4f} | {r['null_p97_5']:.4f} |")
    L.append("\nUncertainty: the request is the independent unit (prompts are distinct enwik9 slices); CIs are percentile bootstraps over requests "
             "(B = 400), so cells with 1–2 requests have no meaningful interval. Marginals are pooled over 1024 tokens per request, so token-level "
             "autocorrelation within a request is inside the unit, not ignored. Resampling with replacement duplicates requests, which biases the "
             "pooled entropy down and the max-load ratio up, so the point estimate can sit at an interval edge: read the intervals as dispersion "
             "across requests, not as coverage intervals. For b = 1 and b = 2 the permutation null below has one or two distinct splits.\n")
    L.append("## 4. Host-side timing distributions (profiler-perturbed; routing-instrumented)\n")
    L.append("| b | KV | wave | pure-decode iters | elapsed ms p50 | p10 | p90 | mean | sd | first-quarter p50 | last-quarter p50 | wall s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in R["iterations"]["cells"]:
        if "pure_decode_iterations" not in c:
            continue
        e = c["elapsed_ms"]
        L.append(f"| {c['b']} | {c['kv_k']}k | {c['wave_observed']} | {c['pure_decode_iterations']} | {e['p50']:.2f} | {e['p10']:.2f} | {e['p90']:.2f} | {e['mean']:.2f} | {e['sd']:.2f} | "
                 f"{c['elapsed_ms_first_quarter_p50']:.2f} | {c['elapsed_ms_last_quarter_p50']:.2f} | {c['wall_s']:.1f} |")
    L.append("\nEngineCore request timings (TTFT = first schedule to first token; TPOT over 1023 decode steps):\n")
    L.append("| b | KV | wave | n | TTFT ms p50 | TTFT p90 | TPOT ms p50 | TPOT p90 | client total ms p50 | client − engine ms p50 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in R["request_timings"]["cells"]:
        L.append(f"| {c['b']} | {c['kv_k']}k | {c['wave']} | {c['requests']} | {c['engine_core_ttft_ms']['p50']:.1f} | {c['engine_core_ttft_ms']['p90']:.1f} | "
                 f"{c['engine_core_tpot_ms']['p50']:.3f} | {c['engine_core_tpot_ms']['p90']:.3f} | {c['client_total_duration_ms']['p50']:.1f} | "
                 f"{c['client_minus_engine_ms_p50'] if c['client_minus_engine_ms_p50'] is None else round(c['client_minus_engine_ms_p50'], 1)} |")
    L.append("\n## 5. Differences from the frozen workload plan\n")
    d = P["arrival_deviation_ms"]
    L.append(f"* arrival deviation (actual submit offset − planned arrival): p50 {d['p50']:.1f} ms, p90 {d['p90']:.1f} ms, max {d['max']:.1f} ms over {d['n']} requests.")
    L.append(f"* consecutive waves overlapping in time: {P['consecutive_wave_time_overlaps']} (plan: 0).")
    odd = [c for c in R['iterations']['cells'] if 'note' in c]
    L.append(f"* cells whose observed pure-decode wave blocks differ from the plan: {len(odd)} {[(c['b'], c['kv_k'], c['note']) for c in odd]}.")
    L.append(f"* waves with submit spread > 1 s: {[(w['b'], w['kv_k'], w['wave'], round(w['submit_spread_ms'])) for w in P['waves'] if w['submit_spread_ms'] > 1000]}.\n")
    L.append("## 6. Autotune tactic cache of this launch vs earlier launches\n")
    L.append("| A | B | common keys | differing | fraction |")
    L.append("|---|---|---|---|---|")
    for r in R["tactics"]["pairs"]:
        L.append(f"| {r['a']} | {r['b']} | {r['common']} | {r['differing']} | {r['fraction']:.3f} |")
    L.append("\n## 7. Measurements that cannot be reconstructed without the Nsight trace\n")
    for s in NOT_RECONSTRUCTABLE:
        L.append(f"* {s}")
    L.append("\nNo coefficient was fitted and no identification partition was written. Machine-readable results: `surviving.json`.\n")
    out_md.write_text("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--scratch", required=True, type=Path)
    ap.add_argument("--caches", nargs="*", default=[], help="label=path of autotune_configs json files to compare")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[7]))  # repo root → alignment package
    from alignment.profiler.record_extraction import extract_metrics_jsonl, extract_request_timings_jsonl

    A, OUT, SCR = a.archive, a.out, a.scratch
    OUT.mkdir(parents=True, exist_ok=True); SCR.mkdir(parents=True, exist_ok=True)
    plan = load_plan(A / "trace.csv")
    replay = load_replay(A / "profile/load_generator/replay.jsonl")
    drive = json.load(open(next((A / "profile/vllm").glob("*_drive_summary.json"))))
    window = (drive["replay_start_monotonic_ns"], drive["replay_end_monotonic_ns"])
    server_log = next((A / "profile/vllm").glob("*_server.log"))
    mlog = SCR / "measurement.log"
    slice_measurement_log(server_log, int(drive["measurement_log_offset"]), mlog)
    n_metrics = extract_metrics_jsonl(mlog, SCR / "metrics_window.jsonl")
    n_timings = extract_request_timings_jsonl(mlog, SCR / "timings_window.jsonl",
                                              expected_request_ids={f"independent_{rid}" for rid in plan})
    R = {"archive": str(A), "window": window, "measurement_log_offset": drive["measurement_log_offset"],
         "extracted": {"metrics_rows_after_offset": n_metrics, "request_timing_rows": n_timings},
         "routing": routing_analysis(A / "profile/routed_experts", plan),
         "iterations": iteration_analysis(SCR / "metrics_window.jsonl", window, plan),
         "request_timings": request_timing_analysis(SCR / "timings_window.jsonl", replay, plan),
         "plan_deviation": plan_deviation(plan, replay),
         "tactics": tactic_agreement({kv.split("=", 1)[0]: Path(kv.split("=", 1)[1]) for kv in a.caches}),
         "not_reconstructable_without_nsight": NOT_RECONSTRUCTABLE}
    (OUT / "surviving.json").write_text(json.dumps(R, indent=1, default=float))
    write_markdown(OUT / "E1-MAIN-V3-SURVIVING-ARTIFACTS.md", R)
    print(f"wrote {OUT}/surviving.json and E1-MAIN-V3-SURVIVING-ARTIFACTS.md; metrics {n_metrics}, timings {n_timings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
