#!/usr/bin/env python3
"""Kernel rows per iteration in the preserved V2 and V3 captures, per wave, per batch size and per rank (CPU-only; reads the
extract tables written by the original post-runs; counts only: no timing model, no rank-dependence parameter, no partition).

    <python with pyarrow> feasibility/e1_rows_per_iteration.py --tables <phase1a-tables dir> --out feasibility
        [--main-inputs E1-MAIN-V3-INPUTS.json]

Inputs: <tables>/<pass>/iterations.parquet (one row per engine iteration: stage, decode_requests, n_prefill_chunks,
n_kernels, t0_ns, t1_ns) and iter_rank.parquet (iteration, rank, n_kernels) for e1_v2_forced_release and e1_v3_scale.
A wave is a run of iterations separated from the next by more than 2 s. Writes E1-ROWS-PER-ITERATION.json and .md.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq

PASSES = ("e1_v2_forced_release", "e1_v3_scale")
TOTAL = {"e1_v2_forced_release": 12_250_832, "e1_v3_scale": 24_501_664}  # kernel rows of the recovered traces (gate records)


def waves_of(tables: Path, name: str) -> list[dict]:
    it = pq.read_table(tables / name / "iterations.parquet").to_pylist()
    ir = pq.read_table(tables / name / "iter_rank.parquet", columns=["iteration", "rank", "n_kernels"]).to_pylist()
    by_rank = collections.defaultdict(dict)
    for r in ir:
        by_rank[r["iteration"]][r["rank"]] = r["n_kernels"]
    it.sort(key=lambda r: r["t0_ns"])
    groups = [[it[0]]]
    for a, b in zip(it, it[1:]):
        if b["t0_ns"] - a["t1_ns"] > 2e9:
            groups.append([])
        groups[-1].append(b)
    out = []
    for g in groups:
        b = max(r["decode_requests"] for r in g)
        pure = [r for r in g if r["stage"] == "decode" and r["decode_requests"] == b and r["n_prefill_chunks"] == 0]
        partial = [r for r in g if r["stage"] == "decode" and r["decode_requests"] < b]
        prefill = [r for r in g if r["stage"] != "decode"]
        ranks = sorted({k for r in g for k in by_rank[r["iteration"]]})
        per_rank = {k: sum(by_rank[r["iteration"]].get(k, 0) for r in g) for k in ranks}
        pure_rank = {k: collections.Counter(by_rank[r["iteration"]].get(k) for r in pure) for k in ranks}
        seq = hashlib.sha256(json.dumps([[r["stage"], r["decode_requests"], r["n_prefill_chunks"], r["n_kernels"], [by_rank[r["iteration"]].get(k) for k in ranks]] for r in g]).encode()).hexdigest()
        out.append({
            "pass": name, "batch": b, "first_t0_s": round(g[0]["t0_ns"] / 1e9, 2), "duration_s": round((g[-1]["t1_ns"] - g[0]["t0_ns"]) / 1e9, 2), "iterations": len(g), "kernel_rows": sum(r["n_kernels"] for r in g),
            "rows_per_iteration_all": round(sum(r["n_kernels"] for r in g) / len(g), 2),
            "pure_decode": {"iterations": len(pure), "rows_per_iteration_mean": round(statistics.fmean(r["n_kernels"] for r in pure), 2), "min": min(r["n_kernels"] for r in pure), "max": max(r["n_kernels"] for r in pure),
                            "distinct_values": dict(sorted(collections.Counter(r["n_kernels"] for r in pure).items()))},
            "partial_batch_decode": {"iterations": len(partial), "rows_per_iteration_mean": round(statistics.fmean(r["n_kernels"] for r in partial), 2) if partial else None},
            "prefill_or_mixed": {"iterations": len(prefill), "rows_per_iteration_mean": round(statistics.fmean(r["n_kernels"] for r in prefill), 2) if prefill else None, "min": min((r["n_kernels"] for r in prefill), default=None), "max": max((r["n_kernels"] for r in prefill), default=None)},
            "rows_per_rank": per_rank, "pure_decode_rows_per_iteration_per_rank": {k: dict(sorted(v.items())) for k, v in pure_rank.items()}, "iteration_sequence_sha256": seq,
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--main-inputs", type=Path)
    a = ap.parse_args(argv)
    waves = [w for name in PASSES for w in waves_of(a.tables, name)]
    totals = {name: sum(w["kernel_rows"] for w in waves if w["pass"] == name) for name in PASSES}
    by_b = collections.defaultdict(list)
    for w in waves:
        by_b[w["batch"]].append(w)
    checks = {
        "sum_of_waves_equals_recovered_trace_rows": {n: totals[n] == TOTAL[n] for n in PASSES},
        "waves_of_a_cell_identical_in_rows_iterations_and_sequence": {f"b{b}": len({(w["kernel_rows"], w["iterations"], w["iteration_sequence_sha256"]) for w in ws}) == 1 for b, ws in sorted(by_b.items())},
        "waves_per_cell": {f"b{b}": [f"{w['pass']} @ {w['first_t0_s']} s" for w in ws] for b, ws in sorted(by_b.items())},
    }
    full = None
    if a.main_inputs:
        d = json.loads(a.main_inputs.read_text())
        dec = sorted(w["pure_decode"]["rows_per_iteration_mean"] for w in waves)
        pre = [w["prefill_or_mixed"] for w in waves]
        lo_d, hi_d, lo_p, hi_p = dec[0], dec[-1], min(p["min"] for p in pre), max(p["max"] for p in pre)
        chunks = {4096: 1, 32768: 2, 98304: 6}
        rows = []
        for p in d["plan"]:
            decode_iters, prefill_iters = d["design"]["OUT_LEN"], p["b"] * chunks[p["kv"]]
            rows.append({"b": p["b"], "kv": p["kv"], "wave": p["wave"], "iterations": decode_iters + prefill_iters, "plan_prefill_iters": p.get("prefill_iters"), "rows_low": decode_iters * lo_d + prefill_iters * lo_p, "rows_high": decode_iters * hi_d + prefill_iters * hi_p})
        measured_like = {b: ws[0] for b, ws in by_b.items()}
        check_b = {f"b{b}": {"formula_iterations": d["design"]["OUT_LEN"] + b * chunks[32768], "measured_iterations": w["iterations"]} for b, w in sorted(measured_like.items())}
        full = {"basis": "ASSUMPTION: every cell's pure-decode iterations have between the lowest and the highest measured rows per iteration (b=48 and b=8 at 32k), and its prefill-carrying iterations between the lowest and highest measured for such iterations; iterations per wave = OUT_LEN + b x prefill chunks (1, 2, 6 for 4k, 32k, 96k), which reproduces the two measured cells",
                "iteration_formula_check_on_measured_cells": check_b, "decode_rows_per_iteration_bracket": [lo_d, hi_d], "prefill_rows_per_iteration_bracket": [lo_p, hi_p], "waves": len(rows), "iterations": sum(r["iterations"] for r in rows),
                "kernel_rows_low": round(sum(r["rows_low"] for r in rows)), "kernel_rows_high": round(sum(r["rows_high"] for r in rows)), "earlier_estimate": 216_000_000,
                "by_kv": {str(kv): {"iterations": sum(r["iterations"] for r in rows if r["kv"] == kv), "rows_low": round(sum(r["rows_low"] for r in rows if r["kv"] == kv)), "rows_high": round(sum(r["rows_high"] for r in rows if r["kv"] == kv))} for kv in chunks}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "E1-ROWS-PER-ITERATION.json").write_text(json.dumps({"waves": waves, "totals": totals, "checks": checks, "full_e1_consistency": full}, indent=1) + "\n")
    md = ["| pass | wave start | batch | iterations | kernel rows | rows / iteration (all) | pure decode: iterations, mean, min – max | partial-batch decode: n, mean | prefill or mixed: n, mean, min – max | rows per rank 0 – 3 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for w in waves:
        pd, pb, pf = w["pure_decode"], w["partial_batch_decode"], w["prefill_or_mixed"]
        md.append(f"| {w['pass']} | {w['first_t0_s']} s | {w['batch']} | {w['iterations']} | {w['kernel_rows']} | {w['rows_per_iteration_all']} | {pd['iterations']}, {pd['rows_per_iteration_mean']}, {pd['min']} – {pd['max']} | {pb['iterations']}, {pb['rows_per_iteration_mean']} | {pf['iterations']}, {pf['rows_per_iteration_mean']}, {pf['min']} – {pf['max']} | {', '.join(str(v) for _, v in sorted(w['rows_per_rank'].items()))} |")
    md += ["", "Pure-decode rows per iteration, by rank (value: number of iterations):", ""]
    for b, ws in sorted(by_b.items()):
        md.append(f"- b={b} ({len(ws)} waves, identical): " + "; ".join(f"rank {k}: {v}" for k, v in sorted(ws[0]["pure_decode_rows_per_iteration_per_rank"].items())))
    md += ["", f"Checks: {json.dumps(checks)}"]
    if full:
        md += ["", f"Full E1 consistency (assumption-based bracket, not a measurement): {json.dumps(full)}"]
    (a.out / "E1-ROWS-PER-ITERATION.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
