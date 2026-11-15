#!/usr/bin/env python3
"""Per-rank GPU execution, collective-wait and kernel-identity report for one nsys pass (CPU, read-only; numpy + pyarrow).

    $S/gb300/venv/bin/python analysis/e1_nsys_rank_kernels.py --profile <profile dir> --tables <Phase 1A tables dir>
        [--cells 8,48] [--ref-profile <dir> --ref-tables <dir> --ref-label quadrant_nsys] [--out report.md] [--json out.json]

Observed measurements (sections A-D) come from the pass itself: Phase 1A `iterations` / `iter_rank` / `collectives` tables
(costmodel.extract) plus `parsed.json` + `parsed.kernels.parquet` for kernel names. Pure-decode iterations of each cell
(decode_requests == b, no prefill, not JIT-excluded) are used.
  A. per-rank GPU execution: span, busy, compute/comm union, idle, launch and sync API time, host->GPU tail, kernels/iteration
  B. rank alignment: spread across the four ranks of the first kernel start and the last kernel end, per iteration
  C. collective wait: per collective k, each rank's comm time minus the fastest rank's at the same (iteration, k) — the time a
     rank spends inside the collective waiting for its peers — summed per iteration; plus the rank spread of collective ends
  D. kernel identities and launch characteristics: per cell, kernels ranked by GPU time per iteration (count/iteration,
     mean duration, streams), with the MoE / GEMM (FlashInfer-tuned) kernels marked
Section E (optional, --ref-*) compares the kernel set and per-rank busy time against another nsys pass of the same recipe
(e.g. Phase 0 quadrant_nsys: hook OFF, vllm_tokens backend, 2026-11-12). That is a cross-session comparison, not a hook
measurement: differences there are hypotheses about the hook at best, and the report says so.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

MOE_GEMM_TOKENS = ("moe", "gemm", "cutlass", "mxfp", "trtllm", "fused_moe", "grouped", "bmm", "cublas", "nvjet", "sm100", "sm103")
PCTS = (50, 90)


def q(a, p):
    a = np.asarray(a, dtype=np.float64)
    return float(np.percentile(a, p)) if a.size else float("nan")


def ms(ns):
    return ns / 1e6


def cols(path: str) -> dict:
    t = pq.read_table(path)
    return {n: t.column(n).to_numpy(zero_copy_only=False) for n in t.column_names}


def load_tables(tables: str):
    return (cols(os.path.join(tables, "iterations.parquet")), cols(os.path.join(tables, "iter_rank.parquet")),
            cols(os.path.join(tables, "collectives.parquet")))


def pure_decode_iterations(it: dict, b: int) -> np.ndarray:
    m = (it["stage"] == "decode") & (it["prefill_tokens"] == 0) & (it["decode_requests"] == b) & (~it["excluded"].astype(bool))
    return it["iteration"][m]


def pct_dict(a, scale_ms=True, nd=4):
    return {f"p{p}": round(ms(q(a, p)) if scale_ms else q(a, p), nd) for p in PCTS}


def section_rank_exec(ir: dict, its: np.ndarray) -> dict:
    m = np.isin(ir["iteration"], its)
    out = {}
    tcols = ["span_ns", "busy_ns", "compute_union_ns", "comm_union_ns", "idle_ns", "comp_in_comm_sum_ns", "seg_idle_sum_ns",
             "api_launch_sum_ns", "api_sync_sum_ns", "api_sync_max_ns", "gpu_tail_after_host_ns", "fwd_window_ns"]
    ccols = ["n_kernels", "n_streams", "api_launch_n", "api_sync_n", "n_comm_episodes"]
    for r in np.unique(ir["rank"][m]):
        mr = m & (ir["rank"] == r)
        d = {"n_iterations": int(mr.sum())}
        for c in tcols:
            if c in ir:
                d[c.replace("_ns", "_ms")] = pct_dict(ir[c][mr])
        for c in ccols:
            if c in ir:
                d[c] = pct_dict(ir[c][mr], scale_ms=False, nd=1)
        out[int(r)] = d
    return out


def rank_matrix(ir: dict, its: np.ndarray, col: str):
    """(iterations, 4) matrix of `col` for the iterations in `its` that have all four ranks."""
    m = np.isin(ir["iteration"], its)
    itv, rk, v = ir["iteration"][m], ir["rank"][m].astype(np.int64), ir[col][m].astype(np.float64)
    u, inv = np.unique(itv, return_inverse=True)
    M = np.full((len(u), 4), np.nan)
    M[inv, rk] = v
    full = ~np.isnan(M).any(axis=1)
    return u[full], M[full]


def section_alignment(ir: dict, its: np.ndarray) -> dict:
    u, S = rank_matrix(ir, its, "first_start_ns")
    _, E = rank_matrix(ir, its, "last_end_ns")
    ss = S.max(axis=1) - S.min(axis=1)
    es = E.max(axis=1) - E.min(axis=1)
    overlap = int((S.max(axis=1) < E.min(axis=1)).sum())
    return {
        "n_iterations_with_4_ranks": int(len(u)),
        "first_start_spread_ms": {"p50": round(ms(q(ss, 50)), 4), "p90": round(ms(q(ss, 90)), 4), "max": round(ms(ss.max()) if ss.size else float("nan"), 4)},
        "last_end_spread_ms": {"p50": round(ms(q(es, 50)), 4), "p90": round(ms(q(es, 90)), 4), "max": round(ms(es.max()) if es.size else float("nan"), 4)},
        "gpu_windows_overlap_all_ranks": overlap,
        "earliest_rank_first_start_counts": {int(r): int(c) for r, c in zip(*np.unique(S.argmin(axis=1), return_counts=True))},
        "latest_rank_last_end_counts": {int(r): int(c) for r, c in zip(*np.unique(E.argmax(axis=1), return_counts=True))},
    }


def section_collective_wait(coll: dict, its: np.ndarray) -> dict:
    m = np.isin(coll["iteration"], its)
    if not m.any():
        return {"note": "no collectives rows for these iterations"}
    itv = coll["iteration"][m].astype(np.int64); k = coll["k"][m].astype(np.int64); rk = coll["rank"][m].astype(np.int64)
    kind = coll["kind"][m].astype(np.int64); comm = coll["comm_sum_ns"][m].astype(np.float64); end = coll["end_rel_ns"][m].astype(np.float64)
    key = itv * 100000 + k
    g, inv = np.unique(key, return_inverse=True)
    n_r = np.bincount(inv, minlength=len(g))
    min_comm = np.full(len(g), np.inf); np.minimum.at(min_comm, inv, comm)
    max_end = np.full(len(g), -np.inf); np.maximum.at(max_end, inv, end)
    min_end = np.full(len(g), np.inf); np.minimum.at(min_end, inv, end)
    ok = n_r[inv] == 4
    wait = comm - min_comm[inv]
    # per (iteration, rank) sums
    u_it, it_pos = np.unique(itv[ok], return_inverse=True)
    key2 = it_pos * 4 + rk[ok]
    wait_sum = np.bincount(key2, weights=wait[ok], minlength=len(u_it) * 4).reshape(len(u_it), 4)
    comm_sum = np.bincount(key2, weights=comm[ok], minlength=len(u_it) * 4).reshape(len(u_it), 4)
    n_coll = np.bincount(key2, minlength=len(u_it) * 4).reshape(len(u_it), 4)
    out = {"n_iterations": int(len(u_it)), "collectives_per_iteration": pct_dict(n_coll[:, 0], scale_ms=False, nd=1), "per_rank": {}, "per_kind": {}}
    for r in range(4):
        out["per_rank"][r] = {"wait_sum_ms": pct_dict(wait_sum[:, r]), "comm_sum_ms": pct_dict(comm_sum[:, r]),
                              "wait_share_of_comm_p50": round(float(np.median(wait_sum[:, r] / np.maximum(comm_sum[:, r], 1))), 4)}
    es = (max_end - min_end)[n_r == 4]
    out["end_spread_ms"] = {"p50": round(ms(q(es, 50)), 5), "p90": round(ms(q(es, 90)), 5), "max": round(ms(es.max()), 4)}
    for kd in np.unique(kind[ok]):
        mk = ok & (kind == kd)
        out["per_kind"][int(kd)] = {"n": int(mk.sum()), "wait_ms_p50": round(ms(q(wait[mk], 50)), 4), "wait_ms_p90": round(ms(q(wait[mk], 90)), 4),
                                    "comm_ms_p50": round(ms(q(comm[mk], 50)), 4)}
    med = np.median(wait_sum, axis=0)
    out["least_waiting_rank_(straggler)"] = int(med.argmin()); out["most_waiting_rank"] = int(med.argmax())
    return out


class Kernels:
    """parsed.json + parsed.kernels.parquet -> per-iteration, per-device kernel rows with names."""

    def __init__(self, profile: str):
        d = json.load(open(os.path.join(profile, "parsed.json")))
        det = d["iteration_details"]
        self.names = {int(k): v for k, v in d["kernel_names"].items()}
        it_of_det = np.array([int(x["iteration"]) for x in det], dtype=np.int64)
        maxr = max(len(x["ranges"]) for x in det)
        rd = np.full(len(det) * maxr, -1, dtype=np.int8)
        for ai, x in enumerate(det):
            for bi, r in enumerate(x["ranges"]):
                rd[ai * maxr + bi] = r["device_id"]
        del d, det
        tab = pq.read_table(os.path.join(profile, "parsed.kernels.parquet"), columns=["detail_index", "range_index", "name_id", "start_ns", "end_ns", "stream_id"])
        di = tab["detail_index"].to_numpy().astype(np.int64)
        ri = tab["range_index"].to_numpy().astype(np.int64)
        self.nid = tab["name_id"].to_numpy().astype(np.int64)
        self.dur = tab["end_ns"].to_numpy().astype(np.int64) - tab["start_ns"].to_numpy().astype(np.int64)
        self.sid = tab["stream_id"].to_numpy().astype(np.int64)
        self.dev = rd[di * maxr + ri].astype(np.int64)
        self.it = it_of_det[di]
        del tab

    def per_cell(self, its: np.ndarray, top: int = 30) -> dict:
        m = np.isin(self.it, its)
        n_it = len(np.unique(self.it[m]))
        res = {"n_iterations": int(n_it), "per_rank_kernels_per_iteration": {}, "top_kernels": []}
        for r in np.unique(self.dev[m]):
            res["per_rank_kernels_per_iteration"][int(r)] = round(float((m & (self.dev == r)).sum()) / max(n_it, 1), 2)
        nid, dur, sid = self.nid[m], self.dur[m], self.sid[m]
        L = max(self.names) + 1
        agg_cnt = np.bincount(nid, minlength=L)
        agg_dur = np.bincount(nid, weights=dur, minlength=L)
        present = np.nonzero(agg_cnt)[0]
        res["n_distinct_names"] = int(len(present))
        order = present[np.argsort(-agg_dur[present])]
        total = float(agg_dur[present].sum())
        for kk in order[:top]:
            streams = np.unique(sid[nid == kk])
            name = self.names.get(int(kk), f"<{kk}>")
            res["top_kernels"].append({
                "name": name[:120], "moe_gemm_family": any(t in name.lower() for t in MOE_GEMM_TOKENS),
                "count_per_iteration_all_ranks": round(float(agg_cnt[kk]) / max(n_it, 1), 2),
                "mean_us": round(float(agg_dur[kk] / agg_cnt[kk]) / 1e3, 2), "share_of_gpu_time": round(float(agg_dur[kk]) / max(total, 1), 4),
                "n_streams": int(len(streams)), "sum_ms_per_iteration_all_ranks": round(float(agg_dur[kk]) / max(n_it, 1) / 1e6, 4),
            })
        res["name_set"] = sorted(self.names[int(kk)] for kk in present)
        res["name_time"] = {self.names[int(kk)]: [float(agg_cnt[kk]) / max(n_it, 1), float(agg_dur[kk]) / max(n_it, 1) / 1e6] for kk in present}
        return res


def md_table(rows, cols_):
    out = ["| " + " | ".join(cols_) + " |", "|" + "---|" * len(cols_)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols_) + " |")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True); ap.add_argument("--tables", required=True)
    ap.add_argument("--label", default="e1_nsys_val"); ap.add_argument("--cells", default="8,48")
    ap.add_argument("--ref-profile"); ap.add_argument("--ref-tables"); ap.add_argument("--ref-label", default="reference")
    ap.add_argument("--top", type=int, default=25); ap.add_argument("--out"); ap.add_argument("--json")
    a = ap.parse_args(argv)
    cells = [int(x) for x in a.cells.split(",")]
    it, ir, coll = load_tables(a.tables)
    K = Kernels(a.profile)
    report = {"label": a.label, "profile": a.profile, "tables": a.tables, "cells": {}}
    lines = [f"# Per-rank execution, collective wait and kernel identities: `{a.label}`", "",
             f"profile `{a.profile}`, tables `{a.tables}`. Pure-decode iterations only (decode_requests == b, no prefill, not JIT-excluded).",
             "Sections A-D are observed on this pass alone. They say what the four ranks did; they do not by themselves",
             "measure the routing hook's effect, which needs a hook-off pass under identical conditions.", ""]
    ref = None
    if a.ref_profile and a.ref_tables:
        rit, rir, _ = load_tables(a.ref_tables)
        ref = (rit, rir, Kernels(a.ref_profile))
    for b in cells:
        its = pure_decode_iterations(it, b)
        c = {"n_pure_decode_iterations": int(len(its))}
        if len(its) == 0:
            report["cells"][b] = c; lines += [f"## cell b={b}: no pure-decode iterations", ""]; continue
        c["A_rank_exec"] = section_rank_exec(ir, its)
        c["B_alignment"] = section_alignment(ir, its)
        c["C_collective_wait"] = section_collective_wait(coll, its)
        c["D_kernels"] = K.per_cell(its, a.top)
        report["cells"][b] = c
        lines += [f"## cell b={b} ({len(its)} pure-decode iterations)", "", "### A. per-rank GPU execution (ms per iteration, p50 / p90)", ""]
        rows = []
        for r, d in c["A_rank_exec"].items():
            g = lambda k: d.get(k, {})
            rows.append({"rank": r, "span": f"{g('span_ms').get('p50')} / {g('span_ms').get('p90')}", "busy": f"{g('busy_ms').get('p50')} / {g('busy_ms').get('p90')}",
                         "compute_union": f"{g('compute_union_ms').get('p50')} / {g('compute_union_ms').get('p90')}", "comm_union": f"{g('comm_union_ms').get('p50')} / {g('comm_union_ms').get('p90')}",
                         "idle": f"{g('idle_ms').get('p50')} / {g('idle_ms').get('p90')}", "api_launch_sum": g("api_launch_sum_ms").get("p50"), "api_sync_sum": g("api_sync_sum_ms").get("p50"),
                         "gpu_tail_after_host": g("gpu_tail_after_host_ms").get("p50"), "kernels": g("n_kernels").get("p50"), "launches": g("api_launch_n").get("p50")})
        lines += md_table(rows, ["rank", "span", "busy", "compute_union", "comm_union", "idle", "api_launch_sum", "api_sync_sum", "gpu_tail_after_host", "kernels", "launches"])
        al = c["B_alignment"]
        lines += ["", "### B. rank alignment", "", f"iterations with all 4 ranks: {al['n_iterations_with_4_ranks']}; GPU windows overlap on all ranks: {al['gpu_windows_overlap_all_ranks']}",
                  f"first-kernel start spread ms p50/p90/max: {al['first_start_spread_ms']['p50']} / {al['first_start_spread_ms']['p90']} / {al['first_start_spread_ms']['max']}",
                  f"last-kernel end spread ms p50/p90/max: {al['last_end_spread_ms']['p50']} / {al['last_end_spread_ms']['p90']} / {al['last_end_spread_ms']['max']}",
                  f"earliest-starting rank counts: {al['earliest_rank_first_start_counts']}; latest-ending rank counts: {al['latest_rank_last_end_counts']}", ""]
        cw = c["C_collective_wait"]
        if "per_rank" in cw:
            lines += ["### C. collective wait (per iteration, ms; wait = rank comm time - fastest rank's comm time at the same collective)", ""]
            rows = [{"rank": r, "wait_sum p50/p90": f"{d['wait_sum_ms']['p50']} / {d['wait_sum_ms']['p90']}", "comm_sum p50/p90": f"{d['comm_sum_ms']['p50']} / {d['comm_sum_ms']['p90']}",
                     "wait share of comm (p50)": d["wait_share_of_comm_p50"]} for r, d in cw["per_rank"].items()]
            lines += md_table(rows, ["rank", "wait_sum p50/p90", "comm_sum p50/p90", "wait share of comm (p50)"])
            lines += ["", f"collectives per iteration p50 {cw['collectives_per_iteration']['p50']}; collective end spread across ranks ms p50/p90/max "
                      f"{cw['end_spread_ms']['p50']} / {cw['end_spread_ms']['p90']} / {cw['end_spread_ms']['max']}; least-waiting (straggler) rank {cw['least_waiting_rank_(straggler)']}, most-waiting rank {cw['most_waiting_rank']}",
                      "per kind (Phase 1A kind code): " + "; ".join(f"kind {k}: n={v['n']} wait p50 {v['wait_ms_p50']} p90 {v['wait_ms_p90']} comm p50 {v['comm_ms_p50']}" for k, v in cw["per_kind"].items()), ""]
        kd = c["D_kernels"]
        lines += [f"### D. kernel identities (top {a.top} by GPU time; {kd['n_distinct_names']} distinct names; kernels/iteration per rank {kd['per_rank_kernels_per_iteration']})", ""]
        rows = [{"#": i + 1, "kernel": f"`{k['name'][:80]}`", "MoE/GEMM": "x" if k["moe_gemm_family"] else "", "count/iter (4 ranks)": k["count_per_iteration_all_ranks"],
                 "mean us": k["mean_us"], "share": k["share_of_gpu_time"], "streams": k["n_streams"]} for i, k in enumerate(kd["top_kernels"])]
        lines += md_table(rows, ["#", "kernel", "MoE/GEMM", "count/iter (4 ranks)", "mean us", "share", "streams"]) + [""]
        if ref is not None:
            rit, rir, RK = ref
            rits = pure_decode_iterations(rit, b)
            e = {"ref_label": a.ref_label, "ref_n_iterations": int(len(rits))}
            if len(rits):
                rk = RK.per_cell(rits, a.top)
                only_a = sorted(set(kd["name_set"]) - set(rk["name_set"])); only_r = sorted(set(rk["name_set"]) - set(kd["name_set"]))
                e["kernel_names_only_in_this_pass"] = only_a; e["kernel_names_only_in_ref"] = only_r
                common = sorted(set(kd["name_set"]) & set(rk["name_set"]))
                diffs = []
                for n in common:
                    ca, ta = kd["name_time"][n]; cr, tr = rk["name_time"][n]
                    if abs(ca - cr) > 0.5 or (max(ta, tr) > 0.05 and abs(ta - tr) / max(ta, tr) > 0.25):
                        diffs.append({"kernel": f"`{n[:90]}`", f"count/iter {a.label}": round(ca, 2), f"count/iter {a.ref_label}": round(cr, 2), f"ms/iter {a.label}": round(ta, 4), f"ms/iter {a.ref_label}": round(tr, 4)})
                diffs.sort(key=lambda d: -abs(d[f"ms/iter {a.label}"] - d[f"ms/iter {a.ref_label}"]))
                e["common_kernels_with_count_or_time_difference"] = diffs[:40]
                ra = section_rank_exec(rir, rits)
                e["ref_rank_busy_ms_p50"] = {r: d["busy_ms"]["p50"] for r, d in ra.items()}
                e["this_rank_busy_ms_p50"] = {r: d["busy_ms"]["p50"] for r, d in c["A_rank_exec"].items()}
                e["ref_rank_span_ms_p50"] = {r: d["span_ms"]["p50"] for r, d in ra.items()}
                e["this_rank_span_ms_p50"] = {r: d["span_ms"]["p50"] for r, d in c["A_rank_exec"].items()}
                lines += [f"### E. cross-session comparison with `{a.ref_label}` ({len(rits)} pure-decode iterations at b={b}) — HYPOTHESIS MATERIAL ONLY", "",
                          f"`{a.ref_label}` was a different session with a different workload backend and the hook OFF; `{a.label}` is hook ON. Any difference below",
                          "mixes session-to-session drift (FlashInfer re-tuning; Stage 1.5 showed ~5 % between sessions at b=48) with any hook effect and cannot separate them.", "",
                          f"kernel names only in `{a.label}` ({len(only_a)}): " + ", ".join(f"`{n[:70]}`" for n in only_a[:15]) + (" …" if len(only_a) > 15 else ""),
                          f"kernel names only in `{a.ref_label}` ({len(only_r)}): " + ", ".join(f"`{n[:70]}`" for n in only_r[:15]) + (" …" if len(only_r) > 15 else ""), "",
                          "per-rank busy ms p50 this / ref: " + "; ".join(f"r{r}: {e['this_rank_busy_ms_p50'][r]} / {e['ref_rank_busy_ms_p50'].get(r)}" for r in e["this_rank_busy_ms_p50"]),
                          "per-rank span ms p50 this / ref: " + "; ".join(f"r{r}: {e['this_rank_span_ms_p50'][r]} / {e['ref_rank_span_ms_p50'].get(r)}" for r in e["this_rank_span_ms_p50"]), ""]
                if diffs:
                    lines += ["common kernels whose count/iteration or time/iteration differs (top 20 by time difference):", ""]
                    lines += md_table(diffs[:20], list(diffs[0].keys())) + [""]
            c["E_reference"] = e
    for b in report["cells"]:
        report["cells"][b].get("D_kernels", {}).pop("name_time", None)
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        open(a.out, "w").write(text)
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
