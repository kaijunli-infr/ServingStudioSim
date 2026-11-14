#!/usr/bin/env python3
"""Pre-registered E1 estimators (identification only; frozen before the capture, hash recorded).

Inputs are the compact tables produced by the frozen Phase 1A extractor (`costmodel.extract`) for the E1
capture, the engine's per-iteration metrics jsonl (cell membership via request ids), and optionally the
packed routed-experts corpus. Nothing here fits or changes a model parameter: the outputs are the
statistics the Phase 1C-0 uncertainty audit could not obtain from the frozen captures.

    python analysis/e1_identify.py --tables <pass table dir (meta.json + parquet)> --label <label> --out <dir>
                                   [--metrics <metrics.jsonl>] [--corpus <token_corpus dir>]
                                   [--calibration-only <partition.json>]
                                   [--frozen-coeffs <coefficients.json>] [--min-n 50] [--draws 1024] [--boot 200]

Without --metrics (smoke test on a frozen Phase 0 table) cells are formed from the table's own
decode_requests and a KV bucket of kv_max; with --metrics, cells are the E1 (b, KV, phase) groups.
A smoke test on a frozen table must pass --calibration-only with the frozen Phase 1A partition: only
calibration-role iterations are read, so the frozen held-out rows stay unopened (protocol section 1).

Estimators per cell (b, KV, phase in {pure, mixed}), on iterations with the cell's modal collective count K:
  W_meas[r]   mean measured wait per rank (sum over collectives of duration - min-over-ranks duration)
  W_med       wait produced by the median path (every rank at its per-segment median)
  W_ind       wait produced by independent per-(rank, segment) draws from the cell's empirical quantiles
  W_cm        wait produced by common-mode draws (the same quantile on every rank)
  f_ind       (W_meas - W_cm) / (W_ind - W_cm), mean over ranks; bootstrap CI over iterations
  rho         variance-weighted mean pairwise cross-rank correlation of per-segment deviations
  c_common    normalized common-mode variance fraction of per-segment deviations ((c - 1/R) / (1 - 1/R))
  arrival     spread across ranks of the arrival at the first collective (median, p90), last-rank histogram
  clock       max-min across ranks of each collective's end time (median, p90): cross-device alignment check
  moe         cross-rank correlation of the MoE group time; with --corpus, R^2 of wall and MoE time on
              per-iteration routing statistics (routing is common-mode by construction under TP without EP)
Decision statistics (pre-registered in E1-AUTHORIZATION-PACKET.md section 8) are computed from the f_ind
table: regime (b >= 24 vs b <= 16 at fixed KV), KV (across KV at fixed b), pass (E1 vs the frozen tables
at matched b; the frozen values come only from calibration-role rows, through the committed coefficients).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P1A = os.path.abspath(os.path.join(HERE, "..", "..", "..", "phase1a"))
sys.path.insert(0, P1A)
from costmodel.compose import critical_path_batch  # noqa: E402
from costmodel.data import PassData  # noqa: E402
from costmodel.decode import _sample  # noqa: E402
from costmodel.predict import Predictor  # noqa: E402

R = 4
TAG = re.compile(r"e1_b(\d+)_kv(\d+)k_w(\d+)_r\d+")
KV_BUCKETS = ((0, 8192, "kv<8k"), (8192, 49152, "kv8-48k"), (49152, 10**9, "kv>48k"))


def kv_bucket(kv: int) -> str:
    for lo, hi, name in KV_BUCKETS:
        if lo <= kv < hi:
            return name
    return "kv?"


# ----------------------------------------------------------------------------------------------- cells
def cells_from_metrics(path: str) -> dict[int, tuple]:
    """iteration -> (b, kv_tokens, phase, wave) from the engine's own per-iteration record."""
    out = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            prog = d.get("decode_request_progress") or []
            tags = {TAG.search(p["request_id"]).groups() for p in prog if TAG.search(p["request_id"])}
            if not tags or len({(t[0], t[1]) for t in tags}) != 1:
                continue                                    # no cell, or two cells overlapping: not a cell iteration
            b, kv, w = next(iter(tags))
            b, kv = int(b), int(kv) * 1024
            if d["decode_requests"] != b:
                continue                                    # ramp-up / ramp-down inside the wave
            phase = "pure" if d["prefill_tokens"] == 0 else "mixed"
            waves = {int(t[2]) for t in tags}
            out[int(d["iteration_index"])] = (b, kv, phase, min(waves) if len(waves) == 1 else -1)
    return out


def cells_from_tables(pd: PassData, allowed: set | None) -> dict[int, tuple]:
    out = {}
    it = pd.it
    for i, iteration in enumerate(it["iteration"]):
        if it["excluded"][i] or it["prefill_tokens"][i] != 0 or it["decode_requests"][i] <= 0:
            continue
        if allowed is not None and int(iteration) not in allowed:
            continue
        out[int(iteration)] = (int(it["decode_requests"][i]), kv_bucket(int(it["kv_max"][i])), "pure", 0)
    return out


# ------------------------------------------------------------------------------------------- estimators
def icdf(sorted_vals: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse empirical CDF with linear interpolation. sorted_vals: (n, *dims) sorted on axis 0; u: (S, *dims)."""
    n = sorted_vals.shape[0]
    pos = u * (n - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = pos - lo
    a = np.take_along_axis(sorted_vals, lo, axis=0)
    b = np.take_along_axis(sorted_vals, hi, axis=0)
    return a * (1 - w) + b * w


def wait_refs(start, seg, intr, dur, S, rng):
    """W_meas, W_med, W_ind, W_cm per rank (ms) from one cell's arrays (n,R), (n,R,K+1), (n,K), (n,R,K)."""
    n, _, K1 = seg.shape
    w_meas = (dur - intr[:, None, :]).sum(axis=2).mean(axis=0) / 1e6
    med = critical_path_batch(np.median(start, axis=0)[None], np.median(seg, axis=0)[None], np.median(intr, axis=0)[None])
    ss, sg, si = np.sort(start, axis=0), np.sort(seg, axis=0), np.sort(intr, axis=0)
    u_i = rng.random((S, K1 - 1))
    ind = critical_path_batch(icdf(ss, rng.random((S, R))), icdf(sg, rng.random((S, R, K1))), icdf(si, u_i))
    u_g = np.repeat(rng.random((S, 1, K1)), R, axis=1)
    u_s = np.repeat(rng.random((S, 1)), R, axis=1)
    com = critical_path_batch(icdf(ss, u_s), icdf(sg, u_g), icdf(si, u_i))
    return (w_meas, med["rank_wait"].mean(axis=0) / 1e6, ind["rank_wait"].mean(axis=0) / 1e6, com["rank_wait"].mean(axis=0) / 1e6)


def f_ind_from(w_meas, w_ind, w_cm) -> float:
    return float(np.clip((w_meas - w_cm) / np.maximum(w_ind - w_cm, 1e-9), -1, 2).mean())


def cross_rank_stats(seg: np.ndarray) -> dict:
    d = seg - np.median(seg, axis=0, keepdims=True)                 # (n,R,K+1)
    var = d.var(axis=0)                                            # (R,K+1)
    w = var.mean(axis=0)                                           # per segment
    rho_k = []
    for k in range(seg.shape[2]):
        c = np.corrcoef(d[:, :, k].T)
        iu = np.triu_indices(R, 1)
        rho_k.append(float(np.nanmean(c[iu])) if np.isfinite(c[iu]).any() else 0.0)
    rho_k = np.asarray(rho_k)
    c_raw = d.mean(axis=1).var(axis=0) / np.maximum(var.mean(axis=0), 1e-9)     # per segment
    c_norm = (c_raw - 1 / R) / (1 - 1 / R)
    wsum = max(float(w.sum()), 1e-9)
    return {"rho": float((rho_k * w).sum() / wsum), "c_common": float((c_norm * w).sum() / wsum),
            "rho_first_segment": float(rho_k[0]), "rho_max_var_segment": float(rho_k[int(np.argmax(w))])}


def arrival_stats(start, seg, dur) -> dict:
    arr0 = start + seg[:, :, 0]                                    # arrival at the first collective (n,R)
    spread = arr0.max(axis=1) - arr0.min(axis=1)
    late = np.argmax(arr0, axis=1)
    end = start[:, :, None] + np.cumsum(seg[:, :, :-1] + dur, axis=2)   # end of collective k per rank (n,R,K)
    dend = end.max(axis=1) - end.min(axis=1)                           # (n,K)
    return {"first_arrival_spread_ms_median": float(np.median(spread) / 1e6), "first_arrival_spread_ms_p90": float(np.quantile(spread, 0.9) / 1e6),
            "last_arriving_rank_hist": np.bincount(late, minlength=R).tolist(),
            "mean_lateness_ms_per_rank": ((arr0 - arr0.min(axis=1, keepdims=True)).mean(axis=0) / 1e6).round(4).tolist(),
            "collective_end_spread_ms_median": float(np.median(dend) / 1e6), "collective_end_spread_ms_p90": float(np.quantile(dend, 0.9) / 1e6)}


def moe_stats(pd: PassData, iters: list[int], routes=None) -> dict:
    g = np.stack([pd.rank_col("g_moe_sum_ns", i) for i in iters]) / 1e6           # (n,R)
    wall = np.asarray([pd.it_value("wall_ns", i) for i in iters]) / 1e6
    c = np.corrcoef(g.T)
    out = {"moe_ms_mean": float(g.mean()), "moe_cross_rank_corr": float(np.nanmean(c[np.triu_indices(R, 1)])),
           "moe_wall_corr": float(np.corrcoef(g.mean(axis=1), wall)[0, 1]) if len(iters) > 2 else None}
    if routes is not None:
        X = np.asarray([routes[i] for i in iters])                                  # (n, 3)
        if len(iters) > 10 and np.isfinite(X).all():
            A = np.c_[np.ones(len(iters)), X]
            for name, y in (("r2_wall_on_routing", wall), ("r2_moe_on_routing", g.mean(axis=1))):
                beta, *_ = np.linalg.lstsq(A, y, rcond=None)
                res = y - A @ beta
                out[name] = float(1 - res.var() / max(y.var(), 1e-12))
            out["routing_stat_means"] = X.mean(axis=0).round(3).tolist()
    return out


# ---------------------------------------------------------------------------------------------- routing
def routing_stats(corpus_dir: str, metrics_path: str, block_m: int = 64) -> dict[int, np.ndarray]:
    """iteration -> (mean over layers of: max expert load, #non-empty experts, padded M / tokens)."""
    man = json.load(open(os.path.join(corpus_dir, "manifest.json")))
    prov = json.load(open(os.path.join(corpus_dir, "provenance.json")))
    L, E, T = man["num_layers"], man["num_experts"], man["top_k"]
    routes = np.fromfile(os.path.join(corpus_dir, man["data_file"]), dtype="<u2").reshape(man["num_tokens"], L, T)
    seg = {s["request"]: (s["start"], s["length"]) for s in prov["request_segments"]}
    out = {}
    with open(metrics_path) as fh:
        for line in fh:
            d = json.loads(line)
            if d["prefill_tokens"] != 0 or not d["decode_requests"]:
                continue
            rows = []
            for p in d["decode_request_progress"]:
                key = "independent_" + p["request_id"] if not p["request_id"].startswith("independent_") else p["request_id"]
                s = seg.get(key)
                j = p["output_tokens_before"]
                if s is None or j < 1 or j - 1 >= s[1]:
                    rows = None
                    break
                rows.append(s[0] + j - 1)
            if not rows:
                continue
            r = routes[np.asarray(rows)]                                            # (b, L, T)
            maxload, nonempty, padded = [], [], []
            for l in range(L):
                h = np.bincount(r[:, l, :].ravel(), minlength=E)
                if h.sum() == 0:
                    continue
                maxload.append(h.max()); nonempty.append((h > 0).sum())
                padded.append((np.ceil(h / block_m) * block_m).sum() / max(h.sum(), 1))
            if maxload:
                out[int(d["iteration_index"])] = np.array([np.mean(maxload), np.mean(nonempty), np.mean(padded)])
    return out


# ----------------------------------------------------------------------------------------------- frozen
def frozen_fractions(coeffs_path: str, bs, S: int, seed: int) -> dict:
    d = Predictor(json.load(open(coeffs_path))).decode
    rng = np.random.default_rng(seed)
    out = {}
    for b in bs:
        p, st = d._lookup(b)
        if (p.get("n") or 0) < 30:
            continue                                    # protocol section 5: n < 30 is reported elsewhere, never compared
        K = p["K"]
        med = critical_path_batch(np.asarray(p["start"], float)[None], np.asarray(p["seg"], float)[None], np.asarray(p["intrinsic"], float)[None])
        ind = critical_path_batch(_sample(p["start_q"], rng.random((S, R))), _sample(p["seg_q"], rng.random((S, R, K + 1))), _sample(p["intrinsic_q"], rng.random((S, K))))
        ug = np.repeat(rng.random((S, 1, K + 1)), R, axis=1); us = np.repeat(rng.random((S, 1)), R, axis=1)
        com = critical_path_batch(_sample(p["start_q"], us), _sample(p["seg_q"], ug), _sample(p["intrinsic_q"], rng.random((S, K))))
        meas = np.asarray(p["wait"], float) / 1e6
        out[int(b)] = {"status": st, "n": p.get("n"), "K": K, "sources": p.get("sources") or p.get("source") or p.get("passes") or p.get("pass"),
                       "f_ind": f_ind_from(meas, ind["rank_wait"].mean(axis=0) / 1e6, com["rank_wait"].mean(axis=0) / 1e6),
                       "w_meas": meas.round(4).tolist(), "w_med": (med["rank_wait"].mean(axis=0) / 1e6).round(4).tolist()}
    return out


# ------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True); ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--metrics"); ap.add_argument("--corpus"); ap.add_argument("--frozen-coeffs", default=os.path.join(P1A, "out", "coefficients.json"))
    ap.add_argument("--calibration-only", help="frozen partition.json: read only calibration-role iterations of --label (smoke tests)")
    ap.add_argument("--min-n", type=int, default=50); ap.add_argument("--draws", type=int, default=1024)
    ap.add_argument("--boot", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    pd = PassData.load(a.label, a.tables)
    allowed = None
    if a.calibration_only:
        roles = json.load(open(a.calibration_only))["passes"][a.label]["role"]
        allowed = {int(k) for k, r in roles.items() if r == "calibration"}
        print(f"calibration-only: {len(allowed)} of {len(roles)} iterations of {a.label} readable")
    elif not a.metrics:
        sys.exit("refusing to read a frozen table without --calibration-only (held-out rows must stay unopened)")
    cells = cells_from_metrics(a.metrics) if a.metrics else cells_from_tables(pd, allowed)
    routes = routing_stats(a.corpus, a.metrics) if (a.corpus and a.metrics) else None
    excluded = {int(i) for i, x in zip(pd.it["iteration"], pd.it["excluded"]) if x}
    groups = defaultdict(list)
    for it, (b, kv, phase, wave) in cells.items():
        if it not in excluded:
            groups[(b, kv, phase)].append(it)

    results = []
    for key in sorted(groups, key=lambda k: (str(k[1]), k[0], k[2])):
        iters = sorted(groups[key])
        segs = {i: pd.segments(i) for i in iters}
        segs = {i: s for i, s in segs.items() if s is not None}
        Ks = Counter(s[1].shape[1] - 1 for s in segs.values())
        if not Ks:
            continue
        K = Ks.most_common(1)[0][0]
        iters = [i for i in iters if i in segs and segs[i][1].shape[1] - 1 == K]
        row = {"b": key[0], "kv": key[1], "phase": key[2], "n": len(iters), "K": K, "n_other_K": sum(v for k, v in Ks.items() if k != K)}
        if len(iters) < a.min_n:
            row["skipped"] = f"n < {a.min_n}"
            results.append(row); continue
        start = np.stack([segs[i][0] for i in iters]).astype(float)
        seg = np.stack([segs[i][1] for i in iters]).astype(float)
        intr = np.stack([segs[i][2] for i in iters]).astype(float)
        dur = np.stack([segs[i][3] for i in iters]).astype(float)
        wm, wmed, wi, wc = wait_refs(start, seg, intr, dur, a.draws, rng)
        f = f_ind_from(wm, wi, wc)
        boots = []
        for _ in range(a.boot):
            idx = rng.integers(0, len(iters), len(iters))
            bm, _, bi, bc = wait_refs(start[idx], seg[idx], intr[idx], dur[idx], max(a.draws // 4, 256), rng)
            boots.append(f_ind_from(bm, bi, bc))
        wall = np.asarray([pd.it_value("wall_ns", i) for i in iters]) / 1e6
        row.update({"wall_ms_median": float(np.median(wall)), "wall_ms_mean": float(wall.mean()), "wall_ms_cv": float(wall.std() / wall.mean()),
                    "w_meas": wm.round(4).tolist(), "w_med": wmed.round(4).tolist(), "w_ind": wi.round(4).tolist(), "w_cm": wc.round(4).tolist(),
                    "f_ind": round(f, 4), "f_ind_ci90": [round(float(np.quantile(boots, 0.05)), 4), round(float(np.quantile(boots, 0.95)), 4)],
                    **cross_rank_stats(seg), **arrival_stats(start, seg, dur), **moe_stats(pd, iters, routes)})
        results.append(row)

    frozen = frozen_fractions(a.frozen_coeffs, sorted({r["b"] for r in results}), a.draws, a.seed) if a.frozen_coeffs and os.path.exists(a.frozen_coeffs) else {}

    # decision statistics (pure phase only)
    pure = {(r["b"], r["kv"]): r for r in results if r["phase"] == "pure" and "f_ind" in r}
    kvs = sorted({k[1] for k in pure}, key=str)
    decision = {"regime": {}, "kv": {}, "pass": {}}
    for kv in kvs:
        lo = [pure[(b, kv)]["f_ind"] for b in (1, 2, 4, 8, 16) if (b, kv) in pure]
        hi = [pure[(b, kv)]["f_ind"] for b in (24, 32, 48) if (b, kv) in pure]
        if lo and hi:
            decision["regime"][str(kv)] = {"f_b_le16": round(float(np.mean(lo)), 4), "f_b_ge24": round(float(np.mean(hi)), 4), "delta": round(float(np.mean(hi) - np.mean(lo)), 4)}
    for b in sorted({k[0] for k in pure}):
        fs = {str(kv): pure[(b, kv)]["f_ind"] for kv in kvs if (b, kv) in pure}
        if len(fs) > 1:
            decision["kv"][str(b)] = {**fs, "range": round(max(fs.values()) - min(fs.values()), 4)}
    for b, fr in frozen.items():
        match = [pure[(b, kv)] for kv in kvs if (b, kv) in pure and (b >= 24 and str(kv) in ("4096", "kv<8k") or b < 24)]
        if match:
            decision["pass"][str(b)] = {"f_frozen": round(fr["f_ind"], 4), "frozen_sources": fr["sources"],
                                       "f_e1": {str(m["kv"]): m["f_ind"] for m in match},
                                       "delta_max_abs": round(max(abs(m["f_ind"] - fr["f_ind"]) for m in match), 4)}

    out = {"label": a.label, "tables": a.tables, "metrics": a.metrics, "corpus": a.corpus, "calibration_only": a.calibration_only,
           "draws": a.draws, "boot": a.boot, "seed": a.seed,
           "n_cells": len(results), "cells": results, "frozen": frozen, "decision": decision}
    json.dump(out, open(os.path.join(a.out, "e1_identify.json"), "w"), indent=1)
    with open(os.path.join(a.out, "E1-IDENTIFY.md"), "w") as fh:
        fh.write(f"# E1 identification statistics — {a.label}\n\n| b | KV | phase | n | K | wall med ms | f_ind | CI90 | rho | c_common | arr. spread p50/p90 ms | last rank hist | coll. end spread p50/p90 ms | MoE xrank corr | R2 wall~routing |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            if "f_ind" not in r:
                fh.write(f"| {r['b']} | {r['kv']} | {r['phase']} | {r['n']} | {r['K']} | – | {r.get('skipped','')} | | | | | | | | |\n"); continue
            fh.write(f"| {r['b']} | {r['kv']} | {r['phase']} | {r['n']} | {r['K']} | {r['wall_ms_median']:.3f} | {r['f_ind']:.2f} | {r['f_ind_ci90'][0]:.2f}–{r['f_ind_ci90'][1]:.2f} | {r['rho']:.2f} | {r['c_common']:.2f} | "
                     f"{r['first_arrival_spread_ms_median']:.3f}/{r['first_arrival_spread_ms_p90']:.3f} | {r['last_arriving_rank_hist']} | {r['collective_end_spread_ms_median']:.4f}/{r['collective_end_spread_ms_p90']:.4f} | {r['moe_cross_rank_corr']:.2f} | {r.get('r2_wall_on_routing', float('nan')):.2f} |\n")
        fh.write("\n## Frozen tables (calibration rows only, via the committed coefficients)\n\n| b | status | n | K | f_ind |\n|---|---|---|---|---|\n")
        for b, fr in sorted(frozen.items()):
            fh.write(f"| {b} | {fr['status']} | {fr['n']} | {fr['K']} | {fr['f_ind']:.2f} |\n")
        fh.write("\n## Decision statistics\n\n```\n" + json.dumps(decision, indent=1) + "\n```\n")
    print(json.dumps({"cells": len(results), "with_estimates": sum('f_ind' in r for r in results), "decision": decision}, indent=1))


if __name__ == "__main__":
    main()
