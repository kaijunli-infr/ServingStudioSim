#!/usr/bin/env python3
"""Rank-dependence audit on existing nsys traces (CPU, read-only; numpy + pyarrow). Nothing is fitted.

    python e1_rank_dependence.py --pass <label>:<tables dir>:<b> [--pass ...] --out <dir> [--boot 100] [--seed 0]

For every pass and its pure-decode cell b (decode_requests == b, no prefill, not JIT-excluded, modal collective count K on all
four ranks) the Phase 1A collectives table gives, per (iteration, rank, k): the rank's arrival at collective k (start_rel_ns),
its completion (end_rel_ns), its time inside the collective (comm_sum_ns) and the compute segment before it (seg_len_ns).

Directly measured (no model):
  * arrival / completion per rank and collective, arrival spread and completion (end) spread across ranks
  * wait_direct[r,k]  = latest arrival at k - arrival of r        (valid when completion is simultaneous; end spread reported)
  * wait_proxy[r,k]   = comm[r,k] - min_r' comm[r',k]             (Phase 1A definition: the fastest rank is the intrinsic proxy)
  * intrinsic_direct[k] = completion - latest arrival;  intrinsic_proxy[k] = min_r comm[r,k]
  * per-rank wait sums per iteration (p50/p90/p99/max), last-arriving-rank frequencies per collective and per iteration,
    critical-path (least-waiting) rank frequency, cross-rank correlation of segments / busy / span / wait, temporal autocorrelation
Model-dependent (composition by the barrier chain, no coefficients): W_ind / W_cm by permutation and quantile pairing of the
measured segments, f_ind = (W_meas - W_cm) / (W_ind - W_cm) with a bootstrap CI over iterations.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq

R = 4


def q(a, p):
    return float(np.percentile(np.asarray(a, float), p))


def ms(ns):
    return round(float(ns) / 1e6, 4)


def pct(a):
    a = np.asarray(a, float) / 1e6
    return {"p50": round(q(a, 50), 4), "p90": round(q(a, 90), 4), "p99": round(q(a, 99), 4), "max": round(float(a.max()), 4), "mean": round(float(a.mean()), 4)}


def load(tables: str, b: int):
    it = pq.read_table(os.path.join(tables, "iterations.parquet")).to_pydict()
    m = [(s == "decode") and (p == 0) and (d == b) and (not e) for s, p, d, e in zip(it["stage"], it["prefill_tokens"], it["decode_requests"], it["excluded"])]
    its = np.asarray(it["iteration"])[np.asarray(m)]
    wall = dict(zip(it["iteration"], np.asarray(it["wall_ns"], np.int64)))
    co = pq.read_table(os.path.join(tables, "collectives.parquet")).to_pydict()
    co = {k: np.asarray(v) for k, v in co.items()}
    sel = np.isin(co["iteration"], its)
    co = {k: v[sel] for k, v in co.items()}
    ir = pq.read_table(os.path.join(tables, "iter_rank.parquet")).to_pydict()
    ir = {k: np.asarray(v) for k, v in ir.items()}
    sel = np.isin(ir["iteration"], its)
    ir = {k: v[sel] for k, v in ir.items()}
    return its, wall, co, ir


def cube(co, its):
    """arrays [n_it, R, K] for iterations where every rank has the modal K collectives with identical kind sequences."""
    key = co["iteration"].astype(np.int64) * 8 + co["rank"].astype(np.int64)
    counts = {}
    for kk in key:
        counts[kk] = counts.get(kk, 0) + 1
    Ks = np.array(list(counts.values()))
    K = int(np.bincount(Ks).argmax())
    good = []
    for i in its:
        if all(counts.get(int(i) * 8 + r, 0) == K for r in range(R)):
            good.append(int(i))
    good = np.asarray(good)
    idx = {int(i): n for n, i in enumerate(good)}
    sel = np.isin(co["iteration"], good)
    n = len(good)
    A = np.zeros((n, R, K), np.int64); E = np.zeros_like(A); D = np.zeros_like(A); SEG = np.zeros_like(A); KIND = np.zeros((n, R, K), np.int16)
    ii = np.array([idx[int(x)] for x in co["iteration"][sel]]); rr = co["rank"][sel].astype(int); kk = co["k"][sel].astype(int)
    A[ii, rr, kk] = co["start_rel_ns"][sel]; E[ii, rr, kk] = co["end_rel_ns"][sel]; D[ii, rr, kk] = co["comm_sum_ns"][sel]; SEG[ii, rr, kk] = co["seg_len_ns"][sel]
    KIND[ii, rr, kk] = co["kind"][sel]
    same_kind = bool((KIND == KIND[:, :1, :]).all())
    return good, K, A, E, D, SEG, KIND[:, 0, :], same_kind


def acf(x, lags):
    x = np.asarray(x, float); x = x - x.mean(); v = float((x * x).mean()) or 1.0
    return {int(l): round(float((x[:-l] * x[l:]).mean() / v), 3) for l in lags if l < len(x)}


def chain_wait(seg, intrinsic, start):
    """Barrier chain: arrival_k[r] = completion_{k-1} + seg[r,k]; completion_k = max_r arrival + intrinsic_k. Returns wait sums per rank [n, R]."""
    n, R_, K = seg.shape
    wait = np.zeros((n, R_)); comp = None
    for k in range(K):
        arr = (start + seg[:, :, 0]) if k == 0 else (comp[:, None] + seg[:, :, k])
        latest = arr.max(axis=1)
        wait += latest[:, None] - arr
        comp = latest + intrinsic[:, k]
    return wait


def analyse(label, tables, b, boot, seed):
    its, wall, co, ir = load(tables, b)
    good, K, A, E, D, SEG, kind, same_kind = cube(co, its)
    n = len(good)
    out = {"label": label, "b": b, "n_pure_decode": int(len(its)), "n_modal_K": n, "K": K, "kind_sequence_identical_across_ranks": same_kind}
    Af = A.astype(float); Ef = E.astype(float); Df = D.astype(float); Sf = SEG.astype(float)
    latest = Af.max(axis=1)                                   # [n, K]
    end_spread = Ef.max(axis=1) - Ef.min(axis=1)
    arr_spread = latest - Af.min(axis=1)
    wait_direct = latest[:, None, :] - Af                     # [n, R, K]
    wait_proxy = Df - Df.min(axis=1)[:, None, :]
    intr_direct = np.median(Ef, axis=1) - latest
    intr_proxy = Df.min(axis=1)
    # 1. arrival / completion reconstruction
    out["reconstruction"] = {
        "collective_end_spread_ms": pct(end_spread.ravel()),
        "arrival_spread_ms_all_k": pct(arr_spread.ravel()),
        "arrival_spread_ms_first_collective": pct(arr_spread[:, 0]),
        "intrinsic_direct_ms_per_iteration_sum": pct(intr_direct.sum(axis=1)),
        "intrinsic_proxy_ms_per_iteration_sum": pct(intr_proxy.sum(axis=1)),
        "intrinsic_direct_minus_proxy_ms_per_collective": pct((intr_direct - intr_proxy).ravel()),
        "fraction_of_collectives_with_end_spread_gt_20us": round(float((end_spread > 20_000).mean()), 5),
        "intrinsic_by_kind_ms_p50": {int(kd): round(q(intr_proxy[:, kind[0] == kd], 50) / 1e6, 5) for kd in np.unique(kind[0])},
        "collectives_per_kind": {int(kd): int((kind[0] == kd).sum()) for kd in np.unique(kind[0])},
    }
    # 2./3. per-rank waiting distributions and tails; separation of intrinsic from waiting
    wd = wait_direct.sum(axis=2); wp = wait_proxy.sum(axis=2)        # [n, R]
    comm = Df.sum(axis=2)
    out["wait_per_rank"] = {int(r): {"direct_sum_ms": pct(wd[:, r]), "proxy_sum_ms": pct(wp[:, r]),
                                     "comm_sum_ms": pct(comm[:, r]), "wait_share_of_comm_p50": round(float(np.median(wp[:, r] / np.maximum(comm[:, r], 1))), 4),
                                     "wait_share_of_wall_p50": round(float(np.median(wp[:, r] / np.array([wall[int(i)] for i in good]))), 4)} for r in range(R)}
    out["wait_direct_vs_proxy_abs_diff_ms_per_iteration"] = pct(np.abs(wd - wp).ravel())
    last_k = Af.argmax(axis=1)                                        # [n, K] last-arriving rank per collective
    out["last_arriving_rank_share_per_collective"] = {int(r): round(float((last_k == r).mean()), 4) for r in range(R)}
    out["last_arriving_rank_share_first_collective"] = {int(r): round(float((last_k[:, 0] == r).mean()), 4) for r in range(R)}
    out["last_arriving_rank_share_final_collective"] = {int(r): round(float((last_k[:, -1] == r).mean()), 4) for r in range(R)}
    crit = wp.argmin(axis=1)                                          # least-waiting rank = critical path (Phase 1A convention)
    out["critical_path_rank_share_least_waiting"] = {int(r): round(float((crit == r).mean()), 4) for r in range(R)}
    # per-kind wait
    out["wait_proxy_by_kind_ms_p50_p90"] = {int(kd): [round(q(wait_proxy[:, :, kind[0] == kd], 50) / 1e6, 5), round(q(wait_proxy[:, :, kind[0] == kd], 90) / 1e6, 5)] for kd in np.unique(kind[0])}
    # per-rank lateness at the first collective (host arrival): mean(arrival_r - min arrival)
    out["first_collective_lateness_ms_mean_per_rank"] = {int(r): round(float((Af[:, r, 0] - Af[:, :, 0].min(axis=1)).mean()) / 1e6, 4) for r in range(R)}
    # 5. covariance / correlation across ranks
    dev = Sf - Sf.mean(axis=0, keepdims=True)                         # per-(rank, segment) deviations [n, R, K]
    rhos = []; wts = []
    for k in range(K):
        x = dev[:, :, k]; v = x.var(axis=0)
        if (v <= 0).any():
            continue
        c = np.corrcoef(x.T); pair = c[np.triu_indices(R, 1)]
        rhos.append(float(pair.mean())); wts.append(float(v.mean()))
    rhos = np.array(rhos); wts = np.array(wts)
    cm = dev.mean(axis=1)                                             # common-mode component per (it, k)
    c = float((cm.var(axis=0) * 1.0).sum() / dev.var(axis=0).mean(axis=0).sum())   # var(mean_r)/mean_r var
    out["segment_cross_rank_correlation"] = {"rho_variance_weighted": round(float((rhos * wts).sum() / wts.sum()), 4), "rho_p10_p50_p90": [round(q(rhos, 10), 3), round(q(rhos, 50), 3), round(q(rhos, 90), 3)],
                                             "c_common_normalized": round((c - 1 / R) / (1 - 1 / R), 4), "n_segments": int(len(rhos))}
    def cmat(name, M):
        c = np.corrcoef(M.T); return {name: [[round(float(v), 3) for v in row] for row in c]}
    busy = np.zeros((n, R)); span = np.zeros((n, R)); moe = np.zeros((n, R))
    idx = {int(i): t for t, i in enumerate(good)}
    sel = np.isin(ir["iteration"], good)
    busy[[idx[int(i)] for i in ir["iteration"][sel]], ir["rank"][sel].astype(int)] = ir["busy_ns"][sel]
    span[[idx[int(i)] for i in ir["iteration"][sel]], ir["rank"][sel].astype(int)] = ir["span_ns"][sel]
    moe[[idx[int(i)] for i in ir["iteration"][sel]], ir["rank"][sel].astype(int)] = ir["g_moe_sum_ns"][sel]
    out["per_iteration_cross_rank_corr"] = {**cmat("busy", busy), **cmat("span", span), **cmat("moe_group_time", moe), **cmat("wait_proxy_sum", wp)}
    out["moe_group_time_ms_p50_per_rank"] = {int(r): ms(np.median(moe[:, r])) for r in range(R)}
    out["corr_rank_own_segment_total_vs_own_wait"] = [round(float(np.corrcoef(Sf.sum(axis=2)[:, r], wp[:, r])[0, 1]), 3) for r in range(R)]
    # temporal autocorrelation (iterations are consecutive within the wave)
    walls = np.array([wall[int(i)] for i in good], float)
    lags = (1, 2, 3, 5, 10, 20, 50, 100)
    out["autocorrelation"] = {"wall": acf(walls, lags), "mean_rank_busy": acf(busy.mean(axis=1), lags), "common_mode_segment_sum": acf(cm.sum(axis=1), lags),
                              "rank0_wait_sum": acf(wp[:, 0], lags), "rank_independent_part_sum": acf((dev - cm[:, None, :]).sum(axis=2)[:, 0], lags),
                              "last_arriving_rank_final_collective_lag1_persistence": round(float((last_k[1:, -1] == last_k[:-1, -1]).mean()), 3),
                              "last_arriving_rank_final_collective_chance": round(float(sum(((last_k[:, -1] == r).mean()) ** 2 for r in range(R))), 3)}
    # model-dependent: barrier-chain composition from the measured segments (permutation = independent, quantile pairing = common-mode)
    rng = np.random.default_rng(seed)
    start = Af[:, :, 0] - Sf[:, :, 0]                                 # rank start of the iteration (arrival at k=0 minus its first segment)
    start = start - start.min(axis=1, keepdims=True)
    W_meas = chain_wait(Sf, intr_proxy, start)
    W_meas_proxy = wp
    def W_ind(idx_):
        s = np.stack([Sf[rng.permutation(idx_), r, :] for r in range(R)], axis=1); st = np.stack([start[rng.permutation(idx_), r] for r in range(R)], axis=1)
        return chain_wait(s, intr_proxy[idx_], st)
    def W_cm(idx_):
        s = np.sort(Sf[idx_], axis=0); st = np.sort(start[idx_], axis=0)   # same quantile index on every rank
        return chain_wait(s, np.sort(intr_proxy[idx_], axis=0), st)
    allidx = np.arange(n)
    wi = W_ind(allidx).mean(axis=0); wc = W_cm(allidx).mean(axis=0); wm = W_meas.mean(axis=0)
    f = ((wm - wc) / (wi - wc)).mean()
    fb = []
    for _ in range(boot):
        bi = rng.integers(0, n, n)
        fb.append(((W_meas[bi].mean(axis=0) - W_cm(bi).mean(axis=0)) / (W_ind(bi).mean(axis=0) - W_cm(bi).mean(axis=0))).mean())
    out["composition_no_coefficients"] = {"W_meas_chain_ms_per_rank": [ms(x) for x in wm], "W_meas_proxy_ms_per_rank": [ms(x) for x in W_meas_proxy.mean(axis=0)],
                                          "W_ind_permutation_ms_per_rank": [ms(x) for x in wi], "W_cm_quantile_ms_per_rank": [ms(x) for x in wc],
                                          "f_ind": round(float(f), 3), "f_ind_ci90": [round(q(fb, 5), 3), round(q(fb, 95), 3)], "boot": boot,
                                          "note": "chain = barrier composition of the measured segments with the fastest-rank intrinsic; 'proxy' = Phase 1A measured wait; both are measured quantities, W_ind/W_cm are counterfactual compositions"}
    return out


def md(results):
    L = ["# Rank dependence from existing traces (CPU audit; nothing fitted)", ""]
    L += ["| pass | b | pure iters | with modal K | K | kinds identical across ranks | end spread p50/p90/max ms | arrival spread all k p50/p90 ms | first-collective arrival spread p50/p90 ms |", "|---|---|---|---|---|---|---|---|---|"]
    for o in results:
        r = o["reconstruction"]
        L.append(f"| {o['label']} | {o['b']} | {o['n_pure_decode']} | {o['n_modal_K']} | {o['K']} | {o['kind_sequence_identical_across_ranks']} | {r['collective_end_spread_ms']['p50']} / {r['collective_end_spread_ms']['p90']} / {r['collective_end_spread_ms']['max']} | {r['arrival_spread_ms_all_k']['p50']} / {r['arrival_spread_ms_all_k']['p90']} | {r['arrival_spread_ms_first_collective']['p50']} / {r['arrival_spread_ms_first_collective']['p90']} |")
    L += ["", "## Intrinsic vs waiting (per iteration sums, ms)", "", "| pass | intrinsic direct p50 | intrinsic proxy p50 | direct−proxy per collective p50/p90 (ms) | collectives with end spread > 20 µs |", "|---|---|---|---|---|"]
    for o in results:
        r = o["reconstruction"]
        L.append(f"| {o['label']} b={o['b']} | {r['intrinsic_direct_ms_per_iteration_sum']['p50']} | {r['intrinsic_proxy_ms_per_iteration_sum']['p50']} | {r['intrinsic_direct_minus_proxy_ms_per_collective']['p50']} / {r['intrinsic_direct_minus_proxy_ms_per_collective']['p90']} | {r['fraction_of_collectives_with_end_spread_gt_20us']} |")
    L += ["", "## Per-rank waiting (sum over the iteration's collectives, ms)", "", "| pass | rank | direct p50 / p90 / p99 / max | proxy p50 / p90 / p99 / max | comm p50 | wait share of comm p50 | wait share of wall p50 | first-collective lateness mean |", "|---|---|---|---|---|---|---|---|"]
    for o in results:
        for r, d in o["wait_per_rank"].items():
            a = d["direct_sum_ms"]; p = d["proxy_sum_ms"]
            L.append(f"| {o['label']} b={o['b']} | {r} | {a['p50']} / {a['p90']} / {a['p99']} / {a['max']} | {p['p50']} / {p['p90']} / {p['p99']} / {p['max']} | {d['comm_sum_ms']['p50']} | {d['wait_share_of_comm_p50']} | {d['wait_share_of_wall_p50']} | {o['first_collective_lateness_ms_mean_per_rank'][r]} |")
    L += ["", "## Which rank is last / on the critical path", "", "| pass | last-arriving share per collective (r0 r1 r2 r3) | first collective | final collective | least-waiting (critical path) rank share | lag-1 persistence of the final-collective last rank (chance) |", "|---|---|---|---|---|---|"]
    for o in results:
        f_ = lambda d: " ".join(f"{d[r]:.3f}" for r in sorted(d))
        L.append(f"| {o['label']} b={o['b']} | {f_(o['last_arriving_rank_share_per_collective'])} | {f_(o['last_arriving_rank_share_first_collective'])} | {f_(o['last_arriving_rank_share_final_collective'])} | {f_(o['critical_path_rank_share_least_waiting'])} | {o['autocorrelation']['last_arriving_rank_final_collective_lag1_persistence']} ({o['autocorrelation']['last_arriving_rank_final_collective_chance']}) |")
    L += ["", "## Cross-rank covariance of timing", "", "| pass | segment ρ (variance-weighted) | ρ p10/p50/p90 over segments | c_common (normalized) | busy corr (min off-diag) | MoE group corr (min off-diag) | wait-sum corr (min/max off-diag) | corr(own segments, own wait) per rank |", "|---|---|---|---|---|---|---|---|"]
    for o in results:
        s = o["segment_cross_rank_correlation"]; c = o["per_iteration_cross_rank_corr"]
        off = lambda M: [M[i][j] for i in range(R) for j in range(R) if i != j]
        L.append(f"| {o['label']} b={o['b']} | {s['rho_variance_weighted']} | {s['rho_p10_p50_p90']} | {s['c_common_normalized']} | {min(off(c['busy']))} | {min(off(c['moe_group_time']))} | {min(off(c['wait_proxy_sum']))} / {max(off(c['wait_proxy_sum']))} | {o['corr_rank_own_segment_total_vs_own_wait']} |")
    L += ["", "## Temporal autocorrelation (lag: 1, 2, 3, 5, 10, 20, 50, 100 iterations)", "", "| pass | series | ACF |", "|---|---|---|"]
    for o in results:
        for k in ("wall", "mean_rank_busy", "common_mode_segment_sum", "rank_independent_part_sum", "rank0_wait_sum"):
            L.append(f"| {o['label']} b={o['b']} | {k} | {o['autocorrelation'][k]} |")
    L += ["", "## Composition without coefficients (barrier chain over the measured segments)", "", "| pass | W_meas chain per rank | W_meas Phase 1A proxy per rank | W_ind (permutation) | W_cm (quantile pairing) | f_ind [90 % CI] |", "|---|---|---|---|---|---|"]
    for o in results:
        c = o["composition_no_coefficients"]
        L.append(f"| {o['label']} b={o['b']} | {c['W_meas_chain_ms_per_rank']} | {c['W_meas_proxy_ms_per_rank']} | {c['W_ind_permutation_ms_per_rank']} | {c['W_cm_quantile_ms_per_rank']} | {c['f_ind']} {c['f_ind_ci90']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="passes", action="append", required=True, help="label:tables:b")
    ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = []
    for spec in a.passes:
        label, tables, b = spec.rsplit(":", 2)
        o = analyse(label, tables, int(b), a.boot, a.seed); res.append(o)
        json.dump(o, open(os.path.join(a.out, f"rank_dependence_{label}_b{b}.json"), "w"), indent=1)
        print(f"{label} b={b}: n={o['n_modal_K']} K={o['K']} f_ind={o['composition_no_coefficients']['f_ind']} rho={o['segment_cross_rank_correlation']['rho_variance_weighted']}")
    open(os.path.join(a.out, "RANK-DEPENDENCE.md"), "w").write(md(res))


if __name__ == "__main__":
    main()
