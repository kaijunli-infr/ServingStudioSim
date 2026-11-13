"""Phase 1C-0 uncertainty audit of the frozen Phase 1B predictions (CPU-only; no refit, no band change).

Per pass, role and class: n, signed bias, median / p90 absolute error, empirical p10-p90 coverage, mean and median
interval width, fraction below / above the interval, and the variation of these across random seeds (pred-mc,
32 draws, seeds 0..S-1). Then the quadrant interval displacement is decomposed into (a) per-(rank, segment) draws
composed through the barrier, (b) the common-mode dispersion factor, (c) the MoE routing band, and compared with
(d) the measured first-collective rank-arrival spread and (e) the per-pass shift of the measured wall distribution
relative to the Phase 1A tables (distribution shift).

Usage (tray14, gb300 venv): python uncertainty/reliability.py --tables <tables> [--seeds 5]
Writes uncertainty/reliability.json and uncertainty/RELIABILITY.md.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P1C = os.path.abspath(os.path.join(HERE, ".."))
P1B = os.path.abspath(os.path.join(P1C, "..", "phase1b"))
P1A = os.path.abspath(os.path.join(P1C, "..", "phase1a"))
sys.path.insert(0, P1B)
import seqsim  # noqa: E402,F401
from costmodel import partition as part  # noqa: E402
from costmodel.compose import critical_path_batch  # noqa: E402
from costmodel.data import NRANK, PassData  # noqa: E402
from costmodel.decode import _sample  # noqa: E402
from seqsim.engine import Engine, measured_from_pass  # noqa: E402
from seqsim.evaluate import by, cls_of, rows_from  # noqa: E402
from seqsim.params import SeqParams  # noqa: E402

R = NRANK
PASSES = ["diverse_nsys_run2", "quadrant_nsys", "diverse_nsys_salvaged"]
DECODE_CLASSES = ["decode b1-2", "decode b3-16", "decode b17-24", "decode b>=25"]


def class_stats(rows) -> dict:
    m = np.array([r["meas_wall_ms"] for r in rows]); p = np.array([r["pred_wall_ms"] for r in rows])
    lo = np.array([r["pred_wall_p10_ms"] for r in rows]); hi = np.array([r["pred_wall_p90_ms"] for r in rows])
    e = p - m
    w = hi - lo
    return dict(n=len(rows), bias_ms=float(e.mean()), bias_pct=float(e.sum() / m.sum() * 100), median_abs_err_ms=float(np.median(np.abs(e))),
                p90_abs_err_ms=float(np.percentile(np.abs(e), 90)), cover_p10_p90_pct=float(((m >= lo) & (m <= hi)).mean() * 100),
                width_mean_ms=float(w.mean()), width_median_ms=float(np.median(w)), width_rel_median_pct=float(np.median(w / np.maximum(m, 1e-9)) * 100),
                below_pct=float((m < lo).mean() * 100), above_pct=float((m > hi).mean() * 100),
                meas_median_ms=float(np.median(m)), pred_median_ms=float(np.median(p)), meas_p10_p90_width_ms=float(np.percentile(m, 90) - np.percentile(m, 10)))


def run(pd, coeffs, params, draws, seed, stochastic, moe_band, recon=None):
    sched = sorted(pd.feats, key=lambda f: f.iteration)
    measured = {f.iteration: m for f in sched if (m := measured_from_pass(pd, f.iteration)) is not None}
    sched = [f for f in sched if f.iteration in measured]
    eng = Engine(coeffs, params, draws=draws, seed=seed, stochastic=stochastic, moe_band=moe_band)
    recs = eng.run(sched, measured=measured, recon=recon)
    return [r for r in rows_from(pd, recs) if not r["excluded"]]


def seed_variation(per_seed: list[dict]) -> dict:
    keys = ["bias_pct", "cover_p10_p90_pct", "width_median_ms", "median_abs_err_ms"]
    out = {}
    for k in keys:
        v = np.array([s[k] for s in per_seed])
        out[k] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0, min=float(v.min()), max=float(v.max()))
    return out


def table_draw_analysis(coeffs, b, S=8192, seed=0):
    """From the Phase 1A decode table alone: median path vs mean of independent per-(rank,segment) draws vs common-mode draws."""
    from costmodel.predict import Predictor
    p, status = Predictor(coeffs).decode._lookup(b)
    rng = np.random.default_rng(seed)
    K = p["K"]
    med = critical_path_batch(np.asarray(p["start"], float)[None], np.asarray(p["seg"], float)[None], np.asarray(p["intrinsic"], float)[None])["wall"][0]
    # independent draws (what the engine does)
    u_s = rng.random((S, R)); u_g = rng.random((S, R, K + 1)); u_i = rng.random((S, K))
    ind = critical_path_batch(_sample(p["start_q"], u_s), _sample(p["seg_q"], u_g), _sample(p["intrinsic_q"], u_i))["wall"]
    # common-mode across ranks (same quantile on every rank per segment)
    u_g2 = np.repeat(rng.random((S, 1, K + 1)), R, axis=1); u_s2 = np.repeat(rng.random((S, 1)), R, axis=1)
    com = critical_path_batch(_sample(p["start_q"], u_s2), _sample(p["seg_q"], u_g2), _sample(p["intrinsic_q"], u_i))["wall"]
    # per-rank independent single-rank path (no barrier): mean of one rank's own path
    seg_r = _sample(p["seg_q"], u_g)
    own = (seg_r.sum(axis=2) + _sample(p["intrinsic_q"], u_i).sum(axis=1)[:, None])
    return dict(b=int(b), status=status, n_table=int(p.get("n", 0)), K=int(K), median_path_ms=float(med / 1e6),
                independent_mean_ms=float(ind.mean() / 1e6), independent_median_ms=float(np.median(ind) / 1e6),
                independent_p10_p90_ms=[float(np.percentile(ind, 10) / 1e6), float(np.percentile(ind, 90) / 1e6)],
                common_mode_mean_ms=float(com.mean() / 1e6), common_mode_median_ms=float(np.median(com) / 1e6),
                common_mode_p10_p90_ms=[float(np.percentile(com, 10) / 1e6), float(np.percentile(com, 90) / 1e6)],
                single_rank_path_mean_ms=float(own.mean() / 1e6), table_wall_median_ms=float(p.get("wall_median", np.nan) / 1e6),
                table_wall_mean_ms=float(p.get("wall_mean", np.nan) / 1e6), table_passes=p.get("passes"))


def arrival_spread(pd, b, n_max=300):
    """Measured first-collective arrival spread across ranks (max-min of start+seg0) for decode iterations with batch b."""
    sp = []
    for f in pd.feats:
        if f.stage != "decode" or f.b != b or f.excluded:
            continue
        seg = pd.segments(f.iteration)
        if seg is None:
            continue
        start_rel, seg_arr, intr, _ = seg
        arr0 = np.asarray(start_rel, float) + np.asarray(seg_arr, float)[:, 0]
        sp.append(arr0.max() - arr0.min())
        if len(sp) >= n_max:
            break
    return dict(n=len(sp), median_ms=float(np.median(sp) / 1e6) if sp else None, p90_ms=float(np.percentile(sp, 90) / 1e6) if sp else None)


def measured_wall_by_b(pd, b):
    w = np.array([f.wall_ns for f in pd.feats if f.stage == "decode" and f.b == b and not f.excluded], float) / 1e6
    if len(w) == 0:
        return None
    return dict(n=int(len(w)), median_ms=float(np.median(w)), mean_ms=float(w.mean()), p10_ms=float(np.percentile(w, 10)), p90_ms=float(np.percentile(w, 90)))


def md_table(rows, cols, fmt="{:.3f}"):
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        L.append("| " + " | ".join(fmt.format(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--draws", type=int, default=32)
    a = ap.parse_args()
    t_all = time.time()
    partition = part.load(os.path.join(P1A, "partition"))
    coeffs = json.load(open(os.path.join(P1A, "out", "coefficients.json")))
    params = SeqParams.load(os.path.join(P1B, "out", "params.json"))
    out = dict(seeds=a.seeds, draws=a.draws, passes={}, displacement={}, table_draws={}, notes=[])
    passes = {lab: PassData.load(lab, os.path.join(a.tables, lab), partition) for lab in PASSES}

    # ---- reliability by class and seed --------------------------------------------------------------
    for lab, pd in passes.items():
        t = time.time()
        det = run(pd, coeffs, params, 1, 0, False, False)
        det_stats = {k: class_stats(v) for k, v in by(det, lambda r: (r["role"], r["cls"])).items()}
        per_seed = {}
        for seed in range(a.seeds):
            rows = run(pd, coeffs, params, a.draws, seed, True, True)
            for k, v in by(rows, lambda r: (r["role"], r["cls"])).items():
                per_seed.setdefault(k, []).append(class_stats(v))
        classes = {}
        for k, lst in per_seed.items():
            classes[" | ".join(k)] = dict(pred_det=det_stats[k], pred_mc_seed0=lst[0], seed_variation=seed_variation(lst), per_seed=lst)
        out["passes"][lab] = dict(seconds=round(time.time() - t, 1), classes=classes)
        print(f"[{lab}] {time.time() - t:.0f}s", flush=True)

    # ---- displacement decomposition on quadrant (and run2 b3-16 for contrast) --------------------------
    p_nodisp = copy.deepcopy(params); p_nodisp.disp_q = {}
    variants = {
        "V0 median path (pred-det)": dict(params=params, stochastic=False, moe_band=False),
        "V1 per-(rank,segment) draws only": dict(params=p_nodisp, stochastic=True, moe_band=False),
        "V2 draws + common-mode dispersion": dict(params=params, stochastic=True, moe_band=False),
        "V3 draws + dispersion + MoE band (= pred-mc)": dict(params=params, stochastic=True, moe_band=True),
        "V4 draws + MoE band, no dispersion": dict(params=p_nodisp, stochastic=True, moe_band=True),
    }
    for lab in ("quadrant_nsys", "diverse_nsys_run2"):
        pd = passes[lab]
        dec = {}
        for name, v in variants.items():
            rows = run(pd, coeffs, params if v["params"] is params else v["params"], 1 if not v["stochastic"] else a.draws, 0, v["stochastic"], v["moe_band"])
            for k, g in by(rows, lambda r: (r["role"], r["cls"])).items():
                if k[1] in DECODE_CLASSES:
                    dec.setdefault(" | ".join(k), {})[name] = class_stats(g)
        out["displacement"][lab] = dec
        print(f"[displacement {lab}]", flush=True)

    # ---- table-only draw analysis, arrival spread, distribution shift -----------------------------------
    bs = sorted({f.b for pd in passes.values() for f in pd.feats if f.stage == "decode" and not f.excluded})
    for b in bs:
        try:
            ta = table_draw_analysis(coeffs, b)
        except Exception as e:  # b outside the tables
            ta = dict(b=int(b), error=str(e))
        ta["measured"] = {lab: measured_wall_by_b(pd, b) for lab, pd in passes.items()}
        ta["arrival_spread"] = {lab: arrival_spread(pd, b) for lab, pd in passes.items()}
        out["table_draws"][str(b)] = ta
    # MoE band contribution by b
    eng = Engine(coeffs, params, draws=1, stochastic=False, moe_band=True)
    moe = {}
    for b in bs:
        try:
            pool = eng._decode_pool(b)
            lo, hi = pool["moe_band"]
            moe[str(b)] = dict(moe_share=pool["moe_share"], band_low=lo, band_high=hi, wall_band_rel_pct=float((hi - lo) * pool["moe_share"] * 100))
        except Exception as e:
            moe[str(b)] = dict(error=str(e))
    out["moe_band"] = moe
    out["seconds"] = round(time.time() - t_all, 1)
    json.dump(out, open(os.path.join(HERE, "reliability.json"), "w"), indent=1, default=float)

    # ---- markdown ---------------------------------------------------------------------------------------
    L = ["# Reliability of the Phase 1B predictions (uncertainty audit, Phase 1C-0)", "",
         f"pred-mc with {a.draws} draws, seeds 0..{a.seeds - 1}; frozen coefficients/partition/params (hashes in ../repro/MANIFEST.json). "
         "Interval = p10-p90 of the draws. Errors are predicted mean minus measured. No parameter was changed.", ""]
    cols = ["class", "n", "bias_pct", "median_abs_err_ms", "p90_abs_err_ms", "cover_p10_p90_pct", "width_mean_ms", "width_median_ms", "below_pct", "above_pct",
            "bias_pct_seed_std", "cover_seed_std", "cover_seed_min", "cover_seed_max"]
    for lab, P in out["passes"].items():
        L += [f"## {lab}", "", "### pred-mc (seed 0) with seed variation", ""]
        rows = []
        for k, c in P["classes"].items():
            s0 = c["pred_mc_seed0"]; sv = c["seed_variation"]
            rows.append(dict(**{"class": k}, n=s0["n"], bias_pct=s0["bias_pct"], median_abs_err_ms=s0["median_abs_err_ms"], p90_abs_err_ms=s0["p90_abs_err_ms"],
                             cover_p10_p90_pct=s0["cover_p10_p90_pct"], width_mean_ms=s0["width_mean_ms"], width_median_ms=s0["width_median_ms"],
                             below_pct=s0["below_pct"], above_pct=s0["above_pct"], bias_pct_seed_std=sv["bias_pct"]["std"], cover_seed_std=sv["cover_p10_p90_pct"]["std"],
                             cover_seed_min=sv["cover_p10_p90_pct"]["min"], cover_seed_max=sv["cover_p10_p90_pct"]["max"]))
        L += md_table(rows, cols) + ["", "### pred-det (median path, no interval)", ""]
        rows = [dict(**{"class": k}, n=c["pred_det"]["n"], bias_pct=c["pred_det"]["bias_pct"], median_abs_err_ms=c["pred_det"]["median_abs_err_ms"],
                     p90_abs_err_ms=c["pred_det"]["p90_abs_err_ms"]) for k, c in P["classes"].items()]
        L += md_table(rows, ["class", "n", "bias_pct", "median_abs_err_ms", "p90_abs_err_ms"]) + [""]
    L += ["## Interval displacement decomposition (decode classes)", "",
          "Variants differ only in which stochastic component is switched on; V0 is the median path, V3 is the reported pred-mc.", ""]
    for lab, dec in out["displacement"].items():
        L += [f"### {lab}", ""]
        rows = []
        for k, vs in dec.items():
            for name, s in vs.items():
                rows.append({"class": k, "variant": name, "n": s["n"], "bias_pct": s["bias_pct"], "pred_median_ms": s["pred_median_ms"], "meas_median_ms": s["meas_median_ms"],
                             "cover_p10_p90_pct": s["cover_p10_p90_pct"], "width_median_ms": s["width_median_ms"], "below_pct": s["below_pct"], "above_pct": s["above_pct"]})
        L += md_table(rows, ["class", "variant", "n", "bias_pct", "pred_median_ms", "meas_median_ms", "cover_p10_p90_pct", "width_median_ms", "below_pct", "above_pct"]) + [""]
    L += ["## Phase 1A decode table: median path vs independent vs common-mode draws, measured walls and first-collective arrival spread by batch size", "",
          "independent = per-(rank, segment) quantile draws (engine behaviour); common-mode = same quantile on all ranks. Measured columns are the per-pass wall distributions of all non-excluded decode iterations of that b.", ""]
    rows = []
    for b, ta in out["table_draws"].items():
        if "error" in ta:
            continue
        row = {"b": ta["b"], "n_table": ta["n_table"], "median_path_ms": ta["median_path_ms"], "indep_mean_ms": ta["independent_mean_ms"], "common_mean_ms": ta["common_mode_mean_ms"],
               "indep_p10_p90_width_ms": ta["independent_p10_p90_ms"][1] - ta["independent_p10_p90_ms"][0], "common_p10_p90_width_ms": ta["common_mode_p10_p90_ms"][1] - ta["common_mode_p10_p90_ms"][0]}
        for lab in PASSES:
            mw = ta["measured"][lab]
            short = {"diverse_nsys_run2": "run2", "quadrant_nsys": "quad", "diverse_nsys_salvaged": "salv"}[lab]
            row[f"{short}_meas_median_ms"] = mw["median_ms"] if mw else float("nan")
            row[f"{short}_meas_p10_p90_width_ms"] = (mw["p90_ms"] - mw["p10_ms"]) if mw else float("nan")
            row[f"{short}_n"] = mw["n"] if mw else 0
            sp = ta["arrival_spread"][lab]
            row[f"{short}_arrival_spread_median_ms"] = sp["median_ms"] if sp["median_ms"] is not None else float("nan")
        row["moe_wall_band_pct"] = out["moe_band"].get(b, {}).get("wall_band_rel_pct", float("nan"))
        rows.append(row)
    cols = ["b", "n_table", "median_path_ms", "indep_mean_ms", "common_mean_ms", "indep_p10_p90_width_ms", "common_p10_p90_width_ms",
            "run2_n", "run2_meas_median_ms", "run2_meas_p10_p90_width_ms", "quad_n", "quad_meas_median_ms", "quad_meas_p10_p90_width_ms", "salv_n", "salv_meas_median_ms",
            "run2_arrival_spread_median_ms", "quad_arrival_spread_median_ms", "moe_wall_band_pct"]
    L += md_table(rows, cols) + [""]
    open(os.path.join(HERE, "RELIABILITY.md"), "w").write("\n".join(L))
    print("done", out["seconds"], "s")


if __name__ == "__main__":
    main()
