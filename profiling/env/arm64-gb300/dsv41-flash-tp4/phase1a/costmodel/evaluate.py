"""Phase 1A evaluation: calibration vs held-out accuracy per metric, residuals by class, extrapolation
and unidentifiable cases, consistency checks (composition rule, long-context slope, MoE range).

Usage:
  python -m costmodel.evaluate --tables <dir> --partition <dir> --coeffs <coefficients.json> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import groups as G
from .compose import reconstruct
from .data import NRANK, Feat, PassData
from .longctx import chunk_predict
from .partition import load as load_partition
from .predict import Predictor

PASSES = ("diverse_nsys_run2", "quadrant_nsys", "diverse_nsys_salvaged")
APE_MIN_MS = 0.05


def metrics(meas, pred) -> dict:
    meas = np.asarray(meas, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ok = np.isfinite(pred) & np.isfinite(meas)
    meas, pred = meas[ok], pred[ok]
    n = len(meas)
    if n == 0:
        return dict(n=0)
    err = pred - meas
    ae = np.abs(err)
    d = meas >= APE_MIN_MS
    ape = ae[d] / meas[d] * 100 if d.any() else np.array([])
    out = dict(n=int(n), median_err_ms=float(np.median(err)), median_abs_err_ms=float(np.median(ae)), p90_abs_err_ms=float(np.percentile(ae, 90)),
               bias_ms=float(err.mean()), worst_abs_err_ms=float(ae.max()), meas_median_ms=float(np.median(meas)))
    out["median_ape_pct"] = float(np.median(ape)) if len(ape) else None
    out["p90_ape_pct"] = float(np.percentile(ape, 90)) if len(ape) else None
    out["worst_ape_pct"] = float(ape.max()) if len(ape) else None
    out["bias_pct"] = float(err.sum() / meas.sum() * 100) if meas.sum() > 0 else None
    return out


def class_group(f: Feat) -> str:
    if f.stage == "decode":
        return f"decode b={f.b}"
    if f.is_chunk16k:
        return f"chunk16k kv0={round(f.kv0 / 16384) * 16}k"
    return f"mixed-{f.mixed_kind}" + ("" if len(f.chunks) <= 1 else " multi-chunk")


def kv_band(f: Feat) -> str:
    return "kv<32k" if f.kv_max < 32768 else ("kv32-96k" if f.kv_max < 98304 else "kv>=96k")


def build_records(pd: PassData, pred: Predictor, role_override: str | None) -> list[dict]:
    recs = []
    for f in pd.feats:
        if f.excluded:
            continue
        p = pred.predict(f)
        rows = pd.rank_rows(f.iteration)
        ir = pd.ir
        gm = pd.group_sums(f.iteration) / 1e6
        busy_m = ir["busy_ns"][rows] / 1e6
        idle_m = ir["idle_ns"][rows] / 1e6
        last_end = ir["last_end_ns"][rows]
        fwd_m = ir["fwd_window_ns"][rows] / 1e6
        rec = dict(
            label=pd.label, it=f.iteration, role=role_override or f.role, cls=f.cls, group=class_group(f), stage=f.stage, regime=f.regime,
            status=p.get("status", "n/a"), kv_status=p.get("kv_status", ""), b=f.b, kv_max=f.kv_max, kv_sum=f.kv_sum, kv_band=kv_band(f), P=f.P,
            kv0=f.kv0, is_chunk=f.is_chunk16k, chain=f.chain, mixed_kind=f.mixed_kind,
            wall_m=f.wall_ns / 1e6, wall_p=p["wall_ns"] / 1e6 if np.isfinite(p.get("wall_ns", np.nan)) else np.nan,
            wall_p0=(p.get("wall_standalone_ns", p.get("wall_det_ns", np.nan)) / 1e6), lead_ms=f.lead_ns / 1e6, prev_stage=f.prev_stage,
            ring_m=bool(ir["c_nccl_allreduce_count"][rows][0] > 0), ring_p=(f.mixed_kind == "ring"),
            slowest_m=int(np.argmax(last_end)), slowest_p=p.get("slowest_rank", -1),
            busy_m=busy_m, idle_m=idle_m, groups_m=gm, fwd_m=fwd_m,
            compute_m=ir["compute_union_ns"][rows] / 1e6, comm_m=ir["comm_union_ns"][rows] / 1e6,
        )
        if f.stage == "decode":
            rec.update(busy_p=p["rank_busy_ns"] / 1e6, idle_p=p["rank_idle_ns"] / 1e6, groups_p=p["groups_ns"] / 1e6,
                       compute_p=p["rank_compute_union_ns"] / 1e6, comm_p=p["rank_comm_union_ns"] / 1e6)
            rec["groups_p"][:, G.GROUP_INDEX["comm"]] = rec["comm_p"]
        elif f.mixed_kind == "graph":
            rec.update(busy_p=np.asarray(p.get("rank_busy_ns", np.full(NRANK, np.nan))) / 1e6, idle_p=np.full(NRANK, np.nan), groups_p=None,
                       host_path_p=p.get("host_path_ns", np.nan) / 1e6, gpu_path_p=p.get("gpu_path_ns", np.nan) / 1e6)
        else:
            gp = p["groups_ns"] / 1e6
            gp[:, G.GROUP_INDEX["comm"]] = p["comm_intrinsic_ns"] / 1e6 + p["rank_wait_ns"] / 1e6
            rec.update(busy_p=p["rank_gpu_path_ns"] / 1e6, idle_p=p["rank_idle_ns"] / 1e6, groups_p=gp, compute_p=p["rank_compute_union_ns"] / 1e6,
                       host_path_p=p["host_path_ns"] / 1e6, gpu_path_p=p["gpu_path_ns"] / 1e6, bound_p=p["bound"], fwd_p=p["host_fwd_ns"] / 1e6,
                       launches_p=p["launches"], launches_m=float(ir["fwd_launch_n"][rows][0]),
                       bound_m="gpu" if rec["compute_m"].max() + (ir["comm_sum_ns"][rows].min() / 1e6) > fwd_m.max() else "host",
                       prefill_status=p["prefill_status"], host_status=p["host_status"])
        recs.append(rec)
    return recs


def by_key(recs, keyf):
    out = {}
    for r in recs:
        out.setdefault(keyf(r), []).append(r)
    return out


def table_wall(recs, keyf) -> list[dict]:
    rows = []
    for key, rs in sorted(by_key(recs, keyf).items(), key=lambda kv: str(kv[0])):
        m = metrics([r["wall_m"] for r in rs], [r["wall_p"] for r in rs])
        m0 = metrics([r["wall_m"] for r in rs], [r["wall_p0"] for r in rs])
        m.update(alt_median_err_ms=m0.get("median_err_ms"), alt_p90_ape_pct=m0.get("p90_ape_pct"), alt_bias_pct=m0.get("bias_pct"))
        m.update(key=key, slowest_rank_match_pct=float(np.mean([r["slowest_m"] == r["slowest_p"] for r in rs if r["slowest_p"] >= 0]) * 100) if any(r["slowest_p"] >= 0 for r in rs) else None)
        rows.append(m)
    return rows


def table_groups(recs, keyf) -> list[dict]:
    rows = []
    for key, rs in sorted(by_key(recs, keyf).items(), key=lambda kv: str(kv[0])):
        rs = [r for r in rs if r.get("groups_p") is not None]
        if not rs:
            continue
        for gi, g in enumerate(G.GROUPS):
            meas = [r["groups_m"][:, gi].max() for r in rs]
            pred = [r["groups_p"][:, gi].max() for r in rs]
            m = metrics(meas, pred)
            m.update(key=key, group=g)
            rows.append(m)
    return rows


def table_rank(recs, keyf, what: str) -> list[dict]:
    rows = []
    for key, rs in sorted(by_key(recs, keyf).items(), key=lambda kv: str(kv[0])):
        rs = [r for r in rs if r.get(f"{what}_p") is not None and np.all(np.isfinite(r[f"{what}_p"]))]
        if not rs:
            continue
        for rank in range(NRANK):
            m = metrics([r[f"{what}_m"][rank] for r in rs], [r[f"{what}_p"][rank] for r in rs])
            m.update(key=key, rank=rank)
            rows.append(m)
    return rows


def kv_independence(recs) -> list[dict]:
    rows = []
    for key, rs in sorted(by_key([r for r in recs if r["stage"] == "decode"], lambda r: (r["label"], r["role"], r["regime"])).items()):
        if len(rs) < 30:
            continue
        resid = np.array([r["wall_p"] - r["wall_m"] for r in rs])
        for var in ("kv_sum", "kv_max"):
            x = np.array([r[var] for r in rs], dtype=np.float64) / 1e5
            if x.std() == 0:
                continue
            X = np.stack([np.ones_like(x), x], axis=1)
            coef, *_ = np.linalg.lstsq(X, resid, rcond=None)
            res = resid - X @ coef
            se = float(np.sqrt(res.var() / max(1, len(x) - 2) / x.var())) if len(x) > 2 else float("nan")
            rows.append(dict(label=key[0], role=key[1], regime=key[2], n=len(rs), variable=var, slope_ms_per_100k=float(coef[1]), slope_se=se,
                             x_range_100k=(float(x.min()), float(x.max())), resid_std_ms=float(resid.std())))
    return rows


def reconstruction_check(pd: PassData, recs, limit=400) -> dict:
    out = {}
    for grp in ("decode", "chunk"):
        sel = [r for r in recs if (r["stage"] == "decode") == (grp == "decode") and (grp == "decode" or r["is_chunk"])]
        sel = sel[:: max(1, len(sel) // limit)][:limit]
        errs, sums = [], []
        for r in sel:
            s = pd.segments(r["it"])
            if s is None:
                continue
            start, seg, intr, dur = s
            cp = reconstruct(start, seg, intr)
            errs.append((cp.wall - r["wall_m"] * 1e6) / 1e6)
            sums.append(((seg.sum(axis=1) + intr.sum()).sum()) / 1e6 / (r["wall_m"]))
        if errs:
            e = np.array(errs)
            out[grp] = dict(n=len(e), median_err_ms=float(np.median(e)), p90_abs_err_ms=float(np.percentile(np.abs(e), 90)), worst_abs_err_ms=float(np.abs(e).max()),
                            median_wall_ms=float(np.median([r["wall_m"] for r in sel])), sum_over_ranks_over_wall_median=float(np.median(sums)))
    return out


def longctx_eval(recs_all, coeffs) -> dict:
    chunk = coeffs["prefill"]["chunk"]
    out = dict(calibration_fit=chunk, heldout=[], slopes={})
    held = [r for r in recs_all if r["is_chunk"] and r["role"] == "heldout" and r["label"] != "diverse_nsys_salvaged"]
    for key, rs in sorted(by_key(held, lambda r: round(r["kv0"] / 16384) * 16).items()):
        wm = np.array([r["wall_m"] for r in rs])
        wp = np.array([chunk_predict(chunk, r["kv0"], "wall")[0] for r in rs])
        im = np.array([r["groups_m"][:, G.GROUP_INDEX["indexer"]].max() for r in rs])
        ip = np.array([chunk["groups"]["indexer"]["a_ms"] + chunk["groups"]["indexer"]["s_ms_per_1k"] * r["kv0"] / 1e3 for r in rs])
        st = sorted({chunk_predict(chunk, r["kv0"], "wall")[1] for r in rs})
        out["heldout"].append(dict(kv0_bucket_k=key, n=len(rs), chains=sorted({r["chain"] for r in rs}), status=st,
                                   wall=metrics(wm, wp), indexer=metrics(im, ip), model_wall=metrics(wm, [r["wall_p"] for r in rs])))
    # slope fitted on held-out only, per chain and pooled; and on the salvaged pass
    def slope(rs, what):
        if len(rs) < 3:
            return None
        kv = np.array([r["kv0"] for r in rs]) / 1e3
        y = np.array([r["wall_m"] if what == "wall" else r["groups_m"][:, G.GROUP_INDEX["indexer"]].max() for r in rs])
        X = np.stack([np.ones_like(kv), kv], axis=1)
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        return dict(n=len(rs), a_ms=float(c[0]), s_ms_per_1k=float(c[1]), kv0_range=(int(kv.min() * 1e3), int(kv.max() * 1e3)))
    for what in ("wall", "indexer"):
        out["slopes"][f"heldout_pooled_{what}"] = slope(held, what)
        for ch, rs in sorted(by_key(held, lambda r: r["chain"]).items()):
            out["slopes"][f"heldout_chain{ch}_{what}"] = slope(rs, what)
        salv = [r for r in recs_all if r["is_chunk"] and r["label"] == "diverse_nsys_salvaged"]
        out["slopes"][f"salvaged_all_{what}"] = slope(salv, what)
        for ch, rs in sorted(by_key(salv, lambda r: r["chain"]).items()):
            if len(rs) >= 4:
                out["slopes"][f"salvaged_chain{ch}_{what}"] = slope(rs, what)
    for g, p in chunk.get("groups", {}).items():
        out["slopes"][f"calibration_group_{g}"] = p
    return out


def moe_eval(recs_all, pred: Predictor) -> list[dict]:
    rows = []
    dec = [r for r in recs_all if r["stage"] == "decode" and r["label"] != "diverse_nsys_salvaged"]
    for b, rs in sorted(by_key(dec, lambda r: r["b"]).items()):
        p = pred.moe.predict(b)
        if "central_ns" not in p:
            continue
        for role in ("calibration", "heldout"):
            sel = [r for r in rs if r["role"] == role]
            if not sel:
                continue
            m = np.array([np.median(r["groups_m"][:, G.GROUP_INDEX["moe"]]) for r in sel])
            rows.append(dict(b=b, role=role, n=len(sel), status=p["status"], meas_p10_ms=float(np.percentile(m, 10)), meas_p50_ms=float(np.median(m)),
                             meas_p90_ms=float(np.percentile(m, 90)), pred_central_ms=p["central_ns"] / 1e6, pred_low_ms=p["low_ns"] / 1e6, pred_high_ms=p["high_ns"] / 1e6,
                             E_mean=p["E_mean"], E_p10=p["E_p10"], E_p90=p["E_p90"], heldout_routing_central_ms=p.get("heldout_routing_central_ns", np.nan) / 1e6,
                             heldout_E_mean=p.get("heldout_E_mean"), central_err_pct=float((p["central_ns"] / 1e6 - np.median(m)) / np.median(m) * 100),
                             measured_spread_covered=bool(p["low_ns"] / 1e6 <= np.median(m) <= p["high_ns"] / 1e6)))
    return rows


def host_path_eval(recs) -> list[dict]:
    rows = []
    mixed = [r for r in recs if r["stage"] == "mixed" and r.get("fwd_p") is not None]
    for key, rs in sorted(by_key(mixed, lambda r: (r["role"], r["mixed_kind"], r["bound_m"])).items()):
        fm = [r["fwd_m"].max() for r in rs]
        fp = [r["fwd_p"].max() for r in rs]
        m = metrics(fm, fp)
        m.update(role=key[0], kind=key[1], measured_bound=key[2], predicted_bound_match_pct=float(np.mean([r["bound_p"] == r["bound_m"] for r in rs]) * 100),
                 launches=metrics([r["launches_m"] for r in rs], [r["launches_p"] for r in rs]), wall=metrics([r["wall_m"] for r in rs], [r["wall_p"] for r in rs]),
                 slowest_host_rank_counts={str(k): int(v) for k, v in zip(*np.unique([int(np.argmax(r["fwd_m"])) for r in rs], return_counts=True))})
        rows.append(m)
    return rows


def mixed_table(recs) -> list[dict]:
    rows = []
    mixed = [r for r in recs if r["stage"] == "mixed"]
    for key, rs in sorted(by_key(mixed, lambda r: (r["role"], r["mixed_kind"], r.get("bound_m", "n/a"))).items()):
        m = metrics([r["wall_m"] for r in rs], [r["wall_p"] for r in rs])
        m0 = metrics([r["wall_m"] for r in rs], [r["wall_p0"] for r in rs])
        rows.append(dict(role=key[0], kind=key[1], measured_bound=key[2], n=len(rs), lead_aware=m, standalone=m0,
                         lead_ms_median=float(np.median([r["lead_ms"] for r in rs])), lead_ms_max=float(max(r["lead_ms"] for r in rs)),
                         comm_kind_match_pct=float(np.mean([r["ring_m"] == r["ring_p"] for r in rs]) * 100),
                         prev_stage_counts={k: len(v) for k, v in by_key(rs, lambda r: r["prev_stage"] or "none").items()}))
    return rows


def full_workload(recs, pd: PassData) -> dict:
    it = pd.it
    total_span_ms = (it["t1_ns"].max() - it["t0_ns"].min()) / 1e6
    excluded_ms = float(it["wall_ns"][it["excluded"]].sum() / 1e6)
    out = dict(measured_span_ms=float(total_span_ms), excluded_iterations=int(it["excluded"].sum()), excluded_wall_ms=excluded_ms,
               sum_measured_walls_all_ms=float(it["wall_ns"].sum() / 1e6))
    for role in ("all", "calibration", "heldout", "crosscheck"):
        rs = recs if role == "all" else [r for r in recs if r["role"] == role]
        rs = [r for r in rs if np.isfinite(r["wall_p"])]
        if not rs:
            continue
        sm = sum(r["wall_m"] for r in rs)
        sp = sum(r["wall_p"] for r in rs)
        out[role] = dict(n=len(rs), sum_measured_ms=sm, sum_predicted_ms=sp, err_pct=(sp - sm) / sm * 100,
                         by_stage={s: dict(n=len(x), meas=sum(r["wall_m"] for r in x), pred=sum(r["wall_p"] for r in x)) for s, x in by_key(rs, lambda r: r["group"].split(" ")[0]).items()})
    out["predicted_coverage_of_span_pct"] = float(sum(r["wall_m"] for r in recs if np.isfinite(r["wall_p"])) / total_span_ms * 100)
    return out


def md_table(rows: list[dict], cols: list[str], fmt=None) -> str:
    if not rows:
        return "_none_\n"
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                cells.append(fmt.get(c, "{:.3f}").format(v) if np.isfinite(v) else "nan")
            elif v is None:
                cells.append("—")
            else:
                cells.append(str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--coeffs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    part = load_partition(a.partition)
    coeffs = json.load(open(a.coeffs))
    pred = Predictor(coeffs)
    results = dict(coefficients_sha256=open(os.path.join(os.path.dirname(a.coeffs), "COEFFICIENTS.sha256")).read().split()[0] if os.path.exists(os.path.join(os.path.dirname(a.coeffs), "COEFFICIENTS.sha256")) else None,
                   partition_sha256=part.part["_sha256"], passes={})
    all_recs = []
    for lab in PASSES:
        d = os.path.join(a.tables, lab)
        if not os.path.isdir(d):
            continue
        pd = PassData.load(lab, d, part)
        recs = build_records(pd, pred, "crosscheck" if lab == "diverse_nsys_salvaged" else None)
        all_recs.extend(recs)
        R = dict(n_records=len(recs), n_excluded=int(pd.it["excluded"].sum()))
        roles = sorted({r["role"] for r in recs})
        R["wall_by_role_group"] = table_wall(recs, lambda r: f"{r['role']} | {r['group']}")
        R["wall_by_role_regime"] = table_wall(recs, lambda r: f"{r['role']} | {r['regime']}")
        R["wall_by_role_kvband"] = table_wall([r for r in recs if r["stage"] == "decode"], lambda r: f"{r['role']} | {r['regime']} | {r['kv_band']}")
        R["wall_by_status"] = table_wall(recs, lambda r: f"{r['role']} | {r['status']} | {r['kv_status'] or '-'}")
        R["groups_by_role_regime"] = table_groups(recs, lambda r: f"{r['role']} | {r['regime']}" + (" | chunk16k" if r["is_chunk"] else ""))
        R["busy_by_role_regime"] = table_rank(recs, lambda r: f"{r['role']} | {r['regime']}", "busy")
        R["idle_by_role_regime"] = table_rank([r for r in recs if r["stage"] == "decode" or r["mixed_kind"] != "graph"], lambda r: f"{r['role']} | {r['regime']}", "idle")
        R["kv_independence"] = kv_independence(recs)
        R["reconstruction"] = reconstruction_check(pd, [r for r in recs if r["role"] in ("heldout", "crosscheck")])
        R["host_path"] = host_path_eval(recs)
        R["mixed"] = mixed_table(recs)
        R["full_workload"] = full_workload(recs, pd)
        R["status_counts"] = {k: len(v) for k, v in sorted(by_key(recs, lambda r: f"{r['role']} | {r['status']}").items())}
        results["passes"][lab] = R
        print(f"{lab}: {len(recs)} records evaluated, {time.time()-t0:.0f}s", flush=True)
    results["longctx"] = longctx_eval(all_recs, coeffs)
    results["moe"] = moe_eval(all_recs, pred)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, np.integer) else str(o)))
    write_markdown(results, coeffs, os.path.join(a.out, "RESULTS.md"))
    print(f"wrote {a.out}/results.json and RESULTS.md in {time.time()-t0:.0f}s", flush=True)


def write_markdown(results: dict, coeffs: dict, path: str):
    W = ["# Phase 1A calibration and held-out results\n",
         f"coefficients sha256 `{results.get('coefficients_sha256')}`; partition sha256 `{results.get('partition_sha256')}`; git rev `{coeffs.get('git_rev')}`\n",
         "Error metrics: n, median signed error (ms), median |error| (ms), p90 |error| (ms), p90 APE (%), bias (mean signed error, ms, and % of Σ measured), worst |error| (ms). "
         f"APE is defined where the measured value is >= {APE_MIN_MS} ms.  Roles: calibration (fitted), heldout (never fitted), crosscheck (salvaged original capture, never fitted).\n"]
    cols = ["key", "n", "median_err_ms", "median_abs_err_ms", "p90_abs_err_ms", "p90_ape_pct", "bias_ms", "bias_pct", "worst_abs_err_ms", "slowest_rank_match_pct", "alt_median_err_ms", "alt_p90_ape_pct", "alt_bias_pct"]
    fmt = {"p90_ape_pct": "{:.1f}", "bias_pct": "{:+.2f}", "slowest_rank_match_pct": "{:.0f}", "alt_p90_ape_pct": "{:.1f}", "alt_bias_pct": "{:+.2f}"}
    W.append("`alt_*` columns: the alternative composition — deterministic median path for decode (vs the stochastic expectation used as the prediction), "
             "standalone max(host, GPU) with lead = 0 for mixed iterations (vs the lead-aware prediction).\n")
    for lab, R in results["passes"].items():
        W.append(f"\n## Pass `{lab}` ({R['n_records']} evaluated iterations, {R['n_excluded']} excluded)\n")
        W.append("\n### Slowest-rank iteration latency by role and class\n" + md_table(R["wall_by_role_group"], cols, fmt))
        W.append("\n### Slowest-rank iteration latency by role and regime\n" + md_table(R["wall_by_role_regime"], cols, fmt))
        if R["wall_by_role_kvband"]:
            W.append("\n### Decode latency by role, regime and KV band\n" + md_table(R["wall_by_role_kvband"], cols, fmt))
        W.append("\n### Latency by prediction status (extrapolation / inferred cases reported separately)\n" + md_table(R["wall_by_status"], cols, fmt))
        gcols = ["key", "group", "n", "median_err_ms", "median_abs_err_ms", "p90_abs_err_ms", "p90_ape_pct", "bias_ms", "worst_abs_err_ms", "meas_median_ms"]
        W.append("\n### Kernel-group accuracy (slowest-rank group time)\n" + md_table(R["groups_by_role_regime"], gcols, fmt))
        W.append("\nNote: in host-bound eager iterations the measured `comm` group is dominated by twoshot kernels spinning while the host launches "
                 "(host-induced wait, 150–190 ms); the model's `comm` is the intrinsic cost plus the GPU-bound wait term, so that row is not a modelling target. "
                 "`comm` for decode = intrinsic + predicted synchronization wait.\n")
        rcols = ["key", "rank", "n", "median_err_ms", "median_abs_err_ms", "p90_abs_err_ms", "p90_ape_pct", "bias_ms", "worst_abs_err_ms", "meas_median_ms"]
        W.append("\n### Per-rank GPU busy accuracy\n" + md_table(R["busy_by_role_regime"], rcols, fmt))
        W.append("\n### Per-rank GPU idle accuracy\n" + md_table(R["idle_by_role_regime"], rcols, fmt))
        W.append("\n### Decode KV-independence test (residual slope vs KV; ms per 100k tokens)\n" + md_table(R["kv_independence"], ["label", "role", "regime", "n", "variable", "slope_ms_per_100k", "slope_se", "x_range_100k", "resid_std_ms"], {"slope_ms_per_100k": "{:+.4f}", "slope_se": "{:.4f}"}))
        W.append("\n### Composition-rule check on held-out iterations (measured own segments → completion = max arrival + intrinsic)\n")
        for k, v in R["reconstruction"].items():
            W.append(f"- {k}: n={v['n']}, median error {v['median_err_ms']:+.3f} ms, p90 |error| {v['p90_abs_err_ms']:.3f} ms, worst {v['worst_abs_err_ms']:.3f} ms "
                     f"(median wall {v['median_wall_ms']:.2f} ms; Σ over ranks / wall = {v['sum_over_ranks_over_wall_median']:.2f})\n")
        if R["host_path"]:
            W.append("\n### Mixed-iteration host critical path (max over ranks of the forward window)\n")
            for h in R["host_path"]:
                W.append(f"- role {h['role']}, kind {h['kind']}, measured bound **{h['measured_bound']}**: n={h['n']}, forward-window median err {h['median_err_ms']:+.1f} ms, "
                         f"p90 APE {h['p90_ape_pct'] if h['p90_ape_pct'] is None else round(h['p90_ape_pct'],1)} %, bound predicted correctly {h['predicted_bound_match_pct']:.0f} %, "
                         f"launch count median err {h['launches']['median_err_ms']:+.0f}, wall median err {h['wall']['median_err_ms']:+.1f} ms (p90 APE {h['wall']['p90_ape_pct'] if h['wall']['p90_ape_pct'] is None else round(h['wall']['p90_ape_pct'],1)} %), slowest host rank counts {h['slowest_host_rank_counts']}\n")
        if R.get("mixed"):
            W.append("\n### Mixed iterations by kind: lead-aware vs standalone composition\n")
            for m in R["mixed"]:
                la, sa = m["lead_aware"], m["standalone"]
                W.append(f"- role {m['role']}, kind {m['kind']}, measured bound **{m['measured_bound']}**, n={m['n']} (previous stage {m['prev_stage_counts']}): "
                         f"lead-aware median err {la['median_err_ms']:+.1f} ms, p90 APE {la['p90_ape_pct'] if la['p90_ape_pct'] is None else round(la['p90_ape_pct'],1)} %, bias {la['bias_pct']:+.1f} %, worst {la['worst_abs_err_ms']:.1f} ms; "
                         f"standalone median err {sa['median_err_ms']:+.1f} ms, p90 APE {sa['p90_ape_pct'] if sa['p90_ape_pct'] is None else round(sa['p90_ape_pct'],1)} %, bias {sa['bias_pct']:+.1f} %; "
                         f"host lead median {m['lead_ms_median']:.1f} ms (max {m['lead_ms_max']:.1f}); all-reduce kind predicted correctly {m['comm_kind_match_pct']:.0f} %\n")
        fw = R["full_workload"]
        W.append(f"\n### Full-workload prediction\nmeasured GPU span {fw['measured_span_ms']/1e3:.2f} s (Σ iteration walls {fw['sum_measured_walls_all_ms']/1e3:.2f} s, "
                 f"{fw['excluded_iterations']} excluded JIT iterations = {fw['excluded_wall_ms']/1e3:.2f} s); predicted iterations cover {fw['predicted_coverage_of_span_pct']:.1f} % of the span.\n")
        for role in ("all", "calibration", "heldout", "crosscheck"):
            if role in fw:
                x = fw[role]
                W.append(f"- {role}: n={x['n']}, Σ measured {x['sum_measured_ms']/1e3:.3f} s, Σ predicted {x['sum_predicted_ms']/1e3:.3f} s, error {x['err_pct']:+.2f} %; by stage: "
                         + ", ".join(f"{s} {v['n']} it. {v['pred']/1e3:.3f}/{v['meas']/1e3:.3f} s" for s, v in x["by_stage"].items()) + "\n")
        W.append("\n### Prediction status counts\n" + "\n".join(f"- {k}: {v}" for k, v in R["status_counts"].items()) + "\n")
    L = results["longctx"]
    W.append("\n## Long-context prefill (16k chunks)\n")
    cf = L["calibration_fit"]
    if cf.get("n"):
        W.append(f"Calibration fit on {cf['n']} clean 16k chunks (kv0 {cf['kv0_min']:.0f}–{cf['kv0_max']:.0f}, chains {cf['chains']}): wall = {cf['wall']['a_ms']:.1f} + {cf['wall']['s_ms_per_1k']:.4f}·kv0_k ms (rmse {cf['wall']['rmse_ms']:.2f}); "
                 f"slowest-rank busy = {cf['busy_slowest']['a_ms']:.1f} + {cf['busy_slowest']['s_ms_per_1k']:.4f}·kv0_k; quadratic term at max kv0 = {cf['wall_quadratic']['q_term_at_max_ms']:+.2f} ms (rmse {cf['wall_quadratic']['rmse_ms']:.2f}).\n")
        W.append(md_table([dict(group=g, **v) for g, v in cf["groups"].items()], ["group", "a_ms", "s_ms_per_1k", "rmse_ms"], {"s_ms_per_1k": "{:+.4f}"}))
    W.append("\n### Held-out chunks by KV offset bucket (chain-level and band hold-outs)\n")
    rows = []
    for h in L["heldout"]:
        rows.append(dict(kv0_bucket_k=h["kv0_bucket_k"], n=h["n"], chains=str(h["chains"]), status=str(h["status"]), wall_median_err_ms=h["wall"]["median_err_ms"], wall_p90_ape=h["wall"]["p90_ape_pct"],
                         wall_worst_ms=h["wall"]["worst_abs_err_ms"], indexer_median_err_ms=h["indexer"]["median_err_ms"], indexer_p90_ape=h["indexer"]["p90_ape_pct"],
                         model_wall_median_err_ms=h["model_wall"]["median_err_ms"], model_wall_p90_ape=h["model_wall"]["p90_ape_pct"]))
    W.append(md_table(rows, ["kv0_bucket_k", "n", "chains", "status", "wall_median_err_ms", "wall_p90_ape", "wall_worst_ms", "indexer_median_err_ms", "indexer_p90_ape", "model_wall_median_err_ms", "model_wall_p90_ape"],
                      {"wall_p90_ape": "{:.1f}", "indexer_p90_ape": "{:.1f}", "model_wall_p90_ape": "{:.1f}"}))
    W.append("\n(`wall`/`indexer` = the dedicated 16k-chunk linear-in-kv0 fit; `model_wall` = the full mixed/prefill component model.)\n")
    W.append("\n### Slopes (ms per 1k KV) fitted independently on held-out and salvaged data\n")
    srows = [dict(fit=k, **v) for k, v in L["slopes"].items() if v and "s_ms_per_1k" in v]
    W.append(md_table(srows, ["fit", "n", "a_ms", "s_ms_per_1k", "kv0_range"], {"s_ms_per_1k": "{:+.4f}"}))
    W.append("\n## MoE statistical model (decode)\n")
    mrows = results["moe"]
    W.append(md_table(mrows, ["b", "role", "n", "status", "meas_p10_ms", "meas_p50_ms", "meas_p90_ms", "pred_central_ms", "pred_low_ms", "pred_high_ms", "central_err_pct", "E_mean", "heldout_E_mean", "heldout_routing_central_ms"],
                      {"central_err_pct": "{:+.1f}", "E_mean": "{:.1f}", "heldout_E_mean": "{:.1f}"}))
    mo = coeffs["moe"]
    W.append(f"\nt_moe(b) = {mo['coef_ns'][0]/1e6:.3f} + {mo['coef_ns'][1]/1e6:.5f}·E(b) + {mo['coef_ns'][2]/1e6:.5f}·b ms; R² {mo['r2']:.3f}, rmse {mo['rmse_ns']/1e6:.3f} ms over {len(mo['fit_rows'])} batch sizes. "
             "Range = p10–p90 of E(b) propagated through β; per-iteration routing is not predicted.\n")
    W.append("\n## Parameter classification\n" + md_table(coeffs["classification"], ["component", "parameter", "status", "evidence"]))
    open(path, "w").write("".join(W))


if __name__ == "__main__":
    main()
