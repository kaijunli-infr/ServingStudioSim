"""Sequence-simulation evaluation against the frozen traces (Phase 1B).

Modes
  recon-gpu  measured per-rank segments and measured host times          -> tests the GPU composition + launch coupling
  recon-seq  measured segments and host durations, host starts from the sequence rules -> tests the inter-iteration rules
  pred-det   fully predicted, deterministic median path (N = 1)
  pred-mc    fully predicted, N stochastic draws (decode segment jitter, ring segment jitter, MoE routing band)

Every table reports n, signed bias (ms and % of the measured sum), median abs error, p90 APE, worst abs error.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from costmodel.data import NRANK, Feat, PassData
from costmodel.evaluate import md_table

from .engine import Engine, IterRecord, measured_from_pass
from .params import SeqParams, kind_of

APE_MIN_MS = 0.05
KV_BANDS_K = (0, 16, 32, 48, 64, 80, 96, 112)


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
    return dict(n=int(n), bias_ms=float(err.mean()), bias_pct=float(err.sum() / meas.sum() * 100) if meas.sum() > 0 else None,
                median_err_ms=float(np.median(err)), median_abs_err_ms=float(np.median(ae)),
                p90_ape_pct=float(np.percentile(ape, 90)) if len(ape) else None, worst_abs_err_ms=float(ae.max()),
                meas_median_ms=float(np.median(meas)))


def decode_bucket(b: int) -> str:
    if b <= 2:
        return "decode b1-2"
    if b <= 16:
        return "decode b3-16"
    if b <= 24:
        return "decode b17-24"
    return "decode b>=25"


def cls_of(f: Feat) -> str:
    k = kind_of(f)
    if k == "decode":
        return decode_bucket(f.b)
    if f.is_chunk16k:
        return "prefill 16k chunk"
    if k == "eager":
        return "mixed-eager (host-bound)"
    if k == "graph":
        return "mixed-graph"
    return "mixed-ring (other)"


def kv_band(f: Feat) -> str:
    if not f.is_chunk16k:
        return "-"
    k = f.kv0 // 1000
    for lo in reversed(KV_BANDS_K):
        if k >= lo:
            return f"kv0 {lo}k"
    return "kv0 0k"


# ---------------------------------------------------------------------------------------------------
def run_mode(pd: PassData, coeffs: dict, params: SeqParams, mode: str, draws: int, seed: int, progress: int = 0) -> tuple[list[IterRecord], float]:
    sched = sorted(pd.feats, key=lambda f: f.iteration)
    measured = {}
    for f in sched:
        m = measured_from_pass(pd, f.iteration)
        if m is not None:
            measured[f.iteration] = m
    # iterations without a usable measured timeline (kind mismatch) are dropped from the schedule
    sched = [f for f in sched if f.iteration in measured]
    stochastic = mode == "pred-mc"
    eng = Engine(coeffs, params, draws=draws if stochastic else 1, seed=seed, stochastic=stochastic, moe_band=stochastic)
    recon = {"recon-gpu": "gpu", "recon-seq": "seq"}.get(mode)
    t = time.time()
    recs = eng.run(sched, measured=measured, recon=recon, progress=progress)
    return recs, time.time() - t


def rows_from(pd: PassData, recs: list[IterRecord]) -> list[dict]:
    byf = {f.iteration: f for f in pd.feats}
    rows = []
    for r in recs:
        f = byf[r.iteration]
        wall = r.wall / 1e6
        hs = pd.rank_col("host_start_ns", f.iteration).astype(float)
        meas_lead = (f.lead_ns / 1e6) if f.prev_stage else np.nan
        rows.append(dict(iteration=f.iteration, kind=r.kind, cls=cls_of(f), role=f.role, status=r.status, kv_status=r.kv_status, excluded=f.excluded,
                         kv_band=kv_band(f), b=f.b, P=f.P, kv0=f.kv0, prev_stage=f.prev_stage,
                         meas_wall_ms=f.wall_ns / 1e6, pred_wall_ms=float(wall.mean()), pred_wall_p10_ms=float(np.percentile(wall, 10)),
                         pred_wall_p90_ms=float(np.percentile(wall, 90)), pred_wall_p05_ms=float(np.percentile(wall, 5)), pred_wall_p95_ms=float(np.percentile(wall, 95)),
                         meas_t0_ms=f.t0_ns / 1e6, meas_t1_ms=f.t1_ns / 1e6, pred_t0_ms=float(r.t0.mean()) / 1e6, pred_t1_ms=float(r.t1.mean()) / 1e6,
                         meas_lead_ms=meas_lead, pred_lead_ms=float(r.lead.mean()) / 1e6,
                         meas_host_start_min_ms=float(hs.min()) / 1e6, pred_host_start_min_ms=float(r.host_start.min(axis=1).mean()) / 1e6,
                         meas_host_skew_ms=float(hs.max() - hs.min()) / 1e6, pred_host_skew_ms=float((r.host_start.max(axis=1) - r.host_start.min(axis=1)).mean()) / 1e6,
                         pred_rank_wait_ms=(r.rank_wait.mean(axis=0) / 1e6).tolist(), host_limited_segments=int(r.host_limited_segments.mean(axis=0).max()),
                         binding=r.binding))
    return rows


def by(rows, keyf):
    out = {}
    for r in rows:
        out.setdefault(keyf(r), []).append(r)
    return dict(sorted(out.items(), key=lambda kv: str(kv[0])))


def table(rows, keyf, what=("meas_wall_ms", "pred_wall_ms")) -> list[dict]:
    out = []
    for key, rs in by(rows, keyf).items():
        m = metrics([r[what[0]] for r in rs], [r[what[1]] for r in rs])
        out.append(dict(key=key if isinstance(key, str) else " | ".join(map(str, key)), **m))
    return out


def coverage(rows, keyf) -> list[dict]:
    out = []
    for key, rs in by(rows, keyf).items():
        m = np.array([r["meas_wall_ms"] for r in rs])
        lo10 = np.array([r["pred_wall_p10_ms"] for r in rs]); hi90 = np.array([r["pred_wall_p90_ms"] for r in rs])
        lo05 = np.array([r["pred_wall_p05_ms"] for r in rs]); hi95 = np.array([r["pred_wall_p95_ms"] for r in rs])
        out.append(dict(key=key if isinstance(key, str) else " | ".join(map(str, key)), n=len(rs),
                        cover_p10_p90_pct=float(((m >= lo10) & (m <= hi90)).mean() * 100), cover_p05_p95_pct=float(((m >= lo05) & (m <= hi95)).mean() * 100),
                        band_p10_p90_median_ms=float(np.median(hi90 - lo10)), band_rel_median_pct=float(np.median((hi90 - lo10) / np.maximum(m, 1e-9) * 100)),
                        below_pct=float((m < lo10).mean() * 100), above_pct=float((m > hi90).mean() * 100)))
    return out


def workload(rows, pd: PassData) -> dict:
    ev = [r for r in rows if not r["excluded"]]
    span_m = (rows[-1]["meas_t1_ms"] - rows[0]["meas_t0_ms"]) / 1e3
    span_p = (rows[-1]["pred_t1_ms"] - rows[0]["pred_t0_ms"]) / 1e3
    byf = {f.iteration: f for f in pd.feats}
    dec_tok = sum(byf[r["iteration"]].b for r in rows if r["kind"] == "decode")
    pre_tok = sum(byf[r["iteration"]].P for r in rows)
    out = dict(n_iterations=len(rows), n_excluded_replayed=len(rows) - len(ev),
               span_measured_s=span_m, span_predicted_s=span_p, span_err_pct=(span_p - span_m) / span_m * 100, span_abs_err_s=abs(span_p - span_m),
               end_drift_ms=rows[-1]["pred_t1_ms"] - rows[-1]["meas_t1_ms"],
               max_abs_drift_ms=float(max(abs(r["pred_t1_ms"] - r["meas_t1_ms"]) for r in rows)),
               decode_tokens=int(dec_tok), prefill_tokens=int(pre_tok),
               throughput_measured_tok_s=dict(decode=dec_tok / span_m, prefill=pre_tok / span_m, total=(dec_tok + pre_tok) / span_m),
               throughput_predicted_tok_s=dict(decode=dec_tok / span_p, prefill=pre_tok / span_p, total=(dec_tok + pre_tok) / span_p),
               sums={})
    for role in ("all", "calibration", "heldout"):
        rs = ev if role == "all" else [r for r in ev if r["role"] == role]
        if not rs:
            continue
        sm = sum(r["meas_wall_ms"] for r in rs) / 1e3
        sp = sum(r["pred_wall_ms"] for r in rs) / 1e3
        st = {}
        for k, g in by(rs, lambda r: r["kind"]).items():
            st[k] = dict(n=len(g), meas_s=sum(r["meas_wall_ms"] for r in g) / 1e3, pred_s=sum(r["pred_wall_ms"] for r in g) / 1e3)
        out["sums"][role] = dict(n=len(rs), sum_measured_s=sm, sum_predicted_s=sp, signed_bias_pct=(sp - sm) / sm * 100, abs_err_s=abs(sp - sm),
                                 sum_abs_iteration_err_s=sum(abs(r["pred_wall_ms"] - r["meas_wall_ms"]) for r in rs) / 1e3, by_kind=st)
    return out


def lead_table(rows) -> list[dict]:
    rs = [r for r in rows if np.isfinite(r["meas_lead_ms"]) and not r["excluded"]]
    out = []
    for key, g in by(rs, lambda r: (r["prev_stage"], r["kind"], r["role"])).items():
        m = np.array([r["meas_lead_ms"] for r in g]); p = np.array([r["pred_lead_ms"] for r in g])
        e = p - m
        sk_m = np.array([r["meas_host_skew_ms"] for r in g]); sk_p = np.array([r["pred_host_skew_ms"] for r in g])
        out.append(dict(key=" | ".join(key), n=len(g), meas_lead_median_ms=float(np.median(m)), pred_lead_median_ms=float(np.median(p)),
                        lead_bias_ms=float(e.mean()), lead_median_abs_err_ms=float(np.median(np.abs(e))), lead_p90_abs_err_ms=float(np.percentile(np.abs(e), 90)),
                        lead_worst_abs_err_ms=float(np.abs(e).max()), meas_host_skew_median_ms=float(np.median(sk_m)), pred_host_skew_median_ms=float(np.median(sk_p))))
    return out


def evaluate_pass(pd: PassData, coeffs: dict, params: SeqParams, modes: list[str], draws: int, seed: int, out_dir: str, progress: int = 0) -> dict:
    res = {"label": pd.label, "modes": {}}
    for mode in modes:
        recs, secs = run_mode(pd, coeffs, params, mode, draws, seed, progress)
        rows = rows_from(pd, recs)
        ev = [r for r in rows if not r["excluded"]]
        R = dict(seconds=round(secs, 1), n=len(rows), n_evaluated=len(ev), draws=draws if mode == "pred-mc" else 1)
        R["by_role_class"] = table(ev, lambda r: (r["role"], r["cls"]))
        R["by_role_class_status"] = table(ev, lambda r: (r["role"], r["cls"], r["status"], r["kv_status"]))
        R["longctx_by_kv_band"] = table([r for r in ev if r["cls"] == "prefill 16k chunk"], lambda r: (r["role"], r["kv_band"], r["status"]))
        R["lead"] = lead_table(rows)
        R["workload"] = workload(rows, pd)
        R["binding"] = {k: int(sum(r["binding"].get(k, 0) for r in rows)) for k in ("serial", "scheduler", "first")}
        if mode == "pred-mc":
            R["coverage"] = coverage(ev, lambda r: (r["role"], r["cls"]))
        res["modes"][mode] = R
        tab = pa.Table.from_pylist([{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in rows])
        pq.write_table(tab, os.path.join(out_dir, f"iterations_{pd.label}_{mode}.parquet"), compression="zstd")
    return res


# ---------------------------------------------------------------------------------------------------
def write_markdown(results: dict, params: SeqParams, path: str):
    L = ["# Phase 1B sequence-simulation results", "",
         f"params sha256 `{results['params_sha256']}`; coefficients sha256 `{results['coefficients_sha256']}`; partition sha256 `{results['partition_sha256']}`; seed {results['seed']}; draws {results['draws']}", "",
         "Modes: recon-gpu (measured segments + measured host times), recon-seq (measured segments and host durations, host starts from the sequence rules), "
         "pred-det (fully predicted, median path), pred-mc (fully predicted, stochastic draws). Metrics: n, signed bias (ms, % of measured sum), median abs error, p90 APE (measured >= 0.05 ms), worst abs error.", ""]
    wc = ["key", "n", "bias_ms", "bias_pct", "median_err_ms", "median_abs_err_ms", "p90_ape_pct", "worst_abs_err_ms", "meas_median_ms"]
    for lab, pr in results["passes"].items():
        L += [f"## Pass `{lab}`", ""]
        for mode, R in pr["modes"].items():
            L += [f"### {mode} ({R['n_evaluated']} evaluated of {R['n']} iterations, {R['seconds']} s, draws {R['draws']})", "",
                  "#### Iteration latency by role and class", md_table(R["by_role_class"], wc), "",
                  "#### By status (extrapolation / inferred / kv-extrapolation reported separately)", md_table(R["by_role_class_status"], wc), ""]
            if R["longctx_by_kv_band"]:
                L += ["#### Long-context 16k chunks by KV offset", md_table(R["longctx_by_kv_band"], wc), ""]
            L += ["#### Host lead (GPU end of i-1 minus earliest host start of i) by transition",
                  md_table(R["lead"], ["key", "n", "meas_lead_median_ms", "pred_lead_median_ms", "lead_bias_ms", "lead_median_abs_err_ms", "lead_p90_abs_err_ms", "lead_worst_abs_err_ms",
                                       "meas_host_skew_median_ms", "pred_host_skew_median_ms"]), ""]
            w = R["workload"]
            L += ["#### Full workload",
                  f"- span: measured {w['span_measured_s']:.3f} s, predicted {w['span_predicted_s']:.3f} s, error {w['span_err_pct']:+.2f} % (abs {w['span_abs_err_s']:.3f} s); "
                  f"end drift {w['end_drift_ms']:+.1f} ms, max |drift| {w['max_abs_drift_ms']:.1f} ms; {w['n_excluded_replayed']} excluded (JIT) iterations replayed from their measured timelines",
                  f"- throughput (tok/s): measured decode {w['throughput_measured_tok_s']['decode']:.1f}, prefill {w['throughput_measured_tok_s']['prefill']:.1f}; "
                  f"predicted decode {w['throughput_predicted_tok_s']['decode']:.1f}, prefill {w['throughput_predicted_tok_s']['prefill']:.1f}"]
            for role, s in w["sums"].items():
                bk = ", ".join(f"{k} {v['n']} it. {v['meas_s']:.3f}/{v['pred_s']:.3f} s" for k, v in s["by_kind"].items())
                L.append(f"- Σ walls {role}: n={s['n']}, measured {s['sum_measured_s']:.3f} s, predicted {s['sum_predicted_s']:.3f} s, signed bias {s['signed_bias_pct']:+.2f} %, "
                         f"abs error {s['abs_err_s']:.3f} s, Σ|per-iteration err| {s['sum_abs_iteration_err_s']:.3f} s; by kind: {bk}")
            L += [f"- host_start binding constraint counts (draw-summed): {R['binding']}", ""]
            if "coverage" in R:
                L += ["#### Uncertainty coverage (fraction of measured walls inside the predicted band)",
                      md_table(R["coverage"], ["key", "n", "cover_p10_p90_pct", "cover_p05_p95_pct", "band_p10_p90_median_ms", "band_rel_median_pct", "below_pct", "above_pct"]), ""]
    L += ["## Sequence parameters (calibration only)", "",
          f"- a_host per rank (ms): {[round(x / 1e6, 3) for x in params.a_host_ns]}; f_idle {params.f_idle_ns / 1e6:.3f} ms; delta_launch {params.delta_launch_ns / 1e6:.3f} ms (inferred)",
          f"- queue depth Q = {params.queue_depth:.0f} launches; evidence {json.dumps(params.queue_evidence)}",
          f"- h_pre (ms per rank): {{{', '.join(f'{k}: {[round(x / 1e6, 2) for x in v]}' for k, v in params.h_pre_ns.items())}}}",
          f"- p_post (ms per rank): {{{', '.join(f'{k}: {[round(x / 1e6, 2) for x in v]}' for k, v in params.p_post_ns.items())}}}", "",
          "### eps_sched (host_start_0(i) - GPU_end(i-2) after a decode iteration), ms", ""]
    rows = []
    for kind, d in params.eps_sched_ns.items():
        for b, v in d.items():
            rows.append(dict(kind=kind, b_prev2=b, n=v["n"], median_ms=round(v["median"] / 1e6, 3), p10_ms=round(v["p10"] / 1e6, 3), p90_ms=round(v["p90"] / 1e6, 3)))
    L += [md_table(rows, ["kind", "b_prev2", "n", "median_ms", "p10_ms", "p90_ms"]), "", "### Parameter status", ""]
    L += [f"- {k}: {v}" for k, v in params.status.items()]
    open(path, "w").write("\n".join(L) + "\n")
