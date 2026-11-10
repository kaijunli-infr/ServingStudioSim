"""Summarise a TP1 fill run directory: per-kind coverage, distributions,
reference ratios, repeatability, step walls, GPU samples, cache delta.

    tp1_report.py RUN_DIR [--db DB]   (prints Markdown; --json for the raw dict)
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
from pathlib import Path

GPU = "NVIDIA GB300"
KINDS = [("single_gemm", "torch_linear"), ("single_gemm", "torch"), ("rms_norm", "flashinfer"),
         ("elementwise", "triton"), ("flashinfer_attn_prefill", "fa2"), ("flashinfer_attn_decode", "fa2")]


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def fmt_us(ms):
    return "-" if ms is None else f"{ms * 1000:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--db")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    run = a.run_dir
    steps = [json.loads(l) for l in (run / "steps.jsonl").read_text().splitlines() if l.strip()]
    wall = {s["step"]: s for s in steps}
    out = {"run_dir": str(run), "steps": steps}
    rows = []
    for kind, backend in KINDS:
        v = load(run / f"validate-{kind}-{backend}.json") or {}
        r = load(run / f"validate-repeat-{kind}-{backend}.json") or {}
        d = v.get("time_ms_distribution") or {}
        refs = v.get("reference_ratios") or {}
        rows.append({
            "kind": kind, "backend": backend, "specs": v.get("specs"), "rows": v.get("rows_present"),
            "missing": v.get("rows_missing"), "statuses": v.get("run_statuses"), "verdict": v.get("verdict"),
            "fill_wall_s": (wall.get(f"fill-{kind}-{backend}") or {}).get("wall_s"),
            "t_min_us": fmt_us(d.get("min")), "t_p50_us": fmt_us(d.get("p50")), "t_p90_us": fmt_us(d.get("p90")),
            "t_max_us": fmt_us(d.get("max")), "outliers": v.get("outlier_rows"), "retried": v.get("retried_rows"),
            "mono_viol": v.get("monotonic_violations"),
            "ratios": {k: (f"{x['gb300_over_ref']['p50']:.2f} (n={x['common_rows']}, {x['gb300_over_ref']['min']:.2f}–{x['gb300_over_ref']['max']:.2f})") for k, x in refs.items()},
            "repeat_pairs": r.get("pairs"), "repeat_rel_p50": (r.get("abs_rel_diff") or {}).get("p50"),
            "repeat_rel_max": (r.get("abs_rel_diff") or {}).get("max"), "repeat_verdict": r.get("verdict"),
            "anomalies": v.get("anomalies"), "problems": v.get("problems"),
        })
    out["kinds"] = rows
    # DB totals
    if a.db:
        con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
        out["db_rows"] = {t: con.execute(f"select count(*) from {t} where gpu_name=?", (GPU,)).fetchone()[0]
                          for t in ("single_gemm", "rms_norm", "elementwise", "flashinfer_attn_prefill", "flashinfer_attn_decode", "kv_cache_append")}
    # GPU samples
    samples = list(csv.reader((run / "gpu-samples.csv").open())) if (run / "gpu-samples.csv").exists() else []
    if len(samples) > 1:
        hdr = [h.strip() for h in samples[0]]
        body = [[c.strip() for c in r] for r in samples[1:] if len(r) == len(hdr)]
        def col(name, conv=float):
            alias = {"clocks.sm": "clocks.current.sm", "clocks.mem": "clocks.current.memory"}
            i = next((i for i, h in enumerate(hdr) if h.startswith(name) or h.startswith(alias.get(name, name))), None)
            if i is None:
                return []
            vals = []
            for r in body:
                try:
                    vals.append(conv(r[i].split()[0]))
                except (ValueError, IndexError):
                    pass
            return vals
        util = col("utilization.gpu"); power = col("power.draw"); sm = col("clocks.sm"); mem = col("memory.used")
        out["gpu_samples"] = {"n": len(body), "seconds": len(body) * 0.5, "util_gt0": sum(1 for u in util if u > 0),
                              "util_ge50": sum(1 for u in util if u >= 50), "util_mean": statistics.fmean(util) if util else None,
                              "power_max_w": max(power) if power else None, "power_mean_busy_w": statistics.fmean([p for p, u in zip(power, util) if u > 0]) if any(u > 0 for u in util) else None,
                              "sm_clock_min_busy": min([c for c, u in zip(sm, util) if u > 0], default=None), "sm_clock_max": max(sm) if sm else None,
                              "mem_used_max_mib": max(mem) if mem else None}
    pre = (run / "cache-bytes-pre.txt"); post = (run / "cache-bytes-post.txt")
    if pre.exists() and post.exists():
        out["cache_delta_bytes"] = int(post.read_text()) - int(pre.read_text())
    out["probe"] = load(run / "probe.json")
    total = sum(s["wall_s"] for s in steps)
    out["total_step_wall_s"] = total
    if a.json:
        print(json.dumps(out, indent=2, default=str)); return 0
    print(f"run {run.name}: {len(steps)} steps, {total/60:.1f} min of step wall")
    print("| kind / backend | specs | rows | missing | verdict | fill wall | time µs min / p50 / p90 / max | outliers | mono viol | GB300/ref p50 | repeat n, |Δ| p50 / max |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        rep = f"{r['repeat_pairs']}, {r['repeat_rel_p50']*100:.1f} % / {r['repeat_rel_max']*100:.1f} %" if r["repeat_rel_p50"] is not None else "-"
        print(f"| {r['kind']} {r['backend']} | {r['specs']} | {r['rows']} | {r['missing']} | {r['verdict']} | {r['fill_wall_s'] and round(r['fill_wall_s'])} s | {r['t_min_us']} / {r['t_p50_us']} / {r['t_p90_us']} / {r['t_max_us']} | {r['outliers']} | {r['mono_viol']} | {'; '.join(f'{k}: {v}' for k, v in r['ratios'].items())} | {rep} |")
        if r["anomalies"] or r["problems"]:
            print(f"  anomalies={r['anomalies']} problems={r['problems']}")
    if "db_rows" in out: print("db rows (GB300):", out["db_rows"])
    if "gpu_samples" in out: print("gpu samples:", json.dumps(out["gpu_samples"]))
    if "cache_delta_bytes" in out: print("cache delta bytes:", out["cache_delta_bytes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
