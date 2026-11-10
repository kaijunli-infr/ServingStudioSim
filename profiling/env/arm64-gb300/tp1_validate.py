"""Validate one TP1 fill batch (or the repeatability sample) against the private DB.

    tp1_validate.py batch  --db DB --kind K --backend B --specs specs/K.json \
                           --results OUT/results.json --stderr OUT.err --report R.json
    tp1_validate.py repeat --db DB --kind K --backend B --results OUT/results.json --report R.json

Exit 0 = pass, 1 = anomalies worth a human look (the orchestrator stops),
2 = systematic failure (failed rows, bad units, missing coverage).
Reference rows (B200, H200) come from the same DB file: the private DB is a
copy of the tracked one plus the GB300 rows this campaign adds.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from pathlib import Path

GPU = "NVIDIA GB300"
REFERENCE_GPUS = ("NVIDIA B200", "NVIDIA H200")
TIME_MAX_MS = 5000.0  # nothing in this grid should take longer than 5 s


def q(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))] if s else None


def dist(values):
    if not values:
        return None
    return {"n": len(values), "min": min(values), "p50": q(values, 0.5), "p90": q(values, 0.9),
            "p99": q(values, 0.99), "max": max(values), "mean": statistics.fmean(values)}


def arg_columns(con, table):
    cols = [r[1] for r in con.execute(f"pragma table_info({table})")]
    skip = {"id", "gpu_name", "backend", "args_hash", "run_key", "profiler_run_at", "verified", "time_ms",
            "tflops", "memory_bandwidth_gbps", "energy_j", "is_outlier", "retry_count", "outlier_reason", "created_at"}
    return [c for c in cols if c not in skip]


def rows_for(con, table, gpu, backend, cols):
    sel = ", ".join(cols)
    out = {}
    for row in con.execute(
        f"select {sel}, time_ms, tflops, memory_bandwidth_gbps, is_outlier, retry_count, outlier_reason, profiler_run_at "
        f"from {table} where gpu_name = ? and backend = ?", (gpu, backend)):
        key = tuple(row[: len(cols)])
        out[key] = {"time_ms": row[len(cols)], "tflops": row[len(cols) + 1], "gbps": row[len(cols) + 2],
                    "is_outlier": row[len(cols) + 3], "retry_count": row[len(cols) + 4],
                    "outlier_reason": row[len(cols) + 5], "run_at": row[len(cols) + 6]}
    return out


def spec_key(spec, cols):
    return tuple(spec.get(c) for c in cols)


def normalise(con, table, cols, specs):
    """Match spec values to the column types (ints vs text) the DB stores."""
    types = {r[1]: (r[2] or "").upper() for r in con.execute(f"pragma table_info({table})")}
    keys = []
    for s in specs:
        vals = []
        for c in cols:
            v = s.get(c)
            if v is not None and "INT" in types.get(c, "") and not isinstance(v, int):
                v = int(v)
            vals.append(v)
        keys.append(tuple(vals))
    return keys


def batch(args) -> int:
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    table = args.kind
    cols = arg_columns(con, table)
    specs = json.loads(Path(args.specs).read_text())["specs"]
    spec_keys = normalise(con, table, cols, specs)
    gb = rows_for(con, table, GPU, args.backend, cols)
    covered = [k for k in spec_keys if k in gb]
    missing = [k for k in spec_keys if k not in gb]
    report = {"kind": table, "backend": args.backend, "arg_columns": cols, "specs": len(specs),
              "rows_present": len(covered), "rows_missing": len(missing),
              "missing_examples": [dict(zip(cols, k)) for k in missing[:10]]}
    # results.json of this run: statuses and errors
    results = json.loads(Path(args.results).read_text()) if Path(args.results).exists() else {}
    statuses = {}
    for r in results.get("results", []):
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    report["run_statuses"] = statuses
    report["run_mode"] = results.get("mode")
    report["run_persisted"] = results.get("persisted")
    err_text = Path(args.stderr).read_text() if args.stderr and Path(args.stderr).exists() else ""
    m = re.search(r"profiled (\d+)/(\d+) specs; (\d+) failed[^\n]*", err_text)
    report["run_summary_line"] = m.group(0) if m else err_text.strip().splitlines()[-1:] if err_text.strip() else None
    failure_reasons = re.findall(r"\(([^()]*)\)\s*$", m.group(0)) if m else []
    report["failure_reasons"] = failure_reasons
    # units / sanity on the covered rows
    times = [gb[k]["time_ms"] for k in covered]
    bad_units = [dict(zip(cols, k)) | gb[k] for k in covered
                 if gb[k]["time_ms"] is None or not (0 < gb[k]["time_ms"] < TIME_MAX_MS)]
    outliers = [dict(zip(cols, k)) | gb[k] for k in covered if gb[k]["is_outlier"]]
    retried = sum(1 for k in covered if (gb[k]["retry_count"] or 0) > 0)
    report["time_ms_distribution"] = dist(times)
    report["bad_unit_rows"] = bad_units[:10]
    report["outlier_rows"] = len(outliers)
    report["outlier_examples"] = outliers[:5]
    report["retried_rows"] = retried
    if "tflops" in (cols + ["tflops"]):
        tf = [gb[k]["tflops"] for k in covered if gb[k]["tflops"]]
        report["tflops_distribution"] = dist(tf)
    gbps = [gb[k]["gbps"] for k in covered if gb[k]["gbps"]]
    report["gbps_distribution"] = dist(gbps)
    # monotonic sanity along the sweep axis, per static config
    axis = next((c for c in ("m", "num_tokens", "total_tokens", "append_len", "input_size_bytes") if c in cols), None)
    if axis is not None:
        ai = cols.index(axis)
        groups = {}
        for k in covered:
            groups.setdefault(tuple(v for i, v in enumerate(k) if i != ai), []).append((k[ai], gb[k]["time_ms"]))
        violations = []
        for g, pts in groups.items():
            pts.sort()
            for (x0, t0), (x1, t1) in zip(pts, pts[1:]):
                if x0 >= 256 and t1 < 0.85 * t0:  # time dropped >15 % while the sweep axis grew
                    violations.append({"static": g, axis: [x0, x1], "time_ms": [t0, t1]})
        report["sweep_axis"] = axis
        report["monotonic_violations"] = len(violations)
        report["monotonic_examples"] = violations[:8]
    # ratios against reference GPUs at identical args
    ratios = {}
    for ref_gpu in REFERENCE_GPUS:
        for ref_backend in {args.backend, "fa2" if args.backend == "fa2" else args.backend}:
            ref = rows_for(con, table, ref_gpu, ref_backend, cols)
            common = [k for k in covered if k in ref and ref[k]["time_ms"]]
            if common:
                rs = [gb[k]["time_ms"] / ref[k]["time_ms"] for k in common]
                ratios[f"{ref_gpu}/{ref_backend}"] = {"common_rows": len(common), "gb300_over_ref": dist(rs)}
    report["reference_ratios"] = ratios
    # verdict
    problems, anomalies = [], []
    if missing:
        problems.append(f"{len(missing)} of {len(specs)} specs have no row")
    if bad_units:
        problems.append(f"{len(bad_units)} rows with time_ms outside (0, {TIME_MAX_MS})")
    failed = statuses.get("missing", 0)
    if failed and failed > max(1, 0.02 * len(specs)):
        problems.append(f"{failed} failed specs in this run ({failure_reasons})")
    if report.get("monotonic_violations", 0) > max(2, 0.03 * len(covered)):
        anomalies.append(f"{report['monotonic_violations']} monotonicity violations")
    for name, r in ratios.items():
        med = r["gb300_over_ref"]["p50"]
        if "B200" in name and (med > 1.5 or med < 0.3):
            anomalies.append(f"median time ratio vs {name} = {med:.2f}")
        if "H200" in name and (med > 1.5 or med < 0.15):
            anomalies.append(f"median time ratio vs {name} = {med:.2f}")
    if len(outliers) > 0.05 * max(1, len(covered)):
        anomalies.append(f"{len(outliers)} rows flagged is_outlier")
    report["problems"], report["anomalies"] = problems, anomalies
    report["verdict"] = "FAIL" if problems else ("ANOMALY" if anomalies else "PASS")
    Path(args.report).write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("kind", "backend", "specs", "rows_present", "rows_missing", "run_statuses",
                                            "time_ms_distribution", "monotonic_violations", "outlier_rows",
                                            "reference_ratios", "problems", "anomalies", "verdict") if k in report},
                     default=str))
    return 0 if not problems and not anomalies else (2 if problems else 1)


def repeat(args) -> int:
    """Compare a --fresh re-measurement (results.json with args) to the stored rows."""
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    table = args.kind
    cols = arg_columns(con, table)
    stored = rows_for(con, table, GPU, args.backend, cols)
    results = json.loads(Path(args.results).read_text())
    specs = results.get("specs", [])
    keys = normalise(con, table, cols, specs)
    pairs = []
    for key, r in zip(keys, results.get("results", [])):
        if r.get("status") != "ok" or key not in stored:
            continue
        fresh = (r.get("metrics") or {}).get("time_ms")
        if fresh:
            pairs.append({"args": dict(zip(cols, key)), "stored_ms": stored[key]["time_ms"], "fresh_ms": fresh,
                          "ratio": fresh / stored[key]["time_ms"]})
    ratios = [p["ratio"] for p in pairs]
    rel = [abs(r - 1) for r in ratios]
    report = {"kind": table, "backend": args.backend, "pairs": len(pairs), "ratio_fresh_over_stored": dist(ratios),
              "abs_rel_diff": dist(rel), "worst": sorted(pairs, key=lambda p: -abs(p["ratio"] - 1))[:5]}
    report["verdict"] = "PASS" if pairs and (q(rel, 0.5) or 0) < 0.05 and max(rel) < 0.25 else ("ANOMALY" if pairs else "FAIL")
    Path(args.report).write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("kind", "backend", "pairs", "ratio_fresh_over_stored", "abs_rel_diff", "verdict")}, default=str))
    return 0 if report["verdict"] == "PASS" else 1


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("batch", "repeat"):
        sp = sub.add_parser(name)
        sp.add_argument("--db", required=True)
        sp.add_argument("--kind", required=True)
        sp.add_argument("--backend", required=True)
        sp.add_argument("--results", required=True)
        sp.add_argument("--report", required=True)
        if name == "batch":
            sp.add_argument("--specs", required=True)
            sp.add_argument("--stderr")
    args = p.parse_args()
    return batch(args) if args.cmd == "batch" else repeat(args)


if __name__ == "__main__":
    raise SystemExit(main())
