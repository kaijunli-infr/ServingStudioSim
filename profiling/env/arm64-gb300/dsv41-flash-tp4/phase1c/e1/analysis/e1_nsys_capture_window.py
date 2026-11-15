#!/usr/bin/env python3
"""Which part of the dry trace did the nsys capture cover? (CPU, read-only)

    python analysis/e1_nsys_capture_window.py --run-root <stage15 run root> --trace traces/e1_dry.csv [--out report.md]

Reads the four Stage 1.5 passes' metrics jsonl (median pure-decode iteration time per cell, for the nsys-overhead factor), the
nsys pass's parsed.json (first/last parsed iteration) and the trace arrival times, and reports where the capture window
(capture_start_delay_seconds after measurement start) fell relative to the b=8 and b=48 waves.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st


def metrics(run_dir):
    m = [json.loads(l) for l in open(glob.glob(run_dir + "/profile/vllm/*_metrics.jsonl")[0])]
    m.sort(key=lambda r: r["iteration_index"])
    return m


def pure(m, b):
    return [r for r in m if r.get("decode_requests") == b and not r.get("prefill_tokens")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True); ap.add_argument("--trace", required=True)
    ap.add_argument("--nsys-pass", default="e1_nsys_val"); ap.add_argument("--passes", default="e1_dry_off_v2,e1_dry_on_v2,e1_dry_off2_v2,e1_nsys_val")
    ap.add_argument("--out")
    a = ap.parse_args()
    lines = ["# nsys capture window vs the dry trace", ""]
    lines += ["| pass | metrics iterations | median pure-decode iteration ms b=8 | b=48 |", "|---|---|---|---|"]
    for p in a.passes.split(","):
        m = metrics(f"{a.run_root}/{p}")
        lines.append(f"| {p} | {len(m)} | {st.median(r['observed_elapsed_ms'] for r in pure(m, 8)):.3f} | {st.median(r['observed_elapsed_ms'] for r in pure(m, 48)):.3f} |")
    m = metrics(f"{a.run_root}/{a.nsys_pass}")
    d = json.load(open(f"{a.run_root}/{a.nsys_pass}/profile/parsed.json"))
    its = [x["iteration"] for x in d["iteration_details"]]
    first_parsed, last_parsed = min(its), max(its)
    cum = 0.0; t = {}
    for r in m:
        t[r["iteration_index"]] = cum  # engine time at the START of this iteration (sum of previous observed_elapsed_ms)
        cum += r["observed_elapsed_ms"]
    p8, p48 = pure(m, 8), pure(m, 48)
    first8, last8 = p8[0]["iteration_index"], p8[-1]["iteration_index"]
    first48, last48 = p48[0]["iteration_index"], p48[-1]["iteration_index"]
    lines += ["", f"nsys pass metrics iterations {m[0]['iteration_index']}..{m[-1]['iteration_index']} ({len(m)}); engine time (sum of observed_elapsed_ms) {cum/1000:.2f} s",
              f"pure b=8 iterations {first8}..{last8} ({len(p8)}): engine time {t[first8]/1000:.2f} s .. {(t[last8] + [r for r in m if r['iteration_index']==last8][0]['observed_elapsed_ms'])/1000:.2f} s",
              f"pure b=48 iterations {first48}..{last48} ({len(p48)}): engine time {t[first48]/1000:.2f} s .. {cum/1000:.2f} s",
              f"parsed (captured) iterations {first_parsed}..{last_parsed} ({len(its)}): engine time at first parsed iteration {t[first_parsed]/1000:.2f} s",
              f"metrics iterations before the first parsed iteration: {sum(1 for r in m if r['iteration_index'] < first_parsed)} "
              f"(pure b=8 among them: {sum(1 for r in p8 if r['iteration_index'] < first_parsed)} of {len(p8)}; pure b=48: {sum(1 for r in p48 if r['iteration_index'] < first_parsed)} of {len(p48)})", ""]
    rows = list(csv.DictReader(open(a.trace)))
    cols = list(rows[0].keys())
    lines.append(f"trace `{a.trace}`: {len(rows)} requests, columns {cols}")
    tcol = next((c for c in cols if "arrival" in c.lower() or "time" in c.lower()), None)
    idcol = next((c for c in cols if "id" in c.lower() or "name" in c.lower()), cols[0])
    if tcol:
        groups = {}
        for r in rows:
            key = "b48" if "b48" in r[idcol] else ("b8" if "b8" in r[idcol] else "other")
            groups.setdefault(key, []).append(float(r[tcol]))
        for k, v in groups.items():
            lines.append(f"trace group {k}: n={len(v)}, {tcol} min {min(v):.0f} max {max(v):.0f}")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        open(a.out, "w").write(text)


if __name__ == "__main__":
    main()
