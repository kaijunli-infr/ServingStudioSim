#!/usr/bin/env python3
"""Revised full-E1 schedule, capture-window and storage simulation from the accepted Stage 1.5 measurements (CPU, stdlib).

    python e1_schedule_sim.py --inputs E1-INPUTS.json --out <dir> [--delay 0 10]

Per-wave durations come from the packet plan (Phase 1A medians x 1.5 margin + 5 s gap) corrected by the nsys-on measurements of
the two accepted passes (b=8 32k wave: 9.02 s measured vs 9.1 s planned; b=48 32k wave: 32.04 s vs 32.7 s). The capture window is
evaluated for a delayed and a no-delay profiler start; the storage and export/parse times scale the measured per-iteration kernel
rows (b=8: 5661 rows/iteration over 4 ranks, b=48: 5682) and the measured export rates.
"""
from __future__ import annotations

import argparse
import json
import os

MEAS = {"b8_32k_wave_s": 9.022, "b48_32k_wave_s": 32.04, "b8_planned_s": 9.1, "b48_planned_s": 32.7,
        "rows_per_iter": {8: 5887244 / 1040, 48: 6363588 / 1120}, "sqlite_B_per_row": 874708992 / 5887244, "rep_B_per_row": 170126141 / 5887244,
        "export_s_per_Mrow": 37 / 5.887, "parse_s_per_Mrow": 103 / 5.887, "parsed_json_B_per_row": 26237219 / 5887244, "kernels_parquet_B_per_row": 37301839 / 5887244,
        "host_timeline_B_per_row": 35069597 / 5887244, "startup_s_measured": 23 * 60 + 17, "first_iteration_after_replay_start_s": 0.038}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--delay", type=float, nargs="*", default=[0.0, 10.0])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    d = json.load(open(a.inputs)); plan = d["main"]["plan"]; des = d["design"]
    ratio = {8: MEAS["b8_32k_wave_s"] / MEAS["b8_planned_s"], 48: MEAS["b48_32k_wave_s"] / MEAS["b48_planned_s"]}
    corr = (ratio[8] + ratio[48]) / 2
    rows = []; total_rows = 0; overlap = 0
    prev_end = -1e9
    for w in plan:
        b = w["b"]; dur = w["est_duration_s"] * corr
        iters = w["prefill_iters"] + des["OUT_LEN"]
        rpi = MEAS["rows_per_iter"][8] if b <= 16 else MEAS["rows_per_iter"][48]
        total_rows += iters * rpi
        start = w["arrival_s"]; end = start + dur
        ov = start < prev_end; overlap += ov; prev_end = max(prev_end, end)
        rows.append({**w, "sim_duration_s": round(dur, 2), "sim_end_s": round(end, 2), "overlaps_previous": ov, "iterations": iters, "kernel_rows_M": round(iters * rpi / 1e6, 2)})
    span = max(r["sim_end_s"] for r in rows)
    windows = {}
    for delay in a.delay:
        lost = [r for r in rows if r["arrival_s"] < delay]   # waves whose first iterations start before the profiler is armed
        partial = [r for r in lost if r["sim_end_s"] > delay]
        windows[str(delay)] = {"waves_starting_before_arm": [f"b{r['b']} kv{r['kv']} w{r['wave']}" for r in lost], "n_waves_partially_captured": len(partial),
                              "n_waves_fully_missed": len(lost) - len(partial), "first_captured_iteration_fully_inside": delay <= MEAS["first_iteration_after_replay_start_s"] or not lost}
    rep = total_rows * MEAS["rep_B_per_row"]; sql = total_rows * MEAS["sqlite_B_per_row"]
    derived = total_rows * (MEAS["parsed_json_B_per_row"] + MEAS["kernels_parquet_B_per_row"] + MEAS["host_timeline_B_per_row"])
    out = {"measured_basis": MEAS, "duration_correction_factor": round(corr, 4), "waves": rows, "n_waves": len(rows), "waves_overlapping_previous": overlap,
           "trace_span_s_sim": round(span, 1), "trace_span_s_packet": d["main"]["trace_span_s"], "iterations_total": sum(r["iterations"] for r in rows),
           "kernel_rows_M": round(total_rows / 1e6, 1), "nsys_rep_GB": round(rep / 1e9, 1), "sqlite_GB": round(sql / 1e9, 1), "parsed_json_kernels_parquet_host_timeline_GB": round(derived / 1e9, 1),
           "routes_and_corpus_GB": round(460800 * 40 * 6 * 2 * 2 / 1e9, 2), "raw_total_GB": round((rep + sql + derived) / 1e9 + 0.5 + 0.2, 1),
           "export_min": round(total_rows / 1e6 * MEAS["export_s_per_Mrow"] / 60, 1), "parse_min": round(total_rows / 1e6 * MEAS["parse_s_per_Mrow"] / 60, 1),
           "gpu_held_min_estimate": {"startup": round(MEAS["startup_s_measured"] / 60, 1), "warmup_and_capture": round(span / 60 + 1.5, 1), "nsys_stop_and_report_finalisation": 20, "total": round(MEAS["startup_s_measured"] / 60 + span / 60 + 1.5 + 20, 1)},
           "capture_windows": windows}
    json.dump(out, open(os.path.join(a.out, "e1_schedule_sim.json"), "w"), indent=1)
    L = ["# Revised full-E1 schedule, capture window and storage (simulation from accepted measurements)", "",
         f"Wave durations = packet plan x {corr:.4f} (measured nsys-on waves: b=8 32k {MEAS['b8_32k_wave_s']} s vs planned {MEAS['b8_planned_s']} s; b=48 32k {MEAS['b48_32k_wave_s']} s vs {MEAS['b48_planned_s']} s).",
         f"Trace span {span:.1f} s (packet {d['main']['trace_span_s']} s); waves overlapping the previous wave: {overlap} of {len(rows)}; iterations {out['iterations_total']}; kernel rows {out['kernel_rows_M']} M.", "",
         "| b | KV | wave | arrival s | sim duration s | sim end s | gap to next arrival s | iterations | kernel rows M |", "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows):
        gap = (rows[i + 1]["arrival_s"] - r["sim_end_s"]) if i + 1 < len(rows) else float("nan")
        L.append(f"| {r['b']} | {r['kv']} | {r['wave']} | {r['arrival_s']} | {r['sim_duration_s']} | {r['sim_end_s']} | {gap:.1f} | {r['iterations']} | {r['kernel_rows_M']} |")
    L += ["", "## Capture window", ""]
    for k, v in windows.items():
        L.append(f"* profiler start delay {k} s: waves starting before the profiler is armed: {v['waves_starting_before_arm'] or 'none'}; partially captured {v['n_waves_partially_captured']}, fully missed {v['n_waves_fully_missed']}.")
    L += ["", "## Storage and post-processing (measured rates: sqlite %.0f B/row, rep %.0f B/row, export %.1f s/Mrow, parse %.1f s/Mrow)" % (MEAS["sqlite_B_per_row"], MEAS["rep_B_per_row"], MEAS["export_s_per_Mrow"], MEAS["parse_s_per_Mrow"]), "",
          f"| .nsys-rep | .sqlite | parsed.json + kernels parquet + host_timeline | routes + corpus | raw total | export | parse |", "|---|---|---|---|---|---|---|",
          f"| {out['nsys_rep_GB']} GB | {out['sqlite_GB']} GB | {out['parsed_json_kernels_parquet_host_timeline_GB']} GB | ~{out['routes_and_corpus_GB']} GB | ~{out['raw_total_GB']} GB | ~{out['export_min']} min | ~{out['parse_min']} min (linear scaling; Phase 0 run2 at 97 M rows took 29 min, i.e. 18 s/Mrow, so this is an upper estimate) |",
          "", f"GPU-held estimate: startup {out['gpu_held_min_estimate']['startup']} min (measured on the b=8 pass) + warm-up and capture {out['gpu_held_min_estimate']['warmup_and_capture']} min + nsys stop/report finalisation ≤ 20 min = ~{out['gpu_held_min_estimate']['total']} min; export and parse are CPU-only on the tray while the launcher still holds the pass."]
    open(os.path.join(a.out, "E1-SCHEDULE-SIM.md"), "w").write("\n".join(L) + "\n")
    print(json.dumps({k: out[k] for k in ("duration_correction_factor", "trace_span_s_sim", "waves_overlapping_previous", "iterations_total", "kernel_rows_M", "nsys_rep_GB", "sqlite_GB", "raw_total_GB", "export_min", "parse_min", "gpu_held_min_estimate")}))
    for k, v in windows.items():
        print("delay", k, v)


if __name__ == "__main__":
    main()
