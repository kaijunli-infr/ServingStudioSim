#!/usr/bin/env python3
"""Resource sizing for the next E1 capture from the V2 and V3 measurements (CPU-only arithmetic; nothing is fitted to data
beyond passing curves through the two controlled points, and no curve is claimed to be a scaling law).

    python3 -I feasibility/e1_scaling_model.py [--out DIR]

Every input is in MEASURED / OBSERVED / ASSUMED below with its source. For a size N (kernel rows) each size-dependent
quantity is computed under several hypotheses, all of which pass through the V3 point:

  power   y3 * (N/N3)**p with p from the V2 and V3 points (two points determine p; this is not a validated exponent)
  line    the straight line through the V2 and V3 points
  prop    proportional to rows from V3 (p = 1)
  stress  p = STRESS_EXPONENT, an assumption with no measurement behind it, kept to show what a super-linear law would cost

The plausible range reported is min..max over power, line and prop; stress is reported next to it. The spread between the
hypotheses at a given N is the model uncertainty; it grows with the distance from the measured sizes.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import json
import math
import sys
from pathlib import Path

GIB = 1024**3
N2, N3 = 12_250_832, 24_501_664  # M: kernel rows, V2 and V3 recovered traces

# M = measured in a controlled run (source in the comment); values at (V2, V3)
MEASURED = {
    "flush_s": (61.9, 112.0),  # stop request -> controller phase importing (lifecycle.json)
    "import_s": (28.9, 53.0),  # V3: importer process first seen -> gone (import probe). V2: phase start -> report mtime, importer not observed (E)
    "importer_rss_gib": (8.7, 15.61),  # V2: offline import only, maximum of 1 s samples; V3: in session (= offline 15.63)
    "stream_gb": (0.348288111, 0.661640958),
    "report_gb": (0.391797303, 0.792004886),
    "export_gb": (1.964322816, 3.954987008),  # in-session sqlite export
    "run_root_gb": (3.16, 6.311),  # peak, storage sampler
    "post_release_export_parse_s": (308.0, 613.0),  # session closed -> launcher rc 0 (run.log): 15:57:04->16:02:12, 21:59:40->22:09:53
    "offline_import_s": (26.0, 56.0),  # post-run recovery of the durable stream
}
V3 = {
    "startup_s": 1523.0,  # launcher start 21:29:07 -> measurement begins ~21:54:30 (monitor; about +-30 s)
    "capture_s": 139.0,  # measurement begins -> stop request 21:56:49
    "gpu_held_s": 1833.0,  # controller and sampler upper edge
    "shutdown_s": 4.8,  # report written -> session closed
    "mem_available_min_gib": 547.2,  # during the import
    "mem_available_idle_gib": 858.0,
    "pre_launch_min": 5.3,  # reservation Running 21:23:51 -> launcher start
    "release_to_closeout_min": 16.2,  # 21:59:40 -> closeout CLEAN 22:15:51 (of which export and parse 10.2)
    "closeout_to_delete_min": 8.3,  # -> reservation deleted 22:24:06
    "reservation_min": 60.2,
    "temp_peak_gb": 5.219,  # post-run; includes 0.70 GB of retained V1 and V2 streams
    "retained_streams_gb_then": 0.70,
    "retained_streams_gb_now": 1.36,  # V1, V2 and V3 tray originals, never deleted
    "archive_over_run_root": 6.947 / 6.285,
    "offline_export_s": 130.0,  # from the post-run storage samples (approximate)
}
STARTUP_RANGE_S = (1487.0, 1640.0)  # M: V2 1487 s; 1640 s is the largest value quoted in earlier analyses and is kept as the upper bound
# observed outside a controlled lifecycle run (the trace was destroyed; row count is the plan estimate, not a measurement)
OBSERVED_E1_MAIN_V3 = {"rows_estimate": 216e6, "flush_s": 865.0, "import_49pct_after_s": 217.0, "importer_rss_gb_at_49pct": 61.0, "capture_s": 896.0}
STRESS_EXPONENT = 1.15  # A
# the limits frozen for V3 (E1-V3-SCALE-PACKET.md section 4 and 8). Reported for comparison only; this analysis changes none of them.
V3_LIMITS = {"flush_s": 300.0, "import_s": 300.0, "stop_to_report_s": 600.0, "gpu_held_s": 2700.0, "reservation_min": 90.0, "importer_rss_gib": 128.0,
             "mem_available_floor_gib": 64.0, "temp_gb": 10.0, "run_root_gb": 8.0, "run_root_plus_archive_gb": 16.0, "server_term_grace_s": 60.0}

OPTIONS = {
    "none (V3 as measured)": {"rows": N3, "capture_s": 139.0, "note": "no further capture; reproduces the V3 measurements"},
    "intermediate ~49 M": {"rows": 2 * N3, "capture_s": 299.0, "note": "the accepted 56-request trace four times (8 waves): rows E = 2 x V3; capture E = 3 x 80 s spacing + 59 s"},
    "full E1 ~216 M": {"rows": 216e6, "capture_s": 896.0, "note": "traces/e1_main.csv, 36 waves; rows are the plan estimate (never measured); capture M in e1_main_v3"},
}


def hypotheses(key: str, rows: float) -> dict:
    y2, y3 = MEASURED[key]
    k = rows / N3
    p = math.log(y3 / y2) / math.log(N3 / N2)
    return {"power": y3 * k**p, "line": y3 + (y3 - y2) / (N3 - N2) * (rows - N3), "prop": y3 * k, "stress": y3 * k**STRESS_EXPONENT, "exponent_from_two_points": p}


def rng(key: str, rows: float) -> tuple[float, float, float]:
    h = hypotheses(key, rows)
    vals = [h["power"], h["line"], h["prop"]]
    return min(vals), max(vals), h["stress"]


def size(rows: float, capture_s: float) -> dict:
    out: dict = {"rows": rows, "scale_vs_v3": rows / N3}
    for key in MEASURED:
        out[key] = rng(key, rows)
    f, i = out["flush_s"], out["import_s"]
    out["stop_to_report_s"] = (f[0] + i[0] + 1.2, f[1] + i[1] + 1.2, f[2] + i[2] + 1.2)  # V3: 112.0 + 53.0 + 1.2 = 166.2
    s2r = out["stop_to_report_s"]
    out["gpu_held_s"] = (STARTUP_RANGE_S[0] + capture_s + s2r[0] + V3["shutdown_s"], STARTUP_RANGE_S[1] + capture_s + s2r[1] + V3["shutdown_s"], STARTUP_RANGE_S[1] + capture_s + s2r[2] + V3["shutdown_s"])
    out["gpu_held_worst_case_under_v3_controller_s"] = STARTUP_RANGE_S[1] + capture_s + V3_LIMITS["stop_to_report_s"] + V3_LIMITS["server_term_grace_s"]
    e = out["post_release_export_parse_s"]
    fixed_min = V3["pre_launch_min"] + (V3["release_to_closeout_min"] - 613.0 / 60) + V3["closeout_to_delete_min"]  # A: these parts do not grow with rows
    out["reservation_min"] = tuple(fixed_min + out["gpu_held_s"][j] / 60 + e[j] / 60 for j in range(3))
    out["reservation_if_deleted_at_gpu_release_min"] = tuple(V3["pre_launch_min"] + out["gpu_held_s"][j] / 60 + 2.0 for j in range(3))  # A: 2 min to verify the release and delete
    r = out["importer_rss_gib"]
    out["mem_available_min_gib"] = tuple(V3["mem_available_min_gib"] - (r[j] - MEASURED["importer_rss_gib"][1]) for j in (1, 0, 2))  # (low, high, stress); A: server footprint as in V3
    new = V3["temp_peak_gb"] - V3["retained_streams_gb_then"]
    k = rows / N3
    pe = hypotheses("export_gb", rows)["exponent_from_two_points"]  # the export dominates the scratch and the offline export time
    out["temp_peak_gb"] = (*sorted(V3["retained_streams_gb_now"] + new * k**x for x in (pe, 1.0)), V3["retained_streams_gb_now"] + new * k**STRESS_EXPONENT)
    rr = out["run_root_gb"]
    out["run_root_plus_archive_gb"] = tuple(x * (1 + V3["archive_over_run_root"]) for x in rr)
    out["offline_export_s"] = (*sorted(V3["offline_export_s"] * k**x for x in (pe, 1.0)), V3["offline_export_s"] * k**STRESS_EXPONENT)
    return out


def largest_rows_within(key: str, limit: float, hypothesis: str) -> float:
    """Largest N for which the hypothesis stays at or below the limit (bisection; every hypothesis is increasing in N)."""
    lo, hi = float(N3), 1e10
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if hypotheses(key, mid)[hypothesis] <= limit else (lo, mid)
    return lo


def fmt(t, digits=0) -> str:
    return f"{t[0]:.{digits}f} – {t[1]:.{digits}f} (stress {t[2]:.{digits}f})"


def render() -> tuple[dict, str]:
    res = {name: size(o["rows"], o["capture_s"]) for name, o in OPTIONS.items()}
    sens = {f"full E1, rows {int(r / 1e6)} M": size(r, 896.0) for r in (173e6, 259e6)}  # the 216 M estimate -20 % / +20 % (A)
    rows = [
        ("kernel rows", lambda s: f"{s['rows'] / 1e6:.1f} M ({s['scale_vs_v3']:.2f} × V3)", "—"),
        ("flush, s", lambda s: fmt(s["flush_s"]), f"{V3_LIMITS['flush_s']:.0f}"),
        ("import, s", lambda s: fmt(s["import_s"]), f"{V3_LIMITS['import_s']:.0f}"),
        ("stop → report, s", lambda s: fmt(s["stop_to_report_s"]), f"{V3_LIMITS['stop_to_report_s']:.0f}"),
        ("GPU-held, s", lambda s: fmt(s["gpu_held_s"]), f"{V3_LIMITS['gpu_held_s']:.0f}"),
        ("GPU-held if the 600 s hard deadline acts, s", lambda s: f"{s['gpu_held_worst_case_under_v3_controller_s']:.0f}", f"{V3_LIMITS['gpu_held_s']:.0f}"),
        ("reservation as in V3 (held through closeout), min", lambda s: fmt(s["reservation_min"]), f"{V3_LIMITS['reservation_min']:.0f}"),
        ("reservation if deleted at verified GPU release, min", lambda s: fmt(s["reservation_if_deleted_at_gpu_release_min"]), f"{V3_LIMITS['reservation_min']:.0f}"),
        ("importer peak memory, GiB", lambda s: fmt(s["importer_rss_gib"], 1), f"{V3_LIMITS['importer_rss_gib']:.0f}"),
        ("MemAvailable minimum, GiB", lambda s: fmt(s["mem_available_min_gib"]), f"floor {V3_LIMITS['mem_available_floor_gib']:.0f}"),
        ("raw stream, GB", lambda s: fmt(s["stream_gb"], 2), "—"),
        ("report, GB", lambda s: fmt(s["report_gb"], 2), "—"),
        ("in-session export, GB", lambda s: fmt(s["export_gb"], 1), "—"),
        ("temporary peak (post-run, with retained streams), GB", lambda s: fmt(s["temp_peak_gb"], 1), f"{V3_LIMITS['temp_gb']:.0f}"),
        ("run root, GB", lambda s: fmt(s["run_root_gb"], 1), f"{V3_LIMITS['run_root_gb']:.0f}"),
        ("run root + archive, GB", lambda s: fmt(s["run_root_plus_archive_gb"], 1), f"{V3_LIMITS['run_root_plus_archive_gb']:.0f}"),
        ("export and parse after release, s", lambda s: fmt(s["post_release_export_parse_s"]), "—"),
        ("offline import, s", lambda s: fmt(s["offline_import_s"]), "—"),
        ("offline export, s", lambda s: fmt(s["offline_export_s"]), "—"),
    ]
    cols = list(res) + list(sens)
    allr = {**res, **sens}
    md = ["| quantity | " + " | ".join(cols) + " | V3 limit (unchanged) |", "|---|" + "---|" * (len(cols) + 1)]
    md += [f"| {label} | " + " | ".join(fn(allr[c]) for c in cols) + f" | {lim} |" for label, fn, lim in rows]
    fit = {key: {h: largest_rows_within(key, lim, h) for h in ("power", "line", "prop", "stress")} for key, lim in (("flush_s", V3_LIMITS["flush_s"]), ("import_s", V3_LIMITS["import_s"]), ("importer_rss_gib", V3_LIMITS["importer_rss_gib"]), ("run_root_gb", V3_LIMITS["run_root_gb"]))}
    md += ["", "Largest size at which each hypothesis stays inside the unchanged V3 limit (M rows):", "", "| limit | power | line | prop | stress |", "|---|---|---|---|---|"]
    md += [f"| {key} ≤ {V3_LIMITS[key]:.0f} | " + " | ".join(f"{fit[key][h] / 1e6:.0f}" for h in ("power", "line", "prop", "stress")) + " |" for key in fit]
    exps = {key: hypotheses(key, N3)["exponent_from_two_points"] for key in MEASURED}
    md += ["", "Exponent that the V2 and V3 points happen to give (two points; not a validated exponent): " + ", ".join(f"{k} {v:.2f}" for k, v in exps.items())]
    o = OBSERVED_E1_MAIN_V3
    chk = {h: hypotheses("flush_s", o["rows_estimate"])[h] for h in ("power", "line", "prop", "stress")}
    md += ["", f"Check against the one uncontrolled observation (e1_main_v3 flush {o['flush_s']:.0f} s at an estimated {o['rows_estimate'] / 1e6:.0f} M rows): " + ", ".join(f"{h} {v:.0f} s" for h, v in chk.items())
           + f". Exponent implied by V3 and that observation: {math.log(o['flush_s'] / MEASURED['flush_s'][1]) / math.log(o['rows_estimate'] / N3):.2f} at 216 M, {math.log(o['flush_s'] / MEASURED['flush_s'][1]) / math.log(173e6 / N3):.2f} at 173 M, {math.log(o['flush_s'] / MEASURED['flush_s'][1]) / math.log(259e6 / N3):.2f} at 259 M."]
    return {"options": res, "sensitivity": sens, "largest_rows_within_v3_limits": fit, "exponents_from_two_points": exps, "inputs": {"measured_v2_v3": MEASURED, "v3": V3, "observed_e1_main_v3": o, "stress_exponent": STRESS_EXPONENT, "v3_limits": V3_LIMITS, "startup_range_s": STARTUP_RANGE_S}}, "\n".join(md) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    data, md = render()
    print(md)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "E1-SCALING-MODEL.json").write_text(json.dumps(data, indent=1) + "\n")
        (a.out / "E1-SCALING-MODEL.md").write_text(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
