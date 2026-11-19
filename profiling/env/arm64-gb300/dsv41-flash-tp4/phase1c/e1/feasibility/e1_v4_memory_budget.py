#!/usr/bin/env python3
"""Host-memory budget for the V4 preflight, from the preserved V3 run (CPU-only; reads records, writes two files).

    python3 -I feasibility/e1_v4_memory_budget.py --v3-run <V3 run root>/e1_v3_scale --inputs E1-V4-SCALE-INPUTS.json
        --out-json feasibility/E1-V4-MEMORY-BUDGET.json --out-md feasibility/E1-V4-MEMORY-BUDGET.md

The preflight minimum is importer ceiling + runtime floor + an allowance for everything else that the run holds while the
importer is alive (server, engine, workers, launcher, Nsight CLI and agent, driver pinnings): the difference between
MemAvailable before the launch and at the stop request. This script measures that difference in V3 from the memory
sampler's record, lists what the V3 number does not cover, and evaluates the registered minimum at its extremes. Nothing is
fitted; the one projection (growth during the capture, continued over the longer V4 capture) is marked as such.
Stdlib only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import json
import sys
from pathlib import Path

KB = 1048576  # kB per GiB


def measure(run: Path) -> dict:
    rows = [json.loads(line) for line in (run / "mem_observations.jsonl").read_text().splitlines() if line.strip()]
    lc = json.loads((run / "profile" / "nsys" / "lifecycle.json").read_text())
    watch = json.loads((run / "mem_watch.json").read_text())
    launch, stop, released = lc["launch_started_at"], lc["stop_requested_at"], lc["gpu"]["released_at"]
    phases = {h["phase"]: h["at"] for h in lc["phase_history"]}

    def avail(t0: float, t1: float) -> list[float]:
        return [r["mem_available_kb"] / KB for r in rows if t0 <= r["at"] < t1 and r.get("mem_available_kb")]

    idle = max(avail(rows[0]["at"], launch + 5))
    load = avail(launch, launch + 300)
    steady = avail(launch + 300, launch + 1300)
    at_stop = avail(stop - 5, stop + 5)
    importing = avail(phases["importing"], phases["report_written"])
    after = avail(released + 30, rows[-1]["at"] + 1)
    series = [(round(r["at"] - launch), r["mem_available_kb"] / KB) for r in rows if launch + 300 <= r["at"] <= stop]
    onset = next((t for t, v in series if v < min(steady) - 0.5), None)  # first sample 0.5 GiB below the steady range
    log = (run / "run.log").read_text(errors="replace") if (run / "run.log").exists() else ""
    m = {
        "run": run.name, "samples": len(rows), "interval_s": watch.get("interval_s"), "max_gap_s": watch.get("max_gap_s"), "mem_total_gib": round(rows[0]["mem_total_kb"] / KB, 1),
        "idle_before_launch_gib": round(idle, 1),
        "load_transient_min_gib": round(min(load), 1), "load_transient_below_idle_gib": round(idle - min(load), 1),
        "steady_after_load_gib": [round(min(steady), 1), round(max(steady), 1)], "steady_below_idle_gib": round(idle - min(steady), 1),
        "decline_onset_s_after_launch": onset, "stop_requested_s_after_launch": round(stop - launch),
        "at_stop_request_gib": round(min(at_stop), 1), "at_stop_request_below_idle_gib": round(idle - min(at_stop), 1),
        "growth_onset_to_stop_gib": round(min(steady) - min(at_stop), 1),
        "during_import_min_gib": round(min(importing), 1), "importer_peak_rss_gib": round(watch["importer_peak_rss_kb"] / KB, 1),
        "run_minimum_gib": round(watch["min_available_kb"] / KB, 1), "run_minimum_s_after_launch": round(watch["min_available_at"] - launch),
        "after_release_min_gib": round(min(after), 1), "sampler_covers_export_and_parse_s": round(rows[-1]["at"] - released),
        "workload_lines_in_run_log": [line.strip()[:60] for line in log.splitlines() if line.startswith(("[replay] warmup", "[profile] frontend ready", "[profile] workload done"))][:6],
    }
    return m


def budget(m: dict, inputs: dict) -> dict:
    c, mb = inputs["ceilings"], inputs["memory_budget"]
    imp, floor, server, minimum = c["importer_rss_ceiling_gb"], c["host_mem_available_floor_gb"], mb["server_allowance_gib"], c["host_mem_available_preflight_min_gb"]

    def left(preflight: float, footprint: float, importer: float) -> float:
        return round(preflight - footprint - importer, 1)

    v3_stop, v3_load = m["at_stop_request_below_idle_gib"], m["load_transient_below_idle_gib"]
    span = (m["stop_requested_s_after_launch"] - m["decline_onset_s_after_launch"]) if m["decline_onset_s_after_launch"] is not None else 0
    projected = round(m["growth_onset_to_stop_gib"] / span * 160, 1) if span > 0 else 0.0  # a projection from one run (see `unmeasured`), shown as a case, not part of the sum
    cases = [
        {"case": "registered minimum, measured V3 footprint at the stop request, importer at its ceiling", "preflight": minimum, "footprint": v3_stop, "importer": imp},
        {"case": "registered minimum, measured footprint plus the growth projected for the longer V4 capture, importer at its ceiling", "preflight": minimum, "footprint": round(v3_stop + projected, 1), "importer": imp},
        {"case": "registered minimum, measured footprint plus twice the projected growth, importer at its ceiling", "preflight": minimum, "footprint": round(v3_stop + 2 * projected, 1), "importer": imp},
        {"case": "earlier proposal (488), measured footprint, importer at its ceiling", "preflight": 488, "footprint": v3_stop, "importer": imp},
        {"case": "earlier proposal (488), measured footprint plus the projected growth, importer at its ceiling", "preflight": 488, "footprint": round(v3_stop + projected, 1), "importer": imp},
        {"case": "first proposal (448), measured footprint, importer at its ceiling", "preflight": 448, "footprint": v3_stop, "importer": imp},
        {"case": "registered minimum, load transient (no importer exists then)", "preflight": minimum, "footprint": v3_load, "importer": 0},
        {"case": "registered minimum, measured footprint, importer as predicted for V4 (stress 35)", "preflight": minimum, "footprint": v3_stop, "importer": 35},
        {"case": "tray idle as measured before V3, measured footprint, importer at its ceiling", "preflight": m["idle_before_launch_gib"], "footprint": v3_stop, "importer": imp},
    ]
    for x in cases:
        x["available_left_gib"] = left(x["preflight"], x["footprint"], x["importer"])
        x["at_or_above_floor"] = x["available_left_gib"] >= floor
    return {"importer_ceiling_gib": imp, "runtime_floor_gib": floor, "server_allowance_gib": server, "preflight_minimum_gib": minimum, "sum_gib": imp + floor + server, "minimum_equals_sum": minimum == imp + floor + server,
            "allowance_covers_measured_v3_footprint": server >= v3_stop, "allowance_margin_over_measured_gib": round(server - v3_stop, 1), "projected_growth_gib": projected, "allowance_margin_over_measured_plus_projected_growth_gib": round(server - v3_stop - projected, 1), "cases": cases}


def unmeasured(m: dict) -> list[dict]:
    rate = None
    if m["decline_onset_s_after_launch"] is not None and m["stop_requested_s_after_launch"] > m["decline_onset_s_after_launch"]:
        rate = m["growth_onset_to_stop_gib"] / (m["stop_requested_s_after_launch"] - m["decline_onset_s_after_launch"])
    extra_s = 160  # two more copies of the trace: 2 x 80 s
    return [
        {"consumer": "growth of the run's footprint during the four additional waves of V4", "status": "NOT MEASURED",
         "note": f"in V3 the footprint grew by {m['growth_onset_to_stop_gib']} GiB between {m['decline_onset_s_after_launch']} s and the stop request at {m['stop_requested_s_after_launch']} s after the launch (warm-up and capture). If growth continued at the same mean rate over the additional {extra_s} s of V4 capture, that would be about {None if rate is None else round(rate * extra_s, 1)} GiB more: a projection from one run, not a measurement; the registered allowance exceeds the V3 measurement by more than that projection (table above), but the growth itself stays unmeasured until V4 runs"},
        {"consumer": "importer at the V4 size", "status": "bounded, not measured at this size", "note": f"V3 peak {m['importer_peak_rss_gib']} GiB; predicted 28 – 31 (35) GiB; bounded by the 128 GiB ceiling (sampler → supervisor), which is the value in the budget"},
        {"consumer": "dirty page cache of the stream and its preserved copy in /tmp (ext4 on the tray's NVMe, not memory-backed)", "status": "inside the V3 measurement at half the size", "note": "0.70 GB in V3, about 1.4 GB predicted for V4; reclaimable after writeback"},
        {"consumer": "other tenants of the node: DaemonSet pods, other sessions of the shared account, CPU-only pods the scheduler may place", "status": "NOT CONTROLLED", "note": "the preflight is one sample; during the run the 64 GiB floor (memory sampler → supervisor) is the protection, not the preflight"},
        {"consumer": "launcher export and parse, post-run offline import and export", "status": "not concurrent with the server", "note": f"they run after the GPU release; MemAvailable minimum after the release in V3 {m['after_release_min_gib']} GiB (sampler covered {m['sampler_covers_export_and_parse_s']} s after the release)"},
        {"consumer": "samplers, supervisor, driver", "status": "inside the V3 measurement", "note": "a few MiB each"},
    ]


def render(m: dict, b: dict, u: list[dict]) -> str:
    out = ["# V4 preflight host-memory budget (CPU-only, from the preserved V3 memory record)", "",
           "Generated by `feasibility/e1_v4_memory_budget.py`; numbers in GiB (1024^3 B). The V3 record is read only.", "",
           "## Measured in V3 (host MemAvailable, 1 s sampler)", "", "| quantity | value |", "|---|---|"]
    for k, label in (("idle_before_launch_gib", "idle before the launch"), ("load_transient_min_gib", "minimum during model load (transient, no importer)"), ("load_transient_below_idle_gib", "… below idle"),
                     ("steady_after_load_gib", "steady after load, until warm-up (min, max)"), ("steady_below_idle_gib", "… below idle"), ("at_stop_request_gib", "at the stop request"),
                     ("at_stop_request_below_idle_gib", "… below idle = the run's footprint when the importer starts"), ("growth_onset_to_stop_gib", "growth from the steady level to the stop request"),
                     ("during_import_min_gib", "minimum during the import"), ("importer_peak_rss_gib", "importer peak RSS"), ("run_minimum_gib", "minimum of the whole run"), ("after_release_min_gib", "minimum after the GPU release (export and parse)")):
        out.append(f"| {label} | {m[k]} |")
    out += ["", "## Registered budget", "",
            f"preflight minimum {b['preflight_minimum_gib']} = importer ceiling {b['importer_ceiling_gib']} + runtime floor {b['runtime_floor_gib']} + allowance for the run {b['server_allowance_gib']} "
            f"(sum {b['sum_gib']}; equal: {b['minimum_equals_sum']}). The allowance covers the footprint measured in V3 at the stop request: {b['allowance_covers_measured_v3_footprint']} (margin {b['allowance_margin_over_measured_gib']} GiB).", "",
            "| case | MemAvailable at the preflight | footprint | importer | left | ≥ floor |", "|---|---|---|---|---|---|"]
    for x in b["cases"]:
        out.append(f"| {x['case']} | {x['preflight']} | {x['footprint']} | {x['importer']} | {x['available_left_gib']} | {'yes' if x['at_or_above_floor'] else '**no**'} |")
    out += ["", "## Not covered by the V3 measurement", "", "| consumer | status | note |", "|---|---|---|"]
    out += [f"| {x['consumer']} | {x['status']} | {x['note']} |" for x in u]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v3-run", required=True, type=Path)
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    a = ap.parse_args(argv)
    m = measure(a.v3_run)
    b, u = budget(m, json.loads(a.inputs.read_text())), unmeasured(m)
    a.out_json.write_text(json.dumps({"measured_v3": m, "budget": b, "not_covered_by_the_v3_measurement": u}, indent=1) + "\n")
    a.out_md.write_text(render(m, b, u))
    print(f"V3 footprint at the stop request {m['at_stop_request_below_idle_gib']} GiB; allowance {b['server_allowance_gib']}; minimum {b['preflight_minimum_gib']} = sum {b['sum_gib']}; cases below the floor: {[x['case'] for x in b['cases'] if not x['at_or_above_floor']]}")
    return 0 if b["minimum_equals_sum"] and b["allowance_covers_measured_v3_footprint"] else 1


if __name__ == "__main__":
    sys.exit(main())
