#!/usr/bin/env python3
"""V3 (intermediate trace size) DRAFT inputs: the accepted V1/V2 recipe with the accepted dry-run trace played twice.

    python3 -I make_e1_inputs_v3scale.py       (from the e1 directory; stdlib only)

Writes traces/e1_v3_scale.csv, configs_v3scale/e1_v3_scale/profile.json and E1-V3-SCALE-INPUTS.draft.json. Nothing here is
pinned or authorized; the draft packet E1-V3-SCALE-PACKET-DRAFT.md explains every number. (`configs_v3/` and
`make_e1_inputs_v3.py` belong to the lost e1_main_v3 capture and are not touched.)

Derivation (mechanical):
  * config = configs_fr/e1_v2_forced_release/profile.json (the V2 config, itself the accepted e1_nsys_val recipe with the
    lifecycle bounds) with name, log_dir and workload.frontend.path changed. Lifecycle bounds unchanged.
  * trace = the 56 rows of traces/e1_dry.csv (waves b=8 at 0 s and b=48 at 18.6272 s), then the same 56 rows again as
    wave index 1, shifted by SECOND_PAIR_OFFSET_MS. Four waves, 112 requests, every one an already accepted cell.
"""

from __future__ import annotations

# ruff: noqa: E501  (input generator: long documentation strings)

import hashlib
import json
from pathlib import Path

E1 = Path(__file__).resolve().parent
S = "/mnt/shared/home/dc/workspace/serving-studio"
PASS = "e1_v3_scale"
SRC = E1 / "configs_fr" / "e1_v2_forced_release" / "profile.json"
SRC_TRACE = E1 / "traces" / "e1_dry.csv"
OUT_DIR = E1 / "configs_v3scale" / PASS
OUT = OUT_DIR / "profile.json"
TRACE = E1 / "traces" / "e1_v3_scale.csv"
# V2 replayed the two waves in 51.19 s (drive summary, replay_end - replay_start); the second pair starts 80 s after the first
SECOND_PAIR_OFFSET_MS = 80000.0

# measured in V2 (runs/v2/run): one pass over the 56-row trace
V2 = {"iterations": 2160, "nvtx_forward_ranges": 8640, "kernel_rows": 12_250_832, "qdstrm_bytes": 348_288_111, "report_bytes": 391_797_303, "sqlite_bytes": 1_964_322_816, "run_dir_gb": 3.0, "replay_s": 51.19}

# stop -> report, all with Nsight 2025.6.3 and the same capture flags; phase split known only where the controller recorded it
CALIBRATION = [
    {"pass": "e1_b8_recovery", "kernel_rows": 5_887_244, "stop_to_report_s": 45},
    {"pass": "e1_nsys_val", "kernel_rows": 6_363_588, "stop_to_report_s": 63},
    {"pass": "e1_v1_lifecycle", "kernel_rows": 12_250_832, "stop_to_report_s": 90.8, "flush_s": 63.8, "import_s": 27.0},
    {"pass": "e1_v2_forced_release", "kernel_rows": 12_250_832, "stop_to_report_s": 90.8, "flush_s": 61.9, "import_s": 28.9, "note": "import end = report mtime; server signalled 1.2 s into the import"},
    {"pass": "quadrant_nsys run1 (Phase 0 recipe)", "kernel_rows": 25_609_900, "stop_to_report_s": 160},
    {"pass": "quadrant_nsys run2 (Phase 0 recipe)", "kernel_rows": 25_626_612, "stop_to_report_s": 163},
    {"pass": "diverse_nsys_run2 (Phase 0 recipe)", "kernel_rows": 97_151_212, "stop_to_report_s": 567},
    {"pass": "e1_main_v3 (lost)", "kernel_rows": 216_000_000, "stop_to_report_s": None, "note": "> 1230 s when destroyed; import 49 % at about 1080 s"},
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    header, *rows = [r for r in SRC_TRACE.read_text().splitlines() if r.strip()]
    out_rows, waves = [], {}
    for index, offset in ((0, 0.0), (1, SECOND_PAIR_OFFSET_MS)):
        for r in rows:
            rid, inp, out, arr = r.split(",")
            assert "_w0_r" in rid, rid
            rid = rid.replace("_w0_r", f"_w{index}_r")
            arrival = round(float(arr) + offset, 1)
            out_rows.append(f"{rid},{inp},{out},{arrival}")
            w = waves.setdefault(rid.rsplit("_r", 1)[0], {"requests": 0, "input_len": int(inp), "output_len": int(out), "arrival_s": arrival / 1000.0})
            w["requests"] += 1
    TRACE.write_text("\n".join([header, *out_rows]) + "\n")

    cfg = json.loads(SRC.read_text())
    cfg["name"] = f"dsv41_flash_1017_gb300_tp4_{PASS}"
    cfg["log_dir"] = f"{S}/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v3scale/{PASS}/profile"
    cfg["workload"]["frontend"]["path"] = str(TRACE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cfg, indent=1) + "\n")

    rows_expected = 2 * V2["kernel_rows"]
    inputs = {
        "status": "DRAFT - not pinned, not frozen, authorizes nothing",
        "pass": PASS,
        "derived_from": {"config": "configs_fr/e1_v2_forced_release/profile.json", "sha256": sha256(SRC), "trace": "traces/e1_dry.csv", "trace_sha256": sha256(SRC_TRACE)},
        "changes": ["name", "log_dir", "workload.frontend.path -> traces/e1_v3_scale.csv (the accepted 56 rows twice; second pair +80 s, wave index 1)"],
        "unchanged": ["server", "python_runtime", "workload.* except the trace path", "nsys capture flags", "nsys lifecycle bounds (V1/V2 values)", "server environment (no tmp_dir)"],
        "lifecycle": {k: v for k, v in cfg["nsys"].items() if k.startswith("finalize_") or k in ("gpu_held_ceiling_s", "preserve_raw_stream")},
        "trace": {"path": "traces/e1_v3_scale.csv", "sha256": sha256(TRACE), "requests": len(out_rows), "waves": waves, "second_pair_offset_ms": SECOND_PAIR_OFFSET_MS},
        "expected": {
            "basis": "2 x the V2 measurement of the same 56 rows (same cells, same recipe); not an extrapolation in trace content",
            "iterations": 2 * V2["iterations"],
            "iterations_band": [4200, 4500],
            "nvtx_per_iteration": 4,
            "nvtx_forward_ranges": 2 * V2["nvtx_forward_ranges"],
            "kernel_rows": rows_expected,
            "kernel_rows_band": [int(rows_expected * 0.95), int(rows_expected * 1.05)],
            "kernel_rows_vs_25.6M": round(rows_expected / 25.6e6, 3),
            "pure_decode_iterations_per_wave": {"b8": 1008, "b48": 928},
            "ranks": 4,
            "routing_files": len(out_rows),
            "request_timings": len(out_rows),
            "replay_s": round(SECOND_PAIR_OFFSET_MS / 1000.0 + V2["replay_s"], 1),
            "qdstrm_bytes": 2 * V2["qdstrm_bytes"],
            "report_bytes": 2 * V2["report_bytes"],
            "sqlite_bytes": 2 * V2["sqlite_bytes"],
            "run_dir_gb": 2 * V2["run_dir_gb"],
            "importer_peak_rss_gb": {"measured_offline_at_12.25M_rows": 8.7, "assumed_upper_for_planning": 64, "note": "scaling with rows is not known; 61 GB was sampled at >= 49 % of the 216 M-row import"},
        },
        "finalisation": {
            "stop_to_report_s_nearest_measured": [160, 163],
            "nearest_measured_at_rows": [25_609_900, 25_626_612],
            "note": "taken from the two captures of nearly the same size, not scaled from V2's 27 s import; their recipe differs (Phase 0 backend, no routing hook), so this is a prediction with unknown error, and the flush/import split at this size has never been recorded",
            "predicted_stop_to_report_s": [140, 190],
            "flush_s_range_considered": [60, 140],
            "import_s_range_considered": [27, 110],
        },
        "phase_deadlines_s": {"flush_stop_to_importing": 300, "import_importing_to_importer_exit": 300, "stop_to_report_written": 600, "session_close_after_report": 300},
        "ceilings": {"reservation_hard_min": 90, "gpu_held_ceiling_s": cfg["nsys"]["gpu_held_ceiling_s"], "gpu_held_nominal_s": 1800, "gpu_release_deadline_s": 90, "temporary_storage_gb": 10, "persistent_storage_gb": 8, "persistent_storage_with_archive_gb": 16, "host_mem_available_min_gb": 128, "startup_limit_min": 30, "capture_limit_min": 10},
        "calibration_stop_to_report": CALIBRATION,
        "config_sha256": sha256(OUT),
    }
    (E1 / "E1-V3-SCALE-INPUTS.draft.json").write_text(json.dumps(inputs, indent=1) + "\n")
    print(f"wrote {TRACE.relative_to(E1)} ({sha256(TRACE)[:16]}, {len(out_rows)} requests), {OUT.relative_to(E1)} ({sha256(OUT)[:16]}), E1-V3-SCALE-INPUTS.draft.json; waves {json.dumps(waves)}")


if __name__ == "__main__":
    main()
