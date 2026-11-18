#!/usr/bin/env python3
"""V2 (forced release during import) inputs: the V1 workload and recipe, unchanged, under the patched launcher.

    python3 -I make_e1_inputs_fr.py            (from the e1 directory; stdlib only; writes configs_fr/, E1-V2-FR-INPUTS.json, INPUTS_fr.sha256)

Derivation (mechanical, from `configs_v2/e1_nsys_val/profile.json`, the recipe accepted at a62e40bd):
  * name / log_dir → e1_v2_forced_release under logs/dsv41_flash_gb300_tp4_e1_v2fr
  * nsys.capture_start_delay_seconds removed (profiler armed before the first request; as e1_b8_recovery and e1_main_v3)
  * workload.frontend.path → this tree's traces/e1_dry.csv (same bytes as the accepted trace; pinned by INPUTS_fr.sha256)
  * nsys.finalize_* / gpu_held_ceiling_s / preserve_raw_stream added: explicit bounds for the stop; they change nothing about the
    capture (`tests/test_profiler_lifecycle.py::test_launch_recipe_is_unchanged…`, preflight `recipe.*` checks)
  * no nsys.tmp_dir: the stream goes to the default TMPDIR (/tmp on the tray root NVMe) exactly as in every accepted capture, so
    the server environment is byte-identical to the accepted launch record
Everything else (server, workload, python_runtime, trace e1_dry.csv) is unchanged.
"""

from __future__ import annotations

# ruff: noqa: E501  (input generator: long documentation strings)

import hashlib
import json
from pathlib import Path

E1 = Path(__file__).resolve().parent
S = "/mnt/shared/home/dc/workspace/serving-studio"
PASS = "e1_v2_forced_release"
SRC = E1 / "configs_v2" / "e1_nsys_val" / "profile.json"
OUT_DIR = E1 / "configs_fr" / PASS
OUT = OUT_DIR / "profile.json"
TRACE = E1 / "traces" / "e1_dry.csv"

LIFECYCLE = {
    "finalize_soft_deadline_s": 300.0,  # escalation record; predicted finalisation ≈ 90 s (calibration below)
    "finalize_hard_deadline_s": 600.0,  # release action (server stop by identity; importer untouched)
    "finalize_stall_timeout_s": 300.0,
    "finalize_on_hard_deadline": "release_gpus",
    "finalize_session_close_timeout_s": 300.0,
    "finalize_max_extension_s": 120.0,  # the 60 s probe + at most one more operator extension
    "gpu_held_ceiling_s": 2700.0,  # 45 min from the launch: startup ≤ 30 + capture ≈ 3 + finalisation ≤ 10 + margin
    "preserve_raw_stream": True,
}

# Calibration of the stop (accepted captures; "Stopping profiler..." → "Profiler stopped." in the server log)
CALIBRATION = [
    {"pass": "e1_b8_recovery", "kernel_rows": 5_887_244, "stop_to_report_s": 45},
    {"pass": "e1_nsys_val", "kernel_rows": 6_363_588, "stop_to_report_s": 63},
    {"pass": "quadrant_nsys (run2)", "kernel_rows": 25_626_612, "stop_to_report_s": 163},
    {"pass": "quadrant_nsys (run1)", "kernel_rows": 25_609_900, "stop_to_report_s": 160},
    {"pass": "diverse_nsys_run2", "kernel_rows": 97_151_212, "stop_to_report_s": 567},
    {
        "pass": "e1_main_v3 (lost)",
        "kernel_rows": 216_000_000,
        "stop_to_report_s": None,
        "note": "> 1230 s when killed; import 49 % at ≈ 1080 s",
    },
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    src = json.loads(SRC.read_text())
    cfg = json.loads(json.dumps(src))
    cfg["name"] = f"dsv41_flash_1017_gb300_tp4_{PASS}"
    cfg["log_dir"] = f"{S}/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v2fr/{PASS}/profile"
    cfg["nsys"].pop("capture_start_delay_seconds", None)
    # the trace of THIS (pinned) tree, not the main clone's copy; same bytes (INPUTS_fr.sha256)
    cfg["workload"]["frontend"]["path"] = str(TRACE)
    cfg["nsys"].update(LIFECYCLE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cfg, indent=1) + "\n")

    rows = [r for r in TRACE.read_text().splitlines()[1:] if r.strip()]
    waves: dict[str, dict] = {}
    for r in rows:
        rid, inp, out, arr = r.split(",")
        key = rid.rsplit("_r", 1)[0]
        w = waves.setdefault(
            key,
            {
                "requests": 0,
                "input_len": int(inp),
                "output_len": int(out),
                "arrival_s": float(arr) / 1000.0,
            },
        )
        w["requests"] += 1
    # expected coverage: both waves are inside the capture (no delayed start). Iterations and rows from the two accepted
    # single-wave captures of the same cells: e1_b8_recovery (b=8 wave, 1040 iterations, 5 887 244 rows) and e1_nsys_val
    # (b=48 wave, 1120 iterations, 6 363 588 rows).
    expected_iterations = 1040 + 1120
    expected_rows = 5_887_244 + 6_363_588
    inputs = {
        "pass": PASS,
        "derived_from": {
            "config": "configs_v2/e1_nsys_val/profile.json",
            "sha256": sha256(SRC),
            "accepted_as": "e1_nsys_val (a62e40bd)",
        },
        "changes": [
            "name",
            "log_dir",
            "nsys.capture_start_delay_seconds removed (profiler armed before the first request)",
            "workload.frontend.path -> this tree's traces/e1_dry.csv (identical bytes to the accepted trace)",
            "nsys.finalize_soft_deadline_s / finalize_hard_deadline_s / finalize_stall_timeout_s / finalize_on_hard_deadline / "
            "finalize_session_close_timeout_s / finalize_max_extension_s / gpu_held_ceiling_s / preserve_raw_stream added",
        ],
        "unchanged": [
            "server",
            "python_runtime",
            "workload.* (trace e1_dry.csv, openai backend, stream false, hook on, warmup, trace-timed)",
            "nsys capture flags (== Phase 0 quadrant_nsys block)",
            "server environment (no tmp_dir)",
        ],
        "lifecycle": LIFECYCLE,
        "trace": {
            "path": "traces/e1_dry.csv",
            "sha256": sha256(TRACE),
            "requests": len(rows),
            "waves": waves,
        },
        "expected": {
            "iterations": expected_iterations,
            "iterations_band": [2100, 2250],
            "nvtx_per_iteration": 4,
            "kernel_rows": expected_rows,
            "kernel_rows_band": [int(expected_rows * 0.95), int(expected_rows * 1.05)],
            "pure_decode_iterations": {"b8": 1008, "b48": 928},
            "ranks": 4,
            "routing_files": len(rows),
            "request_timings": len(rows),
            "finalisation_s_predicted": 90,
            "finalisation_s_worst": 180,
            "report_bytes_predicted": 170_126_141 + 223_141_300,
            "sqlite_bytes_predicted": 874_708_992 + 1_088_155_648,
            "qdstrm_bytes_predicted": int(expected_rows * 23),
        },
        "ceilings": {
            "reservation_hard_min": 90,
            "gpu_held_ceiling_s": LIFECYCLE["gpu_held_ceiling_s"],
            "gpu_held_nominal_min": 28,
            "temporary_storage_gb": 5,
            "persistent_storage_gb": 5,
            "persistent_storage_with_archive_gb": 10,
            "startup_limit_min": 30,
            "capture_limit_min": 10,
        },
        "trigger": {
            "control_line": "release-gpus V2 forced release during import (pre-registered trigger: first observation of phase importing)",
            "condition": "first observation of phase `importing` in lifecycle.json by v2_release_probe.sh (registered in the pid registry; polls every 0.5 s)",
            "deterministic_in": "phase, not wall time: the line is written once, on the first poll that sees `importing`",
            "v1_reference": "V1: importing lasted 27 s (03:22:16 → 03:22:43Z) for the same workload; the trigger needs ≤ 2 s (probe poll 0.5 s + controller poll 1 s)",
        },
        "limits": {
            "gpu_release_deadline_s": 90,
            "gpu_watch_interval_s": 2,
            "session_close_timeout_s": LIFECYCLE["finalize_session_close_timeout_s"],
        },
        "outcomes": {
            "A": "the importer finishes after the server is gone: phase report_written; recovered trace must equal the in-session export",
            "B": "the importer ends without a report: phase stop_failed; recovery from the preserved stream must give a complete trace",
            "neither": "importer still running after the session-close timeout, or GPUs not released: FAIL; registry cleanup procedure",
        },
        "calibration_stop_to_report": CALIBRATION,
        "config_sha256": sha256(OUT),
    }
    (E1 / "E1-V2-FR-INPUTS.json").write_text(json.dumps(inputs, indent=1) + "\n")
    lines = [
        f"{sha256(TRACE)}  traces/e1_dry.csv",
        f"{sha256(OUT)}  configs_fr/{PASS}/profile.json",
        f"{sha256(SRC)}  configs_v2/e1_nsys_val/profile.json",
    ]
    (E1 / "INPUTS_fr.sha256").write_text("\n".join(lines) + "\n")
    print(
        f"wrote {OUT.relative_to(E1)} ({sha256(OUT)[:16]}), E1-V2-FR-INPUTS.json, INPUTS_fr.sha256; waves {json.dumps(waves)}"
    )


if __name__ == "__main__":
    main()
