#!/usr/bin/env python3
"""V4 (intermediate lifecycle scale, about 49 M kernel rows) inputs: the accepted V1/V2 recipe with the accepted dry-run trace
played four times.

    python3 -I make_e1_inputs_v4scale.py           (stdlib only; deterministic: a second run rewrites the same bytes)
    python3 -I make_e1_inputs_v4scale.py --check   (writes nothing in the e1 directory: generates into a temporary directory and
                                                    compares with the five files on disk; exit 1 on any difference)

Writes traces/e1_v4_scale.csv, configs_v4scale/e1_v4_scale/profile.json, E1-V4-SCALE-INPUTS.json, INPUTS_v4scale.sha256 and
checks/e1_identity_policy_v4scale.json. The proposed packet E1-V4-SCALE-PACKET.md explains every number; nothing here
authorizes a launch.

Derivation (mechanical):
  * config = configs_fr/e1_v2_forced_release/profile.json (the V2 config, itself the accepted e1_nsys_val recipe with the
    lifecycle bounds) with name, log_dir and workload.frontend.path changed. Lifecycle bounds unchanged. The result differs
    from the V3 config in the same three fields only.
  * trace = the 56 rows of traces/e1_dry.csv (waves b=8 at 0 s and b=48 at 18.6272 s) COPIES times: copy i has the wave
    index i in every request id and every arrival shifted by i * COPY_OFFSET_MS. Eight waves, 224 requests, two accepted
    cell kinds. The repeats are lifecycle-scale replicates of the same two cells inside one session; they are not
    independent scientific observations and must not be counted as such.
  * identity policy = checks/e1_identity_policy_v1.json with the rule for workload.frontend.path replaced by the
    `derived_trace` rule of checks/e1_launch_identity_v4scale.py and the V4 pins added.
"""

from __future__ import annotations

# ruff: noqa: E501  (input generator: long documentation strings)

import hashlib
import json
import sys
import tempfile
from pathlib import Path

E1 = Path(__file__).resolve().parent
S = "/mnt/shared/home/dc/workspace/serving-studio"
PASS = "e1_v4_scale"
SRC = E1 / "configs_fr" / "e1_v2_forced_release" / "profile.json"
SRC_TRACE = E1 / "traces" / "e1_dry.csv"
POLICY_V1 = E1 / "checks" / "e1_identity_policy_v1.json"
GENERATED = ("traces/e1_v4_scale.csv", f"configs_v4scale/{PASS}/profile.json", "E1-V4-SCALE-INPUTS.json", "INPUTS_v4scale.sha256", "checks/e1_identity_policy_v4scale.json")
COPIES = 4
COPY_OFFSET_MS = 80000.0  # as V3: V2 replayed the two waves in 51.19 s; each further copy starts 80 s after the previous one

# measured (feasibility/E1-ROWS-PER-ITERATION.md): kernel rows of one b=8 wave and one b=48 wave of this trace. The two numbers
# were identical in V2, in V3 wave 0 and in V3 wave 1.
WAVE = {"b8": {"iterations": 1040, "kernel_rows": 5_887_244, "pure_decode": 1008}, "b48": {"iterations": 1120, "kernel_rows": 6_363_588, "pure_decode": 928}}
V2 = {"iterations": 2160, "nvtx_forward_ranges": 8640, "kernel_rows": 12_250_832, "qdstrm_bytes": 348_288_111, "report_bytes": 391_797_303, "sqlite_bytes": 1_964_322_816, "replay_s": 51.19}
V3 = {"iterations": 4320, "kernel_rows": 24_501_664, "qdstrm_bytes": 661_640_958, "report_bytes": 792_004_886, "sqlite_bytes": 3_954_987_008, "flush_s": 112.0, "import_s": 53.0, "stop_to_report_s": 166.2,
      "importer_peak_rss_gib": 15.61, "run_root_peak_gb": 6.311, "temporary_peak_gb": 5.219, "gpu_held_s": 1833, "reservation_min": 60.2, "mem_available_min_gib": 547.2}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(root: Path) -> str:
    """Write the five generated files under `root` (the e1 directory, or a scratch directory for --check)."""
    out_dir = root / "configs_v4scale" / PASS
    cfg_path, trace, policy_path = out_dir / "profile.json", root / "traces" / "e1_v4_scale.csv", root / "checks" / "e1_identity_policy_v4scale.json"
    trace.parent.mkdir(parents=True, exist_ok=True)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    assert WAVE["b8"]["kernel_rows"] + WAVE["b48"]["kernel_rows"] == V2["kernel_rows"] and WAVE["b8"]["iterations"] + WAVE["b48"]["iterations"] == V2["iterations"]
    header, *rows = [r for r in SRC_TRACE.read_text().splitlines() if r.strip()]
    out_rows, waves = [], {}
    for index in range(COPIES):
        for r in rows:
            rid, inp, out, arr = r.split(",")
            assert rid.count("_w0_r") == 1, rid
            rid = rid.replace("_w0_r", f"_w{index}_r")
            arrival = round(float(arr) + index * COPY_OFFSET_MS, 1)
            out_rows.append(f"{rid},{inp},{out},{arrival}")
            w = waves.setdefault(rid.rsplit("_r", 1)[0], {"requests": 0, "input_len": int(inp), "output_len": int(out), "arrival_ms": arrival})
            w["requests"] += 1
    trace.write_text("\n".join([header, *out_rows]) + "\n")

    cfg = json.loads(SRC.read_text())
    cfg["name"] = f"dsv41_flash_1017_gb300_tp4_{PASS}"
    cfg["log_dir"] = f"{S}/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v4scale/{PASS}/profile"
    cfg["workload"]["frontend"]["path"] = str(E1 / "traces" / "e1_v4_scale.csv")  # the registered location, also when checking in a scratch directory
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=1) + "\n")

    rows_expected = COPIES * V2["kernel_rows"]
    iterations = COPIES * V2["iterations"]
    plan = [{"b": b, "kv": 32768, "wave": i, "arrival_s": round(i * COPY_OFFSET_MS / 1000.0 + (0.0 if b == 8 else 18.6), 1), "pure_decode_iters_min": WAVE[f"b{b}"]["pure_decode"]} for i in range(COPIES) for b in (8, 48)]
    inputs = {
        "status": "inputs of the PROPOSED V4 packet; authorizes nothing; V4 GPU execution is NO GO",
        "pass": PASS,
        "purpose": "profiler lifecycle scaling and operational safety at twice the V3 size. Not scientific identification: the eight waves are replicates of two accepted cells within one session.",
        "derived_from": {"config": "configs_fr/e1_v2_forced_release/profile.json", "sha256": sha256(SRC), "trace": "traces/e1_dry.csv", "trace_sha256": sha256(SRC_TRACE)},
        "changes": ["name", "log_dir", f"workload.frontend.path -> traces/e1_v4_scale.csv (the accepted 56 rows {COPIES} times; copy i has wave index i and +i x 80 s)"],
        "unchanged": ["server", "python_runtime", "workload.* except the trace path", "nsys capture flags", "nsys lifecycle bounds (V1/V2/V3 values)", "server environment (no tmp_dir)"],
        "lifecycle": {k: v for k, v in cfg["nsys"].items() if k.startswith("finalize_") or k in ("gpu_held_ceiling_s", "preserve_raw_stream")},
        "trace": {"path": "traces/e1_v4_scale.csv", "sha256": sha256(trace), "requests": len(out_rows), "source_rows": len(rows), "copies": COPIES, "copy_offset_ms": COPY_OFFSET_MS, "waves": waves,
                  "cells": {"b8_kv32768": {"waves": COPIES, "requests_per_wave": 8}, "b48_kv32768": {"waves": COPIES, "requests_per_wave": 48}}},
        "expected": {
            "basis": f"{COPIES} x the V2 measurement of the same 56 rows. Kernel rows per wave were identical in V2, V3 wave 0 and V3 wave 1 (b=8: 5 887 244, b=48: 6 363 588), so this is a prediction from repeated measurements of the same cells, not an extrapolation in trace content",
            "requests": len(out_rows),
            "iterations": iterations,
            "iterations_band": [iterations - 240, iterations + 360],
            "nvtx_per_iteration": 4,
            "nvtx_forward_ranges": 4 * iterations,
            "kernel_rows": rows_expected,
            "kernel_rows_band": [int(rows_expected * 0.95), int(rows_expected * 1.05)],
            "kernel_rows_per_wave": {"b8": WAVE["b8"]["kernel_rows"], "b48": WAVE["b48"]["kernel_rows"]},
            "kernel_rows_vs_v3": round(rows_expected / V3["kernel_rows"], 3),
            "pure_decode_iterations_per_wave": {"b8": WAVE["b8"]["pure_decode"], "b48": WAVE["b48"]["pure_decode"]},
            "ranks": 4,
            "routing_files": len(out_rows),
            "request_timings": len(out_rows),
            "replay_s": round((COPIES - 1) * COPY_OFFSET_MS / 1000.0 + V2["replay_s"], 1),
            "qdstrm_bytes": 2 * V3["qdstrm_bytes"],
            "report_bytes": 2 * V3["report_bytes"],
            "sqlite_bytes": 2 * V3["sqlite_bytes"],
            "importer_peak_rss_gib": {"measured_at_12.25M_rows_offline": 8.7, "measured_at_24.5M_rows": V3["importer_peak_rss_gib"], "predicted": [28.0, 34.6], "note": "prediction, not a gate; feasibility/E1-SCALING-MODEL.md"},
        },
        # predictions registered before the run; none is a gate (the gates use the deadlines and ceilings below)
        "predictions": {"source": "feasibility/E1-SCALING-MODEL.md, column 'intermediate ~49 M': min - max over three curves through the V2 and V3 points, then the stress case", "flush_s": [203, 224, 249], "import_s": [97, 106, 118],
                        "stop_to_report_s": [301, 331, 367], "gpu_held_s": [2092, 2275, 2311], "reservation_min": [75, 78, 81], "temporary_peak_gb": [10.4, 10.5, 11.4], "run_root_gb": [12.6, 12.6, 14.0], "run_root_plus_archive_gb": [26.5, 26.6, 29.5],
                        "export_and_parse_after_release_s": [1220, 1226, 1360], "mem_available_min_gib": [528, 535]},
        # clocks: checks/e1_v3scale_gates.py docstring (the V4 gates use that code unchanged). A phase passes at or below its deadline.
        "phase_deadlines_s": {"stop_accepted": 30, "flush_stop_to_importing": 300, "import_importer_first_seen_to_gone": 300, "stop_to_report_written": 600, "session_close_after_server_sigterm": 300},
        "limits": {"gpu_release_deadline_s": 90, "gpu_watch_interval_s": 2, "gpu_watch_max_gap_s": 10, "sampler_max_gap_s": 30, "import_probe_interval_s": 0.5, "import_probe_min_progress": 2, "mem_watch_interval_s": 1, "storage_watch_interval_s": 2,
                   "supervisor_interval_s": 1, "supervisor_ack_s": 10, "supervisor_stale_s": 60, "supervisor_cleanup_grace_s": 30, "observer_max_s": {"gpu_sampler": 3300, "import_probe": 3300, "memory_sampler": 5400, "storage_sampler": 5400, "supervisor": 5400}},
        # the one line the registered supervisor may write; any accepted control line fails gate G3, so an abort is always a FAIL
        "supervisor": {"control_line": "release-gpus V4 resource-limit abort (registered supervisor response)", "registry": "pids/supervisor.tsv", "acts_on_registry": "pids/owned.tsv"},
        # reservation budget (E1-V4-TIMELINE.md): the three parts add up to the 90 min ceiling
        "reservation": {"prelaunch_max_age_s": 600, "supervisor_guard_s": 4680, "closeout_and_deletion_budget_s": 720, "early_release_after_verified_gpu_release": {"enabled": False, "note": "designed and CPU-tested only (checks/e1_reservation_release.py); not authorized"}},
        "ceilings": {"reservation_hard_min": 90, "gpu_held_ceiling_s": cfg["nsys"]["gpu_held_ceiling_s"], "gpu_held_nominal_s": 2300, "temporary_storage_gb": 15, "persistent_storage_gb": 18, "persistent_storage_with_archive_gb": 38,
                     "host_mem_available_preflight_min_gb": 448, "host_mem_available_floor_gb": 64, "importer_rss_ceiling_gb": 128,
                     "memory_basis": "GiB (1024^3 B). Unchanged from V3: importer ceiling 128 GiB, floor 64 GiB, preflight 448 GiB = ceiling + floor + 256 GiB for the server (measured server footprint in V3: 285 GiB below idle; MemAvailable minimum 547.2 GiB)",
                     "storage_basis": "review decision of 2026-11-19 on 598d4a2d: temporary 15 GB, run directory 18 GB, run directory plus archive 38 GB", "startup_limit_min": 30, "capture_limit_min": 10},
        "identity": {"policy": "checks/e1_identity_policy_v4scale.json", "policy_version": "v4scale-1", "checker": "e1_launch_identity_v4scale.py", "reference_run": "e1_nsys_val (accepted, a62e40bd)"},
        # read by checks/e1_completeness_v2.py (--plan dry) and checks/e1_nsys_validation.py: the eight waves of this trace
        "design": {"OUT_LEN": 1024, "MIN_PURE_DECODE": 400},
        "dry": {"requests": len(out_rows), "trace_span_s": round((COPIES - 1) * COPY_OFFSET_MS / 1000.0 + 72.7, 1), "plan": plan},
        "calibration": {"v2": V2, "v3": V3},
        "config_sha256": sha256(cfg_path),
    }
    (root / "E1-V4-SCALE-INPUTS.json").write_text(json.dumps(inputs, indent=1) + "\n")
    lines = [f"{sha256(trace)}  traces/e1_v4_scale.csv", f"{sha256(cfg_path)}  configs_v4scale/{PASS}/profile.json", f"{sha256(SRC)}  configs_fr/e1_v2_forced_release/profile.json", f"{sha256(SRC_TRACE)}  traces/e1_dry.csv"]
    (root / "INPUTS_v4scale.sha256").write_text("\n".join(lines) + "\n")

    # identity policy: the V1 policy with the trace rule replaced; everything else identical (tests compare)
    p = json.loads(POLICY_V1.read_text())
    p["version"] = "v4scale-1"
    p["purpose"] = ("V4-specific, fail-closed normalization policy for comparing the e1_v4_scale launch record / profile config with the accepted e1_nsys_val run. Identical to e1_identity_policy_v1.json (same launch rules, same profile rules, "
                    "same protected patterns) except for the trace: the rule for workload.frontend.path is `derived_trace`, satisfied only if the V4 trace is proved to be the accepted 56-request trace repeated exactly four times "
                    "(checks/e1_launch_identity_v4scale.py), its complete hash equals the V4 input manifest, and both paths are the documented locations. Not a path exemption and not a hash allow-list. Written before any V4 launch.")
    p["applies_to"] = {"run_name": f"dsv41_flash_1017_gb300_tp4_{PASS}", "reference_name": "dsv41_flash_1017_gb300_tp4_e1_nsys_val"}
    p["pinned"]["v4_trace_sha256"] = sha256(trace)
    p["pinned"]["inputs_manifest_sha256"] = sha256(root / "INPUTS_v4scale.sha256")
    p["pinned"]["inputs_json_sha256"] = sha256(root / "E1-V4-SCALE-INPUTS.json")
    p["pinned"]["pins_note"] = "trace_sha256: accepted e1_dry.csv. v4_trace_sha256: traces/e1_v4_scale.csv. inputs_manifest_sha256 / inputs_json_sha256: INPUTS_v4scale.sha256 and E1-V4-SCALE-INPUTS.json as generated by make_e1_inputs_v4scale.py (the same values are listed in FREEZE_v4scale.sha256)."
    p["trace_derivation"] = {"source_path": "traces/e1_dry.csv", "derived_path": "traces/e1_v4_scale.csv", "manifest_file": "INPUTS_v4scale.sha256", "inputs_file": "E1-V4-SCALE-INPUTS.json", "manifest_entry": "traces/e1_v4_scale.csv",
                             "header": header, "source_rows": len(rows), "copies": COPIES, "rows": len(out_rows), "wave_token": "_w0_r", "wave_token_format": "_w{index}_r", "copy_offset_ms": COPY_OFFSET_MS, "input_output_len": ["32768", "1024"],
                             "waves": [[name, w["arrival_ms"], w["requests"]] for name, w in sorted(waves.items(), key=lambda kv: kv[1]["arrival_ms"])]}
    n = 0
    for r in p["profile_rules"]:
        if r["field"] == "workload.frontend.path":
            r.clear()
            r.update({"field": "workload.frontend.path", "category": "worktree_source_path", "kind": "derived_trace",
                      "doc": "trace file: the reference is the accepted e1_dry.csv; the V4 trace must be proved to be that trace repeated exactly four times (rows, order, fields, wave index, +80 s per copy), with the V4 manifest's hash, at the documented path"})
            n += 1
    assert n == 1
    policy_path.write_text(json.dumps(p, indent=1) + "\n")
    return f"traces/e1_v4_scale.csv ({sha256(trace)[:16]}, {len(out_rows)} requests, {len(waves)} waves), configs_v4scale/{PASS}/profile.json ({sha256(cfg_path)[:16]}), E1-V4-SCALE-INPUTS.json ({sha256(root / 'E1-V4-SCALE-INPUTS.json')[:16]}), INPUTS_v4scale.sha256, checks/e1_identity_policy_v4scale.json; expected {iterations} iterations, {rows_expected} kernel rows"


def main(argv: list[str]) -> int:
    if argv == ["--check"]:
        with tempfile.TemporaryDirectory(prefix="v4scale-inputs-check-") as tmp:
            generate(Path(tmp))
            differing = [f for f in GENERATED if not (E1 / f).is_file() or (E1 / f).read_bytes() != (Path(tmp) / f).read_bytes()]
        print(f"check: {len(GENERATED) - len(differing)} of {len(GENERATED)} generated files on disk equal a fresh generation" + (f"; differing or missing: {differing}" if differing else ""))
        return 1 if differing else 0
    if argv:
        print(__doc__)
        return 2
    print("wrote " + generate(E1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
