#!/usr/bin/env python3
"""Derive the full-E1 v3 inputs (CPU only): the accepted Stage 1.5 recipe (configs_v2/e1_nsys_val: openai backend, stream false,
routing hook on, Phase 0 nsys block) with the profiler armed before the first request (no capture_start_delay_seconds), driving
the frozen full-E1 trace traces/e1_main.csv (sha 2e147402...). Writes configs_v3/e1_main_v3/profile.json, INPUTS_v3.sha256 and
E1-MAIN-V3-INPUTS.json. Nothing else is changed: server block, environment, workload options are copied verbatim.

    python3 make_e1_inputs_v3.py --e1-dir <phase1c/e1> --run-root <log root> [--expect-nsys-val-sha <sha>] [--expect-main-trace-sha <sha>]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os

PHASE0_QUADRANT_NSYS = {"executable": "/usr/local/bin/nsys", "capture_mode": "cuda_profiler_api", "cuda_graph_trace": "node", "cuda_event_trace": False}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-dir", required=True); ap.add_argument("--run-root", required=True)
    ap.add_argument("--expect-nsys-val-sha", default="df949648"); ap.add_argument("--expect-main-trace-sha", default="2e147402b5a24f9e2078bf12e3623691c2fd6d5c8378bab09d7e86ee23bcc2ef")
    a = ap.parse_args()
    E1 = os.path.abspath(a.e1_dir)
    src = os.path.join(E1, "configs_v2", "e1_nsys_val", "profile.json"); trace = os.path.join(E1, "traces", "e1_main.csv")
    assert sha(src).startswith(a.expect_nsys_val_sha), ("nsys_val config sha", sha(src))
    assert sha(trace) == a.expect_main_trace_sha, ("e1_main.csv sha", sha(trace))
    cfg = json.load(open(src)); c = copy.deepcopy(cfg)
    c["name"] = "dsv41_flash_1017_gb300_tp4_e1_main_v3"
    c["log_dir"] = os.path.join(a.run_root, "e1_main_v3", "profile")
    c["workload"]["frontend"]["path"] = trace
    c["nsys"].pop("capture_start_delay_seconds", None)
    assert c["nsys"] == PHASE0_QUADRANT_NSYS, c["nsys"]
    for k in ("server", "python_runtime", "engine", "fork_python", "profile_kind", "gpu", "cuda_visible_devices", "idle"):
        assert c.get(k) == cfg.get(k), k
    for k, v in cfg["workload"].items():
        if k != "frontend":
            assert c["workload"][k] == v, k
    out_dir = os.path.join(E1, "configs_v3", "e1_main_v3"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "profile.json"); json.dump(c, open(out, "w"), indent=2); open(out, "a").write("\n")
    rows = ["traces/e1_main.csv", "configs_v3/e1_main_v3/profile.json", "configs_v2/e1_nsys_val/profile.json", "configs_v2/e1_b8_recovery/profile.json", "traces/e1_b8_recovery.csv", "traces/e1_dry.csv"]
    with open(os.path.join(E1, "INPUTS_v3.sha256"), "w") as f:
        for r in rows:
            f.write(f"{sha(os.path.join(E1, r))}  {r}\n")
    plan = json.load(open(os.path.join(E1, "E1-INPUTS.json")))
    info = {"derived_from": {"config": "configs_v2/e1_nsys_val/profile.json", "sha256": sha(src), "accepted_as": "e1_nsys_val (a62e40bd) and, with the b=8 trace, e1_b8_recovery (223aff6d)"},
            "changes": ["name", "log_dir", "workload.frontend.path -> traces/e1_main.csv", "nsys.capture_start_delay_seconds removed (profiler armed before the first request)"],
            "unchanged": ["server", "python_runtime", "workload.* (openai backend, stream false, capture_routed_experts true, warmup, trace-timed)", "nsys block == Phase 0 quadrant_nsys"],
            "trace": {"path": "traces/e1_main.csv", "sha256": sha(trace), "requests": plan["main"]["requests"], "waves": plan["main"]["waves"], "trace_span_s": plan["main"]["trace_span_s"]},
            "design": plan["design"], "plan": plan["main"]["plan"], "pure_decode_iters_min": plan["main"]["pure_decode_iters_min"], "config_sha256": sha(out)}
    json.dump(info, open(os.path.join(E1, "E1-MAIN-V3-INPUTS.json"), "w"), indent=1)
    print("config", sha(out)[:16], "trace", sha(trace)[:16], "INPUTS_v3.sha256 written")


if __name__ == "__main__":
    main()
