#!/usr/bin/env python3
"""Phase 1.5 (Stage 1.5) launcher configs: matched-wire-mode dry pair, off/off control, short nsys validation.

    python3 make_e1_inputs_v2.py [--run-root DIR] [--devices 0,1,2,3] [--out DIR]

Nothing is launched. Reuses make_e1_inputs.py (frozen Stage 1 generator) for the recipe (server block, environment,
nsys block) and the Stage 1 dry trace `traces/e1_dry.csv` (hash unchanged); writes `configs_v2/<pass>/profile.json`,
`INPUTS_v2.sha256` and `E1-STAGE15-INPUTS.json`. The original `configs/`, `traces/`, `INPUTS.sha256` and `E1-INPUTS.json`
are not touched.

Passes (all cells b=8 and b=48 at KV 32k, 56 requests, 1024 output tokens, trace-timed; GPU runs need a separate go):
  e1_dry_off_v2    workload_metrics, openai, hook off, stream=false  (matched wire mode; needs session_runner --stream)
  e1_dry_on_v2     workload_metrics, openai, hook on (capture_routed_experts), stream=false (explicit; also implied by the hook)
  e1_dry_off2_v2   identical to e1_dry_off_v2: the off/off control (run-to-run variability incl. autotune tactics)
  e1_nsys_val      nsys (Phase 0 settings, 10 s delayed start, no bounded stop), openai, hook on: the short Nsight
                   validation (NVTX ownership, per-rank alignment, collective end spread, routes join, export)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_e1_inputs as v1  # noqa: E402  (frozen Stage 1 generator: recipe, schedule, trace writer)

PASSES_V2 = {
    "e1_dry_off_v2":  dict(kind="workload_metrics", trace="e1_dry.csv", backend="openai", routes=False, stream=False),
    "e1_dry_on_v2":   dict(kind="workload_metrics", trace="e1_dry.csv", backend="openai", routes=True, stream=False),
    "e1_dry_off2_v2": dict(kind="workload_metrics", trace="e1_dry.csv", backend="openai", routes=False, stream=False),
    "e1_nsys_val":    dict(kind="nsys", trace="e1_dry.csv", backend="openai", routes=True, stream=False, nsys=v1.nsys()),
}
# the order the protocol prescribes (control bracketing the hook-on arm, nsys validation last)
RUN_ORDER = ["e1_dry_off_v2", "e1_dry_on_v2", "e1_dry_off2_v2", "e1_nsys_val"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=str(v1.REPO / "logs" / "dsv41_flash_gb300_tp4_e1_stage15"))
    ap.add_argument("--devices", default="0,1,2,3")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "configs_v2").mkdir(parents=True, exist_ok=True)
    trace = out / "traces" / "e1_dry.csv"
    assert trace.is_file(), f"Stage 1 dry trace missing: {trace}"
    assert v1.sha256(trace) == "c6d8e13b7229ffd40cfede127bd57ad345aa17213b7590d39f9b2ef697f60158", "e1_dry.csv hash changed"
    dry_rows, dry_plan = v1.schedule(v1.DRY_CELLS, {b: 1 for b in v1.B})
    assert len(dry_rows) == 56

    hashes = {"traces/e1_dry.csv": v1.sha256(trace)}
    for name, ps in PASSES_V2.items():
        trace_in_repo = v1.E1 / "traces" / ps["trace"]
        run_dir = Path(a.run_root) / name
        wl = v1.workload(trace_in_repo, ps["backend"], ps["routes"])
        wl["stream"] = ps["stream"]         # explicit in every config so the identity check compares like with like;
        #                                      False needs patch/apply_stream_optin.py (launcher) + req-frontend --stream
        cfg = {
            "schema_version": 1,
            "name": f"dsv41_flash_1017_gb300_tp4_{name}",
            "log_dir": str(run_dir / "profile"),
            "gpu": "NVIDIA GB300",
            "cuda_visible_devices": a.devices,
            "profile_kind": ps["kind"],
            "engine": "vllm",
            "fork_python": str(v1.FORK_ENV / "venv" / "bin" / "python"),
            "python_runtime": {"packages": [{"name": "nvtx", "version": "0.2.15", "index_url": "https://pypi.org/simple"}],
                               "environment": v1.ENV, "lock_timeout_seconds": 60, "install_timeout_seconds": 600},
            "server": v1.server(),
            "idle": {"enabled": True, "timeout": 600.0, "poll_interval": 0.5},
            "workload": wl,
        }
        if ps["kind"] == "nsys":
            cfg["nsys"] = ps["nsys"]
        d = out / "configs_v2" / name
        d.mkdir(exist_ok=True)
        (d / "profile.json").write_text(json.dumps(cfg, indent=1) + "\n")
        hashes[f"configs_v2/{name}/profile.json"] = v1.sha256(d / "profile.json")

    total_dry = dry_plan[-1]["arrival_s"] + dry_plan[-1]["est_duration_s"] * v1.MARGIN + v1.GAP_S
    summary = {
        "design": {"B": v1.B, "KV": v1.KVS, "OUT_LEN": v1.OUT_LEN, "WAVES": v1.WAVES, "CHUNK_S": v1.CHUNK_S, "WALL_MS": v1.WALL_MS,
                   "MARGIN": v1.MARGIN, "GAP_S": v1.GAP_S, "MIN_PURE_DECODE": v1.MIN_PURE_DECODE},
        "main": {"plan": []},
        "dry": {"requests": len(dry_rows), "trace_span_s": round(total_dry, 1), "plan": dry_plan},
        "passes": {k: {kk: vv for kk, vv in v.items() if kk != "nsys"} for k, v in PASSES_V2.items()},
        "run_order": RUN_ORDER,
        "sha256": hashes,
    }
    (out / "E1-STAGE15-INPUTS.json").write_text(json.dumps(summary, indent=1) + "\n")
    with (out / "INPUTS_v2.sha256").open("w") as fh:
        for k, v in sorted(hashes.items()):
            fh.write(f"{v}  {k}\n")
    print(json.dumps({"dry": {k: v for k, v in summary["dry"].items() if k != "plan"}, "passes": list(PASSES_V2), "sha256": hashes}, indent=1))


if __name__ == "__main__":
    main()
