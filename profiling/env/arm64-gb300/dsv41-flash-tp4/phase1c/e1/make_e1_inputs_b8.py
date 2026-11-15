#!/usr/bin/env python3
"""b=8 recovery capture inputs (CPU-only; nothing is launched; GPU execution needs a separate go).

    python3 make_e1_inputs_b8.py [--e1-dir DIR] [--run-root DIR] [--delay SEC] [--expect-dry-sha SHA|none]

Derives, from the accepted Stage 1.5 inputs and without touching them:
  traces/e1_b8_recovery.csv            the header and the first 8 rows of traces/e1_dry.csv, byte for byte (the b=8 cell:
                                       e1_b8_kv32k_w0_r00..r07, input 32768, output 1024, arrival 0.0). The 48 b=48 rows are
                                       NOT in the trace: the recovery cannot execute b=48 (documented deviation from the
                                       two-cell dry workload). Request ordinals 0..7 are unchanged, so req-frontend seeds the
                                       same synthetic prompts (TokenProvider seed = ordinal * 9973 over the same 100 M-token
                                       enwik9 pool; executor/independent.rs).
  configs_v2/e1_b8_recovery/profile.json
                                       configs_v2/e1_nsys_val/profile.json with exactly four changes: name, log_dir,
                                       workload.frontend.path (the recovery trace) and the nsys block's
                                       capture_start_delay_seconds removed (default) or set to --delay. Without the delay the
                                       nsys block equals the Phase 0 quadrant_nsys block and the launcher arms CUPTI
                                       synchronously before the first request is released (alignment/runner.py
                                       measurement_ready), so the whole b=8 wave is inside the capture by construction.
  INPUTS_b8.sha256                     sha256 of the two new files plus the two Stage 1.5 inputs they derive from.
  E1-B8-RECOVERY-INPUTS.json           the Stage 1.5 design block, a 'dry' plan holding only the b=8 cell (8 requests), the
                                       pass table and the hashes (format of E1-STAGE15-INPUTS.json, read by
                                       checks/e1_completeness_v2.py and checks/e1_nsys_validation.py via --inputs).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

DRY_SHA = "c6d8e13b7229ffd40cfede127bd57ad345aa17213b7590d39f9b2ef697f60158"   # traces/e1_dry.csv, INPUTS_v2.sha256
NSYS_VAL_SHA = "df949648153e862746cf5a8446bfb0a1ead0169388f2306fc5d9c5acc4a659ff"  # configs_v2/e1_nsys_val/profile.json
PASS = "e1_b8_recovery"
PHASE0_QUADRANT_NSYS = {"executable": "/usr/local/bin/nsys", "capture_mode": "cuda_profiler_api", "cuda_graph_trace": "node", "cuda_event_trace": False}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def derive_trace(dry: Path, out: Path, b: int = 8) -> list[str]:
    """Header + the rows whose id carries the b=<b> tag, in their original order and bytes. They must be the first rows."""
    lines = dry.read_bytes().split(b"\n")
    header, rows = lines[0], [l for l in lines[1:] if l]
    keep = [r for r in rows if r.split(b",")[0].startswith(f"e1_b{b}_".encode())]
    assert keep and rows[:len(keep)] == keep, "the b=%d rows are not the leading rows of the dry trace (ordinals would change)" % b
    assert all(r.split(b",")[0].startswith(f"e1_b{b}_".encode()) for r in keep)
    out.write_bytes(header + b"\n" + b"\n".join(keep) + b"\n")
    return [r.decode() for r in keep]


def derive_config(nsys_val_cfg: dict, trace: Path, run_root: Path, delay: float | None) -> dict:
    cfg = copy.deepcopy(nsys_val_cfg)
    cfg["name"] = f"dsv41_flash_1017_gb300_tp4_{PASS}"
    cfg["log_dir"] = str(run_root / PASS / "profile")
    cfg["workload"]["frontend"]["path"] = str(trace)
    cfg["nsys"].pop("capture_start_delay_seconds", None)
    if delay is not None:
        cfg["nsys"]["capture_start_delay_seconds"] = float(delay)
    assert "capture_duration_seconds" not in cfg["nsys"], "no bounded stop (Phase 0 lesson)"
    assert {k: v for k, v in cfg["nsys"].items() if k != "capture_start_delay_seconds"} == PHASE0_QUADRANT_NSYS
    assert cfg["workload"]["stream"] is False and cfg["workload"]["capture_routed_experts"] is True
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--run-root", default="/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15")
    ap.add_argument("--delay", type=float, default=None, help="capture_start_delay_seconds (default: none = Phase 0 quadrant_nsys block)")
    ap.add_argument("--expect-dry-sha", default=DRY_SHA, help="expected sha256 of traces/e1_dry.csv ('none' to skip)")
    ap.add_argument("--expect-nsys-val-sha", default=NSYS_VAL_SHA)
    a = ap.parse_args()
    e1, run_root = Path(a.e1_dir), Path(a.run_root)
    dry, nsys_val = e1 / "traces" / "e1_dry.csv", e1 / "configs_v2" / "e1_nsys_val" / "profile.json"
    if a.expect_dry_sha != "none":
        assert sha256(dry) == a.expect_dry_sha, "e1_dry.csv hash changed"
    if a.expect_nsys_val_sha != "none":
        assert sha256(nsys_val) == a.expect_nsys_val_sha, "e1_nsys_val/profile.json hash changed"
    stage15 = json.load(open(e1 / "E1-STAGE15-INPUTS.json")) if (e1 / "E1-STAGE15-INPUTS.json").exists() else None

    trace = e1 / "traces" / f"{PASS}.csv"
    rows = derive_trace(dry, trace)
    cfg = derive_config(json.load(open(nsys_val)), trace, run_root, a.delay)
    d = e1 / "configs_v2" / PASS
    d.mkdir(parents=True, exist_ok=True)
    (d / "profile.json").write_text(json.dumps(cfg, indent=1) + "\n")

    hashes = {f"traces/{PASS}.csv": sha256(trace), f"configs_v2/{PASS}/profile.json": sha256(d / "profile.json"),
              "traces/e1_dry.csv": sha256(dry), "configs_v2/e1_nsys_val/profile.json": sha256(nsys_val)}
    with (e1 / "INPUTS_b8.sha256").open("w") as fh:
        for k, v in sorted(hashes.items()):
            fh.write(f"{v}  {k}\n")
    plan = [p for p in (stage15 or {}).get("dry", {}).get("plan", []) if p["b"] == 8] if stage15 else []
    summary = {
        "design": (stage15 or {}).get("design"),
        "derived_from": {"inputs": "E1-STAGE15-INPUTS.json", "trace": "traces/e1_dry.csv rows 1-8 (byte-identical)", "config": "configs_v2/e1_nsys_val/profile.json"},
        "deviation_from_stage15": ["the 48 b=48 requests are absent from the trace (b=8-only recovery; b=48 is not re-run)",
                                   "nsys.capture_start_delay_seconds " + ("removed (Phase 0 quadrant_nsys block; profiler armed before the first request)" if a.delay is None else f"= {a.delay}")],
        "main": {"plan": []},
        "dry": {"requests": len(rows), "trace_span_s": round(plan[0]["est_duration_s"] * (stage15["design"]["MARGIN"] if stage15 else 1.5), 1) if plan else None, "plan": plan},
        "passes": {PASS: {"kind": "nsys", "trace": f"{PASS}.csv", "backend": "openai", "routes": True, "stream": False,
                          "nsys": cfg["nsys"], "delay_s": a.delay}},
        "run_order": [PASS],
        "sha256": hashes,
    }
    (e1 / "E1-B8-RECOVERY-INPUTS.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"trace_rows": rows, "nsys": cfg["nsys"], "sha256": hashes, "plan": plan}, indent=1))


if __name__ == "__main__":
    main()
