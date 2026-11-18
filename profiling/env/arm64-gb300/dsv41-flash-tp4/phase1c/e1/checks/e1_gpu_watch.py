#!/usr/bin/env python3
"""Independent GPU-state sampler (V1 remediation, item E): timestamped nvidia-smi observations as JSONL.

    python3 -I checks/e1_gpu_watch.py --out <run>/gpu_observations.jsonl [--interval 5] [--max-s 3600]
        [--stop-after-released 3]

Meant to be started by a driver through `e1_pids.py spawn` (registered, identity-verified cleanup) before the stop
request, so the release of the GPUs is bracketed by a last "held" and a first "released" observation taken outside
the lifecycle controller. Uses the census of checks/e1_gpu_release_census.py unchanged. Never signals anything.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


def load_census():
    path = Path(__file__).resolve().parent / "e1_gpu_release_census.py"
    spec = importlib.util.spec_from_file_location("e1_gpu_release_census", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def observation(rec: dict, source: str) -> dict:
    return {
        "at": rec["at"],
        "at_iso": rec["at_iso"],
        "host": rec["host"],
        "source": source,
        "memory_used_mib": [g["memory_used_mib"] for g in rec["gpus"]],
        "compute_pids": [c["pid"] for c in rec["compute_apps"]],
        "released": rec["verdict"]["released"],
        "available": rec["verdict"]["available"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--max-s", type=float, default=3600.0)
    ap.add_argument("--max-used-mib", type=int, default=64)
    ap.add_argument("--stop-after-released", type=int, default=0, help="stop after N consecutive released samples that follow a held one (0 = run to --max-s)")
    args = ap.parse_args(argv)
    census = load_census()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    end = time.monotonic() + args.max_s
    seen_held, streak = False, 0
    with args.out.open("a") as f:
        while time.monotonic() < end:
            obs = observation(census.census(args.max_used_mib, None), "e1_gpu_watch")
            f.write(json.dumps(obs) + "\n")
            f.flush()
            if obs["available"] and not obs["released"]:
                seen_held, streak = True, 0
            elif obs["released"] and seen_held:
                streak += 1
                if args.stop_after_released and streak >= args.stop_after_released:
                    break
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
