#!/usr/bin/env python3
"""Independent GPU-state sampler for V3: checks/e1_gpu_watch.py with one more way to end.

    python3 -I checks/e1_gpu_watch_v3.py --out <run>/gpu_observations.jsonl [--interval 2] [--max-s 3300]
        [--stop-after-released 30] [--stop-file FILE] [--after-stop-samples 10]

The observation, its time stamp (taken before the queries), the record format and the `source` label are those of
checks/e1_gpu_watch.py, whose functions are imported unchanged. That sampler ends only after it has seen the GPUs held
and then released; a launch that ends before the GPUs were ever held (a startup failure, an early abort, a CPU-only
rehearsal) would leave it running until --max-s. This one also ends when --stop-file exists, after
--after-stop-samples further samples. It never signals anything. Stdlib only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _base():
    spec = importlib.util.spec_from_file_location("e1_gpu_watch", HERE / "e1_gpu_watch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None, *, census=None, sleep=time.sleep) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--max-s", type=float, default=3600.0)
    ap.add_argument("--max-used-mib", type=int, default=64)
    ap.add_argument("--stop-after-released", type=int, default=0, help="stop after N consecutive released samples that follow a held one (0 = never)")
    ap.add_argument("--stop-file", type=Path)
    ap.add_argument("--after-stop-samples", type=int, default=10)
    args = ap.parse_args(argv)
    base = _base()
    census = census or base.load_census().census
    args.out.parent.mkdir(parents=True, exist_ok=True)
    end = time.monotonic() + args.max_s
    seen_held, streak, after_stop = False, 0, 0
    with args.out.open("a") as f:
        while time.monotonic() < end:
            obs = base.observation(census(args.max_used_mib, None), "e1_gpu_watch")
            f.write(json.dumps(obs) + "\n")
            f.flush()
            if obs["available"] and not obs["released"]:
                seen_held, streak = True, 0
            elif obs["released"] and seen_held:
                streak += 1
                if args.stop_after_released and streak >= args.stop_after_released:
                    break
            if args.stop_file is not None and args.stop_file.exists():
                after_stop += 1
                if after_stop > args.after_stop_samples:
                    break
            sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
