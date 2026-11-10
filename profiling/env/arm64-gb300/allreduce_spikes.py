"""List stored all-reduce rows that fresh repeats do not reproduce (CPU-only).

    allreduce_spikes.py --db DB --num-gpus 2 --repeats r1/results.json r2/... r3/... \
        --out force-specs.json [--threshold 0.25]

A stored row is listed when its time differs by more than `--threshold` from
the median of the `--fresh` repeats of the same spec. The list is the input of
`run_allreduce_fill.sh` with FORCE_SPECS, which re-measures exactly those rows
with the profiler's `--force`; the report keeps the replaced values.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--repeats", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    stored = dict(
        con.execute(
            "select message_size_bytes, time_ms from all_reduce where gpu_name = 'NVIDIA GB300' "
            "and backend = 'nccl' and fabric = 'nvlink' and num_gpus = ?",
            (args.num_gpus,),
        )
    )
    fresh: dict[int, list[float]] = {}
    spec_by_size: dict[int, dict] = {}
    for path in args.repeats:
        results = json.loads(Path(path).read_text())
        for spec, result in zip(results["specs"], results["results"], strict=True):
            if result.get("status") == "ok":
                size = int(spec["message_size_bytes"])
                fresh.setdefault(size, []).append(result["metrics"]["time_ms"])
                spec_by_size[size] = spec
    rows, force = [], []
    for size in sorted(stored):
        median = statistics.median(fresh[size])
        ratio = stored[size] / median
        spike = abs(ratio - 1) > args.threshold
        rows.append(
            {
                "message_size_bytes": size,
                "stored_ms": stored[size],
                "fresh_ms": fresh[size],
                "stored_over_fresh_median": ratio,
                "spike": spike,
            }
        )
        if spike:
            force.append(spec_by_size[size])
    report = {"num_gpus": args.num_gpus, "threshold": args.threshold, "rows": rows, "specs": force}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"spikes": [s["message_size_bytes"] for s in force], "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
