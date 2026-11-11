"""Stage validated graph-replay all-reduce rows as backend `nccl_graph` (CPU-only).

    allreduce_graph_import.py --db STAGING_DB --small phase2f-report.json \
        --large large-report.json --manifest manifest.json

Reads the stability reports of `allreduce_graph_report.py` and inserts one
`all_reduce` row per (num_gpus, message size) under backend `nccl_graph` for
`NVIDIA GB300`, through the profiler's own `Table.insert`. The values are the
`nccl_torch_dist.cuda_graph_replay.ops10` estimates: `--small` supplies the
sizes below 16 MiB, `--large` those from 16 MiB up.

Refuses to write unless: the worktree's profiling code is clean, both TP sizes
have all 19 sizes, every row is complete, stable (4 of 5 groups within 10 %)
and free of SUM mismatches, and the database holds no `nccl_graph` row for
this GPU yet. The eager `nccl` rows are a different key and are verified
unchanged around the insert. The run record carries the commit of this import
(at which `nccl_graph` is a registered backend); the manifest keeps, per row,
the method identifier, the commit and run that measured it, the GPUs' UUIDs,
torch and NCCL versions, and every group estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GPU = "NVIDIA GB300"
BACKEND = "nccl_graph"
SIZES = [4096 << k for k in range(19)]  # 4 KiB .. 1 GiB
BOUNDARY = 16 << 20


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(conn: sqlite3.Connection, backend: str) -> list[tuple]:
    return conn.execute(
        "select num_gpus, message_size_bytes, dtype, fabric, time_ms, algbw_gbps, busbw_gbps,"
        " run_key, profiler_run_at from all_reduce where gpu_name = ? and backend = ?"
        " order by 1, 2",
        (GPU, backend),
    ).fetchall()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--small", required=True)
    parser.add_argument("--large", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    from profiling.db.args import DType
    from profiling.db.registry import find_kernel_profiler_spec
    from profiling.db.table import ProfileRow, Table, _current_git_hash
    from profiling.kernels.all_reduce import KIND, AllReduceArgs
    from profiling.runners.comm.nccl_graph import MEASUREMENT_METHOD, comm_metrics

    stamp = _current_git_hash()
    if stamp.endswith("-dirty") or stamp == "unknown":
        print(f"REFUSED: profiling code stamp is {stamp}")
        return 2

    chosen: dict[tuple[int, int], dict] = {}
    provenance: dict[tuple[str, int], dict] = {}
    for label, path, keep in (
        ("small", args.small, lambda size: size < BOUNDARY),
        ("large", args.large, lambda size: size >= BOUNDARY),
    ):
        for row in json.loads(Path(path).read_text())["rows"]:
            if row.get("provenance"):
                provenance[(label, row["num_gpus"])] = row["provenance"]
            if row["method"] == MEASUREMENT_METHOD and keep(row["message_size_bytes"]):
                chosen[(row["num_gpus"], row["message_size_bytes"])] = row | {"report": label}
    problems = []
    for num_gpus in (2, 4):
        missing = [s for s in SIZES if (num_gpus, s) not in chosen]
        if missing:
            problems.append(f"TP{num_gpus}: no measurement for {missing}")
    for (num_gpus, size), row in chosen.items():
        if not (row["complete"] and row["stable"] and row["sum_mismatches"] == 0):
            problems.append(f"TP{num_gpus} {size}: not complete, stable and correct")
        if not 0 < row["estimate_us"] < 5e6:
            problems.append(f"TP{num_gpus} {size}: implausible time {row['estimate_us']} us")
    if len(chosen) != 2 * len(SIZES):
        problems.append(f"{len(chosen)} rows selected, expected {2 * len(SIZES)}")

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    eager_before = _rows(conn, "nccl")
    if _rows(conn, BACKEND):
        problems.append(f"{args.db} already holds {BACKEND} rows for {GPU}")
    conn.close()
    if problems:
        print("REFUSED:", *problems, sep="\n  ")
        return 2

    sha_before = _sha256(args.db)
    profile_rows = []
    for (num_gpus, size), row in sorted(chosen.items()):
        spec = {"num_gpus": num_gpus, "message_size_bytes": size, "dtype": "bf16"}
        prov = provenance[(row["report"], num_gpus)]
        profile_rows.append(
            ProfileRow(
                args=AllReduceArgs(
                    num_gpus=num_gpus, message_size_bytes=size, dtype=DType.BF16, fabric="nvlink"
                ),
                metrics=comm_metrics(row["estimate_us"] / 1e3, spec, num_gpus),
                gpu_name=GPU,
                backend=BACKEND,
                profiler_git_hash=stamp,
                backend_version="nccl " + ".".join(str(v) for v in prov["nccl"]),
            )
        )
    Table(find_kernel_profiler_spec(KIND, BACKEND), args.db).insert(profile_rows)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    eager_after = _rows(conn, "nccl")
    staged = _rows(conn, BACKEND)
    conn.close()
    if eager_after != eager_before:
        print("ERROR: eager nccl rows changed during the import")
        return 3
    if len(staged) != len(profile_rows):
        print(f"ERROR: {len(staged)} {BACKEND} rows after inserting {len(profile_rows)}")
        return 3

    manifest = {
        "gpu_name": GPU,
        "kind": KIND,
        "backend": BACKEND,
        "measurement_method": MEASUREMENT_METHOD,
        "import_commit": stamp,
        "importer": "profiling/env/arm64-gb300/allreduce_graph_import.py",
        "database": {"path": args.db, "sha256_before": sha_before, "sha256_after": _sha256(args.db)},
        "eager_nccl_rows_unchanged": len(eager_before),
        "reports": {"small": args.small, "large": args.large},
        "rows": [
            {
                "num_gpus": num_gpus,
                "message_size_bytes": size,
                "dtype": "bf16",
                "fabric": "nvlink",
                "time_ms": row["estimate_us"] / 1e3,
                "group_estimates_ms": [e / 1e3 for e in row["group_estimates_us"]],
                "batch_ms": {g: [t / 1e3 for t in v] for g, v in row["batch_us"].items()},
                "groups_within_10pct": row["groups_within_10pct"],
                "cv_pct": row["cv_pct"],
                "sum_mismatches": row["sum_mismatches"],
                "eager_nccl_time_ms": row["stored_eager_us"] / 1e3,
                "measured": provenance[(row["report"], num_gpus)] | {"report": row["report"]},
            }
            for (num_gpus, size), row in sorted(chosen.items())
        ],
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=1) + "\n")
    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    print(f"staged {len(staged)} {BACKEND} rows at {head}; eager rows unchanged: {len(eager_after)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
