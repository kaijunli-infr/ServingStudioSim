"""Phase 2D: one diagnostic NCCL rank group over the small-message sizes (needs GPUs).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
      uv run --no-sync python profiling/env/arm64-gb300/allreduce_diag.py \
        --num-gpus 2 --expect-uuids GPU-.. GPU-.. --group 1 --out-dir DIR

One call spawns one fresh rank group with the profiler's `TorchMpLauncher`
(the launcher of the `all_reduce`/nccl runner) and, for every message size
below 16 MiB, on every rank:

1. checks `dist.all_reduce` (SUM) element by element on deterministic input;
2. warms up exactly as the runner does (`_WARMUP` all-reduces of the timed
   buffer, then `torch.cuda.synchronize()`);
3. records BATCHES steady-state batches of `_REP` all-reduces, each timed the
   way a stored row is: `dist.barrier()`, CUDA event, `_REP` back-to-back
   `dist.all_reduce(tensor)`, CUDA event, synchronize; time = elapsed / `_REP`.

Added observations, none of which changes what the CUDA events bracket:
`time.perf_counter_ns()` after every call returns (one clock read per call,
well under 0.5 % of a 40 us collective), from which the report derives the
host's enqueue pace and its longest gap, and the host wall time from the first
event to the end of the final synchronize. Communicator creation happens in
the first SUM check and is reported apart; teardown is outside every interval.
Nothing is written to a profile database. Every rank writes
`group<g>-rank<r>.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from profiling.runners.comm._batch import _REP, _WARMUP  # noqa: E402
from profiling.runners.comm._launcher import TorchMpLauncher  # noqa: E402

SIZES = [4096 << k for k in range(12)]  # 4 KiB .. 8 MiB: the rows below 16 MiB
BATCHES = 3
PATTERN_MOD = 7


def _diag_per_rank(*, rank: int, world_size: int, out_dir: str, group: int, sizes: list[int]):
    import torch
    import torch.distributed as dist

    device = torch.cuda.current_device()
    report: dict = {
        "group": group,
        "rank": rank,
        "world_size": world_size,
        "uuid": f"GPU-{torch.cuda.get_device_properties(device).uuid}",
        "backend": dist.get_backend(),
        "warmup": _WARMUP,
        "rep": _REP,
        "sizes": [],
    }
    for index, size in enumerate(sizes):
        n = size // 2
        base = torch.arange(n, dtype=torch.int32, device="cuda") % PATTERN_MOD
        check = (base + rank).to(torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(check)  # default op: SUM
        torch.cuda.synchronize()
        if index == 0:
            report["first_collective_s"] = time.perf_counter() - t0  # communicator creation
        expected = (base * world_size + world_size * (world_size - 1) // 2).to(torch.bfloat16)
        mismatches = int((check != expected).sum().item())
        del base, check, expected

        # From here on: the runner's body (profiling/runners/comm/nccl.py), repeated.
        tensor = torch.randn(n, dtype=torch.bfloat16, device="cuda")
        for _ in range(_WARMUP):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        batches = []
        for _ in range(BATCHES):
            stamps = [0] * _REP
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            dist.barrier()
            h0 = time.perf_counter_ns()
            start.record()
            for i in range(_REP):
                dist.all_reduce(tensor)
                stamps[i] = time.perf_counter_ns()
            end.record()
            h1 = time.perf_counter_ns()
            torch.cuda.synchronize()
            h2 = time.perf_counter_ns()
            batches.append(
                {
                    "time_ms": start.elapsed_time(end) / _REP,  # what a stored row holds
                    "host_enqueue_ms": (h1 - h0) / 1e6 / _REP,
                    "host_total_ms": (h2 - h0) / 1e6 / _REP,
                    "call_gaps_ns": [stamps[0] - h0] + [b - a for a, b in zip(stamps, stamps[1:])],
                }
            )
        del tensor
        report["sizes"].append(
            {"message_size_bytes": size, "sum_mismatches": mismatches, "batches": batches}
        )
    Path(out_dir, f"group{group}-rank{rank}.json").write_text(json.dumps(report) + "\n")
    return {"rank": rank, "ok": True}


def _visible_uuids() -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    by_index = dict(line.split(", ") for line in out.strip().splitlines())
    return [by_index[t.strip()] for t in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--expect-uuids", nargs="+", required=True)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problems: list[str] = []
    visible = _visible_uuids()
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        problems.append("CUDA_DEVICE_ORDER is not PCI_BUS_ID")
    if visible != args.expect_uuids or len(visible) != args.num_gpus:
        problems.append(f"visible GPUs {visible} are not the expected {args.expect_uuids}")
    wall = None
    if not problems:
        start = time.time()
        try:
            TorchMpLauncher(args.num_gpus, backend="nccl").run(
                _diag_per_rank, out_dir=str(out_dir), group=args.group, sizes=SIZES
            )
        except Exception as exc:  # noqa: BLE001 - any launcher failure fails the group
            problems.append(f"launcher failed: {exc}")
        wall = time.time() - start
        for rank in range(args.num_gpus):
            path = out_dir / f"group{args.group}-rank{rank}.json"
            if not path.exists():
                problems.append(f"rank {rank} wrote no report")
                continue
            r = json.loads(path.read_text())
            if r["uuid"] != args.expect_uuids[rank]:
                problems.append(f"rank {rank} holds {r['uuid']}")
            if r["backend"] != "nccl":
                problems.append(f"rank {rank} backend is {r['backend']}")
            if len(r["sizes"]) != len(SIZES):
                problems.append(f"rank {rank} has {len(r['sizes'])} sizes")
            bad = [s["message_size_bytes"] for s in r["sizes"] if s["sum_mismatches"]]
            if bad:
                problems.append(f"rank {rank}: SUM mismatches at {bad}")
            if any(len(s["batches"]) != BATCHES for s in r["sizes"]):
                problems.append(f"rank {rank}: incomplete batches")
    summary = {"group": args.group, "launcher_wall_s": wall, "problems": problems}
    (out_dir / f"group{args.group}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"DIAG GROUP {args.group} pass={not problems} wall={wall} problems={problems}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
