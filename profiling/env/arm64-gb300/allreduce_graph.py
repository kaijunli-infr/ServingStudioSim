"""Phase 2F: `torch.distributed.all_reduce` timed by CUDA-graph replay (needs GPUs).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
      uv run --no-sync python profiling/env/arm64-gb300/allreduce_graph.py \
        --num-gpus 2 --expect-uuids GPU-.. GPU-.. --group 1 --out-dir DIR

One fresh rank group through the profiler's `TorchMpLauncher`; the collective
is the `all_reduce`/nccl runner's own call, `dist.all_reduce(tensor)` (SUM) on
a bf16 buffer of `message_size_bytes`, one rank per GPU. Only the timing
boundary differs from that runner: the call is captured in a CUDA graph and
the graph is replayed, as `_time_cuda_graph` in
`profiling/runners/comm/flashinfer_trtllm.py` does for the FlashInfer
collectives ("Match vLLM's serving path"). The conventions are theirs:

- warm-up launches before capture, warm-up replays after it;
- `dist.barrier()`, then host wall time over the replays up to the end of
  `torch.cuda.synchronize()`, divided by the number of collectives;
- the group's figure is the slowest rank's (`ReduceOp.MAX`);
- `OPERATIONS_PER_GRAPH = 10` back-to-back collectives per graph, the value
  `flashinfer_mnnvl.py` uses to amortise the host's replay launch.

Method identifiers written with every observation:

- `nccl_torch_dist.cuda_graph_replay.ops10`: 10 collectives per graph, 100
  replays per batch (1000 collectives). The production candidate.
- `nccl_torch_dist.cuda_graph_replay.ops1`: 1 collective per graph, 100
  replays per batch. Diagnostic: one host launch per collective.
- `nccl_torch_dist.eager_events`: the stored-row method (CUDA events around
  100 eager calls). Boundary sizes only, for the eager-against-graph comparison.

Each graph is checked before it is timed: the buffer is refilled with a
deterministic pattern, the graph replayed once, and the result compared
element by element with the closed-form sum (times world_size^(ops-1) for the
ten-collective graph; all values stay exact in bf16). Three batches per method
and size. CUDA events around the same replay loop are kept as a second clock.
Nothing is written to a profile database. Every rank writes
`group<g>-rank<r>.json`. `--set large` runs the sizes from 16 MiB to 1 GiB with
the same sequence (the eager method again only at 16 and 128 MiB).
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

CANDIDATE_SIZES = [4096 << k for k in range(12)]  # 4 KiB .. 8 MiB: the rows below 16 MiB
LARGE_SIZES = [(16 << 20) << k for k in range(7)]  # 16 MiB .. 1 GiB: the rows from 16 MiB up
BOUNDARY_SIZES = [8 << 20, 16 << 20, 128 << 20]  # eager and graph side by side
BATCHES = 3
REPLAYS = _REP  # 100 graph replays per batch
OPERATIONS_PER_GRAPH = 10  # as profiling/runners/comm/flashinfer_mnnvl.py
PATTERN_MOD = 7
METHOD_GRAPH = "nccl_torch_dist.cuda_graph_replay.ops{ops}"
METHOD_EAGER = "nccl_torch_dist.eager_events"


def _graph_per_rank(
    *, rank: int, world_size: int, out_dir: str, group: int, sizes: list[int], provenance: dict
):
    import torch
    import torch.distributed as dist

    device = torch.cuda.current_device()
    report: dict = {
        "group": group,
        "rank": rank,
        "world_size": world_size,
        "uuid": f"GPU-{torch.cuda.get_device_properties(device).uuid}",
        "backend": dist.get_backend(),
        "torch": torch.__version__,
        "nccl": list(torch.cuda.nccl.version()),
        "provenance": provenance,
        "sizes": [],
    }

    def slowest_rank(local_ms: float) -> float:
        value = torch.tensor(local_ms, dtype=torch.float64, device="cuda")
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return float(value.item())

    for size in sizes:
        n = size // 2
        base = torch.arange(n, dtype=torch.int32, device="cuda") % PATTERN_MOD
        pattern = (base + rank).to(torch.bfloat16)
        expected = (base * world_size + world_size * (world_size - 1) // 2).to(torch.bfloat16)
        del base
        entry: dict = {"message_size_bytes": size, "methods": {}}

        # Eager SUM check, then the runner's warm-up on the timed buffer.
        tensor = pattern.clone()
        dist.all_reduce(tensor)  # default op: SUM
        torch.cuda.synchronize()
        entry["eager_sum_mismatches"] = int((tensor != expected).sum().item())
        tensor.copy_(torch.randn(n, dtype=torch.bfloat16, device="cuda"))
        for _ in range(_WARMUP):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()

        if size in BOUNDARY_SIZES:
            batches = []
            for _ in range(BATCHES):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                dist.barrier()
                start.record()
                for _ in range(_REP):
                    dist.all_reduce(tensor)
                end.record()
                torch.cuda.synchronize()
                batches.append({"time_ms": start.elapsed_time(end) / _REP})
            entry["methods"][METHOD_EAGER] = {"collectives_per_batch": _REP, "batches": batches}

        for ops in (OPERATIONS_PER_GRAPH, 1):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(ops):
                    dist.all_reduce(tensor)
            torch.cuda.synchronize()
            # Correctness of this graph: refill, replay once, compare.
            tensor.copy_(pattern)
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            mismatches = int((tensor != expected * float(world_size ** (ops - 1))).sum().item())
            tensor.copy_(torch.randn(n, dtype=torch.bfloat16, device="cuda"))
            for _ in range(max(1, _WARMUP // ops)):
                graph.replay()
            torch.cuda.synchronize()
            batches = []
            for _ in range(BATCHES):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                dist.barrier()
                t0 = time.perf_counter()
                start.record()
                for _ in range(REPLAYS):
                    graph.replay()
                end.record()
                torch.cuda.synchronize()
                local_ms = (time.perf_counter() - t0) / (REPLAYS * ops) * 1000.0
                batches.append(
                    {
                        "local_time_ms": local_ms,
                        "local_event_time_ms": start.elapsed_time(end) / (REPLAYS * ops),
                        "time_ms": slowest_rank(local_ms),  # the convention's figure
                    }
                )
            entry["methods"][METHOD_GRAPH.format(ops=ops)] = {
                "operations_per_graph": ops,
                "replays_per_batch": REPLAYS,
                "collectives_per_batch": REPLAYS * ops,
                "replay_sum_mismatches": mismatches,
                "batches": batches,
            }
            del graph
        del tensor, pattern, expected
        report["sizes"].append(entry)
    Path(out_dir, f"group{group}-rank{rank}.json").write_text(json.dumps(report) + "\n")
    return {"rank": rank}


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
    parser.add_argument(
        "--set",
        choices=("small", "large"),
        default="small",
        help="small: below 16 MiB plus the boundary sizes (Phase 2F); large: 16 MiB to 1 GiB",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.set == "small":
        sizes = sorted(set(CANDIDATE_SIZES) | set(BOUNDARY_SIZES))
    else:
        sizes = list(LARGE_SIZES)
    git = ["git", "-C", str(REPO_ROOT)]
    provenance = {
        "source_commit": subprocess.run(
            [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "source_dirty": bool(
            subprocess.run(
                [*git, "status", "--porcelain"], capture_output=True, text=True, check=False
            ).stdout.strip()
        ),
        "harness": "profiling/env/arm64-gb300/allreduce_graph.py",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    problems: list[str] = []
    visible = _visible_uuids()
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        problems.append("CUDA_DEVICE_ORDER is not PCI_BUS_ID")
    if visible != args.expect_uuids or len(visible) != args.num_gpus:
        problems.append(f"visible GPUs {visible} are not the expected {args.expect_uuids}")
    if not problems:
        try:
            TorchMpLauncher(args.num_gpus, backend="nccl").run(
                _graph_per_rank,
                out_dir=str(out_dir),
                group=args.group,
                sizes=sizes,
                provenance=provenance,
            )
        except Exception as exc:  # noqa: BLE001 - any launcher failure fails the group
            problems.append(f"launcher failed: {exc}")
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
            if len(r["sizes"]) != len(sizes):
                problems.append(f"rank {rank} has {len(r['sizes'])} sizes")
            for s in r["sizes"]:
                where = f"rank {rank} at {s['message_size_bytes']} bytes"
                if s["eager_sum_mismatches"]:
                    problems.append(f"{where}: eager SUM mismatches")
                for name, method in s["methods"].items():
                    if method.get("replay_sum_mismatches"):
                        problems.append(f"{where}: replay SUM mismatches ({name})")
                    if len(method["batches"]) != BATCHES:
                        problems.append(f"{where}: incomplete batches ({name})")
    print(f"GRAPH GROUP {args.group} pass={not problems} problems={problems}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
