"""Phase 2E: what does an all-reduce row time? One diagnostic rank group (needs GPUs).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
      uv run --no-sync python profiling/env/arm64-gb300/allreduce_semantics.py \
        --num-gpus 2 --expect-uuids GPU-.. GPU-.. --group 1 --mode timing --out-dir DIR

One fresh group through the profiler's `TorchMpLauncher`, the same
`dist.all_reduce(tensor)` on the same bf16 buffers as the `all_reduce`/nccl
runner, at 8 KiB, 512 KiB and 8 MiB. `--mode timing` records, on every rank:

A. `runner`: the stored-row method (warm-up, barrier, CUDA events around 100
   back-to-back calls), three batches, plus the host time spent inside each
   call and between calls.
S. `sync`: 100 calls each followed by `torch.cuda.synchronize()`, host clock:
   the latency of one isolated, eagerly launched collective.
B. `kernel`: the GPU-side start and duration of every NCCL device kernel
   during 100 back-to-back calls, from torch.profiler (Kineto/CUPTI, CUDA
   activity only). Kernel time excludes host enqueue; the gaps between kernels
   are the time the GPU sat idle waiting for the host. Runs last, because the
   CUPTI subscription it leaves behind must not touch methods A and S.

`--mode graph` tests whether `dist.all_reduce` can be captured in a CUDA graph
at all on this stack: capture one call, refill the buffer, replay, and check
the sum. Only if every rank reports a correct replay is replay throughput
timed, with the repository's own `_time_cuda_graph` (the helper of the
FlashInfer all-reduce runners). A capture error is recorded, not worked around.

No profile database is written. Every rank writes `<mode>-group<g>-rank<r>.json`.
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

SIZES = [8192, 524288, 8388608]
BATCHES = 3
PATTERN_MOD = 7


def _pattern(torch, n: int, rank: int, world_size: int):
    base = torch.arange(n, dtype=torch.int32, device="cuda") % PATTERN_MOD
    mine = (base + rank).to(torch.bfloat16)
    expected = (base * world_size + world_size * (world_size - 1) // 2).to(torch.bfloat16)
    return mine, expected


def _timing_per_rank(*, rank: int, world_size: int, out_dir: str, group: int, sizes: list[int]):
    import torch
    import torch.distributed as dist
    from torch.profiler import ProfilerActivity, profile

    device = torch.cuda.current_device()
    report: dict = {
        "mode": "timing",
        "group": group,
        "rank": rank,
        "world_size": world_size,
        "uuid": f"GPU-{torch.cuda.get_device_properties(device).uuid}",
        "backend": dist.get_backend(),
        "sizes": [],
    }
    tensors = {}
    for size in sizes:
        n = size // 2
        check, expected = _pattern(torch, n, rank, world_size)
        dist.all_reduce(check)  # default op: SUM
        torch.cuda.synchronize()
        entry = {
            "message_size_bytes": size,
            "sum_mismatches": int((check != expected).sum().item()),
        }
        del check, expected

        # A. The runner's body (profiling/runners/comm/nccl.py), three batches.
        tensor = torch.randn(n, dtype=torch.bfloat16, device="cuda")
        tensors[size] = tensor
        for _ in range(_WARMUP):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        batches = []
        for _ in range(BATCHES):
            before = [0] * _REP
            after = [0] * _REP
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            dist.barrier()
            start.record()
            for i in range(_REP):
                before[i] = time.perf_counter_ns()
                dist.all_reduce(tensor)
                after[i] = time.perf_counter_ns()
            end.record()
            torch.cuda.synchronize()
            batches.append(
                {
                    "time_ms": start.elapsed_time(end) / _REP,
                    "in_call_ns": [a - b for b, a in zip(before, after, strict=True)],
                    "between_calls_ns": [b - a for a, b in zip(after, before[1:], strict=False)],
                }
            )
        entry["runner"] = batches

        # S. One collective at a time, synchronised: isolated eager latency.
        dist.barrier()
        torch.cuda.synchronize()
        sync_ns = [0] * _REP
        for i in range(_REP):
            t0 = time.perf_counter_ns()
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            sync_ns[i] = time.perf_counter_ns() - t0
        entry["sync_ns"] = sync_ns
        report["sizes"].append(entry)

    # B. GPU-side kernel records, after every host-clocked method is done.
    dist.barrier()
    with profile(activities=[ProfilerActivity.CUDA]):  # start CUPTI once, discard
        for _ in range(5):
            dist.all_reduce(tensors[sizes[0]])
        torch.cuda.synchronize()
    for entry in report["sizes"]:
        tensor = tensors[entry["message_size_bytes"]]
        for _ in range(_WARMUP):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter_ns()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(_REP):
                dist.all_reduce(tensor)
            torch.cuda.synchronize()
        entry["kernel_host_wall_ms"] = (time.perf_counter_ns() - t0) / 1e6
        records = []
        for event in prof.events():
            if "cuda" not in str(event.device_type).lower():
                continue
            records.append(
                {
                    "name": event.name[:120],
                    "start": event.time_range.start,  # profiler clock units, see dur_us
                    "end": event.time_range.end,
                    "dur_us": event.time_range.elapsed_us(),
                }
            )
        entry["kernel_records"] = sorted(records, key=lambda r: r["start"])
    Path(out_dir, f"timing-group{group}-rank{rank}.json").write_text(json.dumps(report) + "\n")
    return {"rank": rank}


def _graph_per_rank(*, rank: int, world_size: int, out_dir: str, group: int, sizes: list[int]):
    import torch
    import torch.distributed as dist

    from profiling.runners.comm.flashinfer_trtllm import _time_cuda_graph

    device = torch.cuda.current_device()
    report: dict = {
        "mode": "graph",
        "group": group,
        "rank": rank,
        "world_size": world_size,
        "uuid": f"GPU-{torch.cuda.get_device_properties(device).uuid}",
        "sizes": [],
    }
    usable = True
    for size in sizes:
        n = size // 2
        entry: dict = {"message_size_bytes": size}
        tensor, expected = _pattern(torch, n, rank, world_size)
        for _ in range(_WARMUP):  # eager warm-up: communicator and buffers exist before capture
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        if usable:
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    dist.all_reduce(tensor)
                torch.cuda.synchronize()
                entry["capture"] = "ok"
            except Exception as exc:  # noqa: BLE001 - the capture error is the finding
                entry["capture"] = f"error: {type(exc).__name__}: {str(exc)[:400]}"
                usable = False
        else:
            entry["capture"] = "skipped after an earlier capture error"
        # Every rank reports its capture outcome before anyone replays.
        flag = torch.tensor([1.0 if entry["capture"] == "ok" else 0.0], device="cuda")
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        all_captured = bool(flag.item() == 1.0)
        entry["all_ranks_captured"] = all_captured
        if all_captured:
            mine, _ = _pattern(torch, n, rank, world_size)
            tensor.copy_(mine)
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            entry["replay_mismatches"] = int((tensor != expected).sum().item())
            ok = torch.tensor([1.0 if entry["replay_mismatches"] == 0 else 0.0], device="cuda")
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            entry["all_ranks_replay_correct"] = bool(ok.item() == 1.0)
            if entry["all_ranks_replay_correct"]:
                data = torch.randn(n, dtype=torch.bfloat16, device="cuda")
                entry["graph_replay_time_ms"] = _time_cuda_graph(
                    torch=torch,
                    dist=dist,
                    launch=lambda data=data: dist.all_reduce(data),
                    input_tensor=data,
                    warmup=_WARMUP,
                    rep=_REP,
                    operations_per_graph=10,
                )
        report["sizes"].append(entry)
    Path(out_dir, f"graph-group{group}-rank{rank}.json").write_text(json.dumps(report) + "\n")
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
    parser.add_argument("--mode", choices=("timing", "graph"), required=True)
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
    if not problems:
        fn = _timing_per_rank if args.mode == "timing" else _graph_per_rank
        try:
            TorchMpLauncher(args.num_gpus, backend="nccl").run(
                fn, out_dir=str(out_dir), group=args.group, sizes=SIZES
            )
        except Exception as exc:  # noqa: BLE001 - any launcher failure fails the group
            problems.append(f"launcher failed: {exc}")
        for rank in range(args.num_gpus):
            path = out_dir / f"{args.mode}-group{args.group}-rank{rank}.json"
            if not path.exists():
                problems.append(f"rank {rank} wrote no report")
                continue
            r = json.loads(path.read_text())
            if r["uuid"] != args.expect_uuids[rank]:
                problems.append(f"rank {rank} holds {r['uuid']}")
            if len(r["sizes"]) != len(SIZES):
                problems.append(f"rank {rank} has {len(r['sizes'])} sizes")
            for s in r["sizes"]:
                if s.get("sum_mismatches"):
                    problems.append(f"rank {rank}: SUM mismatches at {s['message_size_bytes']}")
                if args.mode == "graph":
                    print(
                        f"rank {rank} {s['message_size_bytes']}: "
                        + json.dumps({k: v for k, v in s.items() if k != "message_size_bytes"})[
                            :500
                        ]
                    )
    print(f"SEMANTICS {args.mode} GROUP {args.group} pass={not problems} problems={problems}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
