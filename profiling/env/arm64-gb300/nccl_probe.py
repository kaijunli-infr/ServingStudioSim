"""NCCL rank-group probe for the GB300 all-reduce fill (needs GPUs).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 NCCL_DEBUG=INFO \
      uv run --no-sync python profiling/env/arm64-gb300/nccl_probe.py \
        --num-gpus 2 --expect-uuids GPU-.. GPU-.. --out-dir DIR

Spawns the rank group with the profiler's own `TorchMpLauncher` (the launcher
the `all_reduce`/nccl runner uses) and, inside the live group, on every rank:

1. records which physical GPU the rank holds (rank r must hold the r-th UUID
   of `--expect-uuids`, the order of `CUDA_VISIBLE_DEVICES`);
2. times the first collective, which is where NCCL creates its communicator
   (`init_process_group` only does the TCP rendezvous), separately from
   everything after it;
3. checks `dist.all_reduce` (SUM) element by element against the closed-form
   sum of the ranks' deterministic bf16 inputs, at each `--sizes` value;
4. runs the runner's own per-rank body, `_all_reduce_per_rank_batch`, on the
   same sizes, so the steady-state figure is measured exactly as a row is.

Every rank writes `rank<r>.json`; the parent requires all of them, checks the
rank-to-GPU mapping and the sums, and writes `probe.json`. Exit 0 only if every
check passed. No profile row is written.
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
from profiling.runners.comm.nccl import _all_reduce_per_rank_batch  # noqa: E402

PATTERN_MOD = 7  # inputs are (i % 7) + rank: small integers, exact in bf16


def _probe_per_rank(
    *, rank: int, world_size: int, out_dir: str, sizes: list[int], parent_start: float
) -> dict:
    import torch
    import torch.distributed as dist

    report: dict = {"rank": rank, "world_size": world_size}
    report["group_ready_s"] = time.time() - parent_start  # spawn + import + TCP rendezvous
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    report["device"] = {
        "current_device": device,
        "name": props.name,
        "uuid": f"GPU-{props.uuid}",
        "visible_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    report["backend"] = dist.get_backend()

    # First collective: NCCL communicator creation happens here.
    first = torch.full((1024,), float(rank + 1), dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dist.all_reduce(first)
    torch.cuda.synchronize()
    report["first_collective_s"] = time.perf_counter() - t0
    expected_first = world_size * (world_size + 1) / 2
    report["first_collective_ok"] = bool((first == expected_first).all().item())
    t0 = time.perf_counter()
    dist.all_reduce(first)
    torch.cuda.synchronize()
    report["second_collective_s"] = time.perf_counter() - t0

    checks = []
    for size in sizes:
        n = size // 2
        base = torch.arange(n, dtype=torch.int32, device="cuda") % PATTERN_MOD
        tensor = (base + rank).to(torch.bfloat16)
        dist.all_reduce(tensor)  # default op: SUM
        torch.cuda.synchronize()
        expected = (base * world_size + world_size * (world_size - 1) // 2).to(torch.bfloat16)
        mismatches = int((tensor != expected).sum().item())
        checks.append(
            {
                "message_size_bytes": size,
                "elements": n,
                "mismatches": mismatches,
                "sample": tensor[:8].float().tolist(),
                "expected_sample": expected[:8].float().tolist(),
            }
        )
        del base, tensor, expected
    report["sum_checks"] = checks

    # Steady state through the runner's own body (all ranks in lockstep).
    specs = [
        {"num_gpus": world_size, "message_size_bytes": s, "dtype": "bf16", "fabric": "nvlink"}
        for s in sizes
    ]
    timings = _all_reduce_per_rank_batch(
        rank=rank, world_size=world_size, specs=specs, warmup=_WARMUP, rep=_REP
    )
    report["runner_timings"] = timings  # rank 0 only, by the runner's contract
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _visible_uuids() -> list[str]:
    """UUIDs of the GPUs in CUDA_VISIBLE_DEVICES order (indices, PCI bus order)."""
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
    parser.add_argument("--sizes", type=int, nargs="+", default=[8192, 67108864])
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
    report: dict = {"num_gpus": args.num_gpus, "visible_uuids": visible, "sizes": args.sizes}
    if not problems:
        start = time.time()
        try:
            rank0 = TorchMpLauncher(args.num_gpus, backend="nccl").run(
                _probe_per_rank, out_dir=str(out_dir), sizes=args.sizes, parent_start=start
            )
        except Exception as exc:  # noqa: BLE001 - the probe reports any launcher failure
            rank0 = None
            problems.append(f"launcher failed: {exc}")
        report["launcher_wall_s"] = time.time() - start
        ranks = []
        for rank in range(args.num_gpus):
            path = out_dir / f"rank{rank}.json"
            if not path.exists():
                problems.append(f"rank {rank} wrote no report")
                continue
            r = json.loads(path.read_text())
            ranks.append(r)
            if r["device"]["uuid"] != args.expect_uuids[rank]:
                problems.append(f"rank {rank} holds {r['device']['uuid']}")
            if r["backend"] != "nccl":
                problems.append(f"rank {rank} backend is {r['backend']}")
            if not r["first_collective_ok"]:
                problems.append(f"rank {rank}: first collective sum wrong")
            for check in r["sum_checks"]:
                if check["mismatches"]:
                    problems.append(
                        f"rank {rank}: {check['mismatches']} mismatches "
                        f"at {check['message_size_bytes']} bytes"
                    )
        report["ranks"] = ranks
        if rank0 is None or not rank0.get("runner_timings"):
            problems.append("rank 0 returned no runner timings")
        else:
            report["steady_state"] = [
                {"message_size_bytes": s, **t}
                for s, t in zip(args.sizes, rank0["runner_timings"], strict=True)
            ]
    report["problems"] = problems
    report["pass"] = not problems
    (out_dir / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    for r in report.get("ranks", []):
        print(
            f"rank {r['rank']}: {r['device']['uuid']} group_ready={r['group_ready_s']:.2f}s "
            f"first_collective={r['first_collective_s']:.3f}s "
            f"second={r['second_collective_s'] * 1e3:.3f}ms "
            f"mismatches={[c['mismatches'] for c in r['sum_checks']]}"
        )
    for s in report.get("steady_state", []):
        print(
            f"steady {s['message_size_bytes']} B: {s['time_ms'] * 1e3:.2f} us "
            f"busbw={s['busbw_gbps']:.1f} GB/s"
        )
    print(f"NCCL PROBE pass={report['pass']} problems={problems}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
