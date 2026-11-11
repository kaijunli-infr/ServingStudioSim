"""NCCL (torch.distributed) all-reduce timed by CUDA-graph replay.

Same collective as ``nccl.py``: ``dist.all_reduce(tensor)`` (SUM) on a buffer of
``message_size_bytes``, one rank per GPU, spawned through ``TorchMpLauncher``.
Only the timing boundary differs, which is why this is its own backend
(``nccl_graph``) with its own rows rather than a change to ``nccl``:

- ``nccl`` brackets 100 eager calls with CUDA events. Below about 8 MiB the
  host needs longer to issue a call than the GPUs need to run it, so those rows
  hold the host's eager launch pace (see
  ``profiling/env/arm64-gb300/PHASE2E_RESULTS.md``).
- ``nccl_graph`` captures the call in a CUDA graph and times replays, the
  convention of the FlashInfer collective runners (``_time_cuda_graph`` in
  ``flashinfer_trtllm.py``): warm-up launches, capture, warm-up replays,
  ``dist.barrier()``, host wall time over the replays up to the end of
  ``torch.cuda.synchronize()``, per collective, slowest rank. A graph holds
  ``OPERATIONS_PER_GRAPH`` back-to-back collectives, as in ``flashinfer_mnnvl.py``.

One row is the median over ``GROUPS`` fresh rank groups of each group's median
over ``BATCHES`` batches of ``REPLAYS`` replays. Every graph is checked before
it is timed: the buffer is refilled with a deterministic pattern, the graph
replayed once, and the result compared with the closed-form sum on every rank;
a mismatch fails the chunk. ``fabric`` is a row/key field only.
"""

from __future__ import annotations

from statistics import median

from profiling.db.args import DType
from profiling.runners.comm._batch import _WARMUP, all_error
from profiling.runners.comm._launcher import TorchMpLauncher
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import CommMetrics, RunnerResult

# Recorded with every measurement artifact of this backend.
MEASUREMENT_METHOD = "nccl_torch_dist.cuda_graph_replay.ops10"
GROUPS = 5
BATCHES = 3
REPLAYS = 100
OPERATIONS_PER_GRAPH = 10  # as profiling/runners/comm/flashinfer_mnnvl.py
_PATTERN_MOD = 7  # check input is (i % 7) + rank: sums stay exact in bf16


def profile_all_reduce_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    """Profile a homogeneous chunk with ``GROUPS`` fresh rank groups, each looping
    every size, and reduce to the median of the group medians per spec. A failure
    in any group fails the whole chunk; no partial or eager value is returned."""
    if not kwargs_list:
        return []
    num_gpus = int(kwargs_list[0]["num_gpus"])
    if num_gpus < 2:
        return all_error(len(kwargs_list), "all-reduce needs at least 2 GPUs")
    group_batches: list[list[list[float]]] = []
    for _ in range(GROUPS):
        try:
            payloads = TorchMpLauncher(num_gpus, backend="nccl").run(
                _graph_replay_per_rank_batch,
                specs=kwargs_list,
                warmup=_WARMUP,
                replays=REPLAYS,
                batches=BATCHES,
            )
        except Exception as exc:  # noqa: BLE001 - the rank group tore down; fail the chunk
            return all_error(len(kwargs_list), str(exc))
        if payloads is None or len(payloads) != len(kwargs_list):
            got = 0 if payloads is None else len(payloads)
            return all_error(
                len(kwargs_list),
                f"graph-replay group returned {got} payloads for {len(kwargs_list)} specs",
            )
        group_batches.append(payloads)
    results = []
    for index, spec in enumerate(kwargs_list):
        time_ms = estimate_ms([group[index] for group in group_batches])
        results.append(RunnerResult(metrics=comm_metrics(time_ms, spec, num_gpus)))
    return results


def estimate_ms(batches_by_group: list[list[float]]) -> float:
    """Median over groups of each group's median batch time."""
    return median(median(batches) for batches in batches_by_group)


def comm_metrics(time_ms: float, spec: dict, world_size: int) -> CommMetrics:
    # Whole elements, as the rank body allocates them (bf16: 2 bytes).
    element_size = 2
    actual_bytes = max(1, int(spec["message_size_bytes"]) // element_size) * element_size
    latency_s = time_ms / 1000.0
    algbw_gbps = (actual_bytes / latency_s) / 1e9 if latency_s > 0 else 0.0
    # Ring all-reduce moves 2(N-1)/N of the payload across the bus.
    return CommMetrics(
        time_ms=time_ms,
        algbw_gbps=algbw_gbps,
        busbw_gbps=algbw_gbps * 2 * (world_size - 1) / world_size,
    )


def _graph_replay_per_rank_batch(
    *,
    rank: int,
    world_size: int,
    specs: list[dict],
    warmup: int,
    replays: int,
    batches: int,
) -> list[list[float]] | None:
    """Runs inside each spawned rank. Per size: SUM check, warm-up, capture,
    replay check, warm-up replays, ``batches`` timed batches. No per-size
    try/except, so the ranks never desync (see ``_batch``). Rank 0 returns the
    slowest-rank batch times per spec."""
    try:
        import time

        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for the NCCL all-reduce runner") from exc

    ops = OPERATIONS_PER_GRAPH
    results: list[list[float]] = []
    try:
        for spec in specs:
            torch_dtype = DType.from_value(spec["dtype"]).torch()
            element_size = torch.tensor([], dtype=torch_dtype).element_size()
            n = max(1, int(spec["message_size_bytes"]) // element_size)
            base = torch.arange(n, dtype=torch.int32, device="cuda") % _PATTERN_MOD
            pattern = (base + rank).to(torch_dtype)
            expected = (base * world_size + world_size * (world_size - 1) // 2).to(torch_dtype)
            del base

            tensor = pattern.clone()
            dist.all_reduce(tensor)  # default op: SUM
            torch.cuda.synchronize()
            eager_bad = int((tensor != expected).sum().item())
            tensor.copy_(torch.randn(n, dtype=torch_dtype, device="cuda"))
            for _ in range(warmup):
                dist.all_reduce(tensor)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(ops):
                    dist.all_reduce(tensor)
            torch.cuda.synchronize()
            tensor.copy_(pattern)
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            replay_bad = int((tensor != expected * float(world_size ** (ops - 1))).sum().item())
            # Every rank learns whether any rank saw a wrong sum, then all stop together.
            bad = torch.tensor(float(eager_bad + replay_bad), dtype=torch.float64, device="cuda")
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            if bad.item() != 0:
                raise KernelLaunchFailed(
                    f"all-reduce SUM mismatch at {spec['message_size_bytes']} bytes "
                    f"(rank {rank}: eager {eager_bad}, graph replay {replay_bad})"
                )

            tensor.copy_(torch.randn(n, dtype=torch_dtype, device="cuda"))
            for _ in range(max(1, warmup // ops)):
                graph.replay()
            torch.cuda.synchronize()
            batch_ms = []
            for _ in range(batches):
                dist.barrier()
                start = time.perf_counter()
                for _ in range(replays):
                    graph.replay()
                torch.cuda.synchronize()
                local_ms = (time.perf_counter() - start) / (replays * ops) * 1000.0
                slowest = torch.tensor(local_ms, dtype=torch.float64, device="cuda")
                dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
                batch_ms.append(float(slowest.item()))
            del graph, tensor, pattern, expected
            results.append(batch_ms)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc

    if rank != 0:
        return None
    return results
