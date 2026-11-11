"""CUPTI kernel time of a callable replayed from a CUDA graph (opt-in timing mode).

vLLM serves decode steps from CUDA graphs, back to back: every kernel's input
was just written by the kernel before it, nothing is flushed between steps, and
there is no host launch gap. For small activation-to-activation kernels
(SiLU-and-mul, RoPE) the default ``Timer.cupti`` boundary, one eager launch
after a 64 MiB L2 flush, read 23-56 % above the served kernel on GB300, while
the device time of graph replays read within +7 %
(``profiling/env/arm64-gb300/PHASE3B_PILOT_RESULTS.md``). This module is that
second boundary. It is opt-in: a runner uses it under its own backend name, so
its rows never share a key with the eager rows of the same kernel.

``time_graph_replay`` fixes every count and condition:

1. ``EAGER_WARMUP`` eager calls on a side stream (the ``torch.cuda.graph``
   convention; compiles and autotunes before capture), then one CUPTI capture
   of one eager call that records the kernel names the call launches.
2. Capture of one call into a graph with its own memory pool; inputs are the
   caller's, allocated before capture, and outputs live in the pool.
3. One replay, then ``check(outputs)``: the caller's correctness oracle.
4. ``WARMUP_REPLAYS`` untimed replays, then ``BATCHES`` CUPTI captures of
   ``REPLAYS`` back-to-back replays each. L2 is never flushed. Every replay
   must launch exactly the eager call's kernels, in order; anything else (a
   memcpy kernel, a missing or extra launch) fails the measurement instead of
   being timed.
5. ``check(outputs)`` again after the timed replays.

The time of one replay is the sum of its kernels' CUPTI durations (device
kernel time, not the gaps between kernels); a batch's time is the median over
its replays and the result the median over batches. The batch spread is
returned for the caller to record.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

# Recorded with every measurement of a backend that uses this mode.
MEASUREMENT_METHOD = "cupti.cuda_graph_replay.warm_l2.eager3.warm20.batches3x100"
EAGER_WARMUP = 3
WARMUP_REPLAYS = 20
REPLAYS = 100
BATCHES = 3


class GraphReplayError(RuntimeError):
    """A replay launched something other than the eager call's kernels."""


@dataclass(frozen=True)
class GraphReplayTiming:
    time_ms: float
    batch_ms: tuple[float, ...]
    kernel_names: tuple[str, ...]
    method: str = MEASUREMENT_METHOD


def per_replay_ms(records: Sequence[Any], kernel_names: Sequence[str], replays: int) -> list[float]:
    """Split one capture of ``replays`` replays into per-replay kernel time (ms).

    ``records`` carry ``name``, ``start_ns`` and ``duration_ns``; ``kernel_names``
    is the ordered launch sequence of one eager call. Raises unless the capture
    holds exactly ``replays`` copies of that sequence.
    """
    if replays <= 0 or not kernel_names:
        raise ValueError("replays and kernel_names must be non-empty")
    ordered = sorted(records, key=lambda record: record.start_ns)
    per = len(kernel_names)
    if len(ordered) != replays * per:
        raise GraphReplayError(
            f"{replays} replays launched {len(ordered)} kernels, expected {replays * per} "
            f"({list(kernel_names)} each)"
        )
    times = []
    for index in range(replays):
        launch = ordered[index * per : (index + 1) * per]
        names = [record.name for record in launch]
        if names != list(kernel_names):
            raise GraphReplayError(
                f"replay {index} launched {names}, the eager call {list(kernel_names)}"
            )
        times.append(sum(record.duration_ns for record in launch) / 1e6)
    return times


def time_graph_replay(
    fn: Callable[[], Any],
    check: Callable[[Any], None],
    *,
    torch: Any,
    profiler: Any = None,
) -> tuple[GraphReplayTiming, Callable[[], None]]:
    """Time ``fn`` by graph replay; return the timing and the replay callable.

    ``fn`` launches the measured kernels on the current stream from inputs it
    closes over and returns its outputs; ``check`` raises when they are wrong.
    The returned callable replays the graph (for energy or a further probe) and
    keeps the graph and its pool alive.
    """
    from profiling.profilers.cupti_kernel_profiler import CuptiKernelProfiler

    profiler = profiler or CuptiKernelProfiler()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(EAGER_WARMUP):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    eager = profiler.capture_once(fn, clear_l2_before_run=False, clear_l2_between_launches=False)
    kernel_names = tuple(record.name for record in sorted(eager, key=lambda r: r.start_ns))
    if not kernel_names:
        raise GraphReplayError("the eager call launched no kernel")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
        outputs = fn()
    graph.replay()
    torch.cuda.synchronize()
    check(outputs)

    for _ in range(WARMUP_REPLAYS):
        graph.replay()
    torch.cuda.synchronize()
    batch_ms = []
    for _ in range(BATCHES):
        records = profiler.capture_once(
            graph.replay,
            launches_per_run=REPLAYS,
            clear_l2_before_run=False,
            clear_l2_between_launches=False,
        )
        batch_ms.append(median(per_replay_ms(records, kernel_names, REPLAYS)))
    check(outputs)

    def replay() -> None:
        graph.replay()

    timing = GraphReplayTiming(
        time_ms=float(median(batch_ms)), batch_ms=tuple(batch_ms), kernel_names=kernel_names
    )
    return timing, replay


__all__ = [
    "BATCHES",
    "EAGER_WARMUP",
    "MEASUREMENT_METHOD",
    "REPLAYS",
    "WARMUP_REPLAYS",
    "GraphReplayError",
    "GraphReplayTiming",
    "per_replay_ms",
    "time_graph_replay",
]
