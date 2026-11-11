"""Profile vLLM's native SiLU-and-mul as Inductor compiles it into the served graph.

Two timing boundaries over the same compiled kernel, each its own backend:
``vllm_inductor`` (``Timer.cupti``: one eager launch after an L2 flush) and
``vllm_inductor_graph`` (``profiling.profilers.graph_replay``: back-to-back
CUDA-graph replays, warm L2).
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.elementwise.silu_and_mul_reference import silu_and_mul_reference
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.vllm_inductor import compile_like_vllm

_BACKEND = "silu_and_mul:vllm_inductor"
_GRAPH_BACKEND = "silu_and_mul:vllm_inductor_graph"


def _validate(num_tokens: int, hidden_dim: int, dtype: DType | str, backend: str) -> None:
    for name, value in (("num_tokens", num_tokens), ("hidden_dim", hidden_dim)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{backend} requires dtype=bf16, got {dtype}")


def _compiled_case(torch: Any, num_tokens: int, hidden_dim: int) -> tuple[Any, Any]:
    """``(launch, check)``: one call of the compiled block over seeded inputs,
    and the oracle for its output."""
    from vllm.model_executor.layers.activation import SiluAndMul

    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(0)

    def make_inputs(tokens: int) -> list[Any]:
        return [
            torch.randn(
                (tokens, 2 * hidden_dim), dtype=torch.bfloat16, device=device, generator=generator
            )
        ]

    compiled = compile_like_vllm(torch, SiluAndMul.forward_native, make_inputs, {0: 0}, num_tokens)
    inputs = make_inputs(num_tokens)

    def launch() -> Any:
        return compiled(*inputs)

    def check(actual: Any) -> None:
        torch.testing.assert_close(
            actual, silu_and_mul_reference(torch, inputs[0]), rtol=2e-2, atol=2e-2
        )

    return launch, check


def _metrics(time_ms: float, energy_j: float, num_tokens: int, hidden_dim: int) -> ComputeMetrics:
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(6 * num_tokens * hidden_dim / seconds / 1e9),
        energy_j=float(energy_j),
    )


def _import_torch(backend: str) -> Any:
    try:
        import torch
        import vllm.model_executor.layers.activation  # noqa: F401
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{backend} requires the pinned vLLM environment") from exc
    return torch


def profile_silu_and_mul_vllm_inductor(
    num_tokens: int, hidden_dim: int, dtype: DType | str
) -> ComputeMetrics:
    _validate(num_tokens, hidden_dim, dtype, _BACKEND)
    torch = _import_torch(_BACKEND)
    try:
        launch, check = _compiled_case(torch, num_tokens, hidden_dim)
        actual = launch()
        torch.cuda.synchronize()
        check(actual)

        def timed() -> None:
            launch()

        time_ms = Timer.cupti(timed, kernel_name=None)
        energy_j = Energy.perf(timed, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc
    return _metrics(time_ms, energy_j, num_tokens, hidden_dim)


def profile_silu_and_mul_vllm_inductor_graph(
    num_tokens: int, hidden_dim: int, dtype: DType | str
) -> ComputeMetrics:
    from profiling.profilers.graph_replay import time_graph_replay

    _validate(num_tokens, hidden_dim, dtype, _GRAPH_BACKEND)
    torch = _import_torch(_GRAPH_BACKEND)
    try:
        launch, check = _compiled_case(torch, num_tokens, hidden_dim)
        timing, replay = time_graph_replay(launch, check, torch=torch)
        energy_j = Energy.perf(replay, warmup=5, per_iter_time_ms=timing.time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_GRAPH_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_GRAPH_BACKEND} failed: {exc}") from exc
    return _metrics(timing.time_ms, energy_j, num_tokens, hidden_dim)


__all__ = ["profile_silu_and_mul_vllm_inductor", "profile_silu_and_mul_vllm_inductor_graph"]
