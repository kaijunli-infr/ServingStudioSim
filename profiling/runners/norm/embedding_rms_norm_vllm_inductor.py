"""Profile the embedding gather + first RMSNorm as Inductor compiles vLLM's Llama graph.

The block is vLLM's own code at TP1: ``VocabParallelEmbedding`` looks rows up
with ``F.embedding(input_ids.long(), weight)`` (the unquantized method), and
``LlamaModel.forward`` hands the result to layer 0, whose
``input_layernorm`` with no residual yet returns ``(rms_norm(h), h)``: the
native ``rms_norm`` IR op and the embedding itself as the residual. Inductor
fuses the int32-to-int64 cast, the gather and the norm into one reduction
kernel; its source, compiled this way, matched the served
``triton_red_fused__to_copy_embedding_rms_norm_0`` body for body on GB300
(``profiling/env/arm64-gb300/PHASE3B_PILOT_RESULTS.md``).
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.norm.embedding_rms_norm_reference import embedding_rms_norm_reference
from profiling.runners.vllm_inductor import compile_like_vllm

_BACKEND = "embedding_rms_norm:vllm_inductor"
# Embedding rows of a trained model are small; the scale keeps the bf16 squares
# well inside range like a real table, without changing the kernel.
_TABLE_STD = 0.02
# The residual is a copy, so it must match bit for bit; normalized rows differ
# from the float32 reference by about one bf16 ulp of rounding order.
_NORMED_TOL = 2e-2


def validate_args(
    num_tokens: int,
    hidden_dim: int,
    vocab_size: int,
    dtype: DType | str,
    rms_eps: float,
    tune_tokens: int,
) -> None:
    for name, value in (
        ("num_tokens", num_tokens),
        ("hidden_dim", hidden_dim),
        ("vocab_size", vocab_size),
        ("tune_tokens", tune_tokens),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if vocab_size >= 2**31:
        raise ValueError("vocab_size must fit the int32 token ids vLLM passes")
    if not (isinstance(rms_eps, float) and 0.0 < rms_eps < 1.0):
        raise ValueError(f"rms_eps must be a float in (0, 1), got {rms_eps!r}")
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires dtype=bf16, got {dtype}")


def build_block(rms_norm: Any, embedding: Any, eps: float) -> Any:
    """vLLM's first piece over the given ``rms_norm(x, w, eps)`` and
    ``embedding(ids, table)``: returns ``(residual, normalized)``."""

    def block(ids: Any, table: Any, weight: Any) -> tuple[Any, Any]:
        hidden = embedding(ids.long(), table)
        return hidden, rms_norm(hidden, weight, eps)

    return block


def check_outputs(
    torch: Any, ids: Any, table: Any, weight: Any, eps: float, residual: Any, normed: Any
) -> None:
    """Raise unless both planes match the independent reference."""
    want_residual, want_normed = embedding_rms_norm_reference(torch, ids, table, weight, eps)
    if residual.shape != want_residual.shape or not torch.equal(residual, want_residual):
        raise KernelLaunchFailed(f"{_BACKEND}: the residual differs from the gathered rows")
    torch.testing.assert_close(
        normed.float(), want_normed.float(), rtol=_NORMED_TOL, atol=_NORMED_TOL
    )


def bytes_moved(num_tokens: int, hidden_dim: int) -> int:
    element = 2
    return num_tokens * (4 + 3 * hidden_dim * element) + hidden_dim * element


def profile_embedding_rms_norm_vllm_inductor(
    num_tokens: int,
    hidden_dim: int,
    vocab_size: int,
    dtype: DType | str,
    rms_eps: float,
    tune_tokens: int,
) -> ComputeMetrics:
    validate_args(num_tokens, hidden_dim, vocab_size, dtype, rms_eps, tune_tokens)
    try:
        import torch
        from vllm import ir
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the pinned vLLM environment") from exc

    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(0)
    table = _TABLE_STD * torch.randn(
        (vocab_size, hidden_dim), dtype=torch.bfloat16, device=device, generator=generator
    )
    weight = 1.0 + 0.1 * torch.randn(
        hidden_dim, dtype=torch.bfloat16, device=device, generator=generator
    )
    block = build_block(
        ir.ops.rms_norm.impls["native"].impl_fn, torch.nn.functional.embedding, rms_eps
    )

    def make_inputs(tokens: int) -> list[Any]:
        ids = torch.randint(
            0, vocab_size, (tokens,), dtype=torch.int32, device=device, generator=generator
        )
        return [ids, table, weight]

    try:
        compiled = compile_like_vllm(
            torch, block, make_inputs, {0: 0}, num_tokens, tune_tokens=tune_tokens
        )
        inputs = make_inputs(num_tokens)
        residual, normed = compiled(*inputs)
        torch.cuda.synchronize()
        check_outputs(torch, inputs[0], table, weight, rms_eps, residual, normed)

        def launch() -> None:
            compiled(*inputs)

        time_ms = Timer.cupti(launch, kernel_name=None)
        energy_j = Energy.perf(launch, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(bytes_moved(num_tokens, hidden_dim) / seconds / 1e9),
        energy_j=float(energy_j),
    )


__all__ = [
    "build_block",
    "bytes_moved",
    "check_outputs",
    "profile_embedding_rms_norm_vllm_inductor",
    "validate_args",
]
