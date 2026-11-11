"""Profile GQA RoPE as vLLM's compiled Llama attention runs it.

The kernel this measures is the one the GB300 Llama 3.1 8B TP1 vLLM capture
shows between the QKV GEMM and the KV write, ``triton_poi_fused_<n>``. Its
generated source (vLLM's ``torch_compile_cache`` of that run) reads the query
and key slices of the QKV GEMM output in place (row stride (Hq + 2 Hkv) * D),
the position of each token and the cos/sin cache row it indexes, rotates every
head in NeoX halves, and writes contiguous key and query tensors.

The block reproduces ``LlamaAttention.forward`` up to attention: the QKV split
and ``rotary_emb.forward_native``. ``forward_native``, not ``rotary_emb(...)``:
RotaryEmbedding is a CustomOp, and the enabled op is an opaque CUDA kernel
Inductor cannot fuse, while the served graph took the native path. The value
slice is returned untouched by the block, as the attention op consumes it; it
adds no kernel.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.gqa_rope_neox_reference import gqa_rope_neox_reference
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.vllm_inductor import compile_like_vllm

_BACKEND = "gqa_rope_neox:vllm_inductor"
# Llama 3.1's rope parameters (config.json rope_scaling, rope_theta). They set
# the cos/sin values only, never the kernel, so they are not part of the key.
LLAMA3_ROPE_PARAMETERS = {
    "rope_type": "llama3",
    "rope_theta": 500000.0,
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}
# bf16 comparison: rounding of the rotated values to bf16 differs between the
# fused kernel and the float32 reference by about one ulp.
_RTOL = 2e-2
_ATOL = 2e-2


def validate_args(
    num_tokens: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    dtype: DType | str,
) -> None:
    for name, value in (
        ("num_tokens", num_tokens),
        ("num_qo_heads", num_qo_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
        ("rotary_dim", rotary_dim),
        ("max_position", max_position),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if num_qo_heads % num_kv_heads:
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
    if rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be even and at most head_dim")
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires dtype=bf16, got {dtype}")


def bytes_moved(
    num_tokens: int, num_qo_heads: int, num_kv_heads: int, head_dim: int, rotary_dim: int
) -> int:
    """Query and key read and written once, one cos/sin row and one int64 position per token."""
    element = 2
    heads = num_qo_heads + num_kv_heads
    return num_tokens * (2 * heads * head_dim * element + rotary_dim * element + 8)


def profile_gqa_rope_neox_vllm_inductor(
    num_tokens: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    dtype: DType | str,
) -> ComputeMetrics:
    validate_args(num_tokens, num_qo_heads, num_kv_heads, head_dim, rotary_dim, max_position, dtype)
    try:
        import torch
        from vllm.model_executor.layers.rotary_embedding import get_rope
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the pinned vLLM environment") from exc

    device = torch.device("cuda", torch.cuda.current_device())
    q_size, kv_size = num_qo_heads * head_dim, num_kv_heads * head_dim
    rope = get_rope(
        head_dim,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters={**LLAMA3_ROPE_PARAMETERS, "partial_rotary_factor": rotary_dim / head_dim},
        dtype=torch.bfloat16,
    ).to(device)
    if rope.rotary_dim != rotary_dim:
        raise ProfilerNotImplemented(f"{_BACKEND}: vLLM built rotary_dim {rope.rotary_dim}")
    generator = torch.Generator(device=device).manual_seed(0)

    def block(positions: Any, qkv: Any) -> tuple[Any, Any]:
        q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
        return rope.forward_native(positions, q, k)

    def make_inputs(tokens: int) -> list[Any]:
        positions = torch.randint(
            0, max_position, (tokens,), device=device, generator=generator, dtype=torch.int64
        )
        qkv = torch.randn(
            (tokens, q_size + 2 * kv_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        return [positions, qkv]

    try:
        compiled = compile_like_vllm(torch, block, make_inputs, {0: 0, 1: 0}, num_tokens)
        inputs = make_inputs(num_tokens)
        q, k = compiled(*inputs)
        torch.cuda.synchronize()
        want_q, want_k = gqa_rope_neox_reference(
            torch,
            inputs[0],
            inputs[1],
            rope.cos_sin_cache,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
        )
        torch.testing.assert_close(q.reshape(want_q.shape), want_q, rtol=_RTOL, atol=_ATOL)
        torch.testing.assert_close(k.reshape(want_k.shape), want_k, rtol=_RTOL, atol=_ATOL)

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
    moved = bytes_moved(num_tokens, num_qo_heads, num_kv_heads, head_dim, rotary_dim)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(moved / seconds / 1e9),
        energy_j=float(energy_j),
    )


__all__ = ["profile_gqa_rope_neox_vllm_inductor", "validate_args", "bytes_moved"]
