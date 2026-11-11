"""Rotary position embedding of the query and key heads of a GQA attention layer."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND = "gqa_rope_neox"


@dataclass(frozen=True)
class GqaRopeNeoxArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens whose query and key heads are rotated.")
    num_qo_heads: int = arg(unit="heads", doc="Query heads per token.")
    num_kv_heads: int = arg(unit="heads", doc="Key (and value) heads per token.")
    head_dim: int = arg(unit="elements", doc="Width of each head.")
    rotary_dim: int = arg(unit="elements", doc="Leading elements of each head that are rotated.")
    max_position: int = arg(unit="positions", doc="Rows of the cos/sin cache the positions index.")
    dtype: DType = arg(doc="Element type of the projections, the cache read and the outputs.")


DOC = KernelDoc(
    title="GQA RoPE (NeoX)",
    summary="Rotate the query and key heads of a GQA layer by their token positions.",
    description=(
        "A plain GQA attention layer (Llama) splits its fused QKV projection output "
        "into query, key and value and applies rotary position embedding to the "
        "query and key heads before the key and value are written to the cache. "
        "Under torch.compile vLLM's native RotaryEmbedding becomes one Inductor "
        "pointwise kernel that reads the query and key slices of the QKV output in "
        "place, indexes the cos/sin cache by each token's position, rotates the "
        "first rotary_dim elements of every head in NeoX halves (element i pairs "
        "with i + rotary_dim/2), and writes contiguous query and key tensors. The "
        "value slice is not touched. Inputs are seeded random bf16 rows and random "
        "positions; the cos/sin values are Llama 3.1's."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "x1, x2 = x[:r/2], x[r/2:r];  out = [x1·cos − x2·sin, x2·cos + x1·sin, x[r:]]",
        "GB/s = num_tokens · (2 · (Hq + Hkv) · head_dim · b + rotary_dim · b + 8) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} The block (QKV split + RotaryEmbedding.forward_native) is "
        "compiled once with a symbolic token count, and its first call, at 8192 "
        "tokens or at the row's token count when larger, picks the launch config, "
        "as vLLM's profile run at max_num_batched_tokens does. Every launch of the "
        "compiled block is counted. "
        "vllm_inductor_graph: the same compiled kernel timed by CUDA-graph replay "
        "(profiling.profilers.graph_replay): the call is captured once, checked after "
        "its first replay and after the timed ones, and timed as the CUPTI kernel time "
        "of 3 batches of 100 back-to-back replays after 20 warm-up replays, with no L2 "
        "flush; the median of the batch medians."
    ),
    caveats=(
        "The cos/sin values (rope theta and scaling) do not change the kernel and "
        "are not part of the key; the runner uses Llama 3.1 8B's.",
        "GB/s counts the query and key read and written once each, one cos/sin row "
        "and one int64 position per token.",
        "vllm_inductor (cold, one eager launch) and vllm_inductor_graph (warm, graph "
        "replay) time one kernel at two boundaries; against the served GB300 kernel, "
        "graph replay read 0 to -5 % and the cold launch +8 to +34 % (8 and 256 tokens).",
    ),
    reference="profiling.runners.attention.gqa_rope_neox_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_inductor",
        # Inductor emits a portable Triton pointwise kernel; Triton lowers bf16 only on SM80+.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gqa_rope_neox_vllm_inductor",
            function_name="profile_gqa_rope_neox_vllm_inductor",
        ),
        table_name=KIND,
        args_schema=GqaRopeNeoxArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's QKV split plus RotaryEmbedding.forward_native, compiled by Inductor "
                "into one Triton kernel, as LlamaAttention.forward runs it."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding/base.py",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_inductor_graph",
        # Same Inductor kernel as vllm_inductor; only the timing boundary differs,
        # so its rows are a separate backend and never share a key with eager rows.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gqa_rope_neox_vllm_inductor",
            function_name="profile_gqa_rope_neox_vllm_inductor_graph",
        ),
        table_name=KIND,
        args_schema=GqaRopeNeoxArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's QKV split plus RotaryEmbedding.forward_native compiled by "
                "Inductor, timed by CUDA-graph replay with a warm L2 (opt-in; separate "
                "rows from vllm_inductor)."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding/base.py",
        ),
    )
)

__all__ = ["KIND", "GqaRopeNeoxArgs"]
