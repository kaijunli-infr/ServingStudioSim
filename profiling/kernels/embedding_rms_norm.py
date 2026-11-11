"""Token embedding gather fused with the first decoder layer's input RMSNorm."""

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

KIND = "embedding_rms_norm"


@dataclass(frozen=True)
class EmbeddingRmsNormArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token ids looked up and normalized together.")
    hidden_dim: int = arg(unit="elements", doc="Width of each embedding row.")
    vocab_size: int = arg(unit="rows", doc="Rows of the embedding table the ids index.")
    dtype: DType = arg(doc="Element type of the table, the weight and both stored planes.")
    rms_eps: float = arg(
        unit="unitless", doc="Stabilizer added to the mean square; the model's rms_norm_eps."
    )
    tune_tokens: int = arg(
        unit="tokens",
        doc=(
            "Token count of the first call, which picks Inductor's launch config: the "
            "deployment's max_num_batched_tokens (vLLM's profile run)."
        ),
    )


DOC = KernelDoc(
    title="Embedding gather + RMSNorm",
    summary=(
        "Look up each token's embedding row by id, store it as the residual, and "
        "store its RMS-normalized copy, in one kernel."
    ),
    description=(
        "The first piece of vLLM's compiled Llama graph is the token embedding "
        "(F.embedding of the int32 ids, cast to int64) followed by the first "
        "decoder layer's input RMSNorm, whose residual is the embedding itself. "
        "Inductor fuses both into one Triton reduction kernel, "
        "triton_red_fused__to_copy_embedding_rms_norm, that reads a token id, "
        "gathers that row of the table, writes it as the residual and writes the "
        "normalized row. A plain RMSNorm over contiguous rows has no gather and no "
        "residual store and is a different kernel. Token ids are seeded uniform "
        "random over the table."
    ),
    category="Normalization",
    subcategory="RMSNorm",
    formula=(
        "r = table[id], one row per token",
        "n = r / √(mean(r²) + ε) · weight, accumulated in fp32",
        "outputs: r (the residual) and n, both in dtype",
        "GB/s = (m·4 + 3·m·h·b + h·b) / time, with m = num_tokens, h = hidden_dim, "
        "b = bytes per element (row read, two row writes, int32 id, weight)",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} The block is compiled once with a symbolic token count, "
        "and its first call, at tune_tokens or at the row's token count when larger, "
        "picks the launch config the row then uses, as vLLM's profile run at "
        "max_num_batched_tokens does. Every launch of the compiled block is counted."
    ),
    caveats=(
        "The gathered rows are random and cold; served ids repeat common tokens, "
        "whose rows may be L2-resident.",
        "GB/s counts the gathered row once although the kernel reads it twice, once "
        "for the mean square and once for the outputs; the second read mostly hits L2.",
    ),
    reference="profiling.runners.norm.embedding_rms_norm_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_inductor",
        # Inductor emits a portable Triton reduction; Triton lowers bf16 only on SM80+.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.norm.embedding_rms_norm_vllm_inductor",
            function_name="profile_embedding_rms_norm_vllm_inductor",
        ),
        table_name=KIND,
        args_schema=EmbeddingRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "VocabParallelEmbedding's F.embedding and vLLM's native rms_norm, "
                "compiled by Inductor into one Triton reduction kernel."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py",
        ),
    )
)

__all__ = ["KIND", "EmbeddingRmsNormArgs"]
