"""SiLU-and-mul activation of a dense gated MLP."""

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

KIND = "silu_and_mul"


@dataclass(frozen=True)
class SiluAndMulArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows entering the activation.")
    hidden_dim: int = arg(
        unit="elements", doc="Width of the gate half and of the up half of each row."
    )
    dtype: DType = arg(doc="Element type of the input and output.")


DOC = KernelDoc(
    title="SiLU and mul",
    summary="Multiply SiLU of the gate half of each row by its up half.",
    description=(
        "A gated MLP applies this activation to the output of its fused gate-up "
        "projection: each input row holds a gate half and an up half, each "
        "hidden_dim wide, and the output row is hidden_dim wide. Under "
        "torch.compile vLLM's native SiluAndMul becomes one Inductor pointwise "
        "kernel; a following quantization of the down projection's input stays a "
        "separate kernel when it is a custom op. Inputs are seeded random bf16 rows."
    ),
    category="Other",
    formula=(
        "y = SiLU(gate) · up, computed in fp32",
        "GB/s = 6 · num_tokens · hidden_dim / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} The block is compiled once with a symbolic token count, "
        "and its first call, at 8192 tokens or at the row's token count when larger, "
        "picks the launch config the row then uses, as vLLM's profile run at "
        "max_num_batched_tokens does. Every launch of the compiled block is counted. "
        "vllm_cuda: the CUDA custom op writes into a preallocated output and is one launch. "
        "vllm_inductor_graph: the same compiled kernel timed by CUDA-graph replay "
        "(profiling.profilers.graph_replay): the call is captured once, checked after "
        "its first replay and after the timed ones, and timed as the CUPTI kernel time "
        "of 3 batches of 100 back-to-back replays after 20 warm-up replays, with no L2 "
        "flush; the median of the batch medians."
    ),
    caveats=(
        "GB/s counts the two input halves and the output once each.",
        "vllm_inductor (cold, one eager launch) and vllm_inductor_graph (warm, graph "
        "replay) time one kernel at two boundaries; against the served GB300 kernel, "
        "graph replay read +3 to +7 % and the cold launch +23 to +56 % (8 and 256 tokens).",
    ),
    reference="profiling.runners.elementwise.silu_and_mul_reference",
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
            module_name="profiling.runners.elementwise.silu_and_mul_vllm_inductor",
            function_name="profile_silu_and_mul_vllm_inductor",
        ),
        table_name=KIND,
        args_schema=SiluAndMulArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="vLLM's SiluAndMul.forward_native compiled by Inductor into one Triton kernel.",
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/activation.py",
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
            module_name="profiling.runners.elementwise.silu_and_mul_vllm_inductor",
            function_name="profile_silu_and_mul_vllm_inductor_graph",
        ),
        table_name=KIND,
        args_schema=SiluAndMulArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's SiluAndMul.forward_native compiled by Inductor, timed by CUDA-graph "
                "replay with a warm L2 (opt-in; separate rows from vllm_inductor)."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/activation.py",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        # csrc/libtorch_stable/activation_kernels.cu act_and_mul_kernel: a plain vectorized
        # elementwise CUDA kernel with no architecture-specific path; vLLM itself
        # serves bf16 only on SM80+ (platforms/cuda.py check_if_supports_dtype).
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.silu_and_mul_vllm_cuda",
            function_name="profile_silu_and_mul_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=SiluAndMulArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="vLLM's SiluAndMul.forward_cuda: the torch.ops._C.silu_and_mul custom op.",
            url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/activation_kernels.cu",
        ),
    )
)

__all__ = ["KIND", "SiluAndMulArgs"]
