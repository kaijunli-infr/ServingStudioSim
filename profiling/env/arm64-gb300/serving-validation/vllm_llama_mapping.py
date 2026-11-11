"""Leaf mapping of a vLLM-specific Llama 3.1 8B BF16 TP1 variant (design data, CPU-only).

Phase 3A of the GB300 serving validation (`PHASE3A_DESIGN.md`). This module is
data plus accounting helpers: it names, for each operation of the proposed
`llama3_dense` vLLM variant, the kernel kind, backend and configuration that
would price it, which `llama3_dense` leaves it replaces, and which kernels of
the measured vLLM trace (roles of `tp1_trace_audit.py`) it covers. Nothing in
the simulator reads it; `test_vllm_llama_mapping.py` checks it against the
profiling registry and against the traced kernel counts.

Facts it encodes come from the TP1 smoke trace and its compiled Inductor code
(`$GB300_ROOT/serving-validation/cache/vllm/torch_compile_cache`):

- `triton_red_fused__to_copy_embedding_rms_norm_0` gathers the embedding row,
  stores it (the residual) and stores its RMSNorm.
- `triton_red_fused_fused_add_rms_norm_0` (post-attention) sums attention
  output and residual and stores only the normed result.
- `triton_red_fused_fused_add_rms_norm_2` (input norm, layers 1-31) sums MLP
  output, attention output and the old residual, and stores the new residual
  and the normed result; the last one (final norm) stores only the normed
  result, in place.
- `triton_poi_fused_<n>` reads the merged QKV output, positions and the cos/sin
  cache and writes rotated Q (32 x 128) and K (8 x 128): NeoX RoPE over the
  full head.
- The GEMMs are `extern_kernels.mm(x, weight.T)`, cuBLAS with the weight in
  `[n, k]` layout, as `F.linear`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NUM_LAYERS = 32
HIDDEN = 4096
INTERMEDIATE = 14336
NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
VOCAB = 128256
KV_BLOCK_SIZE = 16  # the trace's attention kernel name carries `P16`
# Serving configuration the Inductor norms are keyed by (Phase 3B-2): vLLM's
# max_num_batched_tokens (the tuning size) and Llama 3.1's rms_norm_eps.
MAX_NUM_BATCHED_TOKENS = 2048
RMS_EPS = 1e-5
SERVED_NORM = {"tune_tokens": MAX_NUM_BATCHED_TOKENS, "rms_eps": RMS_EPS}

# The leaves of `llama3_dense` (simulator/src/arch/llama3_dense.rs) and how
# many times one iteration charges each.
ORIGINAL_LEAVES = {
    "embedding": 1,
    "input_norm": NUM_LAYERS,
    "qkv_proj": NUM_LAYERS,
    "kv_cache_append": NUM_LAYERS,
    "prefill": NUM_LAYERS,
    "decode": NUM_LAYERS,
    "o_proj": NUM_LAYERS,
    "post_norm": NUM_LAYERS,
    "up_gate_proj": NUM_LAYERS,
    "activation": NUM_LAYERS,
    "down_proj": NUM_LAYERS,
    "final_norm": 1,
    "lm_head": 1,
}


@dataclass(frozen=True)
class Leaf:
    name: str
    kind: str
    backend: str
    config: dict
    per_iteration: int
    replaces: tuple[str, ...]
    trace_roles: tuple[str, ...]
    semantics: str  # equivalent | approximate | blocked | new_kind
    note: str = ""
    kernels_per_call: dict = field(default_factory=dict)  # trace role -> kernels per call


GEMM_BACKEND = "torch_linear"  # F.linear with weight [n, k]: the layout vLLM's graph calls

VARIANT_LEAVES: tuple[Leaf, ...] = (
    Leaf(
        "first_norm",
        "embedding_rms_norm",
        "vllm_inductor",
        {
            "hidden_dim": HIDDEN,
            "vocab_size": VOCAB,
            "dtype": "bf16",
            **SERVED_NORM,
        },
        1,
        ("embedding", "input_norm"),
        ("embed_input_norm",),
        "equivalent",
        "Phase 3B-2 kind: the served fused kernel (gather by int32 id, store the row as "
        "the residual and its RMSNorm). Replaces the Phase 3A 1-addend approximation, "
        "which the runner rejects and which the pilot measured as a different kernel.",
    ),
    Leaf(
        "qkv_proj",
        "single_gemm",
        GEMM_BACKEND,
        {"n": (NUM_QO_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM, "k": HIDDEN, "dtype": "bf16"},
        NUM_LAYERS,
        ("qkv_proj",),
        ("qkv_proj",),
        "equivalent",
    ),
    Leaf(
        "rope",
        "gqa_rope_neox",
        "vllm_inductor_graph",
        {
            "num_qo_heads": NUM_QO_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "rotary_dim": HEAD_DIM,
            "max_position": 131072,
            "dtype": "bf16",
        },
        NUM_LAYERS,
        (),
        ("rope",),
        "equivalent",
        "New kind (Phase 3B-0): no earlier kind applied RoPE to Q and K of a plain GQA layer. "
        "Graph-replay rows (Phase 3B-2).",
    ),
    Leaf(
        "kv_cache_append",
        "kv_cache_append",
        "vllm_cuda",
        {
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "block_size": KV_BLOCK_SIZE,
            "input_dtype": "bf16",
            "kv_dtype": "bf16",
            "cache_layout": "HND_PACKED",
            "scale_granularity": "tensor",
            "value_row_stride": (NUM_QO_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM,
        },
        NUM_LAYERS,
        ("kv_cache_append",),
        ("kv_cache_append",),
        "equivalent",
        "Same kernel over the served layout (Phase 3B-2): vLLM's packed FlashInfer cache, "
        "V read in place from the QKV output.",
    ),
    Leaf(
        "attention_decode",
        "paged_attn_batch_decode",
        "flashinfer_trtllm_gen",
        {
            "num_qo_heads": NUM_QO_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "q_dtype": "bf16",
            "kv_dtype": "bf16",
            "o_dtype": "bf16",
            "page_size": KV_BLOCK_SIZE,
        },
        NUM_LAYERS,
        ("decode",),
        ("attention_aux_fill", "attention"),
        "equivalent",
        "Same call and launch unit (counter fill + trtllm-gen kernel); the backend takes a "
        "bf16 query over a bf16 cache since Phase 3B-0.",
        {"attention_aux_fill": 1, "attention": 1},
    ),
    Leaf(
        "attention_prefill",
        "paged_attn_batch_prefill",
        "flashinfer_trtllm_gen",
        {
            "num_qo_heads": NUM_QO_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "q_dtype": "bf16",
            "kv_dtype": "bf16",
            "o_dtype": "bf16",
            "page_size": KV_BLOCK_SIZE,
        },
        NUM_LAYERS,
        ("prefill",),
        (),
        "equivalent",
        "Zero in a pure decode step.",
    ),
    Leaf(
        "o_proj",
        "single_gemm",
        GEMM_BACKEND,
        {"n": HIDDEN, "k": NUM_QO_HEADS * HEAD_DIM, "dtype": "bf16"},
        NUM_LAYERS,
        ("o_proj",),
        ("o_proj",),
        "equivalent",
        "Split-K 2 plus a reduce kernel in the trace; the row's CUPTI union includes both.",
        {"o_proj": 2},
    ),
    Leaf(
        "post_norm",
        "multi_add_rms_norm",
        "vllm_inductor",
        {
            "hidden_dim": HIDDEN,
            "num_addends": 2,
            "store_sum": False,
            "store_normed": True,
            "store_fp8": False,
            "dtype": "bf16",
            **SERVED_NORM,
        },
        NUM_LAYERS,
        ("post_norm",),
        ("post_norm",),
        "equivalent",
    ),
    Leaf(
        "up_gate_proj",
        "single_gemm",
        GEMM_BACKEND,
        {"n": 2 * INTERMEDIATE, "k": HIDDEN, "dtype": "bf16"},
        NUM_LAYERS,
        ("up_gate_proj",),
        ("up_gate_proj",),
        "equivalent",
    ),
    Leaf(
        "activation",
        "silu_and_mul",
        "vllm_inductor_graph",
        {"hidden_dim": INTERMEDIATE, "dtype": "bf16"},
        NUM_LAYERS,
        ("activation",),
        ("activation",),
        "equivalent",
    ),
    Leaf(
        "down_proj",
        "single_gemm",
        GEMM_BACKEND,
        {"n": HIDDEN, "k": INTERMEDIATE, "dtype": "bf16"},
        NUM_LAYERS,
        ("down_proj",),
        ("down_proj",),
        "equivalent",
        "Split-K 7 plus a reduce kernel in the trace; the row's CUPTI union includes both.",
        {"down_proj": 2},
    ),
    Leaf(
        "next_input_norm",
        "multi_add_rms_norm",
        "vllm_inductor",
        {
            "hidden_dim": HIDDEN,
            "num_addends": 3,
            "store_sum": True,
            "store_normed": True,
            "store_fp8": False,
            "dtype": "bf16",
            **SERVED_NORM,
        },
        NUM_LAYERS - 1,
        ("input_norm",),
        ("input_norm",),
        "equivalent",
        "The input norm of layers 1-31, issued at the end of the previous layer.",
    ),
    Leaf(
        "final_norm",
        "multi_add_rms_norm",
        "vllm_inductor",
        {
            "hidden_dim": HIDDEN,
            "num_addends": 3,
            "store_sum": False,
            "store_normed": True,
            "store_fp8": False,
            "dtype": "bf16",
            **SERVED_NORM,
        },
        1,
        ("final_norm",),
        ("final_norm",),
        "equivalent",
    ),
    Leaf(
        "lm_head",
        "single_gemm",
        GEMM_BACKEND,
        {"n": VOCAB, "k": HIDDEN, "dtype": "bf16"},
        1,
        ("lm_head",),
        ("lm_head",),
        "equivalent",
        "In the trace it runs in the sample phase, on the 8 sampled rows.",
    ),
)

# Trace roles deliberately left outside the architecture model (serving execution model).
OUTSIDE_ARCHITECTURE = ("runtime",)


def charged_original_leaves() -> dict[str, int]:
    """How many times per iteration each original leaf is replaced by the variant."""
    out: dict[str, int] = {name: 0 for name in ORIGINAL_LEAVES}
    for leaf in VARIANT_LEAVES:
        for name in leaf.replaces:
            out[name] += leaf.per_iteration
    return out


def charged_trace_kernels(prefill: bool = False) -> dict[str, int]:
    """Traced kernels per iteration that the variant's leaves account for."""
    out: dict[str, int] = {}
    for leaf in VARIANT_LEAVES:
        if leaf.name == "attention_prefill" and not prefill:
            continue
        for role in leaf.trace_roles:
            out[role] = out.get(role, 0) + leaf.per_iteration * leaf.kernels_per_call.get(role, 1)
    return out
