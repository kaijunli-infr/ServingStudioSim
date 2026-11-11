"""Operation-level accounting tests for the Phase 3A vLLM Llama TP1 mapping (CPU-only).

    uv run pytest profiling/env/arm64-gb300/serving-validation/test_vllm_llama_mapping.py

The traced kernel counts are those of the 183 clean 8-request decode iterations
of the GB300 TP1 smoke trace (`tp1-audit/tp1_audit_summary.json`).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import vllm_llama_mapping as m  # noqa: E402

TRACE = json.loads((HERE / "tp1-audit" / "tp1_audit_summary.json").read_text())
TRACE_COUNTS = {
    role: int(n)
    for role, n in TRACE["kernel_counts"]["per_role_per_iteration"].items()
    if n and role not in ("unknown",)
}


def test_every_traced_kernel_is_charged_exactly_once():
    charged = m.charged_trace_kernels()
    for role, n in TRACE_COUNTS.items():
        if role in m.OUTSIDE_ARCHITECTURE:
            assert role not in charged
            continue
        assert charged.get(role) == n, (role, charged.get(role), n)
    assert set(charged) <= set(TRACE_COUNTS)


def test_every_original_leaf_is_replaced_exactly_as_often_as_it_was_charged():
    replaced = m.charged_original_leaves()
    for name, per_iteration in m.ORIGINAL_LEAVES.items():
        if name == "embedding":
            # Folded into the first norm: replaced once, by one fused leaf.
            assert replaced[name] == 1
            continue
        assert replaced[name] == per_iteration, (name, replaced[name], per_iteration)


def test_residual_add_and_rms_norm_are_not_double_counted():
    norm_leaves = [leaf for leaf in m.VARIANT_LEAVES if leaf.kind == "multi_add_rms_norm"]
    # One norm leaf per traced norm kernel: 1 first + 32 post + 31 next-input + 1 final.
    assert (
        sum(leaf.per_iteration for leaf in norm_leaves)
        == (
            TRACE_COUNTS["embed_input_norm"]
            + TRACE_COUNTS["post_norm"]
            + TRACE_COUNTS["input_norm"]
            + TRACE_COUNTS["final_norm"]
        )
        == 65
    )
    # The adds live only inside the norm leaves: no other leaf prices a residual add,
    # and llama3_dense had no residual-add leaf to remove.
    assert all("residual" not in leaf.name for leaf in m.VARIANT_LEAVES)
    assert "residual_add" not in m.ORIGINAL_LEAVES
    # A sum is stored only where the trace stores it.
    stored = {leaf.name: leaf.config["store_sum"] for leaf in norm_leaves}
    assert stored == {
        "first_norm": True,
        "post_norm": False,
        "next_input_norm": True,
        "final_norm": False,
    }


def test_embedding_and_first_norm_are_one_leaf():
    first = [leaf for leaf in m.VARIANT_LEAVES if "embedding" in leaf.replaces]
    assert len(first) == 1 and first[0].per_iteration == 1
    assert set(first[0].replaces) == {"embedding", "input_norm"}
    # Layer 0's input norm is inside it, so the per-layer input norm covers 31 layers.
    nxt = next(leaf for leaf in m.VARIANT_LEAVES if leaf.name == "next_input_norm")
    assert nxt.per_iteration == m.NUM_LAYERS - 1
    assert m.charged_original_leaves()["input_norm"] == m.NUM_LAYERS


def test_activation_is_charged_once():
    act = [leaf for leaf in m.VARIANT_LEAVES if "activation" in leaf.trace_roles]
    assert len(act) == 1 and act[0].per_iteration == TRACE_COUNTS["activation"] == 32


def test_attention_fill_is_inside_the_attention_leaf():
    holders = [leaf for leaf in m.VARIANT_LEAVES if "attention_aux_fill" in leaf.trace_roles]
    assert [leaf.name for leaf in holders] == ["attention_decode"]
    assert holders[0].kernels_per_call == {"attention_aux_fill": 1, "attention": 1}


def test_split_k_reduce_is_inside_its_gemm_leaf():
    # o_proj and down_proj run two kernels per call (GEMM + split-K reduce); each is
    # one leaf priced by a row whose CUPTI union covers both.
    for name in ("o_proj", "down_proj"):
        leaf = next(leaf for leaf in m.VARIANT_LEAVES if leaf.name == name)
        assert leaf.kernels_per_call == {name: 2}
        assert TRACE_COUNTS[name] == 2 * leaf.per_iteration
    reduce_holders = [
        leaf for leaf in m.VARIANT_LEAVES if leaf.kernels_per_call.get(leaf.name) == 2
    ]
    assert {leaf.name for leaf in reduce_holders} == {"o_proj", "down_proj"}


def test_gemm_shapes_are_the_simulators():
    shapes = {
        leaf.name: (leaf.config["n"], leaf.config["k"])
        for leaf in m.VARIANT_LEAVES
        if leaf.kind == "single_gemm"
    }
    assert shapes == {
        "qkv_proj": (6144, 4096),
        "o_proj": (4096, 4096),
        "up_gate_proj": (28672, 4096),
        "down_proj": (4096, 14336),
        "lm_head": (128256, 4096),
    }


# ── registry: kinds, backends, fields ─────────────────────────────────────────


def _registry():
    pytest.importorskip("profiling.db.registry")
    from profiling.db import registry

    return registry


@pytest.mark.parametrize("leaf", m.VARIANT_LEAVES, ids=lambda leaf: leaf.name)
def test_every_kind_and_backend_is_registered_with_the_needed_fields(leaf):
    registry = _registry()
    assert leaf.backend in registry.known_backends(leaf.kind)
    schema = registry.find_args_schema(leaf.kind, leaf.backend)
    fields = {f.name for f in dataclasses.fields(schema)}
    assert set(leaf.config) <= fields, set(leaf.config) - fields


def test_no_leaf_is_blocked_or_unimplemented():
    assert {leaf.semantics for leaf in m.VARIANT_LEAVES} <= {"equivalent", "approximate"}


def test_paged_attention_backend_takes_bf16_and_keeps_fp8():
    registry = _registry()
    from profiling.db.args import DType

    for kind in ("paged_attn_batch_decode", "paged_attn_batch_prefill"):
        for dtype in (DType.BF16, DType.FP8_E4M3):
            assert registry.backend_supports(
                kind, "flashinfer_trtllm_gen", dtype, dtype, gpu="NVIDIA GB300"
            )
