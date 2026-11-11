#!/usr/bin/env python
"""CPU-only: vLLM's Llama 3.1 rotary embedding against the independent reference.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<worktree>:<vllm-fork> \\
        <serving venv>/bin/python rope_vllm_cpu_check.py --out rope_cpu_check.json

Builds the RotaryEmbedding the gqa_rope_neox runner builds (same get_rope call)
and checks, with no GPU and no weights:

1. frequencies: vLLM's float32 cos/sin cache against `llama3_cos_sin_cache`
   (float64). vLLM forms `position * inv_freq` in float32, so the angle carries
   up to about one float32 rounding of `position`; each row must stay within
   `position * 2**-23 + 1e-6`.
2. bf16 cache: exactly the float32 cache rounded to bf16.
3. semantics: `forward_native` on the runner's QKV split, in float32, equals
   `gqa_rope_neox_reference` (same cache) to 1e-6.
4. the runner's oracle: in bf16, `forward_native` passes the runner's
   `assert_close(rtol, atol)` against the reference (bf16 arithmetic rounds the
   intermediate products; float32 reference rounds once).
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from profiling.runners.attention import gqa_rope_neox_vllm_inductor as runner
from profiling.runners.attention.gqa_rope_neox_reference import (
    gqa_rope_neox_reference,
    llama3_cos_sin_cache,
)

HQ, HKV, D, MAX_POSITION = 32, 8, 128, 131072


def _rope(dtype: torch.dtype):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope

    with set_current_vllm_config(VllmConfig()):
        return get_rope(
            D,
            max_position=MAX_POSITION,
            is_neox_style=True,
            rope_parameters={**runner.LLAMA3_ROPE_PARAMETERS, "partial_rotary_factor": 1.0},
            dtype=dtype,
        )


def _apply(rope, dtype: torch.dtype):
    g = torch.Generator().manual_seed(0)
    tokens = 64
    positions = torch.cat(
        [torch.arange(8), torch.randint(0, MAX_POSITION, (tokens - 8,), generator=g)]
    )
    qkv = torch.randn(tokens, (HQ + 2 * HKV) * D, generator=g).to(dtype)
    q, k, _ = qkv.split([HQ * D, HKV * D, HKV * D], dim=-1)
    got_q, got_k = rope.forward_native(positions, q.clone(), k.clone())
    want_q, want_k = gqa_rope_neox_reference(
        torch, positions, qkv, rope.cos_sin_cache, HQ, HKV, D, D
    )
    return got_q.reshape(want_q.shape), got_k.reshape(want_k.shape), want_q, want_k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    report: dict = {"torch": torch.__version__}
    problems: list[str] = []

    rope32, rope16 = _rope(torch.float32), _rope(torch.bfloat16)
    report["class"] = type(rope32).__name__
    report["rotary_dim"] = rope32.rotary_dim
    report["cache_shape"] = list(rope32.cos_sin_cache.shape)

    ours = llama3_cos_sin_cache(torch, D, MAX_POSITION).double()
    theirs = rope32.cos_sin_cache.double()
    bound = torch.arange(MAX_POSITION, dtype=torch.float64) * 2.0**-23 + 1e-6
    ratio = ((theirs - ours).abs().amax(dim=1) / bound).max().item()
    report["frequencies"] = {
        "max_abs_diff": float((theirs - ours).abs().max()),
        "max_abs_diff_positions_0_15": float((theirs[:16] - ours[:16]).abs().max()),
        "max_diff_over_float32_angle_bound": ratio,
    }
    if ratio > 1.0:
        problems.append(f"cos/sin cache exceeds the float32 angle bound (ratio {ratio:.3g})")

    bf16_exact = torch.equal(rope16.cos_sin_cache, rope32.cos_sin_cache.to(torch.bfloat16))
    report["bf16_cache_is_rounded_float32_cache"] = bf16_exact
    if not bf16_exact:
        problems.append("bf16 cache is not the float32 cache rounded to bf16")

    got_q, got_k, want_q, want_k = _apply(rope32, torch.float32)
    semantics = max(float((got_q - want_q).abs().max()), float((got_k - want_k).abs().max()))
    report["float32_forward_native_vs_reference_max_abs"] = semantics
    if semantics > 1e-6:
        problems.append(f"float32 forward_native differs from the reference by {semantics:.3g}")

    got_q, got_k, want_q, want_k = _apply(rope16, torch.bfloat16)
    entry = {"rtol": runner._RTOL, "atol": runner._ATOL, "dtype": str(got_q.dtype)}
    for name, got, want in (("q", got_q, want_q), ("k", got_k, want_k)):
        diff = (got.float() - want.float()).abs()
        entry[f"{name}_max_abs"] = float(diff.max())
        entry[f"{name}_mismatched"] = int((got != want).sum())
        try:
            torch.testing.assert_close(got, want, rtol=runner._RTOL, atol=runner._ATOL)
        except AssertionError as exc:
            problems.append(f"bf16 {name} fails the runner oracle: {str(exc).splitlines()[0]}")
    report["bf16_runner_oracle"] = entry

    report["problems"] = problems
    report["verdict"] = "PASS" if not problems else "FAIL"
    text = json.dumps(report, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
