#!/usr/bin/env python
"""CPU-only: vLLM's Llama 3.1 rotary embedding against the independent reference.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<worktree>:<vllm-fork> \\
        <serving venv>/bin/python rope_vllm_cpu_check.py --out rope_cpu_check.json

Builds the RotaryEmbedding the gqa_rope_neox runner builds (same get_rope call),
runs its `forward_native` on the QKV split the runner's block makes, in float32
and in bf16, and compares (1) vLLM's cos/sin cache with `llama3_cos_sin_cache`
and (2) the rotated Q and K with `gqa_rope_neox_reference`. No GPU, no weights.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from profiling.runners.attention.gqa_rope_neox_reference import (
    gqa_rope_neox_reference,
    llama3_cos_sin_cache,
)
from profiling.runners.attention.gqa_rope_neox_vllm_inductor import LLAMA3_ROPE_PARAMETERS

HQ, HKV, D, MAX_POSITION = 32, 8, 128, 131072


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from vllm.model_executor.layers.rotary_embedding import get_rope

    report: dict = {"torch": torch.__version__}
    problems: list[str] = []
    for dtype, tol in ((torch.float32, 1e-4), (torch.bfloat16, 2e-2)):
        rope = get_rope(
            D,
            max_position=MAX_POSITION,
            is_neox_style=True,
            rope_parameters={**LLAMA3_ROPE_PARAMETERS, "partial_rotary_factor": 1.0},
            dtype=dtype,
        )
        name = str(dtype).removeprefix("torch.")
        entry = {"class": type(rope).__name__, "rotary_dim": rope.rotary_dim}
        ours = llama3_cos_sin_cache(torch, D, MAX_POSITION)
        theirs = rope.cos_sin_cache.float()
        entry["cache_shape"] = list(theirs.shape)
        # vLLM forms pos * inv_freq in float32, so the angle carries a float32
        # rounding error growing with the position; compare per position band.
        bands = {}
        for lo, hi in ((0, 8192), (8192, 32768), (32768, MAX_POSITION)):
            bands[f"{lo}-{hi}"] = float((theirs[lo:hi] - ours[lo:hi]).abs().max())
        entry["cache_max_abs_diff_by_position"] = bands
        g = torch.Generator().manual_seed(0)
        tokens = 64
        positions = torch.cat(
            [torch.arange(8), torch.randint(0, MAX_POSITION, (tokens - 8,), generator=g)]
        )
        qkv = torch.randn(tokens, (HQ + 2 * HKV) * D, generator=g).to(dtype)
        q, k, _ = qkv.split([HQ * D, HKV * D, HKV * D], dim=-1)
        got_q, got_k = rope.forward_native(positions, q.clone(), k.clone())
        rope_dtype = got_q.dtype
        want_q, want_k = gqa_rope_neox_reference(
            torch, positions, qkv, rope.cos_sin_cache, HQ, HKV, D, D
        )
        # Same reference over the independent cache, first 8192 positions only
        # (where the float32 angle error is below the bf16 tolerance).
        early = positions < 8192
        ind_q, _ = gqa_rope_neox_reference(torch, positions, qkv, ours, HQ, HKV, D, D)
        diffs = {
            "q_vs_reference": float(
                (got_q.float().reshape(want_q.shape) - want_q.float()).abs().max()
            ),
            "k_vs_reference": float(
                (got_k.float().reshape(want_k.shape) - want_k.float()).abs().max()
            ),
            "q_vs_independent_cache_pos_lt_8192": float(
                (got_q.float().reshape(want_q.shape)[early] - ind_q.float()[early]).abs().max()
            ),
        }
        entry["max_abs_diff"] = diffs
        entry["tolerance"] = tol
        entry["output_dtype"] = str(rope_dtype)
        for key, value in diffs.items():
            if value > tol:
                problems.append(f"{name} {key} {value:.3g} > {tol}")
        if bands["0-8192"] > 1e-4:
            problems.append(
                f"{name} cache differs from llama3 scaling below 8192: {bands['0-8192']}"
            )
        report[name] = entry
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
