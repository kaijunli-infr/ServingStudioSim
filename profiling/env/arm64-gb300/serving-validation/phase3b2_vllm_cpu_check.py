#!/usr/bin/env python
"""CPU-only (Phase 3B-2): the new and changed blocks over vLLM's real native ops.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<worktree>:<vllm-fork> \\
        <serving venv>/bin/python phase3b2_vllm_cpu_check.py --out phase3b2_vllm_cpu_check.json

No GPU, no weights, no compile (Inductor's CPU backend emits C++, not the
Triton kernel served on GB300, so compiling here would say nothing about the
launch config). Eager, in the serving environment, it checks the wiring the
GPU runs will compile:

1. embedding_rms_norm: ``build_block`` over vLLM's native ``rms_norm`` IR op and
   ``F.embedding`` (VocabParallelEmbedding's unquantized path) with int32 ids,
   eps 1e-5: the residual equals the gathered table rows bit for bit and the
   normalized rows pass the runner's oracle against the independent reference.
2. multi_add_rms_norm: the runner's own ``_build_block`` (vLLM's native
   ``fused_add_rms_norm`` / ``rms_norm`` and QuantFP8) for the post-attention
   (2, normed), next-input (3, sum + normed) and final (3, normed) configs at
   eps 1e-5 returns exactly its stored planes, each matching the reference.
3. vLLM's Llama source still stores the embedding as layer 0's residual and
   normalizes it (the operation embedding_rms_norm keys).
"""

from __future__ import annotations

import argparse
import inspect
import json

import torch

EPS = 1e-5


def check_embedding() -> dict:
    from vllm import ir

    from profiling.runners.norm import embedding_rms_norm_vllm_inductor as runner
    from profiling.runners.norm.embedding_rms_norm_reference import embedding_rms_norm_reference

    g = torch.Generator().manual_seed(0)
    vocab, hidden, tokens = 1024, 4096, 37
    table = (0.02 * torch.randn(vocab, hidden, generator=g)).to(torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(hidden, generator=g)).to(torch.bfloat16)
    ids = torch.randint(0, vocab, (tokens,), generator=g, dtype=torch.int32)
    block = runner.build_block(
        ir.ops.rms_norm.impls["native"].impl_fn, torch.nn.functional.embedding, EPS
    )
    residual, normed = block(ids, table, weight)
    runner.check_outputs(torch, ids, table, weight, EPS, residual, normed)
    _, want = embedding_rms_norm_reference(torch, ids, table, weight, EPS)
    return {
        "residual_bitwise": bool(torch.equal(residual, table[ids.long()])),
        "normed_max_abs": float((normed.float() - want.float()).abs().max()),
        "oracle": "pass",
    }


def check_norms() -> dict:
    from profiling.runners.norm import multi_add_rms_norm_vllm_inductor as runner
    from profiling.runners.norm.multi_add_rms_norm_reference import multi_add_rms_norm_reference

    out = {}
    g = torch.Generator().manual_seed(1)
    weight = (1 + 0.1 * torch.randn(4096, generator=g)).to(torch.bfloat16)
    scale = torch.tensor(1 / 16)
    for name, (addends, store_sum) in {
        "post_attention": (2, False),
        "next_input": (3, True),
        "final": (3, False),
    }.items():
        xs = [torch.randn(19, 4096, generator=g).to(torch.bfloat16) for _ in range(addends)]
        block = runner._build_block(store_sum, True, False, addends, EPS)
        planes = block(*xs, weight, scale)
        total, normed, _ = multi_add_rms_norm_reference(torch, xs, weight, scale, EPS)
        want = ([total] if store_sum else []) + [normed]
        assert len(planes) == len(want), (name, len(planes))
        errors = []
        for got, ref in zip(planes, want, strict=True):
            torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)
            errors.append(float((got.float() - ref.float()).abs().max()))
        out[name] = {"planes": len(planes), "max_abs": errors, "oracle": "pass"}
    return out


def check_llama_source() -> dict:
    from vllm.model_executor.models import llama

    source = inspect.getsource(llama.LlamaDecoderLayer.forward)
    stores = "residual = hidden_states" in source
    stores = stores and "self.input_layernorm(hidden_states)" in source
    model = inspect.getsource(llama.LlamaModel.forward)
    first = "residual = None" in model and "embed_input_ids" in model
    if not (stores and first):
        raise AssertionError("vLLM's Llama no longer stores the embedding as layer 0's residual")
    return {"layer0_residual_is_embedding": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    args = parser.parse_args()
    import vllm

    report = {
        "torch": torch.__version__,
        "vllm": getattr(vllm, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "embedding_rms_norm": check_embedding(),
        "multi_add_rms_norm": check_norms(),
        "llama_source": check_llama_source(),
    }
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
