"""CPU tests for the fused embedding gather + first RMSNorm kind."""

import dataclasses

import pytest
import torch

from profiling.db.registry import find_kernel_profiler_spec
from profiling.kernels.embedding_rms_norm import EmbeddingRmsNormArgs
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.norm import embedding_rms_norm_vllm_inductor as runner
from profiling.runners.norm.embedding_rms_norm_reference import embedding_rms_norm_reference


def _rms_norm(x, weight, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def _case(tokens=6, vocab=50, hidden=32, eps=1e-5):
    g = torch.Generator().manual_seed(3)
    table = (0.02 * torch.randn(vocab, hidden, generator=g)).to(torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(hidden, generator=g)).to(torch.bfloat16)
    ids = torch.randint(0, vocab, (tokens,), generator=g, dtype=torch.int32)
    return ids, table, weight, eps


def test_kind_registers_the_inductor_backend_with_every_key_field():
    # Defect caught: the kind missing from the barrel, or a key without the
    # table size, epsilon or tuning size the compiled kernel bakes in.
    spec = find_kernel_profiler_spec("embedding_rms_norm", "vllm_inductor")
    assert spec.subprocess_env == "vllm_env"
    assert spec.args_schema is EmbeddingRmsNormArgs
    assert [f.name for f in dataclasses.fields(EmbeddingRmsNormArgs)] == [
        "num_tokens",
        "hidden_dim",
        "vocab_size",
        "dtype",
        "rms_eps",
        "tune_tokens",
    ]


def test_block_gathers_by_id_and_returns_residual_and_normed():
    # The served operation: the embedding rows (the residual) and their
    # RMSNorm, from int32 ids, against the independent reference.
    ids, table, weight, eps = _case()
    block = runner.build_block(_rms_norm, torch.nn.functional.embedding, eps)
    residual, normed = block(ids, table, weight)
    want_residual, want_normed = embedding_rms_norm_reference(torch, ids, table, weight, eps)
    assert torch.equal(residual, want_residual)
    assert torch.equal(residual, table[ids.long()])
    torch.testing.assert_close(normed.float(), want_normed.float(), rtol=2e-2, atol=2e-2)
    runner.check_outputs(torch, ids, table, weight, eps, residual, normed)


def test_check_rejects_the_standalone_norm_substitute():
    # Defect caught: a contiguous 1-addend norm (no gather) passing as the
    # fused kernel, or a block that drops the residual store.
    ids, table, weight, eps = _case()
    rows = table[: ids.numel()]
    with pytest.raises(KernelLaunchFailed, match="residual"):
        runner.check_outputs(torch, ids, table, weight, eps, rows, _rms_norm(rows, weight, eps))
    residual = table[ids.long()]
    wrong_eps = _rms_norm(residual * 1e-3, weight, 0.5)
    with pytest.raises(AssertionError):
        runner.check_outputs(torch, ids, table, weight, eps, residual, wrong_eps)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"vocab_size": 0}, ValueError),
        ({"tune_tokens": 0}, ValueError),
        ({"rms_eps": 0.0}, ValueError),
        ({"vocab_size": 2**31}, ValueError),
        ({"dtype": "fp16"}, ProfilerNotImplemented),
    ],
)
def test_validation_rejects_unprofilable_args(overrides, error):
    args = dict(
        num_tokens=8,
        hidden_dim=4096,
        vocab_size=128256,
        dtype="bf16",
        rms_eps=1e-5,
        tune_tokens=2048,
    )
    with pytest.raises(error):
        runner.validate_args(**{**args, **overrides})


def test_bytes_count_the_id_one_row_read_and_two_row_writes():
    assert runner.bytes_moved(8, 4096) == 8 * (4 + 3 * 4096 * 2) + 4096 * 2
