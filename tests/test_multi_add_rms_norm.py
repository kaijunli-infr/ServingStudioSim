"""Behavioral tests for the summed RMSNorm with a selectable output set."""

import pytest
import torch

from profiling.runners.norm.multi_add_rms_norm_reference import multi_add_rms_norm_reference
from profiling.runners.norm.multi_add_rms_norm_vllm_inductor import _validate_args


def test_reference_normalizes_the_sum_of_every_addend() -> None:
    # A reference that dropped the recomputed third addend would normalize a
    # different row and accept a kernel that skipped one input read.
    addends = [torch.full((1, 4), value, dtype=torch.bfloat16) for value in (1.0, 2.0, -6.0)]
    weight = torch.full((4,), 2.0, dtype=torch.bfloat16)

    total, normalized, fp8 = multi_add_rms_norm_reference(
        torch, addends, weight, torch.tensor(0.5), 0.0
    )

    torch.testing.assert_close(total.float(), torch.full((1, 4), -3.0))
    torch.testing.assert_close(normalized.float(), torch.full((1, 4), -2.0))
    torch.testing.assert_close(fp8.float(), torch.full((1, 4), -4.0))


@pytest.mark.parametrize(
    ("num_addends", "store_sum", "store_normed", "store_fp8"),
    [(1, True, True, False), (2, True, False, False)],
)
def test_runner_rejects_output_sets_vllm_never_compiles(
    num_addends: int, store_sum: bool, store_normed: bool, store_fp8: bool
) -> None:
    # A one-addend "sum" is the input itself, and a norm with no output plane is
    # dead code Inductor deletes; profiling either would store a fake row.
    with pytest.raises(ValueError):
        _validate_args(8, 5120, num_addends, store_sum, store_normed, store_fp8, "bf16")


def _torch_rms_norm(x, weight, eps):
    """vLLM's native rms_norm contract, written independently in float32."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def _torch_fused_add_rms_norm(x, residual, weight, eps):
    """vLLM's native fused_add_rms_norm contract: (normed, x + residual)."""
    total = (x.float() + residual.float()).to(x.dtype)
    return _torch_rms_norm(total, weight, eps), total


# (num_addends, store_sum, store_normed) of the served Llama norms.
_SERVED_NORMS = {
    "post_attention": (2, False, True),
    "next_input": (3, True, True),
    "final": (3, False, True),
}


@pytest.mark.parametrize("which", sorted(_SERVED_NORMS))
def test_served_norm_configs_store_exactly_their_planes(which: str) -> None:
    # Defect caught: an output set wired to the wrong plane. post stores only
    # the normed rows (the next norm re-adds its sum), next-input stores the
    # new residual and the normed rows, final only the normed rows; each must
    # equal the independent reference with Llama 3.1's eps.
    from profiling.runners.norm.multi_add_rms_norm_vllm_inductor import block_from_ops

    num_addends, store_sum, store_normed = _SERVED_NORMS[which]
    g = torch.Generator().manual_seed(1)
    addends = [torch.randn(5, 64, generator=g).to(torch.bfloat16) for _ in range(num_addends)]
    weight = (1 + 0.1 * torch.randn(64, generator=g)).to(torch.bfloat16)
    scale = torch.tensor(1 / 16)
    block = block_from_ops(
        _torch_fused_add_rms_norm,
        _torch_rms_norm,
        lambda normed, s: normed,
        store_sum=store_sum,
        store_normed=store_normed,
        store_fp8=False,
        num_addends=num_addends,
        eps=1e-5,
    )
    outputs = block(*addends, weight, scale)
    assert len(outputs) == store_sum + store_normed
    total, normed, _ = multi_add_rms_norm_reference(torch, addends, weight, scale, 1e-5)
    expected = ([total] if store_sum else []) + ([normed] if store_normed else [])
    for got, want in zip(outputs, expected, strict=True):
        torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)
    # The eps is the block's argument, not a constant baked into the runner.
    if store_normed:
        coarse = block_from_ops(
            _torch_fused_add_rms_norm,
            _torch_rms_norm,
            lambda normed, s: normed,
            store_sum=store_sum,
            store_normed=True,
            store_fp8=False,
            num_addends=num_addends,
            eps=10.0,
        )(*addends, weight, scale)[-1]
        assert not torch.equal(coarse, outputs[-1])


def test_tuning_size_and_eps_are_keyed_and_legacy_values_keep_old_keys() -> None:
    # Defect caught: rows tuned at 2048 sharing the key of rows tuned at 8192
    # (different launch config on GB300), or spelling out 8192 / 1e-6 moving
    # every existing row's key.
    from profiling.db.batch import coerce_args
    from profiling.db.registry import find_kernel_profiler_spec
    from profiling.db.table import Table

    spec = find_kernel_profiler_spec("multi_add_rms_norm", "vllm_inductor")
    table = Table(spec, "unused.db")
    base = dict(
        num_tokens=8,
        hidden_dim=4096,
        num_addends=3,
        store_sum=True,
        store_normed=True,
        store_fp8=False,
        dtype="bf16",
    )

    def key(**extra):
        return table.args_hash(table.db_key(coerce_args(spec.args_schema, {**base, **extra})))

    legacy = key()
    assert key(tune_tokens=8192, rms_eps=1e-6) == legacy
    served = key(tune_tokens=2048, rms_eps=1e-5)
    assert len({legacy, served, key(tune_tokens=2048), key(rms_eps=1e-5)}) == 4
    args = coerce_args(spec.args_schema, {**base, "tune_tokens": 8192, "rms_eps": 1e-6})
    assert (args.tune_tokens, args.rms_eps) == (None, None)


@pytest.mark.parametrize(("tune_tokens", "rms_eps"), [(0, 1e-5), (2048, 0.0), (2048, 1)])
def test_runner_rejects_unusable_tuning_and_eps(tune_tokens: int, rms_eps: float) -> None:
    with pytest.raises(ValueError):
        _validate_args(8, 4096, 2, False, True, False, "bf16", tune_tokens, rms_eps)
