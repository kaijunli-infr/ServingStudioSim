"""The ``nccl_graph`` all-reduce backend: CUDA-graph replay timing, kept apart
from the eager ``nccl`` rows.

CPU tier: registry introspection, the runner's group aggregation behind a fake
launcher, and row separation in a temporary profile database. No GPU, no rank
group, no shared ``profile.db``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from profiling.db.args import DType
from profiling.db.registry import (
    backend_supports,
    find_kernel_profiler_spec,
    known_backends,
    supported_backends,
)
from profiling.db.table import MissingEntry, ProfileRow, Table
from profiling.kernels.all_reduce import KIND, AllReduceArgs
from profiling.runners.comm import nccl_graph
from profiling.runners.metrics import CommMetrics

REPO_ROOT = Path(__file__).resolve().parents[1]


def _args(num_gpus: int = 2, size: int = 8192) -> AllReduceArgs:
    return AllReduceArgs(
        num_gpus=num_gpus, message_size_bytes=size, dtype=DType.BF16, fabric="nvlink"
    )


def _spec(num_gpus: int = 2, size: int = 8192) -> dict:
    return {"num_gpus": num_gpus, "message_size_bytes": size, "dtype": "bf16", "fabric": "nvlink"}


# ── registration ─────────────────────────────────────────────────────────────


def test_nccl_graph_is_registered_beside_the_eager_backends():
    assert set(known_backends(KIND)) == {"nccl", "nccl_graph", "nvshmem"}
    spec = find_kernel_profiler_spec(KIND, "nccl_graph")
    assert spec.table_name == KIND  # same table, distinguished by the backend column
    assert spec.runner_ref.module_name == "profiling.runners.comm.nccl_graph"
    assert spec.runner_ref.function_name == "profile_all_reduce_batch"
    assert spec.list_native is True
    assert spec.gpu_count_fn({"num_gpus": 4}) == 4
    # The eager backend still points at the eager runner.
    eager = find_kernel_profiler_spec(KIND, "nccl")
    assert eager.runner_ref.module_name == "profiling.runners.comm.nccl"


def test_nccl_graph_is_a_legal_option_on_every_gpu_and_dtype():
    for gpu in (None, "NVIDIA H200", "NVIDIA B200", "NVIDIA GB300"):
        assert backend_supports(KIND, "nccl_graph", DType.BF16, gpu=gpu)
        assert "nccl_graph" in supported_backends(KIND, DType.BF16, gpu=gpu)


# ── selection is explicit ────────────────────────────────────────────────────


def test_no_arch_or_public_preset_selects_nccl_graph_by_default():
    # Graph-timed rows exist for GB300 only. An arch default or a public preset
    # naming nccl_graph would change H200/B200 runs or fail them for lack of rows.
    for path in (REPO_ROOT / "simulator" / "src").rglob("*.rs"):
        assert "nccl_graph" not in path.read_text(), path
    for path in (REPO_ROOT / "presets").iterdir():
        if path.is_file():
            assert "nccl_graph" not in path.read_text(), path


def test_llama3_dense_tp_default_all_reduce_backend_is_eager_nccl():
    source = (REPO_ROOT / "simulator/src/arch/llama3_dense_tp.rs").read_text()
    assert re.search(r'const ALLREDUCE_BACKENDS: &\[&str\] = &\["nccl"\];', source)


def test_gb300_graph_presets_select_the_backend_explicitly_for_both_all_reduces():
    for tp in (2, 4):
        preset = json.loads(
            (
                REPO_ROOT
                / f"profiling/env/arm64-gb300/presets/predict_llama3_8b_gb300_tp{tp}_graph.json"
            ).read_text()
        )
        assert preset["gpu"] == "NVIDIA GB300"
        assert preset["arch"]["iter"]["tp_size"] == tp
        assert preset["backends"]["main"] == {
            "unified.attn_block.tp_allreduce": ["nccl_graph"],
            "unified.mlp_block.tp_allreduce": ["nccl_graph"],
        }


# ── rows: separate keys, no fallback ─────────────────────────────────────────


def _row(backend: str, time_ms: float, gpu: str = "NVIDIA GB300") -> ProfileRow:
    return ProfileRow(
        args=_args(),
        metrics=CommMetrics(time_ms=time_ms, algbw_gbps=1.0, busbw_gbps=1.0),
        gpu_name=gpu,
        backend=backend,
        profiler_git_hash="0" * 40,
        cuda_version="13.0",
        driver_version="test",
        backend_version="test",
    )


def test_missing_graph_row_is_missing_even_when_the_eager_row_exists(tmp_path):
    db = tmp_path / "profile.db"
    Table(find_kernel_profiler_spec(KIND, "nccl"), db).insert([_row("nccl", 0.05)])
    graph = Table(find_kernel_profiler_spec(KIND, "nccl_graph"), db)

    [entry] = graph.query([_args()], gpu_name="NVIDIA GB300")
    assert isinstance(entry, MissingEntry)
    assert entry.backend == "nccl_graph"


def test_graph_and_eager_rows_coexist_under_one_key_each(tmp_path):
    db = tmp_path / "profile.db"
    eager = Table(find_kernel_profiler_spec(KIND, "nccl"), db)
    graph = Table(find_kernel_profiler_spec(KIND, "nccl_graph"), db)
    eager.insert([_row("nccl", 0.05)])
    graph.insert([_row("nccl_graph", 0.014)])

    [eager_metrics] = eager.query([_args()], gpu_name="NVIDIA GB300")
    [graph_metrics] = graph.query([_args()], gpu_name="NVIDIA GB300")
    assert eager_metrics.time_ms == pytest.approx(0.05)  # not overwritten
    assert graph_metrics.time_ms == pytest.approx(0.014)
    # Another GPU has neither.
    [other] = graph.query([_args()], gpu_name="NVIDIA H200")
    assert isinstance(other, MissingEntry)


# ── runner: group aggregation behind a fake launcher ─────────────────────────


class _FakeLauncher:
    """Stands in for TorchMpLauncher: returns rank 0's batch times per spec."""

    payloads: list = []
    calls: list = []

    def __init__(self, num_gpus: int, *, backend: str):
        type(self).calls.append((num_gpus, backend))

    def run(self, per_rank_fn, **kwargs):
        assert per_rank_fn is nccl_graph._graph_replay_per_rank_batch
        assert kwargs["replays"] == nccl_graph.REPLAYS
        assert kwargs["batches"] == nccl_graph.BATCHES
        payload = type(self).payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload


@pytest.fixture
def fake_launcher(monkeypatch):
    _FakeLauncher.payloads = []
    _FakeLauncher.calls = []
    monkeypatch.setattr(nccl_graph, "TorchMpLauncher", _FakeLauncher)
    return _FakeLauncher


def test_runner_takes_the_median_of_five_group_medians(fake_launcher):
    # One spec; group medians 0.010, 0.011, 0.012, 0.013, 0.050 -> 0.012.
    fake_launcher.payloads = [
        [[0.010, 0.010, 0.900]],  # a spike inside a group does not move its median
        [[0.011, 0.011, 0.011]],
        [[0.012, 0.012, 0.012]],
        [[0.013, 0.013, 0.013]],
        [[0.050, 0.050, 0.050]],  # a slow group does not move the row
    ]
    [result] = nccl_graph.profile_all_reduce_batch([_spec(num_gpus=4, size=1 << 20)])

    assert result.error is None
    assert result.metrics.time_ms == pytest.approx(0.012)
    algbw = (1 << 20) / 0.012e-3 / 1e9
    assert result.metrics.algbw_gbps == pytest.approx(algbw)
    assert result.metrics.busbw_gbps == pytest.approx(algbw * 2 * 3 / 4)
    # Five fresh groups, all through the NCCL process-group backend.
    assert fake_launcher.calls == [(4, "nccl")] * nccl_graph.GROUPS


def test_runner_fails_the_whole_chunk_when_a_group_fails(fake_launcher):
    fake_launcher.payloads = [
        [[0.01] * 3, [0.02] * 3],
        RuntimeError("all-reduce SUM mismatch at 8192 bytes"),
    ]
    results = nccl_graph.profile_all_reduce_batch([_spec(), _spec(size=16384)])

    assert [r.metrics for r in results] == [None, None]
    assert all("SUM mismatch" in r.error for r in results)


def test_runner_rejects_a_short_group_payload_and_single_gpu(fake_launcher):
    fake_launcher.payloads = [[[0.01] * 3]]
    results = nccl_graph.profile_all_reduce_batch([_spec(), _spec(size=16384)])
    assert all("returned 1 payloads for 2 specs" in r.error for r in results)

    [single] = nccl_graph.profile_all_reduce_batch([_spec(num_gpus=1)])
    assert "at least 2 GPUs" in single.error
    assert nccl_graph.profile_all_reduce_batch([]) == []


def test_method_constants_match_the_validated_harness():
    # The harness that produced the stored rows and the runner must agree.
    harness = (REPO_ROOT / "profiling/env/arm64-gb300/allreduce_graph.py").read_text()
    assert "OPERATIONS_PER_GRAPH = 10" in harness
    assert "BATCHES = 3" in harness
    assert nccl_graph.OPERATIONS_PER_GRAPH == 10
    assert (nccl_graph.GROUPS, nccl_graph.BATCHES, nccl_graph.REPLAYS) == (5, 3, 100)
    assert nccl_graph.MEASUREMENT_METHOD == "nccl_torch_dist.cuda_graph_replay.ops10"
    assert 'METHOD_GRAPH = "nccl_torch_dist.cuda_graph_replay.ops{ops}"' in harness
