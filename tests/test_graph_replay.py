"""CPU tests for the CUDA-graph replay timing mode's record parsing."""

from dataclasses import dataclass

import pytest

from profiling.profilers.graph_replay import (
    BATCHES,
    EAGER_WARMUP,
    MEASUREMENT_METHOD,
    REPLAYS,
    WARMUP_REPLAYS,
    GraphReplayError,
    per_replay_ms,
)


@dataclass(frozen=True)
class _Rec:
    name: str
    start_ns: int
    duration_ns: int


def _records(names, replays, duration_ns=1000, gap_ns=500):
    out, t = [], 0
    for _ in range(replays):
        for name in names:
            out.append(_Rec(name, t, duration_ns))
            t += duration_ns + gap_ns
    return out


def test_counts_are_fixed_and_named_in_the_method():
    # Provenance: the method string records the counts a row was timed with.
    assert (EAGER_WARMUP, WARMUP_REPLAYS, BATCHES, REPLAYS) == (3, 20, 3, 100)
    assert MEASUREMENT_METHOD.startswith("cupti.cuda_graph_replay.warm_l2")
    assert "3x100" in MEASUREMENT_METHOD and "warm20" in MEASUREMENT_METHOD


def test_replay_time_is_the_sum_of_its_kernels_not_the_gaps():
    times = per_replay_ms(_records(["a", "b"], 4), ["a", "b"], 4)
    assert times == [0.002] * 4


def test_out_of_order_records_are_sorted_by_start():
    records = list(reversed(_records(["a", "b"], 3)))
    assert per_replay_ms(records, ["a", "b"], 3) == [0.002] * 3


@pytest.mark.parametrize(
    "records",
    [
        _records(["a", "b"], 3)[:-1],  # a missing launch
        _records(["a", "b", "memcpy"], 3),  # an unrelated kernel in the graph
        _records(["b", "a"], 3),  # another kernel order
    ],
)
def test_anything_but_the_eager_sequence_fails(records):
    with pytest.raises(GraphReplayError):
        per_replay_ms(records, ["a", "b"], 3)
