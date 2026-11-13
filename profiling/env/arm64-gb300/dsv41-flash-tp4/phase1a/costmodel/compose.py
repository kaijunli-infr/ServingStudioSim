"""Component 6: per-rank stream and synchronization composition (TP4 critical path).

The iteration is a per-rank ordered timeline of compute windows separated by K collective calls.
A collective completes on every rank at `max_r(arrival_r) + intrinsic_k`, where `intrinsic_k` is
the cost of the call when no rank waits (measured as the minimum duration over ranks).  Iteration
completion is the latest rank end after the tail window.  Nothing is summed over ranks.

`critical_path` is the deterministic forward model; `critical_path_batch` evaluates it for many
sampled timelines at once (stochastic composition: per-window jitter makes the expected critical
path longer than the path through the median windows, because every synchronization point waits
for whichever rank happens to be late); `reconstruct` applies the rule to one measured iteration
(its own windows and intrinsic costs) and is the evidence that the synchronization rule holds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PathResult:
    wall: float                 # iteration completion (same unit as the inputs)
    rank_end: np.ndarray        # (R,) end of the tail window per rank
    rank_wait: np.ndarray       # (R,) total time a rank spent waiting in collectives for other ranks
    rank_compute: np.ndarray    # (R,) Σ compute windows (span, i.e. incl. intra-window idle)
    comm_intrinsic: float       # Σ_k intrinsic_k (identical on all ranks)
    coll_done: np.ndarray       # (K,) completion time of each collective
    slowest_rank: int           # rank whose arrival determined the most collectives
    critical_count: np.ndarray  # (R,) number of collectives each rank was last to arrive at


def critical_path(start: np.ndarray, seg: np.ndarray, intrinsic: np.ndarray) -> PathResult:
    """start: (R,) first-kernel offsets; seg: (R, K+1) compute windows; intrinsic: (K,)."""
    start = np.asarray(start, dtype=np.float64)
    seg = np.asarray(seg, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    R, K1 = seg.shape
    K = K1 - 1
    if len(intrinsic) != K:
        raise ValueError(f"segments imply {K} collectives, got {len(intrinsic)} intrinsic costs")
    t = start.copy()
    wait = np.zeros(R)
    crit = np.zeros(R, dtype=np.int64)
    done = np.zeros(K)
    for k in range(K):
        arrival = t + seg[:, k]
        last = int(np.argmax(arrival))
        crit[last] += 1
        done[k] = arrival[last] + intrinsic[k]
        wait += arrival[last] - arrival          # time each rank idles inside the collective
        t = np.full(R, done[k])
    end = t + seg[:, K]
    return PathResult(wall=float(end.max()), rank_end=end, rank_wait=wait, rank_compute=seg.sum(axis=1),
                      comm_intrinsic=float(intrinsic.sum()), coll_done=done, slowest_rank=int(np.argmax(crit)),
                      critical_count=crit)


def critical_path_batch(start: np.ndarray, seg: np.ndarray, intrinsic: np.ndarray) -> dict:
    """Vectorised over N sampled timelines: start (N, R), seg (N, R, K+1), intrinsic (N, K).

    Returns wall (N,), rank_end (N, R), rank_wait (N, R), critical_count (N, R).
    """
    start = np.asarray(start, dtype=np.float64)
    seg = np.asarray(seg, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    N, R, K1 = seg.shape
    K = K1 - 1
    t = start.copy()
    wait = np.zeros((N, R))
    crit = np.zeros((N, R), dtype=np.int64)
    rows = np.arange(N)
    for k in range(K):
        arrival = t + seg[:, :, k]
        last = np.argmax(arrival, axis=1)
        amax = arrival[rows, last]
        crit[rows, last] += 1
        wait += amax[:, None] - arrival
        t = np.repeat((amax + intrinsic[:, k])[:, None], R, axis=1)
    end = t + seg[:, :, K]
    return dict(wall=end.max(axis=1), rank_end=end, rank_wait=wait, critical_count=crit)


def reconstruct(start_rel: np.ndarray, seg_len: np.ndarray, intrinsic: np.ndarray) -> PathResult:
    """Apply the synchronization rule to one measured iteration (ns)."""
    return critical_path(start_rel, seg_len, intrinsic)


def sum_of_rank_durations(start: np.ndarray, seg: np.ndarray, intrinsic: np.ndarray) -> float:
    """The quantity that must NOT be used as iteration time (kept for tests and reports)."""
    seg = np.asarray(seg, dtype=np.float64)
    return float((seg.sum(axis=1) + np.asarray(intrinsic, dtype=np.float64).sum()).sum())
