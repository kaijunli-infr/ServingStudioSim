"""Interval accounting primitives (component 6, stream-level composition).

All functions take integer (or float) start/end arrays and return exact union / overlap / idle
quantities.  `union_length` is the measured "busy" time of a set of kernels on one device;
`sum - union` is the multi-stream overlap; `span - union` is the idle time inside the span.
"""
from __future__ import annotations

import numpy as np


def union_length(start: np.ndarray, end: np.ndarray) -> int:
    """Length of the union of the intervals [start, end)."""
    if len(start) == 0:
        return 0
    o = np.argsort(start, kind="stable")
    s = np.asarray(start)[o]
    e = np.asarray(end)[o]
    cm = np.maximum.accumulate(e)
    prev = np.empty_like(cm)
    prev[0] = s[0]
    prev[1:] = cm[:-1]
    contrib = e - np.maximum(s, prev)
    return int(np.clip(contrib, 0, None).sum())


def windowed_union(start: np.ndarray, end: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Union length of the intervals clipped to each window [bounds[k], bounds[k+1]).

    Every interval is assigned to the window that contains its start; the part of the interval
    beyond the window end is clipped (reported separately by the caller as cross-window overlap).
    Returns an array of len(bounds) - 1 union lengths.  Vectorised (no per-window Python loop).
    """
    nwin = len(bounds) - 1
    out = np.zeros(nwin, dtype=np.int64)
    if nwin <= 0 or len(start) == 0:
        return out
    s = np.asarray(start, dtype=np.int64)
    e = np.asarray(end, dtype=np.int64)
    w = np.searchsorted(bounds, s, side="right") - 1
    keep = (w >= 0) & (w < nwin)
    s, e, w = s[keep], e[keep], w[keep]
    if len(s) == 0:
        return out
    cs = np.maximum(s, bounds[w])
    ce = np.minimum(e, bounds[w + 1])
    ce = np.maximum(ce, cs)
    o = np.lexsort((cs, w))
    cs, ce, w = cs[o], ce[o], w[o]
    cm = np.maximum.accumulate(ce)
    prev = np.empty_like(cm)
    prev[0] = cs[0]
    prev[1:] = cm[:-1]
    # windows are disjoint and ordered, so a clipped end from an earlier window never exceeds a
    # later window's start: the running max resets implicitly at window boundaries.
    contrib = np.clip(ce - np.maximum(cs, prev), 0, None)
    np.add.at(out, w, contrib)
    return out


def accounting(start: np.ndarray, end: np.ndarray, is_comm: np.ndarray) -> dict[str, int]:
    """Iteration-level accounting on one device.

    Returns span, busy (union of all kernels), compute_union, comm_union, compute_sum, comm_sum,
    idle = span - busy, overlap_total = sum - busy, overlap_comp_comm = compute_union + comm_union - busy
    (time when a compute kernel and a communication kernel run concurrently),
    overlap_comp_comp = compute_sum - compute_union.
    """
    start = np.asarray(start, dtype=np.int64)
    end = np.asarray(end, dtype=np.int64)
    if len(start) == 0:
        return dict(span=0, busy=0, compute_union=0, comm_union=0, compute_sum=0, comm_sum=0, idle=0,
                    overlap_total=0, overlap_comp_comm=0, overlap_comp_comp=0)
    busy = union_length(start, end)
    cu = union_length(start[~is_comm], end[~is_comm])
    mu = union_length(start[is_comm], end[is_comm])
    csum = int((end[~is_comm] - start[~is_comm]).sum())
    msum = int((end[is_comm] - start[is_comm]).sum())
    span = int(end.max() - start.min())
    return dict(span=span, busy=busy, compute_union=cu, comm_union=mu, compute_sum=csum, comm_sum=msum,
                idle=span - busy, overlap_total=csum + msum - busy, overlap_comp_comm=cu + mu - busy,
                overlap_comp_comp=csum - cu)
