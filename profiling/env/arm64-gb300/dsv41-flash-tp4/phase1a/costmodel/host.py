"""Component 7: host scheduling and synchronization.

What the frozen traces identify:
* Inter-iteration GPU gap (first kernel of iteration i+1 − last kernel of iteration i): ≈ 0 in every
  class → the GPU timeline is contiguous, the host runs ahead and scheduling never shows up as a gap
  between iterations.  Full-workload GPU time is therefore Σ iteration walls (component 6 output).
* Decode run-ahead: host NVTX window (≈4 ms) ends before the GPU step completes (gpu_tail 3–14 ms);
  no *Synchronize call appears inside the window → no host-side critical path in steady decode.
* Rank-specific host delay in decode: for b = 1–2 one rank's graph launch is late, visible as
  idle in the first compute window of that rank only (carried per rank in the decode table, so the
  composition reproduces the resulting collective wait on the other ranks).  The cause (rank-0
  sampling / bookkeeping) is inferred, not measured.
* Eager iterations: per-rank per-launch host cost (component 2), rank 2 consistently slowest; the
  cause is not identifiable from the GPU/API trace (gap G4).
* Back-pressure: launch API time inflates when the GPU is behind (ring iterations), so the host
  window is not additive to GPU time.

This module computes the evidence tables (per class) that calibrate.py stores next to the
coefficients and that evaluate.py reports; the parameters it owns are the measured gap medians.
"""
from __future__ import annotations

import numpy as np

from .data import NRANK, PassData


def gap_table(passes: list[PassData], partition) -> dict:
    out = {}
    for pd in passes:
        if pd.label not in partition.calibration_passes:
            continue
        feats = pd.feats
        byit = {f.iteration: f for f in feats}
        cls_gaps: dict[str, list] = {}
        for f in feats:
            nxt = byit.get(f.iteration + 1)
            if nxt is None or f.excluded or nxt.excluded or f.role != "calibration" or nxt.role != "calibration":
                continue
            c = f"decode b={nxt.b}" if nxt.stage == "decode" else f"mixed-{nxt.mixed_kind}"
            cls_gaps.setdefault(c, []).append((nxt.t0_ns - f.t1_ns) / 1e6)
        for c, v in cls_gaps.items():
            a = np.array(v)
            out.setdefault(c, []).append(dict(pass_=pd.label, n=len(a), median_ms=float(np.median(a)), p90_ms=float(np.percentile(a, 90)), max_ms=float(a.max())))
    return out


def decode_host_evidence(passes: list[PassData], partition, batches=(1, 2, 3, 6, 12, 24, 48)) -> dict:
    out = {}
    for pd in passes:
        if pd.label not in partition.calibration_passes:
            continue
        for b in batches:
            ids = [f.iteration for f in pd.feats if f.stage == "decode" and f.b == b and not f.excluded and f.role == "calibration"][:300]
            if len(ids) < 5:
                continue
            rows = np.array([pd.rank_rows(it) for it in ids])          # (n, R)
            med = lambda c: np.median(pd.ir[c][rows].astype(np.float64), axis=0)
            seg0 = []
            for it in ids[:100]:
                c = pd.collectives(it)
                if c is not None:
                    seg0.append(c["seg_idle_ns"][:, 0])
            out[f"{pd.label} b={b}"] = dict(
                n=len(ids), host_window_ms=(med("host_window_ns") / 1e6).tolist(), gpu_tail_after_host_ms=(med("gpu_tail_after_host_ns") / 1e6).tolist(),
                sync_calls=med("api_sync_n").tolist(), sync_max_ms=(med("api_sync_max_ns") / 1e6).tolist(), launches=med("fwd_launch_n").tolist(),
                first_window_idle_ms=(np.median(np.stack(seg0), axis=0) / 1e6).tolist() if seg0 else None,
                first_kernel_offset_ms=((med("first_start_ns") - np.median([pd.it_value("t0_ns", it) for it in ids])) / 1e6).tolist(),
            )
    return out


def backpressure_evidence(passes: list[PassData], partition, c_api_ns: float | None) -> dict:
    out = {}
    for pd in passes:
        if pd.label not in partition.calibration_passes:
            continue
        for kind in ("eager", "ring"):
            ids = [f.iteration for f in pd.feats if f.stage == "mixed" and f.mixed_kind == kind and not f.excluded and f.role == "calibration"]
            if not ids:
                continue
            rows = np.array([pd.rank_rows(it) for it in ids])
            L = pd.ir["fwd_launch_n"][rows].astype(np.float64)
            api_per_launch = pd.ir["fwd_launch_sum_ns"][rows] / np.maximum(L, 1) / 1e3
            fwd = pd.ir["fwd_window_ns"][rows] / 1e6
            busy = pd.ir["busy_ns"][rows] / 1e6
            walls = np.array([pd.it_value("wall_ns", it) for it in ids]) / 1e6
            out[f"{pd.label} {kind}"] = dict(
                n=len(ids), api_us_per_launch_median=float(np.median(api_per_launch)), api_us_per_launch_p90=float(np.percentile(api_per_launch, 90)),
                api_us_per_launch_max=float(api_per_launch.max()), fwd_window_ms_median_per_rank=np.median(fwd, axis=0).tolist(),
                gpu_busy_ms_median_per_rank=np.median(busy, axis=0).tolist(), wall_ms_median=float(np.median(walls)),
                wall_minus_max_fwd_ms_median=float(np.median(walls - fwd.max(axis=1))), wall_minus_max_busy_ms_median=float(np.median(walls - busy.max(axis=1))),
                unthrottled_api_us_per_launch=None if c_api_ns is None else c_api_ns / 1e3,
            )
    return out
