"""Deterministic CPU tests for the Phase 1B sequence simulator (synthetic data; no captures needed)."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import seqsim  # noqa: E402  (adds ../phase1a to sys.path)
from costmodel import groups as G  # noqa: E402
from costmodel.data import NRANK, Feat  # noqa: E402
from costmodel.partition import LeakageError, Partition  # noqa: E402
from seqsim.engine import Engine, Measured, _QueueLog  # noqa: E402
from seqsim.params import SeqParams  # noqa: E402

R = NRANK
MS = 1e6


# ------------------------------------------------------------------------------------------------
# synthetic coefficients (Phase 1A schema) and sequence parameters
# ------------------------------------------------------------------------------------------------
def _decode_entry(b, K=3, seg_ms=(1.5, 1.5, 1.5, 0.2), skew_ms=(0.0, 0.0, 0.0, 0.0), intr_us=5.0, jitter_ms=0.0):
    seg = np.array([[s + sk for s in seg_ms] for sk in skew_ms])      # (R, K+1)
    start = np.zeros(R)
    intr = np.full(K, intr_us * 1e3)
    q = np.linspace(-1, 1, 10) * jitter_ms * MS
    seg_q = seg[:, :, None] * MS + q[None, None, :]
    groups = np.zeros((R, len(G.GROUPS)))
    groups[:, G.GROUP_INDEX["moe"]] = seg.sum(axis=1).mean() * MS * 0.5
    groups[:, G.GROUP_INDEX["other"]] = seg.sum(axis=1).mean() * MS * 0.5
    return dict(n=50, regime="b<=16" if b <= 16 else ("b17-24" if b <= 24 else "b>=25"), K=K, kinds=[0] * 6, wall_median=float(seg.sum(axis=1).max() * MS),
                wall_mean=float(seg.sum(axis=1).max() * MS), passes=["synthetic"],
                start=start.tolist(), seg=(seg * MS).tolist(), intrinsic=intr.tolist(), wait=[0.0] * R, groups=groups.tolist(),
                compute_union=(seg.sum(axis=1) * MS).tolist(), comm_union=[0.0] * R, idle=[0.0] * R, busy=(seg.sum(axis=1) * MS).tolist(),
                overlap_cc=[0.0] * R, overlap_cm=[0.0] * R,
                start_q=np.zeros((R, 10)).tolist(), seg_q=seg_q.tolist(), intrinsic_q=np.repeat(intr[:, None], 10, axis=1).tolist())


def coeffs(jitter_ms=0.0, decode_skew=(0.0, 0.0, 0.0, 0.0)):
    K = 3
    kinds = [3, 3, 4]              # nccl_allreduce, nccl_allreduce, nccl_allgather
    prefill_coef = {}
    for g in ("fmha", "indexer", "sparse_meta", "moe", "other"):
        prefill_coef[g] = [[1e6, 2e3 if g != "indexer" else 1e3, 0.0, (2.5e5 if g == "indexer" else 0.0), 0.0] for _ in range(R)]
    return {
        "decode": dict(kv_max_observed=131_063, kv_sum_observed=600_000, n_fit=100, min_table_n=5, quantiles=list(np.linspace(0.05, 0.95, 10)), mc_draws=8, mc_seed=1,
                       table={"4": _decode_entry(4, K, jitter_ms=jitter_ms, skew_ms=decode_skew), "8": _decode_entry(8, K, seg_ms=(1.7, 1.7, 1.7, 0.2), jitter_ms=jitter_ms, skew_ms=decode_skew),
                              "32": _decode_entry(32, K, seg_ms=(2.0, 2.0, 2.0, 0.2))}),
        "collectives": dict(eager={"twoshot": dict(a_us=3.0, c_us_per_token=0.025, tokens_min=100, tokens_max=8000, n=10, rmse_us=0.5, status="measured"),
                                   "nccl_allreduce": dict(a_us=23.0, c_us_per_token=0.024, tokens_min=8192, tokens_max=16384, n=10, rmse_us=1.0, status="measured"),
                                   "nccl_allgather": dict(a_us=29.0, c_us_per_token=0.032, tokens_min=100, tokens_max=16384, n=10, rmse_us=1.0, status="measured")},
                            counts={"ring": {"nccl_allreduce": 2, "nccl_allgather": 1}, "eager": {"twoshot": 2, "nccl_allgather": 1}}, wait_ms={}),
        "prefill": dict(features=["const", "T", "Q", "KVW", "b"], coef_ns=prefill_coef, rmse_ns={g: 1e5 for g in prefill_coef},
                        ranges={"T": [300.0, 16384.0], "Q": [0.0, 16384.0], "KVW_k": [0.0, 98.0], "b": [0.0, 48.0]}, n_fit=50,
                        idle_ns={"ring": [1e5] * R, "eager": [1e5] * R}, overlap_cc_ns={"ring": [0.0] * R, "eager": [0.0] * R}, wait_ns={"ring": [0.0] * R},
                        chunk={}),
        "mixed": dict(launch=dict(l0=100.0, l1_per_P=0.0, l2_per_extra_chunk=0.0, l3_per_kv16=0.0, rmse=1.0, n=10, status="measured"),
                      c_host_ns=[100e3, 100e3, 110e3, 100e3], c_api_ns=13e3, outside_fwd_ns=[7e6] * R, tail_ns=2.0e6, c_kv_ns=0.0, c_kv_evidence={"status": "inferred"},
                      graph_table={}, host_bound_ids=[], n_fit=10, ranges={"P": [300.0, 16384.0], "chunks": [1.0, 4.0], "kv16": [0.0, 6.0], "L": [100.0, 200.0]},
                      lead_evidence_ms={}, decomposition={}, status={}),
        "moe": dict(coef_ns=[0.878e6, 21e3, 34e3], rmse_ns=1e5, r2=0.98, fit_rows=[dict(b=1), dict(b=48)], corpus={}, raw_b_fit={},
                    E_calibration={"4": dict(E_mean=21.0, E_p10=19.0, E_p90=23.0), "8": dict(E_mean=37.0, E_p10=34.0, E_p90=40.0), "32": dict(E_mean=120.0, E_p10=115.0, E_p90=125.0)},
                    E_heldout={}, status={}),
        "_kinds": kinds,
    }


def params(a_host_ms=0.2, eps_ms=0.6, f_idle_ms=0.6, Q=800.0, K=3, kinds=(3, 3, 4), c_pre_ring_ms=7.0):
    share = np.full((R, K + 1), 1.0 / (K + 1))
    share_q = np.repeat(share[:, :, None], 10, axis=2)
    prof = dict(K=K, kinds=list(kinds), share=share.tolist(), share_q=share_q.tolist(), n=10, source="synthetic", n_dense=10)
    return SeqParams(
        eps_sched_ns={"decode": {"4": dict(median=eps_ms * MS, p10=eps_ms * MS, p90=eps_ms * MS, n=50), "8": dict(median=eps_ms * MS, p10=0, p90=0, n=50),
                                 "32": dict(median=1.7 * MS, p10=0, p90=0, n=50)},
                      "ring": {"*": dict(median=3.7 * MS, p10=0, p90=0, n=20)}, "eager": {"*": dict(median=1.0 * MS, p10=0, p90=0, n=20)}},
        a_host_ns=[a_host_ms * MS] * R, f_idle_ns=f_idle_ms * MS, delta_launch_ns=20e3,
        h_pre_ns={"decode": [1.0 * MS] * R, "ring": [c_pre_ring_ms * MS] * R, "eager": [5.5 * MS] * R},
        p_post_ns={"decode": [0.5 * MS] * R, "ring": [1.6 * MS] * R, "eager": [1.5 * MS] * R},
        hostwin_dec_ns={"4": dict(median=[0.5 * MS] * R, n=50), "8": dict(median=[0.5 * MS] * R, n=50), "32": dict(median=[0.5 * MS] * R, n=50)},
        n_launch_dec={"4": dict(median=30.0, n=50), "8": dict(median=30.0, n=50), "32": dict(median=30.0, n=50)},
        queue_depth=Q, queue_evidence={}, profile={"ring": prof, "eager": prof}, idle_share={"ring": [0.0] * R, "eager": [0.0] * R},
        graph_mixed={}, status={}, evidence={}, n_fit=10)


def feat(it, stage="decode", b=4, P=0, chunks=None, kv_max=1000, excluded=False):
    f = Feat(iteration=it, stage=stage, excluded=excluded, b=b, kv_max=kv_max, kv_sum=kv_max * max(b, 1), P=P, chunks=chunks or ([] if P == 0 else [[0, P]]),
             wall_ns=0, t0_ns=0, t1_ns=0)
    f.regime = "b<=16"
    f.mixed_kind = "" if stage == "decode" else ("ring" if P + b >= 8192 else ("graph" if P + b <= 256 else "eager"))
    f.is_chunk16k = stage == "mixed" and len(f.chunks) == 1 and f.chunks[0][1] >= 16000
    f.kv0 = f.chunks[0][0] if f.is_chunk16k else 0
    return f


def run(sched, c=None, p=None, **kw):
    c = c or coeffs()
    p = p or params()
    first = Measured(start_rel=np.zeros(R), seg=np.ones((R, 1)) * MS, intrinsic=np.zeros(0), host_start=np.zeros(R), host_end=np.ones(R) * MS,
                     fwd_start=np.zeros(R), fwd_end=np.ones(R) * MS, launches=np.full(R, 30.0), t0=0.0, t1=MS)
    eng = Engine(c, p, **kw)
    return eng.run(sched, measured={sched[0].iteration: first}), eng


# ------------------------------------------------------------------------------------------------
# 1. collective barriers and rank skew
# ------------------------------------------------------------------------------------------------
def test_collective_barrier_rank_skew_not_a_sum():
    """A rank that is 0.3 ms late on every window delays the collective; the other ranks wait, nothing is summed."""
    c = coeffs(decode_skew=(0.0, 0.0, 0.3, 0.0))
    recs, _ = run([feat(i) for i in range(6)], c=c)
    r = recs[4]
    wall = r.wall[0] / MS
    # critical path: windows 1.8, 1.8, 1.8, 0.5 ms on rank 2 + 3 collectives of 5 us (not 4 ranks x ...); rank 2 also frees
    # its GPU 0.3 ms after the others at the end of i-1, so its first window of i starts 0.3 ms late (carried over, not summed)
    assert abs(wall - (3 * 1.8 + 0.5 + 3 * 0.005 + 0.3)) < 1e-6
    assert abs((recs[3].rank_end[0, 2] - recs[3].rank_end[0, 0]) / MS - 0.3) < 1e-6
    assert r.rank_wait[0, 2] == 0.0 and all(abs(r.rank_wait[0, q] / MS - (3 * 0.3 + 0.3)) < 1e-6 for q in (0, 1, 3))
    s = (3 * (3 * 1.5 + 0.2) + (3 * 1.8 + 0.5)) + 4 * 3 * 0.005          # sum of rank durations
    assert wall < s / 2


def test_barrier_completion_is_max_arrival_plus_intrinsic():
    """Mixed ring iteration: completion of each collective = max over ranks of arrival + intrinsic (explicit barrier)."""
    c = coeffs()
    c["mixed"]["c_host_ns"] = [1e3] * R                                  # host negligible -> GPU-bound
    recs, eng = run([feat(0), feat(1), feat(2), feat(3, "mixed", b=0, P=16000)], c=c, p=params(Q=1e9))
    r = recs[3]
    spec = eng._mixed_spec(feat(3, "mixed", b=0, P=16000), "ring")
    tot = spec["tot"]
    # all ranks equal windows -> no wait, wall = Σ windows (max rank) + Σ intrinsic + f_idle offset (GPU idle before)
    expect = tot.max() + spec["intr"].sum()
    assert abs((r.rank_end[0].max() - (r.host_start[0, 0] + 7.0 * MS + 20e3)) - expect) < 1.0
    assert np.all(r.rank_wait[0] == 0.0)


# ------------------------------------------------------------------------------------------------
# 2. compute / communication overlap preserved (collective waiting is not an intrinsic cost)
# ------------------------------------------------------------------------------------------------
def test_wait_is_separate_from_intrinsic_and_overlap_is_union():
    c = coeffs(decode_skew=(0.0, 0.5, 0.0, 0.0))
    recs, _ = run([feat(i) for i in range(6)], c=c)
    r = recs[4]
    # ranks 0,2,3 wait 0.5 ms at each of the 3 collectives plus 0.5 ms carried over from rank 1's late tail of i-1;
    # intrinsic is 5 us per collective on every rank and is never counted as waiting
    assert np.allclose(r.rank_wait[0, [0, 2, 3]] / MS, 2.0) and r.rank_wait[0, 1] == 0.0
    # overlap accounting (Phase 1A union) is reused: busy = compute_union + comm - overlap, never the sum of streams
    from costmodel.intervals import union_length
    s = np.array([0, 5, 20], dtype=np.int64); e = np.array([10, 15, 30], dtype=np.int64)
    assert union_length(s, e) == 25 and (e - s).sum() == 30


# ------------------------------------------------------------------------------------------------
# 3. host lead and synchronisation
# ------------------------------------------------------------------------------------------------
def test_decode_steady_state_is_gpu_bound_with_scheduler_lead():
    recs, _ = run([feat(i) for i in range(6)])
    r = recs[4]
    wall = r.wall[0]
    # host of i starts eps after GPU end of i-2 -> lead = wall(i-1) - eps
    assert abs(r.lead[0] - (recs[3].wall[0] - 0.6 * MS)) < 1.0
    assert r.binding == {"serial": 0, "scheduler": 1}
    # GPU continuous: no gap between iterations
    assert abs(recs[4].t0[0] - recs[3].t1[0]) < 1.0
    assert abs(wall / MS - (3 * 1.5 + 0.2 + 3 * 0.005)) < 1e-6


def test_host_bound_eager_mixed_is_paced_by_the_slowest_host_rank():
    """Eager mixed with an expensive host: rank 2 (slowest host) paces its own GPU; the others wait at every collective."""
    c = coeffs()
    c["mixed"]["c_host_ns"] = [1e6, 1e6, 1.2e6, 1e6]                     # 1 ms per launch, rank 2 20 % slower
    recs, eng = run([feat(0), feat(1), feat(2), feat(3, "mixed", b=4, P=2000)], c=c, p=params(Q=1e9))
    r = recs[3]
    L = 100.0
    hs = r.host_start[0]
    # rank 2 host path: h_pre + L * 1.2 ms (+ tail after last launch + delta)
    assert abs(r.rank_end[0, 2] - (hs[2] + 5.5 * MS + L * 1.2e6 + 20e3 + 2.0 * MS)) < 1.0
    assert r.rank_wait[0, 2] == 0.0 and (r.rank_wait[0, [0, 1, 3]] > 0).all()
    assert r.host_limited_segments[0].max() == 4                          # every segment waited for the host
    # the GPU finishes tail + delta after the slowest host's last launch (host end = last launch + p_post)
    assert abs((r.t1[0] - r.host_end[0, 2]) - (2.0 * MS + 20e3 - 1.5 * MS)) < 1.0


def test_lead_accumulates_along_a_chunk_chain_and_saturates_at_the_queue_depth():
    """GPU-bound 16k chunks: the host runs ahead by (GPU wall - host window) per chunk until Q launches ahead."""
    c = coeffs()
    c["mixed"]["c_host_ns"] = [50e3] * R                                   # host 100 launches x 50 us = 5 ms << GPU (~70 ms)
    sched = [feat(0)] + [feat(i, "mixed", b=0, P=16000, chunks=[[16000 * (i - 1), 16000]]) for i in range(1, 7)]
    recs, _ = run(sched, c=c, p=params(Q=60.0))
    leads = [r.lead[0] / MS for r in recs[2:]]
    walls = [r.wall[0] / MS for r in recs[2:]]
    assert leads[0] > 0 and leads[1] > leads[0]                            # lead grows chunk to chunk
    # throttle: host cannot be more than Q = 60 launches (0.6 iteration) ahead -> lead saturates below 0.6 wall
    assert max(leads) < 0.65 * max(walls)
    assert abs(leads[-1] - leads[-2]) < 0.05 * walls[-1]                   # saturated
    # with an unbounded queue the only cap is the scheduler's one-step-ahead rule: lead -> wall(i-1) - eps_sched(ring)
    recs2, _ = run(sched, c=c, p=params(Q=1e9))
    leads2 = [r.lead[0] / MS for r in recs2[2:]]
    walls2 = [r.wall[0] / MS for r in recs2[2:]]
    assert leads2[-1] > leads[-1] + 10
    assert abs(leads2[-1] - (walls2[-2] - 3.7)) < 0.5


def test_scheduler_constraint_binds_after_decode_and_serial_after_mixed():
    recs, _ = run([feat(0), feat(1), feat(2), feat(3), feat(4, "mixed", b=0, P=16000), feat(5)])
    assert recs[4].binding == {"serial": 0, "scheduler": 1}
    assert recs[5].binding == {"serial": 1, "scheduler": 0}
    r = recs[5]
    # after a mixed iteration the host of the next iteration starts a_host after its own host end (per rank)
    assert np.allclose(r.host_start[0] - recs[4].host_end[0], 0.2 * MS)


# ------------------------------------------------------------------------------------------------
# 4. CUDA graph decode versus eager/ring prefill and mixed iterations
# ------------------------------------------------------------------------------------------------
def test_graph_decode_is_not_launch_paced_but_eager_is():
    c = coeffs()
    c["mixed"]["c_host_ns"] = [2e6] * R                                   # absurdly slow host: 2 ms per launch
    p = params(Q=1e9)
    recs, _ = run([feat(0), feat(1), feat(2), feat(3), feat(4, "mixed", b=4, P=2000), feat(5)], c=c, p=p)
    dec = recs[3]; eag = recs[4]
    assert dec.host_limited_segments.max() == 0 and dec.kind == "decode"           # graph: one launch, GPU path from the table
    assert eag.host_limited_segments.max() == 4 and eag.kind == "eager"             # eager: every segment waits for the host
    assert abs(dec.wall[0] / MS - (3 * 1.5 + 0.2 + 0.015)) < 1e-6
    assert eag.wall[0] > 100 * 2.0 * MS
    # kind decision follows the payload rule (graph <= 256 tokens, ring >= 8192)
    from costmodel.data import mixed_kind
    assert mixed_kind(200, 4) == "graph" and mixed_kind(3000, 4) == "eager" and mixed_kind(16000, 0) == "ring"


# ------------------------------------------------------------------------------------------------
# 5. KV-range boundaries
# ------------------------------------------------------------------------------------------------
def test_kv_range_boundaries_are_flagged_not_silently_extrapolated():
    recs, _ = run([feat(0), feat(1, kv_max=131_063), feat(2, kv_max=131_064)])
    assert recs[1].kv_status == "in-range" and recs[2].kv_status == "kv-extrapolation"
    # indexer KV dependence only inside the validated KVW range: beyond it the prefill status is extrapolation
    recs2, _ = run([feat(0), feat(1, "mixed", b=0, P=16000, chunks=[[90_000, 16000]]), feat(2, "mixed", b=0, P=16000, chunks=[[120_000, 16000]])])
    assert recs2[1].status in ("in-range", "inferred") and recs2[1].kv_status == "in-range"
    assert recs2[2].status == "extrapolation" and recs2[2].kv_status == "kv-extrapolation"
    # and the slope is applied (longer at larger kv0)
    assert recs2[2].wall[0] > recs2[1].wall[0]


# ------------------------------------------------------------------------------------------------
# 6. missing routing information
# ------------------------------------------------------------------------------------------------
def test_missing_routing_gives_band_not_point_and_no_per_iteration_claim():
    c = coeffs()
    recs_det, eng = run([feat(0), feat(1, b=4), feat(2, b=4)], c=c, draws=1, stochastic=False, moe_band=False)
    recs_mc, _ = run([feat(0), feat(1, b=4), feat(2, b=4)], c=c, draws=16, seed=3, stochastic=True, moe_band=True)
    w = recs_mc[1].wall / MS
    assert w.std() > 0 and w.min() < recs_det[1].wall[0] / MS < w.max()
    # a padded batch without a corpus E(b) estimate -> no band, status extrapolation, no crash
    from costmodel.moe import MoeModel
    m = MoeModel.from_dict(c["moe"])
    out = m.predict(64)
    assert out["status"] == "extrapolation" and "central_ns" not in out
    recs3, _ = run([feat(0), feat(1, b=64)], c=c, draws=4, stochastic=True, moe_band=True)
    assert recs3[1].status == "extrapolation"


# ------------------------------------------------------------------------------------------------
# 7. stochastic reproducibility
# ------------------------------------------------------------------------------------------------
def test_stochastic_runs_are_reproducible_and_seed_sensitive():
    c = coeffs(jitter_ms=0.05)
    sched = [feat(i) for i in range(8)]
    a, _ = run(sched, c=c, draws=8, seed=11, stochastic=True, moe_band=True)
    b, _ = run(sched, c=c, draws=8, seed=11, stochastic=True, moe_band=True)
    d, _ = run(sched, c=c, draws=8, seed=12, stochastic=True, moe_band=True)
    assert all(np.array_equal(x.t1, y.t1) for x, y in zip(a, b))
    assert any(not np.array_equal(x.t1, y.t1) for x, y in zip(a, d))
    # expected wall under jitter >= deterministic median path (max over ranks is convex)
    det, _ = run(sched, c=c, draws=1, stochastic=False)
    assert a[5].wall.mean() >= det[5].wall[0] - 1.0


def test_params_roundtrip_is_exact():
    p = params()
    q = SeqParams.from_dict(json.loads(json.dumps(p.to_dict())))
    assert q.to_dict() == p.to_dict()


# ------------------------------------------------------------------------------------------------
# 8. holdout leakage
# ------------------------------------------------------------------------------------------------
def test_holdout_leakage_is_refused_by_the_frozen_partition():
    part = {"passes": {"synthetic": {"role": {"1": "calibration", "2": "heldout"}, "cls": {"1": "x", "2": "y"}}},
            "moe_calibration_requests": [], "moe_heldout_requests": []}
    P = Partition(part)
    assert P.calibration_only("synthetic", [1]) == [1]
    with pytest.raises(LeakageError):
        P.calibration_only("synthetic", [1, 2])


def test_seq_params_fit_uses_only_calibration_rows():
    """SeqParams.fit must call the leakage guard with calibration ids only."""
    class Guard:
        calibration_passes = ("synthetic",)
        calls = []

        def calibration_only(self, label, ids):
            self.calls.append(list(ids))
            if any(i == 2 for i in ids):
                raise LeakageError("held-out id in a fit")
            return list(ids)

    class PD:
        label = "synthetic"
        feats = [feat(1), feat(2), feat(3)]
        ir = {k: np.zeros(12) for k in ("host_start_ns", "host_end_ns", "fwd_start_ns", "fwd_end_ns", "fwd_launch_n", "compute_union_ns", "idle_ns")}

        def rank_rows(self, it):
            return np.arange(4)

    PD.feats[0].role = "calibration"; PD.feats[1].role = "heldout"; PD.feats[2].role = "calibration"
    g = Guard()
    SeqParams.fit([PD()], g, {})
    assert g.calls and all(2 not in c for c in g.calls)


# ------------------------------------------------------------------------------------------------
# 9. preservation of existing simulator predictions / artifacts
# ------------------------------------------------------------------------------------------------
def test_phase1a_and_simulator_sources_untouched():
    """Phase 1B is additive: outside its own directory the tree equals the Phase 1A commit a7725255 (skips without git)."""
    repo = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", ".."))
    if not os.path.isdir(os.path.join(repo, ".git")):
        pytest.skip("not inside the git checkout")
    base = "a7725255"
    try:
        subprocess.run(["git", "-C", repo, "cat-file", "-e", f"{base}^{{commit}}"], check=True, capture_output=True)
    except Exception:
        pytest.skip("base commit not available")
    diff = subprocess.run(["git", "-C", repo, "diff", "--name-only", base, "HEAD", "--", ".", ":(exclude)profiling/env/arm64-gb300/dsv41-flash-tp4/phase1b"],
                          capture_output=True, text=True, check=True).stdout.split()
    assert diff == [], f"files outside phase1b changed since {base}: {diff}"
    status = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--", ".", ":(exclude)profiling/env/arm64-gb300/dsv41-flash-tp4/phase1b"],
                            capture_output=True, text=True, check=True).stdout.split("\n")
    dirty = [s for s in status if s.strip() and "__pycache__" not in s and ".pytest_cache" not in s]
    assert dirty == [], f"uncommitted changes outside phase1b: {dirty}"


def test_phase1a_coefficients_hash_recorded_and_unchanged():
    p1a = os.path.join(HERE, "..", "..", "phase1a", "out")
    if not os.path.exists(os.path.join(p1a, "COEFFICIENTS.sha256")):
        pytest.skip("Phase 1A outputs not present")
    import hashlib
    want = open(os.path.join(p1a, "COEFFICIENTS.sha256")).read().split()[0]
    have = hashlib.sha256(open(os.path.join(p1a, "coefficients.json"), "rb").read()).hexdigest()
    assert want == have == "78f937c500ff9f205db425a1573789d182df22cf67e8b84de39b22a7ae01c1c9"


def test_no_global_multiplier_in_engine_source():
    src = open(os.path.join(HERE, "..", "seqsim", "engine.py")).read()
    for banned in ("GLOBAL_MULT", "latency_multiplier", "fudge", "correction_factor"):
        assert banned not in src


def test_queue_log_interpolates_launch_completion():
    q = _QueueLog(2)
    q.push(0, np.array([0.0, 0.0]), np.array([10.0, 10.0]), np.array([100.0, 200.0]), np.array([200.0, 400.0]))
    t = q.done_of_launch(0, np.array([5.0, 2.5]))
    assert np.allclose(t, [150.0, 250.0])
    assert (q.done_of_launch(0, np.array([-1.0, -1.0])) < -1e29).all()
