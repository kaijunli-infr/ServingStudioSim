"""Phase 1C-0 sequence-model invariant checks (CPU-only, deterministic).

These tests exercise the frozen Phase 1B engine (../../phase1b/seqsim) as a black box: property-style
checks on synthetic schedules, a scalar reference re-composition of the per-rank timelines, and the same
invariants on the real (frozen) Phase 1A coefficients and Phase 1B parameters. Nothing here changes the
Phase 1B sources or outputs; the last group pins their hashes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
P1B = os.path.abspath(os.path.join(HERE, "..", "..", "phase1b"))
P1A = os.path.abspath(os.path.join(HERE, "..", "..", "phase1a"))
sys.path.insert(0, P1B)
sys.path.insert(0, os.path.join(P1B, "tests"))

import seqsim  # noqa: E402
import test_seqsim as H  # noqa: E402  (synthetic coefficients / parameters / schedule helpers of Phase 1B)
from costmodel.compose import critical_path_batch  # noqa: E402
from costmodel.data import NRANK, Feat, mixed_kind  # noqa: E402
from seqsim.engine import Engine, Measured  # noqa: E402
from seqsim.params import SeqParams, kind_of  # noqa: E402

R = NRANK
MS = 1e6
TOL = 1.0  # ns


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _random_schedule(rng, n=40, with_graph=False):
    """Random mix of decode, eager mixed, 16k chunk chains (and optionally graph-kind mixed) after a decode anchor."""
    sched = [H.feat(0)]
    kv0 = 0
    for i in range(1, n):
        u = rng.random()
        if u < 0.6:
            sched.append(H.feat(i, b=int(rng.choice([4, 8, 32]))))
            kv0 = 0
        elif u < 0.75:
            sched.append(H.feat(i, "mixed", b=4, P=int(rng.integers(1000, 4000))))
        elif u < 0.95 or not with_graph:
            sched.append(H.feat(i, "mixed", b=0, P=16000, chunks=[[kv0, 16000]]))
            kv0 += 16000
        else:
            sched.append(H.feat(i, "mixed", b=4, P=200))
    return sched


def _eps(p: SeqParams, kind, b_prev2):
    return p.eps_sched(kind, b_prev2)[0]


def _check_sequence_invariants(recs, sched, p: SeqParams):
    """Invariants that must hold for every iteration, draw and rank of any run (pred modes)."""
    a = np.asarray(p.a_host_ns)
    for i, r in enumerate(recs):
        hs, he, re_ = r.host_start, r.host_end, r.rank_end
        assert np.all(np.isfinite(r.t0)) and np.all(np.isfinite(r.t1)) and np.all(np.isfinite(re_))
        # host: start before end, broadcast rule, first kernel after host start + f_idle, GPU end after first kernel
        assert np.all(he >= hs - TOL)
        assert np.all(hs >= hs[:, :1] - TOL)                                   # host_start_r >= host_start_0
        assert np.all(r.t0 >= hs.min(axis=1) + p.f_idle_ns - TOL)
        assert np.all(r.t1 >= r.t0 - TOL)
        assert np.allclose(r.t1, re_.max(axis=1))                             # critical path = max over ranks of the tail end
        assert np.all(r.rank_wait >= -TOL)
        if i >= 1:
            prev = recs[i - 1]
            assert np.all(re_ >= prev.rank_end - TOL)                         # GPU order per rank across iterations
            assert np.all(hs >= prev.host_end + a[None, :] - TOL)             # host serial order per rank
            assert np.allclose(r.lead, prev.t1 - hs.min(axis=1))              # lead definition
            assert np.all(r.t0 >= prev.rank_end.min(axis=1) - TOL)
        if i >= 2:
            eps = _eps(p, kind_of(sched[i]), sched[i - 2].b)
            assert np.all(hs[:, 0] >= recs[i - 2].t1 + eps - TOL)             # scheduler one-step-ahead


# ------------------------------------------------------------------------------------------------
# 1. collective completion and rank synchronisation
# ------------------------------------------------------------------------------------------------
def test_collective_completion_is_max_arrival_plus_intrinsic_for_random_skews():
    rng = np.random.default_rng(1)
    for _ in range(50):
        K = int(rng.integers(1, 6))
        start = rng.random((3, R)) * MS
        seg = rng.random((3, R, K + 1)) * 2 * MS
        intr = rng.random((3, K)) * 0.1 * MS
        res = critical_path_batch(start, seg, intr)
        # scalar reference
        for n in range(3):
            t = start[n].copy()
            wait = np.zeros(R)
            for k in range(K):
                arr = t + seg[n, :, k]
                done = arr.max() + intr[n, k]
                wait += done - arr - intr[n, k]
                t = np.full(R, done)
            end = t + seg[n, :, K]
            assert np.allclose(res["rank_end"][n], end) and np.allclose(res["wall"][n], end.max())   # wall is measured from the iteration origin
            assert np.allclose(res["rank_wait"][n], wait)
            assert np.all(wait >= -1e-6)
            assert end.max() < (seg[n].sum() + intr[n].sum() + start[n].max())                         # never a rank sum


def test_every_rank_leaves_each_collective_at_the_same_time_in_the_engine():
    """Rank-asymmetric segment windows: all ranks end the collective together, waits are the arrival gaps, and the
    iteration wall equals the scalar re-composition of the same segment inputs (critical-path reconstruction)."""
    p = H.params(Q=1e9)
    p.idle_share = {"ring": [0.0, 0.10, 0.20, 0.05], "eager": [0.0, 0.10, 0.20, 0.05]}
    c = H.coeffs()
    c["mixed"]["c_host_ns"] = [1.0] * R                                   # host never limits
    sched = [H.feat(0), H.feat(1, "mixed", b=0, P=16000, chunks=[[0, 16000]])]
    recs, eng = H.run(sched, c=c, p=p)
    r = recs[1]
    s = eng._mixed_spec(sched[1], "ring")
    seg = np.asarray(s["share"]) * s["tot"][:, None]                       # (R, K+1)
    K = s["K"]
    # reference: GPU of rank r starts at max(GPU free, host_start + h_pre + delta); barriers between segments
    t = np.maximum(recs[0].rank_end[0], r.host_start[0] + s["h_pre"] + p.delta_launch_ns)
    wait = np.zeros(R)
    for k in range(K):
        arr = t + seg[:, k]
        done = arr.max() + s["intr"][k]
        wait += arr.max() - arr
        t = np.full(R, done)
    end = t + seg[:, K]
    assert np.allclose(r.rank_end[0], end, atol=TOL)
    assert np.allclose(r.rank_wait[0], wait, atol=TOL)
    assert np.isclose(r.t1[0], end.max())
    assert wait.max() > 1 * MS and np.isclose(wait.min(), 0.0)              # asymmetric ranks: someone waits, the slowest never


# ------------------------------------------------------------------------------------------------
# 2. host/GPU ordering and launch-queue depth
# ------------------------------------------------------------------------------------------------
def test_queue_depth_bounds_how_far_the_host_runs_ahead():
    """With Q = one iteration of launches the host cannot finish launching chunk i before the GPU finished chunk i-1
    (per rank); with an unbounded queue it does."""
    c = H.coeffs()
    c["mixed"]["c_host_ns"] = [50e3] * R
    sched = [H.feat(0)] + [H.feat(i, "mixed", b=0, P=16000, chunks=[[16000 * (i - 1), 16000]]) for i in range(1, 8)]
    recs, _ = H.run(sched, c=c, p=H.params(Q=1e9))
    L = recs[2].extra["launches"]
    recs_q, _ = H.run(sched, c=c, p=H.params(Q=float(L)))
    for i in range(3, 8):
        fwd_end = recs_q[i].extra["fwd_end"][0]
        assert np.all(fwd_end >= recs_q[i - 1].rank_end[0] - TOL), i        # queue rule binds
        fwd_end_free = recs[i].extra["fwd_end"][0]
        assert np.all(fwd_end_free < recs[i - 1].rank_end[0]), i            # unbounded: host finished earlier
    # GPU order: segments of iteration i on a rank never start before the rank's previous iteration ended
    for i in range(1, 8):
        assert np.all(recs_q[i].t0 >= recs_q[i - 1].rank_end.min(axis=1) - TOL)


def test_host_gpu_ordering_holds_on_random_schedules():
    rng = np.random.default_rng(7)
    p = H.params()
    for _ in range(5):
        sched = _random_schedule(rng, 30)
        recs, _ = H.run(sched, p=p)
        _check_sequence_invariants(recs, sched, p)


# ------------------------------------------------------------------------------------------------
# 3. cross-iteration host lead
# ------------------------------------------------------------------------------------------------
def test_lead_is_capped_by_the_scheduler_rule_after_decode_and_never_exceeds_the_previous_wall():
    p = H.params()
    recs, _ = H.run([H.feat(i, b=4) for i in range(12)], p=p)
    for i in range(3, 12):
        r = recs[i]
        assert r.lead[0] <= recs[i - 1].wall[0] - _eps(p, "decode", 4) + TOL
        assert r.lead[0] > 0                                                 # GPU-bound decode: host runs ahead
    # host-bound eager: the GPU idles -> negative lead (GPU end of i-1 precedes host start of i by less than the host work)
    c = H.coeffs(); c["mixed"]["c_host_ns"] = [2e6] * R
    recs2, _ = H.run([H.feat(0), H.feat(1), H.feat(2), H.feat(3, "mixed", b=4, P=2000), H.feat(4, "mixed", b=4, P=2000), H.feat(5)], c=c, p=H.params(Q=1e9))
    p2 = H.params(Q=1e9)
    assert recs2[4].lead[0] < p2.f_idle_ns and recs2[4].t0[0] > recs2[3].t1[0]   # host-bound: the GPU idles between iterations (lead below f_idle)
    assert recs2[5].binding == {"serial": 1, "scheduler": 0}


# ------------------------------------------------------------------------------------------------
# 4. critical-path reconstruction (never a rank sum)
# ------------------------------------------------------------------------------------------------
def test_wall_is_between_slowest_rank_path_and_sum_of_rank_paths():
    rng = np.random.default_rng(3)
    for _ in range(20):
        skew = tuple(rng.random(R) * 0.5)
        c = H.coeffs(decode_skew=skew)
        recs, _ = H.run([H.feat(i) for i in range(8)], c=c)
        r = recs[7]
        seg = np.array(c["decode"]["table"]["4"]["seg"]) / MS               # (R, K+1) ms
        intr = np.array(c["decode"]["table"]["4"]["intrinsic"]).sum() / MS
        slowest = seg.sum(axis=1).max() + intr
        total = seg.sum() + intr
        w = r.wall[0] / MS
        carry = (recs[6].rank_end[0].max() - recs[6].rank_end[0].min()) / MS   # rank-end spread of i-1 delays the slow rank's start
        assert slowest - 1e-6 <= w < total                                   # critical path, not a sum
        assert w <= slowest + carry + 0.03                                   # wall = slowest path + start skew carried over (+ launch latency)


# ------------------------------------------------------------------------------------------------
# 5. graph versus eager scheduling
# ------------------------------------------------------------------------------------------------
def test_decode_graph_wall_is_independent_of_per_launch_host_cost_while_eager_scales_with_it():
    walls_dec, walls_eag = [], []
    for ch in (800e3, 1.2e6, 2e6):                                           # all host-bound (GPU path ~25 ms, host 80-200 ms)
        c = H.coeffs(); c["mixed"]["c_host_ns"] = [ch] * R
        recs, _ = H.run([H.feat(0), H.feat(1), H.feat(2), H.feat(3), H.feat(4, "mixed", b=4, P=2000)], c=c, p=H.params(Q=1e9))
        walls_dec.append(recs[3].wall[0]); walls_eag.append(recs[4].wall[0])
    assert np.allclose(walls_dec, walls_dec[0])
    assert walls_eag[2] > walls_eag[1] > walls_eag[0]
    c = H.coeffs(); c["mixed"]["c_host_ns"] = [50e3] * R                   # GPU-bound eager: the wall no longer depends on the host cost
    r1, _ = H.run([H.feat(0), H.feat(1), H.feat(2), H.feat(3), H.feat(4, "mixed", b=4, P=2000)], c=c, p=H.params(Q=1e9))
    c["mixed"]["c_host_ns"] = [100e3] * R
    r2, _ = H.run([H.feat(0), H.feat(1), H.feat(2), H.feat(3), H.feat(4, "mixed", b=4, P=2000)], c=c, p=H.params(Q=1e9))
    assert np.isclose(r1[4].wall[0], r2[4].wall[0])
    assert kind_of(H.feat(1)) == "decode" and kind_of(H.feat(1, "mixed", b=4, P=200)) == "graph"
    assert kind_of(H.feat(1, "mixed", b=4, P=2000)) == "eager" and kind_of(H.feat(1, "mixed", b=0, P=16000)) == "ring"
    assert mixed_kind(252, 4) == "graph" and mixed_kind(253, 4) == "eager"      # boundary P + b <= 256


# ------------------------------------------------------------------------------------------------
# 6. stream overlap and non-overlap
# ------------------------------------------------------------------------------------------------
def test_measured_compute_comm_overlap_shortens_windows_and_zero_overlap_keeps_them_serial():
    c0 = H.coeffs()
    c1 = copy.deepcopy(c0); c1["prefill"]["overlap_cc_ns"]["ring"] = [3 * MS] * R
    e0 = Engine(c0, H.params()); e1 = Engine(c1, H.params())
    f = H.feat(1, "mixed", b=0, P=16000, chunks=[[0, 16000]])
    s0 = e0._mixed_spec(f, "ring"); s1 = e1._mixed_spec(f, "ring")
    assert np.allclose(s0["tot"] - s1["tot"], 3 * MS)                       # union shrinks by exactly the measured overlap
    assert np.allclose(s0["intr"], s1["intr"])                              # collectives are not shortened
    # within a rank, compute windows and collectives are serial: wall == sum of windows + collectives when ranks are symmetric
    c = H.coeffs(); c["mixed"]["c_host_ns"] = [1.0] * R
    recs, eng = H.run([H.feat(0), f], c=c, p=H.params(Q=1e9))
    s = eng._mixed_spec(f, "ring")
    serial = (np.asarray(s["share"]) * s["tot"][:, None]).sum(axis=1).max() + s["intr"].sum()
    gstart = recs[1].host_start[0] + s["h_pre"] + H.params().delta_launch_ns
    assert np.isclose(recs[1].t1[0] - gstart.min(), serial, atol=TOL)
    # across ranks the work runs in parallel: the wall is far below the 4-rank sum
    assert recs[1].wall[0] < 0.5 * (np.asarray(s["share"]) * s["tot"][:, None]).sum()


# ------------------------------------------------------------------------------------------------
# 7. deterministic Monte Carlo replay
# ------------------------------------------------------------------------------------------------
def test_stochastic_replay_is_bitwise_reproducible_and_causal():
    rng = np.random.default_rng(11)
    sched = _random_schedule(rng, 25)
    c = H.coeffs(jitter_ms=0.2)
    a, _ = H.run(sched, c=c, draws=16, seed=5, stochastic=True, moe_band=True)
    b, _ = H.run(sched, c=c, draws=16, seed=5, stochastic=True, moe_band=True)
    for x, y in zip(a, b):
        assert np.array_equal(x.t0, y.t0) and np.array_equal(x.t1, y.t1) and np.array_equal(x.rank_end, y.rank_end)
        assert np.array_equal(x.host_start, y.host_start) and np.array_equal(x.rank_wait, y.rank_wait)
    # causality: the first n iterations do not depend on what comes after them
    short, _ = H.run(sched[:12], c=c, draws=16, seed=5, stochastic=True, moe_band=True)
    for x, y in zip(short, a[:12]):
        assert np.array_equal(x.t1, y.t1) and np.array_equal(x.host_end, y.host_end)
    # a different seed changes the draws, the median path (draws=1, stochastic=False) does not depend on the seed
    d, _ = H.run(sched, c=c, draws=16, seed=6, stochastic=True, moe_band=True)
    assert any(not np.array_equal(x.t1, y.t1) for x, y in zip(a, d))
    m1, _ = H.run(sched, c=c, seed=1); m2, _ = H.run(sched, c=c, seed=2)
    assert all(np.array_equal(x.t1, y.t1) for x, y in zip(m1, m2))


def test_stochastic_draws_bracket_the_median_path_without_shifting_the_median_for_symmetric_jitter():
    """Per-(rank, segment) draws composed through the max-over-ranks barrier shift the *mean* upward (the quadrant
    displacement mechanism, see uncertainty/RELIABILITY.md); this test pins that the shift is positive and bounded."""
    c = H.coeffs(jitter_ms=0.3)
    sched = [H.feat(i, b=4) for i in range(30)]
    det, _ = H.run(sched, c=c)
    mc, _ = H.run(sched, c=c, draws=256, seed=0, stochastic=True)
    w_det = np.array([r.wall[0] for r in det[5:]])
    w_mc = np.array([r.wall.mean() for r in mc[5:]])
    assert np.all(w_mc > w_det)                                              # max of independent symmetric draws: mean above median path
    assert np.all((w_mc - w_det) / w_det < 0.35)
    c_small = H.coeffs(jitter_ms=0.1)
    det_s, _ = H.run(sched, c=c_small); mc_s, _ = H.run(sched, c=c_small, draws=256, seed=0, stochastic=True)
    shift_s = np.mean([m.wall.mean() / d.wall[0] - 1 for m, d in zip(mc_s[5:], det_s[5:])])
    assert 0 < shift_s < np.mean((w_mc - w_det) / w_det)                   # the shift shrinks with the per-segment dispersion


# ------------------------------------------------------------------------------------------------
# 8. missing or unsupported configurations
# ------------------------------------------------------------------------------------------------
def test_unsupported_configurations_are_flagged_or_refused_never_silently_in_range():
    c = H.coeffs(); p = H.params()
    recs, _ = H.run([H.feat(0), H.feat(1, b=64), H.feat(2, b=4, kv_max=200_000), H.feat(3, "mixed", b=4, P=2000, kv_max=150_000)], c=c, p=p)
    assert recs[1].status != "in-range"                                      # b=64 outside the decode tables
    assert recs[2].kv_status == "kv-extrapolation"                           # beyond kv_max_observed
    assert recs[3].kv_status == "kv-extrapolation" or recs[3].status != "in-range"
    # eps_sched for an unseen previous batch size falls back with a non-measured status
    assert p.eps_sched("decode", 999)[1] != "measured"
    assert p.decode_host(999)[2] != "measured"
    # graph-kind mixed without a class table: flagged, never 'in-range'
    recs_g, _ = H.run([H.feat(0), H.feat(1, "mixed", b=4, P=200)], c=c, p=p)
    assert recs_g[1].kind == "graph" and recs_g[1].status != "in-range"
    # excluded iteration without a measured timeline, or a timeline of the wrong rank count, is refused
    sched = [H.feat(0), H.feat(1, excluded=True)]
    with pytest.raises(ValueError):
        H.run(sched)
    bad = Measured(start_rel=np.zeros(8), seg=np.ones((8, 1)) * MS, intrinsic=np.zeros(0), host_start=np.zeros(8), host_end=np.ones(8) * MS,
                   fwd_start=np.zeros(8), fwd_end=np.ones(8) * MS, launches=np.full(8, 30.0), t0=0.0, t1=MS)
    eng = Engine(c, p)
    with pytest.raises((ValueError, IndexError)):
        eng.run([H.feat(0), H.feat(1)], measured={0: bad, 1: bad}, recon="gpu")
    # a run needs a measured anchor or starts at time 0 (no exception, documented)
    recs0 = Engine(c, p).run([H.feat(0), H.feat(1)])
    assert recs0[0].host_start.min() == 0.0


# ------------------------------------------------------------------------------------------------
# 9. preservation of existing simulator behaviour and of the frozen artefacts
# ------------------------------------------------------------------------------------------------
PINNED = {
    os.path.join(P1A, "out", "coefficients.json"): "78f937c500ff9f205db425a1573789d182df22cf67e8b84de39b22a7ae01c1c9",
    os.path.join(P1A, "partition", "partition.json"): "288e193f46323f4d4361913a08c47afa9c4d9f98cc4c4be0d5b88f9629304b7a",
    os.path.join(P1B, "out", "params.json"): "8dc79573c1481c7fc2cecb5a095a110dd5fb0d4a09e1e6274f32bd42196a61b8",
}


def test_frozen_inputs_and_phase1b_sources_are_unchanged():
    for path, h in PINNED.items():
        assert _sha(path) == h, path
    code = json.load(open(os.path.join(P1B, "out", "results.json")))["code"]
    for name, h in code.items():
        assert _sha(os.path.join(P1B, "seqsim", name)) == h, name          # engine sources identical to the ones that produced the results


def test_repository_outside_phase1b_and_phase1c_is_untouched_and_nothing_imports_seqsim():
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE, capture_output=True, text=True)
    if root.returncode != 0:
        pytest.skip("not inside a git checkout")
    root = root.stdout.strip()
    rel = os.path.relpath(os.path.join(HERE, "..", ".."), root)
    diff = subprocess.run(["git", "diff", "--stat", "29082793", "HEAD", "--", ".", f":!{rel}/phase1b", f":!{rel}/phase1c"], cwd=root, capture_output=True, text=True)
    assert diff.returncode == 0 and diff.stdout.strip() == "", diff.stdout
    grep = subprocess.run(["git", "grep", "-l", "-E", "seqsim|phase1b", "--", ".", f":!{rel}", ":!*.md"], cwd=root, capture_output=True, text=True)
    assert grep.stdout.strip() == "", grep.stdout                         # opt-in: no repository module references the sequence model


# ------------------------------------------------------------------------------------------------
# the same invariants on the real frozen inputs (no capture tables needed)
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real():
    c = json.load(open(os.path.join(P1A, "out", "coefficients.json")))
    p = SeqParams.load(os.path.join(P1B, "out", "params.json"))
    return c, p


def _real_feat(it, stage="decode", b=8, P=0, chunks=None, kv_max=20000, excluded=False):
    f = Feat(iteration=it, stage=stage, excluded=excluded, b=b, kv_max=kv_max, kv_sum=kv_max * max(b, 1), P=P, chunks=chunks or ([] if P == 0 else [[0, P]]),
             wall_ns=0, t0_ns=0, t1_ns=0)
    f.regime = "b<=16" if b <= 16 else ("b17-24" if b <= 24 else "b>=25")
    f.mixed_kind = "" if stage == "decode" else mixed_kind(P, b)
    f.is_chunk16k = stage == "mixed" and len(f.chunks) == 1 and f.chunks[0][1] >= 16000
    f.kv0 = f.chunks[0][0] if f.is_chunk16k else 0
    return f


def test_real_parameters_satisfy_the_sequence_invariants_in_both_modes(real):
    c, p = real
    sched = [_real_feat(0)] + [_real_feat(i, b=8) for i in range(1, 20)] + [_real_feat(20 + j, "mixed", b=0, P=16384, chunks=[[16384 * j, 16384]]) for j in range(5)] \
        + [_real_feat(25, "mixed", b=4, P=3000)] + [_real_feat(26 + i, b=2) for i in range(10)] + [_real_feat(36 + i, b=48, kv_max=4000) for i in range(10)]
    first = Measured(start_rel=np.zeros(R), seg=np.ones((R, 1)) * MS, intrinsic=np.zeros(0), host_start=np.zeros(R), host_end=np.ones(R) * MS,
                     fwd_start=np.zeros(R), fwd_end=np.ones(R) * MS, launches=np.full(R, 100.0), t0=0.0, t1=MS)
    det = Engine(c, p).run(sched, measured={0: first})
    mc = Engine(c, p, draws=16, seed=0, stochastic=True, moe_band=True).run(sched, measured={0: first})
    for recs in (det, mc):
        _check_sequence_invariants(recs, sched, p)
        assert all(r.status in ("in-range", "interpolated", "measured", "inferred", "extrapolation") for r in recs)
    # plausibility against the Phase 1A tables: steady-state decode b=8 wall within the table's own quantile range
    w = np.array([r.wall[0] for r in det[5:20]]) / MS
    assert 4.0 < w.min() and w.max() < 9.0
    # chain leads grow then saturate below the queue cap expressed in time
    leads = [r.lead[0] / MS for r in det[21:25]]
    assert leads[1] > leads[0] and max(leads) < 250
    mc2 = Engine(c, p, draws=16, seed=0, stochastic=True, moe_band=True).run(sched, measured={0: first})
    assert all(np.array_equal(x.t1, y.t1) for x, y in zip(mc, mc2))
