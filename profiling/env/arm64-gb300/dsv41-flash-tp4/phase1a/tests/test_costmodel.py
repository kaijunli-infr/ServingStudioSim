"""CPU tests for the Phase 1A cost model (synthetic inputs only; no trace files needed).

Run from the phase1a directory:  python -m pytest tests -q
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from costmodel import compose, groups as G, partition  # noqa: E402
from costmodel.collectives import CollectiveModel  # noqa: E402
from costmodel.data import Feat, decode_regime, mixed_kind, padded_batch  # noqa: E402
from costmodel.decode import MIN_TABLE_N, QUANTILES, DecodeModel  # noqa: E402
from costmodel.intervals import accounting, union_length, windowed_union  # noqa: E402
from costmodel.longctx import PrefillComputeModel, chunk_predict  # noqa: E402
from costmodel.mixed import MixedModel  # noqa: E402
from costmodel.moe import MoeModel  # noqa: E402

R = 4


# ---------------------------------------------------------------- kernel overlap / union accounting
def test_union_overlap_idle_accounting():
    s = np.array([0, 5, 20, 22, 40])
    e = np.array([10, 12, 30, 25, 45])
    assert union_length(s, e) == 12 + 10 + 5                      # [0,12) ∪ [20,30) ∪ [40,45)
    acc = accounting(s, e, is_comm=np.array([False, False, True, False, False]))
    assert acc["busy"] == 27 and acc["span"] == 45 and acc["idle"] == 45 - 27
    assert acc["compute_sum"] == 10 + 7 + 3 + 5 and acc["comm_sum"] == 10
    assert acc["overlap_total"] == acc["compute_sum"] + acc["comm_sum"] - acc["busy"]
    assert acc["overlap_comp_comm"] == 3                           # [22,25) runs under the comm kernel [20,30)
    assert acc["overlap_comp_comp"] == 5                           # [5,10) under [0,10)
    # the sum of durations must never be used as busy time
    assert acc["compute_sum"] + acc["comm_sum"] > acc["busy"]


def test_windowed_union_clips_and_resets_at_window_boundaries():
    s = np.array([0, 8, 15, 30])
    e = np.array([12, 9, 25, 35])
    bounds = np.array([0, 10, 20, 40])
    u = windowed_union(s, e, bounds)
    assert u.tolist() == [10, 5, 5]                               # [0,12) clipped to [0,10); [15,25) clipped to [15,20); [30,35)
    assert windowed_union(s[:0], e[:0], bounds).tolist() == [0, 0, 0]


# ---------------------------------------------------------------- four-rank critical path
def _timeline():
    start = np.array([0.0, 0.0, 2.0, 0.0])                         # rank 2 launches 2 units late
    seg = np.array([[10, 10, 5], [10, 10, 5], [10, 12, 5], [10, 10, 5]], dtype=float)
    intr = np.array([3.0, 4.0])
    return start, seg, intr


def test_critical_path_is_not_a_sum_over_ranks():
    start, seg, intr = _timeline()
    res = compose.critical_path(start, seg, intr)
    # collective 0: arrivals 10,10,12,10 → done 15; collective 1: arrivals 25,25,27,25 → done 31; tail 5 → 36
    assert res.wall == pytest.approx(36.0)
    assert res.coll_done.tolist() == [15.0, 31.0]
    assert compose.sum_of_rank_durations(start, seg, intr) == pytest.approx(4 * 7 + 25 + 25 + 27 + 25)
    assert compose.sum_of_rank_durations(start, seg, intr) > 3.5 * res.wall
    assert res.slowest_rank == 2 and res.critical_count.tolist() == [0, 0, 2, 0]


def test_collective_wait_is_skew_not_intrinsic():
    start, seg, intr = _timeline()
    res = compose.critical_path(start, seg, intr)
    assert res.comm_intrinsic == pytest.approx(7.0)                 # intrinsic cost paid once, by every rank
    assert res.rank_wait.tolist() == [2 + 2, 2 + 2, 0, 2 + 2]       # ranks 0,1,3 wait 2 at each collective for rank 2
    # with no skew there is no wait, and the wall is compute + intrinsic
    res0 = compose.critical_path(np.zeros(R), np.full((R, 3), 10.0), intr)
    assert res0.rank_wait.tolist() == [0, 0, 0, 0] and res0.wall == pytest.approx(30 + 7)


def test_batch_critical_path_matches_scalar_version():
    start, seg, intr = _timeline()
    rng = np.random.default_rng(1)
    N = 16
    S = start[None] + rng.random((N, R))
    SG = seg[None] + rng.random((N, R, 3))
    IN = intr[None] + rng.random((N, 2))
    res = compose.critical_path_batch(S, SG, IN)
    for i in range(N):
        one = compose.critical_path(S[i], SG[i], IN[i])
        assert res["wall"][i] == pytest.approx(one.wall)
        assert np.allclose(res["rank_wait"][i], one.rank_wait)


# ---------------------------------------------------------------- decode table: stochastic vs deterministic, out of range
def _decode_table_entry(b, K=3, jitter=0.0, n=50):
    seg = np.full((R, K + 1), 10.0)
    start = np.zeros(R)
    intr = np.full(K, 1.0)
    q = QUANTILES
    spread = (q - 0.5) * 2 * jitter                                # symmetric jitter around the median
    entry = dict(n=n, regime=decode_regime(b), K=K, kinds=np.zeros(len(G.COMM_KINDS)), wall_median=0.0, wall_mean=0.0, passes=["synthetic"],
                 start=start, seg=seg, intrinsic=intr, wait=np.zeros(R), groups=np.ones((R, len(G.GROUPS))), compute_union=np.full(R, 30.0),
                 comm_union=np.full(R, 3.0), idle=np.zeros(R), busy=np.full(R, 33.0), overlap_cc=np.zeros(R), overlap_cm=np.zeros(R),
                 start_q=start[:, None] + spread[None, :], seg_q=seg[:, :, None] + spread[None, None, :], intrinsic_q=intr[:, None] + 0 * spread[None, :])
    return entry


def _feat(b=4, stage="decode", P=0, chunks=None, kv_max=1000):
    return Feat(iteration=1, stage=stage, excluded=False, b=b, kv_max=kv_max, kv_sum=kv_max * max(b, 1), P=P, chunks=chunks or [],
                wall_ns=0, t0_ns=0, t1_ns=0)


def test_stochastic_composition_exceeds_deterministic_only_with_jitter():
    m = DecodeModel()
    m.kv_max_observed = 5000
    m.table[4] = _decode_table_entry(4, jitter=0.0)
    p = m.predict(_feat(4))
    assert p["wall_det_ns"] == pytest.approx(40 + 3)
    assert p["wall_ns"] == pytest.approx(p["wall_det_ns"], rel=1e-9)   # no jitter → identical
    m.table[4] = _decode_table_entry(4, jitter=1.0)
    p = m.predict(_feat(4))
    assert p["wall_det_ns"] == pytest.approx(43)
    assert p["wall_ns"] > 43.0                                         # every sync point waits for the latest rank
    assert p["rank_wait_ns"].min() > 0


def test_decode_interpolation_extrapolation_and_min_n():
    m = DecodeModel()
    m.kv_max_observed = 5000
    m.table[4] = _decode_table_entry(4)
    m.table[8] = _decode_table_entry(8)
    m.table[8]["seg"] = m.table[8]["seg"] * 2
    m.table[8]["seg_q"] = m.table[8]["seg_q"] * 2
    m.table[6] = _decode_table_entry(6, n=MIN_TABLE_N - 1)             # too few samples: evidence only
    p6 = m.predict(_feat(6))
    assert p6["status"] == "interpolated" and p6["wall_det_ns"] == pytest.approx((43 + 83) / 2)
    assert m.predict(_feat(4))["status"] == "measured"
    p12 = m.predict(_feat(12))
    assert p12["status"] == "extrapolation"                             # beyond the calibrated range of the regime
    assert m.predict(_feat(4, kv_max=9000))["kv_status"] == "kv-extrapolation"
    with pytest.raises(KeyError):
        m.predict(_feat(30))                                            # no calibrated entry in regime b>=25


def test_decode_prediction_is_deterministic_and_roundtrips():
    m = DecodeModel()
    m.kv_max_observed = 5000
    m.table[4] = _decode_table_entry(4, jitter=0.7)
    a = m.predict(_feat(4))
    b = m.predict(_feat(4))
    assert a["wall_ns"] == b["wall_ns"] and np.array_equal(a["rank_wait_ns"], b["rank_wait_ns"])
    m2 = DecodeModel.from_dict(json.loads(json.dumps(m.to_dict())))
    c = m2.predict(_feat(4))
    assert c["wall_ns"] == pytest.approx(a["wall_ns"], rel=1e-12)


# ---------------------------------------------------------------- host / GPU composition of mixed iterations
def _mixed_models():
    prefill = PrefillComputeModel.from_dict(dict(
        features=["const", "T", "Q", "KVW", "b"],
        coef_ns={g: [[1e6, 2.5e3, 0.0, (1e5 if g == "indexer" else 0.0), 0.0]] * R for g in ("fmha", "indexer", "sparse_meta", "moe", "other")},
        rmse_ns={}, ranges={"T": (300, 16384), "Q": (0, 16384), "KVW_k": (0, 100), "b": (1, 48)}, n_fit=10,
        idle_ns={"ring": [1e6] * R}, overlap_cc_ns={"ring": [0] * R, "eager": [0] * R}, wait_ns={"ring": [0] * R},
        chunk={"n": 10, "kv0_min": 0, "kv0_max": 100000, "wall": {"a_ms": 220.0, "s_ms_per_1k": 0.25, "rmse_ms": 1}, "groups": {}}))
    coll = CollectiveModel.from_dict(dict(eager={"twoshot": dict(a_us=3, c_us_per_token=0.025, tokens_min=300, tokens_max=7300, n=5, rmse_us=1),
                                                "nccl_allreduce": dict(a_us=23, c_us_per_token=0.024, tokens_min=8000, tokens_max=16400, n=5, rmse_us=1),
                                                "nccl_allgather": dict(a_us=29, c_us_per_token=0.032, tokens_min=300, tokens_max=16400, n=5, rmse_us=1)},
                                         counts={"eager": {"twoshot": 81, "nccl_allgather": 5}, "ring": {"nccl_allreduce": 81, "nccl_allgather": 5}}, wait_ms={}))
    mixed = MixedModel.from_dict(dict(launch=dict(l0=1480, l1_per_P=0.0085, l2_per_extra_chunk=10, l3_per_kv16=82, rmse=5, n=10),
                                      c_host_ns=[120e3, 119e3, 130e3, 124e3], c_api_ns=14e3, outside_fwd_ns=[8e6] * R, tail_ns=3e6, c_kv_ns=5e3,
                                      c_kv_evidence={}, graph_table={"128,47": dict(n=3, wall_ns=36e6, busy_ns=[30e6] * R, fwd_ns=[32e6] * R, passes=[])},
                                      host_bound_ids=[], n_fit=10, ranges={"P": (300, 16384), "chunks": (1, 13), "kv16": (0, 7), "L": (1400, 2100)}))
    return prefill, coll, mixed


def test_host_bound_eager_iteration_follows_slowest_host_and_lead_reduces_it():
    prefill, coll, mixed = _mixed_models()
    f = _feat(b=8, stage="mixed", P=1000, chunks=[[0, 1000]])
    f.mixed_kind = mixed_kind(f.P, f.b)
    assert f.mixed_kind == "eager"
    p = mixed.predict(f, prefill, coll)
    L = 1480 + 0.0085 * 1000
    assert p["launches"] == pytest.approx(L)
    assert p["host_path_ns"] == pytest.approx(130e3 * L + 3e6)         # rank 2 (slowest host) + GPU tail
    assert p["bound"] == "host" and p["wall_ns"] == pytest.approx(p["host_path_ns"])
    assert p["gpu_path_ns"] < p["host_path_ns"]
    # the host had already run 50 ms ahead while the previous iteration was on the GPU
    q = mixed.predict(f, prefill, coll, lead_ns=50e6)
    assert q["wall_ns"] == pytest.approx(max(p["gpu_path_ns"], p["host_path_ns"] - 50e6))
    assert q["wall_standalone_ns"] == pytest.approx(p["wall_ns"])
    # host/GPU overlap: the host window is never added to the GPU path
    assert q["wall_ns"] < p["host_path_ns"] + p["gpu_path_ns"]


def test_gpu_bound_ring_iteration_uses_gpu_path_and_ring_collectives():
    prefill, coll, mixed = _mixed_models()
    f = _feat(b=4, stage="mixed", P=16384, chunks=[[0, 16384]])
    f.mixed_kind = mixed_kind(f.P, f.b)
    assert f.mixed_kind == "ring"
    p = mixed.predict(f, prefill, coll)
    comm_expected = (23 + 0.024 * 16388) * 81e3 + (29 + 0.032 * 16388) * 5e3
    assert p["comm_intrinsic_ns"] == pytest.approx(comm_expected)
    assert p["bound"] == "gpu" and p["wall_ns"] == pytest.approx(p["gpu_path_ns"])
    assert mixed_kind(128, 47) == "graph" and mixed_kind(7200, 11) == "eager" and mixed_kind(8262, 12) == "ring"


# ---------------------------------------------------------------- sparse indexer KV slope and out-of-range status
def test_indexer_kv_slope_and_out_of_range():
    prefill, coll, mixed = _mixed_models()
    f0 = _feat(b=4, stage="mixed", P=16384, chunks=[[0, 16384]])
    f1 = _feat(b=4, stage="mixed", P=16384, chunks=[[50000, 16384]])
    g0 = prefill.predict_groups(f0)
    g1 = prefill.predict_groups(f1)
    i = G.GROUP_INDEX["indexer"]
    assert g1[0, i] - g0[0, i] == pytest.approx(1e5 * 50.0)              # slope × 50k KV (KVW in thousands)
    for gname in ("fmha", "moe", "other", "sparse_meta"):
        assert g1[0, G.GROUP_INDEX[gname]] == pytest.approx(g0[0, G.GROUP_INDEX[gname]])   # only the indexer depends on KV
    assert prefill.status(f1) == "in-range"
    f2 = _feat(b=4, stage="mixed", P=16384, chunks=[[120000, 16384]])
    assert prefill.status(f2) == "extrapolation"
    assert chunk_predict(prefill.chunk, 50000, "wall") == (pytest.approx(220 + 0.25 * 50), "in-range")
    assert chunk_predict(prefill.chunk, 120000, "wall")[1] == "extrapolation"
    f3 = _feat(b=4, stage="mixed", P=16384, chunks=[[0, 8000], [40000, 8384]])
    assert prefill.status(f3) == "inferred-structure"                        # chunk-length × KV interaction (gap G1)


# ---------------------------------------------------------------- MoE routing uncertainty
def test_moe_statistical_model_range_and_padding():
    m = MoeModel()
    m.coef_ns = [1e6, 3e4, 1e4]
    m.fit_rows = [dict(b=1), dict(b=48)]
    m.E_cal = {8: dict(b=8, E_mean=40.0, E_p10=35.0, E_p50=40.0, E_p90=46.0), 16: dict(b=16, E_mean=70.0, E_p10=64.0, E_p50=70.0, E_p90=76.0)}
    m.E_held = {16: dict(b=16, E_mean=75.0, E_p10=70.0, E_p50=75.0, E_p90=80.0)}
    assert [padded_batch(b) for b in (1, 2, 3, 4, 5, 8, 9, 16, 17, 24, 25, 48)] == [1, 2, 4, 4, 8, 8, 16, 16, 24, 24, 32, 48]
    p9, p12 = m.predict(9), m.predict(12)
    assert p9["central_ns"] == p12["central_ns"] == pytest.approx(1e6 + 3e4 * 70 + 1e4 * 16)   # same graph bucket → same cost
    assert p9["low_ns"] <= p9["central_ns"] <= p9["high_ns"] and p9["high_ns"] - p9["low_ns"] == pytest.approx(3e4 * 12)
    assert p9["heldout_routing_central_ns"] > p9["central_ns"]                                  # distribution shift is reported, not hidden
    assert m.predict(5)["central_ns"] == pytest.approx(1e6 + 3e4 * 40 + 1e4 * 8)
    assert m.predict(60)["status"] == "extrapolation"


# ---------------------------------------------------------------- hold-out integrity and leakage prevention
def test_partition_freeze_and_leakage_guard(tmp_path):
    split = {"diverse_nsys_run2": {"exclusions": [7], "held_out_batch_sizes": [5], "held_out_chains": [], "held_out_kv0_band": [1, 2],
                                   "classes": {"decode nreq=5": {"calibration": [], "heldout": [1, 2], "excluded": []},
                                               "decode nreq=3": {"calibration": [3, 4], "heldout": [5], "excluded": [7]}}},
             "diverse_nsys_salvaged": {"exclusions": [], "held_out_batch_sizes": [], "held_out_chains": [], "held_out_kv0_band": [1, 2],
                                       "classes": {"x": {"calibration": [10], "heldout": [], "excluded": []}}}}
    routing = {"per_request": [dict(request=f"r{i}", tokens=(2 if i < 3 else 500), imb_mean=float(i)) for i in range(16)]}
    sp = tmp_path / "split.json"
    rp = tmp_path / "routing.json"
    sp.write_text(json.dumps(split))
    rp.write_text(json.dumps(routing))
    out = tmp_path / "partition"
    partition.materialize(str(sp), str(rp), str(out))
    part = partition.load(str(out))
    assert part.ids("diverse_nsys_run2", "calibration") == {3, 4}
    assert part.ids("diverse_nsys_run2", "heldout") == {1, 2, 5}
    assert part.role_of("diverse_nsys_run2", 7) == "excluded" and part.role_of("diverse_nsys_run2", 999) == "excluded"
    assert part.calibration_only("diverse_nsys_run2", [3, 4]) == [3, 4]
    with pytest.raises(partition.LeakageError):
        part.calibration_only("diverse_nsys_run2", [3, 5])                 # held-out id in a fit
    with pytest.raises(partition.LeakageError):
        part.calibration_only("diverse_nsys_run2", [7])                    # excluded id in a fit
    with pytest.raises(partition.LeakageError):
        part.calibration_only("diverse_nsys_salvaged", [10])               # cross-check pass never calibrates
    # MoE hold-out: 2-token requests excluded; 5 least + 5 most imbalanced of the rest
    assert len(part.part["moe_heldout_requests"]) == 10 and "r0" not in part.part["moe_heldout_requests"]
    assert part.part["moe_heldout_requests"][:5] == ["r3", "r4", "r5", "r6", "r7"]
    # any edit to the frozen partition is refused
    p = out / "partition.json"
    txt = p.read_text()
    assert '"1": "heldout"' in txt
    p.write_text(txt.replace('"1": "heldout"', '"1": "calibration"'))
    with pytest.raises(partition.LeakageError):
        partition.load(str(out))


def test_partition_rejects_duplicate_iteration(tmp_path):
    split = {"p": {"exclusions": [], "held_out_batch_sizes": [], "held_out_chains": [], "held_out_kv0_band": [0, 0],
                   "classes": {"a": {"calibration": [1], "heldout": [], "excluded": []}, "b": {"calibration": [], "heldout": [1], "excluded": []}}}}
    sp = tmp_path / "s.json"
    rp = tmp_path / "r.json"
    sp.write_text(json.dumps(split))
    rp.write_text(json.dumps({"per_request": []}))
    with pytest.raises(ValueError):
        partition.materialize(str(sp), str(rp), str(tmp_path / "out"))


# ---------------------------------------------------------------- deterministic reproduction from frozen inputs
def test_calibration_file_hashes_are_declared(tmp_path):
    from costmodel.calibrate import code_hashes
    h = code_hashes()
    assert {"decode.py", "mixed.py", "compose.py", "extract.py", "partition.py"} <= set(h)
    assert all(len(v) == 64 for v in h.values())
