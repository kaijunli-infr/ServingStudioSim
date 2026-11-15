"""Regression tests for the Phase 1C E1 completeness checker v2 (Phase 1.5 correction).

Each test names the v1 defect or the pre-registered rule it protects. The fixture builds a small but structurally
faithful profile directory: per-iteration engine records (schema 4 fields used by the checker), routed_experts/*.npy
with shape (OUT_LEN, L, K), and a token_corpus (routes.u16 = concatenated npy[1:], manifest, provenance).
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

E1 = Path(__file__).resolve().parents[1] / "profiling" / "env" / "arm64-gb300" / "dsv41-flash-tp4" / "phase1c" / "e1"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


v2 = _load("e1_completeness_v2", E1 / "checks" / "e1_completeness_v2.py")
v1 = _load("e1_completeness_v1", E1 / "checks" / "e1_completeness.py")

OUT_LEN, L, K, E = 8, 2, 2, 16
CELLS = [(2, 32768, 0), (3, 4096, 0)]  # (b, kv tokens, wave)


def _rid(b, kv, w, r):
    return f"independent_e1_b{b}_kv{kv // 1024}k_w{w}_r{r:02d}"


def build_records(cells=CELLS, out_len=OUT_LEN, scale=1.0):
    """Engine records: one prefill iteration per request (ramp), then pure decode, then a tail (ramp)."""
    recs = []
    it = 100
    for b, kv, w in cells:
        reqs = [_rid(b, kv, w, r) for r in range(b)]
        emitted = {r: 0 for r in reqs}
        active = []
        for r in reqs:  # prefill iterations: new request emits token 0; already active ones decode (ramp)
            prog = []
            for a in active:
                prog.append({"request_id": a, "output_tokens_before": emitted[a], "emitted_tokens": 1}); emitted[a] += 1
            prog.append({"request_id": r, "output_tokens_before": 0, "emitted_tokens": 1}); emitted[r] = 1
            active.append(r)
            recs.append({"schema_version": 4, "iteration_index": it, "observed_elapsed_ms": 50.0 * scale, "prefill_tokens": kv,
                         "decode_requests": len(active) - 1, "decode_request_progress": prog}); it += 1
        while any(emitted[r] < out_len for r in reqs):  # decode iterations
            live = [r for r in reqs if emitted[r] < out_len]
            prog = []
            for a in live:
                prog.append({"request_id": a, "output_tokens_before": emitted[a], "emitted_tokens": 1}); emitted[a] += 1
            recs.append({"schema_version": 4, "iteration_index": it, "observed_elapsed_ms": (4.0 + 0.5 * b) * scale, "prefill_tokens": 0,
                         "decode_requests": len(live), "decode_request_progress": prog}); it += 1
    return recs


def build_profile(root: Path, recs=None, routes=True, out_len=OUT_LEN, seed=0):
    root.mkdir(parents=True, exist_ok=True)
    recs = build_records(out_len=out_len) if recs is None else recs
    (root / "vllm").mkdir(exist_ok=True)
    with open(root / "vllm" / "x_metrics.jsonl", "w") as fh:
        for d in recs:
            fh.write(json.dumps(d) + "\n")
    reqs = sorted({p["request_id"] for d in recs for p in d["decode_request_progress"]})
    json.dump({"profile_kind": "workload_metrics", "request_timing_count": len(reqs)}, open(root / "profile_result.json", "w"))
    if routes:
        rng = np.random.default_rng(seed)
        rd = root / "routed_experts"; rd.mkdir(exist_ok=True)
        arrays = {}
        for r in reqs:
            a = np.stack([np.stack([rng.choice(E, K, replace=False) for _ in range(L)]) for _ in range(out_len)]).astype(np.uint16)
            np.save(rd / f"{r}.npy", a); arrays[r] = a
        corpus = root / "token_corpus"; corpus.mkdir(exist_ok=True)
        segs, rows, start = [], [], 0
        for r in reqs:
            seg = arrays[r][1:]
            segs.append({"request": r, "start": start, "length": seg.shape[0]}); start += seg.shape[0]; rows.append(seg)
        data = np.concatenate(rows).astype("<u2")
        data.tofile(corpus / "routes.u16")
        json.dump({"schema_version": 1, "data_file": "routes.u16", "num_tokens": int(data.shape[0]), "num_layers": L, "num_experts": E, "top_k": K,
                   "checksum_fnv1a64": v2.fnv1a64(data.tobytes())}, open(corpus / "manifest.json", "w"))
        json.dump({"scope": "accepted generated tokens; no prompt", "request_segments": segs}, open(corpus / "provenance.json", "w"))
    return root


def inputs(tmp_path, cells=CELLS, mins=None):
    mins = mins or {(2, 32768, 0): 6, (3, 4096, 0): 5}
    d = {"design": {"OUT_LEN": OUT_LEN, "MIN_PURE_DECODE": 400}, "main": {"plan": []},
         "dry": {"plan": [{"b": b, "kv": kv, "wave": w, "pure_decode_iters_min": mins[(b, kv, w)]} for b, kv, w in cells]}}
    p = tmp_path / "E1-INPUTS.json"; json.dump(d, open(p, "w")); return str(p)


def msgs(rep, status=None):
    return [m for s, m in rep.checks if status is None or s == status]


# ---------------------------------------------------------------- (a) KV tag units
def test_tag_parses_kv_in_units_of_1024_tokens():
    assert v2.parse_tag("independent_e1_b8_kv32k_w0_r01") == (8, 32768, 0, 1)
    assert v2.parse_tag("e1_b48_kv4k_w1_r07") == (48, 4096, 1, 7)
    assert v2.parse_tag("e1_b1_kv96k_w0_r00") == (1, 98304, 0, 0)
    assert v2.parse_tag("independent_07_quadrant_interference_c48-0001") is None


def test_v1_regression_cell_keys_did_not_match_plan_keys():
    """Documents the v1 defect: cell_counts keyed cells by the raw tag number (32) while the plan uses tokens (32768)."""
    recs = build_records()
    pure_v1, *_ = v1.cell_counts(recs, OUT_LEN)
    assert (2, 32, 0) in pure_v1 and (2, 32768, 0) not in pure_v1
    acc = v2.account(recs)
    assert (2, 32768, 0) in acc.pure and (2, 32, 0) not in acc.pure


# ---------------------------------------------------------------- (b) per-cell minimum
def test_clean_fixture_passes(tmp_path):
    P = build_profile(tmp_path / "p")
    rep = v2.run(str(P), inputs(tmp_path))
    assert rep.fails == 0, msgs(rep, "FAIL")
    assert any("cell b=2 kv=32768 wave=0: pure-decode iterations 6 (need >= 6)" in m for m in msgs(rep, "PASS"))
    assert any("cell b=3 kv=4096 wave=0: pure-decode iterations 5 (need >= 5)" in m for m in msgs(rep, "PASS"))


def test_per_cell_minimum_fails_when_short(tmp_path):
    P = build_profile(tmp_path / "p")
    rep = v2.run(str(P), inputs(tmp_path, mins={(2, 32768, 0): 7, (3, 4096, 0): 5}))
    assert any("cell b=2 kv=32768 wave=0: pure-decode iterations 6 (need >= 7)" in m for m in msgs(rep, "FAIL"))


def test_planned_cell_absent_is_a_failure(tmp_path):
    P = build_profile(tmp_path / "p", recs=build_records(cells=[CELLS[0]]))
    rep = v2.run(str(P), inputs(tmp_path))
    assert any("cell b=3 kv=4096 wave=0: pure-decode iterations 0" in m for m in msgs(rep, "FAIL"))


def test_plan_selection_by_request_count_and_explicit_choice(tmp_path):
    # two plans: dry = CELLS (2 + 3 = 5 requests), main = one extra cell; auto must pick dry from the request count and
    # must not demand main's cells; --plan main must flag dry-only runs as incomplete
    P = build_profile(tmp_path / "p")
    d = json.load(open(inputs(tmp_path)))
    d["dry"]["requests"] = 5
    d["main"] = {"requests": 6, "plan": d["dry"]["plan"] + [{"b": 1, "kv": 98304, "wave": 0, "pure_decode_iters_min": 5}]}
    p = tmp_path / "inputs2.json"; json.dump(d, open(p, "w"))
    rep = v2.run(str(P), str(p))
    assert rep.data["plan"] == "dry" and not msgs(rep, "FAIL")
    rep = v2.run(str(P), str(p), plan="main")
    assert rep.data["plan"] == "main"
    assert any("cell b=1 kv=98304 wave=0: pure-decode iterations 0" in m for m in msgs(rep, "FAIL"))


def test_unplanned_cell_is_a_failure(tmp_path):
    P = build_profile(tmp_path / "p", recs=build_records(cells=CELLS + [(1, 98304, 0)]))
    rep = v2.run(str(P), inputs(tmp_path))
    assert any("observed cells not in the plan: [(1, 98304, 0)]" in m for m in msgs(rep, "FAIL"))


# ---------------------------------------------------------------- (c) ramp vs pure accounting
def test_ramp_tokens_are_accounted_and_do_not_fail_the_join():
    recs = build_records()
    acc = v2.account(recs)
    r0 = _rid(2, 32768, 0, 0)
    ramp_positions = [pos for pos, pure in acc.pos_pure[r0].items() if 1 <= pos <= OUT_LEN - 1 and not pure]
    pure_positions = [pos for pos, pure in acc.pos_pure[r0].items() if 1 <= pos <= OUT_LEN - 1 and pure]
    assert ramp_positions == [1] and pure_positions == [2, 3, 4, 5, 6, 7]
    # v1's join sub-check demanded pure-decode count == corpus rows (7) and would therefore fail on this correct data
    per_req_pure_v1 = sum(1 for d in recs if d["prefill_tokens"] == 0 for p in d["decode_request_progress"]
                          if p["request_id"] == r0 and 1 <= p["output_tokens_before"] <= OUT_LEN - 1 and p["emitted_tokens"] == 1)
    assert per_req_pure_v1 == 6 != OUT_LEN - 1


def test_token_accounting_reports_ramp_statistics(tmp_path):
    P = build_profile(tmp_path / "p")
    rep = v2.run(str(P), inputs(tmp_path))
    assert rep.data["token_accounting"]["ramp_emitted_min"] >= 1
    assert any("emitted in pure + ramp iterations == 7: all" in m for m in msgs(rep, "PASS"))


# ---------------------------------------------------------------- (d) join uniqueness
def test_duplicate_emission_is_a_duplicate_join(tmp_path):
    recs = build_records()
    r0 = _rid(2, 32768, 0, 0)
    dup = next(d for d in recs if d["prefill_tokens"] == 0 and any(p["request_id"] == r0 and p["output_tokens_before"] == 3 for p in d["decode_request_progress"]))
    recs.append({**dup, "iteration_index": 999})  # same position emitted again in another iteration
    P = build_profile(tmp_path / "p", recs=recs)
    rep = v2.run(str(P), inputs(tmp_path))
    assert any("duplicate join): 2" in m for m in msgs(rep, "FAIL"))  # both requests of the cell were in that iteration


def test_missing_emission_is_a_missing_join(tmp_path):
    recs = build_records()
    r0 = _rid(2, 32768, 0, 0)
    for d in recs:
        d["decode_request_progress"] = [p for p in d["decode_request_progress"] if not (p["request_id"] == r0 and p["output_tokens_before"] == 4)]
    P = build_profile(tmp_path / "p", recs=recs)
    rep = v2.run(str(P), inputs(tmp_path))
    assert any("missing join): 1" in m for m in msgs(rep, "FAIL"))


def test_untagged_request_id_is_malformed(tmp_path):
    recs = build_records()
    recs[5]["decode_request_progress"].append({"request_id": "independent_warmup-0001", "output_tokens_before": 0, "emitted_tokens": 1})
    P = build_profile(tmp_path / "p", recs=recs, routes=False)
    rep = v2.run(str(P), inputs(tmp_path))
    assert any("request ids without an E1 tag: 1" in m for m in msgs(rep, "FAIL"))


# ---------------------------------------------------------------- (e) routing records
def test_missing_npy_fails(tmp_path):
    P = build_profile(tmp_path / "p")
    os.remove(P / "routed_experts" / f"{_rid(2, 32768, 0, 1)}.npy")
    rep = v2.run(str(P), inputs(tmp_path))
    assert any("routed_experts files 4 vs requests 5" in m for m in msgs(rep, "FAIL"))


def test_malformed_npy_shape_and_dtype_fail(tmp_path):
    P = build_profile(tmp_path / "p")
    r = _rid(3, 4096, 0, 2)
    np.save(P / "routed_experts" / f"{r}.npy", np.zeros((OUT_LEN - 1, L, K), dtype=np.uint16))
    rep = v2.run(str(P), inputs(tmp_path))
    assert any("shape != (8, 2, 2): 1" in m for m in msgs(rep, "FAIL"))
    r2 = _rid(3, 4096, 0, 1)
    a = np.load(P / "routed_experts" / f"{r2}.npy").astype(np.int32)
    np.save(P / "routed_experts" / f"{r2}.npy", a)
    rep = v2.run(str(P), inputs(tmp_path))
    assert any("dtype != uint16: 1" in m for m in msgs(rep, "FAIL"))


def test_duplicate_expert_in_topk_and_out_of_range_id_fail(tmp_path):
    P = build_profile(tmp_path / "p")
    r = _rid(2, 32768, 0, 0)
    a = np.load(P / "routed_experts" / f"{r}.npy")
    a[3, 1, :] = 5  # duplicate expert inside one (token, layer) top-k
    a[4, 0, 0] = E  # expert id out of range
    np.save(P / "routed_experts" / f"{r}.npy", a)
    rep = v2.run(str(P), inputs(tmp_path))
    assert any("duplicate expert in the top-2: 1" in m for m in msgs(rep, "FAIL"))
    assert any("expert id >= 16: 1" in m for m in msgs(rep, "FAIL"))
    assert any("corpus segment != npy[1:]" in m and ": 1" in m for m in msgs(rep, "FAIL"))


def test_overlapping_segments_and_size_mismatch_fail(tmp_path):
    P = build_profile(tmp_path / "p")
    prov = json.load(open(P / "token_corpus" / "provenance.json"))
    prov["request_segments"][1]["start"] -= 1
    json.dump(prov, open(P / "token_corpus" / "provenance.json", "w"))
    rep = v2.run(str(P), inputs(tmp_path))
    assert any("contiguous, non-overlapping" in m for m in msgs(rep, "FAIL"))
    with open(P / "token_corpus" / "routes.u16", "ab") as fh:
        fh.write(b"\x00\x00")
    rep = v2.run(str(P), inputs(tmp_path))
    assert any("corpus file size" in m for m in msgs(rep, "FAIL"))


def test_segment_name_mismatch_fails(tmp_path):
    P = build_profile(tmp_path / "p")
    prov = json.load(open(P / "token_corpus" / "provenance.json"))
    prov["request_segments"][0]["request"] = "independent_e1_b2_kv32k_w0_r09"
    json.dump(prov, open(P / "token_corpus" / "provenance.json", "w"))
    rep = v2.run(str(P), inputs(tmp_path))
    assert any("segment names == metrics request ids" in m for m in msgs(rep, "FAIL"))


def test_checksum_verified_when_requested(tmp_path):
    P = build_profile(tmp_path / "p")
    rep = v2.run(str(P), inputs(tmp_path), checksum=True)
    assert any("checksum_fnv1a64" in m for m in msgs(rep, "PASS"))


# ---------------------------------------------------------------- overhead gate
def test_overhead_point_gate_preserved_and_interval_reported(tmp_path):
    off = build_profile(tmp_path / "off", routes=False)
    on_fail = build_profile(tmp_path / "on_fail", recs=build_records(scale=1.03), routes=False)
    rep = v2.run(str(on_fail), inputs(tmp_path), compare=str(off), draws=200)
    assert any("overhead b2_kv32768_w0: median iteration ratio this/other = 1.0300" in m for m in msgs(rep, "FAIL"))
    on_ok = build_profile(tmp_path / "on_ok", recs=build_records(scale=1.005), routes=False)
    rep = v2.run(str(on_ok), inputs(tmp_path), compare=str(off), draws=200)
    assert any("ratio this/other = 1.0050" in m for m in msgs(rep, "PASS"))
    assert any("prospective criterion" in m for m in msgs(rep))
    assert rep.data["overhead"]["b2_kv32768_w0"]["block_bootstrap_ci95"]
