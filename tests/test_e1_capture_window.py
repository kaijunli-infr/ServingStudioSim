"""CPU regression tests for the E1 nsys capture-window coverage (phase1c/e1/analysis/e1_capture_window.py).

Background: the Stage 1.5 `e1_nsys_val` pass armed the profiler 10 s after the replay started; the b=8 wave of the dry trace
(arrival 0, done by ~9 s) was entirely outside the window and only the b=48 wave was captured. These tests pin the timing
model (window relative to replay start, iteration captured iff fully inside, straddling iterations flagged) on a synthetic
two-wave timeline with delayed-start boundary cases, and reproduce the real outcome when the Stage 1.5 run root is present.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
E1 = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1"
RUN_ROOT = Path("/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


cw = _load("e1_capture_window", E1 / "analysis" / "e1_capture_window.py")


def _rec(index, start_s, dur_ms, decode, prefill, ids):
    return {"iteration_index": index, "observed_start_monotonic_ns": int(1e9 + start_s * 1e9),
            "observed_end_monotonic_ns": int(1e9 + start_s * 1e9 + dur_ms * 1e6), "observed_elapsed_ms": dur_ms,
            "decode_requests": decode, "prefill_tokens": prefill,
            "decode_request_progress": [{"request_id": f"independent_{i}", "output_tokens_before": 0, "emitted_tokens": 1} for i in ids]}


def synthetic_wave(b: int, t0: float, start_index: int, kv="32k", n_pure=10, pre_ms=200.0, dec_ms=5.0):
    """Shape of a real wave: 2 prefill-only chunks, then (prefill chunk, decode ramp) pairs, then n_pure pure iterations."""
    ids = [f"e1_b{b}_kv{kv}_w0_r{r:02d}" for r in range(b)]
    recs, t, k = [], t0, start_index
    for _ in range(2):
        recs.append(_rec(k, t, pre_ms, 0, 16384, [])); t += pre_ms / 1e3; k += 1
    for r in range(1, b):
        recs.append(_rec(k, t, pre_ms, r, 16384, ids[:r])); t += pre_ms / 1e3; k += 1      # mixed
        recs.append(_rec(k, t, dec_ms, r, 0, ids[:r])); t += dec_ms / 1e3; k += 1           # decode ramp
    for _ in range(n_pure):
        recs.append(_rec(k, t, dec_ms, b, 0, ids)); t += dec_ms / 1e3; k += 1
    return recs, t


def two_wave_timeline():
    """b=8 wave at 0 s, idle gap, b=48 wave later (the dry trace shape); returns (iterations, bounds)."""
    r8, end8 = synthetic_wave(8, 0.033, 100, n_pure=20)
    r48, end48 = synthetic_wave(48, end8 + 5.0, 1000, n_pure=30)
    its = [cw.iteration_from_record(r, int(1e9)) for r in r8 + r48]
    return its, {"end8": end8, "start48": end8 + 5.0, "end48": end48}


def test_kinds_and_wave_bounds():
    its, b = two_wave_timeline()
    kinds = {}
    for it in its:
        kinds[it.kind] = kinds.get(it.kind, 0) + 1
    assert kinds["pure_b8"] == 20 and kinds["pure_b48"] == 30
    assert kinds["prefill_only"] == 4 and kinds["mixed_b8"] == 7 and kinds["decode_ramp_b8"] == 7
    w8 = cw.wave_bounds(its, 8)
    assert w8["first_iteration"] == 100 and w8["n_pure"] == 20 and w8["first_pure_iteration"] == 100 + 2 + 14
    assert w8["n_owned"] == 2 + 14 + 20
    assert cw.wave_bounds(its, 48)["first_start_s"] == pytest.approx(b["start48"])


def test_no_delay_captures_everything_without_straddling():
    its, _ = two_wave_timeline()
    s, e = cw.window(None, 60.0)
    cov = cw.coverage(its, s, e)
    assert cov["n_captured"] == len(its) and cov["n_straddling"] == 0
    assert cw.cell_gate(cov, 8, 20)[0] and cw.cell_gate(cov, 48, 30)[0]


def test_delay_in_idle_gap_misses_the_first_wave_entirely():
    """The e1_nsys_val failure mode: the window starts after the b=8 wave has finished."""
    its, b = two_wave_timeline()
    s, e = cw.window(b["end8"] + 1.0, 60.0)
    cov = cw.coverage(its, s, e)
    assert cov["captured"].get("pure_b8", 0) == 0 and cov["outside"]["pure_b8"] == 20
    assert cov["captured"]["pure_b48"] == 30 and cov["n_straddling"] == 0
    ok, msg = cw.cell_gate(cov, 8, 20)
    assert not ok and "captured 0 of 20" in msg


def test_delay_inside_prefill_straddles_one_iteration_but_keeps_all_pure():
    its, _ = two_wave_timeline()
    mixed = [it for it in its if it.kind == "mixed_b8"][0]
    s, e = cw.window((mixed.start_s + mixed.end_s) / 2, 60.0)
    cov = cw.coverage(its, s, e)
    assert cov["n_straddling"] == 1 and cov["straddling_indices"] == [mixed.index]
    assert cov["captured"]["pure_b8"] == 20
    assert not cw.cell_gate(cov, 8, 20)[0], "a straddling iteration would miss NVTX ranges: gate must fail"


def test_boundaries_are_inclusive_at_iteration_edges():
    its, _ = two_wave_timeline()
    first_pure = [it for it in its if it.kind == "pure_b8"][0]
    prev = [it for it in its if it.index == first_pure.index - 1][0]
    # window starting exactly at the first pure iteration's start: that iteration is captured, the previous one is outside
    cov = cw.coverage(its, first_pure.start_s, float("inf"))
    assert cov["captured"]["pure_b8"] == 20 and cov["n_straddling"] == 0 and cov["first_captured"] == first_pure.index
    assert prev.index not in cov["straddling_indices"]
    # window starting one microsecond later: the first pure iteration straddles and is lost
    cov = cw.coverage(its, first_pure.start_s + 1e-6, float("inf"))
    assert cov["captured"]["pure_b8"] == 19 and cov["straddling_indices"] == [first_pure.index]
    assert not cw.cell_gate(cov, 8, 19)[0]


def test_bounded_duration_cuts_the_tail():
    its, _ = two_wave_timeline()
    pure = [it for it in its if it.kind == "pure_b8"]
    s, e = cw.window(0.0, 60.0, duration_s=pure[9].end_s)          # ends exactly at the end of the 10th pure iteration
    cov = cw.coverage(its, s, e)
    assert cov["captured"]["pure_b8"] == 10 and cov["n_straddling"] == 0 and cov["last_captured"] == pure[9].index
    s, e = cw.window(0.0, 60.0, duration_s=pure[9].end_s - 1e-6)   # one microsecond earlier: the 10th straddles
    cov = cw.coverage(its, s, e)
    assert cov["captured"]["pure_b8"] == 9 and cov["straddling_indices"] == [pure[9].index]


def test_delay_after_the_wave_captures_nothing_and_gate_fails_on_empty():
    its, b = two_wave_timeline()
    cov = cw.coverage(its, b["end48"] + 1.0, float("inf"))
    assert cov["n_captured"] == 0
    assert not cw.cell_gate(cov, 8, 1)[0]


def test_overrun_and_untagged_kinds():
    r = _rec(5, 1.0, 5.0, 10, 0, ["e1_b8_kv32k_w0_r00", "e1_b48_kv32k_w0_r00"])
    assert cw.iteration_from_record(r, int(1e9)).kind == "overrun"
    r = _rec(6, 1.0, 5.0, 3, 0, ["foo", "bar"])
    assert cw.iteration_from_record(r, int(1e9)).kind == "decode_ramp_untagged"


def test_simulation_cli_on_synthetic_run(tmp_path):
    r8, end8 = synthetic_wave(8, 0.033, 100, n_pure=20)
    d = tmp_path / "pass_x" / "profile" / "vllm"; d.mkdir(parents=True)
    (d / "x_drive_summary.json").write_text(json.dumps({"replay_start_monotonic_ns": int(1e9), "replay_end_monotonic_ns": int(1e9 + (end8 + 0.5) * 1e9)}))
    (d / "x_metrics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in r8))
    out = tmp_path / "sim.json"
    assert cw.main(["simulate", "--run-root", str(tmp_path), "--passes", "pass_x", "--delays", f"none,0,{end8 + 1}", "--cell", "8", "--need", "20", "--json", str(out)]) == 0
    rows = json.load(open(out))["passes"]["pass_x"]["rows"]
    assert [r["gate"] for r in rows] == ["PASS", "PASS", "FAIL"]
    c = tmp_path / "census.json"
    assert cw.main(["census", "--run", str(tmp_path / "pass_x"), "--json", str(c)]) == 0
    assert json.load(open(c))["waves"]["b8"]["n_pure"] == 20


@pytest.mark.skipif(not (RUN_ROOT / "e1_nsys_val" / "profile" / "parsed.json").exists(), reason="Stage 1.5 run root not present")
def test_real_stage15_passes_reproduce_the_nsys_val_outcome():
    """With the real timelines: the 10 s delay captured 0 pure b=8 / 928 pure b=48; no delay would capture all 1008 b=8."""
    for p in ("e1_dry_off_v2", "e1_dry_on_v2", "e1_dry_off2_v2", "e1_nsys_val"):
        pt = cw.load_pass(str(RUN_ROOT / p))
        w8 = cw.wave_bounds(pt.iterations, 8)
        assert w8["n_pure"] == 1008 and 3.5 < w8["first_pure_start_s"] < 4.0 and w8["wave_end_s"] < 9.5, (p, w8)
        cov10 = cw.coverage(pt.iterations, *cw.window(10.0, pt.replay_span_s))
        assert cov10["captured"].get("pure_b8", 0) == 0 and cov10["captured"]["pure_b48"] == 928 and cov10["n_straddling"] == 0
        cov0 = cw.coverage(pt.iterations, *cw.window(None, pt.replay_span_s))
        assert cov0["captured"]["pure_b8"] == 1008 and cov0["n_straddling"] == 0
        cov2 = cw.coverage(pt.iterations, *cw.window(2.0, pt.replay_span_s))
        assert cov2["captured"]["pure_b8"] == 1008, "a 2 s delay keeps every pure b=8 iteration (but straddles a prefill chunk)"
    c = cw.census(str(RUN_ROOT / "e1_nsys_val"))
    assert c["parsed"]["iterations"] == 1120 and c["parsed"]["kinds"]["pure_b48"] == 928 and c["parsed"]["kinds"].get("pure_b8", 0) == 0
