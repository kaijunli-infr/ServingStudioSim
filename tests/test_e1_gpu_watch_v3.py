"""V3 GPU sampler: the V2 sampler's observation and ending rule, plus an end through a stop file when the GPUs were never held."""

from __future__ import annotations

# ruff: noqa: E501

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKS = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/checks"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, CHECKS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _census(states):
    it = iter(states)
    t = [1000.0]

    def census(max_used_mib, registry):
        s = next(it, "released")
        t[0] += 2.2
        held = s == "held"
        return {"at": t[0], "at_iso": "x", "host": "h", "gpus": [{"memory_used_mib": 260_000 if held else 0}] * 4, "compute_apps": [{"pid": 1}] if held else [], "verdict": {"released": s == "released", "available": s != "unavailable"}}

    return census


def _rows(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def test_same_record_and_ending_rule_as_the_v2_sampler(tmp_path):
    v3, v2 = _load("e1_gpu_watch_v3"), _load("e1_gpu_watch")
    out = tmp_path / "g.jsonl"
    rc = v3.main(["--out", str(out), "--interval", "0", "--stop-after-released", "3", "--max-s", "60"], census=_census(["released", "held", "held", "released", "released", "released", "released"]), sleep=lambda s: None)
    rows = _rows(out)
    assert rc == 0 and [r["released"] for r in rows] == [True, False, False, True, True, True]  # idle before the load does not count as a release
    assert all(r["source"] == "e1_gpu_watch" for r in rows) and set(rows[0]) == set(v2.observation(_census(["held"])(64, None), "e1_gpu_watch"))


def test_never_held_run_ends_through_the_stop_file_after_the_tail_samples(tmp_path):
    v3 = _load("e1_gpu_watch_v3")
    out, stop = tmp_path / "g.jsonl", tmp_path / "stop"
    n = [0]

    def sleep(_s):
        n[0] += 1
        if n[0] == 5:
            stop.write_text("")

    rc = v3.main(["--out", str(out), "--interval", "0", "--stop-after-released", "30", "--max-s", "600", "--stop-file", str(stop), "--after-stop-samples", "4"], census=_census([]), sleep=sleep)
    assert rc == 0 and len(_rows(out)) == 5 + 5  # five before the stop file, then four tail samples and the one that ends it


def test_stop_file_does_not_cut_the_release_tail_short_before_the_tail_samples(tmp_path):
    v3 = _load("e1_gpu_watch_v3")
    out, stop = tmp_path / "g.jsonl", tmp_path / "stop"
    stop.write_text("")
    rc = v3.main(["--out", str(out), "--interval", "0", "--stop-after-released", "30", "--max-s", "600", "--stop-file", str(stop), "--after-stop-samples", "10"], census=_census(["held"] * 3), sleep=lambda s: None)
    rows = _rows(out)
    assert rc == 0 and len(rows) == 11 and [r["released"] for r in rows[:4]] == [False, False, False, True]
