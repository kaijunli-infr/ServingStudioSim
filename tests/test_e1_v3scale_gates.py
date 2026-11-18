"""CPU tests for the prospective V3 gates (checks/e1_v3scale_gates.py): the rehearsal's fixture matrix in process (positive,
boundary and one registered defect per negative fixture), the sampled-release interval, sampler coverage, and the command line."""

from __future__ import annotations

# ruff: noqa: E501

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
E1 = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load(E1 / "e1_v3scale_rehearsal.py")
G = _load(E1 / "checks" / "e1_v3scale_gates.py")
FIXTURES = R.fixtures()


def _evaluate(a: dict) -> dict:
    res = G.evaluate(a["run"], json.loads(a["inputs"].read_text()), pin=G.load(a["pin"]), provenance=G.load(a["provenance"]), binary_now=G._mod("e1_frontend_binary_gate").sha256_file(a["binary"]),
                     importer=a["importer"] and str(a["importer"]), nsys=a["nsys"] and str(a["nsys"]), scratch=a["scratch"], durable_stream=a["durable"], identity=a["identity"], post_storage=a["post_storage"],
                     post_memory=a["post_memory"], reservation_deleted=a["deleted"], extras=a["extras"])
    return {"verdict": G.verdict_of(res), "gates": res}


def test_fixture_matrix_is_what_the_packet_says():
    kinds = [f.kind for f in FIXTURES]
    assert len(FIXTURES) == 87 and kinds.count("positive") == 3 and kinds.count("boundary") == 3 and kinds.count("negative") == 81
    assert len({f.name for f in FIXTURES}) == len(FIXTURES)
    assert all(f.verdict == "PASS" and not f.not_pass for f in FIXTURES if f.kind != "negative")
    assert all(f.verdict in ("FAIL", "INCONCLUSIVE") and f.not_pass for f in FIXTURES if f.kind == "negative")
    for f in FIXTURES:  # a FAIL verdict needs a failing gate; an INCONCLUSIVE one must have none
        assert ("FAIL" in f.not_pass.values()) == (f.verdict == "FAIL"), f.name
    touched = {g for f in FIXTURES for g in f.not_pass}
    assert touched == set(R.ALL)  # every gate is made to not pass by at least one fixture


@pytest.mark.parametrize("fx", FIXTURES, ids=[f.name for f in FIXTURES])
def test_fixture(fx, tmp_path):
    report = _evaluate(R.build(fx, tmp_path))
    assert R.judge(report, fx.verdict, fx.not_pass) == []


def test_importer_states_are_distinguished(tmp_path):
    want = {"nominal": ("PASS", {"verified_activity": True, "completed": True}), "importer_temporarily_inactive": ("PASS", {"temporarily_inactive_intervals": 2}), "importer_present_never_active": ("INCONCLUSIVE", {"identified": True, "verified_activity": False}),
            "importer_not_found": ("INCONCLUSIVE", {"identified": False}), "importer_ambiguous": ("INCONCLUSIVE", {"ambiguous": True}), "importer_terminated": ("FAIL", {"terminated": True, "completed": False})}
    for name, (status, fields) in want.items():
        fx = next(f for f in FIXTURES if f.name == name)
        root = tmp_path / name
        root.mkdir()
        g5 = next(g for g in _evaluate(R.build(fx, root))["gates"] if g["gate"] == "G5.importer_observed")
        assert g5["status"] == status and all(g5["data"]["summary"][k] == v for k, v in fields.items()), name


def _obs(rows):
    return [{"at": at, "available": state != "unavailable", "released": state == "released"} for at, state in rows]


def test_release_interval_uses_reading_windows_not_stamps():
    """The V2 numbers: held stamped 1.159 s before the SIGTERM, released stamped 1.018 s after it, next stamp 3.100 s later."""
    sig = 1000.0
    obs = _obs([(sig - 3.335, "held"), (sig - 1.159, "held"), (sig + 1.018, "released"), (sig + 4.118, "released"), (sig + 6.3, "released")])
    ri = G.release_interval(obs, sig, 2.0, 10.0)
    assert ri["last_held_window"] == [-1.159, -0.982] and ri["first_released_window"] == [1.018, 2.118] and ri["upper_edge_s"] == 2.118 and ri["problem"] is None and ri["held_again"] is False


def test_release_interval_problems():
    sig = 1000.0
    assert G.release_interval(_obs([(sig - 2, "held"), (sig + 1, "released")]), None, 2.0, 10.0)["problem"] == "the server SIGTERM time is not recorded"
    assert G.release_interval(_obs([(sig + 1, "released")]), sig, 2.0, 10.0)["problem"] == "no 'held' observation"
    assert "no 'released'" in G.release_interval(_obs([(sig - 2, "held"), (sig + 1, "held")]), sig, 2.0, 10.0)["problem"]
    assert "samples missing" in G.release_interval(_obs([(sig - 6, "held"), (sig + 6, "released"), (sig + 8.2, "released")]), sig, 2.0, 10.0)["problem"]
    assert "samples missing" in G.release_interval(_obs([(sig - 2, "held"), (sig + 0.2, "unavailable"), (sig + 2.4, "released"), (sig + 4.6, "released")]), sig, 2.0, 10.0)["problem"]
    again = G.release_interval(_obs([(sig - 2, "held"), (sig + 0.2, "released"), (sig + 2.4, "held"), (sig + 4.6, "released"), (sig + 6.8, "released")]), sig, 2.0, 10.0)
    assert again["held_again"] is True
    idle_first = G.release_interval(_obs([(sig - 900, "released"), (sig - 2, "held"), (sig + 0.2, "released"), (sig + 2.4, "released")]), sig, 2.0, 10.0)
    assert idle_first["held_again"] is False and idle_first["problem"] is None  # idle before the model load is not a release


def test_last_observation_window_uses_the_largest_gap():
    obs = _obs([(0.0, "held"), (3.1, "held"), (5.3, "released")])
    win = G.reading_windows(obs, 2.0)
    assert [w[0] for w in win] == [0.0, 3.1, 5.3] and [w[1] for w in win] == pytest.approx([1.1, 3.3, 6.4])


def test_coverage_reports_why_a_peak_is_unmeasured():
    rows = [{"at": float(t)} for t in range(0, 100, 2)]
    assert G.coverage(rows, 0.0, 98.0, 30) is None
    assert G.coverage(None, 0.0, 98.0, 30) == "no samples" and G.coverage([], 0.0, 98.0, 30) == "no samples"
    assert G.coverage(rows, None, 98.0, 30) == "the interval to cover is not known"
    assert "after the start" in G.coverage(rows, -40.0, 98.0, 30)
    assert "before the end" in G.coverage(rows, 0.0, 140.0, 30)
    assert "largest gap" in G.coverage([r for r in rows if not 20 <= r["at"] <= 60], 0.0, 98.0, 30)
    assert G.coverage([r for r in rows if not 20 < r["at"] < 50], 0.0, 98.0, 30) is None  # a 30 s gap is at the limit


@pytest.mark.parametrize("name, rc", [("nominal", 0), ("flush_over_deadline", 1), ("postrun_storage_unmeasured", 3)])
def test_command_line_exit_codes_and_report(tmp_path, name, rc):
    fx = next(f for f in FIXTURES if f.name == name)
    a = R.build(fx, tmp_path)
    out = tmp_path / "gates.json"
    p = subprocess.run(R.cli_args(a, out), capture_output=True, text=True)
    report = json.loads(out.read_text())
    assert p.returncode == rc and report["verdict"] == fx.verdict and R.judge(report, fx.verdict, fx.not_pass) == []
    assert f"→ {fx.verdict}" in p.stdout.splitlines()[-1]


def test_missing_lifecycle_record_fails(tmp_path):
    a = R.build(FIXTURES[0], tmp_path)
    (a["run"] / "profile" / "nsys" / "lifecycle.json").unlink()
    report = _evaluate(a)
    assert report["verdict"] == "FAIL" and len(report["gates"]) == 1


def test_registered_inputs_carry_every_limit_the_gates_read():
    d = json.loads((E1 / "E1-V3-SCALE-INPUTS.json").read_text())
    assert set(d["phase_deadlines_s"]) == {"stop_accepted", "flush_stop_to_importing", "import_importer_first_seen_to_gone", "stop_to_report_written", "session_close_after_server_sigterm"}
    assert {"gpu_release_deadline_s", "gpu_watch_interval_s", "gpu_watch_max_gap_s", "sampler_max_gap_s"} <= set(d["limits"])
    assert {"reservation_hard_min", "gpu_held_ceiling_s", "temporary_storage_gb", "persistent_storage_gb", "host_mem_available_floor_gb", "importer_rss_ceiling_gb"} <= set(d["ceilings"])
    assert d["expected"]["iterations"] == 4320 and d["trace"]["requests"] == 112 and len(d["dry"]["plan"]) == 4
