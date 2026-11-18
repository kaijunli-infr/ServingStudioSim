"""The final V2 gate rehearsal (e1_fr_rehearsal.py) as a test: the synthetic positive fixtures pass every gate, every
negative fixture stops the validation at its registered gate, and the V1-fixture expectation is strict."""

from __future__ import annotations

# ruff: noqa: E501

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
E1 = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1"


def _load():
    spec = importlib.util.spec_from_file_location("e1_fr_rehearsal", E1 / "e1_fr_rehearsal.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e1_fr_rehearsal"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_synthetic_matrix_meets_every_expectation(tmp_path, capsys):
    mod = _load()
    assert mod.main(["synthetic", "--out", str(tmp_path / "out")]) == 0
    assert "REHEARSAL EXPECTATIONS MET" in capsys.readouterr().out
    summary = json.loads((tmp_path / "out" / "SUMMARY.json").read_text())
    rows = summary["fixtures"]
    assert summary["ok"] is True and all(r["ok"] for r in rows)
    for category in mod.REQUIRED:
        assert summary["required_categories"][category], category
    positive = [r for r in rows if r["category"] == "positive"]
    assert len(positive) >= 2 and all(
        r["verdict"] == "PASS" and r["not_pass"] == {} and r["gates_evaluated"] == 20
        for r in positive
    )
    negative = [r for r in rows if r["category"] != "positive"]
    assert len(negative) >= 40
    assert all(r["verdict"] in ("FAIL", "INCONCLUSIVE") and r["exit_code"] != 0 for r in negative)
    # an import that completed before the signal is never a pass and never a failure of the system
    assert all(
        r["verdict"] == "INCONCLUSIVE"
        for r in rows
        if r["category"] == "import_completed_before_sigterm"
    )
    for category in ("late_gpu_release", "binary_mutation", "destructive_importer_signal"):
        assert all(r["verdict"] == "FAIL" for r in rows if r["category"] == category), category
    # every fixture leaves its gate text; the JSON is kept for the positive ones
    assert len(list((tmp_path / "out").glob("*.gates.txt"))) == len(rows)
    assert len(list((tmp_path / "out").glob("*.gates.json"))) == len(positive)


def _v1_like(mod, **status):
    st = dict.fromkeys(mod.REAL_DATA_GATES, "PASS")
    st.update(dict.fromkeys(mod.FORCED_RELEASE_GATES, "FAIL"))
    st.update({"H.host_identity": "INCONCLUSIVE", "S.binary_unchanged": "FAIL"})
    st.update(status)
    verdict = "FAIL" if "FAIL" in st.values() else "INCONCLUSIVE"
    report = {"verdict": verdict, "gates": [{"gate": g, "status": s} for g, s in st.items()]}
    lifecycle = {
        "phase_history": [
            {"phase": "importing", "at": 1.0},
            {"phase": "report_written", "at": 2.0},
        ],
        "control_commands": [{"command": "note V1 operator probe: control file read"}],
        "actions": [{"pid": 1, "action": "SIGTERM", "reason": "shutdown"}],
        "gpu": {"release_requested_at": None, "released_at": 3.0},
    }
    return report, lifecycle


def test_v1_fixture_expectation_is_met_only_for_a_run_without_a_forced_release():
    mod = _load()
    report, lc = _v1_like(mod)
    ok, lines = mod.expect_v1(report, lc)
    assert ok and lines[-1].startswith("# V1 fixture expectation: MET")
    # a forced-release gate that passes on V1 data would mean the gate does not test the forced release
    report, lc = _v1_like(mod, **{"FR5.gpu_release_within_deadline": "PASS"})
    assert mod.expect_v1(report, lc)[0] is False
    # the recovery gates must hold on the real data
    report, lc = _v1_like(mod, **{"FR8b.recovered_trace_complete": "FAIL"})
    assert mod.expect_v1(report, lc)[0] is False
    # an unexplained failure elsewhere is not hidden behind the expected verdict
    report, lc = _v1_like(mod, **{"S.closeout": "FAIL"})
    ok, lines = mod.expect_v1(report, lc)
    assert ok is False and any("UNEXPECTED" in line for line in lines)
    # a record that does show a release request is not "V1 without a forced release"
    report, lc = _v1_like(mod)
    lc["gpu"]["release_requested_at"] = 2.5
    assert mod.expect_v1(report, lc)[0] is False
    # a PASS verdict is never the expected result
    report, lc = _v1_like(mod)
    report["verdict"] = "PASS"
    assert mod.expect_v1(report, lc)[0] is False


def test_expect_v1_cli(tmp_path, capsys):
    mod = _load()
    report, lc = _v1_like(mod)
    (tmp_path / "g.json").write_text(json.dumps(report))
    (tmp_path / "l.json").write_text(json.dumps(lc))
    argv = [
        "expect-v1",
        "--gates",
        str(tmp_path / "g.json"),
        "--lifecycle",
        str(tmp_path / "l.json"),
    ]
    assert mod.main([*argv, "--out", str(tmp_path / "o.json")]) == 0
    assert json.loads((tmp_path / "o.json").read_text())["ok"] is True
    assert "verdict_from_forced_release_gates_alone" in capsys.readouterr().out
