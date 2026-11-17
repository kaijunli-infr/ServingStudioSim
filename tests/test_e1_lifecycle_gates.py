"""CPU tests for the V1 lifecycle acceptance gates (phase1c/e1/checks/e1_lifecycle_gates.py).

A synthetic run directory with the controller's record, a profile result and a launch record
stands in for a real capture; a PASS record, a record with a signal to the importer, a missing
probe and a GPU-held overrun must each produce the right verdict. Any FAIL is a STOP.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
E1 = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1"
SCRIPT = E1 / "checks" / "e1_lifecycle_gates.py"


def _load():
    spec = importlib.util.spec_from_file_location("e1_lifecycle_gates", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


INPUTS = {
    "expected": {
        "iterations_band": [2100, 2250],
        "nvtx_per_iteration": 4,
        "kernel_rows_band": [11_638_290, 12_863_373],
    },
    "control_probe": {
        "lines": [
            "note V1 operator probe: control file read",
            "extend 60 V1 operator probe: extension logged",
        ],
        "extension_s": 60,
    },
}


def _run_dir(tmp_path: Path, **over) -> Path:
    run = tmp_path / "e1_v1_lifecycle"
    nsys = run / "profile" / "nsys"
    vllm = run / "profile" / "vllm"
    nsys.mkdir(parents=True)
    vllm.mkdir(parents=True)
    rep = nsys / "x.nsys-rep"
    rep.write_bytes(b"r" * 4096)
    preserved = tmp_path / "x.raw-preserve" / "x.qdstrm"
    preserved.parent.mkdir()
    preserved.write_bytes(b"q" * 2048)
    t0 = 1_800_000_000.0
    record = {
        "schema_version": 1,
        "phase": "report_written",
        "launch_started_at": t0,
        "stop_requested_at": t0 + 1500,
        "phase_history": [
            {"phase": p, "at": t0 + 1500 + i * 10}
            for i, p in enumerate(
                [
                    "stop_requested",
                    "stop_accepted",
                    "capture_ended",
                    "stream_written",
                    "importing",
                    "report_written",
                ]
            )
        ],
        "http": {"status": "ok", "requests_sent": 1, "duplicate_requests_suppressed": 1},
        "stream": {
            "path": "/tmp/nsys-report-1234.qdstrm",
            "size": 2048,
            "preserved_path": str(preserved),
            "preserved_size": 2048,
            "discovered_by": "log",
        },
        "report": {"path": str(rep), "size": 4096, "written_at": t0 + 1560},
        "deadlines": {"hard_extension_s": 60.0},
        "policy": {
            "soft_deadline_s": 300,
            "hard_deadline_s": 600,
            "gpu_held_ceiling_s": 2700,
        },
        "escalations": [],
        "control_commands": [
            {"command": INPUTS["control_probe"]["lines"][0], "result": "noted"},
            {
                "command": INPUTS["control_probe"]["lines"][1],
                "result": "hard deadline extended by 60 s (operator)",
            },
        ],
        "warnings": [],
        "nsys": {"pid": 4000, "alive": False},
        "server": {"api_server": {"pid": 4100}, "alive": False},
        "gpu": {"released_at": t0 + 1600},
        "session_closed_at": t0 + 1620,
        "actions": [{"at": t0 + 1565, "reason": "shutdown", "pid": 4100, "action": "SIGTERM"}],
    }
    for key, value in over.items():
        record[key] = value
    (nsys / "lifecycle.json").write_text(json.dumps(record))
    (run / "profile" / "profile_result.json").write_text(
        json.dumps(
            {"validation": {"ok": True, "kernel_rows": 12_250_000, "nvtx_forward_ranges": 8640}}
        )
    )
    (vllm / "x_launch.json").write_text(
        json.dumps({"argv": ["/opt/nsys"], "nsys_profiler": {"version": "2025.6.3"}})
    )
    return run


def _statuses(res):
    return {r["gate"]: r["status"] for r in res}


def test_clean_record_passes_every_gate(tmp_path, capsys):
    mod = _load()
    run = _run_dir(tmp_path)
    res = mod.evaluate(run, INPUTS, importer=None, scratch=None, extras={"completeness_v2": 0})
    st = _statuses(res)
    fails = [g for g, s in st.items() if s == "FAIL"]
    assert fails == [], fails
    assert st["L10.recovery_proof"] == "INFO"  # not run without an importer
    assert st["S.completeness_v2"] == "PASS"


def test_signal_to_the_importer_before_the_report_is_a_stop(tmp_path):
    mod = _load()
    run = _run_dir(
        tmp_path,
        actions=[
            {"at": 1_800_001_540.0, "reason": "hard deadline", "pid": 4000, "action": "SIGINT"},
            {"at": 1_800_001_565.0, "reason": "shutdown", "pid": 4100, "action": "SIGTERM"},
        ],
    )
    st = _statuses(mod.evaluate(run, INPUTS, importer=None, scratch=None, extras={}))
    assert st["L5.no_signal_to_importer"] == "FAIL"


def test_missing_probe_overrun_and_duplicate_post_fail(tmp_path):
    mod = _load()
    run = _run_dir(
        tmp_path,
        control_commands=[],
        gpu={"released_at": 1_800_000_000.0 + 2800},
        http={"status": "ok", "requests_sent": 2, "duplicate_requests_suppressed": 0},
        escalations=[{"kind": "soft_deadline", "elapsed_s": 301}],
    )
    st = _statuses(
        mod.evaluate(run, INPUTS, importer=None, scratch=None, extras={"nsys_validation": 1})
    )
    assert st["L7.control_probe"] == "FAIL"
    assert st["L6.gpu_held_within_ceiling"] == "FAIL"
    assert st["L1.idempotent_stop"] == "FAIL"
    assert st["L8.no_escalation"] == "FAIL"
    assert st["L8c.extension_accounting"] == "FAIL"
    assert st["S.nsys_validation"] == "FAIL"


def test_recovery_proof_runs_the_importer_offline(tmp_path):
    mod = _load()
    run = _run_dir(tmp_path)
    fake = tmp_path / "QdstrmImporter"
    fake.write_text(
        "#!/bin/sh\n# -i <stream> -o <report>: write a report the size of the in-session one\n"
        'head -c 4096 /dev/zero > "$4"\n'
    )
    fake.chmod(0o755)
    res = mod.evaluate(run, INPUTS, importer=str(fake), scratch=tmp_path / "scratch", extras={})
    st = _statuses(res)
    assert st["L10.recovery_proof"] == "PASS"
    assert (tmp_path / "scratch" / "recovery.nsys-rep").stat().st_size == 4096


def test_cli_verdict_is_stop_on_any_fail(tmp_path, capsys):
    mod = _load()
    run = _run_dir(tmp_path, phase_history=[{"phase": "stop_requested", "at": 1_800_001_500.0}])
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(INPUTS))
    out = tmp_path / "gates.json"
    rc = mod.main(["--run", str(run), "--inputs", str(inputs), "--out", str(out)])
    assert rc == 1
    assert json.loads(out.read_text())["verdict"] == "STOP"
    assert "STOP" in capsys.readouterr().out
