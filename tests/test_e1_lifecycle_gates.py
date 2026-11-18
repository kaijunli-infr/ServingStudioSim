"""CPU tests for the V1 lifecycle acceptance gates (phase1c/e1/checks/e1_lifecycle_gates.py).

A synthetic run directory with the controller's record, a profile result, a launch record and the
independent nvidia-smi census stands in for a real capture; a PASS record, a record with a signal
to the importer, a missing probe, a GPU-held overrun, a missing or busy census and a recovered
report with fewer kernels must each produce the right verdict. Any FAIL is a STOP.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
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
        "kernel_rows_band": [1_163, 1_286],  # scaled-down bands (the fake export inserts rows)
    },
    "control_probe": {
        "lines": [
            "note V1 operator probe: control file read",
            "extend 60 V1 operator probe: extension logged",
        ],
        "extension_s": 60,
    },
}


IN_SESSION = {
    "ok": True,
    "kernel_rows": 1_225,
    "nvtx_forward_ranges": 8640,
    "kernel_span_ms": 500.0,
}


def _census(run: Path, at: float, *, released: bool = True, name: str = "t20") -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / f"gpu_release_census_{name}.json").write_text(
        json.dumps(
            {
                "host": "tray",
                "at": at,
                "at_iso": "x",
                "max_used_mib": 64,
                "compute_apps": [] if released else [{"pid": 4100, "used_mib": 180_000}],
                "gpus": [
                    {"index": i, "memory_used_mib": 0 if released else 180_000} for i in range(4)
                ],
                "verdict": {"available": True, "released": released, "reason": "test"},
            }
        )
    )


def _run_dir(tmp_path: Path, census: bool = True, **over) -> Path:
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
        "host": "tray",
        "session_closed_at": t0 + 1620,
        "actions": [{"at": t0 + 1565, "reason": "shutdown", "pid": 4100, "action": "SIGTERM"}],
    }
    for key, value in over.items():
        record[key] = value
    (nsys / "lifecycle.json").write_text(json.dumps(record))
    (run / "profile" / "profile_result.json").write_text(
        json.dumps({"validation": dict(IN_SESSION)})
    )
    if census:
        _census(run, t0 + 1700)
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


def _fake_tools(tmp_path: Path, *, kernel_rows: int, nvtx: int) -> tuple[Path, Path]:
    """A fake QdstrmImporter (writes a report) and a fake nsys whose `export --type sqlite` creates
    the two tables the in-session validation counts, with the given row counts."""
    imp = tmp_path / "QdstrmImporter"
    imp.write_text(
        "#!/bin/sh\n# -i <stream> -o <report>: write a report the size of the in-session one\n"
        'head -c 4096 /dev/zero > "$4"\n'
    )
    imp.chmod(0o755)
    nsys = tmp_path / "nsys"
    nsys.write_text(
        f"""#!/usr/bin/env python3
import sqlite3, sys
# argv: export --type sqlite --force-overwrite true -o <out.sqlite> <report>
out = sys.argv[sys.argv.index("-o") + 1]
con = sqlite3.connect(out)
con.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
con.execute("CREATE TABLE NVTX_EVENTS (text TEXT, textId INTEGER)")
con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER)")
con.executemany("INSERT INTO NVTX_EVENTS VALUES (?, NULL)",
                [(f"vllm_iteration({{i // 4}}): forward",) for i in range({nvtx})])
con.execute("INSERT INTO NVTX_EVENTS VALUES ('vllm_iteration(1): other', NULL)")
con.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?)",
                [(1_000_000 + i, 1_000_000 + i + 1) for i in range({kernel_rows})])
con.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end = 501_000_000 WHERE rowid = 1")
con.commit(); con.close()
"""
    )
    nsys.chmod(0o755)
    return imp, nsys


def test_recovery_proof_imports_exports_and_compares_the_in_session_counts(tmp_path):
    mod = _load()
    run = _run_dir(tmp_path)
    imp, nsys = _fake_tools(tmp_path, kernel_rows=1_225, nvtx=8640)
    res = mod.evaluate(
        run, INPUTS, importer=str(imp), nsys=str(nsys), scratch=tmp_path / "scratch", extras={}
    )
    st = _statuses(res)
    assert st["L4.stream_preserved"] == "PASS"
    assert st["L10.recovery_proof"] == "PASS"
    assert st["L10b.recovery_semantic_completeness"] == "PASS"
    assert (tmp_path / "scratch" / "recovery.nsys-rep").stat().st_size == 4096
    with sqlite3.connect(tmp_path / "scratch" / "recovery.sqlite") as con:
        assert con.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0] == 1_225
    detail = next(r["detail"] for r in res if r["gate"] == "L10b.recovery_semantic_completeness")
    assert "kernel_rows 1225 (in-session 1225)" in detail and "2160 iterations" in detail


def test_recovered_report_with_fewer_kernels_or_ranges_is_a_stop(tmp_path):
    mod = _load()
    run = _run_dir(tmp_path)
    imp, nsys = _fake_tools(tmp_path, kernel_rows=1_200, nvtx=8640)
    st = _statuses(
        mod.evaluate(
            run, INPUTS, importer=str(imp), nsys=str(nsys), scratch=tmp_path / "s1", extras={}
        )
    )
    assert st["L10.recovery_proof"] == "PASS"
    assert st["L10b.recovery_semantic_completeness"] == "FAIL"
    (tmp_path / "b").mkdir(exist_ok=True)
    imp, nsys = _fake_tools(tmp_path / "b", kernel_rows=1_225, nvtx=8636)
    st = _statuses(
        mod.evaluate(
            run, INPUTS, importer=str(imp), nsys=str(nsys), scratch=tmp_path / "s2", extras={}
        )
    )
    assert st["L10b.recovery_semantic_completeness"] == "FAIL"
    # size alone no longer decides anything: no nsys for the export is a STOP, not a pass
    st = _statuses(mod.evaluate(run, INPUTS, importer=str(imp), scratch=tmp_path / "s3", extras={}))
    assert st["L10.recovery_proof"] == "PASS"
    assert st["L10b.recovery_semantic_completeness"] == "FAIL"


def test_preserved_stream_must_be_byte_complete_on_disk(tmp_path):
    mod = _load()
    run = _run_dir(tmp_path)
    lc = run / "profile" / "nsys" / "lifecycle.json"
    rec = json.loads(lc.read_text())
    Path(rec["stream"]["preserved_path"]).write_bytes(b"q" * 2000)  # truncated copy
    lc.write_text(json.dumps(rec))
    st = _statuses(mod.evaluate(run, INPUTS, importer="/bin/true", scratch=None, extras={}))
    assert st["L4.stream_preserved"] == "FAIL"
    assert st["L10.recovery_proof"] == "INFO"  # not attempted on an incomplete stream


def test_gpu_release_needs_an_independent_census_after_the_controller_timestamp(tmp_path):
    mod = _load()
    run = _run_dir(tmp_path, census=False)
    st = _statuses(mod.evaluate(run, INPUTS, importer=None, scratch=None, extras={}))
    assert st["L6.gpu_held_within_ceiling"] == "PASS"
    assert st["L6c.gpu_release_verified_independently"] == "FAIL"
    _census(run, 1_800_000_000.0 + 1700, released=False)
    st = _statuses(mod.evaluate(run, INPUTS, importer=None, scratch=None, extras={}))
    assert st["L6c.gpu_release_verified_independently"] == "FAIL"
    _census(
        run, 1_800_000_000.0 + 1590, released=True, name="t0"
    )  # before the controller's timestamp
    run2 = _run_dir(tmp_path / "two", census=False)
    _census(run2, 1_800_000_000.0 + 1590, released=True, name="t0")
    st = _statuses(mod.evaluate(run2, INPUTS, importer=None, scratch=None, extras={}))
    assert st["L6c.gpu_release_verified_independently"] == "FAIL"
    _census(run2, 1_800_000_000.0 + 1700, released=True, name="t20")
    st = _statuses(mod.evaluate(run2, INPUTS, importer=None, scratch=None, extras={}))
    assert st["L6c.gpu_release_verified_independently"] == "PASS"


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
