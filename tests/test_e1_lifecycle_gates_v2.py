"""CPU tests for the corrected V1 gates (checks/e1_lifecycle_gates_v2.py); the pinned gates and their tests are unchanged."""

from __future__ import annotations

# ruff: noqa: E501

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import test_e1_lifecycle_gates as base  # synthetic run directory and fake tools of the pinned-gate tests

REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/checks/e1_lifecycle_gates_v2.py"
)
T0 = 1_800_000_000.0


def _load():
    spec = importlib.util.spec_from_file_location("e1_lifecycle_gates_v2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(tmp_path: Path, *, stops: int = 1, host: str = "tray", **over) -> Path:
    run = base._run_dir(tmp_path, census=False, **over)
    (run / "run.log").write_text(
        f"[e1_v1_lifecycle] start 2026-11-18T02:53:23Z on {host} launcher_tree=x\n"
    )
    log = run / "profile" / "vllm" / "x_server.log"
    log.write_text(
        "".join(
            "INFO [api_router.py:31] Stopping profiler...\nINFO [api_router.py:33] Profiler stopped.\n"
            for _ in range(stops)
        )
    )
    # the in-session export the recovered report is compared with (same generator as the fake nsys)
    ins = tmp_path / "insession"
    ins.mkdir()
    _, nsys = _fake_tools(ins, kernel_rows=1_225, nvtx=8640)
    subprocess.run(
        [str(nsys), "export", "-o", str(run / "profile" / "nsys" / "x.sqlite"), "x"], check=True
    )
    # the preserved stream was last written before the session closed (independent of the wall clock)
    rec = json.loads((run / "profile" / "nsys" / "lifecycle.json").read_text())
    os.utime(rec["stream"]["preserved_path"], (T0 + 1550, T0 + 1550))
    return run


def _fake_tools(tmp_path: Path, *, kernel_rows: int, nvtx: int) -> tuple[Path, Path]:
    """Fake QdstrmImporter and fake `nsys export` with the tables the corrected L10b reads (the
    pinned tests' fake has only the two count columns)."""
    imp = tmp_path / "QdstrmImporter"
    imp.write_text('#!/bin/sh\nhead -c 4096 /dev/zero > "$4"\n')
    imp.chmod(0o755)
    nsys = tmp_path / "nsys"
    nsys.write_text(
        f"""#!/usr/bin/env python3
import sqlite3, sys
out = sys.argv[sys.argv.index("-o") + 1]
con = sqlite3.connect(out)
con.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
con.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER)")
con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, globalPid INTEGER)")
con.execute("CREATE TABLE ANALYSIS_DETAILS (globalVid INTEGER, duration INTEGER, startTime INTEGER, stopTime INTEGER)")
con.execute("CREATE TABLE META_DATA_CAPTURE (name TEXT, value TEXT)")
con.execute("CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT)")
con.execute("CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER, utcTime TEXT, localTime TEXT, monotonic INTEGER)")
con.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL)",
                [(10 * i, 10 * i + 5, f"vllm_iteration({{i // 4}}): forward") for i in range({nvtx})])
con.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?)",
                [(1_000_000 + i, 1_000_000 + i + 1, i % 4) for i in range({kernel_rows})])
con.execute("INSERT INTO ANALYSIS_DETAILS VALUES (1, 500, 0, 500)")
con.execute("INSERT INTO META_DATA_CAPTURE VALUES ('START_TRIGGER', 'Immediate')")
con.execute("INSERT INTO META_DATA_EXPORT VALUES ('EXPORT_PRODUCT_VERSION', '2025.6.3.541')")
con.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (1, 'a', 'a', 2)")
con.commit(); con.close()
"""
    )
    nsys.chmod(0o755)
    return imp, nsys


def _status(res):
    return {r["gate"]: r["status"] for r in res}


def _obs(tmp_path: Path, rows) -> Path:
    p = tmp_path / "obs.jsonl"
    p.write_text(
        "".join(
            json.dumps(
                {
                    "at": T0 + at,
                    "host": host,
                    "source": "test",
                    "memory_used_mib": [mem] * 4,
                    "compute_pids": pids,
                }
            )
            + "\n"
            for at, host, mem, pids in rows
        )
    )
    return p


GOOD_OBS = [(1560, "tray", 260_000, [1, 2, 3, 4]), (1625, "tray", 0, [])]


def _evaluate(mod, run, tmp_path, **kw):
    kw.setdefault("observations", _obs(tmp_path, GOOD_OBS))
    return mod.evaluate(run, base.INPUTS, **kw)


def test_one_effective_stop_passes_with_zero_or_with_suppressed_duplicates(tmp_path):
    mod = _load()
    for i, http in enumerate(
        (
            {"status": "ok", "requests_sent": 1},  # the real V1 record: no duplicate at all
            {"status": "ok", "requests_sent": 1, "duplicate_requests_suppressed": 5},
        )
    ):
        (tmp_path / str(i)).mkdir()
        run = _run(tmp_path / str(i), http=http)
        assert _status(_evaluate(mod, run, tmp_path))["L1.one_effective_stop"] == "PASS"


def test_a_second_effective_stop_or_finalisation_fails(tmp_path):
    mod = _load()
    (tmp_path / "a").mkdir()
    run = _run(tmp_path / "a", http={"status": "ok", "requests_sent": 2})
    assert _status(_evaluate(mod, run, tmp_path))["L1.one_effective_stop"] == "FAIL"
    (tmp_path / "b").mkdir()
    run = _run(
        tmp_path / "b",
        stops=2,
        http={"status": "ok", "requests_sent": 1, "duplicate_requests_suppressed": 1},
    )
    assert (
        _status(_evaluate(mod, run, tmp_path))["L1.one_effective_stop"] == "FAIL"
    )  # server saw two stops
    (tmp_path / "c").mkdir()
    twice = [
        {"phase": p, "at": T0 + 1500 + i}
        for i, p in enumerate(
            [
                "stop_requested",
                "stop_accepted",
                "capture_ended",
                "importing",
                "report_written",
                "importing",
                "report_written",
            ]
        )
    ]
    run = _run(tmp_path / "c", phase_history=twice)
    assert (
        _status(_evaluate(mod, run, tmp_path))["L1.one_effective_stop"] == "FAIL"
    )  # duplicate finalisation


def test_stream_gate_ignores_the_early_size_sample_and_records_size_and_sha(tmp_path):
    mod = _load()
    run = _run(tmp_path)
    lc = run / "profile" / "nsys" / "lifecycle.json"
    rec = json.loads(lc.read_text())
    rec["stream"]["size"] = 2018  # sampled while Nsight was still appending (the V1 situation)
    lc.write_text(json.dumps(rec))
    res = _evaluate(mod, run, tmp_path)
    g = next(r for r in res if r["gate"] == "L4.stream_final_and_stable")
    assert (
        g["status"] == "PASS" and g["data"]["final_size"] == 2048 and len(g["data"]["sha256"]) == 64
    )
    assert g["data"]["recorded_size_sample"] == 2018


def test_stream_gate_fails_without_finalisation_on_mutation_or_late_write(tmp_path, monkeypatch):
    mod = _load()
    (tmp_path / "a").mkdir()
    run = _run(tmp_path / "a", session_closed_at=None)
    st = _status(_evaluate(mod, run, tmp_path))
    assert st["L4.stream_final_and_stable"] == "FAIL" and st["L10.offline_import"] == "FAIL"
    (tmp_path / "b").mkdir()
    run = _run(tmp_path / "b")
    seq = iter(["a" * 64, "b" * 64, "c" * 64])
    lc = json.loads((run / "profile" / "nsys" / "lifecycle.json").read_text())
    res: list = []
    assert (
        mod.gate_l4(res, lc, None, hasher=lambda p: next(seq)) is None
    )  # content changed between two reads
    assert res[0]["status"] == "FAIL"
    (tmp_path / "c").mkdir()
    run = _run(
        tmp_path / "c", session_closed_at=T0 - 10_000_000_000
    )  # stream written long after the session closed
    assert _status(_evaluate(mod, run, tmp_path))["L4.stream_final_and_stable"] == "FAIL"
    (tmp_path / "d").mkdir()
    run = _run(tmp_path / "d")
    other = tmp_path / "copy.qdstrm"
    other.write_bytes(b"x" * 2048)  # a "durable copy" that is not the recorded stream
    assert (
        _status(_evaluate(mod, run, tmp_path, stream=other))["L4.stream_final_and_stable"] == "FAIL"
    )


def test_recovery_is_mandatory_and_semantic(tmp_path):
    mod = _load()
    run = _run(tmp_path)
    st = _status(_evaluate(mod, run, tmp_path))
    assert (
        st["L10.offline_import"] == "INCONCLUSIVE"
        and mod.verdict_of(_evaluate(mod, run, tmp_path)) != "PASS"
    )
    (tmp_path / "t1").mkdir()
    imp, nsys = _fake_tools(tmp_path / "t1", kernel_rows=1_225, nvtx=8640)
    res = _evaluate(mod, run, tmp_path, importer=str(imp), nsys=str(nsys), scratch=tmp_path / "s1")
    st = _status(res)
    assert (
        st["L10.offline_import"] == "PASS" and st["L10b.recovery_semantic_completeness"] == "PASS"
    )
    checks = next(r for r in res if r["gate"] == "L10b.recovery_semantic_completeness")["data"][
        "checks"
    ]
    assert set(checks) >= {
        "kernel_rows",
        "nvtx_forward_ranges",
        "distinct_iterations",
        "kernel_boundaries_ns",
        "every_table_row_count",
        "capture_metadata",
    }
    assert not (tmp_path / "s1" / "recovery.sqlite").exists()  # derived scratch removed
    (tmp_path / "t2").mkdir()
    imp, nsys = _fake_tools(tmp_path / "t2", kernel_rows=1_224, nvtx=8640)
    st = _status(
        _evaluate(mod, run, tmp_path, importer=str(imp), nsys=str(nsys), scratch=tmp_path / "s2")
    )
    assert st["L10b.recovery_semantic_completeness"] == "FAIL"
    (run / "profile" / "nsys" / "x.sqlite").unlink()
    (tmp_path / "t3").mkdir()
    imp, nsys = _fake_tools(tmp_path / "t3", kernel_rows=1_225, nvtx=8640)
    st = _status(
        _evaluate(mod, run, tmp_path, importer=str(imp), nsys=str(nsys), scratch=tmp_path / "s3")
    )
    assert (
        st["L10b.recovery_semantic_completeness"] == "INCONCLUSIVE"
    )  # stable size is not completeness


def test_gpu_release_is_an_observed_interval_with_the_controller_time_separate(tmp_path):
    mod = _load()
    run = _run(
        tmp_path
    )  # controller released_at = T0 + 1600; lifecycle.json has no host (as in V1)
    res = _evaluate(mod, run, tmp_path)
    g = next(r for r in res if r["gate"] == "L6c.gpu_release_observed_interval")
    assert g["status"] == "PASS"
    d = g["data"]
    assert (
        d["last_held_observation"],
        d["first_released_observation"],
        d["controller_released_at"],
    ) == (T0 + 1560, T0 + 1625, T0 + 1600)
    assert d["run_host"] == "tray" and d["exact_release_established"] is False
    cases = {
        "none": [],
        "held_after_controller_time": [
            (1560, "tray", 260_000, [1]),
            (1640, "tray", 260_000, [1]),
            (1700, "tray", 0, []),
        ],
        "never_released": [(1560, "tray", 260_000, [1])],
        "other_host": [(1560, "tray", 260_000, [1]), (1625, "tray99", 0, [])],
        "released_only_before_controller_time": [(1500, "tray", 0, [])],
        "memory_still_held": [(1625, "tray", 9_000, [])],
    }
    for name, rows in cases.items():
        st = _status(mod.evaluate(run, base.INPUTS, observations=_obs(tmp_path, rows)))
        assert st["L6c.gpu_release_observed_interval"] == "FAIL", name
    # only an upper bound (no held observation) is still an acceptable, explicitly labelled interval
    st = _status(
        mod.evaluate(run, base.INPUTS, observations=_obs(tmp_path, [(1625, "tray", 0, [])]))
    )
    assert st["L6c.gpu_release_observed_interval"] == "PASS"


def test_cli_keeps_the_original_verdict_separate(tmp_path, capsys):
    mod = _load()
    run = _run(tmp_path)
    (tmp_path / "t").mkdir()
    imp, nsys = _fake_tools(tmp_path / "t", kernel_rows=1_225, nvtx=8640)
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(base.INPUTS))
    original = tmp_path / "V1-GATES.json"
    original.write_text(
        json.dumps({"verdict": "STOP", "gates": [{"gate": "L1.idempotent_stop", "status": "FAIL"}]})
    )
    before = original.read_bytes()
    ident = tmp_path / "identity_v2.json"
    ident.write_text(json.dumps({"verdict": "PASS", "normalized": []}))
    out = tmp_path / "retro.json"
    rc = mod.main(
        [
            "--run",
            str(run),
            "--inputs",
            str(inputs),
            "--importer",
            str(imp),
            "--nsys",
            str(nsys),
            "--scratch",
            str(tmp_path / "s"),
            "--observations",
            str(_obs(tmp_path, GOOD_OBS)),
            "--identity",
            str(ident),
            "--original-gates",
            str(original),
            "--out",
            str(out),
        ]
    )
    rep = json.loads(out.read_text())
    assert rc == 0 and rep["retrospective_assessment"] == "PASS"
    assert rep["original_pinned_verdict"]["verdict"] == "STOP" and original.read_bytes() == before
    assert "original verdict under the pinned gates" in capsys.readouterr().out
