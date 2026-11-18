"""CPU tests for the prospective V2 gates (checks/e1_forced_release_gates.py): both admissible outcomes pass, and
missing streams, incomplete imports, failed semantic recovery, wrong triggers, importer signals, late or unobserved
GPU release, host conflicts and a hung importer each stop the validation."""

from __future__ import annotations

# ruff: noqa: E501

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from test_e1_lifecycle_gates_v2 import _fake_tools

REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/checks/e1_forced_release_gates.py"
)
T0 = 1_800_000_000.0
HOST = "pod4-gb300-10-tray16-f3"
LINE = "release-gpus V2 forced release during import"
INPUTS = {
    "expected": {
        "iterations_band": [2100, 2250],
        "nvtx_per_iteration": 4,
        "kernel_rows_band": [1_163, 1_286],
        "ranks": 4,
    },
    "trigger": {"control_line": LINE},
    "limits": {"gpu_release_deadline_s": 90},
}


def _load():
    spec = importlib.util.spec_from_file_location("e1_forced_release_gates", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(tmp: Path, outcome: str = "A", mutate=None, obs=None) -> tuple[Path, Path]:
    run = tmp / "e1_v2_forced_release"
    nsys_dir, vllm = run / "profile" / "nsys", run / "profile" / "vllm"
    nsys_dir.mkdir(parents=True)
    vllm.mkdir(parents=True)
    data = b"q" * 2048
    preserved = tmp / "x.raw-preserve" / "x.qdstrm"
    preserved.parent.mkdir()
    preserved.write_bytes(data)
    durable = run / "preserved" / "x.qdstrm"
    durable.parent.mkdir()
    durable.write_bytes(data)
    phases = ["stop_requested", "stop_accepted", "capture_ended", "stream_written", "importing"] + (
        ["report_written"] if outcome == "A" else ["stop_failed"]
    )
    at = [1500, 1501, 1564, 1564, 1565, 1592]
    lc = {
        "schema_version": 2,
        "host": HOST,
        "host_identity": {"hostname": HOST, "boot_id": "54a59f08"},
        "phase": phases[-1],
        "launch_started_at": T0,
        "stop_requested_at": T0 + 1500,
        "phase_history": [{"phase": p, "at": T0 + t} for p, t in zip(phases, at)],
        "http": {"status": "error: connection reset", "returned_at": T0 + 1570, "requests_sent": 1},
        "stream": {
            "path": "/tmp/nsys-report-1.qdstrm",
            "size": 2018,
            "preserved_path": str(preserved),
            "preserved_size": 2048,
            "observations": [
                {
                    "at": T0 + 1565,
                    "phase": "importing",
                    "size": 2018,
                    "mtime_ns": 1,
                    "path": str(preserved),
                }
            ],
            "final": {
                "available": True,
                "size": 2048,
                "sha256": hashlib.sha256(data).hexdigest(),
                "taken_at": T0 + 1601,
                "session_closed_at": T0 + 1600,
            },
        },
        "policy": {
            "gpu_held_ceiling_s": 2700,
            "session_close_timeout_s": 300,
            "hard_deadline_s": 600,
        },
        "escalations": [],
        "warnings": [],
        "control_commands": [
            {"at": T0 + 1568, "command": LINE, "phase": "importing", "result": "release requested"}
        ],
        "nsys": {"pid": 4000, "alive": False},
        "server": {
            "api_server": {"pid": 4100},
            "engine_processes": [{"pid": 4101}, {"pid": 4102}],
            "alive": False,
        },
        "gpu": {
            "release_requested_at": T0 + 1568,
            "released_at": T0 + 1575,
            "release_reason": f"operator: {LINE}",
            "release_context": {
                "phase": "importing",
                "import_progress_pct": 12,
                "nsys_alive": True,
                "report_written": False,
            },
        },
        "session_closed_at": T0 + 1600,
        "actions": [
            {"at": T0 + 1568.2, "reason": f"operator: {LINE}", "pid": 4100, "action": "SIGTERM"},
            {"at": T0 + 1571, "reason": f"operator: {LINE}", "pid": 4101, "action": "SIGKILL"},
        ],
    }
    probe = {
        "saw_phase": "importing",
        "nsys_pid": 4000,
        "nsys_alive": True,
        "wrote_trigger_at": T0 + 1567.5,
    }
    observations = (
        obs
        if obs is not None
        else [
            (1560, HOST, 260_000, [1]),
            (1566, HOST, 260_000, [1]),
            (1574, HOST, 0, []),
            (1580, HOST, 0, []),
        ]
    )
    if mutate:
        mutate(lc, probe)
    (nsys_dir / "lifecycle.json").write_text(json.dumps(lc))
    (run / "probe.json").write_text(json.dumps(probe))
    (run / "reservation.json").write_text(json.dumps({"node": HOST, "phase": "Running"}))
    (run / "run.log").write_text(f"[e1_v2_forced_release] start 2026-11-19T00:00:00Z on {HOST} x\n")
    (run / "gpu_observations.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "at": T0 + a,
                    "host": h,
                    "source": "e1_gpu_watch",
                    "memory_used_mib": [m] * 4,
                    "compute_pids": p,
                }
            )
            + "\n"
            for a, h, m, p in observations
        )
    )
    (vllm / "x_server.log").write_text(
        "(APIServer pid=4100) INFO 11-19 00:25:00 [api_router.py:31] Stopping profiler...\n"
    )
    if outcome == "A":
        (tmp / "ins").mkdir()
        _, nsys = _fake_tools(tmp / "ins", kernel_rows=1_225, nvtx=8640)
        subprocess.run([str(nsys), "export", "-o", str(nsys_dir / "x.sqlite"), "x"], check=True)
    return run, durable


def _eval(mod, run, durable, tmp, kernel_rows=1_225, nvtx=8640, **kw):
    (tmp / "tools").mkdir(exist_ok=True)
    imp, nsys = _fake_tools(tmp / "tools", kernel_rows=kernel_rows, nvtx=nvtx)
    kw.setdefault("importer", str(imp))
    kw.setdefault("nsys", str(nsys))
    return mod.evaluate(run, INPUTS, scratch=tmp / "scratch", durable_stream=durable, **kw)


def _st(res):
    return {r["gate"]: r["status"] for r in res}


@pytest.mark.parametrize("outcome", ["A", "B"])
def test_both_pre_registered_outcomes_pass(tmp_path, outcome):
    mod = _load()
    run, durable = _run(tmp_path, outcome)
    ident = tmp_path / "identity.json"
    ident.write_text(json.dumps({"verdict": "PASS", "normalized": []}))
    res = _eval(mod, run, durable, tmp_path, identity=ident, extras={"closeout": 0})
    assert [r for r in res if r["status"] != "PASS"] == []
    assert mod.verdict_of(res) == "PASS"
    rel = next(r for r in res if r["gate"] == "FR8b.recovered_trace_complete")["data"]["relative"]
    assert (rel is not None) == (outcome == "A")  # equality with an original only where one exists


def _set(path, value):
    def mutate(lc, probe):
        target = lc
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


CASES = {
    "trigger_in_wrong_phase": (
        _set(["control_commands", 0, "phase"], "stop_accepted"),
        "FR2.registered_trigger",
    ),
    "unregistered_extra_command": (
        lambda lc, p: lc["control_commands"].append(
            {"command": "extend 60 x", "result": "ok", "phase": "importing"}
        ),
        "FR2.registered_trigger",
    ),
    "release_after_the_report": (
        _set(
            ["gpu", "release_context"],
            {"phase": "report_written", "nsys_alive": True, "report_written": True},
        ),
        "FR3.import_active_at_release",
    ),
    "probe_saw_no_live_importer": (
        lambda lc, p: p.__setitem__("nsys_alive", False),
        "FR3.import_active_at_release",
    ),
    "signal_to_the_importer": (
        lambda lc, p: lc["actions"].append(
            {"at": T0 + 1569, "reason": "hard deadline", "pid": 4000, "action": "SIGINT"}
        ),
        "FR4.no_signal_to_importer",
    ),
    "signal_to_an_unrecorded_pid": (
        lambda lc, p: lc["actions"].append(
            {"at": T0 + 1569, "reason": "x", "pid": 777, "action": "SIGKILL"}
        ),
        "FR4.no_signal_to_importer",
    ),
    "importer_hung": (
        lambda lc, p: (
            lc["nsys"].__setitem__("alive", True),
            lc.__setitem__("session_closed_at", None),
            lc.__setitem__("phase", "importing"),
        ),
        "FR6.session_ended",
    ),
    "no_final_stream_record": (_set(["stream", "final"], None), "FR7.stream_final_stable_durable"),
    "final_record_differs_from_the_durable_copy": (
        _set(["stream", "final", "sha256"], "0" * 64),
        "FR7.stream_final_stable_durable",
    ),
    "second_stop_request": (_set(["http", "requests_sent"], 2), "FR1.one_stop_request"),
    "stop_error_before_the_release": (
        _set(["http", "returned_at"], T0 + 1510),
        "FR1.one_stop_request",
    ),
    "escalation": (
        lambda lc, p: lc["escalations"].append({"kind": "hard_deadline", "phase": "importing"}),
        "FR9.bounds",
    ),
    "lifecycle_host_conflict": (_set(["host"], "pod4-gb300-10-tray14-f3"), "H.host_identity"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_each_violation_stops_the_validation(tmp_path, case):
    mod = _load()
    mutate, gate = CASES[case]
    run, durable = _run(tmp_path, "A", mutate=mutate)
    res = _eval(mod, run, durable, tmp_path)
    assert _st(res)[gate] == "FAIL", (case, [r for r in res if r["gate"] == gate])
    assert mod.verdict_of(res) == "FAIL"


@pytest.mark.parametrize(
    ("obs", "why"),
    [
        ([(1560, HOST, 260_000, [1]), (1700, HOST, 0, [])], "released only after the deadline"),
        ([(1574, HOST, 0, [])], "no held observation before the request: release not bracketed"),
        ([(1560, HOST, 260_000, [1])], "never observed released"),
        (
            [(1560, HOST, 260_000, [1]), (1574, HOST, 0, []), (1590, HOST, 200_000, [9])],
            "held again after the release",
        ),
        ([], "no independent observation at all"),
    ],
)
def test_gpu_release_needs_bracketing_independent_observations_within_the_deadline(
    tmp_path, obs, why
):
    mod = _load()
    run, durable = _run(tmp_path, "A", obs=obs)
    assert _st(_eval(mod, run, durable, tmp_path))["FR5.gpu_release_within_deadline"] == "FAIL", why


def test_missing_stream_incomplete_import_and_failed_semantic_recovery(tmp_path):
    mod = _load()
    (tmp_path / "a").mkdir()
    run, durable = _run(tmp_path / "a", "B")
    durable.unlink()  # missing stream
    st = _st(_eval(mod, run, durable, tmp_path / "a"))
    assert (
        st["FR7.stream_final_stable_durable"] == "FAIL"
        and st["FR8.offline_import"] == "FAIL"
        and st["FR8b.recovered_trace_complete"] == "FAIL"
    )
    (tmp_path / "b").mkdir()
    run, durable = _run(tmp_path / "b", "B")
    broken = tmp_path / "b" / "QdstrmImporter"
    broken.write_text(
        "#!/bin/sh\necho 'truncated stream: unexpected end of file' >&2\nexit 1\n"
    )  # incomplete import
    broken.chmod(0o755)
    res = _eval(mod, run, durable, tmp_path / "b", importer=str(broken))
    st = _st(res)
    assert st["FR7.stream_final_stable_durable"] == "PASS"  # stable and checksummed …
    assert (
        st["FR8.offline_import"] == "FAIL" and st["FR8b.recovered_trace_complete"] == "FAIL"
    )  # … yet not a complete trace
    assert "truncated stream" in next(r for r in res if r["gate"] == "FR8.offline_import")["detail"]
    (tmp_path / "c").mkdir()
    run, durable = _run(tmp_path / "c", "B")
    st = _st(
        _eval(mod, run, durable, tmp_path / "c", kernel_rows=900)
    )  # imports, but too few kernels
    assert st["FR8.offline_import"] == "PASS" and st["FR8b.recovered_trace_complete"] == "FAIL"
    (tmp_path / "d").mkdir()
    run, durable = _run(tmp_path / "d", "A")
    st = _st(
        _eval(mod, run, durable, tmp_path / "d", kernel_rows=1_224)
    )  # in band, but not equal to the original
    assert st["FR8b.recovered_trace_complete"] == "FAIL"
    (tmp_path / "e").mkdir()
    run, durable = _run(tmp_path / "e", "B")
    res = mod.evaluate(run, INPUTS, durable_stream=durable)  # no tools: the proof is mandatory
    assert _st(res)["FR8.offline_import"] == "INCONCLUSIVE" and mod.verdict_of(res) != "PASS"


def test_cli_verdict_and_exit_code(tmp_path, capsys):
    mod = _load()
    run, durable = _run(
        tmp_path, "B", mutate=_set(["control_commands", 0, "phase"], "capture_ended")
    )
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(INPUTS))
    out = tmp_path / "V2-GATES.json"
    rc = mod.main(
        [
            "--run",
            str(run),
            "--inputs",
            str(inputs),
            "--durable-stream",
            str(durable),
            "--out",
            str(out),
        ]
    )
    assert rc == 1 and json.loads(out.read_text())["verdict"] == "FAIL"
    assert "STOP (no retry, no advance)" in capsys.readouterr().out
