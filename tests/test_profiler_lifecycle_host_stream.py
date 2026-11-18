"""Controller record: launch host identity, stream observations, final stream record (V2 readiness, item A).

The stop, importer isolation, termination and deadline behaviour is covered, unchanged, by test_profiler_lifecycle.py.
"""

from __future__ import annotations

# ruff: noqa: E501

import hashlib
import json
import os
from pathlib import Path

from alignment.profiler import lifecycle, procident
from alignment.profiler.lifecycle import COMPLETENESS_NOTE, RETENTION_NAME, Phase
from test_profiler_lifecycle import NSYS_PID, Harness, _policy


def _harness(tmp_path, monkeypatch, **kw):
    monkeypatch.setattr(procident, "is_descendant_of", lambda pid, root: True)
    return Harness(tmp_path, policy=_policy(), stop_return_at=1000.0 + 120, **kw)


def _stream(h, tmp_path: Path, data: bytes) -> Path:
    """Make the controller track a real file as the announced raw stream."""
    path = tmp_path / "nsys-report-1.qdstrm"
    path.write_bytes(data)
    h.ctl.s.stream["path"] = str(path)
    h.ctl.s.stream["discovered_by"] = "log"
    return path


def test_record_carries_the_launch_host_identity(tmp_path, monkeypatch):
    h = _harness(tmp_path, monkeypatch)
    status = h.ctl.status_dict()
    assert status["schema_version"] == 2
    assert status["host"] == status["host_identity"]["hostname"] == os.uname().nodename
    assert status["host_identity"]["boot_id"] == procident.boot_id()
    assert status["host_identity"]["launcher_pid"] == os.getpid()
    assert "launcher process" in status["host_identity"]["source"]
    assert (
        json.loads(h.ctl.status_path.read_text())["host"] == status["host"]
    )  # on disk from the start


def test_host_identity_comes_from_the_launch_context_not_from_a_default(tmp_path, monkeypatch):
    monkeypatch.setattr(procident, "is_descendant_of", lambda pid, root: True)
    ctl = lifecycle.ProfilerStopController(
        name="x",
        server_log=tmp_path / "s.log",
        report_stem=tmp_path / "r",
        status_path=tmp_path / "lifecycle.json",
        stop_request=lambda: None,
        host_identity=lambda: {
            "hostname": "tray-under-test",
            "boot_id": "b",
            "launcher_pid": 1,
            "source": "test",
        },
    )
    assert ctl.status_dict()["host"] == "tray-under-test"


def test_stream_mutation_between_sampling_and_finalisation_is_recorded_not_hidden(
    tmp_path, monkeypatch
):
    h = _harness(tmp_path, monkeypatch)
    h.script_normal_stop(flush_s=60, import_s=50)
    h.ctl.request_stop()
    stream = _stream(h, tmp_path, b"a" * 1000)
    h.ctl._track_stream()  # sample while Nsight is still writing (the V1 situation)
    with stream.open("ab") as fh:
        fh.write(b"tail" * 8)  # Nsight appends after the sample
    assert h.ctl.wait_for_report() == Phase.REPORT_WRITTEN
    h.procs.alive[NSYS_PID] = False  # session closes
    h.ctl.poll()
    st = h.ctl.status_dict()["stream"]
    final = st["final"]
    assert final["available"] and final["size"] == 1032
    assert final["sha256"] == hashlib.sha256(b"a" * 1000 + b"tail" * 8).hexdigest()
    assert (
        final["basis"].startswith("after the nsys session closed")
        and final["session_closed_at"] is not None
    )
    sizes = [o["size"] for o in st["observations"]]
    assert (
        sizes[0] == 1000 and sizes[-1] == 1032
    )  # the earlier observation is kept, with its time and phase
    assert all({"at", "phase", "size", "mtime_ns", "path"} <= set(o) for o in st["observations"])
    assert final["completeness"] == COMPLETENESS_NOTE and "NOT established" in COMPLETENESS_NOTE
    record = json.loads((Path(st["preserved_path"]).parent / RETENTION_NAME).read_text())
    assert (record["size"], record["sha256"]) == (final["size"], final["sha256"])
    assert (
        record["completeness"] == COMPLETENESS_NOTE
        and record["written_by"] == "lifecycle controller"
    )


def test_no_final_record_before_the_session_closes(tmp_path, monkeypatch):
    h = _harness(tmp_path, monkeypatch)
    h.script_normal_stop(flush_s=60, import_s=50)
    h.ctl.request_stop()
    _stream(h, tmp_path, b"a" * 1000)
    assert h.ctl.wait_for_report() == Phase.REPORT_WRITTEN
    assert h.ctl.status_dict()["stream"]["final"] is None  # nsys still alive: a writer may remain


def test_missing_stream_at_finalisation_is_recorded_with_a_warning(tmp_path, monkeypatch):
    h = _harness(tmp_path, monkeypatch)
    h.script_normal_stop(flush_s=60, import_s=50)
    h.ctl.request_stop()
    stream = _stream(h, tmp_path, b"a" * 1000)
    h.ctl.s.stream["preserved_path"] = False  # hard link failed earlier: stream unprotected
    assert h.ctl.wait_for_report() == Phase.REPORT_WRITTEN
    stream.unlink()  # Nsight removes its stream after the import
    h.procs.alive[NSYS_PID] = False
    h.ctl.poll()
    status = h.ctl.status_dict()
    assert status["stream"]["final"]["available"] is False
    assert any("final stream record" in w["message"] for w in status["warnings"])


def test_release_records_what_the_capture_was_doing(tmp_path, monkeypatch):
    h = _harness(tmp_path, monkeypatch)
    h.script_normal_stop(flush_s=60, import_s=500)
    h.ctl.request_stop()
    while h.ctl.poll() != Phase.IMPORTING:
        h.clock.sleep(1.0)
    h.ctl.control_path.write_text("release-gpus V2 forced release during import\n")
    h.ctl._apply_control()
    status = h.ctl.status_dict()
    ctx = status["gpu"]["release_context"]
    assert (
        ctx["phase"] == "importing" and ctx["nsys_alive"] is True and ctx["report_written"] is False
    )
    assert all(pid != NSYS_PID for pid, _ in h.procs.signals)  # the importer is never signalled
    assert status["control_commands"][-1]["result"] == "release requested"
