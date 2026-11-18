"""Preflight: a retained inactive recovery stream is not an active profiler session; real conflicts still fail."""

from __future__ import annotations

# ruff: noqa: E501

import hashlib
import json
import os
from pathlib import Path

import pytest

from alignment.profiler import lifecycle, preflight
from test_profiler_preflight import _config, _fake_install

PLANNED = "dsv41_test_next_run"


def _preserve(
    tmp: Path, name: str = "old_run", data: bytes = b"q" * 4096, record: bool = True, **over
) -> Path:
    d = tmp / f"{name}.raw-preserve"
    d.mkdir()
    (d / f"{name}.qdstrm").write_bytes(data)
    if record:
        rec = {
            "schema_version": 1,
            "name": name,
            "stream_file": f"{name}.qdstrm",
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "session_closed_at": 1.0,
            "nsys": {"pid": 2**22 + 12345, "starttime": 1},
            "written_by": "lifecycle controller",
        }
        rec.update(over)
        (d / lifecycle.RETENTION_NAME).write_text(json.dumps(rec))
    return d


def test_retained_inactive_stream_is_recognised_and_left_untouched(tmp_path):
    d = _preserve(tmp_path)
    before = sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in d.iterdir())
    verdict = preflight.classify_preserve_dir(d, PLANNED)
    assert verdict["state"] == "retained_inactive" and len(verdict["sha256"]) == 64
    assert sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in d.iterdir()) == before


@pytest.mark.parametrize(
    "case",
    [
        "no_record",
        "collision",
        "sha_mismatch",
        "truncated",
        "open",
        "nsys_alive",
        "no_close_time",
        "missing_stream",
        "malformed_record",
    ],
)
def test_genuine_conflicts_and_unsafe_collisions_are_rejected(tmp_path, case):
    identity = lambda pid: None  # noqa: E731 - recorded nsys gone
    name, kw, holder = "old_run", {}, None
    if case == "no_record":
        d = _preserve(tmp_path, record=False)
    elif case == "collision":
        d = _preserve(tmp_path, name=PLANNED)
    elif case == "sha_mismatch":
        d = _preserve(tmp_path, sha256="0" * 64)
    elif case == "truncated":
        d = _preserve(tmp_path)
        (d / f"{name}.qdstrm").write_bytes(b"q" * 4000)
    elif case == "open":
        d = _preserve(tmp_path)
        holder = (d / f"{name}.qdstrm").open(
            "rb"
        )  # some process (this one) still has the stream open
    elif case == "nsys_alive":
        d = _preserve(tmp_path)

        class Live:
            starttime = 1

        identity = lambda pid: Live()  # noqa: E731
    elif case == "no_close_time":
        d = _preserve(tmp_path, session_closed_at=None)
    elif case == "missing_stream":
        d = _preserve(tmp_path)
        (d / f"{name}.qdstrm").unlink()
    else:
        d = _preserve(tmp_path, record=False)
        (d / lifecycle.RETENTION_NAME).write_text("{not json")
    try:
        verdict = preflight.classify_preserve_dir(d, PLANNED, identity=identity, **kw)
    finally:
        if holder:
            holder.close()
    assert verdict["state"] == "conflict", (case, verdict)


def test_location_check_passes_with_a_retained_stream_and_fails_with_a_live_one(
    tmp_path, monkeypatch
):
    cfg = _config(tmp_path, _fake_install(tmp_path))
    stream_dir = Path(cfg.nsys.tmp_dir)
    monkeypatch.setattr(preflight.procident, "identity", lambda pid: None)
    _preserve(stream_dir)

    def leftovers():
        return {c.name: c for c in preflight.check_locations(cfg)}

    checks = leftovers()
    assert checks["locations.stream_leftovers"].status == "PASS"
    assert "retained inactive" in checks["locations.stream_leftovers"].detail
    assert checks["locations.retained_streams"].status == "INFO"
    (stream_dir / "nsys-report-abcd.qdstrm").write_bytes(
        b"live"
    )  # Nsight's working stream = a running session
    assert leftovers()["locations.stream_leftovers"].status == "FAIL"
    (stream_dir / "nsys-report-abcd.qdstrm").unlink()
    _preserve(stream_dir, name="unknown", record=False)  # a preserve dir of unknown origin
    assert leftovers()["locations.stream_leftovers"].status == "FAIL"


def test_adoption_writes_a_record_only_for_a_closed_unopened_stream(tmp_path):
    d = _preserve(tmp_path, record=False)
    stream = d / "old_run.qdstrm"
    before = (stream.read_bytes(), stream.stat().st_mtime_ns)
    status = tmp_path / "lifecycle.json"
    base = {
        "name": "old_run",
        "phase": "report_written",
        "session_closed_at": 5.0,
        "stream": {"preserved_path": str(stream)},
        "nsys": {"identity": {"pid": 2**22 + 12345, "starttime": 1}},
    }
    status.write_text(json.dumps({**base, "session_closed_at": None}))
    with pytest.raises(ValueError, match="session-closed"):
        preflight.adopt_retained_stream(d, status, identity=lambda pid: None)
    status.write_text(json.dumps(base))
    with stream.open("rb"):
        with pytest.raises(ValueError, match="open in process"):
            preflight.adopt_retained_stream(d, status, identity=lambda pid: None)
    rec = preflight.adopt_retained_stream(d, status, identity=lambda pid: None)
    assert (
        rec["written_by"].startswith("operator adoption")
        and "NOT established" in rec["completeness"]
    )
    assert (
        stream.read_bytes(),
        stream.stat().st_mtime_ns,
    ) == before  # the stream itself is untouched
    assert (
        preflight.classify_preserve_dir(d, PLANNED, identity=lambda pid: None)["state"]
        == "retained_inactive"
    )
    assert preflight.main(["--adopt-retained", str(tmp_path / "nope"), str(status)]) == 1
    assert os.path.exists(d / lifecycle.RETENTION_NAME)
