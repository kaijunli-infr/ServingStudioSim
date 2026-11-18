"""Controller evidence for a forced release: exact API-server SIGTERM time, importer identity and state after it."""

from __future__ import annotations

# ruff: noqa: E501

import signal

from alignment.profiler import procident
from alignment.profiler.lifecycle import Phase
from test_profiler_lifecycle import API_PID, NSYS_PID, Harness, _policy


def _released_during_import(tmp_path, monkeypatch):
    monkeypatch.setattr(procident, "is_descendant_of", lambda pid, root: True)
    h = Harness(tmp_path, policy=_policy(), stop_return_at=1000.0 + 120)
    h.script_normal_stop(flush_s=60, import_s=500)
    h.ctl.request_stop()
    while h.ctl.poll() != Phase.IMPORTING:
        h.clock.sleep(1.0)
    h.ctl.control_path.write_text("release-gpus V2 forced release during import\n")
    h.ctl._apply_control()
    return h, h.ctl.status_dict()


def test_api_server_sigterm_time_is_recorded_at_delivery_and_matches_its_action(
    tmp_path, monkeypatch
):
    h, status = _released_during_import(tmp_path, monkeypatch)
    stamp = status["gpu"]["api_server_sigterm"]
    assert stamp["pid"] == API_PID and stamp["outcome"] == "signalled"
    term = [
        a for a in status["actions"] if a.get("pid") == API_PID and a.get("action") == "SIGTERM"
    ]
    assert len(term) == 1 and term[0]["at"] == stamp["at"]
    assert status["gpu"]["release_requested_at"] <= stamp["at"]
    assert all("at" in a for a in status["actions"])
    assert h.procs.signals[0] == (API_PID, signal.SIGTERM)


def test_importer_identity_and_state_are_checked_after_the_signal_and_it_is_never_signalled(
    tmp_path, monkeypatch
):
    h, status = _released_during_import(tmp_path, monkeypatch)
    after = status["gpu"]["importer_after_signal"]
    assert [s["label"] for s in after] == [
        "after the API-server SIGTERM",
        "after the server termination pass",
    ]
    assert all(
        s["nsys_pid"] == NSYS_PID and s["identity_matches"] is True and s["phase"] == "importing"
        for s in after
    )
    assert after[0]["at"] >= status["gpu"]["api_server_sigterm"]["at"]
    assert all(pid != NSYS_PID for pid, _ in h.procs.signals)
    assert status["gpu"]["release_context"]["phase"] == "importing"
