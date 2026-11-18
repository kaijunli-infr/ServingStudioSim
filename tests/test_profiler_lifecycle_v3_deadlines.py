"""The unchanged stop controller under the V3 policy (soft 300 s, hard 600 s): the soft warning is informational and never
moves the hard deadline; the total is counted from the original stop request; the control file is read only in the
finalisation loop; the supervisor's registered release line is an identity-verified server-only stop."""

from __future__ import annotations

# ruff: noqa: E501

import signal

from alignment.profiler import procident
from alignment.profiler.lifecycle import Phase

from test_profiler_lifecycle import API_PID, NSYS_PID, Harness, _policy

LINE = "release-gpus V3 resource-limit abort (registered supervisor response)"


def _v3_policy(**kw):
    base = dict(soft_deadline_s=300.0, hard_deadline_s=600.0, stall_timeout_s=300.0, poll_interval_s=1.0, session_close_timeout_s=300.0, server_term_grace_s=60.0, max_extension_s=120.0, gpu_held_ceiling_s=2700.0)
    base.update(kw)
    return _policy(**base)


def test_soft_warning_does_not_move_the_hard_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(procident, "is_descendant_of", lambda pid, root: True)
    h = Harness(tmp_path, policy=_v3_policy(session_close_timeout_s=20), stop_return_at=None)
    h.script_normal_stop(flush_s=250, import_s=5000, with_report=False)  # the import would never finish in time
    h.ctl.request_stop()
    stop_at = h.ctl.status_dict()["stop_requested_at"]
    phase = h.ctl.wait_for_report()
    st = h.ctl.status_dict()
    kinds = [(e["kind"], e["elapsed_s"]) for e in st["escalations"]]
    assert [k for k, _ in kinds] == ["soft_deadline", "hard_deadline"]
    assert 300 < kinds[0][1] <= 302 and 600 < kinds[1][1] <= 602  # warning at 300 s, action at 600 s, both from the stop request
    assert st["deadlines"]["hard_at"] == stop_at + 600.0 and st["deadlines"]["soft_at"] == stop_at + 300.0 and st["deadlines"]["hard_extension_s"] == 0.0
    assert st["gpu"]["release_reason"] == "hard deadline" and 600 < st["gpu"]["release_requested_at"] - stop_at <= 602
    assert (API_PID, signal.SIGTERM) in h.procs.signals and NSYS_PID not in {pid for pid, _ in h.procs.signals}
    assert st["control_commands"] == [] and phase == Phase.IMPORTING


def test_finalisation_between_the_soft_and_the_hard_deadline_completes_with_only_the_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(procident, "is_descendant_of", lambda pid, root: True)
    h = Harness(tmp_path, policy=_v3_policy(), stop_return_at=1000.0 + 392)
    h.script_normal_stop(flush_s=250, import_s=140)
    h.ctl.request_stop()
    phase = h.ctl.wait_for_report()
    st = h.ctl.status_dict()
    assert phase == Phase.REPORT_WRITTEN and [e["kind"] for e in st["escalations"]] == ["soft_deadline"]
    assert h.procs.signals == [] and st["gpu"].get("release_requested_at") is None and st["deadlines"]["hard_extension_s"] == 0.0
    report_at = next(p["at"] for p in st["phase_history"] if p["phase"] == "report_written")
    assert 390 <= report_at - st["stop_requested_at"] <= 396


def test_the_total_is_counted_from_the_original_stop_request(tmp_path, monkeypatch):
    monkeypatch.setattr(procident, "is_descendant_of", lambda pid, root: True)
    h = Harness(tmp_path, policy=_v3_policy(session_close_timeout_s=20), stop_return_at=None)
    h.script_normal_stop(flush_s=250, import_s=5000, with_report=False)
    h.ctl.request_stop()
    first = dict(h.ctl.status_dict()["deadlines"])
    stop_at = h.ctl.status_dict()["stop_requested_at"]
    h.at(100, h.ctl.request_stop)  # later stop calls are no-ops: they must not restart any clock
    h.at(450, h.ctl.request_stop)
    h.ctl.wait_for_report()
    st = h.ctl.status_dict()
    assert st["stop_requested_at"] == stop_at and st["deadlines"]["hard_at"] == first["hard_at"] and st["http"]["requests_sent"] == 1 and st["http"]["duplicate_requests_suppressed"] == 2
    hard = next(e for e in st["escalations"] if e["kind"] == "hard_deadline")
    assert 600 < hard["at"] - stop_at <= 602


def test_registered_release_line_during_the_import_is_a_server_only_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(procident, "is_descendant_of", lambda pid, root: True)
    h = Harness(tmp_path, policy=_v3_policy(session_close_timeout_s=30), stop_return_at=None)
    h.script_normal_stop(flush_s=100, import_s=200, with_report=False)
    h.at(150, lambda: h.ctl.control_path.write_text(LINE + "\n"))
    h.ctl.request_stop()
    h.ctl.wait_for_report()
    st = h.ctl.status_dict()
    cmd = st["control_commands"][0]
    assert cmd["command"] == LINE and cmd["result"] == "release requested" and cmd["phase"] == "importing"
    assert 150 <= cmd["at"] - st["stop_requested_at"] <= 152  # consumed within one poll
    assert {pid for pid, _ in h.procs.signals} <= {API_PID, 4200, 4301, 4302, 4303, 4304} and (API_PID, signal.SIGTERM) in h.procs.signals
    assert NSYS_PID not in {pid for pid, _ in h.procs.signals} and st["gpu"]["release_reason"] == f"operator: {LINE}"


def test_control_file_is_read_only_inside_the_finalisation_loop(tmp_path, monkeypatch):
    """Why the supervisor uses the registry cleanup before a stop request: a line written during startup or capture is
    not acted on until the controller enters wait_for_report()."""
    monkeypatch.setattr(procident, "is_descendant_of", lambda pid, root: True)
    h = Harness(tmp_path, policy=_v3_policy(session_close_timeout_s=30), stop_return_at=None)
    h.ctl.control_path.write_text(LINE + "\n")
    for _ in range(50):  # the capture phase: the launcher only polls
        h.ctl.poll()
        h.clock.sleep(1.0)
    assert h.ctl.status_dict()["control_commands"] == [] and h.procs.signals == []
    h.script_normal_stop(flush_s=100, import_s=200, with_report=False)
    h.scheduled = [(t + 50.0, fn) for t, fn in h.scheduled]
    h.ctl.request_stop()
    h.ctl.wait_for_report()
    st = h.ctl.status_dict()
    assert st["control_commands"][0]["command"] == LINE and st["control_commands"][0]["at"] - st["stop_requested_at"] <= 2  # delayed delivery: at the first poll after the stop
