"""Concurrent and repeated profiler-stop requests: exactly one effective stop (V1 remediation, item B).

Separate fixture tests on the unchanged controller; the pinned repeated-call test stays in test_profiler_lifecycle.py.
"""

from __future__ import annotations

# ruff: noqa: E501

import threading

from alignment.profiler import procident
from alignment.profiler.lifecycle import Phase
from test_profiler_lifecycle import Harness, _policy


def test_concurrent_stop_requests_send_one_http_request_and_finalise_once(tmp_path, monkeypatch):
    monkeypatch.setattr(procident, "is_descendant_of", lambda pid, root: True)
    h = Harness(tmp_path, policy=_policy(), stop_return_at=1000.0 + 120)
    h.script_normal_stop(flush_s=60, import_s=50)
    n = 16
    barrier = threading.Barrier(n)

    def call():
        barrier.wait()
        h.ctl.request_stop()

    threads = [threading.Thread(target=call) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert h.ctl.wait_for_report() == Phase.REPORT_WRITTEN
    status = h.ctl.status_dict()
    assert h.stop.calls == 1
    assert status["http"]["requests_sent"] == 1
    assert status["http"]["duplicate_requests_suppressed"] == n - 1
    phases = [p["phase"] for p in status["phase_history"]]
    assert phases.count("stop_requested") == 1 and phases.count("report_written") == 1


def test_stop_requests_after_the_report_are_suppressed_and_do_not_finalise_again(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(procident, "is_descendant_of", lambda pid, root: True)
    h = Harness(tmp_path, policy=_policy(), stop_return_at=1000.0 + 120)
    h.script_normal_stop(flush_s=60, import_s=50)
    h.ctl.request_stop()
    assert h.ctl.wait_for_report() == Phase.REPORT_WRITTEN
    before = [p["phase"] for p in h.ctl.status_dict()["phase_history"]]
    for _ in range(3):
        h.ctl.request_stop()
    status = h.ctl.status_dict()
    assert h.stop.calls == 1 and status["http"]["duplicate_requests_suppressed"] == 3
    assert [p["phase"] for p in status["phase_history"]] == before


def test_a_single_stop_records_no_duplicate(tmp_path, monkeypatch):
    # the situation of the real V1 run: one request, no duplicate entry at all
    monkeypatch.setattr(procident, "is_descendant_of", lambda pid, root: True)
    h = Harness(tmp_path, policy=_policy(), stop_return_at=1000.0 + 120)
    h.script_normal_stop(flush_s=60, import_s=50)
    h.ctl.request_stop()
    assert h.ctl.wait_for_report() == Phase.REPORT_WRITTEN
    http = h.ctl.status_dict()["http"]
    assert http["requests_sent"] == 1 and http.get("duplicate_requests_suppressed", 0) == 0
