"""V4 run supervisor: the unchanged V3 supervisor plus the reservation guard. The guard is a cause like the others (one
response, registry cleanup outside finalisation); the V3 behaviour is untouched; the wrapper signals nothing."""

from __future__ import annotations

# ruff: noqa: E501

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECKS = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/checks"
SCRIPT = CHECKS / "e1_supervisor_v4.py"
LINE = "release-gpus V4 resource-limit abort (registered supervisor response)"
GUARD = 4680.0


def _load():
    spec = importlib.util.spec_from_file_location("e1_supervisor_v4", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V4 = _load()


def stamp(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


class World:
    """A run directory and a fake clock that starts `age` seconds after the reservation's creation."""

    def __init__(self, tmp: Path, *, age: float, phase="capturing", stop_requested=False, reservation: dict | str | None = "default"):
        self.run = tmp / "root" / "e1_v4_scale"
        self.nsys = self.run / "profile" / "nsys"
        self.nsys.mkdir(parents=True)
        (tmp / "root" / "pids").mkdir()
        self.registry, self.out, self.res = tmp / "root" / "pids" / "owned.tsv", self.run / "supervisor.json", self.run / "reservation.json"
        self.created = 1_800_000_000.0
        self.t, self.cleanups, self.steps = self.created + age, [], []
        if reservation == "default":
            reservation = {"pod": "p", "node": "n", "phase": "Running", "creation_timestamp": stamp(self.created)}
        if reservation is not None:
            self.res.write_text(reservation if isinstance(reservation, str) else json.dumps(reservation))
        self.lc = {"phase": phase, "launch_started_at": self.t - 10, "stop_requested_at": self.t - 5 if stop_requested else None, "gpu": {}, "control_commands": [], "updated_at": self.t}
        self.beat()
        (self.nsys / "lifecycle.json").write_text(json.dumps(self.lc))

    def beat(self):
        for n in ("mem_observations.jsonl", "storage_observations.jsonl", "gpu_observations.jsonl"):
            p = self.run / n
            p.write_text("{}\n")
            os.utime(p, (self.t, self.t))

    def sleep(self, s):
        self.t += s
        self.lc["updated_at"] = self.t
        control = self.nsys / "lifecycle.control"
        if control.exists() and not self.lc["gpu"].get("release_requested_at"):
            self.lc["gpu"] = {"release_requested_at": self.t, "release_reason": f"operator: {control.read_text().strip()}"}
        (self.nsys / "lifecycle.json").write_text(json.dumps(self.lc))
        self.beat()
        if self.steps:
            self.steps.pop(0)()

    def cleanup(self, registry, grace, log):
        self.cleanups.append((registry, grace))
        return {"command": ["cleanup"], "rc": 0, "result_line": "RESULT survivors 0", "log": str(log), "started_at": self.t, "ended_at": self.t}

    def go(self, max_s=6.0, guard=GUARD):
        rc = V4.supervise(self.run, self.registry, self.out, LINE, self.res, guard, gpu_held_ceiling_s=2700.0, ack_s=10.0, stale_s=60.0, cleanup_grace=30.0, interval=1.0, max_s=max_s,
                          now=lambda: self.t, sleep=self.sleep, cleanup=self.cleanup)
        return rc, json.loads(self.out.read_text())


def test_before_the_guard_nothing_happens(tmp_path):
    rc, rec = World(tmp_path, age=GUARD - 60).go()
    assert rc == 0 and rec["response"] is None and rec["causes"] == [] and rec["complete"] is True and rec["monitors_effective"] is True


def test_exactly_at_the_guard_is_not_a_cause_and_just_past_it_is(tmp_path):
    w = World(tmp_path / "at", age=GUARD - 0.0)
    rc, rec = w.go(max_s=0.0)  # one sample exactly at the guard
    assert rc == 0 and rec["response"] is None
    w = World(tmp_path / "past", age=GUARD + 0.5)
    rc, rec = w.go(max_s=0.0)
    assert rc == 5 and [c["cause"] for c in rec["causes"]] == ["reservation_guard"] and rec["causes"][0]["reservation_age_s"] == pytest.approx(GUARD + 0.5) and rec["causes"][0]["limit_s"] == GUARD


def test_guard_after_the_gpu_release_runs_the_registry_cleanup_once(tmp_path):
    w = World(tmp_path, age=GUARD - 2.5, phase="report_written", stop_requested=True)  # CPU-only phase: export and parse
    w.lc["gpu"] = {"released_at": w.t - 100}
    rc, rec = w.go()
    steps = rec["response"]["steps"]
    assert rc == 5 and [s["kind"] for s in steps] == ["R2 registry cleanup"] and len(w.cleanups) == 1 and w.cleanups[0] == (w.registry, 30.0)
    assert rec["causes"][0]["kind"] == "run not ended when the reservation guard passed" and rec["signals_sent_by_supervisor"] == 0 and rec["monitors_effective"] is True
    assert not (w.nsys / "lifecycle.control").exists()


def test_guard_before_any_stop_request_runs_the_registry_cleanup(tmp_path):
    w = World(tmp_path, age=GUARD + 1)
    rc, rec = w.go()
    assert rc == 5 and [s["kind"] for s in rec["response"]["steps"]] == ["R2 registry cleanup"] and "not in its finalisation loop" in rec["response"]["steps"][0]["why"]


def test_guard_during_finalisation_goes_through_the_controller(tmp_path):
    w = World(tmp_path, age=GUARD + 1, phase="importing", stop_requested=True)
    rc, rec = w.go()
    steps = rec["response"]["steps"]
    assert rc == 5 and [s["kind"] for s in steps] == ["R1 control line"] and steps[0]["acknowledged"] is True and w.cleanups == []
    assert (w.nsys / "lifecycle.control").read_text() == LINE + "\n"


@pytest.mark.parametrize("reservation", [None, "not json", {"pod": "p"}, {"creation_timestamp": "yesterday"}])
def test_unreadable_reservation_record_is_a_cause(tmp_path, reservation):
    rc, rec = World(tmp_path, age=10, reservation=reservation).go(max_s=0.0)
    assert rc == 5 and rec["causes"][0]["cause"] == "reservation_guard" and "unreadable" in rec["causes"][0]["kind"]


def test_v3_causes_still_work_and_are_recorded_together_with_the_guard(tmp_path):
    w = World(tmp_path, age=GUARD + 5)
    (w.run / "STORAGE-BREACH").write_text(json.dumps({"kind": "persistent_storage_above_ceiling", "value_gb": 18.2, "limit_gb": 18, "at": w.t}) + "\n")
    rc, rec = w.go()
    assert rc == 5 and sorted(c["cause"] for c in rec["causes"]) == ["breach", "reservation_guard"] and len(w.cleanups) == 1
    w2 = World(tmp_path / "ceiling", age=100)
    w2.lc["launch_started_at"] = w2.t - 2701
    (w2.nsys / "lifecycle.json").write_text(json.dumps(w2.lc))
    rc, rec = w2.go(max_s=0.0)
    assert rc == 5 and [c["cause"] for c in rec["causes"]] == ["gpu_held_ceiling_before_stop"]


def test_the_v3_supervisor_module_is_not_modified_by_the_wrapper(tmp_path):
    spec = importlib.util.spec_from_file_location("e1_supervisor_plain", CHECKS / "e1_supervisor.py")
    plain = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plain)
    World(tmp_path, age=GUARD + 5).go()
    assert plain.ceiling_cause({"launch_started_at": 0.0, "stop_requested_at": None}, 10.0, 2700.0) == []  # no reservation cause leaks into a fresh import
    scripts = (CHECKS.parent / "SCRIPTS_v3scale.sha256").read_text()
    import hashlib

    assert f"{hashlib.sha256((CHECKS / 'e1_supervisor.py').read_bytes()).hexdigest()}  checks/e1_supervisor.py" in scripts


def test_wrapper_source_has_no_signalling_or_process_call():
    code = SCRIPT.read_text().split('"""', 2)[2]
    assert not any(x in code for x in ("os.kill", "import signal", "killpg", "send_signal", ".terminate(", ".kill(", "subprocess"))


def test_command_line_requires_the_reservation_and_a_positive_guard(tmp_path):
    w = World(tmp_path, age=10)
    base = [sys.executable, "-I", str(SCRIPT), "--run", str(w.run), "--registry", str(w.registry), "--out", str(w.out), "--control-line", LINE]
    assert subprocess.run(base, capture_output=True).returncode == 2
    assert subprocess.run([*base, "--reservation", str(w.res), "--reservation-guard-s", "0"], capture_output=True).returncode == 2
    stop = tmp_path / "stop"
    stop.write_text("")
    w.res.write_text(json.dumps({"creation_timestamp": stamp(time.time() - 30)}))
    for n in ("mem_observations.jsonl", "storage_observations.jsonl", "gpu_observations.jsonl"):
        (w.run / n).write_text("{}\n")
    p = subprocess.run([*base, "--reservation", str(w.res), "--reservation-guard-s", "4680", "--stop-file", str(stop), "--max-s", "5"], capture_output=True, text=True)
    rec = json.loads(w.out.read_text())
    assert p.returncode == 0 and rec["response"] is None and rec["complete"] is True
