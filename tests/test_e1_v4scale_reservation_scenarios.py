"""V4 reservation guard, budget and response latch, scenario by scenario, on the ACTUAL supervisor code (checks/e1_supervisor_v4.py
with the helpers of the unchanged checks/e1_supervisor.py) with the ACTUAL pinned registry cleanup (checks/e1_pids.py) acting on real stand-in
processes. CPU only: the stand-ins are `sleep` processes started through the registry tool; nothing touches a GPU.

In every outcome: the only way a process is ended is the pinned, identity-verified cleanup of the registered tree; the
supervisor sends no signal; processes that are not in the registry (the stand-ins of the importer and of the detached Nsight
agent) are alive afterwards. An earlier response never leaves the reservation guard unenforced, and there is never more than
one release line or more than one cleanup.
"""

from __future__ import annotations

# ruff: noqa: E501

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
E1 = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1"
CHECKS = E1 / "checks"
LINE = "release-gpus V4 resource-limit abort (registered supervisor response)"
D = json.loads((E1 / "E1-V4-SCALE-INPUTS.json").read_text())
GUARD, CEILING = float(D["reservation"]["supervisor_guard_s"]), float(D["ceilings"]["gpu_held_ceiling_s"])
LATEST = float(D["reservation"]["launch_max_age_s"])


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_scenarios"), CHECKS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V4 = _load("e1_supervisor_v4.py")


def stamp(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def alive(pid: int) -> bool:
    try:
        return open(f"/proc/{pid}/stat").read().rsplit(") ", 1)[-1][0] not in "ZX"
    except (OSError, IndexError):
        return False


class Run:
    """A run directory with a real registry of real stand-in processes, an unregistered importer stand-in, and a fake clock."""

    def __init__(self, tmp: Path, *, age: float, launch_age: float = 300.0, phase: str = "capturing", stop_requested: bool = False, released: bool = False, ack: bool = True, release_completes_after: float | None = None):
        self.ack, self.release_completes_after = ack, release_completes_after  # the controller's side of a release line: acknowledgement, and the release completing (None: it stalls)
        self.root = tmp / "root"
        self.run = self.root / "e1_v4_scale"
        self.nsys = self.run / "profile" / "nsys"
        self.nsys.mkdir(parents=True)
        (self.root / "pids").mkdir()
        self.registry, self.out, self.res = self.root / "pids" / "owned.tsv", self.run / "supervisor.json", self.run / "reservation.json"
        self.created = 1_800_000_000.0
        self.t = self.created + age
        self.res.write_text(json.dumps({"pod": "p", "node": "n", "phase": "Running", "creation_timestamp": stamp(self.created)}))
        self.lc = {"phase": phase, "launch_started_at": self.created + launch_age, "stop_requested_at": self.t - 400 if stop_requested else None, "gpu": {"released_at": self.t - 300} if released else {}, "control_commands": [], "updated_at": self.t}
        self.silent: set[str] = set()
        self.registered = [self._spawn(label) for label in ("driver:e1_v4_scale", "memwatch:e1_v4_scale", "storagewatch:e1_v4_scale")]
        self.importer = subprocess.Popen(["sleep", "120"], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # not registered, not a descendant of a registered process
        self.agent = subprocess.Popen(["sleep", "120"], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # the detached Nsight agent's stand-in: not registered either
        self.stream = tmp / "stream.raw-preserve"
        self.stream.write_bytes(b"preserved stream " * 1000)
        self.stream_sha = hashlib.sha256(self.stream.read_bytes()).hexdigest()
        self.events: list = []
        self.sync()

    def _spawn(self, label: str) -> int:
        p = subprocess.run([sys.executable, "-I", str(CHECKS / "e1_pids.py"), "spawn", "--registry", str(self.registry), "--label", label, "--", "sleep", "120"], capture_output=True, text=True, check=True)
        return int(re.search(r'"pid": (\d+)', p.stdout).group(1))

    def sync(self) -> None:
        for n in ("mem_observations.jsonl", "storage_observations.jsonl", "gpu_observations.jsonl"):
            if n not in self.silent:
                (self.run / n).write_text("{}\n")
                os.utime(self.run / n, (self.t, self.t))
        self.lc["updated_at"] = self.t
        control = self.nsys / "lifecycle.control"
        if control.exists() and self.ack and not self.lc["gpu"].get("release_requested_at"):  # the controller's acknowledgement of a control line
            self.lc["gpu"] = {**self.lc["gpu"], "release_requested_at": self.t, "release_reason": f"operator: {control.read_text().strip().splitlines()[0]}"}
        req = self.lc["gpu"].get("release_requested_at")
        if control.exists() and req is not None and self.release_completes_after is not None and not self.lc["gpu"].get("released_at") and self.t - req >= self.release_completes_after:
            self.lc["gpu"]["released_at"] = self.t  # the server is gone and the GPUs are free
        (self.nsys / "lifecycle.json").write_text(json.dumps(self.lc))

    def sleep(self, s: float) -> None:
        self.t += s
        while self.events and self.events[0][0] <= self.t - self.created:
            self.events.pop(0)[1]()
        self.sync()

    def at(self, age: float, action) -> None:
        self.events.append((age, action))
        self.events.sort(key=lambda e: e[0])

    def go(self, max_s: float = 6.0, stop_file: Path | None = None):
        rc = V4.supervise(self.run, self.registry, self.out, LINE, self.res, GUARD, gpu_held_ceiling_s=CEILING, ack_s=10.0, stale_s=60.0, cleanup_grace=1.0, interval=1.0, max_s=max_s, stop_file=stop_file,
                          now=lambda: self.t, sleep=self.sleep)  # no `cleanup` argument: the supervisor's own pinned cleanup runs
        return rc, json.loads(self.out.read_text())

    def cleanup_log(self) -> dict:
        return json.loads((self.root / "cpu" / "supervisor_cleanup.jsonl").read_text().splitlines()[-1])

    def close(self) -> None:
        for own in (self.importer, self.agent):  # the test's own children, by their handles
            own.terminate()
            own.wait(timeout=10)
        left = [p for p in self.registered if alive(p)]
        if left:  # only through the pinned verified cleanup of this test's registry
            subprocess.run([sys.executable, "-I", str(CHECKS / "e1_pids.py"), "cleanup", "--registry", str(self.registry), "--execute", "--grace", "2", "--kill9"], capture_output=True)


@pytest.fixture
def make(tmp_path):
    made: list[Run] = []

    def _make(**kw) -> Run:
        made.append(Run(tmp_path / f"w{len(made)}", **kw))
        return made[-1]

    yield _make
    for w in made:
        w.close()


def assert_frozen_cleanup_only(w: Run, rec: dict) -> None:
    """The response was the pinned registry cleanup; it signalled registered processes only; the importer's stand-in and the stream are untouched."""
    steps = rec["response"]["steps"]
    assert [s["kind"] for s in steps] == ["R2 registry cleanup"] and rec["signals_sent_by_supervisor"] == 0
    assert steps[0]["command"] == [str(CHECKS / "e1_pids.py"), "cleanup", "--registry", str(w.registry), "--execute", "--grace", "1.0", "--kill9", "--log", str(w.root / "cpu" / "supervisor_cleanup.jsonl")]
    assert steps[0]["rc"] in (0, 2) and "survivors 0" in steps[0]["result_line"]  # 2 = a lookalike elsewhere on the host was reported (never signalled)
    assert_cleanup_log(w, rec)
    assert not (w.nsys / "lifecycle.control").exists() and rec["latch"] == {"release_lines_written": 0, "registry_cleanups_started": 1}


def assert_cleanup_log(w: Run, rec: dict) -> None:
    """Exactly one cleanup ran; it signalled the registered processes only, each with a verified identity; importer, agent and stream are untouched."""
    lines = (w.root / "cpu" / "supervisor_cleanup.jsonl").read_text().splitlines()
    log = json.loads(lines[-1])
    signalled = {a["pid"] for a in log["actions"] if a["action"] in ("SIGTERM", "SIGKILL")}
    assert len(lines) == 1 and rec["latch"]["registry_cleanups_started"] == 1 and rec["signals_sent_by_supervisor"] == 0
    assert signalled == set(w.registered) and not log["plan"]["refused"] and log["survivors"] == []
    assert not any(alive(p) for p in w.registered)
    assert w.importer.poll() is None and w.agent.poll() is None and not {w.importer.pid, w.agent.pid} & signalled  # importer and Nsight agent isolation: no signal of any kind
    assert not {w.importer.pid, w.agent.pid} & {o["pid"] for o in log["plan"]["owned"]}
    assert hashlib.sha256(w.stream.read_bytes()).hexdigest() == w.stream_sha


def assert_release_then_cleanup(w: Run, rec: dict) -> dict:
    """An acknowledged release line, then, at the reservation guard, the one registry cleanup."""
    resp = rec["response"]
    assert [s["kind"] for s in resp["steps"]] == ["R1 control line", "R2 registry cleanup"] and resp["steps"][0]["acknowledged"] is True and resp["steps"][1]["state"] == "finished"
    assert rec["latch"] == {"release_lines_written": 1, "registry_cleanups_started": 1} and (w.nsys / "lifecycle.control").read_text() == LINE + "\n"  # one line, once
    esc = resp["escalation"]
    assert [c["cause"] for c in esc["causes"]] == ["reservation_guard"] and GUARD < esc["causes"][0]["reservation_age_s"] <= GUARD + 1 and esc["seconds_from_guard_seen"] <= 45
    assert esc["completed_at"] - w.created <= GUARD + 46 and rec["complete"] is True and rec["ended_at"] >= esc["completed_at"]
    assert "the reservation guard passed after the earlier release request" in resp["steps"][1]["why"]
    assert_cleanup_log(w, rec)
    return esc


def assert_nothing_ended(w: Run) -> None:
    assert all(alive(p) for p in w.registered) and w.importer.poll() is None and w.agent.poll() is None and not (w.root / "cpu" / "supervisor_cleanup.jsonl").exists()


def breach(w: Run, kind: str = "importer_rss_above_ceiling") -> None:
    (w.run / "MEMORY-BREACH").write_text(json.dumps({"kind": kind, "value_gb": 129.0, "limit_gb": 128, "at": w.t}) + "\n")


# ---- GPU-held ceiling


def test_gpu_held_ceiling_reached_before_any_stop_request(make):
    w = make(age=LATEST + 20 + CEILING - 2, launch_age=LATEST + 20)  # launched at the latest permitted instant; 2 s before the ceiling
    rc, rec = w.go()
    assert rc == 5 and [c["cause"] for c in rec["causes"]] == ["gpu_held_ceiling_before_stop"] and rec["causes"][0]["limit_s"] == CEILING and CEILING < rec["causes"][0]["held_s"] <= CEILING + 2
    assert rec["response"]["decided_at"] - w.created < GUARD  # the ceiling acts long before the reservation guard
    assert_frozen_cleanup_only(w, rec)


def test_gpu_held_ceiling_exactly_reached_is_not_yet_a_cause(make):
    w = make(age=300 + CEILING, launch_age=300)
    rc, rec = w.go(max_s=0.0)
    assert rc == 0 and rec["response"] is None
    assert_nothing_ended(w)


def test_gpu_held_ceiling_after_the_stop_request_is_the_controllers(make):
    w = make(age=300 + CEILING + 30, launch_age=300, phase="importing", stop_requested=True)
    rc, rec = w.go()
    assert rc == 0 and rec["response"] is None and rec["causes"] == []  # the unchanged controller enforces its ceiling during finalisation (tests/test_profiler_lifecycle_v3_deadlines.py)
    assert_nothing_ended(w)


# ---- the reservation guard


def test_normal_completion_just_before_the_guard(make, tmp_path):
    w = make(age=GUARD - 3, phase="report_written", stop_requested=True, released=True)
    stop = tmp_path / "samplers.stop"
    w.at(GUARD - 1, lambda: stop.write_text(""))  # the driver ends the run 1 s before the guard
    rc, rec = w.go(max_s=30.0, stop_file=stop)
    assert rc == 0 and rec["response"] is None and rec["causes"] == [] and rec["complete"] is True and rec["ended_at"] - w.created == pytest.approx(GUARD - 1)
    assert_nothing_ended(w)


def test_run_ending_exactly_at_the_guard_gets_no_response(make, tmp_path):
    w = make(age=GUARD - 2, phase="report_written", stop_requested=True, released=True)
    stop = tmp_path / "samplers.stop"
    w.at(GUARD, lambda: stop.write_text(""))
    rc, rec = w.go(max_s=30.0, stop_file=stop)
    assert rc == 0 and rec["response"] is None and rec["ended_at"] - w.created == pytest.approx(GUARD)
    assert_nothing_ended(w)


def test_export_still_running_when_the_guard_passes(make):
    w = make(age=GUARD - 2, phase="report_written", stop_requested=True, released=True)  # CPU-only phase: export and parse
    rc, rec = w.go()
    c = rec["causes"][0]
    assert rc == 5 and [x["cause"] for x in rec["causes"]] == ["reservation_guard"] and c["kind"] == "run not ended when the reservation guard passed" and GUARD < c["reservation_age_s"] <= GUARD + 1
    assert rec["response"]["stop_requested"] is True and rec["response"]["phase"] == "report_written" and "not in its finalisation loop" in rec["response"]["steps"][0]["why"]
    assert rec["response"]["completed_at"] - w.created <= GUARD + 45  # the closeout and deletion budget keeps 675 s
    assert_frozen_cleanup_only(w, rec)


def test_import_still_running_after_a_forced_release_when_the_guard_passes(make):
    """The controller released the GPUs at a deadline and left the importer untouched; it is still importing at the guard."""
    w = make(age=GUARD - 2, phase="importing", stop_requested=True)
    w.lc["gpu"] = {"release_requested_at": w.t - 900, "release_reason": "hard deadline", "released_at": w.t - 880}
    w.sync()
    rc, rec = w.go()
    assert rc == 5 and [x["cause"] for x in rec["causes"]] == ["reservation_guard"]
    assert_frozen_cleanup_only(w, rec)  # the registered tree is ended; the importer is not in it and is not signalled


def test_guard_during_finalisation_ends_the_run_through_the_registry_cleanup(make):
    """Not reachable under the launch rule (latest launch + ceiling + release < guard). The release line would only release the GPUs;
    the guard requires the run to end, so the response is the registry cleanup and no control line is written."""
    w = make(age=GUARD + 1, phase="importing", stop_requested=True)
    rc, rec = w.go()
    assert rc == 5 and [c["cause"] for c in rec["causes"]] == ["reservation_guard"] and "the release line would release the GPUs only" in rec["response"]["steps"][0]["why"]
    assert_frozen_cleanup_only(w, rec)


# ---- the latch: an earlier response does not leave the reservation guard unenforced


def test_earlier_release_acknowledged_and_completing_and_the_run_ends_before_the_guard(make, tmp_path):
    w = make(age=GUARD - 120, phase="importing", stop_requested=True, release_completes_after=3.0)
    breach(w)
    stop = tmp_path / "samplers.stop"
    w.at(GUARD - 100, lambda: w.lc.update(phase="report_written"))  # the importer finished after the release
    w.at(GUARD - 20, lambda: stop.write_text(""))  # the driver ends the run before the guard
    rc, rec = w.go(max_s=300.0, stop_file=stop)
    resp = rec["response"]
    assert rc == 5 and [s["kind"] for s in resp["steps"]] == ["R1 control line"] and resp["steps"][0]["acknowledged"] is True and resp["steps"][0]["latency_s"] <= 10 and "escalation" not in resp
    assert rec["latch"] == {"release_lines_written": 1, "registry_cleanups_started": 0} and (w.nsys / "lifecycle.control").read_text() == LINE + "\n"
    assert rec["release_after_response"]["stalled"] is False and rec["release_after_response"]["gpu_released_at"] is not None and rec["complete"] is True and rec["ended_at"] - w.created < GUARD
    assert_nothing_ended(w)  # the release went through the controller; the supervisor ended nothing


def test_earlier_release_acknowledged_and_completed_but_the_run_is_still_going_at_the_guard(make):
    w = make(age=GUARD - 30, phase="importing", stop_requested=True, release_completes_after=3.0)
    breach(w)
    w.at(GUARD - 15, lambda: w.lc.update(phase="report_written"))  # the import finished; the launcher's export and parse continue
    rc, rec = w.go(max_s=120.0)
    assert rc == 5 and [c["cause"] for c in rec["causes"]] == ["breach"]
    esc = assert_release_then_cleanup(w, rec)
    assert esc["gpu_released_at"] is not None and esc["gpu_released_at"] < esc["decided_at"] and esc["phase"] == "report_written"


def test_earlier_release_acknowledged_but_stalled(make):
    """The controller took the line and the release never completed. Reported as stalled (monitor: ABORT-CONDITION) and left to the
    controller's own deadlines; at the reservation guard the supervisor ends the run."""
    w = make(age=GUARD - 200, phase="importing", stop_requested=True, release_completes_after=None)
    breach(w)
    seen = []
    w.at(GUARD - 10, lambda: seen.append((json.loads(w.out.read_text()), all(alive(p) for p in w.registered), (w.root / "cpu" / "supervisor_cleanup.jsonl").exists())))
    rc, rec = w.go(max_s=300.0)
    before, registered_alive, cleanup_log_exists = seen[0]
    assert before["release_after_response"]["stalled"] is True and before["release_after_response"]["seconds_since_acknowledgement"] > 150 and [s["kind"] for s in before["response"]["steps"]] == ["R1 control line"]
    assert registered_alive and not cleanup_log_exists  # before the guard nothing was ended by the supervisor
    esc = assert_release_then_cleanup(w, rec)
    assert rc == 5 and esc["gpu_released_at"] is None and esc["release_requested_at"] is not None
    assert "ABORT-CONDITION the release requested by the supervisor has not completed" in (E1 / "monitor_e1_v4scale.sh").read_text()


def test_missing_acknowledgement_runs_the_cleanup_once_and_the_guard_adds_nothing(make):
    w = make(age=GUARD - 8, phase="importing", stop_requested=True, ack=False)
    breach(w)
    rc, rec = w.go(max_s=120.0)
    steps = rec["response"]["steps"]
    assert rc == 5 and [s["kind"] for s in steps] == ["R1 control line", "R2 registry cleanup"] and steps[0]["acknowledged"] is False and "not acknowledged within 10.0 s" in steps[1]["why"]
    assert steps[1]["decided_at"] - steps[0]["written_at"] > 10.0  # the cleanup never overlaps the wait for the acknowledgement
    assert steps[1]["decided_at"] - w.created > GUARD and "escalation" not in rec["response"]  # the guard passed during the wait: still one cleanup, not two
    assert rec["latch"] == {"release_lines_written": 1, "registry_cleanups_started": 1} and (w.nsys / "lifecycle.control").read_text() == LINE + "\n"
    assert_cleanup_log(w, rec)


def test_simultaneous_breach_and_reservation_deadline_during_finalisation(make):
    w = make(age=GUARD + 0.5, phase="importing", stop_requested=True)
    breach(w)
    rc, rec = w.go()
    assert rc == 5 and sorted(c["cause"] for c in rec["causes"]) == ["breach", "reservation_guard"] and len({c["seen_at"] for c in rec["causes"]}) == 1
    assert "the reservation guard requires the run to end; the release line would release the GPUs only" in rec["response"]["steps"][0]["why"]
    assert_frozen_cleanup_only(w, rec)  # one response: the cleanup; no control line; importer and agent untouched


def test_importer_still_active_when_the_guard_escalates_an_earlier_release(make):
    """As V2's outcome A: the server was stopped through the release line, the importer keeps importing. At the guard the registered tree
    is ended; the importer and the detached agent are not in it, receive no signal and keep running; the stream stays."""
    w = make(age=GUARD - 30, phase="importing", stop_requested=True, release_completes_after=2.0)  # the phase stays `importing`
    breach(w, "host_mem_available_below_floor")
    rc, rec = w.go(max_s=120.0)
    esc = assert_release_then_cleanup(w, rec)
    assert rc == 5 and esc["phase"] == "importing" and esc["gpu_released_at"] is not None
    assert w.importer.poll() is None and w.agent.poll() is None  # still running after the deadline response


# ---- supervisor or sampler failure


def test_sampler_failure_before_the_guard(make):
    w = make(age=1000)
    w.silent.add("storage_observations.jsonl")  # the storage sampler stops producing samples
    rc, rec = w.go(max_s=90.0)
    assert rc == 5 and [(c["cause"], c["source"]) for c in rec["causes"]] == [("monitor_lost", "storage sampler")] and rec["monitors_effective"] is False and 60 < rec["causes"][0]["age_s"] <= 62
    assert_frozen_cleanup_only(w, rec)


def test_sampler_failure_together_with_the_guard(make):
    w = make(age=GUARD - 60.5, phase="report_written", stop_requested=True, released=True)
    w.silent.add("mem_observations.jsonl")
    rc, rec = w.go(max_s=90.0)
    assert rc == 5 and sorted(c["cause"] for c in rec["causes"]) == ["monitor_lost", "reservation_guard"] and len(rec["response"]["steps"]) == 1
    assert_frozen_cleanup_only(w, rec)


def test_supervisor_failure_leaves_an_incomplete_record_and_ends_nothing(make):
    w = make(age=GUARD - 5, phase="report_written", stop_requested=True, released=True)

    def crash():
        raise RuntimeError("supervisor stopped")

    w.at(GUARD - 3, crash)
    with pytest.raises(RuntimeError):
        w.go(max_s=30.0)
    rec = json.loads(w.out.read_text())
    assert rec["complete"] is False and rec["response"] is None and rec["last_check_at"] - w.created < GUARD  # gate G11: INCONCLUSIVE; monitor: ABORT-CONDITION after 30 s without a check
    assert_nothing_ended(w)  # nothing automatic ends the run any more: the operator's monitor line is the remaining control
    # ... and the closeout can no longer call such a run CLEAN (tests/test_e1_closeout_verdict.py): the supervision statement is INVALID
    spec = importlib.util.spec_from_file_location("e1_closeout_verdict_scenarios", CHECKS / "e1_closeout_verdict.py")
    verdict = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verdict)
    (w.run / "run.log").write_text("[e1_v4_scale] start 2026-11-19T00:00:00Z on host\n")
    state, why = verdict.supervision(w.run)
    assert state == "INVALID" and "supervisor lost" in why[0]
    assert verdict.evaluate(w.run, 0, None, 5400.0, w.t)["result"] == "NOT CLEAN"
    monitor = (E1 / "monitor_e1_v4scale.sh").read_text()
    assert "ABORT-CONDITION the supervisor has not checked for" in monitor and "ABORT-CONDITION the reservation is older than 78 min and the run has not ended" in monitor


# ---- simultaneous causes


def test_guard_and_storage_breach_in_the_same_sample(make):
    w = make(age=GUARD - 1.5, phase="report_written", stop_requested=True, released=True)  # the export fills the run root as the guard passes

    def breach():
        (w.run / "STORAGE-BREACH").write_text(json.dumps({"kind": "persistent_storage_above_ceiling", "value_gb": 18.2, "limit_gb": 18, "at": w.t}) + "\n")

    w.at(GUARD + 0.4, breach)
    rc, rec = w.go()
    assert rc == 5 and sorted((c["cause"], c.get("kind")) for c in rec["causes"]) == [("breach", "persistent_storage_above_ceiling"), ("reservation_guard", "run not ended when the reservation guard passed")]
    assert len({c["seen_at"] for c in rec["causes"]}) == 1 and len(rec["response"]["steps"]) == 1 and "later_causes" not in rec
    assert_frozen_cleanup_only(w, rec)


def test_guard_and_memory_breach_in_the_cpu_only_phase(make):
    w = make(age=GUARD - 1.5, phase="report_written", stop_requested=True, released=True)
    (w.run / "mem_watch.json").write_text(json.dumps({"breaches": []}))

    def breach():
        (w.run / "MEMORY-BREACH").write_text(json.dumps({"kind": "host_mem_available_below_floor", "value_gb": 63.2, "limit_gb": 64, "at": w.t}) + "\n")

    w.at(GUARD + 0.4, breach)
    rc, rec = w.go()
    assert rc == 5 and sorted(c["cause"] for c in rec["causes"]) == ["breach", "reservation_guard"] and len(rec["response"]["steps"]) == 1
    assert_frozen_cleanup_only(w, rec)


def test_guard_reached_after_an_earlier_breach_response(make):
    """The case the first latch left open (review of 76d95289): a breach during finalisation is answered through the controller; the run
    is then still going at the guard. The guard is now ENFORCED: recorded as a later cause AND answered by the one registry cleanup."""
    w = make(age=GUARD - 30, phase="importing", stop_requested=True)
    breach(w)
    rc, rec = w.go(max_s=120.0)
    assert rc == 5 and [c["cause"] for c in rec["causes"]] == ["breach"] and "reservation_guard" in [c["cause"] for c in rec["later_causes"]]
    assert_release_then_cleanup(w, rec)


# ---- delayed closeout or deletion


def test_delayed_deletion_is_a_gate_failure_and_the_guard_leaves_the_registered_budget():
    r = D["reservation"]
    parts = r["closeout_and_deletion_parts_s"]
    assert r["hard_ceiling_s"] - r["supervisor_guard_s"] == r["closeout_and_deletion_budget_s"] == 720 == sum(parts.values())
    assert parts["supervisor_cleanup_response"] + parts["observers_drain_and_driver_end"] + parts["watcher_evidence_and_permit"] + parts["deleter_permit_to_request"] + parts["deletion_request_to_observed"] <= 300  # the mechanical path; the rest is slack
    rehearsal = (E1 / "e1_v4scale_rehearsal.py").read_text()
    for name, verdict in (("reservation_exactly_90_min", '"PASS", {}'), ("reservation_deleted_1s_late", "F, np(G10=F, G16=F)"), ("reservation_over_90_min", "F, np(G10=F, G16=F)"), ("reservation_deleted_record_missing", "F, np(G10=INC, G12=INC, G16=F)"),
                          ("deletion_not_deleted", "F, np(G10=INC, G12=INC, G16=F)"), ("deletion_unverified", "F, np(G16=F)"), ("deletion_watcher_record_missing", "F, np(G16=F)")):
        line = next(ln for ln in rehearsal.splitlines() if f'Fixture("{name}"' in ln)
        assert verdict in line, name  # late or unverifiable is a FAIL, never INCONCLUSIVE
    assert "e1_reservation_guard.py deadline --reservation $RUN/$PASS/reservation.json --ceiling-s 5400" in (E1 / "closeout_e1_v4scale.sh").read_text()
    assert "ABORT-CONDITION the reservation is older than 90 min and no deletion record exists" in (E1 / "monitor_e1_v4scale.sh").read_text()


def test_the_supervisor_and_its_wrapper_are_the_pinned_files():
    scripts = (E1 / "SCRIPTS_v3scale.sha256").read_text()
    for name in ("e1_supervisor.py", "e1_pids.py"):
        assert f"{hashlib.sha256((CHECKS / name).read_bytes()).hexdigest()}  checks/{name}" in scripts, name  # the frozen V3 files, unchanged
    code = (CHECKS / "e1_supervisor_v4.py").read_text().split('"""', 2)[2]
    assert not any(x in code for x in ("os.kill", "import signal", "killpg", "send_signal", ".terminate(", ".kill(", "subprocess"))
