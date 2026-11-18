"""Run supervisor: breach markers and sampler heartbeats lead to exactly one bounded response, through the controller's
control file during finalisation and through the pinned registry cleanup otherwise; the supervisor itself signals nothing."""

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
SCRIPT = CHECKS / "e1_supervisor.py"
LINE = "release-gpus V3 resource-limit abort (registered supervisor response)"


def _load():
    spec = importlib.util.spec_from_file_location("e1_supervisor", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class World:
    """A run directory and a fake clock; `steps` run one per supervisor sleep. `controller` plays the controller's reading of the control file."""

    def __init__(self, tmp: Path, steps=(), *, phase="importing", stop_requested=True, ack_after: float | None = 0.4):
        self.run = tmp / "root" / "e1_v3_scale"
        self.nsys = self.run / "profile" / "nsys"
        self.nsys.mkdir(parents=True)
        (tmp / "root" / "pids").mkdir()
        self.registry, self.out = tmp / "root" / "pids" / "owned.tsv", self.run / "supervisor.json"
        self.t, self.steps, self.ack_after, self.cleanups, self.record_after = 1000.0, list(steps), ack_after, [], 61.0
        self.lc = {"phase": phase, "stop_requested_at": 990.0 if stop_requested else None, "gpu": {}, "control_commands": [], "updated_at": self.t}
        self.beat()
        self.write_lc()

    def write_lc(self):
        (self.nsys / "lifecycle.json").write_text(json.dumps(self.lc))

    def beat(self, names=("mem_observations.jsonl", "storage_observations.jsonl", "gpu_observations.jsonl"), age=0.0):
        for n in names:
            p = self.run / n
            p.write_text("{}\n")
            os.utime(p, (self.t - age, self.t - age))

    def sleep(self, s):
        self.t += s
        self.lc["updated_at"] = self.t if self.lc.get("controller_alive", True) else self.lc["updated_at"]
        control = self.nsys / "lifecycle.control"
        if self.ack_after is not None and control.exists() and not self.lc["control_commands"]:
            written = getattr(self, "_seen_line_at", None)
            if written is None:
                self._seen_line_at = self.t
            elif self.t - self._seen_line_at >= self.ack_after:
                # like the controller: the release request (with the line as its reason) is written at once, the command record only
                # after the release action has finished, `record_after` seconds later
                line = control.read_text().splitlines()[0]
                if self.lc["gpu"].get("release_requested_at") is None:
                    self.lc["gpu"].update({"release_requested_at": self.t, "release_reason": f"operator: {line}", "release_context": {"phase": self.lc["phase"]}, "api_server_sigterm": {"at": self.t, "pid": 4100, "outcome": "signalled"}})
                if self.t - self.lc["gpu"]["release_requested_at"] >= self.record_after:
                    self.lc["control_commands"].append({"at": self.t, "command": line, "phase": self.lc["phase"], "result": "release requested"})
        self.write_lc()
        if self.steps:
            self.steps.pop(0)(self)
        if not getattr(self, "stale", False):
            self.beat()

    def cleanup(self, registry, grace, log):
        self.cleanups.append((str(registry), grace))
        self.t += 2.0
        return {"command": ["e1_pids.py", "cleanup"], "rc": 0, "result_line": "RESULT CLEAN: signalled 5", "log": str(log), "started_at": self.t - 2.0, "ended_at": self.t}

    def go(self, mod, **kw):
        kw.setdefault("max_s", 30.0)
        return mod.supervise(self.run, self.registry, self.out, LINE, now=lambda: self.t, sleep=self.sleep, cleanup=self.cleanup, **kw)

    def record(self):
        return json.loads(self.out.read_text())


def marker(name="MEMORY-BREACH", kind="importer_rss_above_ceiling", value=130.0, limit=128.0):
    def step(w):
        with (w.run / name).open("a") as fh:
            fh.write(json.dumps({"at": w.t - 0.5, "kind": kind, "value_gb": value, "limit_gb": limit}) + "\n")

    return step


def test_quiet_run_ends_without_a_response(tmp_path):
    mod, w = _load(), World(tmp_path)
    assert w.go(mod, max_s=5.0) == 0
    rec = w.record()
    assert rec["response"] is None and rec["causes"] == [] and rec["monitors_effective"] is True and rec["complete"] is True and rec["samples"] >= 5
    assert not (w.nsys / "lifecycle.control").exists() and w.cleanups == [] and rec["signals_sent_by_supervisor"] == 0
    assert all(h["age_s"] is not None and h["age_s"] <= 1.0 for h in rec["heartbeats"].values())


def test_marker_during_finalisation_is_delivered_through_the_controller(tmp_path):
    mod, w = _load(), World(tmp_path, steps=[lambda w: None, marker()])
    assert w.go(mod) == 5
    rec = w.record()
    step = rec["response"]["steps"]
    assert (w.nsys / "lifecycle.control").read_text() == LINE + "\n" and w.cleanups == []
    assert [s["kind"] for s in step] == ["R1 control line"] and step[0]["acknowledged"] is True and 0 < step[0]["latency_s"] <= 10
    assert rec["causes"][0]["source"] == "MEMORY-BREACH" and rec["causes"][0]["kind"] == "importer_rss_above_ceiling" and rec["causes"][0]["breach_at"] < rec["causes"][0]["seen_at"]
    assert rec["response"]["outcome"] == "release requested through the controller" and rec["response"]["phase"] == "importing" and rec["monitors_effective"] is True
    assert rec["response"]["decided_at"] - rec["causes"][0]["breach_at"] <= 2.0  # marker written, seen and acted on within two polls


def test_delayed_acknowledgement_inside_the_bound_is_still_r1(tmp_path):
    mod, w = _load(), World(tmp_path, steps=[marker()], ack_after=8.0)
    assert w.go(mod) == 5
    step = w.record()["response"]["steps"]
    assert [s["kind"] for s in step] == ["R1 control line"] and 8.0 <= step[0]["latency_s"] <= 10.0 and w.cleanups == []


@pytest.mark.parametrize("ack_after", [12.0, None])
def test_missing_or_late_acknowledgement_falls_back_to_the_registry_cleanup(tmp_path, ack_after):
    mod, w = _load(), World(tmp_path, steps=[marker()], ack_after=ack_after)
    assert w.go(mod) == 5
    rec = w.record()
    steps = rec["response"]["steps"]
    assert [s["kind"] for s in steps] == ["R1 control line", "R2 registry cleanup"] and steps[0]["acknowledged"] is False and "not acknowledged within 10.0 s" in steps[1]["why"]
    assert w.cleanups == [(str(w.registry), 30.0)] and (w.nsys / "lifecycle.control").read_text().count(LINE) == 1
    assert 10.0 <= rec["response"]["seconds_from_first_cause_seen"] <= 14.0 and rec["complete"] is True


@pytest.mark.parametrize("phase, stop_requested", [("capturing", False), ("report_written", True), ("stop_failed", True)])
def test_outside_finalisation_the_response_is_the_registry_cleanup(tmp_path, phase, stop_requested):
    mod, w = _load(), World(tmp_path, steps=[marker("STORAGE-BREACH", "persistent_storage_above_ceiling", 8.1, 8.0)], phase=phase, stop_requested=stop_requested)
    assert w.go(mod) == 5
    rec = w.record()
    assert [s["kind"] for s in rec["response"]["steps"]] == ["R2 registry cleanup"] and "not in its finalisation loop" in rec["response"]["steps"][0]["why"]
    assert not (w.nsys / "lifecycle.control").exists() and w.cleanups == [(str(w.registry), 30.0)] and rec["response"]["outcome"] == "registry cleanup rc 0"


def test_no_lifecycle_record_yet_or_a_dead_controller_means_cleanup(tmp_path):
    mod, w = _load(), World(tmp_path, steps=[marker()])
    (w.nsys / "lifecycle.json").unlink()
    w.write_lc = lambda: None
    assert w.go(mod) == 5 and [s["kind"] for s in w.record()["response"]["steps"]] == ["R2 registry cleanup"]
    w2 = World(tmp_path / "b", steps=[lambda w: None] * 8 + [marker()])
    (tmp_path / "b").mkdir(exist_ok=True)
    w2.lc["controller_alive"] = False  # the record stops being refreshed: the controller is not reading the control file
    assert w2.go(mod) == 5 and [s["kind"] for s in w2.record()["response"]["steps"]] == ["R2 registry cleanup"]


def test_simultaneous_breaches_give_one_response_with_both_causes(tmp_path):
    both = lambda w: (marker()(w), marker("STORAGE-BREACH", "temporary_storage_above_ceiling", 10.2, 10.0)(w))  # noqa: E731
    mod, w = _load(), World(tmp_path, steps=[both, marker("MEMORY-BREACH", "host_mem_available_below_floor", 60.0, 64.0)])
    assert w.go(mod) == 5
    rec = w.record()
    assert sorted(c["source"] for c in rec["causes"]) == ["MEMORY-BREACH", "STORAGE-BREACH"] and len(rec["response"]["steps"]) == 1
    assert (w.nsys / "lifecycle.control").read_text() == LINE + "\n" and w.cleanups == []
    assert [c["kind"] for c in rec.get("later_causes", [])] == ["host_mem_available_below_floor"]  # recorded, no second response


def test_breach_in_the_samplers_record_without_a_marker_file_is_still_acted_on(tmp_path):
    def summary_only(w):
        (w.run / "mem_watch.json").write_text(json.dumps({"breaches": [{"at": w.t, "kind": "host_mem_available_below_floor", "value_gb": 50.0, "limit_gb": 64.0}]}))

    mod, w = _load(), World(tmp_path, steps=[summary_only])
    assert w.go(mod) == 5
    c = w.record()["causes"][0]
    assert c["source"] == "mem_watch.json (marker file missing)" and c["marker_mtime"] is None and c["kind"] == "host_mem_available_below_floor"


@pytest.mark.parametrize("name, fname", [("memory sampler", "mem_observations.jsonl"), ("storage sampler", "storage_observations.jsonl"), ("GPU sampler", "gpu_observations.jsonl")])
def test_a_silent_sampler_is_a_lost_monitor(tmp_path, name, fname):
    def go_silent(w):
        w.stale = True

    def others(w):
        w.beat([f for f in ("mem_observations.jsonl", "storage_observations.jsonl", "gpu_observations.jsonl") if f != fname])

    mod, w = _load(), World(tmp_path, steps=[go_silent] + [others] * 80)
    assert w.go(mod, max_s=90.0) == 5
    rec = w.record()
    assert [(c["cause"], c["source"]) for c in rec["causes"]] == [("monitor_lost", name)] and rec["monitors_effective"] is False
    assert 60.0 < rec["causes"][0]["age_s"] <= 63.0 and rec["response"]["steps"][0]["kind"] == "R1 control line"


def test_gpu_sampler_may_stop_after_the_recorded_release_and_missing_files_get_a_grace(tmp_path):
    def released(w):
        w.lc["gpu"]["released_at"] = w.t
        w.lc["phase"] = "report_written"
        w.stale = True

    def others(w):
        w.beat(["mem_observations.jsonl", "storage_observations.jsonl"])

    mod, w = _load(), World(tmp_path, steps=[released] + [others] * 80)
    assert w.go(mod, max_s=80.0) == 0 and w.record()["heartbeats"]["GPU sampler"]["required"] is False
    w2 = World(tmp_path / "b")
    for f in ("mem_observations.jsonl", "storage_observations.jsonl", "gpu_observations.jsonl"):
        (w2.run / f).unlink()
    w2.stale = True
    assert w2.go(mod, max_s=50.0) == 0  # younger than --stale-s: samplers may still be starting
    assert w2.go(mod, max_s=70.0) == 5 and {c["kind"] for c in w2.record()["causes"]} == {"no sample file"}


def test_stop_file_ends_the_supervision(tmp_path):
    mod, w = _load(), World(tmp_path)
    stop = tmp_path / "stop"
    w.steps = [lambda w: None, lambda w: stop.write_text("")]
    assert w.go(mod, max_s=600.0, stop_file=stop) == 0 and w.record()["samples"] == 3


def test_supervisor_source_cannot_signal():
    src = SCRIPT.read_text()
    code = src.split('"""', 2)[2]
    assert "os.kill" not in code and "import signal" not in code and "killpg" not in code and "send_signal" not in code and ".terminate(" not in code and ".kill(" not in code
    assert code.count("subprocess.run") == 1 and '"cleanup", "--registry"' in code  # its only child is the pinned registry cleanup


def test_control_line_must_be_a_release_line(tmp_path):
    mod = _load()
    with pytest.raises(SystemExit):
        mod.main(["--run", str(tmp_path), "--registry", str(tmp_path / "r"), "--out", str(tmp_path / "o"), "--control-line", "extend 60 because"])


def test_real_registry_cleanup_reaches_the_owned_tree_and_nothing_else(tmp_path):
    """R2 with the real pinned e1_pids.py: a registered tree is terminated; a detached, unregistered 'importer' and the supervisor's
    own process (registered in a separate registry) are not touched; nothing is refused."""
    mod = _load()
    env = {**os.environ, "E1_LOOKALIKE_RE": "no-lookalike-pattern-in-this-test"}
    pids = tmp_path / "pids"
    pids.mkdir()
    owned, sup = pids / "owned.tsv", pids / "supervisor.tsv"
    tree = subprocess.Popen([sys.executable, "-c", "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)']); time.sleep(300)"], start_new_session=True)
    detached = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True)  # stands for the agent's importer
    try:
        time.sleep(0.5)
        subprocess.run([sys.executable, "-I", str(CHECKS / "e1_pids.py"), "register", "--registry", str(owned), "--label", "driver:test", str(tree.pid)], check=True, capture_output=True, env=env)
        subprocess.run([sys.executable, "-I", str(CHECKS / "e1_pids.py"), "register", "--registry", str(sup), "--label", "supervisor:test", str(os.getpid())], check=True, capture_output=True, env=env)
        res = mod.run_cleanup(owned, 5.0, tmp_path / "cleanup.jsonl", runner=lambda cmd, **kw: subprocess.run(cmd, env=env, **kw))
        tree.wait(timeout=10)
        log = json.loads((tmp_path / "cleanup.jsonl").read_text().splitlines()[-1])
        assert res["rc"] == 0 and "RESULT CLEAN" in res["result_line"] and not log["plan"]["refused"]
        assert sorted(a["action"] for a in log["actions"]) == ["SIGTERM", "SIGTERM"] and tree.pid in {a["pid"] for a in log["actions"]}
        assert detached.poll() is None and detached.pid not in {a["pid"] for a in log["actions"]}
    finally:
        for p in (tree, detached):
            if p.poll() is None:
                p.kill()
            p.wait()


def test_gpu_held_ceiling_before_any_stop_request_is_a_cause(tmp_path):
    """The controller watches its ceiling only after the stop request; before it, the supervisor does."""
    mod, w = _load(), World(tmp_path, phase="capturing", stop_requested=False)
    w.lc["launch_started_at"] = w.t - 2695.0
    w.write_lc()
    assert w.go(mod, gpu_held_ceiling_s=2700.0, max_s=20.0) == 5
    rec = w.record()
    assert [(c["cause"], c["limit_s"]) for c in rec["causes"]] == [("gpu_held_ceiling_before_stop", 2700.0)] and 2700 < rec["causes"][0]["held_s"] <= 2702
    assert [s["kind"] for s in rec["response"]["steps"]] == ["R2 registry cleanup"] and rec["monitors_effective"] is True
    w2 = World(tmp_path / "b")  # stop already requested: the controller owns the ceiling
    w2.lc["launch_started_at"] = w2.t - 5000.0
    w2.write_lc()
    assert w2.go(mod, gpu_held_ceiling_s=2700.0, max_s=5.0) == 0


def test_acknowledgement_is_the_release_request_not_the_later_command_record(tmp_path):
    """The controller blocks in its release action for up to the 60 s grace and records the command only afterwards. Waiting for
    that record would time out and run the registry cleanup against a controller that is doing exactly what was asked."""
    mod, w = _load(), World(tmp_path, steps=[marker()])
    assert w.record_after > 10.0
    assert w.go(mod) == 5
    step = w.record()["response"]["steps"]
    assert [s["kind"] for s in step] == ["R1 control line"] and step[0]["acknowledged"] is True and step[0]["evidence"] == "gpu.release_requested_at with this line as the reason"
    assert step[0]["api_server_sigterm"]["outcome"] == "signalled" and w.cleanups == [] and w.lc["control_commands"] == []
    lc = {"gpu": {"release_requested_at": 5.0, "release_reason": "hard deadline"}, "control_commands": []}
    assert mod.acknowledged(lc, LINE) is None  # a release for another reason is not an acknowledgement of this line
