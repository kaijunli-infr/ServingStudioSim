"""Early reservation release (design only, not enabled for V4): the decision check allows a deletion only on independently
verified GPU release, with every run process outside every pod cgroup, an isolated placeholder pod and the preserved stream
in place; everything unknown refuses; with the registered inputs the command line can never return "allowed"."""

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
E1 = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1"
SCRIPT = E1 / "checks" / "e1_reservation_release.py"
POD = json.loads((REPO / "tests/fixtures/e1_v4_scale/reserve_pod.json").read_text())
spec = importlib.util.spec_from_file_location("e1_reservation_release", SCRIPT)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
HOST, NOW = "pod4-gb300-10-tray16-f3", 1_800_003_000.0
REL = NOW - 40.0
USER_SLICE = "0::/user.slice/user-1000.slice/session-9831.scope\n"
POD_CGROUP = "0::/kubepods.slice/kubepods-besteffort.slice/kubepods-besteffort-pod1234.slice/cri-containerd-abcd.scope\n"


class World:
    def __init__(self, tmp: Path):
        self.root, self.run, self.proc = tmp / "root", tmp / "root" / "e1_v4_scale", tmp / "proc"
        (self.run / "profile" / "nsys").mkdir(parents=True)
        (self.root / "pids").mkdir()
        self.stream = tmp / "x.raw-preserve" / "x.qdstrm"
        self.stream.parent.mkdir()
        self.stream.write_bytes(b"q" * 4096)
        self.lc = {"gpu": {"released_at": REL, "release_requested_at": None}, "server": {"alive": False}, "nsys": {"pid": 4000},
                   "stream": {"preserved_path": str(self.stream), "preserved_size": 4096, "final": {"size": 4096, "sha256": "x"}}}
        self.obs = [{"at": REL - 6 + 2.2 * i, "host": HOST, "available": True, "memory_used_mib": [260_000] * 4 if i < 3 else [0] * 4, "compute_pids": [1] if i < 3 else []} for i in range(20)]
        self.census = {"host": HOST, "at": NOW - 5, "compute_apps": [], "gpus": [{"index": i, "memory_used_mib": 0} for i in range(4)]}
        self.pod = json.loads(json.dumps(POD))
        self.reservation = {"pod": "dsv41-gb300-reserve-tray16", "node": HOST, "phase": "Running", "creation_timestamp": "2026-11-19T00:00:00Z"}
        self.probe = {"importer": {"pid": 5000, "agent": {"pid": 4990}}}
        self.procs = {3000: USER_SLICE, 3001: USER_SLICE, 3900: USER_SLICE, 5000: USER_SLICE, 4990: USER_SLICE}  # driver, sampler, supervisor, importer, agent (the Nsight CLI 4000 has exited)
        self.registries = {"owned.tsv": [3000, 3001], "supervisor.tsv": [3900]}

    def write(self):
        (self.run / "profile" / "nsys" / "lifecycle.json").write_text(json.dumps(self.lc))
        (self.run / "gpu_observations.jsonl").write_text("".join(json.dumps(o) + "\n" for o in self.obs))
        (self.run / "reservation.json").write_text(json.dumps(self.reservation))
        (self.run / "import_probe.json").write_text(json.dumps(self.probe))
        for name, pids in self.registries.items():
            (self.root / "pids" / name).write_text("# pid\tstarttime\n" + "".join(f"{p}\t1\tboot\t1000\tlabel\n" for p in pids))
        for pid, cg in self.procs.items():
            d = self.proc / str(pid)
            d.mkdir(parents=True, exist_ok=True)
            (d / "stat").write_text(f"{pid} (python3) S 1 1\n")
            if cg is not None:
                (d / "cgroup").write_text(cg)

    def decide(self, **kw):
        self.write()
        return R.decide(self.run, self.pod, self.census, now=NOW, host=HOST, proc=self.proc, **kw)


def test_verified_release_with_isolated_processes_would_allow(tmp_path):
    rep = World(tmp_path).decide()
    assert rep["decision"] == "WOULD_ALLOW" and rep["refused_by"] == [] and len(rep["checks"]) == 6
    assert rep["evidence"]["gpu_released_at"] == REL and len(rep["evidence"]["sampler_tail"]) == 5 and len(rep["evidence"]["processes_alive"]) == 5


def _refused(tmp_path, mutate, **kw):
    w = World(tmp_path)
    mutate(w)
    rep = w.decide(**kw)
    assert rep["decision"] == "REFUSE"
    return rep["refused_by"]


CASES = {
    "controller has not recorded the release": (lambda w: w.lc["gpu"].update(released_at=None), ["C1.controller_release_recorded", "C2.independent_sampler_released", "C3.census_now"]),
    "server still alive in the controller record": (lambda w: w.lc["server"].update(alive=True), ["C1.controller_release_recorded"]),
    "sampler still sees memory held": (lambda w: [o.update(memory_used_mib=[260_000] * 4) for o in w.obs[-2:]], ["C2.independent_sampler_released"]),
    "sampler sees a compute process after the release": (lambda w: w.obs[10].update(compute_pids=[77]), ["C2.independent_sampler_released"]),
    "only four released observations": (lambda w: setattr(w, "obs", w.obs[:7]), ["C2.independent_sampler_released"]),
    "sampler unavailable in the newest observation": (lambda w: w.obs[-1].update(available=False), ["C2.independent_sampler_released"]),
    "sampler observations are old": (lambda w: ([o.update(at=o["at"] - 200) for o in w.obs], w.lc["gpu"].update(released_at=REL - 200)), ["C2.independent_sampler_released"]),
    "sampler observations from another host": (lambda w: [o.update(host="pod4-gb300-10-tray14-f3") for o in w.obs], ["C2.independent_sampler_released"]),
    "no sampler file content": (lambda w: setattr(w, "obs", []), ["C2.independent_sampler_released"]),
    "census shows a compute process": (lambda w: w.census.update(compute_apps=[{"pid": 9}]), ["C3.census_now"]),
    "census shows memory held": (lambda w: w.census["gpus"][2].update(memory_used_mib=5000), ["C3.census_now"]),
    "census is older than a minute": (lambda w: w.census.update(at=NOW - 61), ["C3.census_now"]),
    "census taken before the release": (lambda w: w.census.update(at=REL - 1), ["C3.census_now"]),
    "census from another host": (lambda w: w.census.update(host="other"), ["C3.census_now"]),
    "no census": (lambda w: setattr(w, "census", None), ["C3.census_now"]),
    "the importer runs inside a pod cgroup": (lambda w: w.procs.update({5000: POD_CGROUP}), ["C4.run_processes_outside_every_pod"]),
    "a registered process runs inside a pod cgroup": (lambda w: w.procs.update({3001: POD_CGROUP}), ["C4.run_processes_outside_every_pod"]),
    "cgroup of a live process unreadable": (lambda w: w.procs.update({3900: None}), ["C4.run_processes_outside_every_pod"]),
    "no process recorded at all": (lambda w: (setattr(w, "registries", {}), setattr(w, "probe", {}), w.lc.update(nsys={})), ["C4.run_processes_outside_every_pod"]),
    "pod specification missing": (lambda w: setattr(w, "pod", None), ["C5.pod_is_an_isolated_placeholder"]),
    "another pod": (lambda w: w.pod["metadata"].update(name="other-pod"), ["C5.pod_is_an_isolated_placeholder"]),
    "pod on another node": (lambda w: w.pod["spec"].update(nodeName="pod4-gb300-10-tray14-f3"), ["C5.pod_is_an_isolated_placeholder"]),
    "pod shares the host pid namespace": (lambda w: w.pod["spec"].update(hostPID=True), ["C5.pod_is_an_isolated_placeholder"]),
    "pod mounts a host path": (lambda w: w.pod["spec"]["volumes"].append({"name": "tmp", "hostPath": {"path": "/tmp"}}), ["C5.pod_is_an_isolated_placeholder"]),
    "pod runs something other than pause": (lambda w: w.pod["spec"]["containers"].append({"name": "work", "image": "ubuntu:24.04"}), ["C5.pod_is_an_isolated_placeholder"]),
    "preserved stream missing": (lambda w: w.stream.unlink(), ["C6.preserved_stream_present"]),
    "preserved stream truncated": (lambda w: w.stream.write_bytes(b"q" * 100), ["C6.preserved_stream_present"]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_refusals(tmp_path, name):
    mutate, expected = CASES[name]
    assert _refused(tmp_path, mutate) == expected, name


def test_dead_processes_are_not_required_and_boundaries(tmp_path):
    w = World(tmp_path / "a")
    w.procs.pop(3001)  # an observer that has exited
    assert w.decide()["decision"] == "WOULD_ALLOW"
    w = World(tmp_path / "b")
    w.census["at"], w.lc["gpu"]["released_at"] = NOW - 60.0, NOW - 100.0  # the census exactly at its age limit, after an earlier release
    assert w.decide()["decision"] == "WOULD_ALLOW"
    w = World(tmp_path / "c")
    w.obs = w.obs[:8]  # five released observations spanning 8.8 s: below the 10 s span
    assert w.decide()["refused_by"] == ["C2.independent_sampler_released"]


def test_command_line_is_never_allowed_with_the_registered_inputs(tmp_path):
    w = World(tmp_path)
    w.obs = [{**o, "host": os.uname().nodename, "at": time.time() - 45 + 2.2 * i} for i, o in enumerate(w.obs)]
    w.lc["gpu"]["released_at"] = time.time() - 39
    w.census.update(host=os.uname().nodename, at=time.time() - 2)
    w.pod["spec"]["nodeName"] = os.uname().nodename
    w.procs, w.registries, w.probe, w.lc["nsys"] = {}, {"owned.tsv": [os.getpid()]}, {}, {}
    w.write()
    pod, census, out = tmp_path / "pod.json", tmp_path / "census.json", tmp_path / "evidence.json"
    pod.write_text(json.dumps(w.pod))
    census.write_text(json.dumps(w.census))
    cmd = [sys.executable, "-I", str(SCRIPT), "--run", str(w.run), "--pod-spec", str(pod), "--census", str(census), "--out", str(out)]
    p = subprocess.run([*cmd, "--inputs", str(E1 / "E1-V4-SCALE-INPUTS.json")], capture_output=True, text=True)
    rep = json.loads(out.read_text())
    assert p.returncode == 4 and rep["decision"] == "WOULD_ALLOW" and rep["enabled"] is False and "NOT ENABLED" in p.stdout  # this very process: alive, in a session scope, outside every pod
    enabled = tmp_path / "inputs_enabled.json"
    enabled.write_text(json.dumps({"reservation": {"early_release_after_verified_gpu_release": {"enabled": True}}}))
    assert subprocess.run([*cmd, "--inputs", str(enabled)], capture_output=True).returncode == 0  # the flag is the only switch, and the registered inputs do not set it
    census.write_text(json.dumps({**w.census, "compute_apps": [{"pid": 1}]}))
    assert subprocess.run([*cmd, "--inputs", str(enabled)], capture_output=True).returncode == 1


def test_source_never_touches_kubernetes_or_signals():
    code = SCRIPT.read_text().split('"""', 2)[2]
    assert not any(x in code for x in ("kubectl", "subprocess", "os.kill", "import signal", "requests", "urllib", "socket.create"))
    assert json.loads((E1 / "E1-V4-SCALE-INPUTS.json").read_text())["reservation"]["early_release_after_verified_gpu_release"]["enabled"] is False
