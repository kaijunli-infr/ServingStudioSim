"""The V2 trigger probe: importer identity from /proc, import confirmed in progress, no trigger when the import is over."""

from __future__ import annotations

# ruff: noqa: E501

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/checks/e1_release_probe.py"
LINE = "release-gpus V2 forced release during import"
NSYS, CHILD, START = 4000, 4001, 777


def _load():
    spec = importlib.util.spec_from_file_location("e1_release_probe", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class World:
    """A fake /proc and lifecycle.json that advance each time the probe sleeps."""

    def __init__(self, tmp: Path, *, cpu_step: int = 5, starttime: int = START, script=None):
        self.proc, self.nsys_dir = tmp / "proc", tmp / "nsys"
        self.proc.mkdir()
        self.nsys_dir.mkdir()
        self.ticks, self.cpu_step, self.starttime, self.t = 100, cpu_step, starttime, 0
        self.status = {
            "phase": "importing",
            "import_progress_pct": 10,
            "nsys": {"pid": NSYS, "identity": {"pid": NSYS, "starttime": START}},
            "gpu": {},
        }
        self.script = script or {}
        self.alive = True
        self.write()

    def stat(self, pid, comm, ppid, ticks):
        fields = (
            ["S", str(ppid)]
            + ["0"] * 9
            + [str(ticks), "0"]
            + ["0"] * 6
            + [str(self.starttime)]
            + ["0"] * 5
        )
        d = self.proc / str(pid)
        d.mkdir(exist_ok=True)
        (d / "stat").write_text(f"{pid} ({comm}) " + " ".join(fields) + "\n")

    def write(self):
        if self.alive:
            self.stat(NSYS, "nsys", 1, self.ticks)
            self.stat(CHILD, "QdstrmImporter", NSYS, self.ticks * 3)
        (self.nsys_dir / "lifecycle.json").write_text(json.dumps(self.status))

    def sleep(self, s):
        self.t += 1
        self.ticks += self.cpu_step
        if self.t in self.script:
            self.script[self.t](self)
        control = self.nsys_dir / "lifecycle.control"
        if control.exists() and "api_server_sigterm" not in self.status["gpu"]:
            self.status["gpu"]["api_server_sigterm"] = {
                "at": __import__("time").time(),
                "pid": 4100,
                "outcome": "signalled",
            }
        self.write()


def _run(mod, w: World, tmp: Path):
    logs = []
    rc = mod.run(
        w.nsys_dir,
        LINE,
        tmp / "probe.json",
        logs.append,
        poll_s=0,
        confirm_s=2.0,
        gap_s=0,
        after_s=(0, 0, 0),
        proc=w.proc,
        sleep=w.sleep,
    )
    return rc, json.loads((tmp / "probe.json").read_text()), logs


def test_probe_verifies_the_importer_confirms_progress_writes_once_and_checks_after_the_signal(
    tmp_path,
):
    mod = _load()
    w = World(tmp_path)
    rc, rec, _ = _run(mod, w, tmp_path)
    assert rc == 0 and (w.nsys_dir / "lifecycle.control").read_text() == LINE + "\n"
    assert (
        rec["saw_phase"] == "importing"
        and rec["identity_verified"]
        and rec["import_in_progress_confirmed"] is True
    )
    assert (
        len(rec["samples"]) >= 2
        and rec["samples"][-1]["tree_cpu_ticks"] > rec["samples"][0]["tree_cpu_ticks"]
    )
    assert {p["comm"] for p in rec["samples"][0]["tree"]} == {"nsys", "QdstrmImporter"}
    assert (
        rec["samples"][0]["nsys_starttime_proc"]
        == rec["samples"][0]["nsys_starttime_recorded"]
        == START
    )
    assert (
        rec["api_server_sigterm_at_from_controller"] is not None and len(rec["after_signal"]) == 3
    )
    assert all(
        s["at"] >= rec["api_server_sigterm_at_from_controller"] and s["identity_verified"]
        for s in rec["after_signal"]
    )


def test_no_trigger_when_the_stop_is_already_terminal(tmp_path):
    mod = _load()
    w = World(tmp_path)
    w.status["phase"] = "report_written"
    w.write()
    rc, rec, _ = _run(mod, w, tmp_path)
    assert (
        rc == 1
        and not (w.nsys_dir / "lifecycle.control").exists()
        and rec["result"].startswith("NO TRIGGER")
        and "INCONCLUSIVE" in rec["result"]
    )


def test_no_trigger_when_the_import_ends_while_it_is_being_confirmed(tmp_path):
    mod = _load()
    w = World(
        tmp_path, cpu_step=0, script={1: lambda w: w.status.__setitem__("phase", "report_written")}
    )
    rc, rec, _ = _run(mod, w, tmp_path)
    assert (
        rc == 1
        and not (w.nsys_dir / "lifecycle.control").exists()
        and rec["import_in_progress_confirmed"] is False
    )


def test_no_trigger_when_the_pid_is_not_the_recorded_importer(tmp_path):
    mod = _load()
    w = World(tmp_path, starttime=START + 1)  # pid reused by another process
    rc, rec, _ = _run(mod, w, tmp_path)
    assert (
        rc == 1
        and not (w.nsys_dir / "lifecycle.control").exists()
        and rec["samples"][0]["identity_verified"] is False
    )


def test_an_idle_importer_is_released_but_marked_unconfirmed(tmp_path):
    mod = _load()
    w = World(
        tmp_path, cpu_step=0
    )  # alive, phase importing, but no CPU time and no progress between samples
    rc, rec, _ = _run(mod, w, tmp_path)
    assert (
        rc == 0
        and (w.nsys_dir / "lifecycle.control").exists()
        and rec["import_in_progress_confirmed"] is False
    )


def test_importer_gone_after_the_signal_is_recorded_as_such(tmp_path):
    mod = _load()

    def die(w):
        w.alive = False
        for pid in (NSYS, CHILD):
            (w.proc / str(pid) / "stat").unlink()

    w = World(tmp_path, script={3: die})
    rc, rec, _ = _run(mod, w, tmp_path)
    assert (
        rc == 0
        and rec["after_signal"]
        and rec["after_signal"][-1]["identity_verified"] is False
        and rec["after_signal"][-1]["nsys_state"] is None
    )
