"""Importer-specific import-state probe: identifies the importer itself, counts only its own CPU/I-O, fails closed."""

from __future__ import annotations

# ruff: noqa: E501

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/checks/e1_import_probe.py"
EXE = "/opt/nvidia/nsight-systems/2025.6.3/host-linux-armv8/QdstrmImporter"
NSYS_EXE = "/opt/nvidia/nsight-systems/2025.6.3/target-linux-sbsa-armv8/nsys"
STREAM, LINE = "/tmp/nsys-report-a2d6.qdstrm", "release-gpus registered line"
CLI, AGENT, IMP = 1556601, 1556700, 1556900


def _load():
    spec = importlib.util.spec_from_file_location("e1_import_probe", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class World:
    """A fake /proc, report file and lifecycle.json; `steps` run one per probe sleep."""

    def __init__(self, tmp: Path, steps=()):
        self.proc, self.nsys_dir = tmp / "proc", tmp / "nsys"
        self.proc.mkdir()
        self.nsys_dir.mkdir()
        self.report, self.control = tmp / "out.nsys-rep", self.nsys_dir / "lifecycle.control"
        self.out, self.samples, self.lines = tmp / "probe.json", tmp / "probe.jsonl", []
        self.steps, self.t, self.status = list(steps), 1000.0, {"phase": "importing", "import_progress_pct": 8, "stream": {"path": STREAM}}
        self.write_status()

    def write_status(self):
        (self.nsys_dir / "lifecycle.json").write_text(json.dumps(self.status))

    def put(self, pid, comm, exe, argv, *, ppid=1, ticks=0, starttime=500, state="S", io=(0, 0), fds=()):
        d = self.proc / str(pid)
        if d.exists():
            self.remove(pid)
        (d / "fd").mkdir(parents=True)
        u, s = ticks // 2, ticks - ticks // 2
        (d / "stat").write_text(f"{pid} ({comm}) {state} {ppid} {pid} {pid} 0 -1 0 0 0 0 0 {u} {s} 0 0 20 0 1 0 {starttime} 0 0\n")
        (d / "cmdline").write_text("\0".join(argv) + "\0")
        (d / "io").write_text(f"rchar: {io[0]}\nwchar: {io[1]}\nread_bytes: 0\nwrite_bytes: 0\n")
        os.symlink(exe, d / "exe")
        for i, target in enumerate(fds):
            os.symlink(target, d / "fd" / str(i))

    def remove(self, pid):
        d = self.proc / str(pid)
        for f in list((d / "fd").iterdir()) + [d / "exe", d / "stat", d / "cmdline", d / "io"]:
            f.unlink()
        (d / "fd").rmdir()
        d.rmdir()

    def importer(self, *, ticks=10, io=(100, 0), pid=IMP, starttime=900, state="S", report=None, stream=STREAM, ppid=AGENT):
        report = str(self.report) if report is None else report
        self.put(pid, "QdstrmImporter", EXE, [EXE, "--input-file", stream, "--output-file", report, "--force-overwrite"], ppid=ppid, ticks=ticks, starttime=starttime, state=state, io=io, fds=[stream, report])

    def session(self, worker_ticks=135885):
        """The V2 process picture: Nsight CLI with the server and its workers below it, and the detached agent."""
        self.put(CLI, "nsys", NSYS_EXE, [NSYS_EXE, "profile", "-o", "x", "python"], ticks=168, starttime=300975475)
        self.put(1556645, "python", "/venv/bin/python", ["python", "-m", "server"], ppid=CLI, ticks=3030)
        for i in range(4):
            self.put(1557339 + i, "VLLM::Worker_TP", "/venv/bin/python", [f"VLLM::Worker_TP{i}"], ppid=1556645, ticks=worker_ticks + i)
        self.put(AGENT, "nsys", NSYS_EXE, [NSYS_EXE, "--start-agent", "--session-name", f"profile-{CLI}", "--shm-name", "x"], ppid=1, starttime=300975500)

    def sleep(self, _s):
        self.t += 0.5
        if self.steps:
            self.steps.pop(0)(self)

    def run(self, mod, *, trigger=True, **kw):
        kw.setdefault("max_s", 20.0)
        kw.setdefault("confirm_s", 5.0)
        kw.setdefault("nsys_pid", CLI)
        return mod.run(EXE, str(self.report), self.out, self.samples, self.lines.append, nsys_dir=self.nsys_dir, proc=self.proc, uid=os.getuid(), sleep=self.sleep, now=lambda: self.t,
                       control=self.control if trigger else None, trigger_line=LINE if trigger else None, **kw)

    def record(self):
        return json.loads(self.out.read_text())

    def states(self):
        return [t["state"] for t in self.record()["transitions"]]


def advance(dticks=5, dr=1000, dw=0):
    def step(w):
        st = (w.proc / str(IMP) / "stat").read_text().split()
        ticks = int(st[13]) + int(st[14]) + dticks
        io = [int(line.split(":")[1]) for line in (w.proc / str(IMP) / "io").read_text().splitlines()[:2]]
        w.importer(ticks=ticks, io=(io[0] + dr, io[1] + dw))

    return step


def workers_tick(w):
    for i in range(4):
        st = (w.proc / str(1557339 + i) / "stat").read_text().split()
        w.put(1557339 + i, "VLLM::Worker_TP", "/venv/bin/python", [f"VLLM::Worker_TP{i}"], ppid=1556645, ticks=int(st[13]) + int(st[14]) + 40)
    w.status["import_progress_pct"] += 1
    w.write_status()


def finish(size=4096):
    def step(w):
        w.remove(IMP)
        w.report.write_bytes(b"r" * size)

    return step


# ---- identification


def test_importer_is_identified_by_exact_executable_and_session_files(tmp_path):
    mod, w = _load(), World(tmp_path)
    w.session()
    w.importer()
    w.put(7001, "QdstrmImporter", EXE, [EXE, "--version"], ppid=AGENT)  # the agent's version check
    w.put(7002, "QdstrmImporter", EXE, [EXE, "--input-file", "/tmp/other.qdstrm", "--output-file", "/tmp/other.nsys-rep"], ppid=AGENT)  # another session
    w.put(7003, "QdstrmImporter", "/home/x/QdstrmImporter", ["/home/x/QdstrmImporter", "--input-file", STREAM, "--output-file", str(w.report)])  # not the pinned executable
    found = mod.find_importers(EXE, STREAM, str(w.report), proc=w.proc, uid=os.getuid())
    assert [p["pid"] for p in found] == [IMP]
    assert set(found[0]["matched_by"]) == {"--input-file is the session stream", "--output-file is the session report", "holds the session stream open", "holds the session report open"}
    assert mod.find_importers(EXE, STREAM, str(w.report), proc=w.proc, uid=os.getuid() + 1) == []  # another user's process


def test_importer_is_matched_by_open_files_when_arguments_are_unreadable(tmp_path):
    mod, w = _load(), World(tmp_path)
    w.put(IMP, "QdstrmImporter", EXE, [], fds=[STREAM])
    assert mod.find_importers(EXE, STREAM, None, proc=w.proc, uid=os.getuid())[0]["matched_by"] == ["holds the session stream open"]


def test_agent_link_to_the_recorded_cli(tmp_path):
    mod, w = _load(), World(tmp_path)
    w.session()
    w.importer()
    imp = mod.find_importers(EXE, STREAM, str(w.report), proc=w.proc, uid=os.getuid())[0]
    a = mod.agent_of(imp, CLI, proc=w.proc)
    assert a["is_agent"] and a["linked_to_recorded_cli"] is True and a["ppid"] == 1 and a["session_name"] == f"profile-{CLI}"
    assert mod.agent_of(imp, CLI + 1, proc=w.proc)["linked_to_recorded_cli"] is False
    assert mod.agent_of(imp, None, proc=w.proc)["linked_to_recorded_cli"] is None
    w.importer(ppid=4242)  # parent not present
    imp = mod.find_importers(EXE, STREAM, str(w.report), proc=w.proc, uid=os.getuid())[0]
    assert mod.agent_of(imp, CLI, proc=w.proc) == {"pid": 4242, "found": False, "is_agent": False, "linked_to_recorded_cli": False}


# ---- the V2 defect: unrelated CPU time must never count


def test_v2_shaped_tree_without_an_importer_establishes_nothing(tmp_path):
    """The V2 samples: CLI ticks flat at 168, workers ticking, reported progress present, no importer in sight."""
    mod, w = _load(), World(tmp_path, steps=[workers_tick] * 30)
    w.session()
    rc = w.run(mod, max_s=6.0)
    rec = w.record()
    assert rc == 3 and not w.control.exists() and rec["trigger"]["written"] is False
    assert rec["final"]["state"] == mod.NOT_FOUND and "importer" not in rec and w.states() == [mod.NOT_FOUND]


def test_worker_ticks_and_reported_progress_do_not_make_an_idle_importer_active(tmp_path):
    def grow_report(w):
        workers_tick(w)
        w.report.write_bytes(b"x" * (w.report.stat().st_size + 1000 if w.report.exists() else 1000))

    mod, w = _load(), World(tmp_path, steps=[grow_report] * 40)
    w.session()
    w.importer(ticks=10, io=(100, 0))
    rc = w.run(mod, max_s=8.0, confirm_s=3.0)
    rec = w.record()
    assert rc == 3 and not w.control.exists()
    assert mod.ACTIVE not in w.states() and w.states() == [mod.NO_PROGRESS]
    assert "not established within 3.0 s" in rec["trigger"]["withheld"]
    rows = [json.loads(line) for line in w.samples.read_text().splitlines()]
    assert rows[-1]["reported_progress_pct"] > rows[0]["reported_progress_pct"] and rows[-1]["report_size"] > rows[1]["report_size"]  # both moved, neither counted
    assert {r["importer"]["ticks"] for r in rows} == {10}


def test_only_importer_fields_enter_the_progress_decision():
    mod = _load()
    imp = {"pid": 1, "starttime": 2, "state": "R", "ticks": 5, "io": {"rchar": 10, "wchar": 0}}
    a = {"at": 1.0, "importer": imp, "reported_progress_pct": 8, "report_size": 0}
    b = {"at": 1.5, "importer": dict(imp), "reported_progress_pct": 50, "report_size": 10**9}
    assert mod.interval_progress(a, b)["qualifies"] is False
    assert mod.interval_progress(a, {**b, "importer": {**imp, "ticks": 6}})["qualifies"] is True
    assert mod.interval_progress(a, {**b, "importer": {**imp, "io": {"rchar": 11, "wchar": 0}}})["qualifies"] is True
    assert mod.interval_progress(a, {**b, "importer": {**imp, "io": {"rchar": 10, "wchar": 1}}})["qualifies"] is True
    assert mod.interval_progress(a, {**b, "importer": {**imp, "ticks": 9, "starttime": 3}}) == {"from": 1.0, "to": 1.5, "same_importer": False, "ticks_delta": None, "rchar_delta": None, "wchar_delta": None, "qualifies": False}
    assert mod.interval_progress(a, {**b, "importer": {**imp, "ticks": 9, "state": "Z"}})["qualifies"] is False
    assert mod.interval_progress(a, {**b, "importer": None})["qualifies"] is False
    assert mod.interval_progress(a, {**b, "importer": {**imp, "ticks": 9, "io": None}})["qualifies"] is True  # I/O unreadable: CPU alone


# ---- active import


def test_active_needs_two_consecutive_importer_intervals_then_one_trigger(tmp_path):
    mod, w = _load(), World(tmp_path, steps=[advance(), advance(), advance(), advance(), finish()])
    w.session()
    w.importer()
    rc = w.run(mod)
    rec = w.record()
    assert rc == 0 and w.control.read_text() == LINE + "\n"
    assert w.states() == [mod.NO_PROGRESS, mod.ACTIVE, mod.SETTLING, mod.COMPLETED]
    rows = [json.loads(line) for line in w.samples.read_text().splitlines()]
    assert rec["trigger"]["written"] and rec["trigger"]["at"] == rows[2]["at"]  # third sample = second qualifying interval
    assert rec["trigger"]["importer_rechecked"]["pid"] == IMP and rec["importer"]["agent"]["linked_to_recorded_cli"] is True
    assert rec["importer"]["matched_by"] and rec["provenance"]["importer.io"].startswith("/proc/<importer pid>/io")
    assert all("at" in r and "candidates" in r for r in rows) and rows[0]["importer"]["exe"] == EXE


def test_io_alone_establishes_activity(tmp_path):
    mod, w = _load(), World(tmp_path, steps=[advance(0, 5000), advance(0, 5000), advance(0, 0, 700), finish()])
    w.importer()
    assert w.run(mod, trigger=False) == 0 and mod.ACTIVE in w.states()


def test_temporary_lack_of_progress_is_reported_and_recovers(tmp_path):
    idle = advance(0, 0)
    mod, w = _load(), World(tmp_path, steps=[advance(), advance(), idle, idle, advance(), advance(), finish()])
    w.importer()
    assert w.run(mod, trigger=False) == 0
    assert w.states() == [mod.NO_PROGRESS, mod.ACTIVE, mod.NO_PROGRESS, mod.ACTIVE, mod.SETTLING, mod.COMPLETED]


def test_min_progress_is_configurable_and_never_zero(tmp_path):
    mod, w = _load(), World(tmp_path, steps=[advance(), advance(), advance(), finish()])
    w.importer()
    assert w.run(mod, min_progress=3) == 0
    rows = [json.loads(line) for line in w.samples.read_text().splitlines()]
    assert w.record()["trigger"]["at"] == rows[3]["at"]
    with pytest.raises(SystemExit):
        mod.main(["--importer-exe", EXE, "--report", "r", "--stream", "s", "--out", "o", "--samples", "s", "--log", "l", "--min-progress", "0"])


# ---- completed / terminated / not found / ambiguous


def test_import_completed_before_any_trigger_is_completed_not_active(tmp_path):
    mod, w = _load(), World(tmp_path, steps=[advance(), finish()])
    w.importer()
    rc = w.run(mod)
    rec = w.record()
    assert rc == 3 and not w.control.exists() and rec["final"]["state"] == mod.COMPLETED and rec["final"]["importer_seen"] is True
    assert "exit status not observable" in rec["final"]["reason"] and mod.TERMINATED not in w.states()


def test_import_finished_before_the_first_observation(tmp_path):
    mod, w = _load(), World(tmp_path)
    w.report.write_bytes(b"r" * 100)
    rc = w.run(mod)
    rec = w.record()
    assert rc == 3 and rec["final"]["state"] == mod.COMPLETED and rec["final"]["importer_seen"] is False and not w.control.exists()
    assert w.run(mod, trigger=False) == 0


@pytest.mark.parametrize("leftover", [None, b""])
def test_importer_gone_without_a_report_is_terminated(tmp_path, leftover):
    def die(w):
        w.remove(IMP)
        if leftover is not None:
            w.report.write_bytes(leftover)

    mod, w = _load(), World(tmp_path, steps=[advance(), advance(), advance(), die])
    w.importer()
    rc = w.run(mod, trigger=False)
    assert rc == 1 and w.record()["final"]["state"] == mod.TERMINATED and "missing or empty" in w.record()["final"]["reason"]
    assert w.states() == [mod.NO_PROGRESS, mod.ACTIVE, mod.SETTLING, mod.TERMINATED]


def test_pid_reuse_is_not_the_same_importer(tmp_path):
    def replace(w):
        w.importer(ticks=999, io=(10**6, 10**6), starttime=901)  # same pid and files, another process

    mod, w = _load(), World(tmp_path, steps=[advance(), replace, replace, replace, replace])
    w.importer()
    rc = w.run(mod)
    assert rc == 3 and not w.control.exists() and w.record()["final"]["state"] == mod.TERMINATED and mod.ACTIVE not in w.states()


def test_zombie_importer_is_not_alive(tmp_path):
    def zombie(w):
        w.importer(ticks=50, io=(9000, 0), state="Z")

    mod, w = _load(), World(tmp_path, steps=[advance(), zombie, zombie, zombie, zombie])
    w.importer()
    rc = w.run(mod, max_s=3.0)
    assert rc == 3 and not w.control.exists() and mod.ACTIVE not in w.states() and w.record()["final"]["state"] == mod.TERMINATED


def test_two_matching_importers_are_ambiguous(tmp_path):
    def both(w):
        w.importer(ticks=int(w.t * 10), io=(int(w.t * 100), 0))
        w.importer(pid=IMP + 1, ticks=int(w.t * 10), io=(int(w.t * 100), 0))

    mod, w = _load(), World(tmp_path, steps=[both] * 20)
    both(w)
    rc = w.run(mod, max_s=4.0)
    rec = w.record()
    assert rc == 3 and not w.control.exists() and rec["final"]["state"] == mod.AMBIGUOUS and "importer" not in rec


def test_ambiguity_latches_a_survivor_is_never_adopted(tmp_path):
    """Two processes match; one exits; the other keeps working and finishes. The survivor is not singled out, activity is not
    established, no trigger is written, and the session ends AMBIGUOUS."""
    def both(w):
        w.importer(ticks=int(w.t * 10), io=(int(w.t * 100), 0))
        w.importer(pid=IMP + 1, ticks=int(w.t * 10), io=(int(w.t * 100), 0))

    def first_exits(w):
        w.remove(IMP + 1)

    def survivor_works(w):
        w.importer(ticks=int(w.t * 10), io=(int(w.t * 100), 0))

    mod, w = _load(), World(tmp_path, steps=[both, both, first_exits, survivor_works, survivor_works, survivor_works, finish()])
    both(w)
    rc = w.run(mod, max_s=30.0)
    rec = w.record()
    assert rc == 3 and not w.control.exists() and "importer" not in rec and "ambiguity_first_seen_at" in rec
    assert w.states() == [mod.AMBIGUOUS] and rec["final"]["state"] == mod.AMBIGUOUS and rec["trigger"]["withheld"]
    rows = [json.loads(line) for line in w.samples.read_text().splitlines()]
    assert all(r["importer"] is None for r in rows) and [r["candidates"] for r in rows][:5] == [2, 2, 2, 1, 1] and len(rows) < 15  # ended by itself once the report existed
    assert w.run(mod, trigger=False, max_s=5.0) == 0  # a later session on the finished report: nothing ambiguous is running any more


def test_trigger_is_withheld_when_the_importer_changes_before_the_write(tmp_path, monkeypatch):
    mod, w = _load(), World(tmp_path, steps=[advance(), advance()])
    w.importer()
    real = mod.classify

    def classify_then_vanish(samples, **kw):
        c = real(samples, **kw)
        if c["state"] == mod.ACTIVE and (w.proc / str(IMP)).exists():
            w.remove(IMP)
        return c

    monkeypatch.setattr(mod, "classify", classify_then_vanish)
    rc = w.run(mod)
    assert rc == 3 and not w.control.exists() and "final identity check" in w.record()["trigger"]["withheld"]


def test_observer_mode_never_writes_a_control_file(tmp_path):
    mod, w = _load(), World(tmp_path, steps=[advance(), advance(), advance(), finish()])
    w.importer()
    assert w.run(mod, trigger=False) == 0 and not w.control.exists() and w.record()["trigger"] == {"requested": False, "written": False}


def test_generated_marker_is_recorded_not_required(tmp_path):
    mod, w = _load(), World(tmp_path, steps=[advance(), advance(), finish()])
    w.importer()
    log = tmp_path / "server.log"
    log.write_text("Generating '/tmp/nsys-report-a2d6.qdstrm'\n[1/1] [==8%   ] out.nsys-rep\n")
    assert mod.generated_in_log(log, str(w.report)) is False and mod.generated_in_log(None, str(w.report)) is None
    assert w.run(mod, trigger=False, nsys_log=log) == 0
    assert w.record()["final"]["nsight_generated_marker"] is False and "not (yet) logged" in w.record()["final"]["reason"]
    log.write_text(log.read_text() + f"Generated:\n\t{w.report}\n")
    assert mod.generated_in_log(log, str(w.report)) is True


def test_stream_path_comes_from_the_lifecycle_record(tmp_path):
    mod, w = _load(), World(tmp_path, steps=[advance(), advance(), finish()])
    w.importer(report="/elsewhere/other.nsys-rep")  # only the stream identifies it
    w.report.write_bytes(b"")
    w.run(mod, trigger=False)
    assert w.record()["stream"] == STREAM and w.record()["importer"]["matched_by"] == ["--input-file is the session stream", "holds the session stream open"]


def test_cli_requires_a_stream_source_and_paired_trigger_arguments(tmp_path):
    mod = _load()
    base = ["--importer-exe", EXE, "--report", "r", "--out", str(tmp_path / "o"), "--samples", str(tmp_path / "s"), "--log", str(tmp_path / "l")]
    for extra in ([], ["--stream", "s", "--control", "c"], ["--stream", "s", "--trigger-line", "x"]):
        with pytest.raises(SystemExit):
            mod.main(base + extra)


def test_recorded_cli_pid_defaults_to_the_lifecycle_record(tmp_path):
    mod, w = _load(), World(tmp_path, steps=[advance(), advance(), finish()])
    w.session()
    w.importer()
    w.status["nsys"] = {"pid": CLI}
    w.write_status()
    assert w.run(mod, trigger=False, nsys_pid=None) == 0
    assert w.record()["recorded_nsys_cli_pid"] == CLI and w.record()["importer"]["agent"]["linked_to_recorded_cli"] is True


def test_short_importer_flags_identify_the_offline_importer(tmp_path):
    """The post-run starts the importer as `QdstrmImporter -i <stream> -o <report>`."""
    mod, w = _load(), World(tmp_path)
    w.put(IMP, "QdstrmImporter", EXE, [EXE, "-i", STREAM, "-o", str(w.report)], fds=[STREAM])
    found = mod.find_importers(EXE, STREAM, str(w.report), proc=w.proc, uid=os.getuid())
    assert found[0]["matched_by"] == ["--input-file is the session stream", "--output-file is the session report", "holds the session stream open"]
    assert mod.find_importers(EXE, "/tmp/other.qdstrm", "/tmp/other.nsys-rep", proc=w.proc, uid=os.getuid()) == []


def test_stop_file_ends_the_observation_and_keeps_the_state(tmp_path):
    stop = tmp_path / "stop"
    mod, w = _load(), World(tmp_path, steps=[advance(), advance(), lambda w: (advance()(w), stop.write_text(""))] + [advance()] * 5)
    w.importer()
    rc = w.run(mod, trigger=False, stop_file=stop)
    rec = w.record()
    assert rc == 2 and rec["stopped_by"].startswith("stop file") and rec["final"]["state"] == mod.ACTIVE and rec["samples"] == 4 and rec["result"].startswith("STOPPED (ACTIVE)")


def test_record_names_the_host(tmp_path):
    import socket

    mod, w = _load(), World(tmp_path, steps=[advance(), advance(), finish()])
    w.importer()
    w.run(mod, trigger=False)
    assert w.record()["host"] == socket.gethostname()
