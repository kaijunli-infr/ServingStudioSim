#!/usr/bin/env python3
"""CPU-only rehearsal of the V4 acceptance gates (no GPU, nothing launched; not V4 evidence).

    python3 -I e1_v4scale_rehearsal.py synthetic --out DIR [--keep-work] [--only NAME ...]
        Builds synthetic V4 run directories (lifecycle record, import-probe record, GPU / memory / storage samples,
        stream, source and binary records, reservation records) and evaluates each through the real command-line
        validator checks/e1_v4scale_gates.py with the registered inputs (E1-V4-SCALE-INPUTS.json; only the kernel-row
        band is scaled to the fake exporter). For every fixture the verdict and the exact set of gates that are not
        PASS must equal the registered expectation.
    python3 -I e1_v4scale_rehearsal.py expect --gates V4S-GATES.json --expect-not-pass G2,G3,... --verdict FAIL
        Checks one validator output (used for the post-run on the preserved V2 run) the same way.

Positive fixtures: nominal, every phase exactly at its deadline, release upper edge exactly at the deadline, peaks
exactly at the ceilings (storage, memory, reservation 90 min, GPU-held 2700 s), temporary lack of importer progress, soft-deadline warning with every phase in time. Negative fixtures: one registered defect each.
Exit 0 only if every expectation holds. Stdlib only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

E1 = Path(__file__).resolve().parent
GATES = E1 / "checks" / "e1_v4scale_gates.py"
INPUTS = E1 / "E1-V4-SCALE-INPUTS.json"
T0 = 1_800_000_000.0
HOST = "pod4-gb300-10-tray16-f3"
STAND_IN = b"stand-in for the pinned executable (rehearsal fixture)"
PIN_SHA = hashlib.sha256(STAND_IN).hexdigest()  # the fixtures pin the stand-in file, so the validator's own hashing is exercised
OTHER_SHA = hashlib.sha256(STAND_IN + b" changed").hexdigest()
TREES = {"alignment": "7b4df784a5b6daff7c792e81cbc6503ba2c806ec", "launcher": "5acab28980ef9974bf4cb6dcd676481c357e655d"}
GITLINKS = {"alignment/load_generator/req-frontend": "5c20a259618976ecc360bda84abcc3653a04006e", "alignment/profiler/vllm": "0ab18711335780e062d65994a3243da0281d2081"}
API, ENGINE, NSYS, AGENT, IMPORTER = 4100, 4101, 4000, 4990, 5000
KB = 1024 * 1024
ALL = ["G0.source_and_executable", "G1.host_identity", "G2.one_stop_request_returned_ok", "G3.phases_in_order", "G4.phase_deadlines", "G5.importer_observed", "G6.gpu_release_interval", "G7.stream_durable",
       "G8.offline_import", "G8b.recovered_trace_complete", "G9.scientific_completeness", "G10.bounds", "G11.storage_and_memory", "G12.closeout",
       "G13.workload_and_recipe_identity", "G14.four_copy_trace_in_run", "G15.request_iteration_routing_rank_coverage"]


TRACE_BYTES = (E1 / "traces" / "e1_v4_scale.csv").read_bytes()
CONFIG_BYTES = (E1 / "configs_v4scale" / "e1_v4_scale" / "profile.json").read_bytes()
V3_TRACE_BYTES = (E1 / "traces" / "e1_v3_scale.csv").read_bytes()


def identity_ok() -> dict:
    """A complete, passing report of the V4 identity checker (the fields gate G13 reads)."""
    return {"verdict": "PASS", "checker": "e1_launch_identity_v4scale.py", "policy_version": "v4scale-1", "normalized": [], "results": [],
            "trace_identity": {"exercised": True, "expected_checks": 20, "checks": [{"check": f"check{i}", "ok": True} for i in range(20)]}}


def completeness_ok() -> dict:
    """The completeness checker's record for a complete eight-wave capture (the fields gate G15 reads)."""
    d = json.loads(INPUTS.read_text())
    return {"fails": 0, "cells": {f"b{p['b']}_kv{p['kv']}_w{p['wave']}": {"pure": p["pure_decode_iters_min"], "ramp": 30} for p in d["dry"]["plan"]},
            "profile_result": {"request_timing_count": 224, "parsed_device_ids": [0, 1, 2, 3], "validation": {"nvtx_forward_ranges": 34560, "kernel_rows": 49003328, "ok": True}},
            "token_accounting": {"requests": 224}, "corpus": {"segments": 224, "bad_len": 0, "npy_mismatch": 0, "dup_topk": 0}, "join": {"duplicate": 0, "missing": 0, "non_monotone": 0}}


def rehearsal_inputs() -> dict:
    """The registered inputs; only the kernel-row band is scaled (1 : 10 000) to what the fake exporter writes."""
    d = json.loads(INPUTS.read_text())
    d["expected"]["kernel_rows_band"] = [v // 10_000 for v in d["expected"]["kernel_rows_band"]]
    d["rehearsal_note"] = "kernel_rows_band scaled 1:10000 for the fake exporter; every other value is the registered one"
    return d


def fake_tools(tmp: Path, *, kernel_rows: int = 4900, nvtx: int = 34560, importer_rc: int = 0) -> tuple[Path, Path]:
    """Fake QdstrmImporter (`-i <stream> -o <report>`) and fake `nsys export` writing the tables the recovery gate reads."""
    tmp.mkdir(parents=True, exist_ok=True)
    imp = tmp / "QdstrmImporter"
    imp.write_text(f'#!/bin/sh\n[ {importer_rc} = 0 ] || {{ echo "import failed" >&2; exit {importer_rc}; }}\nhead -c 4096 /dev/zero > "$4"\n')
    imp.chmod(0o755)
    nsys = tmp / "nsys"
    nsys.write_text(f"""#!/usr/bin/env python3
import sqlite3, sys
out = sys.argv[sys.argv.index("-o") + 1]
con = sqlite3.connect(out)
con.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
con.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER)")
con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, globalPid INTEGER)")
con.execute("CREATE TABLE ANALYSIS_DETAILS (globalVid INTEGER, duration INTEGER, startTime INTEGER, stopTime INTEGER)")
con.execute("CREATE TABLE META_DATA_CAPTURE (name TEXT, value TEXT)")
con.execute("CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT)")
con.execute("CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER, utcTime TEXT, localTime TEXT, monotonic INTEGER)")
con.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL)", [(10 * i, 10 * i + 5, f"vllm_iteration({{i // 4}}): forward") for i in range({nvtx})])
con.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?)", [(1_000_000 + i, 1_000_000 + i + 1, i % 4) for i in range({kernel_rows})])
con.execute("INSERT INTO ANALYSIS_DETAILS VALUES (1, 500, 0, 500)")
con.execute("INSERT INTO META_DATA_CAPTURE VALUES ('START_TRIGGER', 'Immediate')")
con.execute("INSERT INTO META_DATA_EXPORT VALUES ('EXPORT_PRODUCT_VERSION', '2025.6.3.541')")
con.execute("INSERT INTO META_DATA_EXPORT VALUES ('EXPORT_PARAM_OUTPUT_PATH', ?)", (out,))
con.execute("INSERT INTO META_DATA_EXPORT VALUES ('EXPORT_TIME_UTC', ?)", (out + " now",))
con.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (1, 'a', 'a', 2)")
con.commit(); con.close()
""")
    nsys.chmod(0o755)
    return imp, nsys


@dataclass
class World:
    """Everything one fixture consists of; `mutate(world)` edits it before it is written."""

    t_stop: float = 1620.0
    accepted: float = 0.01
    flush: float = 100.0
    report_after_importing: float = 60.0
    imp_first_after_importing: float = 1.0
    imp_duration: float = 57.0
    http_returned_after_report: float = 1.0
    sigterm_after_report: float = 3.0
    close_after_sigterm: float = 2.5
    release_after_sigterm: float = 1.0
    lc: dict = field(default_factory=dict)
    probe: dict | None = field(default_factory=dict)
    gpu: list | None = None
    census: dict = field(default_factory=dict)
    mem: list | None = None
    storage: list | None = None
    post_storage: list | None = None
    post_memory: list | None = None
    source: dict | None = field(default_factory=dict)
    driver_bin: dict | None = field(default_factory=dict)
    launcher_bin: dict | None = field(default_factory=dict)
    reservation: dict | None = field(default_factory=dict)
    supervisor: dict | None = field(default_factory=dict)
    deleted: dict | None = field(default_factory=dict)
    identity: dict | None = field(default_factory=identity_ok)
    completeness: dict | None = field(default_factory=completeness_ok)
    trace_bytes: bytes | None = TRACE_BYTES
    config_bytes: bytes | None = CONFIG_BYTES
    extras: dict = field(default_factory=lambda: {"completeness_v2": 0, "extract": 0, "nsys_validation": 0, "closeout": 0})
    pin: dict = field(default_factory=dict)
    binary_bytes: bytes = STAND_IN
    server_log: str | None = "(APIServer pid=4100) INFO 11-19 00:27:00 [api_router.py:31] Stopping profiler...\n(APIServer pid=4100) INFO 11-19 00:29:40 [api_router.py:33] Profiler stopped.\n"
    durable_bytes: bytes | None = None
    in_session: dict = field(default_factory=lambda: {"kernel_rows": 4900, "nvtx": 34560})
    tools: dict = field(default_factory=lambda: {"kernel_rows": 4900, "nvtx": 34560, "importer_rc": 0})
    supply_tools: bool = True
    breach_files: list = field(default_factory=list)
    temp_gb: float = 2.7
    persistent_gb: float = 12.6
    scratch_gb: float = 8.6
    importer_rss_gb: float = 31.0
    mem_available_gb: float = 900.0

    # derived clocks (seconds after T0)
    @property
    def t_importing(self) -> float:
        return self.t_stop + self.flush

    @property
    def t_report(self) -> float:
        return self.t_importing + self.report_after_importing

    @property
    def t_sig(self) -> float:
        return self.t_report + self.sigterm_after_report

    @property
    def t_closed(self) -> float:
        return self.t_sig + self.close_after_sigterm

    def fill(self) -> None:
        """Derive every record from the clocks (called once, before `mutate`)."""
        ti, tr, ts, tc = self.t_importing, self.t_report, self.t_sig, self.t_closed
        phases = [("stop_requested", self.t_stop), ("stop_accepted", self.t_stop + self.accepted), ("capture_ended", ti), ("stream_written", ti), ("importing", ti), ("report_written", tr)]
        self.lc = {
            "schema_version": 2, "host": HOST, "host_identity": {"hostname": HOST, "boot_id": "54a59f08"}, "phase": "report_written", "launch_started_at": T0, "stop_requested_at": T0 + self.t_stop,
            "phase_history": [{"phase": p, "at": T0 + t} for p, t in phases], "http": {"status": "ok", "returned_at": T0 + tr + self.http_returned_after_report, "requests_sent": 1},
            "policy": {"gpu_held_ceiling_s": 2700.0, "session_close_timeout_s": 300.0, "hard_deadline_s": 600.0, "soft_deadline_s": 300.0, "server_term_grace_s": 60.0}, "escalations": [], "warnings": [], "control_commands": [],
            "deadlines": {"soft_at": T0 + self.t_stop + 300.0, "hard_at": T0 + self.t_stop + 600.0, "hard_extension_s": 0.0, "max_extension_s": 120.0, "gpu_held_ceiling_at": T0 + 2700.0},
            "nsys": {"pid": NSYS, "alive": False, "identity": {"pid": NSYS, "starttime": 777}}, "server": {"api_server": {"pid": API}, "engine_processes": [{"pid": ENGINE}, {"pid": ENGINE + 1}], "alive": False},
            "gpu": {"release_requested_at": None, "released_at": T0 + ts + 1.5}, "session_closed_at": T0 + tc, "actions": [{"at": T0 + ts, "reason": "shutdown", "pid": API, "action": "SIGTERM"}],
            "report": {"path": "x.nsys-rep", "size": 4096, "written_at": T0 + tr},
        }
        first = ti + self.imp_first_after_importing
        self.probe = {
            "probe": "e1_import_probe.py", "host": HOST, "started_at": T0 + 1.0, "min_progress": 2,
            "importer": {"pid": IMPORTER, "starttime": 900, "exe": "/opt/nvidia/nsight-systems/2025.6.3/host-linux-armv8/QdstrmImporter", "matched_by": ["--input-file is the session stream", "holds the session stream open"], "first_seen_at": T0 + first,
                         "agent": {"pid": AGENT, "found": True, "is_agent": True, "session_name": f"profile-{NSYS}", "ppid": 1, "linked_to_recorded_cli": True}},
            "transitions": [{"at": T0 + 1.0, "state": "NOT_FOUND"}, {"at": T0 + first, "state": "NO_PROGRESS"}, {"at": T0 + first + 1.0, "state": "ACTIVE"}, {"at": T0 + first + self.imp_duration, "state": "SETTLING"}, {"at": T0 + first + self.imp_duration + 1.5, "state": "COMPLETED"}],
            "trigger": {"requested": False, "written": False}, "final": {"state": "COMPLETED", "importer_seen": True, "nsight_generated_marker": True}, "result": "COMPLETED",
        }
        rel = ts + self.release_after_sigterm - 0.2  # the released reading window is [stamp, stamp + 0.2]
        self.gpu = gpu_series(0.0, tc + 60, lambda t: "idle" if t < 60 else ("held" if t < rel else "released"), anchor=rel)
        self.census = {"t0": True, "t20": True}
        self.mem = [{"at": T0 + t, "host": HOST, "mem_total_kb": 1004 * KB, "mem_available_kb": int(self.mem_available_gb * KB), "candidates": 1 if first <= t < first + self.imp_duration else 0,
                     "importer": {"pid": IMPORTER, "starttime": 900, "rss_kb": int(self.importer_rss_gb * KB * min(1.0, (t - first + 1) / 20)), "hwm_kb": int(self.importer_rss_gb * KB * min(1.0, (t - first + 1) / 20))} if first <= t < first + self.imp_duration else None}
                    for t in frange(-5.0, tc + 10, 1.0)]
        self.storage = [{"at": T0 + t, "host": HOST, "groups": {"temporary": {"bytes": int((1.36 + (self.temp_gb - 1.36) * min(1.0, max(0.0, (t - self.t_stop) / max(1.0, self.flush)))) * 1e9), "files": 3, "free_bytes": 10**12},
                                                                 "persistent": {"bytes": int(self.persistent_gb * 1e9 * min(1.0, max(0.05, t / max(1.0, tc)))), "files": 300, "free_bytes": 10**13}}}
                        for t in frange(-5.0, tc + 10, 2.0)]
        now = time.time()
        self.post_storage = [{"at": now + t, "host": HOST, "groups": {"temporary": {"bytes": int((1.36 + 1.32 + self.scratch_gb * min(1.0, max(0.0, t / 20))) * 1e9), "files": 5, "free_bytes": 10**12}, "persistent": {"bytes": int(self.persistent_gb * 1e9), "files": 320, "free_bytes": 10**13}}}
                             for t in frange(-10.0, 240, 2.0)]
        self.post_memory = [{"at": now + t, "host": HOST, "mem_total_kb": 1004 * KB, "mem_available_kb": int(self.mem_available_gb * KB), "candidates": 1, "importer": {"pid": 6000, "starttime": 950, "rss_kb": int(self.importer_rss_gb * KB), "hwm_kb": int(self.importer_rss_gb * KB)}}
                            for t in frange(-10.0, 240, 1.0)]
        self.source = {"commit": "c" * 40, "mode": "real", "trees": dict(TREES), "gitlinks": dict(GITLINKS), "tracked_changes": 0, "descends_from_pinned_commit": True}
        self.supervisor = {"supervisor": "e1_supervisor.py", "host": HOST, "samples": int(tc) + 30, "causes": [], "response": None, "monitors_effective": True, "signals_sent_by_supervisor": 0, "complete": True}
        self.driver_bin = {"sha256_before_launcher": PIN_SHA, "sha256_after_launcher": PIN_SHA, "unchanged": True, "cargo_compiling_lines_in_run_log": 0, "stat_before": "1 2 3", "stat_after": "1 2 3"}
        self.launcher_bin = {"mode": "prebuilt (no compilation in this launch path)", "sha256": PIN_SHA, "pinned_sha256": PIN_SHA, "sha256_after_launch": PIN_SHA, "unchanged": True, "cargo_consulted": False, "build_tool_guard": {"installed": True, "active": True, "blocked": []}}
        self.reservation = {"pod": "dsv41-gb300-reserve-tray16", "node": HOST, "phase": "Running", "creation_timestamp": "2026-11-19T00:00:00Z"}
        self.deleted = {"pod": "dsv41-gb300-reserve-tray16", "deleted_observed_at": "2026-11-19T00:50:00Z", "gpu_pods_on_node": 0}
        self.pin = {"pass": "e1_v4_scale", "code": {"commit": "b" * 40, "tree": dict(TREES), "gitlinks": dict(GITLINKS)}, "session_runner_sha256": PIN_SHA}

    def retime(self) -> None:
        """Rebuild the derived records after a clock field was changed, keeping nothing of the old ones."""
        self.fill()


def frange(a: float, b: float, step: float) -> list:
    n = int((b - a) / step) + 1
    return [round(a + i * step, 3) for i in range(n)]


def gpu_series(start: float, end: float, state, *, interval: float = 2.0, query: float = 0.2, anchor: float | None = None) -> list:
    """Sampler observations every interval + query seconds; `anchor` forces one stamp exactly there. Reading windows are [stamp, stamp + query]."""
    step = interval + query
    stamps = frange(start, end, step)
    if anchor is not None:
        shift = anchor - max(s for s in stamps if s <= anchor)
        stamps = [round(s + shift, 3) for s in stamps]
    out = []
    for s in stamps:
        st = state(s)
        out.append({"at": T0 + s, "host": HOST, "source": "e1_gpu_watch", "available": st != "unavailable", "memory_used_mib": [260_943] * 4 if st == "held" else [0] * 4, "compute_pids": [1, 2, 3, 4] if st == "held" else []})
    return out


def write_world(w: World, root: Path) -> dict:
    """Write the fixture under `root`; returns the validator's arguments."""
    run = root / "e1_v4_scale"
    nsys_dir, vllm, lg = run / "profile" / "nsys", run / "profile" / "vllm", run / "profile" / "load_generator"
    for d in (nsys_dir, vllm, lg):
        d.mkdir(parents=True)
    data = b"q" * 4096
    preserved = root / "x.raw-preserve" / "x.qdstrm"
    preserved.parent.mkdir()
    preserved.write_bytes(data)
    durable = run / "preserved" / "x.qdstrm"
    durable.parent.mkdir()
    durable.write_bytes(data if w.durable_bytes is None else w.durable_bytes)
    lc = dict(w.lc)
    lc.setdefault("stream", {"path": "/tmp/nsys-report-1.qdstrm", "size": 4066, "preserved_path": str(preserved), "preserved_size": 4096, "observations": [],
                             "final": {"available": True, "size": 4096, "sha256": hashlib.sha256(data).hexdigest(), "taken_at": lc.get("session_closed_at"), "session_closed_at": lc.get("session_closed_at")}})
    (nsys_dir / "lifecycle.json").write_text(json.dumps(lc))
    (run / "run.log").write_text(f"[e1_v4_scale] start 2026-11-19T00:00:00Z on {HOST} x\n")
    for name, data in (("trace.csv", w.trace_bytes), ("profile.json", w.config_bytes)):
        if data is not None:
            (run / name).write_bytes(data)
    if w.server_log is not None:
        (vllm / "x_server.log").write_text(w.server_log)
    for name, obj in (("import_probe.json", w.probe), ("launch_source.json", w.source), ("binary_check.json", w.driver_bin), ("reservation.json", w.reservation), ("supervisor.json", w.supervisor)):
        if obj is not None:
            (run / name).write_text(json.dumps(obj))
    if w.launcher_bin is not None:
        (lg / "frontend_binary.json").write_text(json.dumps(w.launcher_bin))
    for name, rows in (("gpu_observations.jsonl", w.gpu), ("mem_observations.jsonl", w.mem), ("storage_observations.jsonl", w.storage)):
        if rows is not None:
            (run / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    for label, released in w.census.items():
        (run / f"gpu_release_census_{label}.json").write_text(json.dumps({"schema_version": 1, "host": HOST, "at": lc.get("session_closed_at", T0) + (300 if label == "t0" else 320), "compute_apps": [] if released else [{"pid": 1}],
                                                                              "gpus": [{"index": i, "memory_used_mib": 0 if released else 260_943} for i in range(4)]}))
    for name in w.breach_files:
        (run / name).write_text("{}\n")
    if w.in_session:
        _, nsys = fake_tools(root / "ins", **w.in_session)
        subprocess.run([str(nsys), "export", "-o", str(nsys_dir / "x.sqlite"), "x"], check=True)
    imp, nsys = fake_tools(root / "tools", **w.tools)
    cpu = root / "cpu"
    cpu.mkdir()
    args = {"run": run, "durable": durable, "importer": imp if w.supply_tools else None, "nsys": nsys if w.supply_tools else None, "scratch": root / "scratch", "extras": dict(w.extras), "binary": root / "session_runner"}
    (cpu / "pin.json").write_text(json.dumps(w.pin))
    (cpu / "provenance.json").write_text(json.dumps({"sha256": PIN_SHA}))
    (cpu / "inputs.json").write_text(json.dumps(rehearsal_inputs()))
    args.update(pin=cpu / "pin.json", provenance=cpu / "provenance.json", inputs=cpu / "inputs.json")
    for key, name, obj in (("identity", "identity.json", w.identity), ("completeness", "completeness.json", w.completeness), ("deleted", "reservation_deleted.json", w.deleted)):
        if obj is not None:
            (cpu / name).write_text(json.dumps(obj))
        args[key] = cpu / name if obj is not None else None
    for key, name, rows in (("post_storage", "storage_postrun.jsonl", w.post_storage), ("post_memory", "mem_postrun.jsonl", w.post_memory)):
        if rows is not None:
            (cpu / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        args[key] = cpu / name if rows is not None else None
    args["binary"].write_bytes(w.binary_bytes)
    return args


@dataclass
class Fixture:
    name: str
    kind: str  # positive | boundary | negative
    what: str
    verdict: str
    not_pass: dict  # gate -> status expected (every other gate must PASS)
    clocks: dict = field(default_factory=dict)
    mutate: object = None


def _set(path: str, value):
    def m(w: World):
        obj, *keys = path.split(".")
        cur = getattr(w, obj)
        for k in keys[:-1]:
            cur = cur[int(k)] if isinstance(cur, list) else cur[k]
        last = keys[-1]
        if isinstance(cur, list):
            cur[int(last)] = value
        else:
            cur[last] = value

    return m


def _attr(name: str, value):
    def m(w: World):
        setattr(w, name, value)

    return m


def _both(*ms):
    def m(w: World):
        for f in ms:
            f(w)

    return m


def _phases(fn):
    def m(w: World):
        w.lc["phase_history"] = fn(w.lc["phase_history"])

    return m


def _signal(pid, action="SIGTERM", at_report_offset=-5.0):
    def m(w: World):
        w.lc["actions"].append({"at": T0 + w.t_report + at_report_offset, "reason": "test", "pid": pid, "action": action})

    return m


def _probe_states(states, final, seen=True, keep_importer=True):
    def m(w: World):
        base = w.probe["importer"]["first_seen_at"]
        w.probe["transitions"] = [{"at": base + i, "state": s} for i, s in enumerate(states)]
        w.probe["final"] = {"state": final, "importer_seen": seen}
        if not keep_importer:
            del w.probe["importer"]

    return m


def _gpu(state_fn, **kw):
    def m(w: World):
        w.gpu = gpu_series(0.0, w.t_closed + 60, lambda t: state_fn(w, t), **kw)

    return m


def _release_at(delta: float):
    """First 'released' reading window ends exactly `delta` s after the SIGTERM."""

    def m(w: World):
        rel = w.t_sig + delta - 0.2
        w.gpu = gpu_series(0.0, w.t_sig + delta + 120, lambda t: "idle" if t < 60 else ("held" if t < rel else "released"), anchor=rel)
        w.lc["session_closed_at"] = T0 + w.t_sig + 2.5
        w.lc["gpu"]["released_at"] = T0 + w.t_sig + delta

    return m


def _drop_gpu_window(a: float, b: float):
    def m(w: World):
        w.gpu = [o for o in w.gpu if not (T0 + w.t_sig + a <= o["at"] <= T0 + w.t_sig + b)]

    return m


def _scale_rows(attr: str, group: str | None, gb: float):
    def m(w: World):
        rows = getattr(w, attr)
        if group:
            peak = max(r["groups"][group]["bytes"] for r in rows)
            for r in rows:
                r["groups"][group]["bytes"] = int(r["groups"][group]["bytes"] * gb * 1e9 / peak)
        else:
            for r in rows:
                if r.get("importer"):
                    r["importer"]["rss_kb"] = r["importer"]["hwm_kb"] = int(gb * KB)

    return m


def _mem_available(gb: float):
    def m(w: World):
        w.mem[len(w.mem) // 2]["mem_available_kb"] = int(gb * KB)

    return m


def _no_importer_in(attr: str):
    def m(w: World):
        for r in getattr(w, attr):
            r["importer"] = None

    return m


def _gap(attr: str, seconds: float):
    def m(w: World):
        rows = getattr(w, attr)
        mid = rows[len(rows) // 2]["at"]
        setattr(w, attr, [r for r in rows if not (mid <= r["at"] <= mid + seconds)])

    return m


def fixtures() -> list[Fixture]:
    G = {g.split(".")[0]: g for g in ALL}

    def np(**kw):  # short names -> full gate names
        return {G[k]: v for k, v in kw.items()}

    F, INC = "FAIL", "INCONCLUSIVE"
    fx = [
        Fixture("nominal", "positive", "normal stop, importer observed active and completed, all peaks measured", "PASS", {}),
        Fixture("phases_exactly_at_deadlines", "boundary", "stop accepted 30 s, flush 300 s, import 300 s, stop→report 600 s, session close 300 s", "PASS", {},
                clocks={"accepted": 30.0, "flush": 300.0, "report_after_importing": 300.0, "imp_first_after_importing": 0.0, "imp_duration": 300.0, "sigterm_after_report": 3.0, "close_after_sigterm": 300.0, "t_stop": 1000.0}),
        Fixture("release_upper_edge_exactly_90s", "boundary", "first 'released' reading window ends 90.0 s after the SIGTERM", "PASS", {}, mutate=_release_at(90.0)),
        Fixture("peaks_exactly_at_ceilings", "boundary", "temporary 15.0 GB, persistent 18.0 GB, importer 128 GiB, MemAvailable exactly at the floor", "PASS", {},
                mutate=_both(_scale_rows("post_storage", "temporary", 15.0), _scale_rows("storage", "persistent", 18.0), _scale_rows("post_storage", "persistent", 18.0), _scale_rows("mem", None, 128.0), _mem_available(64.0))),
        Fixture("soft_warning_total_exactly_600s", "boundary", "flush 300 s + import 300 s = stop → report 600.0 s, with the controller's soft warning recorded at 300 s", "PASS", {},
                clocks={"flush": 300.0, "report_after_importing": 300.0, "imp_first_after_importing": 0.0, "imp_duration": 300.0, "t_stop": 1000.0}, mutate=_set("lc.escalations", [{"kind": "soft_deadline", "phase": "importing", "elapsed_s": 300.0}])),
        Fixture("soft_deadline_warning_recorded", "positive", "stop → report 400 s (flush 250 s, import 140 s): the controller's soft-deadline warning is recorded, every phase is inside its deadline", "PASS", {},
                clocks={"flush": 250.0, "report_after_importing": 150.0, "imp_duration": 140.0}, mutate=_set("lc.escalations", [{"kind": "soft_deadline", "phase": "importing", "elapsed_s": 300.0}])),
        Fixture("importer_temporarily_inactive", "positive", "ACTIVE, then NO_PROGRESS, then ACTIVE again, then completed", "PASS", {}, mutate=_probe_states(["NOT_FOUND", "NO_PROGRESS", "ACTIVE", "NO_PROGRESS", "ACTIVE", "SETTLING", "COMPLETED"], "COMPLETED")),
        # phase deadlines
        Fixture("stop_accepted_late", "negative", "stop accepted after 30.5 s", F, np(G4=F), clocks={"accepted": 30.5}),
        Fixture("flush_over_deadline", "negative", "flush 300.5 s", F, np(G4=F), clocks={"flush": 300.5}),
        Fixture("import_over_deadline", "negative", "importer first seen → gone 300.5 s", F, np(G4=F), clocks={"report_after_importing": 305.0, "imp_duration": 300.5}),
        Fixture("stop_to_report_over_deadline", "negative", "stop → report 600.5 s with each phase inside its own deadline", F, np(G4=F), clocks={"flush": 300.0, "report_after_importing": 300.5, "imp_duration": 290.0, "t_stop": 1000.0}),
        Fixture("soft_warning_then_total_over_deadline", "negative", "soft warning recorded; flush 300 s and import 290 s are inside their deadlines but stop → report is 600.5 s", F, np(G4=F),
                clocks={"flush": 300.0, "report_after_importing": 300.5, "imp_duration": 290.0, "t_stop": 1000.0}, mutate=_set("lc.escalations", [{"kind": "soft_deadline", "phase": "importing", "elapsed_s": 300.0}])),
        Fixture("phases_inside_but_total_over", "negative", "flush 299 s and import 299 s, each inside its deadline; with the gaps between them stop → report is 601 s", F, np(G4=F),
                clocks={"flush": 299.0, "report_after_importing": 302.0, "imp_first_after_importing": 1.5, "imp_duration": 299.0, "t_stop": 1000.0}),
        Fixture("flush_over_while_total_inside", "negative", "flush 300.5 s, import 30 s: stop → report 340.5 s is inside the total, the flush is not inside its own deadline", F, np(G4=F),
                clocks={"flush": 300.5, "report_after_importing": 40.0, "imp_duration": 30.0}),
        Fixture("hard_deadline_extended", "negative", "an operator `extend` moved the controller's hard deadline by 60 s", F, np(G3=F, G4=F),
                mutate=_both(_set("lc.control_commands", [{"at": T0 + 1700, "command": "extend 60 test", "phase": "importing", "result": "hard deadline extended by 60 s (operator)"}]), lambda w: w.lc["deadlines"].update(hard_extension_s=60.0, hard_at=w.lc["deadlines"]["hard_at"] + 60.0))),
        Fixture("hard_deadline_not_the_registered_total", "negative", "the controller ran with a 900 s hard deadline", F, np(G4=F), mutate=_both(_set("lc.policy.hard_deadline_s", 900.0), lambda w: w.lc["deadlines"].update(hard_at=w.lc["deadlines"]["hard_at"] + 300.0))),
        Fixture("hard_deadline_record_missing", "negative", "the lifecycle record has no deadline block", INC, np(G4=INC), mutate=lambda w: w.lc.pop("deadlines")),
        Fixture("session_close_over_deadline", "negative", "session closed 300.5 s after the SIGTERM", F, np(G4=F, G10=F), clocks={"close_after_sigterm": 300.5, "t_stop": 1000.0}),
        Fixture("phase_boundary_unobserved", "negative", "the probe never recorded the importer leaving", INC, np(G4=INC, G5=INC), mutate=_probe_states(["NOT_FOUND", "NO_PROGRESS", "ACTIVE"], "ACTIVE")),
        # phase order, forced release
        Fixture("phase_missing", "negative", "capture_ended missing from the phase history", F, np(G3=F), mutate=_phases(lambda h: [x for x in h if x["phase"] != "capture_ended"])),
        Fixture("stop_failed", "negative", "final phase stop_failed", F, np(G3=F, G4=INC), mutate=_both(_set("lc.phase", "stop_failed"), _phases(lambda h: h[:-1] + [{"phase": "stop_failed", "at": h[-1]["at"]}]))),
        Fixture("hard_deadline_escalation", "negative", "a hard-deadline escalation is recorded", F, np(G3=F), mutate=_set("lc.escalations", [{"kind": "hard_deadline", "phase": "importing", "elapsed_s": 600}])),
        Fixture("gpu_held_ceiling_escalation", "negative", "a GPU-held-ceiling escalation is recorded", F, np(G3=F), mutate=_set("lc.escalations", [{"kind": "gpu_held_ceiling", "phase": "capturing", "elapsed_s": 2700}])),
        Fixture("operator_control_line", "negative", "an `extend` control line was accepted", F, np(G3=F), mutate=_set("lc.control_commands", [{"at": T0 + 1700, "command": "extend 60 test", "phase": "importing", "result": "extended"}])),
        Fixture("forced_release", "negative", "a GPU release was requested (hard deadline)", F, np(G3=F), mutate=_set("lc.gpu.release_requested_at", T0 + 1700)),
        # stop request
        Fixture("stop_request_error", "negative", "the stop request returned an error", F, np(G2=F), mutate=_set("lc.http.status", "error: connection reset")),
        Fixture("two_stop_requests", "negative", "two stop requests left the launcher", F, np(G2=F), mutate=_set("lc.http.requests_sent", 2)),
        Fixture("stop_returned_after_sigterm", "negative", "the stop request returned after the server was signalled", F, np(G2=F), clocks={"http_returned_after_report": 4.0}),
        Fixture("server_log_missing", "negative", "no server log to count the API-server lines", INC, np(G2=INC), mutate=_attr("server_log", None)),
        Fixture("profiler_stopped_line_missing", "negative", "the API server never logged 'Profiler stopped.'", F, np(G2=F), mutate=_attr("server_log", "(APIServer pid=4100) INFO 11-19 00:27:00 [api_router.py:31] Stopping profiler...\n")),
        # importer
        Fixture("probe_record_missing", "negative", "no import-probe record", INC, np(G1=INC, G4=INC, G5=INC), mutate=_attr("probe", None)),
        Fixture("importer_not_found", "negative", "the probe never identified an importer", INC, np(G4=INC, G5=INC), mutate=_probe_states(["NOT_FOUND"], "NOT_FOUND", seen=False, keep_importer=False)),
        Fixture("importer_ambiguous", "negative", "two processes matched the importer criteria", INC, np(G4=INC, G5=INC), mutate=_probe_states(["AMBIGUOUS"], "AMBIGUOUS", seen=False, keep_importer=False)),
        Fixture("importer_present_never_active", "negative", "importer identified, activity never established from its own counters", INC, np(G5=INC), mutate=_probe_states(["NOT_FOUND", "NO_PROGRESS", "SETTLING", "COMPLETED"], "COMPLETED")),
        Fixture("import_finished_before_observation", "negative", "report complete but the importer was never seen", INC, np(G4=INC, G5=INC), mutate=_probe_states(["COMPLETED"], "COMPLETED", seen=False, keep_importer=False)),
        Fixture("importer_terminated", "negative", "importer gone without a report", F, np(G5=F), mutate=_probe_states(["NOT_FOUND", "NO_PROGRESS", "ACTIVE", "SETTLING", "TERMINATED"], "TERMINATED")),
        Fixture("signal_to_importer", "negative", "a SIGTERM to the importer pid is recorded", F, np(G5=F), mutate=_signal(IMPORTER)),
        Fixture("signal_to_agent", "negative", "a SIGKILL to the Nsight agent is recorded", F, np(G5=F), mutate=_signal(AGENT, "SIGKILL")),
        Fixture("signal_to_cli_before_report", "negative", "the Nsight CLI was signalled before the report", F, np(G5=F), mutate=_signal(NSYS, "SIGINT")),
        Fixture("signal_to_unrecorded_pid", "negative", "a signal to a pid that is not a recorded server process", F, np(G5=F), mutate=_signal(9999)),
        # GPU release
        Fixture("release_upper_edge_over_90s", "negative", "first 'released' reading window ends 90.5 s after the SIGTERM", F, np(G6=F), mutate=_release_at(90.5)),
        Fixture("gpu_held_again", "negative", "a 'held' observation after the first 'released' one", F, np(G6=F), mutate=_gpu(lambda w, t: "idle" if t < 60 else ("released" if w.t_sig + 1 <= t < w.t_sig + 8 else "held" if t < w.t_sig + 14 else "released"))),
        Fixture("gpu_never_released", "negative", "no 'released' observation after the last 'held' one", F, np(G6=INC, G10=INC, G12=F), mutate=_both(_gpu(lambda w, t: "idle" if t < 60 else "held"), _set("extras.closeout", 1))),
        Fixture("gpu_samples_missing_across_release", "negative", "no sampler observation for 12 s around the release", INC, np(G6=INC), mutate=_drop_gpu_window(-6.0, 6.0)),
        Fixture("gpu_sampler_unavailable_at_release", "negative", "nvidia-smi unavailable in the observation between held and released", INC, np(G6=INC), mutate=_gpu(lambda w, t: "idle" if t < 60 else ("held" if t < w.t_sig - 2 else "unavailable" if t < w.t_sig + 1 else "released"))),
        Fixture("gpu_sampler_file_missing", "negative", "no sampler file", INC, np(G6=INC, G10=INC), mutate=_attr("gpu", None)),
        Fixture("census_shows_held", "negative", "the t20 census still shows memory held", F, np(G6=F), mutate=_attr("census", {"t0": True, "t20": False})),
        Fixture("census_missing", "negative", "no census file", INC, np(G6=INC), mutate=_attr("census", {})),
        Fixture("sigterm_not_recorded", "negative", "the controller recorded no SIGTERM to the API server", INC, np(G2=INC, G4=INC, G6=INC, G10=INC), mutate=_set("lc.actions", [])),
        Fixture("released_before_sigterm", "negative", "the GPUs read free before the server was signalled", F, np(G6=F), mutate=_release_at(-5.0)),
        # stream and recovery
        Fixture("durable_copy_differs", "negative", "the durable copy is not the recorded stream", F, np(G7=F, G8=F, G8b=F, G11=INC), mutate=_attr("durable_bytes", b"x" * 4096)),
        Fixture("final_stream_record_missing", "negative", "the controller wrote no final stream record", F, np(G7=F, G8=F, G8b=F, G11=INC), mutate=_set("lc.stream", {"preserved_path": None, "final": {}})),
        Fixture("offline_import_fails", "negative", "the importer exits non-zero on the durable stream", F, np(G8=F, G8b=F), mutate=_set("tools.importer_rc", 1)),
        Fixture("recovered_iterations_below_band", "negative", "8000 iterations recovered (band 8400–9000)", F, np(G8b=F), mutate=_both(_set("tools.nvtx", 32000), _set("in_session.nvtx", 32000))),
        Fixture("recovered_rows_below_band", "negative", "kernel rows 5 % below the band", F, np(G8b=F), mutate=_both(_set("tools.kernel_rows", 4400), _set("in_session.kernel_rows", 4400))),
        Fixture("recovered_differs_from_in_session", "negative", "offline trace has fewer kernel rows than the in-session export", F, np(G8b=F), mutate=_set("tools.kernel_rows", 4880)),
        Fixture("half_trace_two_waves_only", "negative", "only four of the eight waves in the trace (4320 iterations)", F, np(G8b=F, G9=F), mutate=_both(_set("tools.nvtx", 17280), _set("in_session.nvtx", 17280), _set("tools.kernel_rows", 2450), _set("in_session.kernel_rows", 2450), _set("extras.completeness_v2", 1))),
        Fixture("recovery_tools_not_supplied", "negative", "importer / exporter not given to the validator", INC, np(G8=INC, G8b=INC, G11=INC), mutate=_attr("supply_tools", False)),
        # scientific
        Fixture("completeness_fails", "negative", "eight-wave completeness checker exit 1", F, np(G9=F), mutate=_set("extras.completeness_v2", 1)),
        Fixture("nsys_validation_fails", "negative", "Nsight validation exit 1", F, np(G9=F), mutate=_set("extras.nsys_validation", 1)),
        Fixture("extract_not_run", "negative", "extract exit code not supplied", INC, np(G9=INC), mutate=lambda w: w.extras.pop("extract")),
        Fixture("config_identity_fails", "negative", "configuration identity report FAIL", F, np(G9=F, G13=F), mutate=_set("identity.verdict", "FAIL")),
        Fixture("config_identity_missing", "negative", "no configuration identity report", INC, np(G9=INC, G13=INC), mutate=_attr("identity", None)),
        # source and executable
        Fixture("executable_changed_during_run", "negative", "driver hashed another executable after the launcher", F, np(G0=F), mutate=_both(_set("driver_bin.sha256_after_launcher", OTHER_SHA), _set("driver_bin.unchanged", False))),
        Fixture("executable_rebuilt_in_place", "negative", "same bytes recorded but inode/mtime changed and a compile line logged", F, np(G0=F), mutate=_both(_set("driver_bin.stat_after", "9 9 3"), _set("driver_bin.cargo_compiling_lines_in_run_log", 1))),
        Fixture("build_tool_requested", "negative", "the guard blocked a cargo request", F, np(G0=F), mutate=_set("launcher_bin.build_tool_guard", {"installed": True, "active": True, "blocked": [{"argv": ["cargo", "build"]}]})),
        Fixture("guard_not_armed", "negative", "launcher record without an armed guard", F, np(G0=F), mutate=_set("launcher_bin.build_tool_guard", {"installed": False, "active": False, "blocked": []})),
        Fixture("launcher_binary_record_missing", "negative", "no launcher record of the executable", INC, np(G0=INC), mutate=_attr("launcher_bin", None)),
        Fixture("executable_differs_at_postrun", "negative", "the executable on disk at post-run time is not the pinned one", F, np(G0=F), mutate=_attr("binary_bytes", STAND_IN + b" changed")),
        Fixture("launch_tree_not_pinned", "negative", "launch tree `alignment` differs from the pin", F, np(G0=F), mutate=_set("source.trees", {"alignment": "0" * 40, "launcher": TREES["launcher"]})),
        Fixture("launch_tree_dirty", "negative", "tracked changes in the launch tree", F, np(G0=F), mutate=_set("source.tracked_changes", 2)),
        Fixture("submodule_not_pinned", "negative", "vLLM gitlink differs from the pin", F, np(G0=F), mutate=_set("source.gitlinks", {**GITLINKS, "alignment/profiler/vllm": "1" * 40})),
        Fixture("rehearsal_mode_launch", "negative", "the launch record says the driver ran in rehearsal mode", F, np(G0=F), mutate=_set("source.mode", "rehearsal")),
        Fixture("launch_source_record_missing", "negative", "no launch source record", INC, np(G0=INC), mutate=_attr("source", None)),
        # host
        Fixture("reservation_other_node", "negative", "reservation record names another node", F, np(G1=F), mutate=_set("reservation.node", "pod4-gb300-10-tray14-f3")),
        Fixture("memory_sampler_other_host", "negative", "memory samples carry another host", F, np(G1=F), mutate=lambda w: [r.update(host="pod4-gb300-10-tray14-f3") for r in w.mem]),
        # storage and memory
        Fixture("temporary_storage_over_ceiling", "negative", "post-run scratch peak makes temporary storage 15.05 GB", F, np(G11=F), mutate=_scale_rows("post_storage", "temporary", 15.05)),
        Fixture("persistent_storage_over_ceiling", "negative", "run directory peak 18.05 GB", F, np(G11=F), mutate=_scale_rows("storage", "persistent", 18.05)),
        Fixture("importer_memory_over_ceiling", "negative", "in-session importer 128.5 GiB", F, np(G11=F), mutate=_scale_rows("mem", None, 128.5)),
        Fixture("offline_importer_memory_over_ceiling", "negative", "offline importer 128.5 GiB", F, np(G11=F), mutate=_scale_rows("post_memory", None, 128.5)),
        Fixture("supervisor_record_missing", "negative", "no supervisor record", INC, np(G1=INC, G11=INC), mutate=_attr("supervisor", None)),
        Fixture("supervisor_record_incomplete", "negative", "the supervisor did not end through its stop file", INC, np(G11=INC), mutate=_set("supervisor.complete", False)),
        Fixture("supervisor_gave_abort_response", "negative", "the supervisor wrote its release line for a memory breach", F, np(G11=F),
                mutate=_both(_set("supervisor.causes", [{"cause": "breach", "source": "MEMORY-BREACH", "kind": "importer_rss_above_ceiling"}]), _set("supervisor.response", {"outcome": "release requested through the controller", "steps": [{"kind": "R1 control line", "acknowledged": True}]}))),
        Fixture("supervisor_lost_a_monitor", "negative", "the storage sampler stopped producing samples; the supervisor responded", F, np(G11=F),
                mutate=_both(_set("supervisor.monitors_effective", False), _set("supervisor.causes", [{"cause": "monitor_lost", "source": "storage sampler", "kind": "last sample too old"}]), _set("supervisor.response", {"outcome": "registry cleanup rc 0", "steps": [{"kind": "R2 registry cleanup"}]}))),
        Fixture("host_memory_below_floor", "negative", "MemAvailable 63.5 GB in one sample", F, np(G11=F), mutate=_mem_available(63.5)),
        Fixture("breach_file_present", "negative", "the memory sampler wrote its breach file", F, np(G11=F), mutate=_attr("breach_files", ["MEMORY-BREACH"])),
        Fixture("run_storage_unmeasured", "negative", "no storage samples during the run", INC, np(G1=INC, G11=INC), mutate=_attr("storage", None)),
        Fixture("postrun_storage_unmeasured", "negative", "no storage samples during the post-run (scratch peak unmeasured, as in V2)", INC, np(G11=INC), mutate=_attr("post_storage", None)),
        Fixture("storage_samples_gap", "negative", "60 s without a storage sample during the run", INC, np(G11=INC), mutate=_gap("storage", 60.0)),
        Fixture("memory_samples_missing", "negative", "no memory samples during the run", INC, np(G1=INC, G11=INC), mutate=_attr("mem", None)),
        Fixture("memory_samples_gap", "negative", "60 s without a memory sample during the run", INC, np(G11=INC), mutate=_gap("mem", 60.0)),
        Fixture("importer_memory_unmeasured", "negative", "the memory sampler never identified the importer", INC, np(G11=INC), mutate=_no_importer_in("mem")),
        Fixture("offline_importer_memory_unmeasured", "negative", "the post-run memory sampler never identified the importer", INC, np(G11=INC), mutate=_no_importer_in("post_memory")),
        # bounds, closeout
        Fixture("gpu_held_over_ceiling", "negative", "GPUs held 2 704 s from the launch", F, np(G10=F), clocks={"t_stop": 2540.0}),
        Fixture("reservation_over_90_min", "negative", "reservation held 91 min", F, np(G10=F), mutate=_set("deleted.deleted_observed_at", "2026-11-19T01:31:00Z")),
        Fixture("reservation_deleted_record_missing", "negative", "no record of the reservation's deletion", INC, np(G10=INC, G12=INC), mutate=_attr("deleted", None)),
        Fixture("gpu_pod_left_on_node", "negative", "a GPU pod is still on the node after the deletion", F, np(G12=F), mutate=_set("deleted.gpu_pods_on_node", 1)),
        Fixture("closeout_not_clean", "negative", "closeout exit 1", F, np(G12=F), mutate=_set("extras.closeout", 1)),
        Fixture("closeout_not_run", "negative", "closeout exit code not supplied", INC, np(G12=INC), mutate=lambda w: w.extras.pop("closeout")),
        # V4: boundaries of the reservation and the GPU-held ceiling
        Fixture("reservation_exactly_90_min", "boundary", "reservation held exactly 90.0 min", "PASS", {}, mutate=_set("deleted.deleted_observed_at", "2026-11-19T01:30:00Z")),
        Fixture("gpu_held_exactly_2700s", "boundary", "GPUs released exactly 2700.0 s after the launch (controller record); sampler upper edge 2699.5 s", "PASS", {}, clocks={"t_stop": 2535.5}),
        # V4: workload and recipe identity (G13)
        Fixture("identity_report_from_the_v1_policy", "negative", "a PASS identity report written by the V1/V2 checker and policy (the V3 defect)", F, np(G13=F), mutate=_both(_set("identity.checker", "e1_launch_identity_v2.py"), _set("identity.policy_version", "v1"))),
        Fixture("identity_trace_rule_not_exercised", "negative", "V4 report, verdict PASS, but the derived-trace rule was not exercised", F, np(G13=F), mutate=_set("identity.trace_identity.exercised", False)),
        Fixture("identity_trace_check_not_holding", "negative", "one trace-identity check does not hold although the report says PASS", F, np(G13=F), mutate=_set("identity.trace_identity.checks.7.ok", False)),
        Fixture("identity_trace_proof_incomplete", "negative", "19 of the 20 trace-identity checks are in the report", F, np(G13=F), mutate=lambda w: w.identity["trace_identity"]["checks"].pop()),
        Fixture("identity_report_inconclusive", "negative", "the identity checker could not read the launch record", INC, np(G9=INC, G13=INC), mutate=_set("identity.verdict", "INCONCLUSIVE")),
        # V4: the trace and config in the run directory (G14)
        Fixture("run_trace_is_the_v3_trace", "negative", "the run used the two-copy V3 trace (112 requests)", F, np(G14=F), mutate=_attr("trace_bytes", V3_TRACE_BYTES)),
        Fixture("run_trace_rows_reordered", "negative", "two rows of the third copy swapped in the run's trace", F, np(G14=F), mutate=lambda w: setattr(w, "trace_bytes", _swap_rows(w.trace_bytes, 130, 131))),
        Fixture("run_trace_missing", "negative", "no trace copy in the run directory", INC, np(G14=INC), mutate=_attr("trace_bytes", None)),
        Fixture("run_config_not_the_registered_one", "negative", "the run's profile config differs from the registered V4 config in one field", F, np(G14=F), mutate=lambda w: setattr(w, "config_bytes", w.config_bytes.replace(b'"stream": false', b'"stream": true'))),
        # V4: request, iteration, routing and rank coverage (G15)
        Fixture("coverage_cell_missing", "negative", "the b=8 cell of the fourth copy is absent from the completeness record", F, np(G15=F), mutate=lambda w: w.completeness["cells"].pop("b8_kv32768_w3")),
        Fixture("coverage_unplanned_cell", "negative", "a cell that is not in the plan appears", F, np(G15=F), mutate=_set("completeness.cells.b16_kv32768_w0", {"pure": 1000, "ramp": 60})),
        Fixture("coverage_pure_decode_below_floor", "negative", "one cell has 399 pure-decode iterations", F, np(G15=F), mutate=_set("completeness.cells.b48_kv32768_w2.pure", 399)),
        Fixture("coverage_one_request_short", "negative", "223 request timings", F, np(G15=F), mutate=_set("completeness.profile_result.request_timing_count", 223)),
        Fixture("coverage_three_ranks", "negative", "kernels from three ranks only", F, np(G15=F), mutate=_set("completeness.profile_result.parsed_device_ids", [0, 1, 2])),
        Fixture("coverage_routing_array_defect", "negative", "one routing array differs from its corpus segment", F, np(G15=F), mutate=_set("completeness.corpus.npy_mismatch", 1)),
        Fixture("coverage_record_missing", "negative", "the completeness checker wrote no record", INC, np(G15=INC), mutate=_attr("completeness", None)),
    ]
    return fx


def _swap_rows(data: bytes, i: int, j: int) -> bytes:
    lines = data.decode().split("\n")
    lines[i], lines[j] = lines[j], lines[i]
    return "\n".join(lines).encode()


def build(fx: Fixture, root: Path) -> dict:
    w = World(**fx.clocks)
    w.fill()
    if fx.mutate:
        fx.mutate(w)
    return write_world(w, root)


def cli_args(a: dict, out: Path) -> list[str]:
    cmd = [sys.executable, "-I", str(GATES), "--binary", str(a["binary"]), "--run", str(a["run"]), "--inputs", str(a["inputs"]), "--pin", str(a["pin"]), "--provenance", str(a["provenance"]), "--durable-stream", str(a["durable"]), "--scratch", str(a["scratch"]), "--out", str(out)]
    for flag, key in (("--importer", "importer"), ("--nsys", "nsys"), ("--identity", "identity"), ("--completeness", "completeness"), ("--postrun-storage", "post_storage"), ("--postrun-memory", "post_memory"), ("--reservation-deleted", "deleted")):
        if a.get(key) is not None:
            cmd += [flag, str(a[key])]
    for k, v in a["extras"].items():
        cmd += ["--extra", f"{k}={v}"]
    return cmd


def judge(report: dict, verdict: str, not_pass: dict) -> list[str]:
    got = {g["gate"]: g["status"] for g in report.get("gates", [])}
    problems = []
    if report.get("verdict") != verdict:
        problems.append(f"verdict {report.get('verdict')} (expected {verdict})")
    for gate, status in not_pass.items():
        if got.get(gate) != status:
            problems.append(f"{gate} is {got.get(gate)} (expected {status})")
    for gate, status in got.items():
        if status != "PASS" and gate not in not_pass:
            problems.append(f"{gate} is {status} (expected PASS)")
    for gate in ALL:
        if gate not in got and not (gate == "G8.offline_import" and "G8.offline_export" in got):
            problems.append(f"{gate} was not evaluated")
    return problems


def run_fixture(fx: Fixture, work: Path, out: Path, index: int) -> dict:
    root = work / f"{index:02d}_{fx.name}"
    root.mkdir(parents=True)
    a = build(fx, root)
    gates_out = out / f"{index:02d}_{fx.name}.json"
    p = subprocess.run(cli_args(a, gates_out), capture_output=True, text=True)
    (out / f"{index:02d}_{fx.name}.txt").write_text(p.stdout + (("\n--- stderr\n" + p.stderr) if p.stderr.strip() else ""))
    report = json.loads(gates_out.read_text()) if gates_out.exists() else {}
    problems = judge(report, fx.verdict, fx.not_pass)
    want_rc = {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3}[fx.verdict]
    if p.returncode != want_rc:
        problems.append(f"exit code {p.returncode} (expected {want_rc})")
    return {"index": index, "name": fx.name, "kind": fx.kind, "what": fx.what, "expected_verdict": fx.verdict, "expected_not_pass": fx.not_pass, "verdict": report.get("verdict"), "exit_code": p.returncode,
            "not_pass": {g["gate"]: g["status"] for g in report.get("gates", []) if g["status"] != "PASS"}, "met": not problems, "problems": problems}


def synthetic(out: Path, keep_work: bool = False, only: list[str] | None = None) -> int:
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="v4scale-rehearsal-"))
    rows = []
    try:
        for i, fx in enumerate(fixtures(), 1):
            if only and fx.name not in only:
                continue
            rows.append(run_fixture(fx, work, out, i))
    finally:
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)
    met = sum(r["met"] for r in rows)
    kinds = {k: sum(r["kind"] == k for r in rows) for k in ("positive", "boundary", "negative")}
    verdicts = {v: sum(r["verdict"] == v for r in rows) for v in ("PASS", "FAIL", "INCONCLUSIVE")}
    lines = [f"# V4 gate rehearsal, synthetic fixtures through {GATES.name} (CPU-only; not V4 evidence) {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
             f"# {len(rows)} fixtures ({kinds}); verdicts {verdicts}; expectations met {met} of {len(rows)}"]
    for r in rows:
        lines.append(f"{'MET    ' if r['met'] else 'NOT MET'} {r['index']:02d} {r['name']:38s} [{r['kind']:8s}] {r['verdict']:12s} not PASS: {sorted(r['not_pass'].items()) or '-'} | {r['what']}" + (f" | PROBLEMS {r['problems']}" if r["problems"] else ""))
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    (out / "SUMMARY.json").write_text(json.dumps({"fixtures": len(rows), "kinds": kinds, "verdicts": verdicts, "met": met, "rows": rows}, indent=1) + "\n")
    print("\n".join(lines))
    return 0 if rows and met == len(rows) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--keep-work", action="store_true")
    s.add_argument("--only", nargs="*")
    t = sub.add_parser("tools", help="stand-in importer / exporter and rehearsal inputs for the end-to-end rehearsal")
    t.add_argument("--out", required=True, type=Path)
    t.add_argument("--python", required=True, help="interpreter the stand-in importer runs as (the memory sampler identifies it by this executable)")
    e = sub.add_parser("expect")
    e.add_argument("--gates", required=True, type=Path)
    e.add_argument("--verdict", required=True, choices=["PASS", "FAIL", "INCONCLUSIVE"])
    e.add_argument("--expect-not-pass", default="", help="comma-separated GATE=STATUS")
    e.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "synthetic":
        return synthetic(args.out, args.keep_work, args.only)
    if args.cmd == "tools":
        _, nsys = fake_tools(args.out)
        for name, rc in (("QdstrmImporter", 0), ("QdstrmImporter-failing", 1)):
            imp = args.out / name  # `<importer> -i <stream> -o <report>`: reads the stream, takes a few seconds, writes a report (or fails)
            imp.write_text(f"#!{args.python}\nimport sys, time\nopen(sys.argv[2], 'rb').read(1 << 20)\ntime.sleep(4)\nif {rc}:\n    sys.stderr.write('stand-in importer: simulated import failure\\n'); sys.exit({rc})\nopen(sys.argv[4], 'wb').write(b'r' * 4096)\n")
            imp.chmod(0o755)
        (args.out / "inputs.json").write_text(json.dumps(rehearsal_inputs(), indent=1) + "\n")
        print(json.dumps({"importer": str(args.out / "QdstrmImporter"), "importer_failing": str(args.out / "QdstrmImporter-failing"), "nsys": str(nsys), "inputs": str(args.out / "inputs.json")}))
        return 0
    not_pass = dict(x.split("=") for x in args.expect_not_pass.split(",") if x)
    report = json.loads(args.gates.read_text())
    problems = judge(report, args.verdict, not_pass)
    got = {g["gate"]: g["status"] for g in report["gates"] if g["status"] != "PASS"}
    text = f"verdict {report['verdict']}; not PASS {got}; expectation {'MET' if not problems else 'NOT MET: ' + '; '.join(problems)}"
    print(text)
    if args.out:
        args.out.write_text(json.dumps({"verdict": report["verdict"], "not_pass": got, "expected_verdict": args.verdict, "expected_not_pass": not_pass, "met": not problems, "problems": problems}, indent=1) + "\n")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
