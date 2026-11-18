"""Host-memory sampler: MemAvailable and the identified importer's resident memory, bounded policy, breach file, no action."""

from __future__ import annotations

# ruff: noqa: E501

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/checks/e1_mem_watch.py"
EXE = "/opt/nvidia/nsight-systems/2025.6.3/host-linux-armv8/QdstrmImporter"
STREAM = "/tmp/nsys-report-a2d6.qdstrm"
KB = 1024 * 1024


def _load():
    spec = importlib.util.spec_from_file_location("e1_mem_watch", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _host(tmp: Path, available_gb: float = 900.0) -> tuple[Path, Path]:
    proc = tmp / "proc"
    proc.mkdir(exist_ok=True)
    mi = tmp / "meminfo"
    mi.write_text(f"MemTotal:       {1004 * KB} kB\nMemFree:        1 kB\nMemAvailable:   {int(available_gb * KB)} kB\n")
    return proc, mi


def _importer(proc: Path, pid: int, rss_gb: float, *, stream: str = STREAM, exe: str = EXE, hwm_gb: float | None = None, report: str = "/run/out.nsys-rep"):
    d = proc / str(pid)
    (d / "fd").mkdir(parents=True)
    (d / "stat").write_text(f"{pid} (QdstrmImporter) S 1 {pid} {pid} 0 -1 0 0 0 0 0 5 5 0 0 20 0 1 0 900 0 0\n")
    (d / "cmdline").write_text("\0".join([exe, "--input-file", stream, "--output-file", report]) + "\0")
    (d / "io").write_text("rchar: 1\nwchar: 1\n")
    (d / "status").write_text(f"Name:\tQdstrmImporter\nVmHWM:\t{int((hwm_gb if hwm_gb is not None else rss_gb) * KB)} kB\nVmRSS:\t{int(rss_gb * KB)} kB\n")
    os.symlink(exe, d / "exe")


def _run(mod, tmp: Path, *extra: str, proc: Path, mi: Path, importer: bool = True) -> tuple[int, list, dict]:
    out, summary = tmp / "mem.jsonl", tmp / "mem.json"
    args = ["--out", str(out), "--summary", str(summary), "--interval", "0.01", "--max-s", "0", "--meminfo", str(mi), "--proc", str(proc), "--uid", str(os.getuid()), *extra]
    if importer:
        args += ["--importer-exe", EXE, "--stream", STREAM, "--report", "/run/out.nsys-rep"]
    rc = mod.main(args)
    return rc, [json.loads(line) for line in out.read_text().splitlines()], json.loads(summary.read_text())


def test_sample_records_host_memory_and_the_identified_importer(tmp_path):
    mod = _load()
    proc, mi = _host(tmp_path)
    _importer(proc, 5000, 17.0, hwm_gb=17.5)
    rc, rows, s = _run(mod, tmp_path, proc=proc, mi=mi)
    assert rc == 0 and len(rows) == 1 and rows[0]["host"] and rows[0]["mem_available_kb"] == 900 * KB and rows[0]["mem_total_kb"] == 1004 * KB
    assert rows[0]["importer"]["pid"] == 5000 and rows[0]["importer"]["rss_kb"] == 17 * KB and rows[0]["importer"]["hwm_kb"] == int(17.5 * KB) and rows[0]["candidates"] == 1
    assert s["importer_peak_rss_kb"] == int(17.5 * KB) and s["importer_samples"] == 1 and s["min_available_kb"] == 900 * KB and s["breaches"] == [] and s["complete"] is True
    assert s["policy"] == {"min_available_gb": 64.0, "max_importer_rss_gb": 256.0, "action": "record + breach file only; the sampler signals nothing"}


def test_other_sessions_importer_and_ambiguity_are_not_measured_as_ours(tmp_path):
    mod = _load()
    proc, mi = _host(tmp_path)
    _importer(proc, 5001, 40.0, stream="/tmp/other.qdstrm", report="/other/out.nsys-rep")
    _importer(proc, 5002, 40.0, exe="/home/x/QdstrmImporter")
    rc, rows, s = _run(mod, tmp_path, proc=proc, mi=mi)
    assert rc == 0 and rows[0]["importer"] is None and rows[0]["candidates"] == 0 and s["importer_samples"] == 0 and s["importer_peak_rss_kb"] is None
    _importer(proc, 5003, 1.0)
    _importer(proc, 5004, 1.0)
    (tmp_path / "mem.jsonl").unlink()
    rc, rows, s = _run(mod, tmp_path, proc=proc, mi=mi)
    assert rows[0]["candidates"] == 2 and rows[0]["importer"] is None and s["ambiguous_samples"] == 1


@pytest.mark.parametrize("available, rss, kinds", [(64.0, 256.0, []), (63.9, 10.0, ["host_mem_available_below_floor"]), (900.0, 256.1, ["importer_rss_above_ceiling"]), (10.0, 300.0, ["host_mem_available_below_floor", "importer_rss_above_ceiling"])])
def test_policy_boundaries_and_breach_file(tmp_path, available, rss, kinds):
    mod = _load()
    proc, mi = _host(tmp_path, available)
    _importer(proc, 5000, rss)
    breach = tmp_path / "MEMORY-BREACH"
    rc, rows, s = _run(mod, tmp_path, "--breach-file", str(breach), proc=proc, mi=mi)
    assert rc == (4 if kinds else 0) and [b["kind"] for b in s["breaches"]] == kinds and rows[0].get("breach", []) == kinds
    assert breach.exists() == bool(kinds)
    if kinds:
        assert [json.loads(line)["kind"] for line in breach.read_text().splitlines()] == kinds
    assert (proc / "5000").exists()  # nothing was signalled or removed


def test_a_breach_is_recorded_once_per_kind_over_many_samples(tmp_path):
    mod = _load()
    proc, mi = _host(tmp_path, 10.0)
    breach = tmp_path / "MEMORY-BREACH"
    out, summary = tmp_path / "mem.jsonl", tmp_path / "mem.json"
    rc = mod.main(["--out", str(out), "--summary", str(summary), "--interval", "0.01", "--max-s", "0.08", "--meminfo", str(mi), "--proc", str(proc), "--breach-file", str(breach)])
    rows = out.read_text().splitlines()
    s = json.loads(summary.read_text())
    assert rc == 4 and len(rows) >= 3 and len(breach.read_text().splitlines()) == 1 and len(s["breaches"]) == 1
    assert s["samples"] == len(rows) and s["max_gap_s"] is not None and s["importer_samples"] == 0


def test_stop_file_and_missing_meminfo(tmp_path):
    mod = _load()
    proc, _ = _host(tmp_path)
    stop = tmp_path / "stop"
    stop.write_text("")
    out, summary = tmp_path / "mem.jsonl", tmp_path / "mem.json"
    rc = mod.main(["--out", str(out), "--summary", str(summary), "--max-s", "600", "--stop-file", str(stop), "--meminfo", str(tmp_path / "absent"), "--proc", str(proc)])
    s = json.loads(summary.read_text())
    assert rc == 0 and s["samples"] == 1 and s["samples_without_meminfo"] == 1 and s["min_available_kb"] is None and s["complete"] is True


def test_importer_arguments_are_required_together(tmp_path):
    mod = _load()
    with pytest.raises(SystemExit):
        mod.main(["--out", str(tmp_path / "o"), "--summary", str(tmp_path / "s"), "--importer-exe", EXE])
