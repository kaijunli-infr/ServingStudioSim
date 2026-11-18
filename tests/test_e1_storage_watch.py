"""Storage sampler: per-group bytes with each inode once, peaks, ceilings, breach file, measured-versus-unmeasured summary."""

from __future__ import annotations

# ruff: noqa: E501

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/checks/e1_storage_watch.py"


def _load():
    spec = importlib.util.spec_from_file_location("e1_storage_watch", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(tmp: Path) -> tuple[Path, Path]:
    t, run = tmp / "tmpfs", tmp / "run"
    (t / "a.raw-preserve").mkdir(parents=True)
    (t / "nsys-report-1.qdstrm").write_bytes(b"s" * 1000)
    os.link(t / "nsys-report-1.qdstrm", t / "a.raw-preserve" / "a.qdstrm")  # the preserved stream is a hard link
    (t / "v3scale-recovery-x").mkdir()
    (t / "v3scale-recovery-x" / "recovery.sqlite").write_bytes(b"q" * 5000)
    (t / "unrelated.bin").write_bytes(b"u" * 9999)
    (run / "pass" / "profile").mkdir(parents=True)
    (run / "pass" / "profile" / "x.sqlite").write_bytes(b"p" * 3000)
    (run / "pass" / "run.log").write_bytes(b"l" * 10)
    os.symlink(run / "pass" / "run.log", run / "link.log")
    return t, run


def _groups(t: Path, run: Path) -> list[str]:
    return ["--group", f"temporary={t}/nsys-report-*.qdstrm,{t}/*.raw-preserve,{t}/v3scale-recovery-*", "--group", f"persistent={run}"]


def test_each_inode_is_counted_once_and_unmatched_files_are_ignored(tmp_path):
    mod = _load()
    t, run = _tree(tmp_path)
    m = mod.measure([f"{t}/nsys-report-*.qdstrm", f"{t}/*.raw-preserve", f"{t}/v3scale-recovery-*"])
    assert m == {"bytes": 6000, "files": 2, "roots": 3}
    assert mod.measure([str(run)]) == {"bytes": 3010, "files": 2, "roots": 1}  # the symlink is not followed
    assert mod.measure([f"{t}/absent-*"]) == {"bytes": 0, "files": 0, "roots": 0}


def test_samples_summary_and_peak(tmp_path):
    mod = _load()
    t, run = _tree(tmp_path)
    out, summary = tmp_path / "st.jsonl", tmp_path / "st.json"
    rc = mod.main(["--out", str(out), "--summary", str(summary), "--interval", "0.01", "--max-s", "0.06", *_groups(t, run)])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    s = json.loads(summary.read_text())
    assert rc == 0 and len(rows) >= 3 and all(r["host"] and r["groups"]["temporary"]["bytes"] == 6000 and r["groups"]["persistent"]["bytes"] == 3010 for r in rows)
    assert rows[0]["groups"]["temporary"]["free_bytes"] > 0
    assert s["samples"] == len(rows) and s["complete"] is True and s["max_gap_s"] is not None and s["breaches"] == []
    assert s["groups"]["temporary"]["peak_bytes"] == 6000 and s["groups"]["persistent"]["peak_bytes"] == 3010 and s["groups"]["temporary"]["peak_at"] == rows[0]["at"]


def test_peak_is_kept_after_the_scratch_is_removed(tmp_path):
    mod = _load()
    t, run = _tree(tmp_path)
    out, summary = tmp_path / "st.jsonl", tmp_path / "st.json"
    assert mod.main(["--out", str(out), "--summary", str(summary), "--max-s", "0", *_groups(t, run)]) == 0
    (t / "v3scale-recovery-x" / "recovery.sqlite").unlink()
    assert mod.main(["--out", str(out), "--summary", str(summary), "--max-s", "0", *_groups(t, run)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["groups"]["temporary"]["bytes"] for r in rows] == [6000, 1000]  # appended: the earlier, larger sample is still in the file


@pytest.mark.parametrize("ceiling_gb, breached", [(6001 / 1e9, False), (5999 / 1e9, True)])
def test_ceiling_boundary_and_breach_file(tmp_path, ceiling_gb, breached):
    mod = _load()
    t, run = _tree(tmp_path)
    breach = tmp_path / "STORAGE-BREACH"
    out, summary = tmp_path / "st.jsonl", tmp_path / "st.json"
    rc = mod.main(["--out", str(out), "--summary", str(summary), "--interval", "0.01", "--max-s", "0.05", *_groups(t, run), "--ceiling-gb", f"temporary={ceiling_gb!r}", "--breach-file", str(breach)])
    s = json.loads(summary.read_text())
    assert rc == (4 if breached else 0) and breach.exists() == breached and len(s["breaches"]) == (1 if breached else 0)
    if breached:
        assert s["breaches"][0]["kind"] == "temporary_storage_above_ceiling" and len(breach.read_text().splitlines()) == 1
    assert (t / "v3scale-recovery-x" / "recovery.sqlite").exists()  # the sampler removes nothing


def test_stop_file_ends_after_one_sample(tmp_path):
    mod = _load()
    t, run = _tree(tmp_path)
    stop = tmp_path / "stop"
    stop.write_text("")
    out, summary = tmp_path / "st.jsonl", tmp_path / "st.json"
    assert mod.main(["--out", str(out), "--summary", str(summary), "--max-s", "600", "--stop-file", str(stop), *_groups(t, run)]) == 0
    assert len(out.read_text().splitlines()) == 1


def test_bad_arguments(tmp_path):
    mod = _load()
    base = ["--out", str(tmp_path / "o"), "--summary", str(tmp_path / "s")]
    for extra in (["--group", "nameonly"], ["--group", "a=/x", "--ceiling-gb", "b=1"]):
        with pytest.raises(SystemExit):
            mod.main(base + extra)
