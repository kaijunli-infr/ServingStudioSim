"""Run root plus archive against the 38 GB ceiling (sparse files: sizes without disk use)."""

from __future__ import annotations

# ruff: noqa: E501

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/checks/e1_archive_size.py"
spec = importlib.util.spec_from_file_location("e1_archive_size", SCRIPT)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def _sparse(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.truncate(size)


@pytest.mark.parametrize("run_b, arc_b, rc", [(18 * 10**9, 20 * 10**9, 0), (18 * 10**9, 20 * 10**9 + 1, 1), (12_600_000_000, 13_900_000_000, 0), (19 * 10**9, 19 * 10**9 + 10**7, 1)])
def test_ceiling_boundary(tmp_path, run_b, arc_b, rc):
    _sparse(tmp_path / "run" / "a.bin", run_b)
    _sparse(tmp_path / "arc" / "sub" / "b.bin", arc_b)
    got, text = A.check(tmp_path / "run", tmp_path / "arc", 38.0)
    assert got == rc and "ceiling 38" in text


def test_hard_links_count_once_and_missing_directories_are_inconclusive(tmp_path):
    _sparse(tmp_path / "run" / "stream.qdstrm", 10**9)
    os.link(tmp_path / "run" / "stream.qdstrm", tmp_path / "run" / "stream.link")
    (tmp_path / "arc").mkdir()
    assert A.size(tmp_path / "run") == 10**9
    assert A.check(tmp_path / "run", tmp_path / "nope", 38.0)[0] == 3


def test_command_line(tmp_path):
    _sparse(tmp_path / "run" / "a.bin", 39 * 10**9)
    (tmp_path / "arc").mkdir()
    cmd = [sys.executable, "-I", str(SCRIPT), "--run", str(tmp_path / "run"), "--archive", str(tmp_path / "arc"), "--ceiling-gb"]
    assert subprocess.run([*cmd, "38"], capture_output=True).returncode == 1
    assert subprocess.run([*cmd, "39"], capture_output=True).returncode == 0
