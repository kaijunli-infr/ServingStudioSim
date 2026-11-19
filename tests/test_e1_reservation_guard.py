"""Reservation age at the launch (first part of the V4 reservation budget) and the budget's arithmetic."""

from __future__ import annotations

# ruff: noqa: E501

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
E1 = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1"
SCRIPT = E1 / "checks" / "e1_reservation_guard.py"
spec = importlib.util.spec_from_file_location("e1_reservation_guard", SCRIPT)
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
T0 = 1_800_000_000.0


def _res(tmp: Path, text) -> Path:
    p = tmp / "reservation.json"
    p.write_text(text if isinstance(text, str) else json.dumps(text))
    return p


@pytest.mark.parametrize("age, rc", [(0.0, 0), (299.0, 0), (600.0, 0), (600.5, 1), (601.0, 1), (5400.0, 1), (-1.0, 1)])
def test_age_boundary(tmp_path, age, rc):
    p = _res(tmp_path, {"creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T0))})
    got, text = G.check(p, 600.0, T0 + age)
    assert got == rc and text.startswith("PASS" if rc == 0 else "FAIL")


@pytest.mark.parametrize("text", ["", "not json", {"pod": "x"}, {"creation_timestamp": "2026-11-19 00:00"}, {"creation_timestamp": None}])
def test_unreadable_record_is_inconclusive(tmp_path, text):
    assert G.check(_res(tmp_path, text), 600.0, T0)[0] == 3
    assert G.check(tmp_path / "missing.json", 600.0, T0)[0] == 3


def test_command_line(tmp_path):
    p = _res(tmp_path, {"creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 30))})
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), "age", "--reservation", str(p), "--max-age-s", "600"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("PASS")
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), "age", "--reservation", str(p), "--max-age-s", "10"], capture_output=True, text=True)
    assert r.returncode == 1 and r.stdout.startswith("FAIL")


def test_registered_budget_adds_up_to_the_reservation_ceiling():
    d = json.loads((E1 / "E1-V4-SCALE-INPUTS.json").read_text())
    r, c, lim = d["reservation"], d["ceilings"], d["limits"]
    assert r["supervisor_guard_s"] + r["closeout_and_deletion_budget_s"] == c["reservation_hard_min"] * 60 == 5400
    # latest launch + GPU-held ceiling + forced release cannot reach the guard: the guard can only fall into the CPU-only phase
    assert r["prelaunch_max_age_s"] + c["gpu_held_ceiling_s"] + lim["gpu_release_deadline_s"] < r["supervisor_guard_s"]
    # the observers and the supervisor outlive the guard, counted from the latest launch
    for k in ("memory_sampler", "storage_sampler", "supervisor"):
        assert lim["observer_max_s"][k] >= r["supervisor_guard_s"] - 0 and lim["observer_max_s"][k] + 0 >= c["reservation_hard_min"] * 60
    assert lim["observer_max_s"]["gpu_sampler"] >= c["gpu_held_ceiling_s"] + lim["gpu_release_deadline_s"] + d["phase_deadlines_s"]["session_close_after_server_sigterm"]
    assert r["early_release_after_verified_gpu_release"]["enabled"] is False
    driver = (E1 / "run_e1_v4scale.sh").read_text()
    assert "RSS_CEIL=128; TMP_CEIL=15; PERS_CEIL=18; RES_AGE=600; RES_GUARD=4680\n" in driver and driver.count("--max-s 5400") == 3 and "e1_reservation_release" not in driver
    for name in ("run_e1_v4scale.sh", "postrun_e1_v4scale.sh", "closeout_e1_v4scale.sh", "archive_e1_v4scale.sh", "monitor_e1_v4scale.sh", "preflight_e1_v4scale.sh"):
        text = (E1 / name).read_text()
        assert "checks/e1_reservation_release.py" not in text and "kubectl delete" not in text and "delete pod" not in text, name  # no V4 script deletes the reservation or calls the early-release check
