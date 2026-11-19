"""V4 reservation budget: the launch rule (remaining time to the guard >= run budget), the deletion deadline, the arithmetic of
the registered budget, and the driver's own lines."""

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
STAMP = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T0))


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


GUARD, BUDGET, LATEST = 4680.0, 4210.0, 470.0


def test_registered_budget_adds_up_to_the_reservation_ceiling():
    d = json.loads((E1 / "E1-V4-SCALE-INPUTS.json").read_text())
    r, c, lim = d["reservation"], d["ceilings"], d["limits"]
    b = r["run_budget_s"]
    assert r["hard_ceiling_s"] == c["reservation_hard_min"] * 60 == 5400 and r["supervisor_guard_s"] == GUARD
    assert r["supervisor_guard_s"] + r["closeout_and_deletion_budget_s"] == r["hard_ceiling_s"] and sum(r["closeout_and_deletion_parts_s"].values()) == r["closeout_and_deletion_budget_s"] == 720
    # every part of the run budget is a registered limit or the registered stress prediction, and each is counted once
    assert b == {"eligibility_check_to_launcher_start": 20, "gpu_held_ceiling": 2700, "gpu_release_after_ceiling": 90, "cpu_only_after_release": 1400}
    assert b["gpu_held_ceiling"] == c["gpu_held_ceiling_s"] == d["lifecycle"]["gpu_held_ceiling_s"] and b["gpu_release_after_ceiling"] == lim["gpu_release_deadline_s"]
    assert sum(r["cpu_only_after_release_s"].values()) == b["cpu_only_after_release"] and r["cpu_only_after_release_s"]["export_and_parse_stress_prediction"] == d["predictions"]["export_and_parse_after_release_s"][2]
    assert r["run_budget_total_s"] == sum(b.values()) == BUDGET and r["launch_max_age_s"] == r["supervisor_guard_s"] - r["run_budget_total_s"] == LATEST
    # the required worst-case path fits with an immediate launch, and a launch window exists
    assert 0 + r["run_budget_total_s"] <= r["supervisor_guard_s"] and r["launch_max_age_s"] > r["measured_v3"]["age_at_launcher_start_s"]
    # the latest launch, the ceiling and the release cannot reach the guard: the guard can only fall into the CPU-only phase
    assert r["launch_max_age_s"] + b["eligibility_check_to_launcher_start"] + b["gpu_held_ceiling"] + b["gpu_release_after_ceiling"] < r["supervisor_guard_s"]
    # the observers and the supervisor outlive the guard and the ceiling, counted from the earliest launch
    for k in ("memory_sampler", "storage_sampler", "supervisor"):
        assert lim["observer_max_s"][k] >= r["hard_ceiling_s"]
    assert lim["observer_max_s"]["gpu_sampler"] >= c["gpu_held_ceiling_s"] + lim["gpu_release_deadline_s"] + d["phase_deadlines_s"]["session_close_after_server_sigterm"]
    assert r["early_release_after_verified_gpu_release"]["enabled"] is False
    for name in ("run_e1_v4scale.sh", "postrun_e1_v4scale.sh", "closeout_e1_v4scale.sh", "archive_e1_v4scale.sh", "monitor_e1_v4scale.sh", "preflight_e1_v4scale.sh"):
        text = (E1 / name).read_text()
        assert "checks/e1_reservation_release.py" not in text and "kubectl delete" not in text and "delete pod" not in text, name  # no V4 script deletes the reservation or calls the early-release check


def test_what_the_correction_did_not_change():
    d = json.loads((E1 / "E1-V4-SCALE-INPUTS.json").read_text())
    c, e, p = d["ceilings"], d["expected"], d["phase_deadlines_s"]
    assert d["trace"]["requests"] == 224 and len(d["trace"]["waves"]) == 8 and d["trace"]["copies"] == 4 and e["iterations"] == 8640 and e["kernel_rows"] == 49003328
    assert (p["flush_stop_to_importing"], p["import_importer_first_seen_to_gone"], p["stop_to_report_written"]) == (300, 300, 600) and c["gpu_held_ceiling_s"] == 2700
    assert (c["temporary_storage_gb"], c["persistent_storage_gb"], c["persistent_storage_with_archive_gb"]) == (15, 18, 38) and d["identity"]["policy_version"] == "v4scale-1"
    cfg = json.loads((E1 / "configs_v4scale" / "e1_v4_scale" / "profile.json").read_text())["nsys"]
    assert cfg["gpu_held_ceiling_s"] == 2700.0 and cfg["finalize_hard_deadline_s"] == 600.0 and cfg["finalize_soft_deadline_s"] == 300.0 and cfg["finalize_session_close_timeout_s"] == 300.0
    assert d["trace"]["sha256"] == "63d79100e5f482854fcc868bda859f206c611e194c18efec40749c8a31d76de1"


@pytest.mark.parametrize(("age", "rc"), [(0.0, 0), (316.0, 0), (LATEST - 1, 0), (LATEST, 0), (LATEST + 0.001, 1), (LATEST + 1, 1), (600.0, 1), (GUARD, 1), (GUARD + 100, 1), (-1.0, 1)])
def test_launch_rule_boundaries(tmp_path, age, rc):
    got, text = G.eligible(_res(tmp_path, {"creation_timestamp": STAMP}), GUARD, BUDGET, T0 + age)
    assert got == rc and text.startswith("PASS  launch eligibility" if rc == 0 else "FAIL  launch eligibility")
    if rc:
        assert "negative" in text or "REFUSED, nothing is started" in text
    else:
        assert f"latest permitted age {LATEST:.0f} s" in text


def test_launch_rule_with_an_immediate_launch_has_the_whole_window_to_spare(tmp_path):
    rc, text = G.eligible(_res(tmp_path, {"creation_timestamp": STAMP}), GUARD, BUDGET, T0)
    assert rc == 0 and "4680.0 s remain" in text and "470.0 s to spare" in text


@pytest.mark.parametrize(("guard", "budget"), [(4680.0, 4680.1), (4680.0, 0.0), (0.0, 10.0), (100.0, 5000.0)])
def test_no_launch_window_is_a_refusal_not_a_silent_extension(tmp_path, guard, budget):
    rc, text = G.eligible(_res(tmp_path, {"creation_timestamp": STAMP}), guard, budget, T0 + 1)
    assert rc == 1 and "no launch window exists" in text


@pytest.mark.parametrize("text", ["", "{", "{}", json.dumps({"creation_timestamp": "yesterday"}), json.dumps({"creation_timestamp": None})])
def test_launch_rule_unreadable_record(tmp_path, text):
    assert G.eligible(_res(tmp_path, text), GUARD, BUDGET, T0)[0] == 3
    assert G.deadline(_res(tmp_path, text), 5400.0, T0)[0] == 3


@pytest.mark.parametrize(("age", "rc", "words"), [(4680.0, 0, "720 s left"), (5399.0, 0, "1 s left"), (5400.0, 0, "0 s left"), (5401.0, 1, "passed 1 s ago"), (6000.0, 1, "passed 600 s ago")])
def test_deletion_deadline(tmp_path, age, rc, words):
    got, text = G.deadline(_res(tmp_path, {"creation_timestamp": STAMP}), 5400.0, T0 + age)
    assert got == rc and words in text and time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T0 + 5400)) in text


def _driver_lines() -> tuple[str, list[str]]:
    lines = (E1 / "run_e1_v4scale.sh").read_text().splitlines()
    name = next(ln for ln in lines if ln.startswith("NAME=dsv41_flash_1017_gb300_tp4_$PASS;"))
    i = next(k for k, ln in enumerate(lines) if ln.startswith("python3 -I $E1/checks/e1_reservation_guard.py eligible "))
    return name, lines[i:i + 2]


def _run_driver_rule(tmp_path: Path, age: float) -> subprocess.CompletedProcess:
    """The driver's own constants line and its two launch-rule lines, executed by bash against a reservation record of the given age."""
    name, rule = _driver_lines()
    run = tmp_path / "root"
    (run / "p").mkdir(parents=True, exist_ok=True)
    (run / "p" / "reservation.json").write_text(json.dumps({"creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - age))}))
    script = "\n".join(["set -u", f"E1={E1}", f"RUN={run}", "PASS=p", name, *rule, "echo SENTINEL-OBSERVERS-AND-LAUNCHER-WOULD-START"])
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_the_drivers_own_lines_permit_late_and_refuse_too_late(tmp_path):
    name, rule = _driver_lines()
    assert name.endswith("RSS_CEIL=128; TMP_CEIL=15; PERS_CEIL=18; RES_GUARD=4680; RES_RUN=4210; GPU_CEIL=2700")
    assert "--guard-s $RES_GUARD --run-budget-s $RES_RUN" in rule[0] and "exit 7" in rule[1] and '"${PIPESTATUS[0]}" = 0' in rule[1]
    ok = _run_driver_rule(tmp_path / "a", LATEST - 8)  # a few seconds before the latest permitted instant
    assert ok.returncode == 0 and "PASS  launch eligibility" in ok.stdout and "SENTINEL-OBSERVERS-AND-LAUNCHER-WOULD-START" in ok.stdout
    late = _run_driver_rule(tmp_path / "b", LATEST + 3)  # immediately beyond it
    assert late.returncode == 7 and "FAIL  launch eligibility" in late.stdout and "LAUNCH REFUSED" in late.stdout and "SENTINEL" not in late.stdout
    log = (tmp_path / "b" / "root" / "p" / "run.log").read_text()
    assert "REFUSED, nothing is started" in log and "LAUNCH REFUSED" in log
    fresh = _run_driver_rule(tmp_path / "c", 10)
    (tmp_path / "c" / "root" / "p" / "reservation.json").write_text("{")
    broken = subprocess.run(["bash", "-c", "\n".join(["set -u", f"E1={E1}", f"RUN={tmp_path / 'c' / 'root'}", "PASS=p", name, *rule, "echo SENTINEL"])], capture_output=True, text=True)
    assert fresh.returncode == 0 and broken.returncode == 7 and "SENTINEL" not in broken.stdout  # an unreadable record refuses as well


def test_the_rule_runs_before_anything_is_started_and_real_mode_cannot_override_it():
    lines = (E1 / "run_e1_v4scale.sh").read_text().splitlines()
    rule = next(k for k, ln in enumerate(lines) if ln.startswith("python3 -I $E1/checks/e1_reservation_guard.py eligible "))
    started = [k for k, ln in enumerate(lines) if ln.startswith(("spawn ", "python3 -I $E1/checks/e1_pids.py spawn ")) or "python -m launcher alignment profile" in ln or "e1_v4scale_mock_launcher.py launch" in ln]
    assert len(started) == 7 and rule + 1 < min(started)  # four observers, the supervisor, the launcher line and its rehearsal stand-in
    begin, end = lines.index('if [ "$MODE" = rehearsal ]; then'), None
    end = next(k for k in range(begin, len(lines)) if lines[k] == "fi")
    outside = [ln for k, ln in enumerate(lines) if "E1_V4S_REHEARSAL_" in ln and not begin < k < end]
    assert outside == [] and sum("E1_V4S_REHEARSAL_RUN_BUDGET_S" in ln for ln in lines[begin:end]) == 1
    text = "\n".join(lines)
    assert "--gpu-held-ceiling-s $GPU_CEIL --ack-s 10 --stale-s 60 --cleanup-grace 30 --interval 1 --max-s 5400" in text and "--reservation-guard-s $RES_GUARD" in text and text.count("--max-s 5400") == 3
    assert "e1_reservation_release" not in text and "RES_AGE" not in text


def test_command_line_eligible_and_deadline(tmp_path):
    p = _res(tmp_path, {"creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 30))})
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), "eligible", "--reservation", str(p), "--guard-s", "4680", "--run-budget-s", "4210"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("PASS  launch eligibility")
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), "eligible", "--reservation", str(p), "--guard-s", "4680", "--run-budget-s", "4660"], capture_output=True, text=True)
    assert r.returncode == 1 and "REFUSED" in r.stdout
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), "deadline", "--reservation", str(p), "--ceiling-s", "5400"], capture_output=True, text=True)
    assert r.returncode == 0 and "delete the reservation by" in r.stdout
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), "deadline", "--reservation", str(p), "--ceiling-s", "10"], capture_output=True, text=True)
    assert r.returncode == 1 and "ceiling passed" in r.stdout
