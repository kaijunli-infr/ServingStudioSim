"""V4 preflight host-memory minimum: 488 GiB = importer ceiling 128 + runtime floor 64 + 296 for the run. Exact boundary in kB;
the runtime floor, the importer ceiling and the breach responses are not touched by the change."""

from __future__ import annotations

# ruff: noqa: E501

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
E1 = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1"
SCRIPT = E1 / "checks" / "e1_preflight_memory.py"
FIX = REPO / "tests" / "fixtures" / "e1_v4_scale"
INPUTS = E1 / "E1-V4-SCALE-INPUTS.json"
GIB = 1048576  # kB


def _load():
    spec = importlib.util.spec_from_file_location("e1_preflight_memory", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()
D = json.loads(INPUTS.read_text())


def meminfo(kb: int) -> str:
    return f"MemTotal:       1004636800 kB\nMemFree:        1000 kB\nMemAvailable:   {kb} kB\nSwapTotal:             0 kB\n"


@pytest.mark.parametrize(("fixture", "kb", "rc"), [
    ("meminfo_487GiB.txt", 487 * GIB, 1),
    ("meminfo_488GiB_minus_1kB.txt", 488 * GIB - 1, 1),
    ("meminfo_488GiB.txt", 488 * GIB, 0),  # at the minimum counts as within
    ("meminfo_488GiB_plus_1kB.txt", 488 * GIB + 1, 0),  # immediately above
    ("meminfo_489GiB.txt", 489 * GIB, 0),
])
def test_boundary_fixtures(fixture, kb, rc):
    text = (FIX / fixture).read_text()
    assert M.mem_available_kb(text) == kb
    got, lines = M.check(text, D)
    assert got == rc and len(lines) == 2 and lines[0].startswith("PASS  " if rc == 0 else "FAIL  ") and lines[1].startswith("PASS  ")
    assert "488 GiB" in lines[0] and "importer ceiling 128 + runtime floor 64 + allowance for the run 296" in lines[0]
    p = subprocess.run([sys.executable, "-I", str(SCRIPT), "--meminfo", str(FIX / fixture), "--inputs", str(INPUTS)], capture_output=True, text=True)
    assert p.returncode == rc and p.stdout.splitlines() == lines


def test_the_old_minimum_would_have_accepted_what_is_now_refused():
    for kb in (448 * GIB, 477 * GIB, 487 * GIB):
        assert M.check(meminfo(kb), D)[0] == 1


def test_registered_values_and_what_did_not_change():
    c, mb = D["ceilings"], D["memory_budget"]
    assert c["host_mem_available_preflight_min_gb"] == 488 == c["importer_rss_ceiling_gb"] + c["host_mem_available_floor_gb"] + mb["server_allowance_gib"]
    assert c["host_mem_available_floor_gb"] == 64 and c["importer_rss_ceiling_gb"] == 128 and mb["server_allowance_gib"] == 296
    driver = (E1 / "run_e1_v4scale.sh").read_text()
    assert "--min-available-gb 64 --max-importer-rss-gb $RSS_CEIL --breach-file $RUN/$PASS/MEMORY-BREACH" in driver and "RSS_CEIL=128;" in driver
    v3 = json.loads((E1 / "E1-V3-SCALE-INPUTS.json").read_text())["ceilings"]
    assert v3["host_mem_available_preflight_min_gb"] == 448  # the frozen V3 record is not rewritten
    for k in ("host_mem_available_floor_gb", "importer_rss_ceiling_gb"):
        assert v3[k] == c[k]
    pre = (E1 / "preflight_e1_v4scale.sh").read_text()
    assert "checks/e1_preflight_memory.py --meminfo /proc/meminfo --inputs $INP" in pre and "448" not in pre


def test_inconsistent_registration_fails_and_unreadable_is_inconclusive():
    bad = json.loads(json.dumps(D))
    bad["ceilings"]["host_mem_available_preflight_min_gb"] = 448
    rc, lines = M.check(meminfo(900 * GIB), bad)
    assert rc == 1 and "is not" in lines[0]
    for broken in ({}, {"ceilings": {}}, {"ceilings": D["ceilings"]}, {"ceilings": {**D["ceilings"], "host_mem_available_preflight_min_gb": True}, "memory_budget": D["memory_budget"]}):
        assert M.check(meminfo(900 * GIB), broken)[0] == 3
    assert M.check("MemTotal: 5 kB\n", D)[0] == 3 and M.check("MemAvailable: lots kB\n", D)[0] == 3 and M.check("MemAvailable: 5 MB\n", D)[0] == 3


@pytest.mark.parametrize(("cg", "rc", "word"), [("max", 0, "PASS"), (str(488 * 1024**3), 0, "PASS"), (str(488 * 1024**3 - 1), 1, "FAIL"), (str(64 * 1024**3), 1, "FAIL"), ("unknown", 3, "INCONCLUSIVE")])
def test_cgroup_limit(cg, rc, word):
    got, lines = M.check(meminfo(600 * GIB), D, cg)
    assert got == rc and lines[1].startswith(word)


def test_a_host_failure_is_not_hidden_by_a_cgroup_result():
    assert M.check(meminfo(100 * GIB), D, "unknown")[0] == 1


def test_missing_files_on_the_command_line(tmp_path):
    p = subprocess.run([sys.executable, "-I", str(SCRIPT), "--meminfo", str(tmp_path / "none"), "--inputs", str(INPUTS)], capture_output=True, text=True)
    assert p.returncode == 3 and p.stdout.startswith("INCONCLUSIVE")
