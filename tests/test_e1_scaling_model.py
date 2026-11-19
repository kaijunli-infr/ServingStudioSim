"""The sizing arithmetic for the next E1 capture: every hypothesis passes through the measured V2 and V3 points, the reported
range brackets them, and the tables in the repository are the script's output."""

from __future__ import annotations

# ruff: noqa: E501

import importlib.util
import json
from pathlib import Path

import pytest

FEAS = Path(__file__).resolve().parents[1] / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/feasibility"
spec = importlib.util.spec_from_file_location("e1_scaling_model", FEAS / "e1_scaling_model.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


@pytest.mark.parametrize("key", list(M.MEASURED))
def test_every_hypothesis_reproduces_the_v3_point_and_power_and_line_the_v2_point(key):
    y2, y3 = M.MEASURED[key]
    h3, h2 = M.hypotheses(key, M.N3), M.hypotheses(key, M.N2)
    assert all(h3[h] == pytest.approx(y3) for h in ("power", "line", "prop", "stress"))
    assert h2["power"] == pytest.approx(y2) and h2["line"] == pytest.approx(y2)


def test_v3_option_reproduces_the_measured_run():
    s = M.size(M.N3, 139.0)
    assert s["stop_to_report_s"][0] == pytest.approx(166.2) and s["stop_to_report_s"][1] == pytest.approx(166.2)
    assert s["gpu_held_s"][0] <= M.V3["gpu_held_s"] <= s["gpu_held_s"][1]
    assert s["reservation_min"][0] <= M.V3["reservation_min"] <= s["reservation_min"][1]
    assert s["mem_available_min_gib"][0] == pytest.approx(547.2)
    assert s["temp_peak_gb"][1] == pytest.approx(M.V3["temp_peak_gb"] - M.V3["retained_streams_gb_then"] + M.V3["retained_streams_gb_now"])


def test_ranges_are_ordered_and_spread_grows_with_distance():
    mid, full = M.size(2 * M.N3, 299.0), M.size(216e6, 896.0)
    for key in M.MEASURED:
        assert mid[key][0] <= mid[key][1] <= mid[key][2] and full[key][0] <= full[key][1] <= full[key][2]
        assert full[key][1] / full[key][0] >= mid[key][1] / mid[key][0]


def test_largest_rows_within_a_limit_is_consistent():
    for h in ("power", "line", "prop", "stress"):
        n = M.largest_rows_within("flush_s", 300.0, h)
        assert M.hypotheses("flush_s", n)[h] == pytest.approx(300.0, rel=1e-6) and n > 2 * M.N3


def test_committed_tables_are_the_script_output():
    data, md = M.render()
    assert (FEAS / "E1-SCALING-MODEL.md").read_text() == md
    assert json.loads((FEAS / "E1-SCALING-MODEL.json").read_text()) == json.loads(json.dumps(data))


def test_no_v3_limit_was_changed_by_the_analysis():
    assert M.V3_LIMITS == {"flush_s": 300.0, "import_s": 300.0, "stop_to_report_s": 600.0, "gpu_held_s": 2700.0, "reservation_min": 90.0, "importer_rss_gib": 128.0,
                           "mem_available_floor_gib": 64.0, "temp_gb": 10.0, "run_root_gb": 8.0, "run_root_plus_archive_gb": 16.0, "server_term_grace_s": 60.0}
    inputs = json.loads((FEAS.parent / "E1-V3-SCALE-INPUTS.json").read_text())
    text = json.dumps(inputs)
    for needle in ("2700", "600", "128"):
        assert needle in text
