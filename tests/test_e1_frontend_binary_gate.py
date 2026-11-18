"""CPU tests for the V2 post-run gate S.binary_unchanged (checks/e1_frontend_binary_gate.py): the pinned executable
before and after the launch in two independent records, nothing compiled, no build tool requested."""

from __future__ import annotations

# ruff: noqa: E501

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
E1 = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1"
PIN = "7f6eb81d45e6b98f3b6ebb56ab20b60ce8aaf4584aa3d4f25e3449b2d05a00c2"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(path.stem, mod)
    spec.loader.exec_module(mod)
    return mod


GATE = _load(E1 / "checks" / "e1_frontend_binary_gate.py")
REHEARSAL = _load(E1 / "e1_fr_rehearsal.py")


def _failed(driver, launcher, pin=PIN, now=None):
    return [c["check"] for c in GATE.evaluate(driver, launcher, pin, now) if not c["ok"]]


def test_the_pin_is_the_original_session_runner_hash():
    assert json.loads((E1 / "BINARY-PROVENANCE.json").read_text())["sha256"] == PIN
    assert REHEARSAL.OTHER_SHA != PIN


def test_pinned_records_pass_every_check():
    driver, launcher = REHEARSAL.binary_records(PIN, "pinned")
    assert _failed(driver, launcher, now=PIN) == []
    assert len(GATE.evaluate(driver, launcher, PIN, PIN)) == 16


@pytest.mark.parametrize(
    ("variant", "must_fail"),
    [
        ("mutated_during_launch", {"driver.sha256_after_is_pinned", "launcher.unchanged"}),
        (
            "not_pinned_before_launch",
            {"driver.sha256_before_is_pinned", "launcher.sha256_before_is_pinned"},
        ),
        ("replaced_and_restored", {"driver.file_identity_unchanged"}),
        ("compiled_during_launch", {"driver.no_compiling_lines"}),
        ("build_tool_requested", {"launcher.no_build_tool_requested"}),
        (
            "no_guard",
            {"launcher.build_tool_guard_installed", "launcher.no_build_tool_requested"},
        ),
        ("records_missing", {"driver.record_present", "launcher.record_present"}),
    ],
)
def test_each_mutation_fails_the_gate(variant, must_fail):
    driver, launcher = REHEARSAL.binary_records(PIN, variant)
    assert must_fail <= set(_failed(driver, launcher))


def test_source_equality_or_self_consistent_records_never_replace_the_pin():
    driver, launcher = REHEARSAL.binary_records(PIN, "pinned")
    other = REHEARSAL.OTHER_SHA
    # both records agree with each other and say "unchanged", but describe other bytes than the pin
    driver.update(sha256_before_launcher=other, sha256_after_launcher=other)
    launcher.update(sha256=other, pinned_sha256=other, sha256_after_launch=other)
    failed = set(_failed(driver, launcher))
    assert {"driver.sha256_before_is_pinned", "launcher.sha256_after_is_pinned"} <= failed
    # the executable was replaced after the launcher ended, before the post-run
    driver, launcher = REHEARSAL.binary_records(PIN, "pinned")
    assert _failed(driver, launcher, now=other) == ["binary.now_is_pinned"]
    assert _failed(driver, launcher, pin=None)  # no pin: nothing can pass


def test_cli_exit_code_and_report(tmp_path, capsys):
    driver, launcher = REHEARSAL.binary_records(PIN, "pinned")
    (tmp_path / "d.json").write_text(json.dumps(driver))
    (tmp_path / "l.json").write_text(json.dumps(launcher))
    (tmp_path / "p.json").write_text(json.dumps({"sha256": PIN}))
    exe = tmp_path / "exe"
    exe.write_bytes(b"not the pinned bytes")
    base = ["--driver-record", str(tmp_path / "d.json"), "--launcher-record"]
    base += [str(tmp_path / "l.json"), "--provenance", str(tmp_path / "p.json")]
    out = tmp_path / "gate.json"
    assert GATE.main([*base, "--out", str(out)]) == 0
    assert json.loads(out.read_text())["verdict"] == "PASS"
    assert GATE.main([*base, "--binary", str(exe), "--out", str(out)]) == 1
    assert json.loads(out.read_text())["verdict"] == "FAIL"
    assert GATE.main([*base, "--binary", str(tmp_path / "gone")]) == 1
    assert GATE.main([*base[:1], str(tmp_path / "absent.json"), *base[2:]]) == 1
    assert "FAIL  driver.record_present" in capsys.readouterr().out
