"""Fail-closed prebuilt load-generator contract (session_runner provenance repair, items B and C).

A launch that names a provenance file must run the exact pinned bytes, must never invoke cargo, must refuse before
any engine start when the executable, its source or its build-state record is wrong, and must verify the bytes again
when the launch ends.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import ast
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

import alignment.runner as profile_runner
import launcher.alignment as launcher_alignment
from alignment.load_generator import runner as frontend

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "e1_nsys_val"
REAL_RUN = subprocess.run
BINARY = b"#!/bin/sh\necho session_runner pinned build\n"


class CargoInvoked(AssertionError):
    pass


@pytest.fixture
def no_cargo(monkeypatch):
    """Any attempt to start cargo, anywhere, fails the test and is recorded."""
    calls = []

    def guarded(argv, *a, **kw):
        first = argv[0] if isinstance(argv, (list, tuple)) else str(argv).split()[0]
        if Path(str(first)).name == "cargo":
            calls.append(list(argv))
            raise CargoInvoked(f"cargo was invoked: {argv}")
        return REAL_RUN(argv, *a, **kw)

    real_popen = subprocess.Popen

    class GuardedPopen(real_popen):
        def __init__(self, argv, *a, **kw):
            first = argv[0] if isinstance(argv, (list, tuple)) else str(argv).split()[0]
            if Path(str(first)).name == "cargo":
                calls.append(list(argv))
                raise CargoInvoked(f"cargo was invoked: {argv}")
            super().__init__(argv, *a, **kw)

    monkeypatch.setattr(subprocess, "run", guarded)
    monkeypatch.setattr(subprocess, "Popen", GuardedPopen)
    return calls


def _git(root: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    return REAL_RUN(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True, env=env
    ).stdout.strip()


def _checkout(tmp: Path, *, record: bool = True, binary: bytes = BINARY) -> tuple[Path, Path, Path]:
    """A req-frontend checkout with a prebuilt binary, its pin and (optionally) its build-state record."""
    root = tmp / "req-frontend"
    (root / "src").mkdir(parents=True)
    (root / "Cargo.toml").write_text('[package]\nname = "x"\n')
    (root / "src" / "main.rs").write_text("fn main() {}\n")
    (root / ".gitignore").write_text("target/\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "pinned source")
    old = time.time() - 3600
    for f in (root / "Cargo.toml", root / "src" / "main.rs"):
        os.utime(f, (old, old))
    exe = root / "target" / "release" / "session_runner"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(binary)
    exe.chmod(0o755)
    prov = tmp / "BINARY-PROVENANCE.json"
    prov.write_text(
        json.dumps(
            {
                "sha256": hashlib.sha256(BINARY).hexdigest(),
                "req_frontend_commit": _git(root, "rev-parse", "HEAD"),
            }
        )
    )
    if record:
        frontend.record_build_state(
            prov, "copied from the reference worktree; sha256 verified", root=root
        )
    return root, exe, prov


def test_valid_binary_with_compatible_build_state_is_accepted_without_cargo(tmp_path, no_cargo):
    root, exe, prov = _checkout(tmp_path)
    (exe.parent / ".fingerprint").mkdir()  # cargo's own state present
    rec = frontend.verify_prebuilt_session_runner(prov, root=root)
    assert rec["sha256"] == rec["pinned_sha256"] == hashlib.sha256(BINARY).hexdigest()
    assert (
        rec["path"] == str(exe)
        and rec["source_commit"] == _git(root, "rev-parse", "HEAD")
        and rec["source_clean"]
    )
    assert rec["build_state"]["origin"].startswith("copied from the reference worktree")
    assert (
        rec["cargo_consulted"] is False
        and rec["cargo_fingerprints_present"] is True
        and no_cargo == []
    )
    assert exe.read_bytes() == BINARY


def test_missing_build_state_refuses(tmp_path, no_cargo):
    root, exe, prov = _checkout(tmp_path, record=False)
    with pytest.raises(frontend.FrontendBinaryError, match="build state missing"):
        frontend.verify_prebuilt_session_runner(prov, root=root)
    assert no_cargo == [] and exe.read_bytes() == BINARY  # refused, not rebuilt


def test_stale_cargo_fingerprints_cannot_trigger_a_rebuild_and_stale_sources_refuse(
    tmp_path, no_cargo
):
    root, exe, prov = _checkout(tmp_path)
    fp = exe.parent / ".fingerprint" / "session_runner-0123"
    fp.mkdir(parents=True)
    (fp / "invoked.timestamp").write_text("stale fingerprint from another path")
    os.utime(fp / "invoked.timestamp", (1, 1))  # cargo would call this build stale and recompile
    rec = frontend.verify_prebuilt_session_runner(
        prov, root=root
    )  # the prebuilt path does not ask cargo
    assert rec["cargo_consulted"] is False and no_cargo == [] and exe.read_bytes() == BINARY
    os.utime(root / "src" / "main.rs", None)  # a source file newer than the build-state record
    with pytest.raises(frontend.FrontendBinaryError, match="build state stale"):
        frontend.verify_prebuilt_session_runner(prov, root=root)
    (tmp_path / "b").mkdir()
    root, exe, prov = _checkout(tmp_path / "b")
    (root / "src" / "main.rs").write_text("fn main() { /* edited */ }\n")
    with pytest.raises(frontend.FrontendBinaryError, match="build state stale"):
        frontend.verify_prebuilt_session_runner(prov, root=root)
    assert no_cargo == []


def test_incorrect_executable_hash_refuses_even_with_identical_source(tmp_path, no_cargo):
    root, exe, prov = _checkout(tmp_path)
    exe.write_bytes(BINARY + b"# rebuilt from the same source in another path\n")
    with pytest.raises(
        frontend.FrontendBinaryError, match="executable hash mismatch.*not sufficient"
    ):
        frontend.verify_prebuilt_session_runner(prov, root=root)
    exe.unlink()
    with pytest.raises(frontend.FrontendBinaryError, match="executable missing"):
        frontend.verify_prebuilt_session_runner(prov, root=root)
    assert no_cargo == []


@pytest.mark.parametrize(
    "case",
    [
        "record_other_sha",
        "record_other_commit",
        "new_commit",
        "malformed_pin",
        "short_sha_pin",
        "record_refused_for_wrong_bytes",
    ],
)
def test_incompatible_build_state_or_pin_refuses(tmp_path, no_cargo, case):
    root, exe, prov = _checkout(tmp_path)
    state = exe.with_name(frontend.BUILD_STATE_NAME)
    if case == "record_other_sha":
        state.write_text(json.dumps({**json.loads(state.read_text()), "sha256": "0" * 64}))
    elif case == "record_other_commit":
        state.write_text(json.dumps({**json.loads(state.read_text()), "source_commit": "1" * 40}))
    elif case == "new_commit":
        (root / "README").write_text("x")
        _git(root, "add", "README")
        _git(root, "commit", "-q", "-m", "moved on")
    elif case == "malformed_pin":
        prov.write_text("{not json")
    elif case == "short_sha_pin":
        prov.write_text(
            json.dumps(
                {"sha256": "7f6eb81d", "req_frontend_commit": _git(root, "rev-parse", "HEAD")}
            )
        )
    else:
        exe.write_bytes(b"other bytes")
        with pytest.raises(frontend.FrontendBinaryError, match="hash mismatch"):
            frontend.record_build_state(prov, "attempt to record a rebuilt binary", root=root)
    with pytest.raises(frontend.FrontendBinaryError):
        frontend.verify_prebuilt_session_runner(prov, root=root)
    assert no_cargo == []


def test_binary_replaced_between_preflight_and_launch_is_refused(tmp_path, no_cargo):
    root, exe, prov = _checkout(tmp_path)
    frontend.verify_prebuilt_session_runner(prov, root=root)  # preflight: passes
    exe.write_bytes(BINARY.replace(b"pinned", b"rebilt"))  # same size, different bytes
    with pytest.raises(frontend.FrontendBinaryError, match="executable hash mismatch"):
        frontend.verify_prebuilt_session_runner(prov, root=root)  # launch: refuses
    assert no_cargo == []


def _cfg(tmp_path: Path):
    from launcher.alignment_config import load_profile_config

    cfg = load_profile_config(FIX / "profile.json", require_python_runtime=False)
    cfg.log_dir = str(tmp_path / "run" / "profile")
    return cfg


class EngineStartReached(Exception):
    pass


def _mock_until_engine_start(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(frontend, "REQ_FRONTEND_ROOT", root)
    monkeypatch.setattr(profile_runner, "require_gpu", lambda *a, **kw: None)
    monkeypatch.setattr(profile_runner.load_generator, "prepare_replay", lambda *a, **kw: object())

    def engine_start(*a, **kw):
        raise EngineStartReached

    monkeypatch.setattr(profile_runner, "_resolve_fork_python", engine_start)


def test_the_real_launch_path_verifies_before_engine_start_and_never_runs_cargo(
    tmp_path, monkeypatch, no_cargo
):
    root, exe, prov = _checkout(tmp_path)
    _mock_until_engine_start(monkeypatch, root)
    cfg = _cfg(tmp_path)
    with pytest.raises(
        EngineStartReached
    ):  # the contract held, so the launch went on to the engine
        profile_runner.run_profile(cfg, frontend_provenance=prov)
    rec = json.loads(
        (Path(cfg.log_dir) / "load_generator" / frontend.FRONTEND_RECORD_NAME).read_text()
    )
    assert (
        rec["sha256"] == hashlib.sha256(BINARY).hexdigest()
        and rec["mode"].startswith("prebuilt")
        and no_cargo == []
    )
    exe.write_bytes(b"replaced after the preflight")
    with pytest.raises(
        frontend.FrontendBinaryError, match="hash mismatch"
    ):  # refused BEFORE the engine start
        profile_runner.run_profile(cfg, frontend_provenance=prov)
    assert no_cargo == []


def test_without_a_provenance_file_the_development_path_is_the_only_one_that_reaches_cargo(
    tmp_path, monkeypatch, no_cargo
):
    root, _, _ = _checkout(tmp_path)
    _mock_until_engine_start(monkeypatch, root)
    with pytest.raises(CargoInvoked):
        profile_runner.run_profile(_cfg(tmp_path))
    assert len(no_cargo) == 1 and no_cargo[0][:2] == ["cargo", "build"]
    # source-level: cargo appears in exactly one function, reached only through the non-prebuilt branch
    src = ast.parse(Path(frontend.__file__).read_text())
    users = {
        f.name
        for f in ast.walk(src)
        if isinstance(f, ast.FunctionDef)
        and any(isinstance(n, ast.Constant) and n.value == "cargo" for n in ast.walk(f))
    }
    assert users == {"build_session_runner"}
    text = Path(profile_runner.__file__).read_text()
    assert text.count("build_session_runner()") == 1
    branch = text[
        text.index("if frontend_provenance is not None:") : text.index(
            "load_generator.build_session_runner()"
        )
    ]
    assert "verify_prebuilt_session_runner" in branch and branch.rstrip().endswith("else:")


def _args(cfg_path: Path, prov: Path, dry: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        config=cfg_path, dry_run=dry, resume=False, import_stream=None, frontend_provenance=prov
    )


def _launcher_mocks(monkeypatch, tmp_path, root, during_launch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(frontend, "REQ_FRONTEND_ROOT", root)
    monkeypatch.setattr(launcher_alignment, "load_profile_config", lambda *a, **kw: cfg)
    monkeypatch.setattr(launcher_alignment, "write_artifact_kind", lambda *a, **kw: None)
    monkeypatch.setattr(launcher_alignment, "_snapshot_config", lambda *a, **kw: None)

    def mocked_launch(config, *, resume=False, frontend_provenance=None, **kw):
        # the real pre-launch step, then a complete mocked capture that executes the load generator
        log_dir = Path(config.log_dir)
        rec = frontend.verify_prebuilt_session_runner(frontend_provenance)
        frontend.write_frontend_record(log_dir, rec)
        out = REAL_RUN([rec["path"]], capture_output=True, text=True, check=True).stdout
        during_launch(Path(rec["path"]))
        return {
            "profile_kind": "nsys",
            "parsed_nsys": str(log_dir / "parsed.json"),
            "ran": out.strip(),
        }

    monkeypatch.setattr(launcher_alignment, "run_profile", mocked_launch)
    return cfg


def test_complete_mocked_launch_leaves_the_executable_byte_identical(
    tmp_path, monkeypatch, no_cargo, capsys
):
    root, exe, prov = _checkout(tmp_path)
    before = exe.read_bytes()
    cfg = _launcher_mocks(monkeypatch, tmp_path, root, during_launch=lambda p: None)
    assert launcher_alignment._run_profile(_args(tmp_path / "profile.json", prov)) == 0
    rec = json.loads(
        (Path(cfg.log_dir) / "load_generator" / frontend.FRONTEND_RECORD_NAME).read_text()
    )
    pinned = hashlib.sha256(BINARY).hexdigest()
    assert (
        rec["sha256"] == rec["sha256_after_launch"] == rec["pinned_sha256"] == pinned
        and rec["unchanged"] is True
    )
    assert exe.read_bytes() == before == BINARY and no_cargo == []
    assert rec["verified_after_launch_at"] >= rec["verified_before_launch_at"]
    assert {"path", "sha256", "source_commit", "build_state", "provenance_file"} <= set(rec)
    out = capsys.readouterr().out
    assert "prebuilt session_runner verified" in out and "(unchanged)" in out


def test_a_binary_replaced_during_the_launch_is_detected_when_it_ends(
    tmp_path, monkeypatch, no_cargo
):
    root, exe, prov = _checkout(tmp_path)
    cfg = _launcher_mocks(
        monkeypatch,
        tmp_path,
        root,
        during_launch=lambda p: p.write_bytes(b"#!/bin/sh\necho rebuilt\n"),
    )
    with pytest.raises(ValueError, match="changed during the launch"):
        launcher_alignment._run_profile(_args(tmp_path / "profile.json", prov))
    rec = json.loads(
        (Path(cfg.log_dir) / "load_generator" / frontend.FRONTEND_RECORD_NAME).read_text()
    )
    assert rec["unchanged"] is False and rec["sha256_after_launch"] != rec["pinned_sha256"]


def test_dry_run_with_a_provenance_file_checks_the_contract_and_refuses_on_mismatch(
    tmp_path, monkeypatch, no_cargo
):
    root, exe, prov = _checkout(tmp_path)
    _launcher_mocks(monkeypatch, tmp_path, root, during_launch=lambda p: None)
    assert launcher_alignment._run_profile(_args(tmp_path / "profile.json", prov, dry=True)) == 0
    exe.write_bytes(b"wrong")
    with pytest.raises(
        ValueError, match="prebuilt load generator refused: executable hash mismatch"
    ):
        launcher_alignment._run_profile(_args(tmp_path / "profile.json", prov, dry=True))
    assert no_cargo == []


def test_cli_verify_and_record(tmp_path, monkeypatch, capsys):
    root, exe, prov = _checkout(tmp_path, record=False)
    monkeypatch.setattr(frontend, "REQ_FRONTEND_ROOT", root)
    assert frontend._main(["verify-prebuilt", "--provenance", str(prov)]) == 1
    assert "build state missing" in capsys.readouterr().out
    assert (
        frontend._main(
            [
                "record-build-state",
                "--provenance",
                str(prov),
                "--origin",
                "restored from wt-e1stage1",
            ]
        )
        == 0
    )
    assert frontend._main(["verify-prebuilt", "--provenance", str(prov)]) == 0
    assert "cargo not consulted" in capsys.readouterr().out
