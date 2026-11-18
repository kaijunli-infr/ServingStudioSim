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


@pytest.fixture(autouse=True)
def _guard_disarmed():
    """The build-tool guard is process-wide once installed; every test starts and ends with it disarmed."""
    frontend._NO_BUILD.update(active=False, blocked=[])
    yield
    frontend._NO_BUILD.update(active=False, blocked=[])


@pytest.fixture
def tripwire(tmp_path, monkeypatch):
    """Fake cargo/rustc/rustup first on PATH: had anything really started one, the marker file would exist."""
    bin_dir = tmp_path / "tripwire-bin"
    bin_dir.mkdir()
    marker = tmp_path / "build-tool-was-started"
    for tool in frontend.BUILD_TOOLS:
        exe = bin_dir / tool
        exe.write_text(f'#!/bin/sh\necho "$0 $@" >> "{marker}"\nexit 97\n')
        exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return marker


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


# --- no build tool can be started in a prebuilt launch, also not through a helper (final rehearsal, item 5) ---


@pytest.mark.parametrize(
    "command",
    [
        "cargo",
        ["cargo", "build"],
        "/home/x/.cargo/bin/cargo",
        ["uv", "run", "cargo", "build", "--release"],
        "cd /x && cargo build --release",
        ["bash", "-lc", "rustc --version"],
        b"cargo",
        Path("/usr/bin/rustup"),
        ["env", "A=1", "/opt/rust/bin/rustc", "x.rs"],
        "(cargo build)",
        [None, ["nice", "cargo", "build"]],
    ],
)
def test_build_tool_commands_are_recognised(command):
    assert frontend._names_build_tool(command)


@pytest.mark.parametrize(
    "command",
    [
        None,
        ["git", "-C", "/x/.cargo/registry", "status", "--short"],
        "/home/x/.cargo/bin/uv run --no-sync python -m launcher",
        ["/x/req-frontend/target/release/session_runner", "--trace", "t.csv", "--stream", "false"],
        ["git", "ls-files", "--", "*.rs", "Cargo.toml", "Cargo.lock"],
        "/data/cargo/model.bin",
        ["/opt/nvidia/nsight-systems/2025.6.3/bin/nsys", "stop", "--session=x"],
    ],
)
def test_other_commands_are_not_mistaken_for_build_tools(command):
    assert frontend._names_build_tool(command) is None


def test_the_guard_is_inert_until_a_prebuilt_launch_arms_it(tripwire):
    assert frontend.build_tool_guard_state()["active"] is False
    # disarmed: the development path may start cargo (here: the tripwire stand-in, exit code 97)
    assert subprocess.run(["cargo", "build"], check=False).returncode == 97
    assert tripwire.read_text().count("cargo build") == 1


def test_armed_guard_refuses_build_tools_through_every_spawn_route(tmp_path, tripwire):
    from launcher import exec as launcher_exec

    state = frontend.forbid_build_tools()
    assert state["installed"] is True and state["active"] is True and state["blocked"] == []
    gone = "/nonexistent-rehearsal-dir/cargo"  # never executable, even if the guard did not hold
    routes = {
        "subprocess.run argv": lambda: subprocess.run(["cargo", "build"], check=False),
        "subprocess.Popen absolute path": lambda: subprocess.Popen([gone, "build"]),
        "wrapped in another program": lambda: subprocess.run(["uv", "run", "cargo", "build"]),
        "shell string": lambda: subprocess.run("cd / && cargo build --release", shell=True),
        "check_output rustc": lambda: subprocess.check_output(["rustc", "--version"]),
        "os.system": lambda: os.system("cargo --version"),
        "os.popen": lambda: os.popen("rustup show"),
        "os.execv": lambda: os.execv(gone, ["cargo", "build"]),
        "os.posix_spawn": lambda: os.posix_spawn(gone, ["cargo", "build"], dict(os.environ)),
        "development build helper": frontend.build_session_runner,
    }
    for name, call in routes.items():
        with pytest.raises(frontend.FrontendBinaryError, match="forbidden"):
            call()
        assert len(frontend.build_tool_guard_state()["blocked"]) >= 1, name
    assert len(frontend.build_tool_guard_state()["blocked"]) >= len(routes)
    # the launcher's own process supervisor (what launcher.exec.cargo_build spawns through)
    before = len(frontend.build_tool_guard_state()["blocked"])
    try:
        result = launcher_exec._PROCESS_SUPERVISOR.run_sync(
            launcher_exec.ProcessSpec(
                argv=["cargo", "build"], cwd=tmp_path, env=dict(os.environ), name="build-simulator"
            )
        )
        assert not result.succeeded
    except frontend.FrontendBinaryError:
        pass
    assert len(frontend.build_tool_guard_state()["blocked"]) > before
    assert not tripwire.exists()  # nothing was started, by any route
    # other programs still run
    assert subprocess.run(["git", "--version"], capture_output=True, check=False).returncode == 0


def test_the_real_launch_path_arms_the_guard_before_the_engine_start(
    tmp_path, monkeypatch, tripwire
):
    root, exe, prov = _checkout(tmp_path)
    _mock_until_engine_start(monkeypatch, root)
    cfg = _cfg(tmp_path)
    with pytest.raises(EngineStartReached):
        profile_runner.run_profile(cfg, frontend_provenance=prov)
    assert frontend.build_tool_guard_state() == {"installed": True, "active": True, "blocked": []}
    # from here on the development build is unreachable in this process, directly or through the launch code
    with pytest.raises(frontend.FrontendBinaryError, match="forbidden"):
        frontend.build_session_runner()
    with pytest.raises(frontend.FrontendBinaryError, match="forbidden"):
        profile_runner.run_profile(cfg)  # the development branch, in an armed process
    assert not tripwire.exists() and exe.read_bytes() == BINARY


def test_a_helper_asking_for_a_build_during_the_launch_is_refused_and_fails_the_launch(
    tmp_path, monkeypatch, tripwire
):
    root, exe, prov = _checkout(tmp_path)

    def helper(path):  # an indirect helper that swallows its own failure, as a build wrapper might
        try:
            subprocess.run(["cargo", "build", "--release"], check=True)
        except Exception:  # noqa: BLE001
            pass

    cfg = _launcher_mocks(monkeypatch, tmp_path, root, during_launch=helper)
    with pytest.raises(ValueError, match="build tool was requested"):
        launcher_alignment._run_profile(_args(tmp_path / "profile.json", prov))
    rec = json.loads(
        (Path(cfg.log_dir) / "load_generator" / frontend.FRONTEND_RECORD_NAME).read_text()
    )
    guard = rec["build_tool_guard"]
    assert guard["installed"] is True and len(guard["blocked"]) == 1
    assert guard["blocked"][0]["event"] == "subprocess.Popen"
    assert "cargo" in guard["blocked"][0]["command"]
    assert rec["unchanged"] is True and exe.read_bytes() == BINARY and not tripwire.exists()


def test_a_clean_mocked_launch_records_an_armed_guard_with_nothing_blocked(
    tmp_path, monkeypatch, tripwire
):
    root, exe, prov = _checkout(tmp_path)
    cfg = _launcher_mocks(monkeypatch, tmp_path, root, during_launch=lambda p: None)
    assert launcher_alignment._run_profile(_args(tmp_path / "profile.json", prov)) == 0
    rec = json.loads(
        (Path(cfg.log_dir) / "load_generator" / frontend.FRONTEND_RECORD_NAME).read_text()
    )
    assert rec["build_tool_guard"] == {"installed": True, "active": True, "blocked": []}
    assert rec["unchanged"] is True and not tripwire.exists()


def _functions(files):
    found: dict[str, list] = {}
    for file in files:
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found.setdefault(node.name, []).append(node)
    return found


def _starts_build_tool(node) -> bool:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # the docstring is text, not a command
    return any(
        isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and frontend._names_build_tool(n.value)
        for stmt in body
        for n in ast.walk(stmt)
    )


def _called_names(node) -> set[str]:
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                names.add(n.func.id)
            elif isinstance(n.func, ast.Attribute):
                names.add(n.func.attr)
    return names


def test_no_function_that_starts_a_build_tool_is_reachable_from_the_profile_launch():
    """Static check over the launch code (by called name, an over-approximation): from the launcher's profile
    entry point the only reachable function that starts a build tool is the development build, which refuses
    when the guard is armed (tested above)."""
    files = [
        REPO / "launcher" / "alignment.py",
        REPO / "launcher" / "exec.py",
        *sorted((REPO / "launcher").glob("process*.py")),
        *sorted((REPO / "launcher").glob("process*/*.py")),
        REPO / "alignment" / "runner.py",
        *sorted((REPO / "alignment" / "load_generator").glob("*.py")),
        *sorted((REPO / "alignment" / "profiler").glob("*.py")),
    ]
    functions = _functions(files)
    starters = {name for name, nodes in functions.items() if any(map(_starts_build_tool, nodes))}
    assert {"build_session_runner", "cargo_build", "_analyzer_build_command"} <= starters
    reached, todo = set(), ["_run_profile"]
    while todo:
        name = todo.pop()
        if name in reached or name not in functions:
            continue
        reached.add(name)
        for node in functions[name]:
            todo.extend(_called_names(node))
    assert {"run_profile", "verify_prebuilt_session_runner", "forbid_build_tools"} <= reached
    assert reached & starters == {"build_session_runner"}, sorted(reached & starters)
    # and that one function refuses first when the guard is armed
    first = functions["build_session_runner"][0].body[1]
    assert isinstance(first, ast.If) and ast.unparse(first.test) == "_NO_BUILD['active']"
    assert isinstance(first.body[-1], ast.Raise)
