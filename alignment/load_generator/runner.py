"""Build and invoke req-frontend's typed trace frontend for a profiling run."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import LoadGeneratorConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
REQ_FRONTEND_ROOT = Path(__file__).resolve().parent / "req-frontend"
REPLAY_MANIFEST = REQ_FRONTEND_ROOT / "Cargo.toml"
SESSION_RUNNER = REQ_FRONTEND_ROOT / "target" / "release" / "session_runner"
# Draws or materializes a canonical trace (`tracegen describe` lists its knobs).
TRACEGEN = REQ_FRONTEND_ROOT / "target" / "release" / "tracegen"


@dataclass(frozen=True)
class PreparedReplay:
    """Resolved req-frontend invocation inputs prepared before vLLM starts."""

    trace_path: Path
    text_file: Path
    tokenizer: str
    log_path: Path
    summary_path: Path


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


#: Record next to the binary saying how these exact bytes came to be in this checkout. An
#: authorized launch refuses to start without it (see `verify_prebuilt_session_runner`).
BUILD_STATE_NAME = "session_runner.build-state.json"
#: Written into the run's log directory: what was verified before the launch and after it.
FRONTEND_RECORD_NAME = "frontend_binary.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class FrontendBinaryError(RuntimeError):
    """The prebuilt load-generator contract does not hold; nothing may be launched."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        raise FrontendBinaryError(
            f"build state incompatible: git {' '.join(args)} failed in {root}: "
            f"{out.stderr.strip()[:200]}"
        )
    return out.stdout.strip()


def _load_pin(provenance: Path) -> tuple[str, str]:
    try:
        pin = json.loads(Path(provenance).read_text())
        sha, commit = str(pin["sha256"]), str(pin["req_frontend_commit"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise FrontendBinaryError(
            f"provenance missing or malformed: {provenance}: {error}"
        ) from error
    if not _SHA256.match(sha) or not _COMMIT.match(commit):
        raise FrontendBinaryError(
            f"provenance missing or malformed: {provenance}: sha256/req_frontend_commit "
            "are not a full sha256 and a full commit id"
        )
    return sha, commit


def verify_prebuilt_session_runner(
    provenance: Path, *, root: Path | None = None, binary: Path | None = None
) -> dict:
    """Fail-closed check of the prebuilt load generator. Never compiles, never writes.

    The launch may proceed only if the executable is byte-identical to the pinned one
    (`provenance`: `sha256`, `req_frontend_commit`), the source checkout is that commit and
    unmodified, and the build-state record next to the binary describes these same bytes
    and this source and is not older than a source change. Source equality alone is never
    accepted in place of the executable's hash. Raises FrontendBinaryError otherwise.
    """
    root = Path(root) if root is not None else REQ_FRONTEND_ROOT
    binary = Path(binary) if binary is not None else root / "target" / "release" / "session_runner"
    want_sha, want_commit = _load_pin(provenance)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FrontendBinaryError(f"executable missing or not executable: {binary}")
    state_path = binary.with_name(BUILD_STATE_NAME)
    try:
        state = json.loads(state_path.read_text())
    except OSError as error:
        raise FrontendBinaryError(
            f"build state missing: no {BUILD_STATE_NAME} next to {binary} (the origin of "
            "these bytes is unrecorded; record it with `record-build-state`)"
        ) from error
    except ValueError as error:
        raise FrontendBinaryError(f"build state incompatible: {state_path}: {error}") from error
    head = _git(root, "rev-parse", "HEAD")
    if head != want_commit:
        raise FrontendBinaryError(
            f"build state incompatible: source checkout is {head}, pinned {want_commit}"
        )
    dirty = _git(root, "status", "--short", "--untracked-files=no")
    if dirty:
        raise FrontendBinaryError(
            f"build state stale: tracked source files are modified: {dirty[:200]}"
        )
    if state.get("sha256") != want_sha or state.get("source_commit") != want_commit:
        raise FrontendBinaryError(
            "build state incompatible: the record describes sha256 "
            f"{str(state.get('sha256'))[:16]} / source {str(state.get('source_commit'))[:12]}, "
            f"pinned {want_sha[:16]} / {want_commit[:12]}"
        )
    recorded_at = state.get("recorded_at")
    if not isinstance(recorded_at, (int, float)):
        raise FrontendBinaryError("build state incompatible: the record has no recorded_at")
    newer = [
        name
        for name in _git(root, "ls-files", "--", "*.rs", "Cargo.toml", "Cargo.lock").splitlines()
        if (root / name).is_file() and (root / name).stat().st_mtime > recorded_at
    ]
    if newer:
        raise FrontendBinaryError(
            f"build state stale: {len(newer)} source file(s) changed after the record "
            f"was made, e.g. {newer[:3]}"
        )
    have_sha = sha256_file(binary)
    if have_sha != want_sha:
        raise FrontendBinaryError(
            f"executable hash mismatch: {binary} is {have_sha}, pinned {want_sha} "
            "(the source commit matching is not sufficient)"
        )
    stat = binary.stat()
    return {
        "schema_version": 1,
        "mode": "prebuilt (no compilation in this launch path)",
        "path": str(binary),
        "sha256": have_sha,
        "pinned_sha256": want_sha,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "source_commit": head,
        "source_clean": True,
        "provenance_file": str(provenance),
        "build_state": state,
        "cargo_fingerprints_present": (binary.parent / ".fingerprint").is_dir(),
        "cargo_consulted": False,
        "verified_before_launch_at": time.time(),
    }


def record_build_state(provenance: Path, origin: str, *, root: Path | None = None) -> dict:
    """Write the build-state record for a binary that already matches the pin.

    Refuses unless the executable is byte-identical to the pinned one and the source
    checkout is the pinned commit, unmodified. It never builds and never touches the binary.
    """
    root = Path(root) if root is not None else REQ_FRONTEND_ROOT
    binary = root / "target" / "release" / "session_runner"
    want_sha, want_commit = _load_pin(provenance)
    if not binary.is_file():
        raise FrontendBinaryError(f"executable missing: {binary}")
    have_sha = sha256_file(binary)
    if have_sha != want_sha:
        raise FrontendBinaryError(f"executable hash mismatch: {have_sha}, pinned {want_sha}")
    head = _git(root, "rev-parse", "HEAD")
    if head != want_commit or _git(root, "status", "--short", "--untracked-files=no"):
        raise FrontendBinaryError(f"source checkout is not the clean pinned commit {want_commit}")
    state = {
        "schema_version": 1,
        "sha256": have_sha,
        "source_commit": head,
        "origin": origin,
        "provenance_file": str(provenance),
        "recorded_at": time.time(),
        "recorded_by": "alignment.load_generator.runner record-build-state",
    }
    target = binary.with_name(BUILD_STATE_NAME)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    os.replace(tmp, target)
    return state


def write_frontend_record(log_dir: Path, record: dict) -> Path:
    target = Path(log_dir) / "load_generator" / FRONTEND_RECORD_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=1) + "\n")
    return target


def verify_after_launch(log_dir: Path) -> dict | None:
    """Hash the executable again after the launch and add the result to its record."""
    target = Path(log_dir) / "load_generator" / FRONTEND_RECORD_NAME
    try:
        record = json.loads(target.read_text())
    except (OSError, ValueError):
        return None
    binary = Path(record["path"])
    after = sha256_file(binary) if binary.is_file() else None
    record["sha256_after_launch"] = after
    record["verified_after_launch_at"] = time.time()
    record["unchanged"] = after == record.get("sha256") == record.get("pinned_sha256")
    target.write_text(json.dumps(record, indent=1) + "\n")
    return record


def build_session_runner() -> Path:
    """Build req-frontend before starting vLLM so compilation is outside nsys.

    Development path only. A launch that names a provenance file goes through
    `verify_prebuilt_session_runner` instead and never reaches this function.
    """
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(REPLAY_MANIFEST),
            "--bin",
            "session_runner",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return SESSION_RUNNER


def prepare_replay(config: LoadGeneratorConfig, log_dir: Path) -> PreparedReplay:
    """Resolve frontend inputs before launching the server."""
    trace_path = _repo_path(config.frontend.path)
    if not trace_path.is_file():
        raise FileNotFoundError(f"profiling trace not found: {trace_path}")
    text_file = _repo_path(config.text_file)
    if not text_file.is_file():
        raise FileNotFoundError(f"replay text corpus not found: {text_file}")
    return PreparedReplay(
        trace_path=trace_path,
        text_file=text_file,
        tokenizer=config.tokenizer,
        log_path=log_dir / "load_generator" / "replay.jsonl",
        summary_path=log_dir / "load_generator" / "summary.json",
    )


def run_replay(
    config: LoadGeneratorConfig,
    prepared: PreparedReplay,
    *,
    base_url: str,
    model: str,
    measurement_ready: Callable[[], None] | None = None,
    routed_experts_dir: Path | None = None,
) -> dict:
    """Replay the shared trace through its explicitly selected wire backend.

    `routed_experts_dir` turns the replay non-streaming and collects the routes
    the server took per request. The caller decides when that trade is worth
    making; this layer only knows that it costs the per-event timeline.
    """
    prepared.log_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_base_url = (
        f"{base_url.rstrip('/')}/v1"
        if config.backend.type == "openai"
        else base_url.rstrip("/")
    )
    argv = [
        str(SESSION_RUNNER),
        "--trace",
        str(prepared.trace_path),
        "--input-file-format",
        config.frontend.input_file_format,
        "--text-file",
        str(prepared.text_file),
        "--tokenizer",
        prepared.tokenizer,
        "--base-url",
        protocol_base_url,
        "--backend",
        config.backend.cli_value,
        "--model",
        model,
        "--log-path",
        str(prepared.log_path),
        "--summary-path",
        str(prepared.summary_path),
        "--stream-idle-timeout-secs",
        str(config.stream_idle_timeout_secs),
    ]
    optional_args = (
        ("--max-items", config.max_items),
        ("--rate", config.rate),
        ("--arrival-mode", config.arrival_mode),
        ("--token-pool-limit", config.token_pool_limit),
        ("--max-concurrency", config.max_concurrency),
        ("--max-model-len", config.max_model_len),
    )
    for flag, value in optional_args:
        if value is not None:
            argv.extend([flag, str(value)])
    if config.context_limit_skip_enabled:
        argv.append("--skip-when-reaching-limit")
    if config.warmup:
        argv.append("--warmup")
    if measurement_ready is not None:
        argv.append("--measurement-gate")
    if routed_experts_dir is not None:
        argv.extend(["--routed-experts-dir", str(routed_experts_dir)])
    if not config.stream:
        # `--stream false` (clap ArgAction::Set): one completed body per request, no routes written.
        argv.extend(["--stream", "false"])
    argv.extend(config.extra_args)

    if measurement_ready is None:
        subprocess.run(argv, cwd=REPO_ROOT, check=True)
    else:
        with subprocess.Popen(
            argv, cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        ) as proc:
            released = False
            try:
                for line in proc.stdout:
                    if line.rstrip("\n") == "REQ_FRONTEND_MEASUREMENT_READY_V1":
                        if released:
                            raise RuntimeError("duplicate frontend measurement boundary")
                        measurement_ready()
                        proc.stdin.write("continue\n")
                        proc.stdin.flush()
                        released = True
                    else:
                        print(line, end="", flush=True)
                returncode = proc.wait()
                if returncode:
                    raise subprocess.CalledProcessError(returncode, argv)
                if not released:
                    raise RuntimeError("frontend exited without a measurement boundary")
            except BaseException:
                proc.kill()
                proc.wait()
                raise
    return {
        "source_trace": str(prepared.trace_path.resolve()),
        "frontend_type": config.frontend.type,
        "backend_type": config.backend.type,
        "log_path": str(prepared.log_path),
        "summary_path": str(prepared.summary_path),
        "warmup": config.warmup,
        "routed_experts_dir": str(routed_experts_dir) if routed_experts_dir else None,
    }


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="prebuilt load-generator contract (never compiles)")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("verify-prebuilt", "record-build-state"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--provenance", required=True, type=Path)
        if name == "record-build-state":
            cmd.add_argument("--origin", required=True)
    args = ap.parse_args(argv)
    try:
        if args.command == "verify-prebuilt":
            rec = verify_prebuilt_session_runner(args.provenance)
            print(
                f"PASS  prebuilt session_runner {rec['path']}: sha256 {rec['sha256']} == pinned; "
                f"source {rec['source_commit']} clean; build state recorded "
                f"({rec['build_state'].get('origin')}); cargo not consulted"
            )
        else:
            state = record_build_state(args.provenance, args.origin)
            print(
                f"PASS  build state recorded: sha256 {state['sha256']} "
                f"source {state['source_commit']}"
            )
    except FrontendBinaryError as error:
        print(f"FAIL  {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
