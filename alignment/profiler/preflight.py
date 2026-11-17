"""Read-only identity preflight for an Nsight ``cuda_profiler_api`` capture.

Verifies, without starting anything, that the processes the lifecycle controller
will deal with are the ones the authorization pins: the Nsight CLI and the
``QdstrmImporter`` it will hand the stream to, the server interpreter and the
vLLM fork it imports, the launch recipe the config expands to, and the
identity-verified cleanup primitives on this host (``procident``). It also
proves that the controller's log parsers match a real server log of the
accepted recipe, and that the run's status/control/stream locations are clean.

    python -m alignment.profiler.preflight --config <profile.json>
        [--pins <pins.json>] [--reference-launch <launch.json>]
        [--reference-server-log <server.log>] [--json <out.json>] [--write-pins <pins.json>]

Exit status 1 if any check FAILs. Nothing is created except the files named by
``--json`` / ``--write-pins``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from . import lifecycle, nsys_capture, procident, vllm_server
from .config import ProfileConfig

PINNED_NSYS_KEYS = ("path", "version", "sha256")
PINNED_IMPORTER_KEYS = ("path", "sha256")
PINNED_SERVER_KEYS = (
    "fork_python",
    "python_realpath",
    "python_sha256",
    "vllm_version",
    "fork_head",
)


@dataclass
class Check:
    name: str
    status: str  # PASS | FAIL | INFO
    detail: str
    data: dict | None = None


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _python_probe(python: str, env: dict[str, str]) -> dict:
    """Ask the server interpreter which vLLM it imports (CPU; nothing is served)."""
    code = (
        "import json, sys\n"
        "import vllm\n"
        "print(json.dumps({'version': vllm.__version__, 'file': vllm.__file__,"
        " 'python': sys.executable}))\n"
    )
    out = subprocess.run(
        [python, "-c", code], capture_output=True, text=True, timeout=600, env=env, check=False
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[-500:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _compare(name: str, actual: dict, pinned: dict | None, keys: tuple[str, ...]) -> Check:
    if pinned is None:
        return Check(name, "INFO", "no pins supplied; recorded only", actual)
    diffs = [k for k in keys if k in pinned and pinned[k] != actual.get(k)]
    if diffs:
        return Check(
            name,
            "FAIL",
            "pin mismatch: " + ", ".join(f"{k} {actual.get(k)!r} != {pinned[k]!r}" for k in diffs),
            actual,
        )
    return Check(name, "PASS", "matches pins: " + ", ".join(keys), actual)


# ----------------------------------------------------------------------- checks
def check_profiler(cfg: ProfileConfig, pins: dict | None) -> tuple[list[Check], dict]:
    checks: list[Check] = []
    facts: dict = {}
    try:
        exe = nsys_capture.resolve_nsys_executable(cfg.nsys.executable)
    except Exception as error:  # noqa: BLE001 - reported as a check
        checks.append(Check("profiler.nsys", "FAIL", f"cannot resolve nsys: {error}"))
        return checks, facts
    nsys_facts = {"path": str(exe.path), "version": exe.version, "sha256": sha256_file(exe.path)}
    facts["nsys"] = nsys_facts
    checks.append(_compare("profiler.nsys", nsys_facts, (pins or {}).get("nsys"), PINNED_NSYS_KEYS))
    try:
        importer = nsys_capture.qdstrm_importer_path(exe, cfg.nsys.qdstrm_importer)
    except FileNotFoundError as error:
        checks.append(Check("profiler.importer", "FAIL", str(error)))
        return checks, facts
    imp_facts = {"path": str(importer), "sha256": sha256_file(importer)}
    facts["importer"] = imp_facts
    checks.append(
        _compare("profiler.importer", imp_facts, (pins or {}).get("importer"), PINNED_IMPORTER_KEYS)
    )
    same_install = importer.resolve().parent.parent == exe.path.resolve().parent.parent
    checks.append(
        Check(
            "profiler.importer_matches_cli",
            "PASS" if same_install else "FAIL",
            f"importer {importer} and CLI {exe.path} share the install root"
            if same_install
            else f"importer {importer} is not from the CLI's install {exe.path.parent.parent}",
        )
    )
    return checks, facts


def check_server(
    cfg: ProfileConfig, pins: dict | None, *, probe=_python_probe
) -> tuple[list[Check], dict]:
    checks: list[Check] = []
    facts: dict = {}
    python = Path(cfg.fork_python)
    if not python.is_file() or not os.access(python, os.X_OK):
        checks.append(Check("server.python", "FAIL", f"fork_python not executable: {python}"))
        return checks, facts
    real = python.resolve()
    server_facts = {
        "fork_python": str(python),
        "python_realpath": str(real),
        "python_sha256": sha256_file(real),
        "fork_head": _git_head(vllm_server.VLLM_SOURCE_ROOT),
        "fork_source_root": str(vllm_server.VLLM_SOURCE_ROOT),
    }
    env = vllm_server.build_server_env(cfg.fork_python, cfg.cuda_visible_devices)
    env["CUDA_VISIBLE_DEVICES"] = ""  # the probe must not touch a GPU
    try:
        probe_result = probe(str(python), env)
        server_facts["vllm_version"] = probe_result.get("version")
        server_facts["vllm_file"] = probe_result.get("file")
    except Exception as error:  # noqa: BLE001
        checks.append(Check("server.vllm_import", "FAIL", f"vLLM import probe failed: {error}"))
        server_facts["vllm_version"] = None
    else:
        from_source_root = str(server_facts["vllm_file"] or "").startswith(
            str(vllm_server.VLLM_SOURCE_ROOT)
        )
        checks.append(
            Check(
                "server.vllm_import",
                "PASS" if from_source_root else "FAIL",
                f"vllm {server_facts['vllm_version']} from {server_facts['vllm_file']}"
                + ("" if from_source_root else " (not the checked-out fork source root)"),
            )
        )
    facts["server"] = server_facts
    checks.append(
        _compare("server.identity", server_facts, (pins or {}).get("server"), PINNED_SERVER_KEYS)
    )
    model = Path(cfg.server.model_path)
    checks.append(
        Check(
            "server.model_path",
            "PASS" if (model / "config.json").is_file() else "FAIL",
            f"{model}/config.json {'present' if (model / 'config.json').is_file() else 'missing'}",
        )
    )
    return checks, facts


def expand_recipe(cfg: ProfileConfig) -> dict:
    """The server argv and the nsys prefix (without `-o <stem>`) the config expands to."""
    server_argv = vllm_server.build_server_argv(cfg.fork_python, cfg.server)
    if getattr(cfg.workload, "capture_routed_experts", False):
        server_argv += ["--enable-return-routed-experts"]
    server_argv += list(vllm_server.NSYS_CAPTURE_SERVER_ARGS) + list(
        vllm_server.CUDA_PROFILER_SERVER_ARGS
    )
    exe = nsys_capture.ResolvedNsysExecutable(Path(cfg.nsys.executable), "pinned", True)
    prefix = nsys_capture.build_nsys_prefix(exe, cfg.nsys, Path("/STEM"))
    i = prefix.index("-o")
    prefix_wo_output = prefix[:i] + prefix[i + 2 :]
    return {"server_argv": server_argv, "nsys_prefix_without_output": prefix_wo_output}


def check_recipe(cfg: ProfileConfig, reference_launch: Path | None) -> tuple[list[Check], dict]:
    checks: list[Check] = []
    recipe = expand_recipe(cfg)
    env = vllm_server.build_server_env(cfg.fork_python, cfg.cuda_visible_devices)
    if cfg.nsys.tmp_dir is not None:
        env["TMPDIR"] = cfg.nsys.tmp_dir  # what runner.py adds for the nsys process
    recipe["tmp_dir"] = cfg.nsys.tmp_dir
    if reference_launch is None:
        checks.append(
            Check("recipe.reference", "INFO", "no reference launch record supplied", recipe)
        )
        return checks, {"recipe": recipe}
    launch = json.loads(Path(reference_launch).read_text())
    ref_argv = launch["argv"]
    i = ref_argv.index("-o")
    ref_prefix = ref_argv[:i] + ref_argv[i + 2 : len(ref_argv) - len(launch["server_argv"])]
    same_server = recipe["server_argv"] == launch["server_argv"]
    same_prefix = recipe["nsys_prefix_without_output"][1:] == ref_prefix[1:]
    try:
        same_exe = Path(cfg.nsys.executable).resolve(strict=True) == Path(ref_argv[0]).resolve()
    except OSError:
        same_exe = False
    checks.append(
        Check(
            "recipe.server_argv",
            "PASS" if same_server else "FAIL",
            "server argv identical to the reference launch"
            if same_server
            else f"server argv differs from {reference_launch}",
        )
    )
    checks.append(
        Check(
            "recipe.nsys_prefix",
            "PASS" if same_prefix else "FAIL",
            "nsys flags identical to the reference launch (output stem excepted)"
            if same_prefix
            else f"nsys flags differ from {reference_launch}: "
            f"{recipe['nsys_prefix_without_output'][1:]} vs {ref_prefix[1:]}",
        )
    )
    checks.append(
        Check(
            "recipe.nsys_executable",
            "PASS" if same_exe else "FAIL",
            f"{cfg.nsys.executable} resolves to the reference launch's {ref_argv[0]}"
            if same_exe
            else f"{cfg.nsys.executable} does not resolve to the reference launch's {ref_argv[0]}",
        )
    )
    ref_env = launch.get("env", {})
    extra = sorted(k for k in ("TMPDIR",) if k in env and k not in ref_env)
    checks.append(
        Check(
            "recipe.env",
            "PASS" if not extra else "FAIL",
            "no environment variable added relative to the reference launch"
            if not extra
            else f"environment adds {extra} relative to the reference launch",
        )
    )
    return checks, {"recipe": recipe}


def check_cleanup_identity(
    *, nvidia_pids=lifecycle._nvidia_smi_compute_pids, signaller=procident.signal_verified
) -> list[Check]:
    checks: list[Check] = []
    me = procident.identity(os.getpid())
    if me is None or not me.boot_id:
        return [
            Check("cleanup.identity", "FAIL", "cannot read this process's own identity from /proc")
        ]
    checks.append(
        Check(
            "cleanup.identity",
            "PASS",
            f"pid {me.pid} starttime {me.starttime} boot {me.boot_id[:8]} uid {me.uid}",
        )
    )
    parent = procident.identity(os.getppid())
    ok_anc = parent is not None and procident.is_descendant_of(os.getpid(), os.getppid())
    checks.append(
        Check(
            "cleanup.ancestry",
            "PASS" if ok_anc else "FAIL",
            "descendant test resolves this process under its parent"
            if ok_anc
            else "ancestry walk failed",
        )
    )
    log: list[dict] = []
    own = signaller(me, signal.SIGTERM, log)
    checks.append(
        Check(
            "cleanup.refuses_self",
            "PASS" if own == "refused" else "FAIL",
            f"signal_verified(own pid) -> {own}",
            {"log": log},
        )
    )
    log = []
    init = procident.identity(1)
    if init is not None:
        r = signaller(init, signal.SIGTERM, log)
        checks.append(
            Check(
                "cleanup.refuses_pid1",
                "PASS" if r == "refused" else "FAIL",
                f"-> {r}",
                {"log": log},
            )
        )
    if parent is not None:
        stale = procident.ProcessIdentity(
            parent.pid,
            parent.starttime + 1,
            parent.boot_id,
            parent.uid,
            parent.ppid,
            parent.name,
            parent.cmd,
        )
        log = []
        r = signaller(stale, signal.SIGTERM, log)
        checks.append(
            Check(
                "cleanup.detects_identity_change",
                "PASS" if r in ("changed", "refused") else "FAIL",
                f"stale starttime on the parent pid -> {r} (nothing signalled)",
                {"log": log},
            )
        )
    pids = nvidia_pids()
    checks.append(
        Check(
            "cleanup.gpu_query",
            "FAIL" if pids is None else "PASS",
            "nvidia-smi compute-app query unavailable"
            if pids is None
            else f"nvidia-smi compute-app query works; {len(pids)} compute pid(s) now",
            {"pids": pids},
        )
    )
    return checks


def check_log_parsers(reference_log: Path | None) -> list[Check]:
    if reference_log is None:
        return [Check("parsers.reference_log", "INFO", "no reference server log supplied")]
    text = Path(reference_log).read_text(errors="replace")
    lines = text.splitlines()
    checks: list[Check] = []
    api = lifecycle.API_SERVER_PID_RE.findall(text)
    engines = lifecycle.ENGINE_PID_RE.findall(text)
    roles = {role.split("_")[0] for role, _ in engines}
    ok = bool(api) and "EngineCore" in roles and "Worker" in roles
    checks.append(
        Check(
            "parsers.pids",
            "PASS" if ok else "FAIL",
            f"API server pid(s) {sorted(set(api))}, engine roles {sorted(roles)}, "
            f"{len({p for _, p in engines})} distinct engine/worker pids",
        )
    )
    order = []
    for name, rx in (
        ("stop_accepted", lifecycle.STOP_ACCEPTED_RE),
        ("capture_ended", lifecycle.CAPTURE_ENDED_RE),
        ("stream_written", lifecycle.STREAM_RE),
        ("import_start", lifecycle.IMPORT_START_RE),
        ("import_progress", lifecycle.IMPORT_PROGRESS_RE),
        ("stop_returned", lifecycle.STOP_RETURNED_RE),
    ):
        idx = next((i for i, line in enumerate(lines) if rx.search(line)), None)
        order.append((name, idx))
    missing = [n for n, i in order if i is None]
    idxs = [i for _, i in order if i is not None]
    in_order = idxs == sorted(idxs)
    checks.append(
        Check(
            "parsers.phases",
            "PASS" if not missing and in_order else "FAIL",
            "every phase line matched in order: " + ", ".join(f"{n}@{i}" for n, i in order)
            if not missing and in_order
            else f"missing {missing}; order {order}",
        )
    )
    return checks


def check_locations(cfg: ProfileConfig) -> list[Check]:
    checks: list[Check] = []
    policy = cfg.nsys.finalize_policy()
    try:
        policy.validate()
        checks.append(
            Check(
                "policy.valid",
                "PASS",
                f"soft {policy.soft_deadline_s:.0f} s, hard {policy.hard_deadline_s:.0f} s, stall "
                f"{policy.stall_timeout_s:.0f} s, max extension {policy.max_extension_s:.0f} s, "
                f"GPU-held ceiling {policy.gpu_held_ceiling_s}, action {policy.on_hard_deadline}, "
                f"preserve stream {policy.preserve_raw_stream}",
            )
        )
    except ValueError as error:
        checks.append(Check("policy.valid", "FAIL", str(error)))
    checks.append(
        Check(
            "policy.ceiling_set",
            "PASS" if policy.gpu_held_ceiling_s is not None else "FAIL",
            "gpu_held_ceiling_s is explicit"
            if policy.gpu_held_ceiling_s is not None
            else "gpu_held_ceiling_s is not set (an authorized capture needs an explicit ceiling)",
        )
    )
    log_dir = Path(cfg.log_dir)
    nsys_dir = log_dir / "nsys"
    for name in ("lifecycle.json", "lifecycle.control"):
        p = nsys_dir / name
        checks.append(
            Check(
                f"locations.{name}",
                "FAIL" if p.exists() else "PASS",
                f"{p} {'exists (previous run?)' if p.exists() else 'absent'}",
            )
        )
    checks.append(
        Check(
            "locations.log_dir",
            "FAIL" if log_dir.exists() else "PASS",
            f"{log_dir} {'exists' if log_dir.exists() else 'absent'}",
        )
    )
    stream_dir = Path(policy.stream_directory())
    if not stream_dir.is_dir():
        checks.append(Check("locations.stream_dir", "FAIL", f"{stream_dir} is not a directory"))
    else:
        import shutil

        free = shutil.disk_usage(stream_dir).free
        checks.append(
            Check(
                "locations.stream_dir",
                "PASS" if free >= policy.min_free_bytes_stream else "FAIL",
                f"{stream_dir}: {free / 2**30:.0f} GiB free "
                f"(floor {policy.min_free_bytes_stream / 2**30:.0f})",
            )
        )
        uid = os.getuid()
        leftovers = []
        for p in list(stream_dir.glob("nsys-report-*.qdstrm")) + list(
            stream_dir.glob("*.raw-preserve")
        ):
            try:
                if p.stat().st_uid == uid:
                    leftovers.append(str(p))
            except OSError:
                continue
        checks.append(
            Check(
                "locations.stream_leftovers",
                "FAIL" if leftovers else "PASS",
                f"own raw streams / preserve dirs in {stream_dir}: {leftovers or 'none'}",
            )
        )
    report_parent = log_dir
    while not report_parent.exists() and report_parent != report_parent.parent:
        report_parent = report_parent.parent
    import shutil

    free = shutil.disk_usage(report_parent).free
    checks.append(
        Check(
            "locations.report_dir",
            "PASS" if free >= policy.min_free_bytes_report else "FAIL",
            f"{report_parent}: {free / 2**30:.0f} GiB free "
            f"(floor {policy.min_free_bytes_report / 2**30:.0f})",
        )
    )
    return checks


# ------------------------------------------------------------------------- main
def run_preflight(
    cfg: ProfileConfig,
    *,
    pins: dict | None = None,
    reference_launch: Path | None = None,
    reference_server_log: Path | None = None,
    probe=_python_probe,
    nvidia_pids=lifecycle._nvidia_smi_compute_pids,
) -> tuple[list[Check], dict]:
    checks: list[Check] = []
    facts: dict = {}
    c, f = check_profiler(cfg, pins)
    checks += c
    facts.update(f)
    c, f = check_server(cfg, pins, probe=probe)
    checks += c
    facts.update(f)
    c, f = check_recipe(cfg, reference_launch)
    checks += c
    facts.update(f)
    checks += check_cleanup_identity(nvidia_pids=nvidia_pids)
    checks += check_log_parsers(reference_server_log)
    checks += check_locations(cfg)
    return checks, facts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument(
        "--pins", type=Path, help="pins JSON (from --write-pins of an accepted environment)"
    )
    ap.add_argument(
        "--reference-launch", type=Path, help="accepted *_launch.json to compare the recipe to"
    )
    ap.add_argument(
        "--reference-server-log", type=Path, help="accepted *_server.log for the parsers"
    )
    ap.add_argument("--json", type=Path, help="write the checks and facts here")
    ap.add_argument("--write-pins", type=Path, help="write the observed identities as a pins file")
    args = ap.parse_args(argv)

    from launcher.alignment_config import load_profile_config

    cfg = load_profile_config(args.config, require_python_runtime=False)
    pins = json.loads(args.pins.read_text()) if args.pins else None
    checks, facts = run_preflight(
        cfg,
        pins=pins,
        reference_launch=args.reference_launch,
        reference_server_log=args.reference_server_log,
    )
    for c in checks:
        print(f"{c.status:4s}  {c.name}: {c.detail}")
    failed = [c for c in checks if c.status == "FAIL"]
    print(f"# identity preflight: {len(checks)} checks, {len(failed)} FAIL")
    if args.json:
        args.json.write_text(
            json.dumps(
                {"checks": [asdict(c) for c in checks], "facts": facts}, indent=1, default=str
            )
        )
    if args.write_pins:
        pins_out = {k: facts[k] for k in ("nsys", "importer", "server") if k in facts}
        pins_out["recipe"] = facts.get("recipe")
        args.write_pins.write_text(json.dumps(pins_out, indent=1, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
