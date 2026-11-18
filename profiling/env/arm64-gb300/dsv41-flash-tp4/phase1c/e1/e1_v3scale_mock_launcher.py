#!/usr/bin/env python3
"""Mock launcher for the CPU-only end-to-end rehearsal of the V3 driver (rehearsal mode only; never V3 evidence).

    uv run --no-sync python e1_v3scale_mock_launcher.py launch <profile.json> --frontend-provenance BINARY-PROVENANCE.json

The driver starts this instead of `python -m launcher alignment profile` when, and only when, E1_V3S_MODE=rehearsal. It
uses no GPU and no Nsight: CUDA must be hidden or it refuses. What is real: the stop controller
(alignment.profiler.lifecycle.ProfilerStopController with the policy of the profile config: deadlines, control file,
identity-verified release action, stream preservation, lifecycle.json), process identities (alignment.profiler.procident)
and the prebuilt-executable contract with the build-tool guard (alignment.load_generator.runner). What is mocked: the
processes. They are plain Python sleepers arranged like the real ones and they print the log lines the controller reads:

  mock Nsight CLI (the recorded pid)  ─ mock API server ─ mock engine ─ two mock workers
  mock agent (detached: forks, parent exits, own session, parent pid 1) ─ mock importer(s)

The mock importer runs as `<python> <this file> importer --input-file <stream> --output-file <report>`, holds the stream
open, uses CPU for the import time and writes the report at the end. The mock API server, like the real one, stops its
engine at once on SIGTERM and then stays alive while a stop request is still open.

Scenario (E1_V3S_MOCK_SCENARIO): success | ambiguous (two importers) | importer_ballast (the importer holds
E1_V3S_MOCK_BALLAST_MB of memory) | storage_ballast (a file of E1_V3S_MOCK_STORAGE_BALLAST_MB appears in the run root during
the capture). Timings: E1_V3S_MOCK_{STARTUP,CAPTURE,FLUSH,IMPORT}_S.
"""

from __future__ import annotations

# ruff: noqa: E501

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ME = os.path.abspath(__file__)
SCENARIO = os.environ.get("E1_V3S_MOCK_SCENARIO", "success")


def env_f(name: str, default: float) -> float:
    return float(os.environ.get(f"E1_V3S_MOCK_{name}", default))


STARTUP_S, CAPTURE_S, FLUSH_S, IMPORT_S = env_f("STARTUP_S", 4), env_f("CAPTURE_S", 6), env_f("FLUSH_S", 5), env_f("IMPORT_S", 14)


def say(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def stamp() -> str:
    return time.strftime("%m-%d %H:%M:%S", time.gmtime())


# ---------------------------------------------------------------- mock process roles
def role_worker(i: str) -> int:
    say(f"(Worker_TP{i} pid={os.getpid()}) INFO {stamp()} [mock] worker ready\n")
    parent = os.getppid()
    while os.getppid() == parent:
        time.sleep(0.5)
    return 0


def role_engine() -> int:
    say(f"(EngineCore pid={os.getpid()}) INFO {stamp()} [mock] engine ready\n")
    kids = [subprocess.Popen([sys.executable, ME, "worker", str(i)]) for i in range(2)]
    parent = os.getppid()
    while os.getppid() == parent:
        time.sleep(0.5)
    for k in kids:
        k.kill()
    return 0


def role_apiserver(work: str) -> int:
    w = Path(work)
    tag = f"(APIServer pid={os.getpid()})"
    say(f"{tag} INFO {stamp()} [mock] API server starting\n")
    engine = subprocess.Popen([sys.executable, ME, "engine"])
    state = {"term": False}

    def on_term(signum, frame):
        state["term"] = True

    signal.signal(signal.SIGTERM, on_term)
    t0, announced, stopping, stopped = time.time(), False, False, False
    while True:
        if not announced and time.time() - t0 >= STARTUP_S:
            say("INFO:     Application startup complete.\n")
            (w / "ready").write_text(str(os.getpid()))
            announced = True
        if not stopping and (w / "cmd_stop").exists():
            say(f"{tag} INFO {stamp()} [api_router.py:31] Stopping profiler...\n")
            stopping = True
        if stopping and not stopped and (w / "stopped").exists():
            say(f"{tag} INFO {stamp()} [api_router.py:33] Profiler stopped.\n")
            stopped = True
        if state["term"]:
            (w / "term").write_text("1")
            say(f"{tag} INFO {stamp()} [launcher.py:128] [shutdown] API server: shutdown triggered\n")
            engine.kill()  # the engine goes at once (its workers follow), as in V2
            engine.wait()
            if stopping and not stopped:
                say("INFO:     Shutting down\nINFO:     Waiting for connections to close. (CTRL+C to force quit)\n")
                while True:  # a stop request is still open: stay until SIGKILL, as the real server does
                    time.sleep(0.5)
            say("INFO:     Shutting down\nINFO:     Application shutdown complete.\n")
            return 0
        time.sleep(0.2)


def role_importer(args: list[str]) -> int:
    stream, report = args[args.index("--input-file") + 1], args[args.index("--output-file") + 1]
    write = "--no-report" not in args
    ballast = bytearray(int(float(os.environ.get("E1_V3S_MOCK_BALLAST_MB", "0")) * 1024 * 1024))
    for i in range(0, len(ballast), 4096):
        ballast[i] = 1
    with open(stream, "rb") as fh:
        fh.read(1 << 20)
        end, x = time.time() + IMPORT_S + (0 if write else 5), 0  # in the ambiguous scenario the second process outlives the first
        while time.time() < end:  # the importer's own CPU time advances throughout
            x = (x * 1103515245 + 12345) % (1 << 31)
        if write:
            tmp = report + ".part"
            with open(tmp, "wb") as out:
                out.write(os.urandom(1 << 20))
            os.replace(tmp, report)
    return 0


def role_agent(args: list[str]) -> int:
    """Detached like the Nsight agent: fork, the parent exits, the child is its own session leader with parent pid 1."""
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    stream, report = args[-2], args[-1]
    kids = [subprocess.Popen([sys.executable, ME, "importer", "--input-file", stream, "--output-file", report, "--force-overwrite"])]
    if SCENARIO == "ambiguous":
        kids.append(subprocess.Popen([sys.executable, ME, "importer", "--input-file", stream, "--output-file", report, "--force-overwrite", "--no-report"]))
    for k in kids:
        k.wait()
    return 0


def role_nsys(work: str, report: str) -> int:
    """Mock Nsight CLI: its stdout is the server log. Starts the target, and on the stop command flushes, 'imports' and exits after the target."""
    w = Path(work)
    api = subprocess.Popen([sys.executable, ME, "apiserver", work])
    while not (w / "cmd_stop").exists():
        if api.poll() is not None:
            return 0
        time.sleep(0.2)
    time.sleep(1.0)  # the API server's "Stopping profiler" line comes first, as in the real log
    say("Capture range started in the application.\nCapture range ended in the application.\n")
    time.sleep(FLUSH_S)
    stream = f"/tmp/nsys-report-e2e{os.getpid():x}.qdstrm"
    with open(stream, "wb") as fh:
        fh.write(os.urandom(2 << 20))
    # the mock CLI does not keep the stream open: every mock process runs under the same interpreter, and a second holder of the
    # stream would look like a second importer; the controller takes the path from the "Generating" line below
    say(f"Generating '{stream}'\n\n\n")
    say(f"[1/1] [0%                          ] {Path(report).name}Processing events...\n")
    subprocess.Popen([sys.executable, ME, "agent", "--start-agent", "--session-name", f"profile-{os.getpid()}", stream, report]).wait()
    t0 = time.time()
    while not (os.path.isfile(report) and os.path.getsize(report) > 0):
        pct = min(99, int(100 * (time.time() - t0) / max(IMPORT_S, 1)))
        say(f"\r[1/1] [{'=' * (pct // 4)}{pct}%{' ' * (25 - pct // 4)}] {Path(report).name}")
        time.sleep(0.5)
        if time.time() - t0 > IMPORT_S + 60:
            break
    say(f"\r[1/1] [========================100%] {Path(report).name}\n")
    try:
        os.unlink(stream)  # Nsight removes its own stream; the controller's hard link keeps the bytes
    except OSError:
        pass
    (w / "stopped").write_text("1")
    api.wait()
    say(f"Generated:\n\t{report}\n")
    return 0


# ---------------------------------------------------------------- the mock launcher itself
def launch(config: Path, provenance: Path) -> int:
    if os.environ.get("E1_V3S_MODE") != "rehearsal" or os.environ.get("CUDA_VISIBLE_DEVICES", "unset") != "":
        print("mock launcher refuses: it runs only with E1_V3S_MODE=rehearsal and CUDA_VISIBLE_DEVICES=''")
        return 9
    sys.path.insert(0, str(Path(ME).parents[6]))  # the repository root (the launcher proper is started with -m from there)
    from alignment.load_generator import runner as frontend
    from alignment.profiler import lifecycle

    cfg = json.loads(config.read_text())
    if not cfg["name"].endswith("_e2e") or "_e1_v3scale_e2e/" not in cfg["log_dir"]:
        print("mock launcher refuses: the profile is not the rehearsal copy (name / log_dir)")
        return 9
    log_dir, name = Path(cfg["log_dir"]), cfg["name"]
    nsys_dir, vllm = log_dir / "nsys", log_dir / "vllm"
    for d in (nsys_dir, vllm, log_dir / "load_generator"):
        d.mkdir(parents=True, exist_ok=True)
    frontend.forbid_build_tools()
    frontend.write_frontend_record(log_dir, frontend.verify_prebuilt_session_runner(provenance))
    print(f"[mock] prebuilt session_runner verified; build-tool guard {frontend.build_tool_guard_state()}", flush=True)
    work = log_dir / "mock"
    work.mkdir(exist_ok=True)
    server_log, stem = vllm / f"{name}_server.log", nsys_dir / name
    report = str(stem.with_suffix(".nsys-rep"))
    n = cfg["nsys"]
    policy = lifecycle.FinalizePolicy(
        soft_deadline_s=n["finalize_soft_deadline_s"], hard_deadline_s=n["finalize_hard_deadline_s"], stall_timeout_s=n["finalize_stall_timeout_s"], on_hard_deadline=n["finalize_on_hard_deadline"],
        session_close_timeout_s=n["finalize_session_close_timeout_s"], max_extension_s=n["finalize_max_extension_s"], gpu_held_ceiling_s=n["gpu_held_ceiling_s"], preserve_raw_stream=n["preserve_raw_stream"])
    with server_log.open("ab") as log_fh:
        proc = subprocess.Popen([sys.executable, ME, "nsys", str(work), report], stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True)

    def stop_request() -> None:
        (work / "cmd_stop").write_text("1")
        api = int((work / "ready").read_text())
        def gone() -> bool:
            try:
                return open(f"/proc/{api}/stat").read().rsplit(") ", 1)[-1][0] == "Z"
            except OSError:
                return True

        while not gone():
            if (work / "stopped").exists() and not (work / "term").exists():
                return  # the import finished and the server answered
            time.sleep(0.2)
        raise RuntimeError("mock /stop_profile failed: Remote end closed connection without response")

    controller = lifecycle.ProfilerStopController(name=name, server_log=server_log, report_stem=stem, status_path=nsys_dir / "lifecycle.json", control_path=nsys_dir / "lifecycle.control", policy=policy, stop_request=stop_request, nsys_pid=proc.pid)
    rc = 1
    try:
        while not (work / "ready").exists():
            if proc.poll() is not None:
                raise RuntimeError("mock server exited during startup")
            time.sleep(0.2)
        controller.discover_server_processes()
        print("[profile] frontend ready; measurement begins", flush=True)
        t_cap, end = time.time(), time.time() + CAPTURE_S
        while time.time() < end:
            controller.poll()  # the capture phase: the launcher only polls (no control file, no deadline)
            if SCENARIO == "storage_ballast" and time.time() - t_cap > 3 and not (work / "ballast.bin").exists():
                (work / "ballast.bin").write_bytes(os.urandom(int(env_f("STORAGE_BALLAST_MB", 8)) << 20))  # pushes the run root over the rehearsal ceiling
            time.sleep(1.0)
        print("[profile] workload done: {'mock': true}", flush=True)
        controller.request_stop()
        phase = controller.wait_for_report()
        print(f"[mock] stop ended in phase {phase.value}", flush=True)
        rc = 0 if phase == lifecycle.Phase.REPORT_WRITTEN else 1
    finally:
        controller.shutdown(proc)
    exporter = os.environ.get("E1_V3S_MOCK_NSYS")
    if rc == 0 and exporter:
        subprocess.run([exporter, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(stem.with_suffix(".sqlite")), report], check=True)
    after = frontend.verify_after_launch(log_dir)
    print(f"[mock] executable after the launch unchanged={after and after.get('unchanged')}; build-tool guard {after and after.get('build_tool_guard')}", flush=True)
    if after and (after.get("build_tool_guard") or {}).get("blocked"):
        return 1
    return rc


def main(argv: list[str]) -> int:
    role = argv[1] if len(argv) > 1 else ""
    if role == "worker":
        return role_worker(argv[2])
    if role == "engine":
        return role_engine()
    if role == "apiserver":
        return role_apiserver(argv[2])
    if role == "importer":
        return role_importer(argv[2:])
    if role == "agent":
        return role_agent(argv[2:])
    if role == "nsys":
        return role_nsys(argv[2], argv[3])
    if role == "launch" and len(argv) == 5 and argv[3] == "--frontend-provenance":
        return launch(Path(argv[2]), Path(argv[4]))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
