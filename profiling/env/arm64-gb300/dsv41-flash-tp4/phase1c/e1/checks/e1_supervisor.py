#!/usr/bin/env python3
"""Registered run supervisor for V3: consumes the resource-breach markers and the samplers' heartbeats and gives one
bounded, identity-verified abort response. It sends no signal itself.

    python3 -I checks/e1_supervisor.py --run <run root>/<pass> --registry <run root>/pids/owned.tsv --out supervisor.json
        --control-line "<registered release line>" [--gpu-held-ceiling-s 2700] [--ack-s 10] [--stale-s 60] [--cleanup-grace 30]
        [--interval 1] [--max-s 3600] [--stop-file FILE]

Started by the driver through `e1_pids.py spawn` into its OWN registry (pids/supervisor.tsv), so that the pinned
cleanup of pids/owned.tsv, which it may have to run, is not a cleanup of its own ancestry.

Inputs, every --interval seconds:
  breach markers   <run>/MEMORY-BREACH, <run>/STORAGE-BREACH (written by the samplers)
  second channel   the samplers' summaries (mem_watch.json, storage_watch.json): a breach recorded there without a
                   marker file is a breach all the same ("marker missing")
  heartbeats       age of the last line of mem_observations.jsonl and storage_observations.jsonl, and of
                   gpu_observations.jsonl until the controller has recorded the GPU release; older than --stale-s, or
                   absent --stale-s after the supervisor started: that independent monitor is no longer effective
  GPU-held ceiling the controller enforces its ceiling only once the stop has been requested; if the ceiling passes (counted
                   from the launch time in lifecycle.json) and no stop has been requested, that is a cause here

One response, on the first sample that shows any of these (all causes present at that moment are recorded together):
  R1  finalisation in progress (stop requested, phase not terminal, no release requested yet, lifecycle record fresh):
      append the registered `release-gpus` line to lifecycle.control. The controller, unchanged, verifies the server's
      identity and stops the server only. The supervisor waits --ack-s for the controller to record the line.
  R2  any other phase, or R1 not acknowledged in time: run the pinned registry cleanup,
      `e1_pids.py cleanup --registry <owned.tsv> --execute --grace <s> --kill9`, which signals only registered processes
      and their descendants whose identity (pid, start time, boot id, uid) still matches.
Neither response can reach the Nsight agent or the importer: they are not in the registry and not descendants of a
registered process, and the controller never signals Nsight. A guessed process is never signalled.

The record (--out) has every cause with the time it was written and the time it was seen, the response, its times, the
acknowledgement or the cleanup result, and the heartbeats; it is rewritten every sample. Exit 0 without a response,
5 after one. Stdlib only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TERMINAL = ("report_written", "stop_failed")
MARKERS = {"MEMORY-BREACH": "mem_watch.json", "STORAGE-BREACH": "storage_watch.json"}
HEARTBEATS = {"memory sampler": "mem_observations.jsonl", "storage sampler": "storage_observations.jsonl", "GPU sampler": "gpu_observations.jsonl"}


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def marker_causes(run: Path, now: float) -> list[dict]:
    """Breaches, from the marker files and, independently, from the samplers' own summaries."""
    causes = []
    for marker, summary in MARKERS.items():
        path, lines = run / marker, []
        if path.exists():
            try:
                lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            except (OSError, ValueError):
                lines = [{"kind": "unreadable marker"}]
            for b in lines or [{"kind": "empty marker"}]:
                causes.append({"cause": "breach", "source": marker, "kind": b.get("kind"), "value_gb": b.get("value_gb"), "limit_gb": b.get("limit_gb"), "breach_at": b.get("at"), "marker_mtime": path.stat().st_mtime, "seen_at": now})
        else:
            for b in (load(run / summary) or {}).get("breaches", []):
                causes.append({"cause": "breach", "source": f"{summary} (marker file missing)", "kind": b.get("kind"), "value_gb": b.get("value_gb"), "limit_gb": b.get("limit_gb"), "breach_at": b.get("at"), "marker_mtime": None, "seen_at": now})
    return causes


def heartbeat_state(run: Path, lc: dict | None, started_at: float, now: float, stale_s: float) -> tuple[dict, list[dict]]:
    """Age of each sampler's last sample; a sampler that is required and silent for more than stale_s is a cause."""
    beats, causes = {}, []
    gpu_done = bool(((lc or {}).get("gpu") or {}).get("released_at"))
    for name, fname in HEARTBEATS.items():
        path = run / fname
        try:
            age = now - path.stat().st_mtime
        except OSError:
            age = None
        required = not (name == "GPU sampler" and gpu_done)
        beats[name] = {"file": fname, "age_s": None if age is None else round(age, 2), "required": required}
        silent = (age is None and now - started_at > stale_s) or (age is not None and age > stale_s)
        if required and silent:
            causes.append({"cause": "monitor_lost", "source": name, "kind": "no sample file" if age is None else "last sample too old", "age_s": None if age is None else round(age, 1), "limit_s": stale_s, "seen_at": now})
    return beats, causes


def ceiling_cause(lc: dict | None, now: float, ceiling_s: float | None) -> list[dict]:
    """The GPU-held ceiling passed before any stop request: the controller is not yet watching it."""
    if ceiling_s is None or not lc or lc.get("launch_started_at") is None or lc.get("stop_requested_at") is not None:
        return []
    held = now - lc["launch_started_at"]
    return [{"cause": "gpu_held_ceiling_before_stop", "source": "lifecycle.json launch_started_at", "kind": "no stop requested when the ceiling passed", "held_s": round(held, 1), "limit_s": ceiling_s, "seen_at": now}] if held > ceiling_s else []


def finalisation_active(lc: dict | None, now: float, fresh_s: float) -> bool:
    """R1 is possible only while the controller is in its finalisation loop: that is the only time it reads the control file."""
    if not lc or lc.get("stop_requested_at") is None or lc.get("phase") in TERMINAL:
        return False
    if (lc.get("gpu") or {}).get("release_requested_at") is not None:
        return False
    updated = lc.get("updated_at")
    return updated is not None and now - updated <= fresh_s


def acknowledged(lc: dict | None, line: str) -> dict | None:
    return next((c for c in (lc or {}).get("control_commands", []) if c.get("command") == line and c.get("result") == "release requested"), None)


def run_cleanup(registry: Path, grace: float, log: Path, runner=subprocess.run) -> dict:
    cmd = [sys.executable, "-I", str(HERE / "e1_pids.py"), "cleanup", "--registry", str(registry), "--execute", "--grace", str(grace), "--kill9", "--log", str(log)]
    t0 = time.time()
    try:
        p = runner(cmd, capture_output=True, text=True, timeout=grace + 120)
        rc, tail = p.returncode, (p.stdout or "").splitlines()[-1:] or [""]
    except (OSError, subprocess.TimeoutExpired) as error:
        rc, tail = None, [f"cleanup could not be run: {error}"]
    return {"command": cmd[2:], "started_at": t0, "ended_at": time.time(), "rc": rc, "result_line": tail[0][:300], "log": str(log)}


def supervise(run: Path, registry: Path, out: Path, control_line: str, *, gpu_held_ceiling_s: float | None = None, ack_s: float = 10.0, stale_s: float = 60.0, cleanup_grace: float = 30.0, interval: float = 1.0, max_s: float = 3600.0, stop_file: Path | None = None,
              now=time.time, sleep=time.sleep, cleanup=run_cleanup) -> int:
    lc_path, control = run / "profile" / "nsys" / "lifecycle.json", run / "profile" / "nsys" / "lifecycle.control"
    started = now()
    rec: dict = {"supervisor": "e1_supervisor.py", "host": socket.gethostname(), "pid": os.getpid(), "started_at": started, "run": str(run), "registry": str(registry), "control_line": control_line,
                 "limits": {"ack_s": ack_s, "stale_s": stale_s, "cleanup_grace_s": cleanup_grace, "interval_s": interval, "gpu_held_ceiling_s": gpu_held_ceiling_s}, "samples": 0, "causes": [], "response": None, "monitors_effective": True, "signals_sent_by_supervisor": 0, "complete": False}

    def write() -> None:
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(rec, indent=1) + "\n")
        os.replace(tmp, out)

    end = started + max_s
    while True:
        t = now()
        lc = load(lc_path)
        beats, lost = heartbeat_state(run, lc, started, t, stale_s)
        causes = marker_causes(run, t) + lost + ceiling_cause(lc, t, gpu_held_ceiling_s)
        rec.update({"samples": rec["samples"] + 1, "last_check_at": t, "heartbeats": beats, "phase_seen": (lc or {}).get("phase")})
        if causes and rec["response"] is None:
            rec["causes"] = causes
            rec["monitors_effective"] = not lost
            resp: dict = {"decided_at": t, "phase": (lc or {}).get("phase"), "stop_requested": bool(lc and lc.get("stop_requested_at") is not None), "steps": []}
            rec["response"] = resp
            if finalisation_active(lc, t, max(5.0, 3 * interval)):
                with control.open("a") as fh:
                    fh.write(control_line + "\n")
                step = {"kind": "R1 control line", "written_at": now(), "ack_deadline_s": ack_s, "acknowledged": False}
                resp["steps"].append(step)
                write()
                while now() - step["written_at"] <= ack_s:
                    ack = acknowledged(load(lc_path), control_line)
                    if ack:
                        step.update({"acknowledged": True, "acknowledged_at": ack.get("at"), "latency_s": round(ack["at"] - step["written_at"], 3) if ack.get("at") else None, "controller_phase": ack.get("phase")})
                        break
                    sleep(min(0.5, interval))
            if not resp["steps"] or not resp["steps"][-1].get("acknowledged"):
                why = "the controller is not in its finalisation loop" if not resp["steps"] else f"the control line was not acknowledged within {ack_s} s"
                (run.parent / "cpu").mkdir(parents=True, exist_ok=True)
                step = {"kind": "R2 registry cleanup", "why": why, **cleanup(registry, cleanup_grace, run.parent / "cpu" / "supervisor_cleanup.jsonl")}
                resp["steps"].append(step)
            resp["completed_at"] = now()
            resp["seconds_from_first_cause_seen"] = round(resp["completed_at"] - t, 3)
            resp["outcome"] = "release requested through the controller" if resp["steps"][-1]["kind"].startswith("R1") else f"registry cleanup rc {resp['steps'][-1].get('rc')}"
            write()
            if resp["steps"][-1]["kind"].startswith("R2"):
                break  # the driver tree is gone; nothing is left to supervise
        elif causes:
            rec["later_causes"] = [c for c in causes if (c["cause"], c["source"], c.get("kind")) not in {(x["cause"], x["source"], x.get("kind")) for x in rec["causes"]}] or rec.get("later_causes", [])
        write()
        if (stop_file is not None and stop_file.exists()) or now() >= end:
            break
        sleep(interval)
    rec["complete"], rec["ended_at"] = True, now()
    write()
    return 5 if rec["response"] else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--registry", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--control-line", required=True)
    ap.add_argument("--gpu-held-ceiling-s", type=float)
    ap.add_argument("--ack-s", type=float, default=10.0)
    ap.add_argument("--stale-s", type=float, default=60.0)
    ap.add_argument("--cleanup-grace", type=float, default=30.0)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=3600.0)
    ap.add_argument("--stop-file", type=Path)
    args = ap.parse_args(argv)
    if not args.control_line.startswith("release-gpus ") or len(args.control_line.split(None, 1)) < 2:
        ap.error("--control-line must be a `release-gpus <reason>` line")
    return supervise(args.run, args.registry, args.out, args.control_line, gpu_held_ceiling_s=args.gpu_held_ceiling_s, ack_s=args.ack_s, stale_s=args.stale_s, cleanup_grace=args.cleanup_grace, interval=args.interval, max_s=args.max_s, stop_file=args.stop_file)


if __name__ == "__main__":
    sys.exit(main())
