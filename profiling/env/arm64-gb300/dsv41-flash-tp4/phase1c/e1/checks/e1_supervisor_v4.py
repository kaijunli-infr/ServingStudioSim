#!/usr/bin/env python3
"""Run supervisor for V4: the causes, the release line, the acknowledgement rule and the registry cleanup of the V3
supervisor (checks/e1_supervisor.py, imported, not modified), one further cause, the reservation guard, and a response
latch under which an earlier response cannot leave the reservation guard unenforced.

    python3 -I checks/e1_supervisor_v4.py <every argument of e1_supervisor.py>
        --reservation <run>/reservation.json --reservation-guard-s 4680 [--release-s 150]

Causes (every --interval seconds): breach markers and the samplers' summaries, a lost monitor, the GPU-held ceiling before
a stop request (all three as in e1_supervisor.py), and the reservation guard: the run has not ended (the driver has not
written the stop file) when more than --reservation-guard-s seconds have passed since the reservation's creation time
(reservation.json). An unreadable reservation record or one without a creation time is a guard cause as well.

Responses (the two of e1_supervisor.py):
  R1  the registered `release-gpus` line through the controller, which stops the server only (identity-verified);
      possible only while the controller is in its finalisation loop; acknowledged within --ack-s or R2 follows
  R2  the pinned registry cleanup of pids/owned.tsv (e1_pids.py cleanup --execute --kill9): ends the registered tree

Latch (single-action coordination):
  * R1 is written at most once. R2 runs at most once and is final: the supervisor ends after it.
  * R2 is never started while R1 is waiting for its acknowledgement; the two are sequential, never concurrent.
  * First cause seen: if the reservation guard is among the causes, R2 (the guard requires the run to END; the release
    line would only release the GPUs). Otherwise R1 during finalisation, R2 in every other phase or without an
    acknowledgement.
  * After an acknowledged R1 the supervisor keeps sampling. Other causes are recorded (`later_causes`) and do not act.
    The reservation guard does: when it passes and the run has not ended, R2 runs (`response.escalation`), whether the
    earlier release completed, is still in progress or stalled. Until then a release that the controller has not
    recorded as completed within --release-s (150 s = its 60 s server grace + the 90 s release bound) is reported
    (`release_after_response.stalled`; monitor: ABORT-CONDITION) and left to the controller's own deadlines.
Neither response can reach the Nsight agent or the importer: they are not in the registry and not descendants of a
registered process, and the controller never signals Nsight. The supervisor sends no signal itself.

The record (--out) is rewritten every sample; an R2 step is recorded as `running` before the cleanup starts. Exit 0
without a response, 5 after one. Stdlib only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import calendar
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _base():
    spec = importlib.util.spec_from_file_location("e1_supervisor_for_v4", HERE / "e1_supervisor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def created_at(path: Path) -> float | None:
    try:
        return float(calendar.timegm(time.strptime(json.loads(path.read_text())["creation_timestamp"], "%Y-%m-%dT%H:%M:%SZ")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def reservation_cause(reservation: Path, guard_s: float, now: float) -> list[dict]:
    """The reservation guard has passed (strictly more than guard_s after the creation time), or cannot be evaluated."""
    created = created_at(reservation)
    if created is None:
        return [{"cause": "reservation_guard", "source": str(reservation.name), "kind": "reservation record unreadable or without a creation time", "limit_s": guard_s, "seen_at": now}]
    age = now - created
    if age > guard_s:
        return [{"cause": "reservation_guard", "source": f"{reservation.name} creation_timestamp", "kind": "run not ended when the reservation guard passed", "reservation_age_s": round(age, 1), "limit_s": guard_s, "seen_at": now}]
    return []


def supervise(run: Path, registry: Path, out: Path, control_line: str, reservation: Path, guard_s: float, *, gpu_held_ceiling_s: float | None = None, ack_s: float = 10.0, stale_s: float = 60.0, cleanup_grace: float = 30.0,
              interval: float = 1.0, max_s: float = 3600.0, stop_file: Path | None = None, release_s: float = 150.0, now=time.time, sleep=time.sleep, cleanup=None) -> int:
    base = _base()  # a private instance; nothing in it is replaced
    cleanup = cleanup or base.run_cleanup
    lc_path, control = run / "profile" / "nsys" / "lifecycle.json", run / "profile" / "nsys" / "lifecycle.control"
    started = now()
    rec: dict = {"supervisor": "e1_supervisor_v4.py", "helpers": "e1_supervisor.py (causes, acknowledgement, registry cleanup; unmodified)", "host": socket.gethostname(), "pid": os.getpid(), "started_at": started, "run": str(run), "registry": str(registry),
                 "control_line": control_line, "limits": {"ack_s": ack_s, "stale_s": stale_s, "cleanup_grace_s": cleanup_grace, "interval_s": interval, "gpu_held_ceiling_s": gpu_held_ceiling_s, "reservation_guard_s": guard_s, "release_s": release_s},
                 "latch": {"release_lines_written": 0, "registry_cleanups_started": 0}, "samples": 0, "causes": [], "response": None, "monitors_effective": True, "signals_sent_by_supervisor": 0, "complete": False}
    latch = rec["latch"]

    def write() -> None:
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(rec, indent=1) + "\n")
        os.replace(tmp, out)

    def key(c: dict) -> tuple:
        return (c["cause"], c["source"], c.get("kind"))

    def registry_cleanup(resp: dict, why: str, decided: float) -> None:
        """R2, at most once: recorded as running before it starts, then with its result."""
        if latch["registry_cleanups_started"]:
            return
        latch["registry_cleanups_started"] += 1
        (run.parent / "cpu").mkdir(parents=True, exist_ok=True)
        head = {"kind": "R2 registry cleanup", "why": why, "decided_at": decided}
        resp["steps"].append({**head, "state": "running"})
        write()
        resp["steps"][-1] = {**head, **cleanup(registry, cleanup_grace, run.parent / "cpu" / "supervisor_cleanup.jsonl"), "state": "finished"}

    def release_line(resp: dict) -> bool:
        """R1, at most once: the registered line through the controller; True if acknowledged within ack_s."""
        if latch["release_lines_written"]:
            return False
        latch["release_lines_written"] += 1
        with control.open("a") as fh:
            fh.write(control_line + "\n")
        step = {"kind": "R1 control line", "written_at": now(), "ack_deadline_s": ack_s, "acknowledged": False}
        resp["steps"].append(step)
        write()
        while now() - step["written_at"] <= ack_s:
            ack = base.acknowledged(base.load(lc_path), control_line)
            if ack:
                step.update({"acknowledged": True, "acknowledged_at": ack.get("at"), "latency_s": round(ack["at"] - step["written_at"], 3) if ack.get("at") else None, "controller_phase": ack.get("phase"), "evidence": ack.get("evidence"), "api_server_sigterm": ack.get("api_server_sigterm")})
                return True
            sleep(min(0.5, interval))
        return False

    def cleaned(resp: dict) -> bool:
        return any(s["kind"].startswith("R2") for s in resp["steps"])

    end = started + max_s
    while True:
        t = now()
        lc = base.load(lc_path)
        beats, lost = base.heartbeat_state(run, lc, started, t, stale_s)
        guard = reservation_cause(reservation, guard_s, t)
        causes = base.marker_causes(run, t) + lost + base.ceiling_cause(lc, t, gpu_held_ceiling_s) + guard
        rec.update({"samples": rec["samples"] + 1, "last_check_at": t, "heartbeats": beats, "phase_seen": (lc or {}).get("phase")})
        if lost:
            rec["monitors_effective"] = False
        resp = rec["response"]
        if resp is None and causes:
            rec["causes"] = causes
            finalising = base.finalisation_active(lc, t, max(5.0, 3 * interval))
            resp = rec["response"] = {"decided_at": t, "phase": (lc or {}).get("phase"), "stop_requested": bool(lc and lc.get("stop_requested_at") is not None), "steps": []}
            if guard:
                registry_cleanup(resp, "the reservation guard requires the run to end" + ("; the release line would release the GPUs only" if finalising else "; the controller is not in its finalisation loop"), t)
            elif finalising:
                if not release_line(resp):
                    registry_cleanup(resp, f"the control line was not acknowledged within {ack_s} s", now())
            else:
                registry_cleanup(resp, "the controller is not in its finalisation loop", t)
            resp["completed_at"] = now()
            resp["seconds_from_first_cause_seen"] = round(resp["completed_at"] - t, 3)
            resp["outcome"] = f"registry cleanup rc {resp['steps'][-1].get('rc')}" if cleaned(resp) else "release requested through the controller"
        elif resp is not None and causes:
            seen = {key(c) for c in rec["causes"]}
            rec["later_causes"] = [c for c in causes if key(c) not in seen] or rec.get("later_causes", [])
            if guard and not cleaned(resp):  # the earlier response does not end the run; the guard does
                gpu = (lc or {}).get("gpu") or {}
                resp["escalation"] = {"decided_at": t, "causes": guard, "earlier_step": resp["steps"][-1]["kind"], "phase": (lc or {}).get("phase"), "gpu_released_at": gpu.get("released_at"), "release_requested_at": gpu.get("release_requested_at")}
                registry_cleanup(resp, "the reservation guard passed after the earlier release request and the run has not ended", t)
                resp["escalation"]["completed_at"] = now()
                resp["escalation"]["seconds_from_guard_seen"] = round(resp["escalation"]["completed_at"] - t, 3)
                resp["outcome"] = f"registry cleanup rc {resp['steps'][-1].get('rc')} at the reservation guard, after the release requested through the controller"
        if resp is not None and not cleaned(resp):
            gpu, ack_at = (lc or {}).get("gpu") or {}, resp["steps"][-1].get("acknowledged_at") or resp["steps"][-1].get("written_at")
            rec["release_after_response"] = {"gpu_released_at": gpu.get("released_at"), "phase": (lc or {}).get("phase"), "seconds_since_acknowledgement": None if ack_at is None else round(t - ack_at, 1),
                                             "stalled": gpu.get("released_at") is None and ack_at is not None and t - ack_at > release_s, "limit_s": release_s}
        write()
        if resp is not None and cleaned(resp):
            break  # the registered tree is gone; nothing is left to supervise
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
    ap.add_argument("--reservation", required=True, type=Path)
    ap.add_argument("--reservation-guard-s", required=True, type=float)
    ap.add_argument("--gpu-held-ceiling-s", type=float)
    ap.add_argument("--ack-s", type=float, default=10.0)
    ap.add_argument("--stale-s", type=float, default=60.0)
    ap.add_argument("--cleanup-grace", type=float, default=30.0)
    ap.add_argument("--release-s", type=float, default=150.0)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=3600.0)
    ap.add_argument("--stop-file", type=Path)
    args = ap.parse_args(argv)
    if not args.control_line.startswith("release-gpus ") or len(args.control_line.split(None, 1)) < 2:
        ap.error("--control-line must be a `release-gpus <reason>` line")
    if args.reservation_guard_s <= 0:
        ap.error("--reservation-guard-s must be positive")
    return supervise(args.run, args.registry, args.out, args.control_line, args.reservation, args.reservation_guard_s, gpu_held_ceiling_s=args.gpu_held_ceiling_s, ack_s=args.ack_s, stale_s=args.stale_s,
                     cleanup_grace=args.cleanup_grace, interval=args.interval, max_s=args.max_s, stop_file=args.stop_file, release_s=args.release_s)


if __name__ == "__main__":
    sys.exit(main())
