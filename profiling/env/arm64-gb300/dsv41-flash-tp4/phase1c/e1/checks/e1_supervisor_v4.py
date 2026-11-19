#!/usr/bin/env python3
"""Run supervisor for V4: the V3 supervisor (checks/e1_supervisor.py, imported and run unchanged) with one further cause,
the reservation guard, so that the 90-minute reservation ceiling is enforced on the tray and not only measured afterwards.

    python3 -I checks/e1_supervisor_v4.py <every argument of e1_supervisor.py>
        --reservation <run>/reservation.json --reservation-guard-s 4680

Reservation guard: if the run has not ended (the driver has not written the stop file) when more than
--reservation-guard-s seconds have passed since the reservation's creation time (reservation.json, written from kubectl
before the launch), that is a cause. It is handled like every other cause of the unchanged supervisor: one response;
during finalisation the registered release line through the controller, otherwise the pinned, identity-verified registry
cleanup of pids/owned.tsv. With the driver's launch rule (no launch unless guard - reservation age >= the run budget of
4210 s, so age <= 470 s; E1-V4-TIMELINE.md) and the GPU-held ceiling of 2700 s, the guard at 4680 s can only fall into the
CPU-only phase after the GPU release, where the response is the registry cleanup: the driver and the launcher's export are ended, the preserved stream stays, the run is a
FAIL, and closeout and the deletion of the reservation fit into the remaining 720 s.

An unreadable reservation record or one without a creation time is itself a cause (the guard cannot work without it).
The supervisor still sends no signal itself. Exit codes as e1_supervisor.py. Stdlib only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import calendar
import importlib.util
import json
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


def supervise(run: Path, registry: Path, out: Path, control_line: str, reservation: Path, guard_s: float, **kw) -> int:
    base = _base()
    ceiling_only, clock = base.ceiling_cause, kw.get("now", time.time)

    def ceiling_and_reservation(lc, now, ceiling_s):
        return ceiling_only(lc, now, ceiling_s) + reservation_cause(reservation, guard_s, clock() if now is None else now)

    base.ceiling_cause = ceiling_and_reservation  # on this private module instance only; e1_supervisor.py is not modified
    return base.supervise(run, registry, out, control_line, **kw)


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
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=3600.0)
    ap.add_argument("--stop-file", type=Path)
    args = ap.parse_args(argv)
    if not args.control_line.startswith("release-gpus ") or len(args.control_line.split(None, 1)) < 2:
        ap.error("--control-line must be a `release-gpus <reason>` line")
    if args.reservation_guard_s <= 0:
        ap.error("--reservation-guard-s must be positive")
    return supervise(args.run, args.registry, args.out, args.control_line, args.reservation, args.reservation_guard_s, gpu_held_ceiling_s=args.gpu_held_ceiling_s, ack_s=args.ack_s, stale_s=args.stale_s,
                     cleanup_grace=args.cleanup_grace, interval=args.interval, max_s=args.max_s, stop_file=args.stop_file)


if __name__ == "__main__":
    sys.exit(main())
