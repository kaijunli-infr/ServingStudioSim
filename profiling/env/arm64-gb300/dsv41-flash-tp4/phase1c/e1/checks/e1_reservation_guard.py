#!/usr/bin/env python3
"""Reservation budget of V4 at the launch and at the closeout (read-only; clock origin: the reservation pod's creation time
in reservation.json, written from kubectl before the launch).

    python3 -I checks/e1_reservation_guard.py eligible --reservation <run>/reservation.json --guard-s 4680 --run-budget-s 4260
    python3 -I checks/e1_reservation_guard.py deadline --reservation <run>/reservation.json --ceiling-s 5400
    python3 -I checks/e1_reservation_guard.py age --reservation <run>/reservation.json --max-age-s 420

eligible  the launch rule. The driver calls it before it starts any observer or the launcher and refuses to start unless
          the time remaining until the supervisor's reservation guard is at least the registered run budget:
              guard - age >= run budget        (equivalently: age <= guard - run budget; equality is permitted)
          The run budget is the registered worst case from this check to the end of the run (E1-V4-TIMELINE.md): the
          allowance from this check to the launcher's start, the GPU-held ceiling, the release after the ceiling acts,
          and the CPU-only processing after the release. Exit 0 permitted, 1 refused (too old, or a creation time in the
          future), 3 if the record cannot be read (the driver refuses as well).
deadline  seconds left until the reservation ceiling (the pod must be deleted by then). Exit 0 while time is left, 1 after
          the ceiling, 3 if the record cannot be read. Information for closeout and monitor; it decides nothing.
age       the plain age limit (kept for the tests of the first proposal; `eligible` is what the driver uses).
Stdlib only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import calendar
import json
import sys
import time
from pathlib import Path


def created_s(reservation: Path) -> float | None:
    try:
        return float(calendar.timegm(time.strptime(json.loads(reservation.read_text())["creation_timestamp"], "%Y-%m-%dT%H:%M:%SZ")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def age_s(reservation: Path, now: float) -> float | None:
    created = created_s(reservation)
    return None if created is None else now - created


def check(reservation: Path, max_age_s: float, now: float) -> tuple[int, str]:
    age = age_s(reservation, now)
    if age is None:
        return 3, f"INCONCLUSIVE reservation age: {reservation} unreadable or without a creation time"
    if age < 0:
        return 1, f"FAIL  reservation age {age:.0f} s is negative: the record's creation time is in the future"
    if age > max_age_s:
        return 1, f"FAIL  reservation age {age:.0f} s exceeds {max_age_s:.0f} s: launching now would not fit the reservation budget"
    return 0, f"PASS  reservation age {age:.0f} s <= {max_age_s:.0f} s at the launch"


def eligible(reservation: Path, guard_s: float, run_budget_s: float, now: float) -> tuple[int, str]:
    """The launch rule: remaining time to the guard >= the registered run budget."""
    age = age_s(reservation, now)
    if age is None:
        return 3, f"INCONCLUSIVE launch eligibility: {reservation} unreadable or without a creation time"
    if run_budget_s <= 0 or guard_s <= 0 or run_budget_s > guard_s:
        return 1, f"FAIL  launch eligibility: no launch window exists (guard {guard_s:.0f} s, run budget {run_budget_s:.0f} s)"
    if age < 0:
        return 1, f"FAIL  launch eligibility: reservation age {age:.1f} s is negative (the record's creation time is in the future)"
    remaining, latest = guard_s - age, guard_s - run_budget_s
    if remaining < run_budget_s:
        return 1, f"FAIL  launch eligibility: reservation age {age:.1f} s; {remaining:.1f} s remain until the reservation guard ({guard_s:.0f} s), fewer than the run budget {run_budget_s:.0f} s (latest permitted age {latest:.0f} s): REFUSED, nothing is started"
    return 0, f"PASS  launch eligibility: reservation age {age:.1f} s; {remaining:.1f} s remain until the reservation guard ({guard_s:.0f} s) >= run budget {run_budget_s:.0f} s (latest permitted age {latest:.0f} s, {latest - age:.1f} s to spare)"


def deadline(reservation: Path, ceiling_s: float, now: float) -> tuple[int, str]:
    created = created_s(reservation)
    if created is None:
        return 3, f"INCONCLUSIVE reservation deadline: {reservation} unreadable or without a creation time"
    left, by = created + ceiling_s - now, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created + ceiling_s))
    if left < 0:
        return 1, f"FAIL  reservation age {now - created:.0f} s: the {ceiling_s:.0f} s ceiling passed {-left:.0f} s ago (deletion was due by {by})"
    return 0, f"PASS  reservation age {now - created:.0f} s: delete the reservation by {by} ({left:.0f} s left of {ceiling_s:.0f} s)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("age")
    a.add_argument("--reservation", required=True, type=Path)
    a.add_argument("--max-age-s", required=True, type=float)
    e = sub.add_parser("eligible")
    e.add_argument("--reservation", required=True, type=Path)
    e.add_argument("--guard-s", required=True, type=float)
    e.add_argument("--run-budget-s", required=True, type=float)
    d = sub.add_parser("deadline")
    d.add_argument("--reservation", required=True, type=Path)
    d.add_argument("--ceiling-s", required=True, type=float)
    args = ap.parse_args(argv)
    now = time.time()
    if args.cmd == "age":
        rc, text = check(args.reservation, args.max_age_s, now)
    elif args.cmd == "eligible":
        rc, text = eligible(args.reservation, args.guard_s, args.run_budget_s, now)
    else:
        rc, text = deadline(args.reservation, args.ceiling_s, now)
    print(text)
    return rc


if __name__ == "__main__":
    sys.exit(main())
