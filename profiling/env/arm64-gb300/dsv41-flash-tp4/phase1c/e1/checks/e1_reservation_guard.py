#!/usr/bin/env python3
"""Age of the GPU reservation at the moment of the launch (V4 reservation budget, first part).

    python3 -I checks/e1_reservation_guard.py age --reservation <run>/reservation.json --max-age-s 600

The driver calls this immediately before the launcher line and refuses to start if the reservation is older than the
limit: a late launch would push the end of the run past the reservation ceiling. Exit 0 if the age is within the limit
(at the limit counts as within), 1 if it is older or negative, 3 if the record cannot be read. Stdlib only; read-only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import calendar
import json
import sys
import time
from pathlib import Path


def age_s(reservation: Path, now: float) -> float | None:
    try:
        text = json.loads(reservation.read_text())["creation_timestamp"]
        return now - calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ"))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def check(reservation: Path, max_age_s: float, now: float) -> tuple[int, str]:
    age = age_s(reservation, now)
    if age is None:
        return 3, f"INCONCLUSIVE reservation age: {reservation} unreadable or without a creation time"
    if age < 0:
        return 1, f"FAIL  reservation age {age:.0f} s is negative: the record's creation time is in the future"
    if age > max_age_s:
        return 1, f"FAIL  reservation age {age:.0f} s exceeds {max_age_s:.0f} s: launching now would not fit the reservation budget"
    return 0, f"PASS  reservation age {age:.0f} s <= {max_age_s:.0f} s at the launch"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("age")
    a.add_argument("--reservation", required=True, type=Path)
    a.add_argument("--max-age-s", required=True, type=float)
    args = ap.parse_args(argv)
    rc, text = check(args.reservation, args.max_age_s, time.time())
    print(text)
    return rc


if __name__ == "__main__":
    sys.exit(main())
