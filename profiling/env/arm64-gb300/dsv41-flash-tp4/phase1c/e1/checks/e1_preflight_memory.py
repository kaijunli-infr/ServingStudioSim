#!/usr/bin/env python3
"""V4 preflight host-memory check (read-only; exact, in kB; no truncation to whole GiB).

    python3 -I checks/e1_preflight_memory.py --meminfo /proc/meminfo --inputs E1-V4-SCALE-INPUTS.json [--cgroup-max max|<bytes>]

PASS if MemAvailable >= the registered preflight minimum (at the minimum counts as within), the minimum equals importer
ceiling + runtime floor + the registered allowance for the run, and no cgroup memory limit below the minimum applies.
Two result lines (host, cgroup). Exit 0 if both PASS, 1 on a FAIL, 3 if MemAvailable or the registered values cannot be
read. The runtime floor and the importer ceiling are enforced during the run by the memory sampler and the supervisor;
this check does not replace them. Stdlib only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import json
import sys
from pathlib import Path

KB_PER_GIB = 1048576


def mem_available_kb(text: str) -> int | None:
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "MemAvailable:" and parts[1].isdigit() and (len(parts) == 2 or parts[2] == "kB"):
            return int(parts[1])
    return None


def registered(inputs: dict) -> tuple[int, int, int, int] | None:
    try:
        c, mb = inputs["ceilings"], inputs["memory_budget"]
        vals = (c["host_mem_available_preflight_min_gb"], c["importer_rss_ceiling_gb"], c["host_mem_available_floor_gb"], mb["server_allowance_gib"])
    except (KeyError, TypeError):
        return None
    return vals if all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in vals) else None


def check(meminfo_text: str, inputs: dict, cgroup_max: str = "max") -> tuple[int, list[str]]:
    reg = registered(inputs)
    if reg is None:
        return 3, ["INCONCLUSIVE host memory: the registered minimum, importer ceiling, floor or allowance cannot be read from the inputs"]
    minimum, importer, floor, server = reg
    basis = f"importer ceiling {importer} + runtime floor {floor} + allowance for the run {server}"
    if minimum != importer + floor + server:
        return 1, [f"FAIL  host memory: the registered preflight minimum {minimum} GiB is not {basis} = {importer + floor + server}"]
    avail = mem_available_kb(meminfo_text)
    if avail is None:
        return 3, ["INCONCLUSIVE host memory: no MemAvailable line in kB"]
    need = minimum * KB_PER_GIB
    lines, rc = [], 0
    if avail >= need:
        lines.append(f"PASS  host MemAvailable {avail} kB ({avail / KB_PER_GIB:.2f} GiB) >= {minimum} GiB ({need} kB) = {basis}; margin {(avail - need) / KB_PER_GIB:.2f} GiB")
    else:
        lines.append(f"FAIL  host MemAvailable {avail} kB ({avail / KB_PER_GIB:.2f} GiB) < {minimum} GiB ({need} kB) = {basis}; short by {(need - avail) / KB_PER_GIB:.2f} GiB")
        rc = 1
    if cgroup_max == "max":
        lines.append(f"PASS  no cgroup memory limit applies to this session (memory.max max); required at least {minimum} GiB")
    elif cgroup_max.isdigit() and int(cgroup_max) >= minimum * 1024**3:
        lines.append(f"PASS  cgroup memory.max {cgroup_max} B ({int(cgroup_max) / 1024**3:.2f} GiB) >= {minimum} GiB")
    elif cgroup_max.isdigit():
        lines.append(f"FAIL  cgroup memory.max {cgroup_max} B ({int(cgroup_max) / 1024**3:.2f} GiB) < {minimum} GiB: the importer ceiling of {importer} GiB is not consistent with it")
        rc = 1
    else:
        lines.append(f"INCONCLUSIVE cgroup memory.max {cgroup_max!r} is neither 'max' nor a byte count")
        rc = rc or 3
    return rc, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meminfo", required=True, type=Path)
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--cgroup-max", default="max")
    a = ap.parse_args(argv)
    try:
        text, inputs = a.meminfo.read_text(), json.loads(a.inputs.read_text())
    except (OSError, ValueError) as error:
        print(f"INCONCLUSIVE host memory: {error}")
        return 3
    rc, lines = check(text, inputs, a.cgroup_max)
    print("\n".join(lines))
    return rc


if __name__ == "__main__":
    sys.exit(main())
