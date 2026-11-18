#!/usr/bin/env python3
"""Host identity cross-check (controller record vs run log vs reservation vs GPU observations).

    python3 -I checks/e1_host_identity.py --run <run root>/<pass> [--reservation reservation.json]
        [--observations gpu_observations.jsonl]

The lifecycle record carries the host the launcher ran on (`host`, `host_identity`). It must be a well-formed host
name and agree with the three independent sources: the driver's `start … on <host>` line in run.log, the node of the
reservation pod (reservation.json, written from kubectl before the launch), and the host of every GPU census /
observation. A missing source is INCONCLUSIVE, a malformed or conflicting one is FAIL. Stdlib only; read-only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import glob
import json
import re
import sys
from pathlib import Path

HOST_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$")


def well_formed(host) -> bool:
    return isinstance(host, str) and len(host) <= 253 and bool(HOST_RE.match(host))


def run_log_host(run: Path) -> str | None:
    try:
        m = re.search(r"^\[[^\]]+\] start \S+ on (\S+)", (run / "run.log").read_text(errors="replace"), re.M)
    except OSError:
        return None
    return m.group(1) if m else None


def gpu_observation_hosts(run: Path, observations: Path | None) -> list:
    hosts = []
    for f in sorted(glob.glob(str(run / "gpu_release_census_*.json"))):
        try:
            hosts.append(json.loads(Path(f).read_text()).get("host"))
        except (OSError, ValueError):
            hosts.append(None)
    for src in [observations, run / "gpu_observations.jsonl"]:
        if src and Path(src).is_file():
            for line in Path(src).read_text().splitlines():
                if line.strip():
                    try:
                        hosts.append(json.loads(line).get("host"))
                    except ValueError:
                        hosts.append(None)
    return hosts


def cross_check(lifecycle: dict | None, log_host: str | None, reservation: dict | None, gpu_hosts: list) -> dict:
    """→ {"status": PASS|FAIL|INCONCLUSIVE, "detail": str, "sources": {...}}"""
    lc_host = (lifecycle or {}).get("host")
    ident = (lifecycle or {}).get("host_identity") or {}
    sources = {
        "lifecycle.host": lc_host,
        "lifecycle.host_identity.hostname": ident.get("hostname"),
        "run_log": log_host,
        "reservation.node": (reservation or {}).get("node"),
        "gpu_observations": sorted({str(h) for h in gpu_hosts}) if gpu_hosts else None,
    }
    problems, missing = [], []
    if lc_host is None:
        missing.append("lifecycle record has no host")
    elif not well_formed(lc_host):
        problems.append(f"lifecycle host is malformed: {lc_host!r}")
    if lc_host is not None and ident.get("hostname") != lc_host:
        problems.append(f"lifecycle host {lc_host!r} != host_identity.hostname {ident.get('hostname')!r}")
    if lc_host is not None and not ident.get("boot_id"):
        problems.append("host_identity has no boot_id")
    for name, value in (("run log", log_host), ("reservation node", (reservation or {}).get("node"))):
        if value is None:
            missing.append(f"{name} host not available")
        elif not well_formed(value):
            problems.append(f"{name} host is malformed: {value!r}")
        elif well_formed(lc_host) and value != lc_host:
            problems.append(f"{name} host {value!r} conflicts with the lifecycle host {lc_host!r}")
    if reservation is not None and reservation.get("phase") not in (None, "Running"):
        problems.append(f"reservation pod phase is {reservation.get('phase')!r}, not Running")
    if not gpu_hosts:
        missing.append("no GPU observation carries a host")
    else:
        bad = sorted({str(h) for h in gpu_hosts if not well_formed(h)})
        other = sorted({h for h in gpu_hosts if well_formed(h) and well_formed(lc_host) and h != lc_host})
        if bad:
            problems.append(f"GPU observation host(s) missing or malformed: {bad}")
        if other:
            problems.append(f"GPU observation host(s) {other} conflict with the lifecycle host {lc_host!r}")
    status = "FAIL" if problems else ("INCONCLUSIVE" if missing else "PASS")
    detail = "; ".join(problems + missing) if (problems or missing) else f"launch host {lc_host} (boot {str(ident.get('boot_id'))[:8]}) agrees with the run log, the reservation node and {len(gpu_hosts)} GPU observation(s)"
    return {"status": status, "detail": detail, "sources": sources}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--reservation", type=Path)
    ap.add_argument("--observations", type=Path)
    args = ap.parse_args(argv)

    def load(p):
        try:
            return json.loads(Path(p).read_text())
        except (OSError, ValueError, TypeError):
            return None

    res = cross_check(load(args.run / "profile" / "nsys" / "lifecycle.json"), run_log_host(args.run), load(args.reservation or args.run / "reservation.json"), gpu_observation_hosts(args.run, args.observations))
    print(f"{res['status']:12s} host identity: {res['detail']}; sources {res['sources']}")
    return {"PASS": 0, "FAIL": 1}.get(res["status"], 3)


if __name__ == "__main__":
    sys.exit(main())
