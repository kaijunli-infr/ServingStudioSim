#!/usr/bin/env python3
"""Temporary- and persistent-storage sampler (V3; registered via checks/e1_pids.py spawn; never deletes or signals).

    python3 -I checks/e1_storage_watch.py --out storage_observations.jsonl --summary storage_watch.json
        --group temporary=/tmp/nsys-report-*.qdstrm,/tmp/*.raw-preserve,/tmp/v3scale-recovery-* --group persistent=<run root>
        [--ceiling-gb temporary=10 --ceiling-gb persistent=8] [--breach-file STORAGE-BREACH]
        [--interval 2] [--max-s 3600] [--stop-file FILE]

Every --interval seconds, for each group: the bytes of all regular files under the paths its patterns match (apparent
size; a file reached twice, e.g. the hard-linked preserved stream, is counted once per group by device and inode) and
the file count; plus the free space of each group's filesystem. One JSON line per sample with the time and the host. The
summary holds each group's peak with its time, the number of samples and the largest gap between samples, so a reader
can tell a measured peak from an unmeasured one. On the first sample above a group's ceiling the sampler writes
--breach-file and records the breach; it takes no action. Exit 0 without a breach, 4 with one. Stdlib only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import glob
import json
import os
import socket
import sys
import time
from pathlib import Path

GB = 1e9


def measure(patterns: list[str]) -> dict:
    seen, total, files, roots = set(), 0, 0, 0

    def add(path: str) -> None:
        nonlocal total, files
        try:
            st = os.stat(path, follow_symlinks=False)
        except OSError:
            return
        if not os.path.isfile(path) or os.path.islink(path) or (st.st_dev, st.st_ino) in seen:
            return
        seen.add((st.st_dev, st.st_ino))
        total += st.st_size
        files += 1

    for pat in patterns:
        for root in sorted(glob.glob(pat)):
            roots += 1
            if os.path.isdir(root) and not os.path.islink(root):
                for d, _dirs, names in os.walk(root):
                    for n in names:
                        add(os.path.join(d, n))
            else:
                add(root)
    return {"bytes": total, "files": files, "roots": roots}


def free_bytes(patterns: list[str]) -> int | None:
    p = Path(patterns[0].split("*")[0] or "/")
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        st = os.statvfs(p)
        return st.f_bavail * st.f_frsize
    except OSError:
        return None


def summarize(samples: list[dict], breaches: list[dict], groups: dict, ceilings: dict, interval: float, host: str) -> dict:
    ats = [s["at"] for s in samples]
    out = {"sampler": "e1_storage_watch.py", "host": host, "samples": len(samples), "first_at": ats[0] if ats else None, "last_at": ats[-1] if ats else None, "interval_s": interval,
           "max_gap_s": max((b - a for a, b in zip(ats, ats[1:])), default=None), "size_basis": "apparent size of regular files, each inode once per group", "groups": {}, "breaches": breaches}
    for name, pats in groups.items():
        vals = [(s["groups"][name]["bytes"], s["at"]) for s in samples]
        peak = max(vals, key=lambda v: v[0], default=(None, None))  # the first time the peak was reached
        out["groups"][name] = {"patterns": pats, "ceiling_gb": ceilings.get(name), "peak_bytes": peak[0], "peak_at": peak[1], "last_bytes": vals[-1][0] if vals else None,
                               "min_free_bytes": min((s["groups"][name]["free_bytes"] for s in samples if s["groups"][name].get("free_bytes") is not None), default=None)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--group", action="append", required=True, metavar="NAME=GLOB[,GLOB...]")
    ap.add_argument("--ceiling-gb", action="append", default=[], metavar="NAME=GB")
    ap.add_argument("--breach-file", type=Path)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--max-s", type=float, default=3600.0)
    ap.add_argument("--stop-file", type=Path)
    args = ap.parse_args(argv)
    groups = {}
    for g in args.group:
        name, _, pats = g.partition("=")
        if not name or not pats:
            ap.error(f"--group needs NAME=GLOB[,GLOB...]: {g!r}")
        groups[name] = [p for p in pats.split(",") if p]
    ceilings = {}
    for c in args.ceiling_gb:
        name, _, v = c.partition("=")
        if name not in groups:
            ap.error(f"--ceiling-gb names an unknown group: {c!r}")
        ceilings[name] = float(v)
    host = socket.gethostname()
    samples, breaches, hit = [], [], set()
    end = time.monotonic() + args.max_s
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as fh:
        while True:
            s = {"at": time.time(), "host": host, "groups": {}}
            for name, pats in groups.items():
                m = measure(pats)
                m["free_bytes"] = free_bytes(pats)
                s["groups"][name] = m
                if name in ceilings and m["bytes"] > ceilings[name] * GB and name not in hit:
                    hit.add(name)
                    b = {"at": s["at"], "kind": f"{name}_storage_above_ceiling", "value_gb": round(m["bytes"] / GB, 3), "limit_gb": ceilings[name]}
                    breaches.append(b)
                    if args.breach_file:
                        with args.breach_file.open("a") as bf:
                            bf.write(json.dumps(b) + "\n")
            samples.append(s)
            fh.write(json.dumps(s) + "\n")
            fh.flush()
            stop = (args.stop_file is not None and args.stop_file.exists()) or time.monotonic() >= end
            if stop or len(samples) % 10 == 0:
                args.summary.write_text(json.dumps({**summarize(samples, breaches, groups, ceilings, args.interval, host), "complete": stop}, indent=1) + "\n")
            if stop:
                break
            time.sleep(args.interval)
    return 4 if breaches else 0


if __name__ == "__main__":
    sys.exit(main())
