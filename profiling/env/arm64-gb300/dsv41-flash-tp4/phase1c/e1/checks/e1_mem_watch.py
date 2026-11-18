#!/usr/bin/env python3
"""Host-memory sampler with a bounded safety policy (V3; registered via checks/e1_pids.py spawn; never signals anything).

    python3 -I checks/e1_mem_watch.py --out mem_observations.jsonl --summary mem_watch.json
        [--importer-exe <QdstrmImporter> --report <out.nsys-rep> (--stream <in.qdstrm> | --nsys-dir <run>/profile/nsys)]
        [--min-available-gb 64] [--max-importer-rss-gb 256] [--breach-file MEMORY-BREACH]
        [--interval 1] [--max-s 3600] [--stop-file FILE]

Every --interval seconds: MemTotal and MemAvailable from /proc/meminfo and, when exactly one process is this session's
importer (identified as in checks/e1_import_probe.py: pinned executable, session stream or report), its VmRSS and VmHWM
from /proc/<pid>/status. One JSON line per sample with the time and the host.

Policy (bounded, explicit): a breach is MemAvailable below --min-available-gb, or the importer's resident memory above
--max-importer-rss-gb. On the first breach of each kind the sampler writes --breach-file (the monitor prints an
ABORT-CONDITION for it) and records the breach; it takes no action itself. What the operator may do is in the packet's
abort procedure. Exit 0 without a breach, 4 with one. Stdlib only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import importlib.util
import json
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KB_PER_GB = 1024 * 1024


def _probe():
    spec = importlib.util.spec_from_file_location("e1_import_probe", HERE / "e1_import_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def meminfo(path: Path = Path("/proc/meminfo")) -> dict:
    out = {}
    try:
        for line in path.read_text().splitlines():
            k, _, v = line.partition(":")
            if k in ("MemTotal", "MemAvailable"):
                out[k] = int(v.split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return out


def proc_mem(pid: int, proc: Path = Path("/proc")) -> dict | None:
    try:
        vals = {}
        for line in (proc / str(pid) / "status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                vals[line.split(":")[0]] = int(line.split()[1])
        return vals or None
    except (OSError, ValueError, IndexError):
        return None


def take_sample(probe, args, stream: str | None, *, host: str, now=time.time, ambiguous: bool = False) -> dict:
    mi = meminfo(args.meminfo)
    s: dict = {"at": now(), "host": host, "mem_total_kb": mi.get("MemTotal"), "mem_available_kb": mi.get("MemAvailable"), "importer": None, "candidates": None}
    if args.importer_exe:
        cands = probe.find_importers(args.importer_exe, stream, args.report, proc=args.proc, uid=args.uid)
        s["candidates"] = len(cands)
        if len(cands) == 1 and not ambiguous:  # after an ambiguous sample no survivor is adopted as the importer
            mem = proc_mem(cands[0]["pid"], args.proc) or {}
            s["importer"] = {"pid": cands[0]["pid"], "starttime": cands[0]["starttime"], "rss_kb": mem.get("VmRSS"), "hwm_kb": mem.get("VmHWM"), "matched_by": cands[0]["matched_by"]}
    return s


def breaches_of(s: dict, min_available_gb: float, max_importer_rss_gb: float) -> list[dict]:
    out = []
    avail = s.get("mem_available_kb")
    if avail is not None and avail < min_available_gb * KB_PER_GB:
        out.append({"at": s["at"], "kind": "host_mem_available_below_floor", "value_gb": round(avail / KB_PER_GB, 2), "limit_gb": min_available_gb})
    imp = s.get("importer") or {}
    peak = max((v for v in (imp.get("rss_kb"), imp.get("hwm_kb")) if v is not None), default=None)
    if peak is not None and peak > max_importer_rss_gb * KB_PER_GB:
        out.append({"at": s["at"], "kind": "importer_rss_above_ceiling", "value_gb": round(peak / KB_PER_GB, 2), "limit_gb": max_importer_rss_gb, "pid": imp.get("pid")})
    return out


def summarize(samples: list[dict], breaches: list[dict], args, host: str) -> dict:
    ats = [s["at"] for s in samples]
    avail = [(s["mem_available_kb"], s["at"]) for s in samples if s.get("mem_available_kb") is not None]
    imp = [s for s in samples if s.get("importer")]
    rss = [v for s in imp for v in (s["importer"].get("rss_kb"), s["importer"].get("hwm_kb")) if v is not None]
    low = min(avail, default=(None, None))
    return {
        "sampler": "e1_mem_watch.py", "host": host, "samples": len(samples), "first_at": ats[0] if ats else None, "last_at": ats[-1] if ats else None,
        "interval_s": args.interval, "max_gap_s": max((b - a for a, b in zip(ats, ats[1:])), default=None),
        "mem_total_kb": next((s["mem_total_kb"] for s in samples if s.get("mem_total_kb")), None),
        "samples_without_meminfo": sum(s.get("mem_available_kb") is None for s in samples),
        "min_available_kb": low[0], "min_available_at": low[1],
        "importer_samples": len(imp), "importer_identities": sorted({(s["importer"]["pid"], s["importer"]["starttime"]) for s in imp}),
        "importer_first_at": imp[0]["at"] if imp else None, "importer_last_at": imp[-1]["at"] if imp else None,
        "importer_peak_rss_kb": max(rss, default=None), "ambiguous_samples": sum((s.get("candidates") or 0) > 1 for s in samples),
        "policy": {"min_available_gb": args.min_available_gb, "max_importer_rss_gb": args.max_importer_rss_gb, "action": "record + breach file only; the sampler signals nothing"},
        "breaches": breaches,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--importer-exe")
    ap.add_argument("--report")
    ap.add_argument("--stream")
    ap.add_argument("--nsys-dir", type=Path)
    ap.add_argument("--min-available-gb", type=float, default=64.0)
    ap.add_argument("--max-importer-rss-gb", type=float, default=256.0)
    ap.add_argument("--breach-file", type=Path)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=3600.0)
    ap.add_argument("--stop-file", type=Path)
    ap.add_argument("--meminfo", type=Path, default=Path("/proc/meminfo"), help=argparse.SUPPRESS)
    ap.add_argument("--proc", type=Path, default=Path("/proc"), help=argparse.SUPPRESS)
    ap.add_argument("--uid", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.importer_exe and not (args.stream or args.nsys_dir or args.report):
        ap.error("--importer-exe needs --report, --stream or --nsys-dir")
    probe, host = _probe(), socket.gethostname()
    samples, breaches, kinds = [], [], set()
    end = time.monotonic() + args.max_s
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as fh:
        while True:
            stream = args.stream
            if stream is None and args.nsys_dir is not None:
                try:
                    stream = (json.loads((args.nsys_dir / "lifecycle.json").read_text()).get("stream") or {}).get("path")
                except (OSError, ValueError):
                    stream = None
            s = take_sample(probe, args, stream, host=host, ambiguous=any((x.get("candidates") or 0) > 1 for x in samples))
            new = breaches_of(s, args.min_available_gb, args.max_importer_rss_gb)
            if new:
                s["breach"] = [b["kind"] for b in new]
            for b in new:
                if b["kind"] not in kinds:
                    kinds.add(b["kind"])
                    breaches.append(b)
                    if args.breach_file:
                        with args.breach_file.open("a") as bf:
                            bf.write(json.dumps(b) + "\n")
            samples.append(s)
            fh.write(json.dumps(s) + "\n")
            fh.flush()
            stop = (args.stop_file is not None and args.stop_file.exists()) or time.monotonic() >= end
            if stop or len(samples) % 10 == 0:
                args.summary.write_text(json.dumps({**summarize(samples, breaches, args, host), "complete": stop}, indent=1) + "\n")
            if stop:
                break
            time.sleep(args.interval)
    return 4 if breaches else 0


if __name__ == "__main__":
    sys.exit(main())
