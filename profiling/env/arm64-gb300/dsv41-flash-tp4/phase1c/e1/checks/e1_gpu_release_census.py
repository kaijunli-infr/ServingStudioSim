#!/usr/bin/env python3
"""Independent GPU-release census (V1 gate L6c): compute-process list and per-GPU memory state.

    python3 -I checks/e1_gpu_release_census.py --out <run>/gpu_release_census_<tag>.json [--max-used-mib 64]
        [--registry <owned.tsv>] [--label t0]

Asks the driver (nvidia-smi), not the lifecycle controller: `--query-compute-apps` for every process
holding a context and `--query-gpu` for memory.used per GPU, and resolves each compute pid against /proc
(comm, uid, starttime) so the record stands on its own. Verdict `released` = no compute process on any GPU
and memory.used ≤ --max-used-mib on every GPU (idle GB300s read 0–3 MiB). Exit 0 when released, 1 when
not, 2 when nvidia-smi is unavailable. Stdlib only; read-only on the system except for --out.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import datetime as dt
import json
import socket
import subprocess
import sys
import time
from pathlib import Path


def _smi(query: str, fields: str) -> tuple[int, list[list[str]], str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-{query}={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return 127, [], str(error)
    rows = [[c.strip() for c in line.split(",")] for line in out.stdout.splitlines() if line.strip()]
    return out.returncode, rows, out.stderr.strip()


def _proc_identity(pid: int) -> dict:
    ident: dict = {"pid": pid, "exists": False}
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        ident["exists"] = True
        ident["comm"] = stat[stat.index("(") + 1 : stat.rindex(")")]
        ident["starttime"] = int(stat[stat.rindex(")") + 2 :].split()[19])
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("Uid:"):
                ident["uid"] = int(line.split()[1])
                break
    except (OSError, ValueError, IndexError):
        pass
    return ident


def census(max_used_mib: int, registry: Path | None) -> dict:
    now = time.time()
    rc_apps, apps, err_apps = _smi("compute-apps", "gpu_uuid,pid,process_name,used_memory")
    rc_gpu, gpus, err_gpu = _smi("gpu", "index,uuid,memory.used,memory.total")
    owned: dict[int, str] = {}
    if registry and registry.is_file():
        for line in registry.read_text().splitlines():
            parts = line.split("\t")
            if parts and parts[0].isdigit():
                owned[int(parts[0])] = parts[1] if len(parts) > 1 else "?"
    compute = []
    for row in apps:
        if len(row) < 4 or not row[1].isdigit():
            continue
        pid = int(row[1])
        compute.append(
            {
                "gpu_uuid": row[0],
                "pid": pid,
                "process_name": row[2],
                "used_mib": int(row[3]) if row[3].isdigit() else None,
                "proc": _proc_identity(pid),
                "owned_label": owned.get(pid),
            }
        )
    gpu_state = []
    for row in gpus:
        if len(row) < 4:
            continue
        gpu_state.append(
            {
                "index": int(row[0]),
                "uuid": row[1],
                "memory_used_mib": int(row[2]) if row[2].isdigit() else None,
                "memory_total_mib": int(row[3]) if row[3].isdigit() else None,
            }
        )
    available = rc_apps == 0 and rc_gpu == 0 and gpu_state
    busy = [g for g in gpu_state if (g["memory_used_mib"] or 0) > max_used_mib]
    released = bool(available) and not compute and not busy
    reason = (
        "nvidia-smi unavailable"
        if not available
        else "no compute process on any GPU; memory.used within the idle bound on every GPU"
        if released
        else f"{len(compute)} compute process(es); {len(busy)} GPU(s) above {max_used_mib} MiB"
    )
    return {
        "schema_version": 1,
        "host": socket.gethostname(),
        "at": now,
        "at_iso": dt.datetime.fromtimestamp(now, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nvidia_smi_rc": {"compute_apps": rc_apps, "gpu": rc_gpu},
        "nvidia_smi_stderr": (err_apps or err_gpu)[:300],
        "max_used_mib": max_used_mib,
        "compute_apps": compute,
        "gpus": gpu_state,
        "verdict": {"available": bool(available), "released": released, "reason": reason},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-used-mib", type=int, default=64)
    ap.add_argument("--registry", type=Path)
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)
    rec = census(args.max_used_mib, args.registry)
    rec["label"] = args.label
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rec, indent=1) + "\n")
    v = rec["verdict"]
    mem = " ".join(str(g["memory_used_mib"]) for g in rec["gpus"])
    print(
        f"# gpu release census {args.label} on {rec['host']} at {rec['at_iso']}: "
        f"{'RELEASED' if v['released'] else 'NOT RELEASED' if v['available'] else 'UNAVAILABLE'} — {v['reason']}; "
        f"compute pids {[c['pid'] for c in rec['compute_apps']]}; memory.used MiB per GPU: {mem}"
    )
    return 0 if v["released"] else (2 if not v["available"] else 1)


if __name__ == "__main__":
    sys.exit(main())
