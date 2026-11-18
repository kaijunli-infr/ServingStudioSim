#!/usr/bin/env python3
"""V2 forced-release trigger and independent importer observer (registered via checks/e1_pids.py spawn).

    python3 -I checks/e1_release_probe.py --nsys-dir <run>/profile/nsys --inputs E1-V2-FR-INPUTS.json
        --out <run>/probe.json --log <run>/probe.log

Reads lifecycle.json only to learn the phase and the recorded Nsight identity; everything else comes from /proc.
1. Waits for the first observation of phase `importing`.
2. Verifies the Nsight (importer) identity: pid alive with the recorded start time. Takes two or more samples of that
   process tree (state, CPU ticks) until the import is confirmed in progress (CPU time or reported progress advanced
   while the phase stayed `importing`), for at most --confirm-s seconds.
3. Appends the one pre-registered line from the inputs file to lifecycle.control. If the phase is already terminal,
   or the Nsight process is not alive, it writes NOTHING and says so (the validation is then INCONCLUSIVE).
4. Waits for the controller's API-server SIGTERM time and samples the importer's identity and state after it.
It never sends a signal. Stdlib only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import json
import os
import sys
import time
from pathlib import Path

TERMINAL = ("report_written", "stop_failed")


def read_status(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def proc_stat(pid: int, proc: Path = Path("/proc")) -> dict | None:
    try:
        stat = (proc / str(pid) / "stat").read_text()
    except OSError:
        return None
    try:
        rest = stat[stat.rindex(")") + 2 :].split()
        return {"pid": pid, "comm": stat[stat.index("(") + 1 : stat.rindex(")")], "state": rest[0], "ppid": int(rest[1]), "cpu_ticks": int(rest[11]) + int(rest[12]), "starttime": int(rest[19])}
    except (ValueError, IndexError):
        return None


def tree(root: int, proc: Path = Path("/proc")) -> list[dict]:
    """The process `root` and its descendants (state and CPU ticks each)."""
    stats = {}
    for entry in proc.iterdir():
        if entry.name.isdigit():
            st = proc_stat(int(entry.name), proc)
            if st:
                stats[st["pid"]] = st
    out, frontier = [], [root]
    while frontier:
        pid = frontier.pop()
        if pid in stats and stats[pid] not in out:
            out.append(stats[pid])
            frontier.extend(p for p, st in stats.items() if st["ppid"] == pid)
    return out


def sample(status: dict | None, proc: Path = Path("/proc")) -> dict:
    nsys = (status or {}).get("nsys") or {}
    pid, recorded = nsys.get("pid"), (nsys.get("identity") or {}).get("starttime")
    procs = tree(pid, proc) if pid is not None else []
    me = next((p for p in procs if p["pid"] == pid), None)
    return {
        "at": time.time(),
        "phase": (status or {}).get("phase"),
        "import_progress_pct": (status or {}).get("import_progress_pct"),
        "nsys_pid": pid,
        "nsys_starttime_recorded": recorded,
        "nsys_starttime_proc": me["starttime"] if me else None,
        "identity_verified": bool(me) and recorded is not None and me["starttime"] == recorded,
        "nsys_state": me["state"] if me else None,
        "tree": [{k: p[k] for k in ("pid", "comm", "state", "cpu_ticks")} for p in procs],
        "tree_cpu_ticks": sum(p["cpu_ticks"] for p in procs),
    }


def in_progress(a: dict, b: dict) -> bool:
    """Both samples see `importing` with the verified importer alive, and work advanced in between."""
    if not (a["phase"] == b["phase"] == "importing" and a["identity_verified"] and b["identity_verified"]):
        return False
    pa, pb = a["import_progress_pct"], b["import_progress_pct"]
    return b["tree_cpu_ticks"] > a["tree_cpu_ticks"] or (pa is not None and pb is not None and pb > pa)


def run(nsys_dir: Path, line: str, out: Path, log, *, poll_s=0.5, confirm_s=3.0, gap_s=0.4, wait_s=3600.0, after_s=(0.3, 2.0, 5.0), proc=Path("/proc"), sleep=time.sleep) -> int:
    status_path, control = nsys_dir / "lifecycle.json", nsys_dir / "lifecycle.control"
    record: dict = {"source": "e1_release_probe.py: /proc of the recorded Nsight pid, read by the probe; lifecycle.json only for phase and identity", "control_line": line, "started_at": time.time(), "samples": []}

    def save(result: str) -> None:
        record["result"] = result
        out.write_text(json.dumps(record, indent=1) + "\n")
        log(f"[probe] {result}")

    end = time.time() + wait_s
    status = None
    while time.time() < end:
        status = read_status(status_path)
        phase = (status or {}).get("phase")
        if phase in TERMINAL:
            record["saw_phase"] = phase
            save("NO TRIGGER: the stop was already terminal before phase importing was observed (import completed before the forced release → INCONCLUSIVE)")
            return 1
        if phase == "importing":
            break
        sleep(poll_s)
    else:
        save("NO TRIGGER: gave up waiting for phase importing")
        return 2
    record["saw_phase"], record["saw_at"] = "importing", time.time()
    first = sample(status, proc)
    record["samples"].append(first)
    if not first["identity_verified"]:
        save("NO TRIGGER: the recorded Nsight process is not alive with the recorded start time")
        return 1
    confirmed, deadline = False, time.time() + confirm_s
    while time.time() < deadline:
        sleep(gap_s)
        cur = sample(read_status(status_path), proc)
        record["samples"].append(cur)
        if cur["phase"] in TERMINAL or not cur["identity_verified"]:
            record["import_in_progress_confirmed"] = False
            save(f"NO TRIGGER: the import ended (phase {cur['phase']}, importer alive={cur['identity_verified']}) before the forced release → INCONCLUSIVE")
            return 1
        if in_progress(first, cur):
            confirmed = True
            break
    record["import_in_progress_confirmed"] = confirmed
    record["nsys_pid"], record["nsys_alive"], record["identity_verified"] = first["nsys_pid"], True, True
    with control.open("a") as fh:
        fh.write(line + "\n")
    record["wrote_trigger_at"] = time.time()
    out.write_text(json.dumps(record, indent=1) + "\n")
    log(f"[probe] trigger line written; import in progress confirmed={confirmed}")
    # after the controller's SIGTERM to the API server: importer identity and state, from /proc
    sig_at, end = None, time.time() + 60
    while time.time() < end and sig_at is None:
        st = read_status(status_path) or {}
        sig_at = ((st.get("gpu") or {}).get("api_server_sigterm") or {}).get("at")
        if sig_at is None:
            sleep(0.2)
    record["api_server_sigterm_at_from_controller"] = sig_at
    record["after_signal"] = []
    if sig_at is not None:
        t0 = time.time()
        for offset in after_s:
            sleep(max(0.0, offset - (time.time() - t0)))
            record["after_signal"].append(sample(read_status(status_path), proc))
    save("trigger written" + ("" if sig_at is not None else "; the controller's SIGTERM time was not seen within 60 s"))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nsys-dir", required=True, type=Path)
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--log", required=True, type=Path)
    args = ap.parse_args(argv)
    line = json.loads(args.inputs.read_text())["trigger"]["control_line"]

    def log(msg: str) -> None:
        with args.log.open("a") as fh:
            fh.write(f"{msg} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")

    log(f"[probe] start pid {os.getpid()} watching {args.nsys_dir}/lifecycle.json")
    return run(args.nsys_dir, line, args.out, log)


if __name__ == "__main__":
    sys.exit(main())
