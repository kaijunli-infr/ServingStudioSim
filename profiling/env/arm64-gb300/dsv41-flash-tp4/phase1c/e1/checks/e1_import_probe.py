#!/usr/bin/env python3
"""Importer-specific import-state probe (V3 preparation; replaces nothing that V2 froze).

    python3 -I checks/e1_import_probe.py --importer-exe <QdstrmImporter> --report <out.nsys-rep>
        (--stream <in.qdstrm> | --nsys-dir <run>/profile/nsys) --out probe.json --samples probe.jsonl --log probe.log
        [--nsys-pid PID (default: lifecycle.json nsys.pid)] [--nsys-log server.log] [--interval 0.5] [--min-progress 2] [--confirm-s 30] [--max-s 3600]
        [--control lifecycle.control --trigger-line "<registered line>"]

Why: the V2 probe confirmed "import in progress" from the CPU time of the recorded Nsight CLI's process tree. That tree
contains the server's workers and does NOT contain the importer: Nsight's importer (`QdstrmImporter --input-file <stream>
--output-file <report>`) is a child of the detached Nsight agent (`nsys --start-agent --session-name profile-<CLI pid>`,
parent pid 1, own session), not a descendant of the CLI. This probe

1. identifies the importer itself: a process of this uid whose executable is exactly the pinned importer and whose
   arguments (or open files) name this session's stream or report; exactly one must match;
2. accepts only evidence read from that process: its own CPU ticks and its own I/O counters (/proc/<pid>/stat, /proc/<pid>/io).
   The report's size and the progress figure the controller read from the Nsight log are recorded with their provenance
   but never count as importer progress, and no other process's CPU time is ever read into the decision;
3. writes every sample, timestamped, with the provenance of each field, to a JSONL file;
4. classifies the state as ACTIVE, NO_PROGRESS (alive, identity intact, nothing advanced in the last interval: temporary),
   COMPLETED (importer gone; report present, non-empty and stable over the settle samples; whether Nsight has logged
   "Generated:" is recorded, and the exit status, which /proc does not show, is stated as not observed), TERMINATED
   (importer gone or replaced and the report missing or empty), NOT_FOUND or AMBIGUOUS;
5. fails closed: "import active" is established only by --min-progress consecutive intervals of importer-specific
   advance; in trigger mode the registered line is written only then, after one more identity check, else withheld.
It never sends a signal. Stdlib only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ACTIVE, NO_PROGRESS, COMPLETED, TERMINATED, NOT_FOUND, AMBIGUOUS, SETTLING = "ACTIVE", "NO_PROGRESS", "COMPLETED", "TERMINATED", "NOT_FOUND", "AMBIGUOUS", "SETTLING"
DEAD_STATES = ("Z", "X", "x")
PROVENANCE = {
    "importer.state/ticks/starttime/ppid": "/proc/<importer pid>/stat, read by this probe",
    "importer.io": "/proc/<importer pid>/io, read by this probe",
    "importer.exe": "readlink /proc/<importer pid>/exe",
    "importer.argv": "/proc/<importer pid>/cmdline",
    "importer.fds": "readlink /proc/<importer pid>/fd/*",
    "agent": "/proc/<importer ppid>/{stat,cmdline,exe}",
    "report_size": "stat() of the report path (file-level, NOT counted as importer progress)",
    "reported_progress_pct": "lifecycle.json: the controller's reading of the Nsight log (NOT counted as importer progress)",
    "phase": "lifecycle.json",
}


def read_proc(pid: int, proc: Path = Path("/proc")) -> dict | None:
    """One process from /proc, or None if it is gone or unreadable."""
    d = proc / str(pid)
    try:
        stat = (d / "stat").read_text()
        rest = stat[stat.rindex(")") + 2 :].split()
        rec = {"pid": pid, "comm": stat[stat.index("(") + 1 : stat.rindex(")")], "state": rest[0], "ppid": int(rest[1]), "pgid": int(rest[2]), "sid": int(rest[3]), "ticks": int(rest[11]) + int(rest[12]), "starttime": int(rest[19])}
        rec["uid"] = d.stat().st_uid
    except (OSError, ValueError, IndexError):
        return None
    try:
        rec["exe"] = os.readlink(d / "exe")
    except OSError:
        rec["exe"] = None
    try:
        rec["argv"] = [a for a in (d / "cmdline").read_text().split("\0") if a]
    except OSError:
        rec["argv"] = []
    try:
        rec["io"] = {k.strip(): int(v) for k, v in (line.split(":") for line in (d / "io").read_text().splitlines() if ":" in line)}
    except (OSError, ValueError):
        rec["io"] = None
    try:
        rec["fds"] = sorted({os.readlink(d / "fd" / n) for n in os.listdir(d / "fd")})
    except OSError:
        rec["fds"] = None
    return rec


def _arg(argv: list[str], flag: str) -> str | None:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a[len(flag) + 1 :]
    return None


def find_importers(importer_exe: str, stream: str | None, report: str | None, *, proc: Path = Path("/proc"), uid: int | None = None) -> list[dict]:
    """Processes that ARE this session's importer: exact pinned executable, this uid, and the stream or report named in
    the arguments or held open. `QdstrmImporter --version` and importers of other sessions do not match."""
    uid = os.getuid() if uid is None else uid
    out = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        p = read_proc(int(entry.name), proc)
        if not p or p["uid"] != uid or p["exe"] != importer_exe:
            continue
        why = []
        if stream and _arg(p["argv"], "--input-file") == stream:
            why.append("--input-file is the session stream")
        if report and _arg(p["argv"], "--output-file") == report:
            why.append("--output-file is the session report")
        for path, name in ((stream, "stream"), (report, "report")):
            if path and p["fds"] and path in p["fds"]:
                why.append(f"holds the session {name} open")
        if why:
            p["matched_by"] = why
            out.append(p)
    return out


def agent_of(importer: dict, nsys_pid: int | None, *, proc: Path = Path("/proc")) -> dict:
    """The importer's parent, and whether it is the Nsight agent of the recorded CLI (`--session-name profile-<pid>`)."""
    parent = read_proc(importer["ppid"], proc)
    if not parent:
        return {"pid": importer["ppid"], "found": False, "is_agent": False, "linked_to_recorded_cli": None if nsys_pid is None else False}
    is_agent = "--start-agent" in parent["argv"] and os.path.basename(parent["exe"] or "") == "nsys"
    session = _arg(parent["argv"], "--session-name")
    linked = None if nsys_pid is None else bool(is_agent and session == f"profile-{nsys_pid}")
    return {"pid": parent["pid"], "found": True, "is_agent": is_agent, "session_name": session, "ppid": parent["ppid"], "sid": parent["sid"], "starttime": parent["starttime"], "linked_to_recorded_cli": linked}


def take_sample(importer_exe: str, stream: str | None, report: str | None, *, known: dict | None = None, status: dict | None = None, proc: Path = Path("/proc"), uid: int | None = None, now=time.time) -> dict:
    """One timestamped observation. `known` is the importer identified earlier (pid + starttime), if any."""
    s: dict = {"at": now(), "phase": (status or {}).get("phase"), "reported_progress_pct": (status or {}).get("import_progress_pct")}
    try:
        st = os.stat(report) if report else None
        s["report_size"], s["report_mtime_ns"] = (st.st_size, st.st_mtime_ns) if st else (None, None)
    except OSError:
        s["report_size"], s["report_mtime_ns"] = None, None
    cands = find_importers(importer_exe, stream, report, proc=proc, uid=uid)
    s["candidates"] = len(cands)
    imp = None
    if known is not None:
        imp = next((c for c in cands if c["pid"] == known["pid"] and c["starttime"] == known["starttime"]), None)
        s["known_importer_present"] = imp is not None
    elif len(cands) == 1:
        imp = cands[0]
    s["importer"] = None if imp is None else {k: imp[k] for k in ("pid", "starttime", "state", "ticks", "io", "ppid", "sid", "exe", "matched_by")}
    return s


def interval_progress(a: dict, b: dict) -> dict:
    """Importer-specific advance between two samples: same identity, alive in both, own ticks or own I/O counters grew."""
    ia, ib = a.get("importer"), b.get("importer")
    r = {"from": a["at"], "to": b["at"], "same_importer": False, "ticks_delta": None, "rchar_delta": None, "wchar_delta": None, "qualifies": False}
    if not ia or not ib or (ia["pid"], ia["starttime"]) != (ib["pid"], ib["starttime"]):
        return r
    r["same_importer"] = True
    if ia["state"] in DEAD_STATES or ib["state"] in DEAD_STATES:
        return r
    r["ticks_delta"] = ib["ticks"] - ia["ticks"]
    if ia.get("io") and ib.get("io"):
        r["rchar_delta"] = ib["io"].get("rchar", 0) - ia["io"].get("rchar", 0)
        r["wchar_delta"] = ib["io"].get("wchar", 0) - ia["io"].get("wchar", 0)
    r["qualifies"] = r["ticks_delta"] > 0 or (r["rchar_delta"] or 0) > 0 or (r["wchar_delta"] or 0) > 0
    return r


def generated_in_log(log_path: Path | None, report: str | None) -> bool | None:
    """Nsight's own completion marker for this report in the given log; None if no log was given or it is unreadable."""
    if not log_path or not report:
        return None
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None
    return bool(re.search(r"Generated:\s*\n\s*" + re.escape(report), text))


def classify(samples: list[dict], *, min_progress: int = 2, generated: bool | None = None) -> dict:
    """State from the samples so far. ACTIVE needs `min_progress` consecutive qualifying intervals ending at the latest sample."""
    if not samples:
        return {"state": NOT_FOUND, "reason": "no sample"}
    last = samples[-1]
    ever = next((s["importer"] for s in samples if s.get("importer")), None)
    if last.get("importer") is None and ever is None and last["candidates"] > 1:
        return {"state": AMBIGUOUS, "reason": f"{last['candidates']} processes match the importer criteria; none is singled out"}
    intervals = [interval_progress(a, b) for a, b in zip(samples, samples[1:])]
    run = 0
    for iv in reversed(intervals):
        if not iv["qualifies"]:
            break
        run += 1
    alive = last.get("importer") is not None and last["importer"]["state"] not in DEAD_STATES
    if alive:
        if run >= min_progress:
            return {"state": ACTIVE, "reason": f"importer pid {last['importer']['pid']} alive; {run} consecutive interval(s) of its own CPU/I-O advance", "qualifying_run": run}
        return {"state": NO_PROGRESS, "reason": f"importer pid {last['importer']['pid']} alive, identity intact; only {run} consecutive qualifying interval(s) (need {min_progress})", "qualifying_run": run}
    size = last.get("report_size")
    stable = len(samples) >= 2 and samples[-2].get("report_size") == size and samples[-2].get("report_mtime_ns") == last.get("report_mtime_ns")
    complete = bool(size) and stable
    marker = {True: "Nsight logged 'Generated:' for it", False: "Nsight has not (yet) logged 'Generated:'", None: "no Nsight log given"}[generated]
    if ever is None:
        if complete:
            return {"state": COMPLETED, "reason": f"no importer process was ever identified, but the report is present, non-empty and stable (import finished before the first observation); {marker}", "importer_seen": False, "nsight_generated_marker": generated}
        return {"state": NOT_FOUND, "reason": "no process matches the importer criteria and there is no complete report", "importer_seen": False}
    if complete:
        return {"state": COMPLETED, "reason": f"importer pid {ever['pid']} is gone; report {size} B, stable; {marker}; exit status not observable from /proc", "importer_seen": True, "nsight_generated_marker": generated}
    return {"state": TERMINATED, "reason": f"importer pid {ever['pid']} is gone or replaced and the report is " + ("missing or empty" if not size else "still changing"), "importer_seen": True}


def run(importer_exe: str, report: str, out: Path, samples_path: Path, log, *, stream: str | None = None, nsys_dir: Path | None = None, nsys_pid: int | None = None, nsys_log: Path | None = None,
        interval: float = 0.5, min_progress: int = 2, confirm_s: float = 30.0, max_s: float = 3600.0, settle: int = 3, control: Path | None = None, trigger_line: str | None = None,
        proc: Path = Path("/proc"), uid: int | None = None, sleep=time.sleep, now=time.time) -> int:
    """Observe until a terminal state (COMPLETED / TERMINATED) or --max-s. In trigger mode, write the line once ACTIVE is
    established within --confirm-s of the first sighting; otherwise withhold it. Exit 0 terminal COMPLETED, 1 TERMINATED,
    2 not established (NOT_FOUND / AMBIGUOUS / timeout), 3 trigger withheld."""
    record: dict = {"probe": "e1_import_probe.py", "provenance": PROVENANCE, "importer_exe": importer_exe, "stream": stream, "report": report, "recorded_nsys_cli_pid": nsys_pid,
                    "min_progress": min_progress, "interval_s": interval, "started_at": now(), "transitions": [], "trigger": {"requested": bool(control and trigger_line), "written": False}}
    samples: list[dict] = []
    known = None
    first_seen_at = None
    gone_streak = 0
    state = None
    end = now() + max_s

    def save(result: str) -> None:
        record.update({"result": result, "ended_at": now(), "samples_file": str(samples_path), "samples": len(samples), "final": classify(samples, min_progress=min_progress, generated=generated_in_log(nsys_log, report))})
        out.write_text(json.dumps(record, indent=1) + "\n")
        log(f"[import-probe] {result}")

    with samples_path.open("w") as fh:
        while now() < end:
            status = None
            if nsys_dir is not None:
                try:
                    status = json.loads((nsys_dir / "lifecycle.json").read_text())
                except (OSError, ValueError):
                    status = None
                if stream is None:
                    stream = ((status or {}).get("stream") or {}).get("path")
                    record["stream"] = stream
                if nsys_pid is None:
                    nsys_pid = ((status or {}).get("nsys") or {}).get("pid")
                    record["recorded_nsys_cli_pid"] = nsys_pid
            s = take_sample(importer_exe, stream, report, known=known, status=status, proc=proc, uid=uid, now=now)
            samples.append(s)
            fh.write(json.dumps(s) + "\n")
            fh.flush()
            if known is None and s.get("importer"):
                known = {"pid": s["importer"]["pid"], "starttime": s["importer"]["starttime"]}
                first_seen_at = s["at"]
                record["importer"] = {**known, "exe": s["importer"]["exe"], "matched_by": s["importer"]["matched_by"], "first_seen_at": first_seen_at, "agent": agent_of(s["importer"], nsys_pid, proc=proc)}
                log(f"[import-probe] importer identified: pid {known['pid']} ({'; '.join(s['importer']['matched_by'])})")
            if known is not None and not s.get("importer"):
                gone_streak += 1
            c = classify(samples, min_progress=min_progress, generated=generated_in_log(nsys_log, report) if known and not s.get("importer") else None)
            if known is not None and 0 < gone_streak < settle:
                c = {"state": SETTLING, "reason": f"importer pid {known['pid']} no longer present; waiting {settle} samples for the report to settle"}
            if c["state"] != state:
                state = c["state"]
                record["transitions"].append({"at": s["at"], "state": state, "reason": c["reason"]})
                log(f"[import-probe] state {state}: {c['reason']}")
            if state == ACTIVE and record["trigger"]["requested"] and not record["trigger"]["written"] and "withheld" not in record["trigger"]:
                check = read_proc(known["pid"], proc)
                if check and check["starttime"] == known["starttime"] and check["state"] not in DEAD_STATES and check["exe"] == importer_exe:
                    with control.open("a") as cf:
                        cf.write(trigger_line + "\n")
                    record["trigger"].update({"written": True, "at": now(), "line": trigger_line, "importer_rechecked": {k: check[k] for k in ("pid", "starttime", "state", "ticks")}})
                    log("[import-probe] trigger line written (import active established from the importer's own counters)")
                else:
                    record["trigger"]["withheld"] = "the importer was gone or changed at the final identity check"
            if record["trigger"]["requested"] and not record["trigger"]["written"] and "withheld" not in record["trigger"]:
                if state in (COMPLETED, TERMINATED, AMBIGUOUS, SETTLING):
                    record["trigger"]["withheld"] = f"state {state}: {c['reason']}"
                elif first_seen_at is not None and s["at"] - first_seen_at > confirm_s:
                    record["trigger"]["withheld"] = f"importer activity not established within {confirm_s} s of the first sighting (state {state})"
            if known is not None and gone_streak >= settle:
                break
            elif known is None and state == COMPLETED:
                break
            out.write_text(json.dumps({**record, "result": "running", "samples": len(samples)}, indent=1) + "\n")
            sleep(interval)
    final = classify(samples, min_progress=min_progress, generated=generated_in_log(nsys_log, report))
    if record["trigger"]["requested"] and not record["trigger"]["written"]:
        record["trigger"].setdefault("withheld", f"import activity never established (final state {final['state']})")
        save(f"TRIGGER WITHHELD: {record['trigger']['withheld']}")
        return 3
    if final["state"] == COMPLETED:
        save(f"COMPLETED: {final['reason']}")
        return 0
    if final["state"] == TERMINATED:
        save(f"TERMINATED: {final['reason']}")
        return 1
    save(f"NOT ESTABLISHED ({final['state']}): {final['reason']}")
    return 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--importer-exe", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--stream")
    ap.add_argument("--nsys-dir", type=Path)
    ap.add_argument("--nsys-pid", type=int)
    ap.add_argument("--nsys-log", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--samples", required=True, type=Path)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--min-progress", type=int, default=2)
    ap.add_argument("--confirm-s", type=float, default=30.0)
    ap.add_argument("--max-s", type=float, default=3600.0)
    ap.add_argument("--control", type=Path)
    ap.add_argument("--trigger-line")
    args = ap.parse_args(argv)
    if not args.stream and not args.nsys_dir:
        ap.error("one of --stream or --nsys-dir is required")
    if bool(args.control) != bool(args.trigger_line):
        ap.error("--control and --trigger-line go together")
    if args.min_progress < 1:
        ap.error("--min-progress must be at least 1")

    def log(msg: str) -> None:
        with args.log.open("a") as fh:
            fh.write(f"{msg} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")

    log(f"[import-probe] start pid {os.getpid()} importer {args.importer_exe} report {args.report}")
    return run(args.importer_exe, args.report, args.out, args.samples, log, stream=args.stream, nsys_dir=args.nsys_dir, nsys_pid=args.nsys_pid, nsys_log=args.nsys_log, interval=args.interval,
               min_progress=args.min_progress, confirm_s=args.confirm_s, max_s=args.max_s, control=args.control, trigger_line=args.trigger_line)


if __name__ == "__main__":
    sys.exit(main())
