#!/usr/bin/env python3
"""Experiment-owned process registry, discovery and verified termination (CPU; Linux /proc; stdlib only).

Discovery and termination are separate. Nothing is ever signalled because its command line, cwd or name *looks* like the
experiment: a process is terminable only if it is recorded in the registry with a matching identity, or is a live descendant
of such a process, and the identity is re-read immediately before every signal.

    e1_pids.py register --registry R --label L <pid>              record (pid, start time, boot id, uid, cmd) of a live process
    e1_pids.py spawn    --registry R --label L [--cwd D] [--log F] -- <argv...>
                                                                  start a detached background process and register it
    e1_pids.py plan     --registry R [--json F]                   verify every record, list owned trees, refusals, lookalikes
    e1_pids.py cleanup  --registry R [--execute] [--grace S] [--kill9] [--log F] [--json F]
                                                                  dry-run by default; with --execute: SIGTERM to verified owned
                                                                  processes (deepest first), wait, re-verify; --kill9 repeats
                                                                  with SIGKILL for survivors
    e1_pids.py verify   <pid> <starttime> [boot_id]               exit 0 iff a live process has exactly that identity

Identity = (pid, start time in clock ticks since boot from /proc/<pid>/stat field 22, boot id, uid). A reused pid has a
different start time, so a stale record is reported and skipped. Refusals (exit 3, nothing signalled): a registered or owned
pid is this process, one of its ancestors (shell, sshd, tmux, ...), pid 1, or belongs to another uid. Lookalikes (processes
whose command line or cwd references the experiment but that are not registered/owned) are reported and never signalled;
their presence is a stop-and-report condition (exit 2).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time

LOOKALIKE_RE = re.compile(os.environ.get("E1_LOOKALIKE_RE", r"dsv41_flash_gb300_tp4_e1|phase1c/e1|20261114_dsv41_flash_gb300_tp4_e1"))
FIELDS = ["pid", "starttime", "boot_id", "uid", "label", "registered_at", "host", "registered_by", "cmd"]


def boot_id() -> str:
    try:
        return open("/proc/sys/kernel/random/boot_id").read().strip()
    except OSError:
        return "unknown"


def identity(pid: int) -> dict | None:
    """Live identity of pid, or None if it does not exist (or vanished while reading)."""
    try:
        stat = open(f"/proc/{pid}/stat").read()
        status = open(f"/proc/{pid}/status").read()
    except OSError:
        return None
    after = stat[stat.rindex(")") + 2:].split()      # fields 3.. ; comm may contain spaces/parentheses
    if after[0] in ("Z", "X"):                        # zombie / dead: cannot be signalled, counts as gone
        return None
    starttime = int(after[19])                        # field 22
    ppid = int(after[1])                              # field 4
    uid = int(next(l for l in status.splitlines() if l.startswith("Uid:")).split()[1])
    name = next(l for l in status.splitlines() if l.startswith("Name:")).split(None, 1)[1]
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        cmd = ""
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        cwd = ""
    return {"pid": pid, "starttime": starttime, "boot_id": boot_id(), "uid": uid, "ppid": ppid, "name": name, "cmd": cmd, "cwd": cwd}


def same_identity(rec: dict, ident: dict | None) -> bool:
    return bool(ident) and int(rec["pid"]) == ident["pid"] and int(rec["starttime"]) == ident["starttime"] \
        and rec["boot_id"] == ident["boot_id"] and int(rec["uid"]) == ident["uid"]


def all_pids() -> list[int]:
    return [int(x) for x in os.listdir("/proc") if x.isdigit()]


def ancestors(pid: int) -> list[int]:
    out, seen = [], set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        ident = identity(pid)
        if not ident:
            break
        pid = ident["ppid"]; out.append(pid)
    return out


def ppid_map() -> dict[int, int]:
    m = {}
    for p in all_pids():
        i = identity(p)
        if i:
            m[p] = i["ppid"]
    return m


def descendants(roots: set[int], pm: dict[int, int]) -> dict[int, int]:
    """{pid: depth} of every live descendant of roots (depth 1 = child)."""
    children: dict[int, list[int]] = {}
    for p, pp in pm.items():
        children.setdefault(pp, []).append(p)
    out, frontier, depth = {}, list(roots), 0
    while frontier:
        depth += 1
        nxt = []
        for r in frontier:
            for c in children.get(r, []):
                if c not in out and c not in roots:
                    out[c] = depth; nxt.append(c)
        frontier = nxt
    return out


# ----------------------------------------------------------------------------------------------------------------------
def read_registry(path: str) -> list[dict]:
    recs = []
    if not os.path.exists(path):
        return recs
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4 or not all(x.strip().lstrip("-").isdigit() for x in parts[:2] + parts[3:4]):
            print(f"WARN  registry line ignored (malformed): {line[:120]!r}", file=sys.stderr); continue
        recs.append(dict(zip(FIELDS, parts + [""] * (len(FIELDS) - len(parts)))))
    return recs


def register(path: str, label: str, pid: int) -> dict:
    ident = identity(pid)
    if not ident:
        raise SystemExit(f"register: pid {pid} is not alive")
    rec = {"pid": pid, "starttime": ident["starttime"], "boot_id": ident["boot_id"], "uid": ident["uid"], "label": label,
           "registered_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "host": socket.gethostname(),
           "registered_by": os.getpid(), "cmd": " ".join(ident["cmd"][:300].split())}      # one line, no tabs/newlines
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a") as fh:
        if new:
            fh.write("# " + "\t".join(FIELDS) + "\n")
        fh.write("\t".join(str(rec[k]) for k in FIELDS) + "\n")
    return rec


def spawn(path: str, label: str, argv: list[str], cwd: str | None, log: str | None) -> dict:
    out = open(log, "ab") if log else subprocess.DEVNULL
    p = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    rec = register(path, label, p.pid)
    rec["argv"] = argv
    return rec


# ----------------------------------------------------------------------------------------------------------------------
def plan(path: str) -> dict:
    me = os.getpid()
    excluded = {me, 1} | set(ancestors(me))
    recs = read_registry(path)
    pm = ppid_map()
    verified, stale, gone, refused = [], [], [], []
    for rec in recs:
        pid = int(rec["pid"]); ident = identity(pid)
        if ident is None:
            gone.append(rec); continue
        if not same_identity(rec, ident):
            stale.append({**rec, "live": {k: ident[k] for k in ("starttime", "uid", "name", "cmd")}, "reason": "identity mismatch: pid reused or different boot/uid"}); continue
        if pid in excluded:
            refused.append({**rec, "reason": "registered pid is this cleanup process or one of its ancestors (shell / ssh / tmux)"}); continue
        if ident["uid"] != os.getuid():
            refused.append({**rec, "reason": f"uid {ident['uid']} is not the current user {os.getuid()}"}); continue
        verified.append({**rec, "live": ident})
    roots = {int(r["pid"]) for r in verified}
    desc = descendants(roots, pm)
    owned = []
    for r in verified:
        owned.append({"pid": int(r["pid"]), "depth": 0, "label": r["label"], "ident": r["live"]})
    for pid, depth in desc.items():
        ident = identity(pid)
        if not ident:
            continue
        if pid in excluded:
            refused.append({"pid": pid, "reason": "an owned descendant is this cleanup process or one of its ancestors: cleanup is running inside the experiment tree", "cmd": ident["cmd"]})
            continue
        if ident["uid"] != os.getuid():
            refused.append({"pid": pid, "reason": f"descendant uid {ident['uid']} differs from the current user", "cmd": ident["cmd"]})
            continue
        owned.append({"pid": pid, "depth": depth, "label": "descendant", "ident": ident})
    owned_pids = {o["pid"] for o in owned}
    lookalikes = []
    for pid in pm:
        if pid in owned_pids or pid in excluded or pid == me:
            continue
        ident = identity(pid)
        if not ident:
            continue
        if "e1_pids.py" in ident["cmd"]:
            continue
        if LOOKALIKE_RE.search(ident["cmd"]) or LOOKALIKE_RE.search(ident["cwd"]):
            lookalikes.append({"pid": pid, "uid": ident["uid"], "name": ident["name"], "cmd": ident["cmd"][:200], "cwd": ident["cwd"]})
    return {"registry": path, "host": socket.gethostname(), "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cleanup_pid": me, "excluded_ancestry": sorted(excluded), "records": len(recs), "verified": verified, "gone": gone,
            "stale": stale, "refused": refused, "owned": sorted(owned, key=lambda o: -o["depth"]), "lookalikes": lookalikes}


def render_plan(p: dict) -> str:
    L = [f"# e1_pids plan on {p['host']} at {p['at']}: registry {p['registry']} ({p['records']} records); cleanup pid {p['cleanup_pid']}, excluded ancestry {p['excluded_ancestry']}"]
    for r in p["gone"]:
        L.append(f"GONE     pid {r['pid']} [{r['label']}] start {r['starttime']}: no such process (already exited)")
    for r in p["stale"]:
        L.append(f"STALE    pid {r['pid']} [{r['label']}] recorded start {r['starttime']}, live start {r['live']['starttime']} ({r['live']['name']}): {r['reason']} -> not touched")
    for r in p["refused"]:
        L.append(f"REFUSED  pid {r['pid']} [{r.get('label', '')}]: {r['reason']} -> STOP, nothing will be signalled")
    for o in p["owned"]:
        L.append(f"OWNED    pid {o['pid']} depth {o['depth']} [{o['label']}] start {o['ident']['starttime']} {o['ident']['name']}: {o['ident']['cmd'][:120]}")
    for l in p["lookalikes"]:
        L.append(f"LOOKALIKE pid {l['pid']} uid {l['uid']} {l['name']}: {l['cmd'][:120]} (cwd {l['cwd']}) -> NOT owned, not touched; stop-and-report")
    L.append(f"summary: owned {len(p['owned'])}, stale {len(p['stale'])}, gone {len(p['gone'])}, refused {len(p['refused'])}, lookalikes {len(p['lookalikes'])}")
    return "\n".join(L)


def signal_verified(target: dict, sig: int, log: list) -> str:
    """Re-read identity immediately before signalling; signal only on an exact match."""
    pid, want = target["pid"], target["ident"]
    now = identity(pid)
    if now is None:
        log.append({"pid": pid, "action": "skip", "reason": "gone before signal"}); return "gone"
    if now["starttime"] != want["starttime"] or now["uid"] != want["uid"] or now["boot_id"] != want["boot_id"]:
        log.append({"pid": pid, "action": "skip", "reason": "identity changed before signal (pid reuse)", "live": now["cmd"][:120]}); return "changed"
    if pid in (os.getpid(), 1) or pid in ancestors(os.getpid()):
        log.append({"pid": pid, "action": "refuse", "reason": "own ancestry"}); return "refused"
    try:
        os.kill(pid, sig)
        log.append({"pid": pid, "action": signal.Signals(sig).name, "cmd": now["cmd"][:120], "starttime": now["starttime"]}); return "signalled"
    except ProcessLookupError:
        log.append({"pid": pid, "action": "skip", "reason": "vanished at signal"}); return "gone"
    except PermissionError:
        log.append({"pid": pid, "action": "error", "reason": "EPERM"}); return "eperm"


def cleanup(path: str, execute: bool, grace: float, kill9: bool, log_path: str | None) -> tuple[int, dict]:
    p = plan(path)
    print(render_plan(p))
    log: list = []
    result = {"plan": p, "execute": execute, "actions": log, "survivors": [], "rc": 0}
    if p["refused"]:
        print("RESULT REFUSED: a registered/owned pid is in this cleanup's own ancestry or another user's; nothing signalled (rc 3)")
        result["rc"] = 3
    elif not execute:
        print(f"RESULT DRY-RUN: {len(p['owned'])} process(es) would receive SIGTERM (deepest first); run with --execute to terminate; lookalikes {len(p['lookalikes'])} would never be touched")
        result["rc"] = 2 if p["lookalikes"] else 0
    else:
        for t in p["owned"]:
            signal_verified(t, signal.SIGTERM, log)
        deadline = time.time() + grace
        while time.time() < deadline and any(same_identity({"pid": t["pid"], "starttime": t["ident"]["starttime"], "boot_id": t["ident"]["boot_id"], "uid": t["ident"]["uid"]}, identity(t["pid"])) for t in p["owned"]):
            time.sleep(0.2)
        survivors = [t for t in p["owned"] if same_identity({"pid": t["pid"], "starttime": t["ident"]["starttime"], "boot_id": t["ident"]["boot_id"], "uid": t["ident"]["uid"]}, identity(t["pid"]))]
        if survivors and kill9:
            for t in survivors:
                signal_verified(t, signal.SIGKILL, log)
            time.sleep(0.5)
            survivors = [t for t in survivors if same_identity({"pid": t["pid"], "starttime": t["ident"]["starttime"], "boot_id": t["ident"]["boot_id"], "uid": t["ident"]["uid"]}, identity(t["pid"]))]
        result["survivors"] = [{"pid": t["pid"], "cmd": t["ident"]["cmd"][:120]} for t in survivors]
        for a in log:
            print("ACTION  ", json.dumps(a))
        for s in survivors:
            print(f"SURVIVOR pid {s['pid']}: {s['ident']['cmd'][:120]}")
        ok = not survivors and not p["lookalikes"]
        print(f"RESULT {'CLEAN' if ok else 'NOT CLEAN'}: signalled {sum(1 for a in log if a['action'] in ('SIGTERM', 'SIGKILL'))}, skipped {sum(1 for a in log if a['action'] == 'skip')}, survivors {len(survivors)}, lookalikes {len(p['lookalikes'])} (never signalled)")
        result["rc"] = 0 if ok else 2
    if log_path:
        with open(log_path, "a") as fh:
            fh.write(json.dumps(result, default=str) + "\n")
    return result["rc"], result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register"); r.add_argument("--registry", required=True); r.add_argument("--label", required=True); r.add_argument("pid", type=int)
    s = sub.add_parser("spawn"); s.add_argument("--registry", required=True); s.add_argument("--label", required=True); s.add_argument("--cwd"); s.add_argument("--log"); s.add_argument("argv", nargs=argparse.REMAINDER)
    p = sub.add_parser("plan"); p.add_argument("--registry", required=True); p.add_argument("--json")
    c = sub.add_parser("cleanup"); c.add_argument("--registry", required=True); c.add_argument("--execute", action="store_true"); c.add_argument("--grace", type=float, default=10.0)
    c.add_argument("--kill9", action="store_true"); c.add_argument("--log"); c.add_argument("--json")
    v = sub.add_parser("verify"); v.add_argument("pid", type=int); v.add_argument("starttime", type=int); v.add_argument("boot_id", nargs="?")
    a = ap.parse_args(argv)
    if a.cmd == "register":
        rec = register(a.registry, a.label, a.pid); print(json.dumps(rec)); return 0
    if a.cmd == "spawn":
        argv = a.argv[1:] if a.argv and a.argv[0] == "--" else a.argv
        if not argv:
            ap.error("spawn needs -- <argv...>")
        rec = spawn(a.registry, a.label, argv, a.cwd, a.log); print(json.dumps(rec)); return 0
    if a.cmd == "plan":
        pl = plan(a.registry); print(render_plan(pl))
        if a.json:
            json.dump(pl, open(a.json, "w"), indent=1, default=str)
        return 3 if pl["refused"] else (2 if pl["lookalikes"] else 0)
    if a.cmd == "cleanup":
        rc, res = cleanup(a.registry, a.execute, a.grace, a.kill9, a.log)
        if a.json:
            json.dump(res, open(a.json, "w"), indent=1, default=str)
        return rc
    if a.cmd == "verify":
        ident = identity(a.pid)
        ok = bool(ident) and ident["starttime"] == a.starttime and (a.boot_id is None or ident["boot_id"] == a.boot_id)
        print(json.dumps({"pid": a.pid, "match": ok, "live": ident and {k: ident[k] for k in ("starttime", "uid", "name", "cmd")}}))
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
