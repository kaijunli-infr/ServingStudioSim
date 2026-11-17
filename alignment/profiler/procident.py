"""Identity-verified process handling for the profiler lifecycle (Linux /proc).

A pid alone is not an identity: it is reused. Every record this module hands
out carries the kernel start time (clock ticks since boot), the boot id and
the uid, and every signal is re-verified against the live `/proc` entry
immediately before `os.kill`. Nothing here matches command lines: a process is
only ever signalled because *we* recorded it (or it descends from a recorded
root) and it still has the recorded identity. Own process and own ancestry are
refused unconditionally.

This mirrors the experiment-side registry (`phase1c/e1/checks/e1_pids.py`); it
is duplicated into the package so the launcher does not import from an
experiment directory. Behaviour differences are defects.
"""

from __future__ import annotations

import os
import signal
from dataclasses import asdict, dataclass

_GONE_STATES = ("Z", "X")


def boot_id() -> str:
    try:
        with open("/proc/sys/kernel/random/boot_id") as fh:
            return fh.read().strip()
    except OSError:
        return "unknown"


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    starttime: int
    boot_id: str
    uid: int
    ppid: int
    name: str
    cmd: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ProcessIdentity:
        return cls(
            pid=int(data["pid"]),
            starttime=int(data["starttime"]),
            boot_id=str(data["boot_id"]),
            uid=int(data["uid"]),
            ppid=int(data.get("ppid", -1)),
            name=str(data.get("name", "")),
            cmd=str(data.get("cmd", "")),
        )


def identity(pid: int) -> ProcessIdentity | None:
    """Live identity of ``pid``; None when it does not exist, is a zombie, or vanished mid-read."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            stat = fh.read()
        with open(f"/proc/{pid}/status") as fh:
            status = fh.read()
    except OSError:
        return None
    after = stat[stat.rindex(")") + 2 :].split()  # comm may contain spaces/parentheses
    if after[0] in _GONE_STATES:
        return None
    starttime = int(after[19])  # field 22 of /proc/<pid>/stat
    ppid = int(after[1])
    uid = int(next(line for line in status.splitlines() if line.startswith("Uid:")).split()[1])
    name = next(line for line in status.splitlines() if line.startswith("Name:")).split(None, 1)[1]
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmd = fh.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        cmd = ""
    return ProcessIdentity(pid, starttime, boot_id(), uid, ppid, name, cmd)


def same_identity(recorded: ProcessIdentity, live: ProcessIdentity | None) -> bool:
    return (
        live is not None
        and recorded.pid == live.pid
        and recorded.starttime == live.starttime
        and recorded.boot_id == live.boot_id
        and recorded.uid == live.uid
    )


def is_alive(recorded: ProcessIdentity) -> bool:
    return same_identity(recorded, identity(recorded.pid))


def ancestors(pid: int) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        ident = identity(pid)
        if ident is None:
            break
        pid = ident.ppid
        out.append(pid)
    return out


def ppid_map() -> dict[int, int]:
    out: dict[int, int] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as fh:
                stat = fh.read()
        except OSError:
            continue
        after = stat[stat.rindex(")") + 2 :].split()
        out[int(entry)] = int(after[1])
    return out


def descendants(root_pid: int) -> list[int]:
    """All live descendants of ``root_pid`` (children first, breadth-first)."""
    parents = ppid_map()
    children: dict[int, list[int]] = {}
    for pid, ppid in parents.items():
        children.setdefault(ppid, []).append(pid)
    out: list[int] = []
    frontier = [root_pid]
    while frontier:
        nxt: list[int] = []
        for p in frontier:
            for c in children.get(p, []):
                out.append(c)
                nxt.append(c)
        frontier = nxt
    return out


def is_descendant_of(pid: int, root_pid: int) -> bool:
    return root_pid in ancestors(pid)


def signal_verified(recorded: ProcessIdentity, sig: int, log: list[dict]) -> str:
    """Re-read the identity and signal only on an exact match. Returns an outcome word.

    Refuses the calling process, pid 1 and the caller's own ancestry whatever the
    record says. Pid reuse (same pid, different start time) is reported as
    ``"changed"`` and never signalled.
    """
    live = identity(recorded.pid)
    if live is None:
        log.append({"pid": recorded.pid, "action": "skip", "reason": "gone before signal"})
        return "gone"
    if not same_identity(recorded, live):
        log.append(
            {
                "pid": recorded.pid,
                "action": "skip",
                "reason": "identity changed before signal (pid reuse)",
                "live": live.cmd[:120],
            }
        )
        return "changed"
    me = os.getpid()
    if recorded.pid in (me, 1) or recorded.pid in ancestors(me):
        log.append({"pid": recorded.pid, "action": "refuse", "reason": "own process or ancestry"})
        return "refused"
    try:
        os.kill(recorded.pid, sig)
    except ProcessLookupError:
        log.append({"pid": recorded.pid, "action": "skip", "reason": "vanished at signal"})
        return "gone"
    except PermissionError:
        log.append({"pid": recorded.pid, "action": "error", "reason": "EPERM"})
        return "eperm"
    log.append(
        {
            "pid": recorded.pid,
            "action": signal.Signals(sig).name,
            "cmd": live.cmd[:120],
            "starttime": live.starttime,
        }
    )
    return "signalled"
