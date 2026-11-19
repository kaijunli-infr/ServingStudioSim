#!/usr/bin/env python3
"""Decision check for an OPTIONAL early deletion of the GPU reservation after an independently verified GPU release.
DESIGNED AND CPU-TESTED ONLY. Not enabled for V4: E1-V4-SCALE-INPUTS.json registers `enabled: false`, the V4 scripts do
not call this file, and it never talks to Kubernetes. The reservation lifecycle of V4 is the existing one (deletion after
a CLEAN closeout).

    python3 -I checks/e1_reservation_release.py --run <run root>/<pass> --pod-spec <kubectl get pod -o json>
        --census <e1_gpu_release_census.py output taken now> --inputs E1-V4-SCALE-INPUTS.json [--out evidence.json]

It answers one question: if the reservation pod were deleted now, could that (a) leave GPUs that this run still holds
unprotected, (b) terminate or disturb a process of this run, (c) destroy a preserved file, or (d) invalidate cleanup
evidence? The decision is WOULD_ALLOW only if every condition holds; anything unknown is REFUSE.

  C1 controller   lifecycle.json: gpu.released_at recorded, server not alive, no release still pending
  C2 sampler      the independent GPU sampler (not the controller): since the recorded release, at least 5 consecutive
                  observations, all available, no compute process, every GPU at or below the idle bound, spanning at
                  least 10 s, none 'held' after the first released one, the newest at most 30 s old
  C3 census       a pinned census taken now (e1_gpu_release_census.py): no compute process, memory at the idle bound,
                  at most 60 s old, taken after the recorded release
  C4 processes    every live registered process (driver tree roots, observers, supervisor), the importer and its agent
                  recorded by the import probe: its /proc/<pid>/cgroup is readable and not inside a Kubernetes pod
                  (no `kubepods` path element). A host process outside every pod cgroup is not a member of the
                  reservation pod and is not signalled when that pod's containers are stopped.
  C5 pod          the pod specification: the reservation pod named in reservation.json, on this node, containers only
                  the pause image, no hostPID / hostNetwork / hostIPC, no volume other than the projected service-account
                  token (so no host path, no preserved file and no run directory is mounted in it)
  C6 files        the preserved stream recorded by the controller exists with the recorded size
  C7 evidence     the record written to --out: every input above with its time. It must exist BEFORE a deletion, because
                  afterwards the scheduler may place another GPU pod on the node and later GPU observations no longer
                  describe this run.

Exit 0 WOULD_ALLOW and enabled; 4 WOULD_ALLOW but not enabled (always the case for V4); 1 REFUSE. Stdlib only; read-only
except for --out.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import json
import socket
import sys
import time
from pathlib import Path

IDLE_MIB = 64
PAUSE_PREFIX = "registry.k8s.io/pause:"


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None


def registry_pids(path: Path) -> list[int]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    return [int(ln.split("\t")[0]) for ln in lines if ln.strip() and not ln.startswith("#") and ln.split("\t")[0].isdigit()]


def cgroup_of(pid: int, proc: Path) -> str | None:
    """Contents of /proc/<pid>/cgroup; None if the process does not exist (or is a zombie being reaped)."""
    try:
        return (proc / str(pid) / "cgroup").read_text()
    except OSError:
        return None


def alive(pid: int, proc: Path) -> bool:
    try:
        return (proc / str(pid) / "stat").read_text().rsplit(") ", 1)[-1][0] not in "ZX"
    except (OSError, IndexError):
        return False


def decide(run: Path, pod_spec: dict | None, census: dict | None, *, now: float, host: str, proc: Path = Path("/proc"), min_obs: int = 5, min_span_s: float = 10.0, max_obs_age_s: float = 30.0, max_census_age_s: float = 60.0) -> dict:
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    lc = load(run / "profile" / "nsys" / "lifecycle.json") or {}
    gpu, released = lc.get("gpu") or {}, (lc.get("gpu") or {}).get("released_at")
    add("C1.controller_release_recorded", released is not None and (lc.get("server") or {}).get("alive") is False, f"gpu.released_at {released}; server alive {(lc.get('server') or {}).get('alive')}; release requested {gpu.get('release_requested_at')}")
    try:
        obs = [json.loads(ln) for ln in (run / "gpu_observations.jsonl").read_text().splitlines() if ln.strip()]
    except (OSError, ValueError):
        obs = []
    obs = [o for o in obs if o.get("host") == host]

    def is_released(o: dict) -> bool:
        return o.get("available") is True and not o.get("compute_pids") and max(o.get("memory_used_mib") or [10**9]) <= IDLE_MIB

    after = [o for o in obs if released is not None and o.get("at", 0) >= released]
    tail = []
    for o in reversed(after):
        if not is_released(o):
            break
        tail.append(o)
    tail.reverse()
    span = tail[-1]["at"] - tail[0]["at"] if tail else 0.0
    first_rel = next((i for i, o in enumerate(after) if is_released(o)), None)
    held_again = first_rel is not None and any(not is_released(o) for o in after[first_rel:])
    add("C2.independent_sampler_released", len(tail) >= min_obs and span >= min_span_s and not held_again and now - tail[-1]["at"] <= max_obs_age_s,
        f"{len(tail)} consecutive released observation(s) on {host} since the recorded release, spanning {span:.1f} s (need {min_obs} over {min_span_s} s); newest {None if not tail else round(now - tail[-1]['at'], 1)} s old (limit {max_obs_age_s}); not released after a released one: {held_again}")
    c_ok = census is not None and census.get("host") == host and not census.get("compute_apps") and all((g.get("memory_used_mib", 10**9) <= IDLE_MIB) for g in census.get("gpus") or [{}]) and len(census.get("gpus") or []) > 0
    c_age = None if census is None or census.get("at") is None else now - census["at"]
    add("C3.census_now", c_ok and c_age is not None and 0 <= c_age <= max_census_age_s and released is not None and census["at"] >= released,
        f"census host {None if census is None else census.get('host')}, compute processes {None if census is None else len(census.get('compute_apps') or [])}, GPU memory {None if census is None else [g.get('memory_used_mib') for g in census.get('gpus') or []]} MiB, age {None if c_age is None else round(c_age, 1)} s (limit {max_census_age_s})")
    probe = load(run / "import_probe.json") or {}
    named = {"importer": (probe.get("importer") or {}).get("pid"), "Nsight agent": ((probe.get("importer") or {}).get("agent") or {}).get("pid"), "Nsight CLI": (lc.get("nsys") or {}).get("pid")}
    pids = {f"registered:{p}": p for reg in ("owned.tsv", "supervisor.tsv") for p in registry_pids(run.parent / "pids" / reg)}
    pids.update({k: v for k, v in named.items() if isinstance(v, int)})
    inside, unreadable, live = [], [], []
    for label, pid in pids.items():
        if not alive(pid, proc):
            continue
        live.append(label)
        cg = cgroup_of(pid, proc)
        if cg is None:
            unreadable.append(label)
        elif "kubepods" in cg:
            inside.append(label)
    add("C4.run_processes_outside_every_pod", bool(pids) and not inside and not unreadable, f"{len(pids)} recorded process(es), {len(live)} alive ({live}); inside a pod cgroup: {inside}; cgroup unreadable: {unreadable}")
    res = load(run / "reservation.json") or {}
    spec = (pod_spec or {}).get("spec") or {}
    problems = []
    if pod_spec is None or pod_spec.get("kind") != "Pod":
        problems.append("no pod specification")
    else:
        if (pod_spec.get("metadata") or {}).get("name") != res.get("pod"):
            problems.append(f"pod {(pod_spec.get('metadata') or {}).get('name')!r} is not the reservation pod {res.get('pod')!r}")
        if spec.get("nodeName") != host:
            problems.append(f"pod on node {spec.get('nodeName')!r}, this host is {host!r}")
        for flag in ("hostPID", "hostNetwork", "hostIPC"):
            if spec.get(flag):
                problems.append(f"{flag} is set")
        images = [c.get("image", "") for c in (spec.get("containers") or []) + (spec.get("initContainers") or [])]
        if not images or any(not i.startswith(PAUSE_PREFIX) for i in images):
            problems.append(f"containers are not only the pause image: {images}")
        other = [v.get("name") for v in spec.get("volumes") or [] if set(v) - {"name"} != {"projected"}]
        if other:
            problems.append(f"volumes other than the projected service-account token: {other}")
        if any(c.get("volumeMounts") and any(not str(m.get("mountPath", "")).startswith("/var/run/secrets/") for m in c["volumeMounts"]) for c in spec.get("containers") or []):
            problems.append("a container mounts a path outside /var/run/secrets")
    add("C5.pod_is_an_isolated_placeholder", not problems, "reservation pod: pause container only, no host namespace, no host or persistent volume" if not problems else f"{problems}")
    stream = lc.get("stream") or {}
    pres = stream.get("preserved_path")
    try:
        size = Path(pres).stat().st_size if pres else None
    except OSError:
        size = None
    add("C6.preserved_stream_present", size is not None and size == (stream.get("final") or {}).get("size", stream.get("preserved_size")), f"preserved stream {pres}: {size} B (recorded {(stream.get('final') or {}).get('size', stream.get('preserved_size'))})")
    ok = all(c["ok"] for c in checks)
    return {"decision": "WOULD_ALLOW" if ok else "REFUSE", "checked_at": now, "host": host, "run": str(run), "checks": checks, "refused_by": [c["check"] for c in checks if not c["ok"]],
            "evidence": {"gpu_released_at": released, "sampler_tail": tail[-min_obs:], "census": census, "processes_alive": live},
            "note": "decision only; this program deletes nothing and is not called by any V4 script"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--pod-spec", required=True, type=Path)
    ap.add_argument("--census", required=True, type=Path)
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    enabled = (((load(args.inputs) or {}).get("reservation") or {}).get("early_release_after_verified_gpu_release") or {}).get("enabled") is True
    rep = decide(args.run, load(args.pod_spec), load(args.census), now=time.time(), host=socket.gethostname())
    rep["enabled"] = enabled
    for c in rep["checks"]:
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['check']}: {c['detail']}")
    if args.out:
        args.out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(f"# early reservation release: {rep['decision']}" + ("" if enabled else " — NOT ENABLED for this experiment (registered in the inputs); the reservation is deleted only after a CLEAN closeout") + (f"; refused by {rep['refused_by']}" if rep["refused_by"] else ""))
    if rep["decision"] != "WOULD_ALLOW":
        return 1
    return 0 if enabled else 4


if __name__ == "__main__":
    sys.exit(main())
