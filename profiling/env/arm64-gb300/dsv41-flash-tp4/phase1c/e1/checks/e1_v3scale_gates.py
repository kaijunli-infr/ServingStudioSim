#!/usr/bin/env python3
"""Prospective acceptance gates for V3 (intermediate trace size, normal stop). Written before any V3 launch.

    python3 -I checks/e1_v3scale_gates.py --run <run root>/<pass> --inputs E1-V3-SCALE-INPUTS.json --pin V3S-PIN.json
        --provenance BINARY-PROVENANCE.json [--binary <session_runner>] --importer <QdstrmImporter> --nsys <nsys>
        --durable-stream <copy on durable storage> [--scratch DIR] [--identity identity.json]
        [--postrun-storage storage_postrun.jsonl] [--postrun-memory mem_postrun.jsonl]
        [--reservation-deleted reservation_deleted.json] [--extra name=rc ...] [--out V3S-GATES.json]

  G0   source and executable: launch record = pinned trees and gitlinks, clean; executable = pinned bytes before and after
  G1   host identity: lifecycle = run log = reservation node = every sampler (GPU, memory, storage, import probe)
  G2   one stop request, returned OK, before the server was signalled
  G3   phases in the registered order; final report_written; no escalation, no control line, no forced release
  G4   phase deadlines on the registered clocks (below)
  G5   importer observed by the importer-specific probe: one importer, activity from its own counters, completed;
       no signal to anything but the recorded server processes
  G6   GPU release as a sampled interval after the server SIGTERM; its upper edge within the deadline
  G7   preserved stream: final record = durable copy = tray original (complete SHA-256)
  G8   offline import of the durable copy and semantic completeness (absolute bands; equality with the in-session export)
  G9   scientific checkers: four-wave completeness and routing records, extract, Nsight validation, configuration identity
  G10  bounds: GPU-held ceiling, session close, reservation
  G11  storage and host memory: measured peaks within the ceilings; an unmeasured peak is INCONCLUSIVE
  G12  closeout clean, reservation deleted

Phase clocks (all wall-clock seconds, each from one named record):
  t_stop      lifecycle.stop_requested_at                     t_accepted  phase_history[stop_accepted].at
  t_importing phase_history[importing].at (controller)        t_report    phase_history[report_written].at
  t_imp_first import probe: importer.first_seen_at            t_imp_gone  import probe: first SETTLING transition
  t_sigterm   lifecycle.actions: first SIGTERM to the API-server pid
  t_closed    lifecycle.session_closed_at
  stop accepted = t_accepted - t_stop;  flush = t_importing - t_stop;  import = t_imp_gone - t_imp_first;
  stop to report = t_report - t_stop;  session close = t_closed - t_sigterm.  A phase PASSes at or below its deadline.

GPU readings: an observation's stamp precedes its queries, and the sampler then sleeps the interval; a reading was
taken within [stamp, next stamp - interval]. The release is reported as (last held window, first released window)
relative to t_sigterm and judged on the upper edge of the first released window.

Verdict: FAIL if any gate FAILs; INCONCLUSIVE if none fails but one could not be evaluated; PASS otherwise.
Exit 0 / 1 / 3. Stdlib only; read-only on the run directory (scratch under --scratch).
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import calendar
import contextlib
import glob
import importlib.util
import io
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["stop_requested", "stop_accepted", "capture_ended", "stream_written", "importing", "report_written"]
API_STOPPING = re.compile(rb"^\(APIServer pid=\d+\).*\[api_router\.py:\d+\] Stopping profiler", re.M)
API_STOPPED = re.compile(rb"^\(APIServer pid=\d+\).*\[api_router\.py:\d+\] Profiler stopped\.", re.M)
GB = 1e9
KB_PER_GB = 1024 * 1024
SCI = ("completeness_v2", "extract", "nsys_validation")


def _mod(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add(res: list, name: str, status: str, detail: str, **data) -> None:
    res.append({"gate": name, "status": status, "detail": detail, **({"data": data} if data else {})})
    print(f"{status:12s} {name}: {detail}")


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None


def load_jsonl(path) -> list | None:
    try:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except (OSError, ValueError, TypeError):
        return None


def iso(t) -> str:
    return "-" if t is None else time.strftime("%H:%M:%S", time.gmtime(t)) + f"{t % 1:.2f}"[1:] + "Z"


def phase_at(lc: dict, phase: str):
    return next((h["at"] for h in lc.get("phase_history", []) if h["phase"] == phase), None)


def sigterm_at(lc: dict):
    """Time of the first SIGTERM the controller sent to the recorded API-server pid."""
    api = ((lc.get("server") or {}).get("api_server") or {}).get("pid")
    return next((a.get("at") for a in lc.get("actions", []) if a.get("action") == "SIGTERM" and api is not None and a.get("pid") == api), None)


# ---- G0


def gate_source(res: list, run: Path, pin: dict | None, provenance: dict | None, binary_now: str | None) -> None:
    src = load(run / "launch_source.json")
    code = (pin or {}).get("code") or {}
    if pin is None or src is None:
        add(res, "G0.source_and_executable", "INCONCLUSIVE", f"{'pin file' if pin is None else 'launch_source.json'} missing or unreadable")
        return
    problems = []
    for key in ("alignment", "launcher"):
        if (src.get("trees") or {}).get(key) != (code.get("tree") or {}).get(key):
            problems.append(f"tree {key} {str((src.get('trees') or {}).get(key))[:12]} != pinned {str((code.get('tree') or {}).get(key))[:12]}")
    for path, want in (code.get("gitlinks") or {}).items():
        if (src.get("gitlinks") or {}).get(path) != want:
            problems.append(f"gitlink {path} {str((src.get('gitlinks') or {}).get(path))[:12]} != pinned {want[:12]}")
    if src.get("tracked_changes") != 0:
        problems.append(f"tracked changes in the launch tree: {src.get('tracked_changes')}")
    if src.get("descends_from_pinned_commit") is not True:
        problems.append("launch commit does not descend from the pinned code commit")
    sha = (provenance or {}).get("sha256")
    if pin.get("session_runner_sha256") != sha:
        problems.append("pin file and BINARY-PROVENANCE.json name different executables")
    driver, launcher = load(run / "binary_check.json"), load(run / "profile" / "load_generator" / "frontend_binary.json")
    checks = _mod("e1_frontend_binary_gate").evaluate(driver, launcher, sha, binary_now)
    bad = [c["check"] for c in checks if not c["ok"]]
    if driver is None or launcher is None:
        add(res, "G0.source_and_executable", "FAIL" if problems else "INCONCLUSIVE", f"binary record missing ({'driver' if driver is None else 'launcher'}); source problems {problems}", binary_checks=checks)
        return
    ok = not problems and not bad
    add(res, "G0.source_and_executable", "PASS" if ok else "FAIL", f"launch commit {str(src.get('commit'))[:12]}, trees alignment {str((src.get('trees') or {}).get('alignment'))[:12]} launcher {str((src.get('trees') or {}).get('launcher'))[:12]}, tracked changes {src.get('tracked_changes')}; executable {sha}: {len(checks) - len(bad)} of {len(checks)} binary checks pass" + (f"; problems {problems + bad}" if not ok else ""), binary_checks=checks)


# ---- G1


def gate_host(res: list, run: Path, lc: dict, reservation: dict | None, probe: dict | None, extra_hosts: dict) -> None:
    hostmod = _mod("e1_host_identity")
    h = hostmod.cross_check(lc, hostmod.run_log_host(run), reservation, hostmod.gpu_observation_hosts(run, None))
    status, detail = h["status"], h["detail"]
    want = lc.get("host")
    others = {"import probe": (probe or {}).get("host"), **extra_hosts}
    missing = [k for k, v in others.items() if v is None]
    conflict = {k: v for k, v in others.items() if v is not None and v != want}
    if conflict:
        status, detail = "FAIL", detail + f"; sampler host conflict {conflict}"
    elif missing and status == "PASS":
        status, detail = "INCONCLUSIVE", detail + f"; no host recorded by: {missing}"
    add(res, "G1.host_identity", status, detail + (f"; import probe, memory and storage samplers on {want}" if status == "PASS" else ""), sources={**h["sources"], **others})


# ---- G2, G3


def gate_stop_request(res: list, lc: dict, server_log: Path | None) -> None:
    http, sig = lc.get("http", {}), sigterm_at(lc)
    phases = [h["phase"] for h in lc.get("phase_history", [])]
    raw = server_log.read_bytes() if server_log and server_log.is_file() else None
    stopping, stopped = (len(API_STOPPING.findall(raw)), len(API_STOPPED.findall(raw))) if raw is not None else (None, None)
    returned = http.get("returned_at")
    ok = http.get("requests_sent") == 1 and phases.count("stop_requested") == 1 and http.get("status") == "ok" and returned is not None and sig is not None and returned <= sig and stopping == 1 and stopped == 1
    unknown = raw is None or sig is None
    hard = http.get("requests_sent") != 1 or phases.count("stop_requested") != 1 or http.get("status") != "ok" or (raw is not None and (stopping != 1 or stopped != 1)) or (sig is not None and returned is not None and returned > sig)
    add(res, "G2.one_stop_request_returned_ok", "FAIL" if hard else ("PASS" if ok else "INCONCLUSIVE" if unknown else "FAIL"), f"requests_sent={http.get('requests_sent')}, stop_requested phases={phases.count('stop_requested')}, http={http.get('status')!r} returned {iso(returned)}, server SIGTERM {iso(sig)}, API-server 'Stopping profiler' lines={stopping}, 'Profiler stopped.' lines={stopped}")


def gate_phases(res: list, lc: dict) -> None:
    phases = [h["phase"] for h in lc.get("phase_history", [])]
    ats = [h["at"] for h in lc.get("phase_history", [])]
    esc, ctl, rel = lc.get("escalations", []), lc.get("control_commands", []), (lc.get("gpu") or {}).get("release_requested_at")
    ok = phases == ORDER and ats == sorted(ats) and lc.get("phase") == "report_written" and not esc and not ctl and rel is None
    add(res, "G3.phases_in_order", "PASS" if ok else "FAIL", f"phases {phases} (registered {ORDER}); times non-decreasing={ats == sorted(ats)}; final phase {lc.get('phase')!r}; escalations {[(e.get('kind'), e.get('phase')) for e in esc]}; control lines {[c.get('command') for c in ctl]}; forced release requested={rel is not None}")


# ---- G4


def phase_clocks(lc: dict, probe: dict | None) -> dict:
    p = probe or {}
    return {
        "t_stop": lc.get("stop_requested_at"), "t_accepted": phase_at(lc, "stop_accepted"), "t_importing": phase_at(lc, "importing"), "t_report": phase_at(lc, "report_written"),
        "t_imp_first": (p.get("importer") or {}).get("first_seen_at"), "t_imp_gone": next((t["at"] for t in p.get("transitions", []) if t["state"] == "SETTLING"), None),
        "t_sigterm": sigterm_at(lc), "t_closed": lc.get("session_closed_at"),
    }


def gate_deadlines(res: list, lc: dict, probe: dict | None, inputs: dict) -> None:
    c, dl = phase_clocks(lc, probe), inputs.get("phase_deadlines_s", {})
    spans = {
        "stop_accepted": ("t_stop", "t_accepted"), "flush_stop_to_importing": ("t_stop", "t_importing"), "import_importer_first_seen_to_gone": ("t_imp_first", "t_imp_gone"),
        "stop_to_report_written": ("t_stop", "t_report"), "session_close_after_server_sigterm": ("t_sigterm", "t_closed"),
    }
    rows, late, unseen = {}, [], []
    for name, (a, b) in spans.items():
        limit = dl.get(name)
        dur = c[b] - c[a] if c[a] is not None and c[b] is not None else None
        rows[name] = {"from": a, "to": b, "seconds": None if dur is None else round(dur, 3), "deadline_s": limit}
        if limit is None or dur is None:
            unseen.append(name)
        elif dur > limit or dur < 0:
            late.append(name)
    status = "FAIL" if late else ("INCONCLUSIVE" if unseen else "PASS")
    add(res, "G4.phase_deadlines", status, "; ".join(f"{n} {r['seconds']} s (deadline {r['deadline_s']})" for n, r in rows.items()) + (f"; exceeded or negative: {late}" if late else "") + (f"; boundary not observed or deadline not registered: {unseen}" if unseen else ""), phases=rows, clocks=c)


# ---- G5


def gate_importer(res: list, lc: dict, probe: dict | None) -> None:
    server = lc.get("server") or {}
    allowed = {p for p in [(server.get("api_server") or {}).get("pid")] + [e.get("pid") for e in server.get("engine_processes", [])] if p is not None}
    nsys_pid, report_at = (lc.get("nsys") or {}).get("pid"), phase_at(lc, "report_written")
    imp = (probe or {}).get("importer") or {}
    protected = {p for p in (imp.get("pid"), (imp.get("agent") or {}).get("pid")) if p is not None}
    bad_signals = []
    for a in lc.get("actions", []):
        pid = a.get("pid")
        if pid in protected:
            bad_signals.append((pid, a.get("action"), "importer or its agent"))
        elif pid == nsys_pid:
            if report_at is None or a.get("at") is None or a["at"] < report_at:
                bad_signals.append((pid, a.get("action"), "Nsight CLI before the report"))
        elif pid not in allowed:
            bad_signals.append((pid, a.get("action"), "not a recorded server process"))
    if probe is None:
        add(res, "G5.importer_observed", "FAIL" if bad_signals else "INCONCLUSIVE", f"no import-probe record; signals outside the server processes {bad_signals}")
        return
    states = [t["state"] for t in probe.get("transitions", [])]
    final = (probe.get("final") or {}).get("state")
    summary = {
        "identified": bool(imp), "verified_activity": "ACTIVE" in states, "temporarily_inactive_intervals": states.count("NO_PROGRESS"), "ambiguous": "AMBIGUOUS" in states or final == "AMBIGUOUS",
        "completed": final == "COMPLETED" and (probe.get("final") or {}).get("importer_seen") is True, "terminated": final == "TERMINATED" or "TERMINATED" in states, "final_state": final,
    }
    if bad_signals or summary["terminated"]:
        status = "FAIL"
    elif summary["identified"] and summary["verified_activity"] and summary["completed"] and not summary["ambiguous"]:
        status = "PASS"
    else:
        status = "INCONCLUSIVE"
    why = "" if status == "PASS" else ("; importer terminated" if summary["terminated"] else "; identity unknown or ambiguous" if not summary["identified"] or summary["ambiguous"] else "; importer present but activity never established from its own counters" if not summary["verified_activity"] else "; completion not observed")
    add(res, "G5.importer_observed", status, f"importer {imp.get('pid')} ({'; '.join(imp.get('matched_by', [])) or 'not identified'}), parent {(imp.get('agent') or {}).get('pid')} agent={(imp.get('agent') or {}).get('is_agent')} linked to the recorded CLI={(imp.get('agent') or {}).get('linked_to_recorded_cli')}; states {states}; final {final}; signals outside the server processes {bad_signals}{why}", summary=summary)


# ---- G6


def reading_windows(obs: list, interval: float) -> list:
    """[(lo, hi)] per observation: taken at or after its stamp and before the next stamp minus the sampler's sleep."""
    gaps = [b["at"] - a["at"] for a, b in zip(obs, obs[1:])]
    slack = max(0.0, max(gaps, default=interval) - interval)
    return [(o["at"], max(o["at"], obs[i + 1]["at"] - interval) if i + 1 < len(obs) else o["at"] + slack) for i, o in enumerate(obs)]


def release_interval(obs: list, sig: float | None, interval: float, max_gap: float) -> dict:
    """The sampled release interval relative to `sig`; `problem` names what prevents a judgement."""
    out: dict = {"observations": len(obs), "problem": None}
    held = [i for i, o in enumerate(obs) if o.get("available", True) and not o["released"]]
    if sig is None:
        out["problem"] = "the server SIGTERM time is not recorded"
        return out
    if not held:
        out["problem"] = "no 'held' observation"
        return out
    h = held[-1]
    after = [i for i in range(h + 1, len(obs)) if obs[i]["released"]]
    seq = [o["released"] for o in obs[held[0]:] if o.get("available", True)]
    out["held_again"] = any(a and not b for a, b in zip(seq, seq[1:]))
    if not after:
        out["problem"] = "no 'released' observation after the last 'held' one"
        return out
    r = after[0]
    win = reading_windows(obs, interval)
    out.update({"last_held_window": [round(win[h][0] - sig, 3), round(win[h][1] - sig, 3)], "first_released_window": [round(win[r][0] - sig, 3), round(win[r][1] - sig, 3)],
                "stamp_gap_s": round(obs[r]["at"] - obs[h]["at"], 3), "unavailable_between": sum(not o.get("available", True) for o in obs[h:r + 1]), "upper_edge_s": round(win[r][1] - sig, 3)})
    if win[h][0] > sig:
        out["note"] = "the last 'held' reading is itself after the SIGTERM"
    if obs[r]["at"] - obs[h]["at"] > max_gap or out["unavailable_between"]:
        out["problem"] = f"samples missing across the release (gap {out['stamp_gap_s']} s, limit {max_gap} s; unavailable readings {out['unavailable_between']})"
    return out


def gate_gpu_release(res: list, run: Path, lc: dict, inputs: dict) -> dict:
    g2 = _mod("e1_lifecycle_gates_v2")
    host = lc.get("host")
    every = g2.collect_observations(run, run / "gpu_observations.jsonl")
    obs = [o for o in every if o.get("source") == "e1_gpu_watch" and (not host or o.get("host") == host)]
    lim = inputs.get("limits", {})
    deadline, interval, max_gap = lim.get("gpu_release_deadline_s"), lim.get("gpu_watch_interval_s", 2), lim.get("gpu_watch_max_gap_s", 10)
    ri = release_interval(obs, sigterm_at(lc), interval, max_gap)
    census = {o["source"].replace("pinned census ", ""): o["released"] for o in every if str(o.get("source", "")).startswith("pinned census")}
    census_ok = len(census) >= 2 and all(v is True for v in census.values())
    upper = ri.get("upper_edge_s")
    if ri.get("held_again") or (upper is not None and deadline is not None and upper > deadline) or (upper is not None and upper < 0) or any(v is False for v in census.values()):
        status = "FAIL"
    elif ri["problem"] or deadline is None or not census_ok:
        status = "INCONCLUSIVE"
    else:
        status = "PASS"
    add(res, "G6.gpu_release_interval", status, f"{ri['observations']} sampler observation(s) on {host}; relative to the server SIGTERM {iso(sigterm_at(lc))}: last 'held' reading within {ri.get('last_held_window')} s, first 'released' reading within {ri.get('first_released_window')} s → released no later than {upper} s after it (deadline {deadline} s; no tighter bound is claimed); held again afterwards={ri.get('held_again')}; census files {census}" + (f"; {ri['problem']}" if ri["problem"] else "") + ("; released before the SIGTERM" if upper is not None and upper < 0 else ""), interval=ri, census=census)
    return ri


# ---- G10


def gate_bounds(res: list, lc: dict, ri: dict, reservation: dict | None, deleted: dict | None, inputs: dict) -> None:
    ceil = inputs.get("ceilings", {})
    policy, gpu, sig = lc.get("policy", {}), lc.get("gpu") or {}, sigterm_at(lc)
    launch = lc.get("launch_started_at")
    ctl_held = gpu["released_at"] - launch if gpu.get("released_at") and launch else None
    smp_held = sig + ri["upper_edge_s"] - launch if sig is not None and launch and ri.get("upper_edge_s") is not None else None
    close_s = lc["session_closed_at"] - sig if sig is not None and lc.get("session_closed_at") else None

    def epoch(text):
        try:
            return calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ"))
        except (TypeError, ValueError):
            return None

    created, gone = epoch((reservation or {}).get("creation_timestamp")), epoch((deleted or {}).get("deleted_observed_at"))
    res_min = (gone - created) / 60 if created is not None and gone is not None else None
    ceiling, close_limit, res_limit = policy.get("gpu_held_ceiling_s"), policy.get("session_close_timeout_s"), ceil.get("reservation_hard_min")
    over = [n for n, v, lim in (("GPU-held (controller)", ctl_held, ceiling), ("GPU-held (sampler upper edge)", smp_held, ceiling), ("session close", close_s, close_limit), ("reservation minutes", res_min, res_limit)) if v is not None and lim is not None and v > lim]
    unknown = [n for n, v, lim in (("GPU-held (controller)", ctl_held, ceiling), ("GPU-held (sampler upper edge)", smp_held, ceiling), ("session close", close_s, close_limit), ("reservation minutes", res_min, res_limit)) if v is None or lim is None]
    if ceiling is not None and ceil.get("gpu_held_ceiling_s") is not None and ceiling != ceil["gpu_held_ceiling_s"]:
        over.append(f"controller ceiling {ceiling} != registered {ceil['gpu_held_ceiling_s']}")
    add(res, "G10.bounds", "FAIL" if over else ("INCONCLUSIVE" if unknown else "PASS"), f"GPU-held {ctl_held and round(ctl_held)} s by the controller, at most {smp_held and round(smp_held)} s by the sampler (ceiling {ceiling}); server SIGTERM → session closed {close_s and round(close_s, 1)} s (limit {close_limit}); reservation {res_min and round(res_min, 1)} min (limit {res_limit})" + (f"; exceeded: {over}" if over else "") + (f"; not measured: {unknown}" if unknown else ""))


# ---- G11


def coverage(rows: list | None, start: float | None, end: float | None, max_gap: float) -> str | None:
    """None if `rows` cover [start, end] with no gap above max_gap, else why not."""
    if not rows:
        return "no samples"
    if start is None or end is None:
        return "the interval to cover is not known"
    ats = [r["at"] for r in rows]
    if ats[0] > start + max_gap:
        return f"first sample {ats[0] - start:.1f} s after the start"
    if ats[-1] < end - max_gap:
        return f"last sample {end - ats[-1]:.1f} s before the end"
    inside = [a for a in ats if start - max_gap <= a <= end + max_gap]
    gap = max((b - a for a, b in zip(inside, inside[1:])), default=0.0)
    return f"largest gap {gap:.1f} s exceeds {max_gap} s" if gap > max_gap else None


def gate_resources(res: list, run: Path, lc: dict, inputs: dict, post_storage: Path | None, post_memory: Path | None, post_window: tuple | None) -> None:
    ceil, lim = inputs.get("ceilings", {}), inputs.get("limits", {})
    max_gap = lim.get("sampler_max_gap_s", 30)
    run_start, run_end = lc.get("launch_started_at"), lc.get("session_closed_at")
    sets = {"run storage": (load_jsonl(run / "storage_observations.jsonl"), run_start, run_end), "run memory": (load_jsonl(run / "mem_observations.jsonl"), run_start, run_end),
            "post-run storage": (load_jsonl(post_storage) if post_storage else None, *(post_window or (None, None))), "post-run memory": (load_jsonl(post_memory) if post_memory else None, *(post_window or (None, None)))}
    unmeasured = {n: why for n, (rows, a, b) in sets.items() if (why := coverage(rows, a, b, max_gap))}
    peaks: dict = {}
    for n in ("run storage", "post-run storage"):
        for row in sets[n][0] or []:
            for g, m in row.get("groups", {}).items():
                peaks[g] = max(peaks.get(g, 0), m.get("bytes", 0))
    mem_rows = (sets["run memory"][0] or []) + (sets["post-run memory"][0] or [])
    avail = [r["mem_available_kb"] for r in mem_rows if r.get("mem_available_kb") is not None]
    rss_run = [v for r in sets["run memory"][0] or [] if r.get("importer") for v in (r["importer"].get("rss_kb"), r["importer"].get("hwm_kb")) if v is not None]
    rss_post = [v for r in sets["post-run memory"][0] or [] if r.get("importer") for v in (r["importer"].get("rss_kb"), r["importer"].get("hwm_kb")) if v is not None]
    for name, vals, source in (("in-session importer memory", rss_run, "run memory"), ("offline importer memory", rss_post, "post-run memory")):
        if not vals and source not in unmeasured:
            unmeasured[name] = "the importer was never identified by the memory sampler"
    if any(g not in peaks for g in ("temporary", "persistent")):
        unmeasured["storage groups"] = f"groups measured: {sorted(peaks)} (need temporary and persistent)"
    measured = {
        "temporary_peak_gb": None if "temporary" not in peaks else round(peaks["temporary"] / GB, 3), "persistent_peak_gb": None if "persistent" not in peaks else round(peaks["persistent"] / GB, 3),
        "min_mem_available_gb": None if not avail else round(min(avail) / KB_PER_GB, 1), "importer_peak_rss_gb_in_session": None if not rss_run else round(max(rss_run) / KB_PER_GB, 2), "importer_peak_rss_gb_offline": None if not rss_post else round(max(rss_post) / KB_PER_GB, 2),
    }
    over = []
    for key, limit, bigger_is_bad in (("temporary_peak_gb", ceil.get("temporary_storage_gb"), True), ("persistent_peak_gb", ceil.get("persistent_storage_gb"), True), ("min_mem_available_gb", ceil.get("host_mem_available_floor_gb"), False),
                                      ("importer_peak_rss_gb_in_session", ceil.get("importer_rss_ceiling_gb"), True), ("importer_peak_rss_gb_offline", ceil.get("importer_rss_ceiling_gb"), True)):
        v = measured[key]
        if v is not None and limit is not None and (v > limit if bigger_is_bad else v < limit):
            over.append(f"{key} {v} vs {limit}")
        if limit is None:
            unmeasured[f"ceiling for {key}"] = "not registered in the inputs"
    breach_files = [p.name for p in (run / "MEMORY-BREACH", run / "STORAGE-BREACH") if p.exists()]
    if breach_files:
        over.append(f"breach file(s) written during the run: {breach_files}")
    add(res, "G11.storage_and_memory", "FAIL" if over else ("INCONCLUSIVE" if unmeasured else "PASS"), f"measured peaks {measured}; ceilings temporary {ceil.get('temporary_storage_gb')} GB, persistent {ceil.get('persistent_storage_gb')} GB, importer {ceil.get('importer_rss_ceiling_gb')} GB, MemAvailable floor {ceil.get('host_mem_available_floor_gb')} GB" + (f"; exceeded: {over}" if over else "") + (f"; NOT MEASURED: {unmeasured}" if unmeasured else ""), measured=measured, unmeasured=unmeasured)


# ---- all


def evaluate(run: Path, inputs: dict, *, pin=None, provenance=None, binary_now=None, importer=None, nsys=None, scratch=None, keep_scratch=False, durable_stream=None, identity=None,
             post_storage=None, post_memory=None, reservation_deleted=None, extras=None) -> list:
    fr, g2, pinned = _mod("e1_forced_release_gates"), _mod("e1_lifecycle_gates_v2"), _mod("e1_lifecycle_gates")
    res: list = []
    lc = load(run / "profile" / "nsys" / "lifecycle.json")
    if lc is None:
        add(res, "G3.phases_in_order", "FAIL", "lifecycle.json missing or unreadable: nothing to assess")
        return res
    extras = dict(extras or {})
    probe, reservation, deleted = load(run / "import_probe.json"), load(run / "reservation.json"), load(reservation_deleted) if reservation_deleted else None
    gate_source(res, run, pin, provenance, binary_now)
    gate_host(res, run, lc, reservation, probe, {"memory sampler": next((r.get("host") for r in load_jsonl(run / "mem_observations.jsonl") or []), None), "storage sampler": next((r.get("host") for r in load_jsonl(run / "storage_observations.jsonl") or []), None)})
    logs = sorted(glob.glob(str(run / "profile" / "vllm" / "*_server.log")))
    gate_stop_request(res, lc, Path(logs[0]) if logs else None)
    gate_phases(res, lc)
    gate_deadlines(res, lc, probe, inputs)
    gate_importer(res, lc, probe)
    ri = gate_gpu_release(res, run, lc, inputs)
    # G7 / G8: the V2 implementations, unchanged, under the V3 gate names
    tmp: list = []
    t_post0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):  # their own lines carry the V2 gate names; re-printed below under the V3 names
        stream = fr.gate_stream(tmp, lc, durable_stream, g2)
        fr.gate_recovery(tmp, run, inputs, stream, importer, nsys, scratch, keep_scratch, g2, pinned)
    t_post1 = time.time()
    names = {"FR7.stream_final_stable_durable": "G7.stream_durable", "FR8.offline_import": "G8.offline_import", "FR8.offline_export": "G8.offline_export", "FR8b.recovered_trace_complete": "G8b.recovered_trace_complete"}
    for r in tmp:
        add(res, names.get(r["gate"], r["gate"]), r["status"], r["detail"], **r.get("data", {}))
    # G9
    missing = [n for n in SCI if n not in extras]
    failed = [n for n in SCI if extras.get(n) not in (None, 0)]
    ident = load(identity) if identity else None
    iv = (ident or {}).get("verdict")
    if failed or iv == "FAIL":
        status = "FAIL"
    elif missing or iv != "PASS":
        status = "INCONCLUSIVE"
    else:
        status = "PASS"
    add(res, "G9.scientific_completeness", status, f"exit codes {({n: extras.get(n) for n in SCI})} (completeness_v2 runs the four-wave plan and the routing-record checks); configuration identity {iv}" + (f"; not run: {missing}" if missing else ""))
    gate_bounds(res, lc, ri, reservation, deleted, inputs)
    gate_resources(res, run, lc, inputs, post_storage, post_memory, (t_post0, t_post1) if stream is not None and importer and nsys else None)
    closeout = extras.get("closeout")
    gone = (deleted or {}).get("gpu_pods_on_node")
    if closeout not in (None, 0) or (gone is not None and gone != 0):
        status = "FAIL"
    elif closeout is None or deleted is None or gone is None:
        status = "INCONCLUSIVE"
    else:
        status = "PASS"
    add(res, "G12.closeout", status, f"closeout exit code {closeout}; reservation deleted record {'present' if deleted else 'missing'} (GPU pods left on the node: {gone})")
    return res


def verdict_of(res: list) -> str:
    if any(r["status"] == "FAIL" for r in res):
        return "FAIL"
    return "INCONCLUSIVE" if any(r["status"] == "INCONCLUSIVE" for r in res) else "PASS"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--pin", type=Path)
    ap.add_argument("--provenance", type=Path)
    ap.add_argument("--binary", type=Path)
    ap.add_argument("--importer")
    ap.add_argument("--nsys")
    ap.add_argument("--scratch", type=Path)
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--durable-stream", type=Path)
    ap.add_argument("--identity", type=Path)
    ap.add_argument("--postrun-storage", type=Path)
    ap.add_argument("--postrun-memory", type=Path)
    ap.add_argument("--reservation-deleted", type=Path)
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=RC")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    extras = {}
    for e in args.extra:
        name, _, rc = e.partition("=")
        extras[name] = int(rc)
    now = None
    if args.binary is not None:
        now = _mod("e1_frontend_binary_gate").sha256_file(args.binary) if args.binary.is_file() else "missing"
    res = evaluate(args.run, json.loads(args.inputs.read_text()), pin=load(args.pin) if args.pin else None, provenance=load(args.provenance) if args.provenance else None, binary_now=now, importer=args.importer, nsys=args.nsys,
                   scratch=args.scratch, keep_scratch=args.keep_scratch, durable_stream=args.durable_stream, identity=args.identity, post_storage=args.postrun_storage, post_memory=args.postrun_memory,
                   reservation_deleted=args.reservation_deleted, extras=extras)
    verdict = verdict_of(res)
    counts = {s: sum(r["status"] == s for r in res) for s in ("PASS", "FAIL", "INCONCLUSIVE")}
    print(f"# V3 gates {args.run.name}: {len(res)} results, {counts['FAIL']} FAIL, {counts['INCONCLUSIVE']} INCONCLUSIVE → {verdict}")
    if args.out:
        args.out.write_text(json.dumps({"verdict": verdict, "counts": counts, "gates": res}, indent=1, default=str) + "\n")
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3}[verdict]


if __name__ == "__main__":
    sys.exit(main())
