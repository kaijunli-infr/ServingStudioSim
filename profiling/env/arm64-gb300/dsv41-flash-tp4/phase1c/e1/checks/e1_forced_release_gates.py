#!/usr/bin/env python3
"""Prospective acceptance gates for V2, the forced-release-during-import validation (frozen before any V2 launch).

    python3 -I checks/e1_forced_release_gates.py --run <run root>/<pass> --inputs E1-V2-INPUTS.json
        --importer <QdstrmImporter> --nsys <nsys> --durable-stream <copy on durable storage>
        [--reservation reservation.json] [--identity identity_v2.json] [--extra name=rc ...] [--out V2-GATES.json]

These gates are independent of the V1 acceptance record: they do not read it and do not change the pinned V1 gates.
They share helper code with checks/e1_lifecycle_gates_v2.py (trace summary) and checks/e1_host_identity.py.

  H    host identity: lifecycle record = run log = reservation node = every GPU observation
  FR1  exactly one profiler-stop request left the launcher; if it ended in an error, that happened after the release
  FR2  the pre-registered trigger: exactly one control line, the registered text, accepted while the phase was `importing`
  FR3  the import was active when the release was requested (controller's release context, phase history, and the
       trigger probe's own observation of the live Nsight process)
  FR4  no destructive signal to the importer: every signalled pid is a recorded server process; the Nsight process is
       signalled only after a report was written, if at all
  FR5  GPUs observed released within the deadline after the release request, by independent observations that bracket
       it (a held observation before, a released one after, none held later); controller time reported separately
  FR6  the session ended: terminal phase and Nsight gone (a still-running or hung importer is a FAIL)
  FR7  preserved stream: controller's final record present; the file is available, stable over two full reads, equal
       to the final record and to the durable copy. Stability and checksum equality are not completeness
  FR8  offline import of the durable copy; FR8b the recovered trace is semantically complete: iterations in band,
       NVTX = 4 x iterations, kernel rows in band, four processes with kernels; and, where an in-session export exists,
       equal to it in kernel rows, NVTX ranges, iterations, capture boundaries, per-process counts, every table's row
       count and capture metadata
  FR9  bounds: GPU-held within the ceiling, session closed within the timeout after the release, no escalation
  S.*  configuration identity (policy) and the exit codes of the scientific checkers / closeout

Verdict: FAIL if any gate FAILs; INCONCLUSIVE if none fails but one could not be evaluated; PASS otherwise.
A PASS of FR7 never substitutes for FR8b. Stdlib only; read-only on the run directory (scratch under --scratch).
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import glob
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TERMINAL = ("report_written", "stop_failed")
API_STOPPING = re.compile(rb"^\(APIServer pid=\d+\).*\[api_router\.py:\d+\] Stopping profiler", re.M)


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


def iso(t) -> str:
    return "-" if t is None else time.strftime("%H:%M:%SZ", time.gmtime(t))


def phase_at(lc: dict, phase: str):
    return next((h["at"] for h in lc.get("phase_history", []) if h["phase"] == phase), None)


def gate_stop_request(res: list, lc: dict, server_log: Path | None) -> None:
    http = lc.get("http", {})
    rel = (lc.get("gpu") or {}).get("release_requested_at")
    phases = [h["phase"] for h in lc.get("phase_history", [])]
    status, returned = http.get("status"), http.get("returned_at")
    error_after_release = isinstance(status, str) and status.startswith("error") and rel is not None and returned is not None and returned >= rel
    stopping = len(API_STOPPING.findall(server_log.read_bytes())) if server_log and server_log.is_file() else None
    ok = http.get("requests_sent") == 1 and phases.count("stop_requested") == 1 and (status == "ok" or error_after_release) and stopping == 1
    detail = f"requests_sent={http.get('requests_sent')}, duplicates suppressed={http.get('duplicate_requests_suppressed', 0)}, stop_requested phases={phases.count('stop_requested')}, http={status!r} returned {iso(returned)} (release requested {iso(rel)}; an error is accepted only at or after the release), API-server 'Stopping profiler' lines={stopping}"
    add(res, "FR1.one_stop_request", "INCONCLUSIVE" if stopping is None and ok is False and http.get("requests_sent") == 1 else ("PASS" if ok else "FAIL"), detail)


def gate_trigger(res: list, lc: dict, inputs: dict) -> None:
    want = inputs.get("trigger", {}).get("control_line")
    cmds = lc.get("control_commands", [])
    ok = len(cmds) == 1 and cmds[0].get("command") == want and cmds[0].get("result") == "release requested" and cmds[0].get("phase") == "importing"
    add(res, "FR2.registered_trigger", "PASS" if ok else "FAIL", f"control commands recorded {[(c.get('command'), c.get('result'), c.get('phase')) for c in cmds]}; expected exactly one: {want!r} → 'release requested' in phase 'importing'")


def gate_import_active(res: list, lc: dict, probe: dict | None) -> None:
    gpu = lc.get("gpu") or {}
    ctx = gpu.get("release_context") or {}
    rel = gpu.get("release_requested_at")
    imp, rep = phase_at(lc, "importing"), phase_at(lc, "report_written")
    controller_ok = ctx.get("phase") == "importing" and ctx.get("nsys_alive") is True and ctx.get("report_written") is False and rel is not None and imp is not None and imp <= rel and (rep is None or rep > rel)
    nsys_pid = (lc.get("nsys") or {}).get("pid")
    probe_ok = bool(probe) and probe.get("saw_phase") == "importing" and probe.get("nsys_alive") is True and probe.get("nsys_pid") == nsys_pid and probe.get("wrote_trigger_at") is not None and rel is not None and probe["wrote_trigger_at"] <= rel + 1.0
    status = "PASS" if controller_ok and probe_ok else ("INCONCLUSIVE" if controller_ok and probe is None else "FAIL")
    add(res, "FR3.import_active_at_release", status, f"controller: release requested {iso(rel)} in phase {ctx.get('phase')!r}, import {ctx.get('import_progress_pct')}%, Nsight alive={ctx.get('nsys_alive')}, report written={ctx.get('report_written')}; importing since {iso(imp)}, report at {iso(rep)}; probe (independent of the controller): {probe}", release_context=ctx, probe=probe)


def gate_no_signal_to_importer(res: list, lc: dict) -> None:
    nsys_pid = (lc.get("nsys") or {}).get("pid")
    server = lc.get("server") or {}
    server_pids = {(server.get("api_server") or {}).get("pid")} | {p.get("pid") for p in server.get("engine_processes", []) or []}
    server_pids.discard(None)
    rep = phase_at(lc, "report_written")
    bad, sigs = [], []
    for a in lc.get("actions", []):
        act = str(a.get("action", ""))
        if "SIG" not in act:
            continue
        sigs.append((a.get("pid"), act, a.get("reason")))
        if a.get("pid") == nsys_pid or "nsys" in act or "pgid" in act:
            if rep is None or a.get("at", 0) < rep:
                bad.append(a)
        elif a.get("pid") not in server_pids:
            bad.append(a)
    release_sigs = [s for s in sigs if str(s[2]).startswith("operator: release-gpus")]
    ok = not bad and bool(release_sigs) and nsys_pid is not None and bool(server_pids)
    add(res, "FR4.no_signal_to_importer", "PASS" if ok else "FAIL", f"Nsight pid {nsys_pid}; recorded server pids {sorted(server_pids)}; signals {sigs}; release signals {len(release_sigs)}; disallowed {bad}")


def gate_gpu_release(res: list, run: Path, lc: dict, inputs: dict, g2, host: str | None) -> None:
    obs = [o for o in g2.collect_observations(run, run / "gpu_observations.jsonl") if not host or o.get("host") == host]
    gpu = lc.get("gpu") or {}
    rel, ctrl = gpu.get("release_requested_at"), gpu.get("released_at")
    deadline = inputs.get("limits", {}).get("gpu_release_deadline_s")
    held_before = [o["at"] for o in obs if not o["released"] and rel is not None and o["at"] <= rel]
    released_after = [o["at"] for o in obs if o["released"] and rel is not None and o["at"] > rel]
    first_rel = min(released_after, default=None)
    held_later = [o["at"] for o in obs if not o["released"] and first_rel is not None and o["at"] > first_rel]
    last_held = max((o["at"] for o in obs if not o["released"] and (first_rel is None or o["at"] < first_rel)), default=None)
    ok = rel is not None and deadline is not None and bool(held_before) and first_rel is not None and first_rel - rel <= deadline and not held_later
    add(res, "FR5.gpu_release_within_deadline", "PASS" if ok else "FAIL", f"{len(obs)} independent observation(s) on {host}; release requested {iso(rel)}; GPUs last observed held {iso(last_held)}, first observed released {iso(first_rel)} → release in ({iso(last_held)}, {iso(first_rel)}], at most {None if first_rel is None or rel is None else round(first_rel - rel, 1)} s after the request (deadline {deadline} s); held observations before the request: {len(held_before)}; held after the first released one: {len(held_later)}; controller time, separate: {iso(ctrl)}", last_held=last_held, first_released=first_rel, release_requested_at=rel, controller_released_at=ctrl)


def gate_session_end(res: list, lc: dict) -> str | None:
    phase = lc.get("phase")
    ended = phase in TERMINAL and (lc.get("nsys") or {}).get("alive") is False and lc.get("session_closed_at") is not None
    outcome = {"report_written": "A: the importer finished and wrote a report", "stop_failed": "B: no report; recovery from the preserved stream"}.get(phase)
    add(res, "FR6.session_ended", "PASS" if ended else "FAIL", f"final phase {phase!r}, Nsight alive={(lc.get('nsys') or {}).get('alive')}, session closed {iso(lc.get('session_closed_at'))} → outcome {outcome or 'none (importer still running or hung: registry cleanup procedure, stream stays preserved)'}")
    return phase if ended else None


def gate_stream(res: list, lc: dict, durable: Path | None, g2, hasher=None) -> dict | None:
    hasher = hasher or g2.sha256_file
    st = lc.get("stream") or {}
    final = st.get("final") or {}
    recorded = Path(st["preserved_path"]) if st.get("preserved_path") else None
    if not final.get("available"):
        add(res, "FR7.stream_final_stable_durable", "FAIL", f"controller has no final stream record: {final or None}")
        return None
    if durable is None or not durable.is_file():
        add(res, "FR7.stream_final_stable_durable", "FAIL", f"durable copy not available: {durable}")
        return None
    before = g2.stat_tuple(durable)
    sha1 = hasher(durable)
    sha2 = hasher(durable)
    after = g2.stat_tuple(durable)
    stable = before[:2] == after[:2] and sha1 == sha2
    matches_final = sha1 == final.get("sha256") and after[0] == final.get("size")
    original_sha = hasher(recorded) if recorded is not None and recorded.is_file() else None
    original_ok = original_sha is None or original_sha == sha1
    ok = stable and matches_final and original_ok
    add(res, "FR7.stream_final_stable_durable", "PASS" if ok else "FAIL", f"controller's final record (taken {iso(final.get('taken_at'))}, after session close): {final.get('size')} B sha256 {final.get('sha256')}; durable copy {durable}: {after[0]} B sha256 {sha1}; two full reads identical={stable}; equals the final record={matches_final}; tray original {'absent' if original_sha is None else 'present'} and equal={original_ok}; {len(st.get('observations') or [])} earlier observation(s) kept; this is not a completeness statement (FR8b decides)", sha256=sha1, size=after[0], observations=st.get("observations"))
    return {"path": durable, "sha256": sha1, "size": after[0]} if ok else None


def gate_recovery(res: list, run: Path, inputs: dict, stream: dict | None, importer, nsys, scratch, keep, g2, pinned) -> None:
    exp = inputs.get("expected", {})
    if stream is None:
        add(res, "FR8.offline_import", "FAIL", "not attempted: FR7 did not establish a final, stable, durable stream")
        add(res, "FR8b.recovered_trace_complete", "FAIL", "not attempted (FR8)")
        return
    if not importer or not nsys:
        add(res, "FR8.offline_import", "INCONCLUSIVE", "QdstrmImporter / nsys not supplied")
        add(res, "FR8b.recovered_trace_complete", "INCONCLUSIVE", "not run (FR8)")
        return
    scratch = scratch or Path(tempfile.mkdtemp(prefix="v2-recovery-"))
    scratch.mkdir(parents=True, exist_ok=True)
    out_rep, out_sql = scratch / "recovery.nsys-rep", scratch / "recovery.sqlite"
    try:
        t0 = time.time()
        subprocess.run([importer, "-i", str(stream["path"]), "-o", str(out_rep)], check=True, capture_output=True, text=True, timeout=7200)
        add(res, "FR8.offline_import", "PASS" if out_rep.stat().st_size > 0 else "FAIL", f"offline import of the durable stream rc 0 in {time.time() - t0:.0f} s → {out_rep.stat().st_size} B")
        subprocess.run([nsys, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(out_sql), str(out_rep)], check=True, capture_output=True, text=True, timeout=7200)
        rec = g2.trace_summary(out_sql, pinned)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, Exception) as error:  # noqa: BLE001
        detail = getattr(error, "stderr", "") or str(error)
        step = "FR8.offline_export" if any(r["gate"] == "FR8.offline_import" for r in res) else "FR8.offline_import"
        add(res, step, "FAIL", f"incomplete or failed import/export of the preserved stream: {str(detail)[:300]}")
        add(res, "FR8b.recovered_trace_complete", "FAIL", "not evaluable (FR8)")
        return
    finally:
        if not keep:
            for f in (out_rep, out_sql):
                f.unlink(missing_ok=True)
    per = exp.get("nvtx_per_iteration", 4)
    it_band, row_band = exp.get("iterations_band"), exp.get("kernel_rows_band")
    procs = rec.get("kernels_per_process") or []
    absolute = {
        "iterations_in_band": bool(it_band) and it_band[0] <= rec["distinct_iterations"] <= it_band[1],
        "nvtx_is_per_iteration_multiple": rec["nvtx_forward_ranges"] == per * rec["distinct_iterations"],
        "kernel_rows_in_band": bool(row_band) and row_band[0] <= rec["kernel_rows"] <= row_band[1],
        "four_processes_with_kernels": len(procs) == exp.get("ranks", 4) and min(procs, default=0) > 0,
        "capture_boundaries_present": bool(rec.get("kernel_boundaries_ns")) and bool(rec.get("nvtx_forward_boundaries_ns")),
    }
    in_sql = sorted(glob.glob(str(run / "profile" / "nsys" / "*.sqlite")))
    relative, table_diff = None, None
    if in_sql:
        ins = g2.trace_summary(Path(in_sql[0]), pinned)
        relative = {
            "kernel_rows": rec["kernel_rows"] == ins["kernel_rows"],
            "nvtx_forward_ranges": rec["nvtx_forward_ranges"] == ins["nvtx_forward_ranges"],
            "distinct_iterations": rec["distinct_iterations"] == ins["distinct_iterations"],
            "iteration_index_range": rec["iteration_index_range"] == ins["iteration_index_range"],
            "kernel_boundaries_ns": rec.get("kernel_boundaries_ns") == ins.get("kernel_boundaries_ns"),
            "nvtx_forward_boundaries_ns": rec["nvtx_forward_boundaries_ns"] == ins["nvtx_forward_boundaries_ns"],
            "kernels_per_process": procs == ins.get("kernels_per_process"),
            "every_table_row_count": rec["table_rows"] == ins["table_rows"],
            "capture_metadata": all(rec["metadata"].get(t) == ins["metadata"].get(t) for t in ("ANALYSIS_DETAILS", "META_DATA_CAPTURE", "TARGET_INFO_SESSION_START_TIME")),
            "export_product_and_schema": g2.export_rows(rec, False) == g2.export_rows(ins, False),
        }
        table_diff = {t: (ins["table_rows"].get(t), rec["table_rows"].get(t)) for t in sorted(set(ins["table_rows"]) | set(rec["table_rows"])) if ins["table_rows"].get(t) != rec["table_rows"].get(t)}
    ok = all(absolute.values()) and (relative is None or all(relative.values()))
    add(
        res,
        "FR8b.recovered_trace_complete",
        "PASS" if ok else "FAIL",
        f"recovered trace: {rec['distinct_iterations']} iterations (band {it_band}), NVTX forward ranges {rec['nvtx_forward_ranges']}, kernel rows {rec['kernel_rows']} (band {row_band}), kernels per process {procs}, kernel boundaries {rec.get('kernel_boundaries_ns')} ns; absolute checks {absolute}; "
        + (f"in-session export exists → equality checks {relative}, differing table row counts {table_diff}" if relative is not None else "no in-session export exists (outcome B): equality with an original is not applicable; the absolute checks and the scientific checkers on the recovered profile decide"),
        absolute=absolute,
        relative=relative,
        table_row_differences=table_diff,
        recovered={k: v for k, v in rec.items() if k != "table_rows"},
    )


def gate_bounds(res: list, lc: dict) -> None:
    policy, gpu = lc.get("policy", {}), lc.get("gpu") or {}
    held = gpu["released_at"] - lc["launch_started_at"] if gpu.get("released_at") and lc.get("launch_started_at") else None
    rel, closed = gpu.get("release_requested_at"), lc.get("session_closed_at")
    close_s = closed - rel if rel and closed else None
    esc = lc.get("escalations", [])
    ok = held is not None and policy.get("gpu_held_ceiling_s") is not None and held <= policy["gpu_held_ceiling_s"] and close_s is not None and close_s <= policy.get("session_close_timeout_s", 0) and not esc
    add(res, "FR9.bounds", "PASS" if ok else "FAIL", f"GPU-held {held and round(held)} s (ceiling {policy.get('gpu_held_ceiling_s')}); release request → session closed {close_s and round(close_s)} s (timeout {policy.get('session_close_timeout_s')}); escalations {[(e.get('kind'), e.get('phase')) for e in esc]}; warnings {[w.get('message') for w in lc.get('warnings', [])][:6]}")


def evaluate(run: Path, inputs: dict, *, importer=None, nsys=None, scratch=None, keep_scratch=False, durable_stream=None, reservation=None, identity=None, extras=None) -> list:
    g2, hostmod, pinned = _mod("e1_lifecycle_gates_v2"), _mod("e1_host_identity"), _mod("e1_lifecycle_gates")
    res: list = []
    lc = load(run / "profile" / "nsys" / "lifecycle.json")
    if lc is None:
        add(res, "FR0.record", "FAIL", "lifecycle.json missing or unreadable")
        return res
    add(res, "FR0.record", "PASS", f"lifecycle.json schema {lc.get('schema_version')} phase {lc.get('phase')}")
    h = hostmod.cross_check(lc, hostmod.run_log_host(run), load(reservation or run / "reservation.json"), hostmod.gpu_observation_hosts(run, None))
    add(res, "H.host_identity", h["status"], h["detail"], sources=h["sources"])
    logs = sorted(glob.glob(str(run / "profile" / "vllm" / "*_server.log")))
    gate_stop_request(res, lc, Path(logs[0]) if logs else None)
    gate_trigger(res, lc, inputs)
    gate_import_active(res, lc, load(run / "probe.json"))
    gate_no_signal_to_importer(res, lc)
    gate_gpu_release(res, run, lc, inputs, g2, lc.get("host"))
    gate_session_end(res, lc)
    stream = gate_stream(res, lc, durable_stream, g2)
    gate_recovery(res, run, inputs, stream, importer, nsys, scratch, keep_scratch, g2, pinned)
    gate_bounds(res, lc)
    if identity is not None:
        rep = load(identity) or {}
        verdict = rep.get("verdict", "INCONCLUSIVE")
        add(res, "S.config_identity", verdict if verdict in ("PASS", "FAIL") else "INCONCLUSIVE", f"{identity.name}: {verdict}; {len(rep.get('normalized', []))} normalized field(s), each listed in the identity report")
    else:
        add(res, "S.config_identity", "INCONCLUSIVE", "no identity report supplied")
    for name, rc in (extras or {}).items():
        add(res, f"S.{name}", "PASS" if rc == 0 else "FAIL", f"exit code {rc}")
    return res


def verdict_of(res: list) -> str:
    if any(r["status"] == "FAIL" for r in res):
        return "FAIL"
    return "INCONCLUSIVE" if any(r["status"] == "INCONCLUSIVE" for r in res) else "PASS"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--importer")
    ap.add_argument("--nsys")
    ap.add_argument("--scratch", type=Path)
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--durable-stream", type=Path)
    ap.add_argument("--reservation", type=Path)
    ap.add_argument("--identity", type=Path)
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    extras = {}
    for e in args.extra:
        name, _, rc = e.partition("=")
        extras[name] = int(rc) if rc.isdigit() else 99
    res = evaluate(args.run, json.loads(args.inputs.read_text()), importer=args.importer, nsys=args.nsys, scratch=args.scratch, keep_scratch=args.keep_scratch, durable_stream=args.durable_stream, reservation=args.reservation, identity=args.identity, extras=extras)
    verdict = verdict_of(res)
    print(f"# V2 forced-release gates {args.run.name}: {len(res)} gates, {sum(r['status'] == 'FAIL' for r in res)} FAIL, {sum(r['status'] == 'INCONCLUSIVE' for r in res)} INCONCLUSIVE → {verdict}" + ("" if verdict == "PASS" else " → STOP (no retry, no advance)"))
    if args.out:
        args.out.write_text(json.dumps({"run": str(args.run), "verdict": verdict, "gates": res}, indent=1, default=str) + "\n")
    return {"PASS": 0, "FAIL": 1}.get(verdict, 3)


if __name__ == "__main__":
    sys.exit(main())
