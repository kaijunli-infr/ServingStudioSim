#!/usr/bin/env python3
"""Corrected lifecycle gates (V1 remediation; the pinned checks/e1_lifecycle_gates.py is unchanged and still decides
the ORIGINAL verdict). This file evaluates evidence retrospectively or for a future run.

    python3 -I checks/e1_lifecycle_gates_v2.py --run <run root>/<pass> --inputs E1-V1-INPUTS.json
        --importer <QdstrmImporter> --nsys <nsys> [--stream <durable copy>] [--observations obs.jsonl]
        [--identity identity_v2.json] [--extra name=rc ...] [--original-gates V1-GATES.json] [--out retro.json]

Unchanged gates (L0, L2, L2b, L3, L3b, L5, L5b, L6, L6b, L7, L8, L8b, L8c, L9, R) are taken from the pinned
evaluator as they are. Corrected gates:

  L1   exactly one effective profiler stop: one request sent and answered ok; zero or more duplicates, each recorded
       as suppressed; every finalisation phase occurs once; the server log (independent of the controller) shows one
       "Stopping profiler" and one "Profiler stopped".
  L4   preserved stream: finalisation confirmed (report written, Nsight gone, session closed), the stream file is
       available, its size and sha256 are taken twice around a full read and do not change, and it was not modified
       after the session closed. No comparison with an earlier size sample. Stability is NOT taken as completeness.
  L10  offline import of that stream; L10b semantic comparison of the recovered report with the in-session export:
       kernel rows, NVTX forward ranges, distinct iterations, capture boundaries, per-process kernel counts, every
       table's row count, capture metadata. Mandatory: without the tools or the in-session export the verdict is not PASS.
  L6c  GPU release as an observed interval from independently timestamped observations (pinned census files plus any
       JSONL observations): (last "held" observation, first "released" observation]. The controller's timestamp is kept
       separately and must not be contradicted. Host taken from the run log, not from lifecycle.json.
  S.config_identity_v2  from checks/e1_launch_identity_v2.py (explicit normalization policy).

Verdict: FAIL if any gate FAILs, else INCONCLUSIVE if any gate could not be evaluated, else PASS. Stdlib only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import glob
import hashlib
import importlib.util
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPLACED = ("L1.", "L4.", "L6c.", "L10.", "L10b.", "S.")
FINAL_PHASES = ("stop_requested", "stop_accepted", "capture_ended", "importing", "report_written")
ITER_RE = re.compile(r"_iteration\((\d+)\): forward$")


def load_pinned():
    spec = importlib.util.spec_from_file_location("e1_lifecycle_gates_pinned", HERE / "e1_lifecycle_gates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add(results: list, name: str, status: str, detail: str, **data) -> None:
    results.append({"gate": name, "status": status, "detail": detail, **({"data": data} if data else {})})
    print(f"{status:12s} {name}: {detail}")


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stat_tuple(path: Path) -> tuple:
    st = path.stat()
    return (st.st_size, st.st_mtime_ns, st.st_ino)


# ---------------------------------------------------------------- L1
def gate_l1(res: list, lc: dict, server_log: Path | None) -> None:
    http = lc.get("http", {})
    sent, dup = http.get("requests_sent"), int(http.get("duplicate_requests_suppressed", 0) or 0)
    phases = [h["phase"] for h in lc.get("phase_history", [])]
    counts = {p: phases.count(p) for p in FINAL_PHASES}
    once = all(c == 1 for c in counts.values()) and phases.count("stream_written") <= 1
    log_counts = None
    if server_log and server_log.is_file():
        text = server_log.read_bytes()
        log_counts = (text.count(b"] Stopping profiler"), text.count(b"] Profiler stopped"))
    ok = sent == 1 and http.get("status") == "ok" and dup >= 0 and once and log_counts == (1, 1)
    detail = (
        f"requests_sent={sent} http={http.get('status')}; duplicates recorded as suppressed={dup} (zero or more permitted); "
        f"finalisation phases each once={once} {counts}; server log 'Stopping profiler'/'Profiler stopped' lines={log_counts}"
    )
    if log_counts is None and sent == 1 and http.get("status") == "ok" and once:
        add(res, "L1.one_effective_stop", "INCONCLUSIVE", detail + " — server log not available for the independent count", duplicates=dup)
    else:
        add(res, "L1.one_effective_stop", "PASS" if ok else "FAIL", detail, duplicates=dup, phase_counts=counts, server_log_counts=log_counts)


# ---------------------------------------------------------------- L4
def gate_l4(res: list, lc: dict, stream_override: Path | None, hasher=sha256_file) -> dict | None:
    st = lc.get("stream", {})
    finalised = (
        lc.get("phase") == "report_written"
        and (lc.get("nsys") or {}).get("alive") is False
        and lc.get("session_closed_at") is not None
    )
    recorded = Path(st["preserved_path"]) if st.get("preserved_path") else None
    path = stream_override or recorded
    info = {"recorded_path": st.get("path"), "recorded_size_sample": st.get("size"), "recorded_preserved_size": st.get("preserved_size"), "verified_path": str(path) if path else None}
    if not finalised:
        add(res, "L4.stream_final_and_stable", "FAIL", f"finalisation not confirmed: phase={lc.get('phase')} nsys alive={(lc.get('nsys') or {}).get('alive')} session_closed_at={lc.get('session_closed_at')}", **info)
        return None
    if path is None or not path.is_file() or path.stat().st_size == 0:
        add(res, "L4.stream_final_and_stable", "FAIL", f"preserved stream not available at {path}", **info)
        return None
    before = stat_tuple(path)
    sha1 = hasher(path)
    mid = stat_tuple(path)
    sha2 = hasher(path)
    after = stat_tuple(path)
    stable = before[:2] == mid[:2] == after[:2] and sha1 == sha2
    closed = lc["session_closed_at"]
    # a copy keeps the source mtime (cp -p); the stream must not have been written after the session closed
    not_modified_later = before[1] / 1e9 <= closed + 1.0
    live = Path(st["path"]) if st.get("path") else None
    live_gone_or_same = live is None or not live.exists() or (recorded is not None and recorded.exists() and live.stat().st_ino == recorded.stat().st_ino)
    same_as_recorded = True
    recorded_sha = None
    if stream_override and recorded is not None and recorded.is_file():
        recorded_sha = hasher(recorded)
        same_as_recorded = recorded_sha == sha1
    ok = stable and not_modified_later and live_gone_or_same and same_as_recorded
    info.update(final_size=after[0], sha256=sha1, mtime=before[1] / 1e9, session_closed_at=closed, recorded_copy_sha256=recorded_sha)
    add(
        res,
        "L4.stream_final_and_stable",
        "PASS" if ok else "FAIL",
        f"finalisation confirmed; {path.name}: final size {after[0]} B, sha256 {sha1}; two full reads identical and size/mtime unchanged around them={stable}; "
        f"last modified {before[1] / 1e9 - closed:+.1f} s relative to session close (must be ≤ +1)={not_modified_later}; Nsight's own stream file gone or same inode={live_gone_or_same}; "
        + (f"equal to the stream at the recorded path={same_as_recorded}; " if stream_override else "")
        + f"earlier size sample in the record {st.get('size')} B is reported only, not compared; completeness is decided by L10b, not here",
        **info,
    )
    return {"path": path, "sha256": sha1, "size": after[0]} if ok else None


# ---------------------------------------------------------------- L10 / L10b
def trace_summary(sqlite_path: Path, pinned) -> dict:
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        out = dict(pinned.validate_sqlite(sqlite_path))
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1")]
        out["table_rows"] = {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
        rows = con.execute(
            """SELECT COALESCE(n.text, s.value), n.start, n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
               WHERE COALESCE(n.text, s.value) LIKE 'vllm_iteration(%): forward' OR COALESCE(n.text, s.value) LIKE 'sglang_iteration(%): forward'"""
        ).fetchall()
        iters = {int(m.group(1)) for t, _, _ in rows if (m := ITER_RE.search(t or ""))}
        out["distinct_iterations"] = len(iters)
        out["iteration_index_range"] = [min(iters), max(iters)] if iters else None
        out["nvtx_forward_boundaries_ns"] = [min(r[1] for r in rows), max(r[2] for r in rows if r[2] is not None)] if rows else None
        if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
            out["kernel_boundaries_ns"] = list(con.execute("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone())
            out["kernels_per_process"] = sorted(r[1] for r in con.execute("SELECT globalPid, COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY 1"))
        meta = {}
        for t in ("ANALYSIS_DETAILS", "META_DATA_CAPTURE", "TARGET_INFO_SESSION_START_TIME", "META_DATA_EXPORT"):
            if t in tables:
                meta[t] = sorted(json.dumps(r, default=str) for r in con.execute(f'SELECT * FROM "{t}"'))
        out["metadata"] = meta
    finally:
        con.close()
    return out


def gate_l10(res: list, run: Path, exp: dict, stream: dict | None, importer: str | None, nsys: str | None, scratch: Path | None, keep: bool, pinned) -> None:
    pr = load(run / "profile" / "profile_result.json") or {}
    val = pr.get("validation") or {}
    in_sql = sorted(glob.glob(str(run / "profile" / "nsys" / "*.sqlite")))
    if stream is None:
        add(res, "L10.offline_import", "FAIL", "not attempted: L4 did not establish a final, stable stream")
        add(res, "L10b.recovery_semantic_completeness", "FAIL", "not attempted (L10)")
        return
    if not importer or not nsys:
        add(res, "L10.offline_import", "INCONCLUSIVE", "QdstrmImporter / nsys not supplied: the mandatory recovery proof was not run")
        add(res, "L10b.recovery_semantic_completeness", "INCONCLUSIVE", "not run (L10)")
        return
    scratch = scratch or Path(tempfile.mkdtemp(prefix="v1-recovery-v2-"))
    scratch.mkdir(parents=True, exist_ok=True)
    out_rep, out_sql = scratch / "recovery.nsys-rep", scratch / "recovery.sqlite"
    try:
        t0 = time.time()
        subprocess.run([importer, "-i", str(stream["path"]), "-o", str(out_rep)], check=True, capture_output=True, text=True, timeout=7200)
        t1 = time.time()
        size = out_rep.stat().st_size
        in_rep = sorted(glob.glob(str(run / "profile" / "nsys" / "*.nsys-rep")))
        in_size = Path(in_rep[0]).stat().st_size if in_rep else None
        add(res, "L10.offline_import", "PASS" if size > 0 else "FAIL", f"offline import of the stream (sha256 {stream['sha256'][:16]}…, {stream['size']} B) rc 0 in {t1 - t0:.0f} s → report {size} B (in-session {in_size} B; sizes informational)", recovered_report_size=size, in_session_report_size=in_size)
        subprocess.run([nsys, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(out_sql), str(out_rep)], check=True, capture_output=True, text=True, timeout=7200)
        rec = trace_summary(out_sql, pinned)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, sqlite3.Error) as error:
        detail = getattr(error, "stderr", "") or str(error)
        step = "L10.offline_export" if any(r["gate"] == "L10.offline_import" for r in res) else "L10.offline_import"
        add(res, step, "FAIL", f"offline import/export failed: {str(detail)[:300]}")
        add(res, "L10b.recovery_semantic_completeness", "FAIL", "not evaluable (L10)")
        return
    finally:
        if not keep:
            for f in (out_rep, out_sql):
                f.unlink(missing_ok=True)
    if not in_sql:
        add(res, "L10b.recovery_semantic_completeness", "INCONCLUSIVE", "in-session sqlite export not available: boundaries, tables and metadata cannot be compared", recovered=rec)
        return
    ins = trace_summary(Path(in_sql[0]), pinned)
    band = exp.get("iterations_band")
    per = exp.get("nvtx_per_iteration", 4)
    checks = {
        "kernel_rows": rec["kernel_rows"] == ins["kernel_rows"] == val.get("kernel_rows"),
        "nvtx_forward_ranges": rec["nvtx_forward_ranges"] == ins["nvtx_forward_ranges"] == val.get("nvtx_forward_ranges"),
        "distinct_iterations": rec["distinct_iterations"] == ins["distinct_iterations"],
        "iteration_index_range": rec["iteration_index_range"] == ins["iteration_index_range"],
        "iterations_in_band": bool(band) and band[0] <= rec["nvtx_forward_ranges"] / per <= band[1] and rec["distinct_iterations"] * per == rec["nvtx_forward_ranges"],
        "kernel_boundaries_ns": rec.get("kernel_boundaries_ns") == ins.get("kernel_boundaries_ns"),
        "nvtx_forward_boundaries_ns": rec["nvtx_forward_boundaries_ns"] == ins["nvtx_forward_boundaries_ns"],
        "kernels_per_process": rec.get("kernels_per_process") == ins.get("kernels_per_process"),
        "every_table_row_count": rec["table_rows"] == ins["table_rows"],
        "capture_metadata": all(rec["metadata"].get(t) == ins["metadata"].get(t) for t in ("ANALYSIS_DETAILS", "META_DATA_CAPTURE", "TARGET_INFO_SESSION_START_TIME")),
        "export_metadata": rec["metadata"].get("META_DATA_EXPORT") == ins["metadata"].get("META_DATA_EXPORT"),
    }
    table_diff = {t: (ins["table_rows"].get(t), rec["table_rows"].get(t)) for t in sorted(set(ins["table_rows"]) | set(rec["table_rows"])) if ins["table_rows"].get(t) != rec["table_rows"].get(t)}
    ok = all(checks.values())
    add(
        res,
        "L10b.recovery_semantic_completeness",
        "PASS" if ok else "FAIL",
        f"recovered vs in-session export: kernel rows {rec['kernel_rows']} / {ins['kernel_rows']}; NVTX forward ranges {rec['nvtx_forward_ranges']} / {ins['nvtx_forward_ranges']}; "
        f"distinct iterations {rec['distinct_iterations']} / {ins['distinct_iterations']} (index range {rec['iteration_index_range']}, band {band}); "
        f"kernel boundaries {rec.get('kernel_boundaries_ns')} / {ins.get('kernel_boundaries_ns')} ns; NVTX boundaries {rec['nvtx_forward_boundaries_ns']} / {ins['nvtx_forward_boundaries_ns']} ns; "
        f"kernels per process {rec.get('kernels_per_process')}; tables {len(rec['table_rows'])} / {len(ins['table_rows'])}, differing row counts {table_diff}; checks {checks}",
        checks=checks,
        table_row_differences=table_diff,
        recovered={k: v for k, v in rec.items() if k != "table_rows"},
        in_session={k: v for k, v in ins.items() if k != "table_rows"},
        table_rows=rec["table_rows"],
    )


# ---------------------------------------------------------------- L6c
def run_host(run: Path) -> str | None:
    try:
        m = re.search(r"^\[[^\]]+\] start \S+ on (\S+)", (run / "run.log").read_text(errors="replace"), re.M)
    except OSError:
        return None
    return m.group(1) if m else None


def collect_observations(run: Path, extra: Path | None) -> list:
    obs = []
    for f in sorted(glob.glob(str(run / "gpu_release_census_*.json"))):
        c = load(Path(f))
        if c:
            obs.append({"at": c.get("at"), "host": c.get("host"), "source": f"pinned census {Path(f).name}", "memory_used_mib": [g.get("memory_used_mib") for g in c.get("gpus", [])], "compute_pids": [a.get("pid") for a in c.get("compute_apps", [])], "available": c.get("verdict", {}).get("available", True)})
    if extra and extra.is_file():
        for line in extra.read_text().splitlines():
            if line.strip():
                obs.append(json.loads(line))
    for o in obs:
        mem = [m for m in o.get("memory_used_mib") or [] if m is not None]
        o["released"] = bool(o.get("available", True)) and not o.get("compute_pids") and bool(mem) and max(mem) <= 64
    return sorted((o for o in obs if o.get("at") is not None), key=lambda o: o["at"])


def gate_l6c(res: list, run: Path, lc: dict, extra: Path | None) -> None:
    iso = lambda t: "-" if t is None else time.strftime("%H:%M:%SZ", time.gmtime(t))  # noqa: E731
    host = run_host(run)
    obs = collect_observations(run, extra)
    ctrl = (lc.get("gpu") or {}).get("released_at")
    wrong_host = [o for o in obs if host and o.get("host") and o["host"] != host]
    use = [o for o in obs if o not in wrong_host]
    held = [o for o in use if not o["released"]]
    released = [o for o in use if o["released"]]
    last_held = max((o["at"] for o in held), default=None)
    first_rel = min((o["at"] for o in released if last_held is None or o["at"] > last_held), default=None)
    held_after_release = [o for o in held if first_rel is not None and o["at"] > first_rel]
    contradicted = ctrl is not None and (last_held is not None and last_held > ctrl + 1.0)
    ok = (
        host is not None
        and not wrong_host
        and first_rel is not None
        and not held_after_release
        and ctrl is not None
        and not contradicted
        and any(o["at"] >= ctrl for o in released)
    )
    interval = {"last_held_observation": last_held, "first_released_observation": first_rel, "controller_released_at": ctrl, "run_host": host, "observations": len(use), "exact_release_established": False}
    add(
        res,
        "L6c.gpu_release_observed_interval",
        "PASS" if ok else "FAIL",
        f"run host {host} (from run.log); {len(use)} independent observation(s) on that host, {len(wrong_host)} from another host; GPUs observed held last at {iso(last_held)} and released first at {iso(first_rel)} → "
        f"release lies in ({iso(last_held)}, {iso(first_rel)}] ({'no held observation: only an upper bound' if last_held is None else f'{first_rel - last_held:.0f} s wide' if first_rel else 'no released observation'}); "
        f"controller timestamp, kept separately: {iso(ctrl)} ({'contradicted by a later held observation' if contradicted else 'not contradicted; inside the interval' if (ctrl and first_rel and (last_held is None or last_held <= ctrl) and ctrl <= first_rel) else 'not contradicted'}); held observations after the first released one: {len(held_after_release)}",
        **interval,
        sources=sorted({o.get("source", "?") for o in use}),
    )


# ---------------------------------------------------------------- main
def evaluate(run: Path, inputs: dict, *, importer=None, nsys=None, scratch=None, keep_scratch=False, stream=None, observations=None, identity=None, extras=None) -> list:
    pinned = load_pinned()
    res: list = []
    lc = load(run / "profile" / "nsys" / "lifecycle.json")
    if lc is None:
        add(res, "L0.record", "FAIL", "lifecycle.json missing or unreadable")
        return res
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        base = pinned.evaluate(run, inputs, importer=None, scratch=None, extras={})
    for r in base:
        if not r["gate"].startswith(REPLACED):
            add(res, r["gate"], r["status"], r["detail"] + "  [pinned gate, unchanged]")
    logs = sorted(glob.glob(str(run / "profile" / "vllm" / "*_server.log")))
    gate_l1(res, lc, Path(logs[0]) if logs else None)
    final = gate_l4(res, lc, stream)
    gate_l10(res, run, inputs.get("expected", {}), final, importer, nsys, scratch, keep_scratch, pinned)
    gate_l6c(res, run, lc, observations)
    if identity is not None:
        rep = load(identity)
        verdict = (rep or {}).get("verdict", "INCONCLUSIVE")
        add(res, "S.config_identity_v2", verdict if verdict in ("PASS", "FAIL") else "INCONCLUSIVE", f"{identity.name}: {verdict}; {len((rep or {}).get('normalized', []))} normalized field(s): {[(n['scope'] + '.' + n['field'], n['category']) for n in (rep or {}).get('normalized', [])]}")
    else:
        add(res, "S.config_identity_v2", "INCONCLUSIVE", "no identity report supplied")
    for name, rc in (extras or {}).items():
        add(res, f"S.{name}", "PASS" if rc == 0 else "FAIL", f"exit code {rc}")
    return res


def verdict_of(res: list) -> str:
    if any(r["status"] == "FAIL" for r in res):
        return "FAIL"
    if any(r["status"] == "INCONCLUSIVE" for r in res):
        return "INCONCLUSIVE"
    return "PASS"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--importer")
    ap.add_argument("--nsys")
    ap.add_argument("--scratch", type=Path)
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--stream", type=Path, help="durable copy of the preserved stream (must equal the recorded one if that still exists)")
    ap.add_argument("--observations", type=Path, help="JSONL of independent GPU observations (at, host, source, memory_used_mib, compute_pids)")
    ap.add_argument("--identity", type=Path, help="report of e1_launch_identity_v2.py")
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--original-gates", type=Path, help="the pinned post-run's gates file; referenced, never modified")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    extras = {}
    for e in args.extra:
        name, _, rc = e.partition("=")
        extras[name] = int(rc) if rc.isdigit() else 99
    res = evaluate(args.run, json.loads(args.inputs.read_text()), importer=args.importer, nsys=args.nsys, scratch=args.scratch, keep_scratch=args.keep_scratch, stream=args.stream, observations=args.observations, identity=args.identity, extras=extras)
    verdict = verdict_of(res)
    original = None
    if args.original_gates and args.original_gates.is_file():
        o = json.loads(args.original_gates.read_text())
        original = {"file": str(args.original_gates), "sha256": sha256_file(args.original_gates), "verdict": o.get("verdict"), "failed_gates": [g["gate"] for g in o.get("gates", []) if g["status"] == "FAIL"]}
        print(f"# original verdict under the pinned gates (unchanged record, sha256 {original['sha256'][:16]}…): {original['verdict']} — failed {original['failed_gates']}")
    print(f"# corrected gates {args.run.name}: {len(res)} gates, {sum(r['status'] == 'FAIL' for r in res)} FAIL, {sum(r['status'] == 'INCONCLUSIVE' for r in res)} INCONCLUSIVE → retrospective evidence assessment: {verdict}")
    if args.out:
        args.out.write_text(json.dumps({"run": str(args.run), "gates_version": "v2 (corrected L1, L4, L6c, L10, L10b, config identity)", "original_pinned_verdict": original, "retrospective_assessment": verdict, "gates": res}, indent=1, default=str) + "\n")
    return {"PASS": 0, "FAIL": 1}.get(verdict, 3)


if __name__ == "__main__":
    sys.exit(main())
