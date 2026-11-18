#!/usr/bin/env python3
"""Lifecycle acceptance gates for a patched-launcher capture (V1 packet §7, §9).

    python3 -I checks/e1_lifecycle_gates.py --run <run root>/<pass> --inputs E1-V1-INPUTS.json
        [--importer <QdstrmImporter> --nsys <nsys>] [--extra name=rc ...] [--out gates.json]

Reads `<run>/profile/nsys/lifecycle.json` (the controller's record), `<run>/profile/profile_result.json`,
the launch record and the independent GPU-release census files `<run>/gpu_release_census_*.json` (written by
checks/e1_gpu_release_census.py from nvidia-smi, outside the controller), and decides PASS/FAIL per gate.
Every FAIL is a STOP for the programme: it is never permission to advance to the next validation step.
`--extra name=rc` folds in the exit codes of the scientific checkers (completeness v2, nsys validation,
config identity, recipe test) so the verdict covers both families. `--importer` + `--nsys` run the
preserved-stream recovery proof: offline import of the preserved `.qdstrm` into a scratch report, `nsys export
--type sqlite` of that report, and the same counts the in-session validation used (NVTX forward ranges →
iterations, CUPTI kernel rows, kernel span) compared with the in-session values; size is reported, not judged.
Without them the proof is reported NOT RUN. Stdlib only; read-only on the run directory except for the scratch
report under --scratch (default: a temp dir).
"""

from __future__ import annotations

# ruff: noqa: E501  (operational checker: long diagnostic strings stay on one line for grep-ability)

import argparse
import glob
import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ORDER = [
    "capturing",
    "stop_requested",
    "stop_accepted",
    "capture_ended",
    "stream_written",
    "importing",
    "report_written",
]


def gate(results: list, name: str, ok: bool | None, detail: str) -> None:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    results.append({"gate": name, "status": status, "detail": detail})
    print(f"{status:4s}  {name}: {detail}")


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# the in-session validation's queries (alignment/profiler/nsys_capture.py::validate_export), repeated here
# verbatim so the recovery proof measures the recovered report with the same yardstick
NVTX_FORWARD_SQL = """
    SELECT COUNT(*) FROM NVTX_EVENTS n
    LEFT JOIN StringIds s ON n.textId = s.id
    WHERE COALESCE(n.text, s.value) LIKE 'vllm_iteration(%): forward'
       OR COALESCE(n.text, s.value) LIKE 'sglang_iteration(%): forward'
"""
KERNEL_SQL = "SELECT MIN(start), MAX(end), COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"


def validate_sqlite(sqlite_path: Path) -> dict:
    con = sqlite3.connect(str(sqlite_path))
    try:
        (n_forward,) = con.execute(NVTX_FORWARD_SQL).fetchone()
        has_kernels = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_KERNEL'"
        ).fetchone()
        kmin, kmax, krows = con.execute(KERNEL_SQL).fetchone() if has_kernels else (None, None, 0)
    finally:
        con.close()
    return {
        "nvtx_forward_ranges": int(n_forward or 0),
        "kernel_rows": int(krows or 0),
        "kernel_span_ms": ((kmax - kmin) / 1e6) if (kmin and kmax) else 0.0,
        "ok": bool(n_forward) and bool(krows),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(
    run: Path,
    inputs: dict,
    *,
    importer: str | None,
    scratch: Path | None,
    extras: dict,
    nsys: str | None = None,
) -> list:
    res: list = []
    exp = inputs.get("expected", {})
    lc = load(run / "profile" / "nsys" / "lifecycle.json")
    pr = load(run / "profile" / "profile_result.json")
    launches = glob.glob(str(run / "profile" / "vllm" / "*_launch.json"))
    launch = load(Path(launches[0])) if launches else None
    if lc is None:
        gate(res, "L0.record", False, "lifecycle.json missing or unreadable → STOP")
        return res
    gate(
        res,
        "L0.record",
        True,
        f"lifecycle.json schema {lc.get('schema_version')} phase {lc.get('phase')}",
    )

    # L1 stop-request idempotence
    http = lc.get("http", {})
    sent, dup = http.get("requests_sent"), http.get("duplicate_requests_suppressed", 0)
    gate(
        res,
        "L1.idempotent_stop",
        sent == 1 and dup >= 1 and http.get("status") == "ok",
        f"requests_sent={sent} duplicates_suppressed={dup} http={http.get('status')} "
        "(the runner's finally path re-requests the stop; exactly one POST must have left)",
    )

    # L2 phase ordering
    hist = lc.get("phase_history", [])
    phases = [h["phase"] for h in hist]
    idx = [ORDER.index(p) for p in phases if p in ORDER]
    increasing = idx == sorted(idx) and len(set(idx)) == len(idx)
    times = [h["at"] for h in hist]
    monotone = times == sorted(times)
    complete = phases[-1:] == ["report_written"] and {
        "stop_requested",
        "stop_accepted",
        "capture_ended",
        "importing",
    } <= set(phases)
    gate(
        res,
        "L2.phase_order",
        increasing
        and monotone
        and complete
        and (lc.get("stop_requested_at") or 0) <= (times[0] if times else 0),
        f"history {phases}; strictly increasing={increasing} timestamps monotone={monotone} final=report_written={phases[-1:] == ['report_written']}",
    )
    stream_phase = "stream_written" in phases
    gate(
        res,
        "L2b.stream_phase",
        stream_phase,
        "Nsight announced the raw stream" if stream_phase else "no stream_written phase",
    )

    # L3 report completeness
    rep = lc.get("report", {})
    rep_path = Path(rep.get("path") or "")
    rep_ok = rep_path.is_file() and rep_path.stat().st_size > 0
    val = (pr or {}).get("validation", {}) if pr else {}
    rows = val.get("kernel_rows")
    band = exp.get("kernel_rows_band")
    rows_ok = band is None or (rows is not None and band[0] <= rows <= band[1])
    gate(
        res,
        "L3.report_complete",
        rep_ok and bool(val.get("ok")) and rows_ok,
        f"report {rep_path.name} size={rep.get('size')} validation.ok={val.get('ok')} kernel_rows={rows} band={band}",
    )
    iters = None
    if pr and pr.get("parsed_nsys"):
        iters = (pr.get("validation") or {}).get("nvtx_forward_ranges")
    nvtx = val.get("nvtx_forward_ranges")
    it_band = exp.get("iterations_band")
    if nvtx is not None and it_band:
        n_iter = nvtx / exp.get("nvtx_per_iteration", 4)
        gate(
            res,
            "L3b.coverage",
            it_band[0] <= n_iter <= it_band[1],
            f"NVTX forward ranges {nvtx} → {n_iter:.0f} iterations (band {it_band}); both waves expected (b=8 at 0 s, b=48 at 18.6 s)",
        )
    else:
        gate(
            res, "L3b.coverage", None, f"nvtx_forward_ranges={nvtx} band={it_band} (not evaluable)"
        )
    del iters

    # L4 raw stream preserved: recorded, present on disk now, and byte-complete (on-disk size == recorded
    # preserved size == the stream size Nsight finished writing); its sha256 goes on record for the proof
    st = lc.get("stream", {})
    preserved = st.get("preserved_path")
    pres_path = Path(preserved) if preserved else None
    on_disk = pres_path.stat().st_size if pres_path and pres_path.is_file() else None
    pres_ok = (
        bool(st.get("path"))
        and on_disk is not None
        and on_disk > 0
        and st.get("preserved_size") == on_disk
        and (st.get("size") is None or st.get("size") == on_disk)
    )
    pres_sha = sha256_file(pres_path) if pres_ok else None
    gate(
        res,
        "L4.stream_preserved",
        pres_ok,
        f"path={st.get('path')} discovered_by={st.get('discovered_by')} preserved={preserved} "
        f"size={st.get('size')} preserved_size={st.get('preserved_size')} on_disk={on_disk} sha256={pres_sha}",
    )

    # L5 nothing signalled nsys before the report; server stopped by verified identity only
    nsys_pid = (lc.get("nsys") or {}).get("pid")
    api = ((lc.get("server") or {}).get("api_server") or {}).get("pid")
    rep_at = rep.get("written_at") or float("inf")
    bad = []
    server_sigs = []
    for a in lc.get("actions", []):
        act = str(a.get("action", ""))
        if "SIG" not in act:
            continue
        pid = a.get("pid")
        if pid == nsys_pid or "nsys" in act:
            if a.get("at", 0) < rep_at or a.get("reason") not in (
                "nsys lingering after report",
                "nsys ignored SIGINT",
            ):
                bad.append(a)
        elif pid == api:
            server_sigs.append(a)
        else:
            bad.append(a)
    gate(
        res,
        "L5.no_signal_to_importer",
        not bad,
        f"actions={len(lc.get('actions', []))}; server signals={[(s.get('action'), s.get('reason')) for s in server_sigs]}; "
        f"disallowed={bad}",
    )
    gate(
        res,
        "L5b.server_stopped_by_identity",
        any(s.get("reason") == "shutdown" and s.get("action") == "SIGTERM" for s in server_sigs),
        f"api_server pid {api}: {[(s.get('action'), s.get('reason')) for s in server_sigs]}",
    )

    # L6 GPU-held and finalisation within the authorized bounds
    policy = lc.get("policy", {})
    held = None
    if lc.get("gpu", {}).get("released_at") and lc.get("launch_started_at"):
        held = lc["gpu"]["released_at"] - lc["launch_started_at"]
    ceiling = policy.get("gpu_held_ceiling_s")
    fin = None
    if rep.get("written_at") and lc.get("stop_requested_at"):
        fin = rep["written_at"] - lc["stop_requested_at"]
    gate(
        res,
        "L6.gpu_held_within_ceiling",
        held is not None and ceiling is not None and held <= ceiling,
        f"GPU-held {held and round(held)} s (launch → release), ceiling {ceiling} s",
    )
    gate(
        res,
        "L6b.finalisation_within_hard_deadline",
        fin is not None and fin <= policy.get("hard_deadline_s", 0),
        f"stop request → report {fin and round(fin)} s; soft {policy.get('soft_deadline_s')} hard {policy.get('hard_deadline_s')}",
    )
    # L6c independent release verification: nvidia-smi census files written by the driver after the launcher
    # exited (checks/e1_gpu_release_census.py); the controller's timestamp alone is not accepted
    census_files = sorted(glob.glob(str(run / "gpu_release_census_*.json")))
    samples = [c for c in (load(Path(f)) for f in census_files) if c]
    released_at = lc.get("gpu", {}).get("released_at")
    good = [
        c
        for c in samples
        if c.get("verdict", {}).get("released") is True
        and released_at is not None
        and c.get("at", 0) >= released_at
        and (not lc.get("host") or c.get("host") == lc.get("host"))
    ]
    last = samples[-1] if samples else None
    gate(
        res,
        "L6c.gpu_release_verified_independently",
        bool(samples) and last in good,
        f"{len(samples)} census sample(s) {[Path(f).name for f in census_files]}; last: "
        + (
            f"{last.get('at_iso')} released={last.get('verdict', {}).get('released')} compute_pids={[c.get('pid') for c in last.get('compute_apps', [])]} "
            f"memory.used MiB={[g.get('memory_used_mib') for g in last.get('gpus', [])]} (bound {last.get('max_used_mib')}); "
            f"after controller release_at={released_at}: {bool(last.get('at', 0) >= (released_at or float('inf')))}"
            if last
            else "none (driver did not run the census)"
        ),
    )

    # L7 control-file probe: exactly the two probe lines, each accepted, nothing else
    probe = inputs.get("control_probe", {}).get("lines", [])
    cmds = lc.get("control_commands", [])
    got = [(c.get("command"), c.get("result")) for c in cmds]
    want_ok = len(cmds) == len(probe) and all(
        c.get("command") == p and not str(c.get("result", "")).startswith("rejected")
        for c, p in zip(cmds, probe)
    )
    gate(res, "L7.control_probe", want_ok, f"recorded {got}; expected {probe}")

    # L8 escalations, warnings, extension accounting
    esc = lc.get("escalations", [])
    warn = lc.get("warnings", [])
    ext = (lc.get("deadlines") or {}).get("hard_extension_s", 0)
    accepted_ext = 0.0
    for c in cmds:
        words = str(c.get("command", "")).split()
        if words[:1] == ["extend"] and not str(c.get("result", "")).startswith("rejected"):
            try:
                accepted_ext += float(words[1])
            except (IndexError, ValueError):
                pass
    gate(
        res,
        "L8.no_escalation",
        not esc,
        f"escalations={[(e.get('kind'), round(e.get('elapsed_s', 0))) for e in esc]}",
    )
    fatal = [
        w
        for w in warn
        if "could not hard-link" in w.get("message", "")
        or "not a descendant" in w.get("message", "")
    ]
    gate(
        res,
        "L8b.warnings",
        not fatal,
        f"{len(warn)} warning(s): {[w.get('message') for w in warn][:6]}",
    )
    gate(
        res,
        "L8c.extension_accounting",
        ext == accepted_ext,
        f"hard_extension_s={ext} == sum of accepted extend commands {accepted_ext} (no silent extension)",
    )

    # L9 session closed, nothing left
    closed = lc.get("session_closed_at") is not None and not (lc.get("nsys") or {}).get(
        "alive", True
    )
    server_dead = (lc.get("server") or {}).get("alive") is False
    gate(
        res,
        "L9.session_closed",
        closed and server_dead,
        f"nsys alive={(lc.get('nsys') or {}).get('alive')} server alive={(lc.get('server') or {}).get('alive')} closed_at={lc.get('session_closed_at')}",
    )

    # L10 preserved-stream recovery proof (offline import + export + the in-session counts, CPU)
    if importer and pres_ok:
        scratch = scratch or Path(tempfile.mkdtemp(prefix="v1-recovery-"))
        scratch.mkdir(parents=True, exist_ok=True)
        out_rep = scratch / "recovery.nsys-rep"
        out_sql = scratch / "recovery.sqlite"
        try:
            subprocess.run(
                [importer, "-i", str(pres_path), "-o", str(out_rep)],
                check=True,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            size = out_rep.stat().st_size
            ratio = size / rep.get("size", 1)
            gate(
                res,
                "L10.recovery_proof",
                size > 0,
                f"offline import of {pres_path.name} (sha256 {pres_sha[:16]}) → {size} B vs in-session {rep.get('size')} B (ratio {ratio:.4f}; informational)",
            )
            if nsys:
                subprocess.run(
                    [
                        nsys,
                        "export",
                        "--type",
                        "sqlite",
                        "--force-overwrite",
                        "true",
                        "-o",
                        str(out_sql),
                        str(out_rep),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
                rec = validate_sqlite(out_sql)
                r_iter = rec["nvtx_forward_ranges"] / exp.get("nvtx_per_iteration", 4)
                span_in = val.get("kernel_span_ms")
                span_ok = (
                    span_in is None
                    or span_in == 0
                    or abs(rec["kernel_span_ms"] - span_in) / span_in <= 0.001
                )
                same = (
                    rec["ok"]
                    and rec["kernel_rows"] == rows
                    and rec["nvtx_forward_ranges"] == nvtx
                    and (not it_band or it_band[0] <= r_iter <= it_band[1])
                    and span_ok
                )
                gate(
                    res,
                    "L10b.recovery_semantic_completeness",
                    same,
                    f"recovered report: kernel_rows {rec['kernel_rows']} (in-session {rows}), NVTX forward ranges {rec['nvtx_forward_ranges']} (in-session {nvtx}) → {r_iter:.0f} iterations (band {it_band}), "
                    f"kernel span {rec['kernel_span_ms']:.1f} ms (in-session {span_in}); equality required",
                )
            else:
                gate(
                    res,
                    "L10b.recovery_semantic_completeness",
                    False,
                    "no --nsys for the sqlite export of the recovered report → STOP",
                )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
            detail = getattr(error, "stderr", "") or str(error)
            gate(
                res,
                "L10.recovery_proof",
                False,
                f"offline import/export failed: {str(detail)[:300]}",
            )
    else:
        gate(
            res,
            "L10.recovery_proof",
            None,
            "NOT RUN (no --importer, or L4 failed: no byte-complete preserved stream)",
        )

    # launch record sanity
    if launch is not None:
        gate(
            res,
            "R.launch_record",
            True,
            f"nsys {launch.get('nsys_profiler', {}).get('version')} exe {launch.get('argv', [''])[0]}",
        )
    else:
        gate(res, "R.launch_record", False, "no *_launch.json")

    # scientific checkers folded in
    for name, rc in extras.items():
        gate(res, f"S.{name}", rc == 0, f"exit code {rc}")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", required=True, type=Path, help="<run root>/<pass>")
    ap.add_argument(
        "--inputs", required=True, type=Path, help="E1-V1-INPUTS.json (expected bands, probe lines)"
    )
    ap.add_argument("--importer", help="QdstrmImporter for the recovery proof")
    ap.add_argument(
        "--nsys",
        help="nsys CLI for the sqlite export of the recovered report (same install as --importer)",
    )
    ap.add_argument("--scratch", type=Path, help="where the recovery-proof report goes")
    ap.add_argument("--extra", action="append", default=[], help="name=rc of a scientific checker")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    inputs = json.loads(args.inputs.read_text())
    extras = {}
    for e in args.extra:
        name, _, rc = e.partition("=")
        extras[name] = int(rc) if rc.isdigit() else 99
    res = evaluate(
        args.run,
        inputs,
        importer=args.importer,
        scratch=args.scratch,
        extras=extras,
        nsys=args.nsys,
    )
    fails = [r for r in res if r["status"] == "FAIL"]
    verdict = (
        "PASS"
        if not fails
        else "FAIL → STOP (no advance to the next validation step; review required)"
    )
    print(f"# lifecycle gates {args.run.name}: {len(res)} gates, {len(fails)} FAIL → {verdict}")
    if args.out:
        args.out.write_text(
            json.dumps(
                {"run": str(args.run), "gates": res, "verdict": "PASS" if not fails else "STOP"},
                indent=1,
            )
        )
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
