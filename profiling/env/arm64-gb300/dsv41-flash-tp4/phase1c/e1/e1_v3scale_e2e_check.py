#!/usr/bin/env python3
"""Checks one scenario of the CPU-only end-to-end rehearsal of the V3 driver (rehearse_e2e_v3scale.sh). Not V3 evidence.

    python3 -I e1_v3scale_e2e_check.py --scenario S --root <rehearsal run root> --out DIR --driver-rc N --closeout-rc N
        [--postrun-rc N] [--archive-rc N] [--gates V3S-GATES.json] [--gates-failed-recovery V3S-GATES.json]

Every expectation below was written before the rehearsal ran. One PASS / FAIL line each; exit 1 on any FAIL.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

F, INC = "FAIL", "INCONCLUSIVE"
BASE = {"G0.source_and_executable": F, "G6.gpu_release_interval": INC, "G9.scientific_completeness": F, "G10.bounds": INC}  # a rehearsal launch; no GPU was held; no scientific profile
EXPECT = {
    "success": BASE,
    "ambiguous": {**BASE, "G4.phase_deadlines": INC, "G5.importer_observed": INC, "G11.storage_and_memory": INC},
}
LINE = "release-gpus V3 resource-limit abort (registered supervisor response)"


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def sha(p) -> str | None:
    if p is None:
        return None
    try:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def alive(pid) -> bool:
    try:
        return open(f"/proc/{pid}/stat").read().rsplit(") ", 1)[-1][0] not in "ZX"
    except (OSError, IndexError):
        return False


def registry(path: Path) -> list[dict]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    head = ["pid", "starttime", "boot_id", "uid", "label", "registered_at", "host", "registered_by", "cmd"]
    return [dict(zip(head, ln.split("\t"))) for ln in lines if ln.strip() and not ln.startswith("#")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, choices=["success", "ambiguous", "memory_breach", "storage_breach"])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--driver-rc", type=int, required=True)
    ap.add_argument("--closeout-rc", type=int, required=True)
    ap.add_argument("--postrun-rc", type=int)
    ap.add_argument("--archive-rc", type=int)
    ap.add_argument("--gates", type=Path)
    ap.add_argument("--gates-failed-recovery", type=Path)
    a = ap.parse_args(argv)
    run = a.root / "e1_v3_scale"
    lc, probe, sup = load(run / "profile" / "nsys" / "lifecycle.json") or {}, load(run / "import_probe.json") or {}, load(run / "supervisor.json") or {}
    mem, sto, src = load(run / "mem_watch.json") or {}, load(run / "storage_watch.json") or {}, load(run / "launch_source.json") or {}
    fb, bc = load(run / "profile" / "load_generator" / "frontend_binary.json") or {}, load(run / "binary_check.json") or {}
    results = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    owned, supreg = registry(a.root / "pids" / "owned.tsv"), registry(a.root / "pids" / "supervisor.tsv")
    labels = sorted(r.get("label", "").split(":")[0] for r in owned)
    check("registry.records", labels == ["driver", "gpuwatch", "importprobe", "memwatch", "storagewatch"] and [r.get("label", "").split(":")[0] for r in supreg] == ["supervisor"], f"owned registry labels {labels}; supervisor registry {[r.get('label') for r in supreg]}")
    left = [r["pid"] for r in owned + supreg if alive(r["pid"])]
    check("registry.all_gone", not left, f"registered processes still alive: {left}")
    server = lc.get("server") or {}
    mock_pids = [p for p in [(lc.get("nsys") or {}).get("pid"), (server.get("api_server") or {}).get("pid")] + [e.get("pid") for e in server.get("engine_processes", [])] if p]
    check("mock_processes.gone", bool(mock_pids) and not [p for p in mock_pids if alive(p)], f"mock Nsight CLI / server pids {mock_pids}; alive {[p for p in mock_pids if alive(p)]}")
    check("launch.rehearsal_mode_recorded", src.get("mode") == "rehearsal", f"launch_source.json mode {src.get('mode')!r} (gate G0 fails on it: never V3 evidence)")
    check("launch.real_controller_record", lc.get("schema_version") is not None and bool(lc.get("host")) and lc.get("policy", {}).get("hard_deadline_s") == 600.0 and lc.get("policy", {}).get("soft_deadline_s") == 300.0 and lc.get("policy", {}).get("server_term_grace_s") == 60.0,
          f"lifecycle.json written by the real controller: schema {lc.get('schema_version')}, policy soft/hard/grace {lc.get('policy', {}).get('soft_deadline_s')}/{lc.get('policy', {}).get('hard_deadline_s')}/{lc.get('policy', {}).get('server_term_grace_s')}")
    guard = fb.get("build_tool_guard") or {}
    if a.scenario == "storage_breach":  # the driver was terminated before it wrote its after-launch records; the rehearsal script compares the executable before and after instead
        check("no_build_tool", fb.get("mode", "").startswith("prebuilt") and fb.get("cargo_consulted") is False and fb.get("sha256") == fb.get("pinned_sha256"), f"launcher record before the launch: mode {fb.get('mode')!r}, cargo consulted {fb.get('cargo_consulted')}, executable = pinned {fb.get('sha256') == fb.get('pinned_sha256')}")
    else:
        check("no_build_tool", guard.get("installed") is True and guard.get("active") is True and guard.get("blocked") == [] and bc.get("unchanged") is True and bc.get("cargo_compiling_lines_in_run_log") == 0 and bc.get("stat_before") == bc.get("stat_after") and fb.get("unchanged") is True,
              f"build-tool guard installed/active {guard.get('installed')}/{guard.get('active')}, blocked {guard.get('blocked')}; executable unchanged {bc.get('unchanged')} (same inode/mtime/size {bc.get('stat_before') == bc.get('stat_after')}); compile lines {bc.get('cargo_compiling_lines_in_run_log')}")
    gpu = [json.loads(ln) for ln in (run / "gpu_observations.jsonl").read_text().splitlines() if ln.strip()] if (run / "gpu_observations.jsonl").exists() else []
    held = [o for o in gpu if o.get("compute_pids") or max(o.get("memory_used_mib") or [0]) > 64]
    check("no_gpu_use", bool(gpu) and not held, f"{len(gpu)} sampler observations during the run; with a compute process or memory above the idle bound: {len(held)}")
    signalled = [(x.get("pid"), x.get("action"), x.get("reason")) for x in lc.get("actions", [])]
    allowed = {p for p in mock_pids[1:]}
    imp_pid = (probe.get("importer") or {}).get("pid")
    check("signals.server_only", all(p in allowed for p, _, _ in signalled) and imp_pid not in [p for p, _, _ in signalled], f"controller signals {signalled}; recorded server pids {sorted(allowed)}; importer pid {imp_pid} never signalled")
    check("supervisor.sent_no_signal", sup.get("signals_sent_by_supervisor") == 0, f"signals sent by the supervisor itself: {sup.get('signals_sent_by_supervisor')}")

    if a.scenario in ("success", "ambiguous"):
        check("driver.rc", a.driver_rc == 0, f"driver exit code {a.driver_rc}")
        phases = [h["phase"] for h in lc.get("phase_history", [])]
        check("lifecycle.phases", phases == ["stop_requested", "stop_accepted", "capture_ended", "stream_written", "importing", "report_written"] and not lc.get("control_commands") and not lc.get("escalations"), f"phases {phases}; control lines {len(lc.get('control_commands', []))}; escalations {len(lc.get('escalations', []))}")
        check("closeout.clean", a.closeout_rc == 0, f"closeout exit code {a.closeout_rc}")
        check("supervisor.no_response", sup.get("response") is None and sup.get("monitors_effective") is True and sup.get("complete") is True and sup.get("samples", 0) > 10, f"response {sup.get('response')}; monitors effective {sup.get('monitors_effective')}; complete {sup.get('complete')}; samples {sup.get('samples')}")
        final = (lc.get("stream") or {}).get("final") or {}
        pres, dur = (lc.get("stream") or {}).get("preserved_path"), next(iter(sorted((run / "preserved").glob("*.qdstrm"))), None)
        check("stream.checksums", bool(final.get("sha256")) and sha(pres) == final.get("sha256") == sha(dur), f"controller final record {str(final.get('sha256'))[:16]}…, preserved original {str(sha(pres))[:16]}…, durable copy {str(sha(dur))[:16]}…")
        check("archive.verified", a.archive_rc == 0, f"archive script exit code {a.archive_rc}")
        check("samplers.measured", mem.get("samples", 0) > 10 and sto.get("samples", 0) > 5 and mem.get("breaches") == [] and sto.get("breaches") == [], f"memory samples {mem.get('samples')}, storage samples {sto.get('samples')}, breaches {mem.get('breaches')} {sto.get('breaches')}")
        gates = load(a.gates) or {}
        got = {g["gate"]: g["status"] for g in gates.get("gates", []) if g["status"] != "PASS"}
        check("gates.as_registered", gates.get("verdict") == "FAIL" and got == EXPECT[a.scenario] and a.postrun_rc == 1, f"verdict {gates.get('verdict')} (post-run exit {a.postrun_rc}); not PASS {got}; registered {EXPECT[a.scenario]}")
    if a.scenario == "success":
        states = [t["state"] for t in probe.get("transitions", [])]
        check("importer.unique_active_completed", bool(probe.get("importer")) and "ACTIVE" in states and (probe.get("final") or {}).get("state") == "COMPLETED" and mem.get("importer_samples", 0) > 0 and mem.get("ambiguous_samples") == 0,
              f"probe importer {imp_pid}, matched by {(probe.get('importer') or {}).get('matched_by')}, parent pid {(probe.get('importer') or {}).get('agent', {}).get('ppid')} (detached), states {states}; memory sampler importer samples {mem.get('importer_samples')}")
        fr = load(a.gates_failed_recovery) or {}
        gotf = {g["gate"]: g["status"] for g in fr.get("gates", []) if g["status"] != "PASS"}
        check("recovery_failure.detected", fr.get("verdict") == "FAIL" and gotf == {**BASE, "G8.offline_import": F, "G8b.recovered_trace_complete": F}, f"second post-run with a failing importer: verdict {fr.get('verdict')}, not PASS {gotf}")
    if a.scenario == "ambiguous":
        states = [t["state"] for t in probe.get("transitions", [])]
        check("importer.ambiguity_recorded", "importer" not in probe and "AMBIGUOUS" in states and "ACTIVE" not in states and mem.get("ambiguous_samples", 0) > 0 and mem.get("importer_samples") == 0,
              f"probe states {states}, no importer singled out; memory sampler ambiguous samples {mem.get('ambiguous_samples')}, importer samples {mem.get('importer_samples')}")
        check("importer.run_continued_under_monitors", lc.get("phase") == "report_written" and sup.get("response") is None and sup.get("monitors_effective") is True, f"final phase {lc.get('phase')}; supervisor response {sup.get('response')}; monitors effective {sup.get('monitors_effective')}")
    if a.scenario == "memory_breach":
        step = ((sup.get("response") or {}).get("steps") or [{}])[0]
        cc = lc.get("control_commands", [])
        check("breach.marker_and_cause", (run / "MEMORY-BREACH").exists() and [c.get("kind") for c in sup.get("causes", [])] == ["importer_rss_above_ceiling"], f"MEMORY-BREACH present {(run / 'MEMORY-BREACH').exists()}; supervisor causes {[(c.get('source'), c.get('kind'), c.get('value_gb'), c.get('limit_gb')) for c in sup.get('causes', [])]}")
        check("breach.delivered_through_controller", step.get("kind") == "R1 control line" and step.get("acknowledged") is True and (step.get("latency_s") or 99) <= 10 and len(sup["response"]["steps"]) == 1, f"supervisor step {step.get('kind')}, acknowledged {step.get('acknowledged')} after {step.get('latency_s')} s (bound 10 s)")
        check("breach.controller_record", [(c.get("command"), c.get("result")) for c in cc] == [(LINE, "release requested")] and cc[0].get("phase") in ("stream_written", "importing"), f"control commands {[(c.get('command')[:40], c.get('result'), c.get('phase')) for c in cc]}")
        acts = [(x.get("action"), x.get("at")) for x in lc.get("actions", [])]
        term = next((t for act, t in acts if act == "SIGTERM"), None)
        kill = max((t for act, t in acts if act == "SIGKILL"), default=None)
        delay = None if not cc or term is None or not (sup.get("causes") or [{}])[0].get("breach_at") else term - sup["causes"][0]["breach_at"]
        check("breach.bounded_response", term is not None and delay is not None and delay <= 15 and (kill is None or kill - term <= 62), f"breach sample → identity-verified SIGTERM to the API server {None if delay is None else round(delay, 1)} s (bound 15 s); SIGTERM → SIGKILL {None if kill is None else round(kill - term, 1)} s (grace 60 s)")
        check("breach.importer_isolated", bool(imp_pid) and (probe.get("final") or {}).get("state") == "COMPLETED" and lc.get("phase") == "report_written", f"importer {imp_pid} was not signalled and finished: probe final {(probe.get('final') or {}).get('state')}, lifecycle phase {lc.get('phase')}")
        gates = load(a.gates) or {}
        got = {g["gate"]: g["status"] for g in gates.get("gates", []) if g["status"] != "PASS"}
        check("breach.run_fails", gates.get("verdict") == "FAIL" and got.get("G3.phases_in_order") == F and got.get("G11.storage_and_memory") == F, f"verdict {gates.get('verdict')}; not PASS {got}")
    if a.scenario == "storage_breach":
        steps = (sup.get("response") or {}).get("steps") or [{}]
        check("breach.marker_and_cause", (run / "STORAGE-BREACH").exists() and [c.get("kind") for c in sup.get("causes", [])] == ["persistent_storage_above_ceiling"], f"STORAGE-BREACH present {(run / 'STORAGE-BREACH').exists()}; supervisor causes {[(c.get('source'), c.get('kind'), c.get('value_gb'), c.get('limit_gb')) for c in sup.get('causes', [])]}")
        check("breach.registry_cleanup", [s.get("kind") for s in steps] == ["R2 registry cleanup"] and steps[0].get("rc") in (0, 2) and lc.get("stop_requested_at") is None and not (run / "profile" / "nsys" / "lifecycle.control").exists(),
              f"supervisor steps {[s.get('kind') for s in steps]} because {steps[0].get('why')}; cleanup rc {steps[0].get('rc')}: {steps[0].get('result_line')}; stop requested {lc.get('stop_requested_at')}; control file written {(run / 'profile' / 'nsys' / 'lifecycle.control').exists()}")
        log = load(a.root / "cpu" / "supervisor_cleanup.jsonl") if (a.root / "cpu" / "supervisor_cleanup.jsonl").exists() else None
        if log is None and (a.root / "cpu" / "supervisor_cleanup.jsonl").exists():
            log = json.loads((a.root / "cpu" / "supervisor_cleanup.jsonl").read_text().splitlines()[-1])
        acts = (log or {}).get("actions", [])
        reg_pids = {int(r["pid"]) for r in owned}
        plan_owned = {o["pid"] for o in ((log or {}).get("plan") or {}).get("owned", [])}
        check("breach.identity_verified_registry_only", bool(acts) and not ((log or {}).get("plan") or {}).get("refused") and {x["pid"] for x in acts if x.get("action") in ("SIGTERM", "SIGKILL")} <= plan_owned and reg_pids & plan_owned == {p for p in reg_pids if p in plan_owned},
              f"cleanup signalled {sorted({x['pid'] for x in acts if x.get('action') in ('SIGTERM', 'SIGKILL')})} ⊆ verified owned tree of {len(plan_owned)} processes (registered roots {sorted(reg_pids)}); refused {len(((log or {}).get('plan') or {}).get('refused', []))}; survivors {(log or {}).get('survivors')}")
        resp = sup.get("response") or {}
        check("breach.bounded_response", resp.get("seconds_from_first_cause_seen") is not None and resp["seconds_from_first_cause_seen"] <= 45, f"first cause seen → cleanup finished {resp.get('seconds_from_first_cause_seen')} s (grace 30 s + SIGKILL pass)")
        check("driver.did_not_finish", a.driver_rc != 0 and "rc=" not in "".join(ln for ln in (run / "run.log").read_text(errors="replace").splitlines() if ln.startswith("[e1_v3_scale] rc=")), f"driver exit code {a.driver_rc} (terminated by the cleanup); no launcher result line")
        check("closeout.registry_clean_not_clean_overall", a.closeout_rc == 1, f"closeout exit code {a.closeout_rc} (NOT CLEAN is required after an abort: non-terminal phase, breach marker, supervisor response)")
    ok = all(r["ok"] for r in results)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "E2E-CHECK.json").write_text(json.dumps({"scenario": a.scenario, "ok": ok, "checks": results, "pid_namespace_note": os.uname().nodename}, indent=1) + "\n")
    print(f"# scenario {a.scenario}: {sum(r['ok'] for r in results)} of {len(results)} checks pass → {'MET' if ok else 'NOT MET'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
