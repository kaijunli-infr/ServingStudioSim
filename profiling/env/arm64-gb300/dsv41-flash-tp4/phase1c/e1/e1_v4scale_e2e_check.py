#!/usr/bin/env python3
"""Checks one scenario of the CPU-only end-to-end rehearsal of the V4 driver (rehearse_e2e_v4scale.sh). Not V4 evidence.

    python3 -I e1_v4scale_e2e_check.py --scenario S --root <rehearsal run root> --out DIR --driver-rc N --closeout-rc N
        [--postrun-rc N] [--archive-rc N] [--gates V4S-GATES.json] [--gates-failed-recovery V4S-GATES.json]

Every expectation below was written before the rehearsal ran. One PASS / FAIL line each; exit 1 on any FAIL.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import calendar
import hashlib
import json
import os
import sys
import time
from pathlib import Path

F, INC = "FAIL", "INCONCLUSIVE"
BASE = {"G0.source_and_executable": F, "G6.gpu_release_interval": INC, "G9.scientific_completeness": F, "G10.bounds": INC,  # a rehearsal launch; no GPU was held; no scientific profile
        "G13.workload_and_recipe_identity": INC, "G14.four_copy_trace_in_run": F, "G15.request_iteration_routing_rank_coverage": INC}  # no launch record (mock launcher); the rehearsal copy of the config is renamed; no completeness record
EXPECT = {
    "success": BASE,
    "launch_late_permitted": BASE,
    "complete_near_guard": BASE,
    "supervisor_lost": {**BASE, "G11.storage_and_memory": INC},  # the supervisor's record is not complete
    "ambiguous": {**BASE, "G4.phase_deadlines": INC, "G5.importer_observed": INC, "G11.storage_and_memory": INC},
}
LINE = "release-gpus V4 resource-limit abort (registered supervisor response)"
ABORTED = ("storage_breach", "reservation_guard", "ceiling_before_stop", "export_near_guard", "guard_and_storage", "sampler_lost")  # ended by the supervisor's registry cleanup
REHEARSAL_GUARD_S, REGISTERED_GUARD_S, REGISTERED_RUN_BUDGET_S = 420.0, 4680.0, 4210.0


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
    ap.add_argument("--scenario", required=True, choices=["success", "ambiguous", "memory_breach", "storage_breach", "reservation_guard", *"launch_late_permitted launch_refused ceiling_before_stop complete_near_guard export_near_guard guard_and_storage sampler_lost supervisor_lost".split()])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--driver-rc", type=int, required=True)
    ap.add_argument("--closeout-rc", type=int, required=True)
    ap.add_argument("--postrun-rc", type=int)
    ap.add_argument("--archive-rc", type=int)
    ap.add_argument("--gates", type=Path)
    ap.add_argument("--gates-failed-recovery", type=Path)
    a = ap.parse_args(argv)
    run = a.root / "e1_v4_scale"
    lc, probe, sup = load(run / "profile" / "nsys" / "lifecycle.json") or {}, load(run / "import_probe.json") or {}, load(run / "supervisor.json") or {}
    mem, sto, src = load(run / "mem_watch.json") or {}, load(run / "storage_watch.json") or {}, load(run / "launch_source.json") or {}
    fb, bc = load(run / "profile" / "load_generator" / "frontend_binary.json") or {}, load(run / "binary_check.json") or {}
    results = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    runlog = (run / "run.log").read_text(errors="replace") if (run / "run.log").exists() else ""
    try:
        created = float(calendar.timegm(time.strptime((load(run / "reservation.json") or {})["creation_timestamp"], "%Y-%m-%dT%H:%M:%SZ")))
    except (KeyError, ValueError, TypeError):
        created = None
    rule = next((ln for ln in runlog.splitlines() if "launch eligibility" in ln), "")
    try:
        rule_age = float(rule.split("reservation age ")[1].split(" s")[0])
    except (IndexError, ValueError):
        rule_age = None
    if a.scenario == "launch_refused":
        owned = registry(a.root / "pids" / "owned.tsv")
        latest = REGISTERED_GUARD_S - REGISTERED_RUN_BUDGET_S
        check("refusal.exit_code", a.driver_rc == 7, f"driver exit code {a.driver_rc} (7 = launch refused by the reservation budget)")
        check("refusal.rule_and_reason_recorded", rule.startswith("FAIL  launch eligibility") and "REFUSED" in rule and f"run budget {REGISTERED_RUN_BUDGET_S:.0f} s" in rule and f"({REGISTERED_GUARD_S:.0f} s)" in rule and "LAUNCH REFUSED" in runlog and rule_age is not None and latest < rule_age <= latest + 120,
              f"run.log: {rule[:230]!r}; age at the check {rule_age} s, latest permitted {latest:.0f} s (registered guard and run budget, not rehearsal values)")
        check("refusal.nothing_started", [r.get("label", "").split(":")[0] for r in owned] == ["driver"] and not (a.root / "pids" / "supervisor.tsv").exists() and not lc and not sup and not src and not (run / "gpu_observations.jsonl").exists() and not (run / "mem_observations.jsonl").exists() and not (run / "profile").exists() and "[e1_v4_scale] start " not in runlog,
              f"owned registry labels {[r.get('label') for r in owned]}; supervisor registry exists {(a.root / 'pids' / 'supervisor.tsv').exists()}; lifecycle record {bool(lc)}; supervisor record {bool(sup)}; launch source record {bool(src)}; observer files {sorted(p.name for p in run.glob('*_observations.jsonl'))}; profile directory {(run / 'profile').exists()}; launcher start line {'[e1_v4_scale] start ' in runlog}")
        check("refusal.no_process_left", not [r["pid"] for r in owned if alive(r["pid"])], f"registered processes still alive: {[r['pid'] for r in owned if alive(r['pid'])]}")
        ok = all(r["ok"] for r in results)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "E2E-CHECK.json").write_text(json.dumps({"scenario": a.scenario, "ok": ok, "checks": results, "pid_namespace_note": os.uname().nodename}, indent=1) + "\n")
        print(f"# scenario {a.scenario}: {sum(r['ok'] for r in results)} of {len(results)} checks pass → {'MET' if ok else 'NOT MET'}")
        return 0 if ok else 1
    owned, supreg = registry(a.root / "pids" / "owned.tsv"), registry(a.root / "pids" / "supervisor.tsv")
    labels = sorted(r.get("label", "").split(":")[0] for r in owned)
    check("registry.records", labels == ["driver", "gpuwatch", "importprobe", "memwatch", "storagewatch"] and [r.get("label", "").split(":")[0] for r in supreg] == ["supervisor"], f"owned registry labels {labels}; supervisor registry {[r.get('label') for r in supreg]}")
    left = [r["pid"] for r in owned + supreg if alive(r["pid"])]
    check("registry.all_gone", not left, f"registered processes still alive: {left}")
    server = lc.get("server") or {}
    mock_pids = [p for p in [(lc.get("nsys") or {}).get("pid"), (server.get("api_server") or {}).get("pid")] + [e.get("pid") for e in server.get("engine_processes", [])] if p]
    check("mock_processes.gone", bool(mock_pids) and not [p for p in mock_pids if alive(p)], f"mock Nsight CLI / server pids {mock_pids}; alive {[p for p in mock_pids if alive(p)]}")
    check("launch.rehearsal_mode_recorded", src.get("mode") == "rehearsal", f"launch_source.json mode {src.get('mode')!r} (gate G0 fails on it: never V4 evidence)")
    check("launch.real_controller_record", lc.get("schema_version") is not None and bool(lc.get("host")) and lc.get("policy", {}).get("hard_deadline_s") == 600.0 and lc.get("policy", {}).get("soft_deadline_s") == 300.0 and lc.get("policy", {}).get("server_term_grace_s") == 60.0,
          f"lifecycle.json written by the real controller: schema {lc.get('schema_version')}, policy soft/hard/grace {lc.get('policy', {}).get('soft_deadline_s')}/{lc.get('policy', {}).get('hard_deadline_s')}/{lc.get('policy', {}).get('server_term_grace_s')}")
    guard = fb.get("build_tool_guard") or {}
    if a.scenario in ABORTED:  # the driver was terminated before it wrote its after-launch records; the rehearsal script compares the executable before and after instead
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

    if a.scenario in ("success", "ambiguous", "launch_late_permitted", "complete_near_guard"):
        check("driver.rc", a.driver_rc == 0, f"driver exit code {a.driver_rc}")
        phases = [h["phase"] for h in lc.get("phase_history", [])]
        check("lifecycle.phases", phases == ["stop_requested", "stop_accepted", "capture_ended", "stream_written", "importing", "report_written"] and not lc.get("control_commands") and not lc.get("escalations"), f"phases {phases}; control lines {len(lc.get('control_commands', []))}; escalations {len(lc.get('escalations', []))}")
        check("closeout.clean", a.closeout_rc == 0, f"closeout exit code {a.closeout_rc}")
        check("supervisor.no_response", sup.get("response") is None and sup.get("monitors_effective") is True and sup.get("complete") is True and sup.get("samples", 0) > 10, f"response {sup.get('response')}; monitors effective {sup.get('monitors_effective')}; complete {sup.get('complete')}; samples {sup.get('samples')}")
        final = (lc.get("stream") or {}).get("final") or {}
        pres, dur = (lc.get("stream") or {}).get("preserved_path"), next(iter(sorted((run / "preserved").glob("*.qdstrm"))), None)
        check("stream.checksums", bool(final.get("sha256")) and sha(pres) == final.get("sha256") == sha(dur), f"controller final record {str(final.get('sha256'))[:16]}…, preserved original {str(sha(pres))[:16]}…, durable copy {str(sha(dur))[:16]}…")
        if a.scenario == "success":
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
        check("importer.ambiguity_recorded", "importer" not in probe and "AMBIGUOUS" in states and "ACTIVE" not in states and (probe.get("final") or {}).get("state") == "AMBIGUOUS" and mem.get("ambiguous_samples", 0) > 0 and mem.get("importer_samples") == 0,
              f"probe states {states}, final {(probe.get('final') or {}).get('state')}, no importer singled out even after one of the two exited; memory sampler ambiguous samples {mem.get('ambiguous_samples')}, importer samples {mem.get('importer_samples')}")
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
    if a.scenario in ("reservation_guard", "export_near_guard"):
        c0 = (sup.get("causes") or [{}])[0]
        check("guard.cause", [c.get("cause") for c in sup.get("causes", [])] == ["reservation_guard"] and c0.get("kind") == "run not ended when the reservation guard passed" and (c0.get("reservation_age_s") or 0) > c0.get("limit_s", 1e9) and not (run / "STORAGE-BREACH").exists() and not (run / "MEMORY-BREACH").exists(),
              f"supervisor causes {[(c.get('cause'), c.get('kind'), c.get('reservation_age_s'), c.get('limit_s')) for c in sup.get('causes', [])]}; no breach marker")
    if a.scenario in ABORTED:
        steps = (sup.get("response") or {}).get("steps") or [{}]
        check("breach.marker_and_cause", a.scenario != "storage_breach" or (run / "STORAGE-BREACH").exists() and [c.get("kind") for c in sup.get("causes", [])] == ["persistent_storage_above_ceiling"], f"STORAGE-BREACH present {(run / 'STORAGE-BREACH').exists()}; supervisor causes {[(c.get('source'), c.get('kind'), c.get('value_gb'), c.get('limit_gb')) for c in sup.get('causes', [])]}")
        check("breach.registry_cleanup", [s.get("kind") for s in steps] == ["R2 registry cleanup"] and steps[0].get("rc") in (0, 2) and "survivors 0" in str(steps[0].get("result_line")) and (lc.get("stop_requested_at") is None) == (a.scenario != "export_near_guard") and not (run / "profile" / "nsys" / "lifecycle.control").exists(),
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
        check("driver.did_not_finish", a.driver_rc != 0 and "[e1_v4_scale] start " in runlog and "[e1_v4_scale] rc=" not in runlog, f"driver exit code {a.driver_rc} (terminated by the cleanup); launcher started {'[e1_v4_scale] start ' in runlog}; launcher result line present {'[e1_v4_scale] rc=' in runlog}")
        check("closeout.registry_clean_not_clean_overall", a.closeout_rc == 1, f"closeout exit code {a.closeout_rc} (NOT CLEAN is required after an abort: non-terminal phase, breach marker, supervisor response)")
    causes = [(c.get("cause"), c.get("source"), c.get("kind")) for c in sup.get("causes", [])]
    later = [(c.get("cause"), c.get("source"), c.get("kind")) for c in sup.get("later_causes", [])]
    steps = (sup.get("response") or {}).get("steps") or []
    if a.scenario == "launch_late_permitted":
        latest = REGISTERED_GUARD_S - REGISTERED_RUN_BUDGET_S
        check("launch.permitted_late_under_the_registered_rule", rule.startswith("PASS  launch eligibility") and f"run budget {REGISTERED_RUN_BUDGET_S:.0f} s" in rule and rule_age is not None and latest - 70 <= rule_age <= latest,
              f"run.log: {rule[:230]!r}; age at the check {rule_age} s, latest permitted {latest:.0f} s (registered guard and run budget)")
    if a.scenario == "ceiling_before_stop":
        c0 = (sup.get("causes") or [{}])[0]
        check("ceiling.cause", [c[0] for c in causes] == ["gpu_held_ceiling_before_stop"] and (c0.get("held_s") or 0) > c0.get("limit_s", 1e9) and c0.get("limit_s") == 45.0 and not (run / "STORAGE-BREACH").exists() and not (run / "MEMORY-BREACH").exists(),
              f"supervisor causes {causes}; held {c0.get('held_s')} s, limit {c0.get('limit_s')} s (rehearsal value of the 2700 s ceiling); no breach marker")
    if a.scenario == "export_near_guard":
        final, pres = (lc.get("stream") or {}).get("final") or {}, (lc.get("stream") or {}).get("preserved_path")
        rel, decided = (lc.get("gpu") or {}).get("released_at"), (sup.get("response") or {}).get("decided_at")
        check("export.guard_fell_into_the_cpu_only_phase", lc.get("phase") == "report_written" and rel is not None and decided is not None and rel < decided and "[mock] export and parse (CPU only) continue" in runlog and "[mock] export and parse ended" not in runlog,
              f"lifecycle phase {lc.get('phase')}; GPU release recorded {rel}; guard response decided {decided}; export started {'[mock] export and parse (CPU only) continue' in runlog}, ended {'[mock] export and parse ended' in runlog}")
        check("export.importer_isolated_and_stream_kept", bool(imp_pid) and (probe.get("final") or {}).get("state") == "COMPLETED" and bool(final.get("sha256")) and sha(pres) == final.get("sha256"),
              f"importer {imp_pid}: probe final {(probe.get('final') or {}).get('state')} (finished before the guard; not in the registry, never signalled); preserved stream {str(sha(pres))[:16]}… = controller's final record {str(final.get('sha256'))[:16]}…")
    if a.scenario == "guard_and_storage":
        check("simultaneous.one_response_for_whatever_was_present", len(steps) == 1 and bool(causes) and {c[0] for c in causes} <= {"reservation_guard", "breach"} and {c[0] for c in later} <= {"reservation_guard", "breach"},
              f"causes at the decision {causes}; recorded later {later}; steps {[s.get('kind') for s in steps]}; both present at the decision: {len({c[0] for c in causes}) == 2}")
    if a.scenario == "sampler_lost":
        check("sampler.heartbeat_cause", [(c[0], c[1]) for c in causes] == [("monitor_lost", "storage sampler")] and sup.get("monitors_effective") is False and (sup.get("causes") or [{}])[0].get("limit_s") == 60.0,
              f"supervisor causes {causes}, age {(sup.get('causes') or [{}])[0].get('age_s')} s (limit 60 s); monitors effective {sup.get('monitors_effective')}")
    if a.scenario == "complete_near_guard":
        ended = sup.get("ended_at")
        check("near_guard.ended_before_the_guard_without_a_response", created is not None and ended is not None and REHEARSAL_GUARD_S - 45 <= ended - created <= REHEARSAL_GUARD_S and sup.get("response") is None and sup.get("causes") == [],
              f"supervisor ended through its stop file {None if None in (created, ended) else round(ended - created, 1)} s after the reservation's creation (rehearsal guard {REHEARSAL_GUARD_S:.0f} s); response {sup.get('response')}; causes {sup.get('causes')}")
    if a.scenario == "supervisor_lost":
        phases = [h["phase"] for h in lc.get("phase_history", [])]
        check("supervisor_lost.record_incomplete", sup.get("complete") is not True and sup.get("response") is None and (sup.get("last_check_at") or 0) < (lc.get("stop_requested_at") or 0),
              f"supervisor record complete {sup.get('complete')}, response {sup.get('response')}, last check {sup.get('last_check_at')} before the stop request {lc.get('stop_requested_at')}")
        check("supervisor_lost.run_completed_under_the_controller", a.driver_rc == 0 and phases[-1:] == ["report_written"] and not lc.get("escalations") and not lc.get("control_commands"), f"driver exit code {a.driver_rc}; phases {phases}; escalations {len(lc.get('escalations', []))}; control lines {len(lc.get('control_commands', []))}")
        gates = load(a.gates) or {}
        got = {g["gate"]: g["status"] for g in gates.get("gates", []) if g["status"] != "PASS"}
        check("supervisor_lost.gates_do_not_pass", gates.get("verdict") == "FAIL" and got == EXPECT[a.scenario], f"verdict {gates.get('verdict')}; not PASS {got}; registered {EXPECT[a.scenario]}")
    ok = all(r["ok"] for r in results)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "E2E-CHECK.json").write_text(json.dumps({"scenario": a.scenario, "ok": ok, "checks": results, "pid_namespace_note": os.uname().nodename}, indent=1) + "\n")
    print(f"# scenario {a.scenario}: {sum(r['ok'] for r in results)} of {len(results)} checks pass → {'MET' if ok else 'NOT MET'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
