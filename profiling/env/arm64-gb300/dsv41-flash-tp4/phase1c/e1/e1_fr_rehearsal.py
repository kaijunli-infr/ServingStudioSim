#!/usr/bin/env python3
"""Final CPU-only rehearsal of the V2 forced-release acceptance gates (no GPU, nothing launched; not V2 evidence).

    uv run --no-sync python e1_fr_rehearsal.py synthetic --out DIR [--keep-work]
        Builds synthetic V2 run directories (lifecycle record, probe record, GPU observations, stream, binary records),
        evaluates each through the real command-line validators (checks/e1_forced_release_gates.py and
        checks/e1_frontend_binary_gate.py) and compares verdict and gate statuses with the registered expectation:
        the positive fixtures must pass every gate; every negative fixture must end FAIL or INCONCLUSIVE at the named gate.
    python3 -I e1_fr_rehearsal.py expect-v1 --gates V2-GATES.json --lifecycle lifecycle.json [--out FILE.json]
        Checks the complete V2 post-run on the preserved V1 fixture: the verdict must be FAIL or INCONCLUSIVE, and that
        because V1 performed no forced release during its import (every forced-release gate is not PASS and the record
        shows no release request), while the gates that do not depend on a forced release pass on the real V1 data.

The synthetic fixtures come from the gate tests (tests/test_e1_forced_release_gates.py: same builder, same fake importer
and exporter), so `synthetic` needs the test environment; `expect-v1` is stdlib only. Exit 0 only if every expectation holds.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import contextlib
import importlib.util
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

E1 = Path(__file__).resolve().parent
REPO = E1.parents[5]
FAIL, INC, OK = "FAIL", "INCONCLUSIVE", "PASS"
ALL_GATES = [
    "FR0.record", "H.host_identity", "FR1.one_stop_request", "FR2.registered_trigger", "FR3.import_active_at_release",
    "FR3b.importer_checked_after_signal", "FR4.no_signal_to_importer", "FR5.gpu_release_within_deadline", "FR6.session_ended",
    "FR7.stream_final_stable_durable", "FR8.offline_import", "FR8b.recovered_trace_complete", "FR9.bounds", "S.config_identity",
    "S.binary_unchanged", "S.launcher_recovery", "S.completeness_v2", "S.extract", "S.nsys_validation", "S.closeout",
]  # fmt: skip
#: gates that can only pass if a forced release happened while the import was active
FORCED_RELEASE_GATES = [
    "FR2.registered_trigger", "FR3.import_active_at_release", "FR3b.importer_checked_after_signal", "FR4.no_signal_to_importer",
    "FR5.gpu_release_within_deadline", "FR9.bounds",
]  # fmt: skip
#: gates that do not depend on a forced release: they must pass on the real V1 data
REAL_DATA_GATES = [
    "FR0.record", "FR1.one_stop_request", "FR6.session_ended", "FR7.stream_final_stable_durable", "FR8.offline_import",
    "FR8b.recovered_trace_complete", "S.launcher_recovery", "S.completeness_v2", "S.extract", "S.nsys_validation",
]  # fmt: skip
#: other gates that may be not PASS on a V1 fixture, with the reason (anything else that is not PASS is unexpected)
V1_OTHER = {
    "H.host_identity": "the V1 record is schema 1 (no host field) and V1 wrote no reservation record",
    "S.binary_unchanged": "V1 has neither binary record, and its launcher rebuilt the executable (ADDENDUM-session-runner-rebuild.md)",
    "S.config_identity": "the fixture is a copy of the V1 run under another directory and name",
}
REQUIRED = [
    "positive", "import_completed_before_sigterm", "missing_importer_observations", "late_gpu_release", "binary_mutation",
    "destructive_importer_signal", "incomplete_recovery",
]  # fmt: skip
OTHER_SHA = "95a1526c3a841eebf84d54eafa254cf24590f847fbba233603dc8f0e158aa549"  # the executable V1's launcher rebuilt


def _mod(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Fixture:
    name: str
    category: str
    verdict: str
    expect: dict = field(default_factory=dict)
    why: str = ""
    outcome: str = "B"
    mutate: object = None
    obs: object = None
    drop: tuple = ()
    drop_durable: bool = False
    kernel_rows: int = 1_225
    nvtx: int = 8640
    broken_importer: bool = False
    no_tools: bool = False
    binary: str = "pinned"
    extras: dict = field(default_factory=dict)


def binary_records(pin: str, variant: str) -> tuple[dict | None, dict | None]:
    driver = {"binary": "alignment/load_generator/req-frontend/target/release/session_runner", "sha256_before_launcher": pin, "sha256_after_launcher": pin, "unchanged": True, "cargo_compiling_lines_in_run_log": 0, "stat_before": "1234 1794990000 9000000", "stat_after": "1234 1794990000 9000000"}
    launcher = {"mode": "prebuilt (no compilation in this launch path)", "sha256": pin, "pinned_sha256": pin, "sha256_after_launch": pin, "unchanged": True, "cargo_consulted": False, "build_tool_guard": {"installed": True, "active": True, "blocked": []}}
    if variant == "mutated_during_launch":
        driver.update(sha256_after_launcher=OTHER_SHA, unchanged=False, stat_after="5678 1794991000 9000100")
        launcher.update(sha256_after_launch=OTHER_SHA, unchanged=False)
    elif variant == "not_pinned_before_launch":
        driver.update(sha256_before_launcher=OTHER_SHA, sha256_after_launcher=OTHER_SHA)
        launcher.update(sha256=OTHER_SHA, sha256_after_launch=OTHER_SHA)
    elif variant == "replaced_and_restored":  # same bytes at both ends, but the file was replaced in between
        driver.update(stat_after="5678 1794991000 9000000")
    elif variant == "compiled_during_launch":
        driver.update(cargo_compiling_lines_in_run_log=3)
    elif variant == "build_tool_requested":
        launcher["build_tool_guard"]["blocked"] = [{"at": 1.0, "event": "subprocess.Popen", "command": "cargo build --release"}]
    elif variant == "no_guard":
        launcher.pop("build_tool_guard")
    elif variant == "records_missing":
        return None, None
    elif variant != "pinned":
        raise ValueError(variant)
    return driver, launcher


def fixtures(T) -> list[Fixture]:
    T0, HOST, LINE = T.T0, T.HOST, T.LINE
    held = [(1560, HOST, 260_000, [1]), (1566, HOST, 260_000, [1])]

    def boundary(at):
        def mutate(lc, probe):
            lc["gpu"]["released_at"] = T0 + at - 1
            lc["session_closed_at"] = T0 + at + 2

        return mutate

    def terminal_trigger(lc, probe):
        lc["control_commands"][0]["phase"] = "report_written"
        lc["gpu"]["release_context"] = {"phase": "report_written", "nsys_alive": True, "report_written": True}

    def append_action(action):
        return lambda lc, probe: lc["actions"].append(action)

    F = Fixture
    return [
        F("positive_outcome_A_importer_finishes", "positive", OK, outcome="A", why="import verified active by controller and probe immediately before the SIGTERM, signal time recorded, importer identity intact after it, GPUs released 6 s after the request, the importer then wrote its report; offline recovery equals the in-session export"),
        F("positive_outcome_B_recovery_from_stream", "positive", OK, why="same, but no report: complete semantic recovery from the preserved stream"),
        F("positive_release_at_exactly_90_s", "positive", OK, obs=held + [(1658, HOST, 0, [])], mutate=boundary(1658), why="first released observation exactly 90.0 s after the request: the deadline is inclusive and unchanged"),
        # 4a. the import completed before the SIGTERM
        F("import_report_written_before_sigterm", "import_completed_before_sigterm", INC, {"FR3.import_active_at_release": INC}, outcome="A", mutate=T._completed_first, why="report written at 1567 s, API server signalled at 1568.2 s"),
        F("release_context_already_report_written", "import_completed_before_sigterm", INC, {"FR3.import_active_at_release": INC}, outcome="A", mutate=T._set(["gpu", "release_context"], {"phase": "report_written", "nsys_alive": True, "report_written": True}), why="the controller's own release context says the report existed"),
        F("trigger_read_only_in_terminal_phase", "import_completed_before_sigterm", INC, {"FR2.registered_trigger": INC, "FR3.import_active_at_release": INC}, outcome="A", mutate=terminal_trigger, why="the registered line was accepted only after the report was written"),
        F("no_release_import_finished_first", "import_completed_before_sigterm", INC, dict.fromkeys(FORCED_RELEASE_GATES, INC), outcome="A", mutate=T._never_released, obs=[(1560, HOST, 260_000, [1]), (1590, HOST, 260_000, [1]), (1600, HOST, 0, [])], why="the probe never wrote the trigger; normal shutdown"),
        # 4b. importer observations missing
        F("probe_record_absent", "missing_importer_observations", INC, {"FR3.import_active_at_release": INC, "FR3b.importer_checked_after_signal": INC}, drop=("probe.json",), why="no independent importer observation at all"),
        F("probe_saw_importing_once", "missing_importer_observations", INC, {"FR3.import_active_at_release": INC}, mutate=lambda lc, p: p.__setitem__("samples", p["samples"][:1]), why="one sample cannot show progress"),
        F("import_not_confirmed_in_progress", "missing_importer_observations", INC, {"FR3.import_active_at_release": INC}, mutate=lambda lc, p: p.__setitem__("import_in_progress_confirmed", False)),
        F("no_probe_check_after_signal", "missing_importer_observations", INC, {"FR3b.importer_checked_after_signal": INC}, mutate=lambda lc, p: p.__setitem__("after_signal", [])),
        F("no_controller_check_after_signal", "missing_importer_observations", INC, {"FR3b.importer_checked_after_signal": INC}, mutate=T._set(["gpu", "importer_after_signal"], [])),
        F("probe_trigger_written_after_sigterm", "missing_importer_observations", INC, {"FR3.import_active_at_release": INC}, mutate=lambda lc, p: p.__setitem__("wrote_trigger_at", T0 + 1569.0), why="the probe's confirmation does not precede the signal"),
        F("importer_identity_not_verified", "missing_importer_observations", FAIL, {"FR3.import_active_at_release": FAIL}, mutate=lambda lc, p: p["samples"][0].__setitem__("identity_verified", False)),
        F("probe_watched_another_process", "missing_importer_observations", FAIL, {"FR3.import_active_at_release": FAIL}, mutate=lambda lc, p: [s.__setitem__("nsys_pid", 9) for s in p["samples"]]),
        F("controller_release_context_missing", "missing_importer_observations", FAIL, {"FR3.import_active_at_release": FAIL}, mutate=T._set(["gpu", "release_context"], {})),
        # 4c. late GPU release
        F("gpu_released_90_5_s_after_request", "late_gpu_release", FAIL, {"FR5.gpu_release_within_deadline": FAIL}, obs=held + [(1658.5, HOST, 0, [])], mutate=boundary(1658.5), why="0.5 s over the frozen 90 s deadline"),
        F("gpu_released_after_132_s", "late_gpu_release", FAIL, {"FR5.gpu_release_within_deadline": FAIL}, obs=held[:1] + [(1700, HOST, 0, [])]),
        F("gpu_never_observed_released", "late_gpu_release", FAIL, {"FR5.gpu_release_within_deadline": FAIL}, obs=held),
        F("gpu_release_not_bracketed", "late_gpu_release", FAIL, {"FR5.gpu_release_within_deadline": FAIL}, obs=[(1574, HOST, 0, [])], why="no held observation before the request"),
        F("gpu_held_again_after_release", "late_gpu_release", FAIL, {"FR5.gpu_release_within_deadline": FAIL}, obs=held + [(1574, HOST, 0, []), (1590, HOST, 200_000, [9])]),
        F("gpu_release_only_controller_time", "late_gpu_release", FAIL, {"FR5.gpu_release_within_deadline": FAIL}, obs=[], why="the controller's timestamp alone is not accepted"),
        # 4d. binary mutation
        F("binary_changed_during_launch", "binary_mutation", FAIL, {"S.binary_unchanged": FAIL}, binary="mutated_during_launch"),
        F("binary_not_pinned_before_launch", "binary_mutation", FAIL, {"S.binary_unchanged": FAIL}, binary="not_pinned_before_launch"),
        F("binary_replaced_and_restored", "binary_mutation", FAIL, {"S.binary_unchanged": FAIL}, binary="replaced_and_restored", why="pinned bytes at both ends, but another file identity"),
        F("binary_compiled_during_launch", "binary_mutation", FAIL, {"S.binary_unchanged": FAIL}, binary="compiled_during_launch"),
        F("build_tool_requested_and_blocked", "binary_mutation", FAIL, {"S.binary_unchanged": FAIL}, binary="build_tool_requested"),
        F("launcher_record_without_guard", "binary_mutation", FAIL, {"S.binary_unchanged": FAIL}, binary="no_guard"),
        F("binary_records_missing", "binary_mutation", FAIL, {"S.binary_unchanged": FAIL}, binary="records_missing"),
        # 4e. destructive signal to the importer
        F("sigint_to_importer_before_report", "destructive_importer_signal", FAIL, {"FR4.no_signal_to_importer": FAIL}, mutate=append_action({"at": T0 + 1569, "reason": "hard deadline", "pid": 4000, "action": "SIGINT"})),
        F("sigkill_to_importer_process_group", "destructive_importer_signal", FAIL, {"FR4.no_signal_to_importer": FAIL}, mutate=append_action({"at": T0 + 1570, "reason": f"operator: {LINE}", "pid": 4000, "action": "SIGKILL pgid"})),
        F("sigterm_to_importer_in_outcome_A_before_report", "destructive_importer_signal", FAIL, {"FR4.no_signal_to_importer": FAIL}, outcome="A", mutate=append_action({"at": T0 + 1580, "reason": "session close", "pid": 4000, "action": "SIGTERM"}), why="report written at 1592 s; the signal came 12 s earlier"),
        F("signal_to_unrecorded_process", "destructive_importer_signal", FAIL, {"FR4.no_signal_to_importer": FAIL}, mutate=append_action({"at": T0 + 1569, "reason": "x", "pid": 777, "action": "SIGKILL"})),
        F("sigterm_time_not_recorded", "destructive_importer_signal", FAIL, {"FR4.no_signal_to_importer": FAIL}, mutate=lambda lc, p: lc["gpu"].pop("api_server_sigterm"), why="without the exact signal time nothing can be ordered against the importer"),
        # 4f. incomplete recovery
        F("durable_stream_missing", "incomplete_recovery", FAIL, {"FR7.stream_final_stable_durable": FAIL, "FR8.offline_import": FAIL, "FR8b.recovered_trace_complete": FAIL}, drop_durable=True),
        F("stream_differs_from_final_record", "incomplete_recovery", FAIL, {"FR7.stream_final_stable_durable": FAIL}, mutate=T._set(["stream", "final", "sha256"], "0" * 64)),
        F("no_final_stream_record", "incomplete_recovery", FAIL, {"FR7.stream_final_stable_durable": FAIL}, mutate=T._set(["stream", "final"], None)),
        F("offline_import_fails_truncated_stream", "incomplete_recovery", FAIL, {"FR7.stream_final_stable_durable": OK, "FR8.offline_import": FAIL, "FR8b.recovered_trace_complete": FAIL}, broken_importer=True, why="a stable, checksummed stream that does not import"),
        F("recovered_trace_too_few_kernels", "incomplete_recovery", FAIL, {"FR8.offline_import": OK, "FR8b.recovered_trace_complete": FAIL}, kernel_rows=900),
        F("recovered_trace_nvtx_not_4_per_iteration", "incomplete_recovery", FAIL, {"FR8b.recovered_trace_complete": FAIL}, nvtx=8638, why="2160 iterations but 8638 forward ranges"),
        F("recovered_trace_differs_from_in_session_export", "incomplete_recovery", FAIL, {"FR8b.recovered_trace_complete": FAIL}, outcome="A", kernel_rows=1_224, why="one kernel row fewer than the in-session export, still inside the band"),
        F("recovery_tools_not_supplied", "incomplete_recovery", INC, {"FR8.offline_import": INC, "FR8b.recovered_trace_complete": INC}, no_tools=True, why="the recovery proof is mandatory"),
        F("launcher_recovery_failed", "incomplete_recovery", FAIL, {"S.launcher_recovery": FAIL}, extras={"launcher_recovery": 1}),
        F("scientific_completeness_failed", "incomplete_recovery", FAIL, {"S.completeness_v2": FAIL}, extras={"completeness_v2": 1}),
        # further stop conditions
        F("importer_still_running", "other", FAIL, {"FR6.session_ended": FAIL}, mutate=lambda lc, p: (lc["nsys"].__setitem__("alive", True), lc.__setitem__("session_closed_at", None), lc.__setitem__("phase", "importing"))),
        F("trigger_in_wrong_phase", "other", FAIL, {"FR2.registered_trigger": FAIL}, mutate=T._set(["control_commands", 0, "phase"], "stop_accepted")),
        F("second_stop_request", "other", FAIL, {"FR1.one_stop_request": FAIL}, mutate=T._set(["http", "requests_sent"], 2)),
        F("host_conflict", "other", FAIL, {"H.host_identity": FAIL}, mutate=T._set(["host"], "pod4-gb300-10-tray14-f3")),
        F("escalation_recorded", "other", FAIL, {"FR9.bounds": FAIL}, mutate=lambda lc, p: lc["escalations"].append({"kind": "hard_deadline", "phase": "importing"})),
        F("configuration_identity_failed", "other", FAIL, {"S.config_identity": FAIL}, extras={"identity": "FAIL"}),
    ]  # fmt: skip


def run_fixture(fx: Fixture, T, gates, bgate, pin: str, work: Path, out: Path, index: int) -> dict:
    tmp = work / fx.name
    tmp.mkdir()
    run, durable = T._run(tmp, fx.outcome, mutate=fx.mutate, obs=fx.obs)
    for rel in fx.drop:
        (run / rel).unlink()
    if fx.drop_durable:
        durable.unlink()
    (tmp / "tools").mkdir()
    imp, nsys = T._fake_tools(tmp / "tools", kernel_rows=fx.kernel_rows, nvtx=fx.nvtx)
    if fx.broken_importer:
        imp.write_text("#!/bin/sh\necho 'truncated stream: unexpected end of file' >&2\nexit 1\n")
    driver, launcher = binary_records(pin, fx.binary)
    prov = tmp / "BINARY-PROVENANCE.json"
    prov.write_text(json.dumps({"sha256": pin}))
    if driver is not None:
        (run / "binary_check.json").write_text(json.dumps(driver))
        (run / "profile" / "load_generator").mkdir(parents=True)
        (run / "profile" / "load_generator" / "frontend_binary.json").write_text(json.dumps(launcher))
    stem = f"{index:02d}_{fx.name}"
    with (out / f"{stem}.binary_gate.txt").open("w") as fh, contextlib.redirect_stdout(fh):
        brc = bgate.main(["--driver-record", str(run / "binary_check.json"), "--launcher-record", str(run / "profile" / "load_generator" / "frontend_binary.json"), "--provenance", str(prov)])
    extras = {"binary_unchanged": brc, "launcher_recovery": 0, "completeness_v2": 0, "extract": 0, "nsys_validation": 0, "closeout": 0, **{k: v for k, v in fx.extras.items() if k != "identity"}}
    ident = tmp / "identity.json"
    ident.write_text(json.dumps({"verdict": fx.extras.get("identity", OK), "normalized": []}))
    inputs = tmp / "inputs.json"
    inputs.write_text(json.dumps(T.INPUTS))
    gates_json = out / f"{stem}.gates.json"
    argv = ["--run", str(run), "--inputs", str(inputs), "--scratch", str(tmp / "scratch"), "--durable-stream", str(durable), "--identity", str(ident), "--out", str(gates_json)]
    if not fx.no_tools:
        argv += ["--importer", str(imp), "--nsys", str(nsys)]
    for name, rc in extras.items():
        argv += ["--extra", f"{name}={rc}"]
    with (out / f"{stem}.gates.txt").open("w") as fh, contextlib.redirect_stdout(fh):
        rc = gates.main(argv)
    report = json.loads(gates_json.read_text())
    st = {g["gate"]: g["status"] for g in report["gates"]}
    if fx.category != "positive":
        gates_json.unlink()  # the text record is kept for every fixture; the JSON only for the positive ones
    mismatches = [f"{g}: expected {want}, got {st.get(g)}" for g, want in fx.expect.items() if st.get(g) != want]
    if report["verdict"] != fx.verdict:
        mismatches.append(f"verdict: expected {fx.verdict}, got {report['verdict']}")
    if {OK: 0, FAIL: 1}.get(report["verdict"], 3) != rc:
        mismatches.append(f"exit code {rc} does not match the verdict {report['verdict']}")
    if fx.category == "positive":
        mismatches += [f"{g}: {st.get(g)}" for g in ALL_GATES if st.get(g) != OK]
        mismatches += [f"unregistered gate {g}" for g in st if g not in ALL_GATES]
        after = next(g for g in report["gates"] if g["gate"] == "FR3b.importer_checked_after_signal")["data"]
        if not all(s.get("identity_matches") is True for s in after["controller"]) or not all(s.get("identity_verified") is True for s in after["probe"]):
            mismatches.append("importer identity not intact after the signal")
    elif fx.verdict == OK:
        mismatches.append("a negative fixture may not expect PASS")
    return {"index": index, "name": fx.name, "category": fx.category, "why": fx.why, "expected_verdict": fx.verdict, "verdict": report["verdict"], "exit_code": rc, "expected_gates": fx.expect, "not_pass": {g: s for g, s in st.items() if s != OK}, "gates_evaluated": len(st), "binary_gate_rc": brc, "ok": not mismatches, "mismatches": mismatches}


def synthetic(out: Path, keep_work: bool = False) -> int:
    sys.path[:0] = [str(REPO / "tests"), str(REPO)]
    import test_e1_forced_release_gates as T

    gates = _mod(E1 / "checks" / "e1_forced_release_gates.py")
    bgate = _mod(E1 / "checks" / "e1_frontend_binary_gate.py")
    pin = json.loads((E1 / "BINARY-PROVENANCE.json").read_text())["sha256"]
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="e1-fr-rehearsal-"))
    try:
        rows = [run_fixture(fx, T, gates, bgate, pin, work, out, i) for i, fx in enumerate(fixtures(T), 1)]
    finally:
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)
    covered = {c: [r["name"] for r in rows if r["category"] == c] for c in REQUIRED}
    missing = [c for c, names in covered.items() if not names]
    ok = all(r["ok"] for r in rows) and not missing
    lines = [f"# synthetic V2 gate rehearsal at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: {len(rows)} fixtures through the command-line validators; CPU only; not V2 evidence", f"# {'fixture':52s} {'category':32s} {'expected':13s} {'verdict':13s} gates not PASS"]
    for r in rows:
        lines.append(f"{'OK      ' if r['ok'] else 'MISMATCH'} {r['index']:02d} {r['name']:49s} {r['category']:32s} {r['expected_verdict']:13s} {r['verdict']:13s} {', '.join(f'{g}={s}' for g, s in r['not_pass'].items()) or '-'}" + (f"   !! {r['mismatches']}" if r["mismatches"] else ""))
    for c in REQUIRED:
        lines.append(f"# required category {c}: {len(covered[c])} fixture(s), all as expected: {all(r['ok'] for r in rows if r['category'] == c) and bool(covered[c])}")
    by = {v: sum(r["verdict"] == v for r in rows) for v in (OK, INC, FAIL)}
    lines.append(f"# result: {'REHEARSAL EXPECTATIONS MET' if ok else 'REHEARSAL EXPECTATIONS NOT MET'}: {sum(r['ok'] for r in rows)}/{len(rows)} fixtures as expected; verdicts PASS {by[OK]}, INCONCLUSIVE {by[INC]}, FAIL {by[FAIL]}; categories without a fixture: {missing or 'none'}")
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    (out / "SUMMARY.json").write_text(json.dumps({"ok": ok, "fixtures": rows, "required_categories": covered, "pinned_sha256": pin}, indent=1) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


def expect_v1(report: dict, lc: dict) -> tuple[bool, list[str]]:
    st = {g["gate"]: g["status"] for g in report.get("gates", [])}
    gpu = lc.get("gpu") or {}
    lines, ok = [], True

    def check(name: str, good: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and bool(good)
        lines.append(f"{'PASS' if good else 'FAIL'}  {name}: {detail}")

    check("verdict_is_not_pass", report.get("verdict") in (FAIL, INC), f"V2 verdict on the V1 fixture: {report.get('verdict')}")
    imported = any(h.get("phase") == "importing" for h in lc.get("phase_history", []))
    releases = [c.get("command") for c in lc.get("control_commands", []) if str(c.get("command", "")).startswith("release-gpus")]
    release_signals = [a for a in lc.get("actions", []) if str(a.get("reason", "")).startswith("operator: release-gpus")]
    check("v1_had_an_import_but_no_forced_release", imported and gpu.get("release_requested_at") is None and not gpu.get("api_server_sigterm") and not releases and not release_signals, f"import phase recorded={imported}; release requested at {gpu.get('release_requested_at')}; API-server SIGTERM record {gpu.get('api_server_sigterm')}; release-gpus control lines {releases}; release signals {len(release_signals)}")
    for g in FORCED_RELEASE_GATES:
        check(f"forced_release_gate_not_pass[{g}]", st.get(g) in (FAIL, INC), f"{st.get(g)}")
    forced_only = FAIL if any(st.get(g) == FAIL for g in FORCED_RELEASE_GATES) else (INC if any(st.get(g) == INC for g in FORCED_RELEASE_GATES) else OK)
    check("verdict_from_forced_release_gates_alone", forced_only != OK, f"{forced_only}: with every other gate set aside the validation still stops, because no release was forced during the import")
    for g in REAL_DATA_GATES:
        check(f"real_data_gate_pass[{g}]", st.get(g) == OK, f"{st.get(g)}")
    for g, s in st.items():
        if s == OK or g in FORCED_RELEASE_GATES or g in REAL_DATA_GATES:
            continue
        check(f"other_gate_explained[{g}]", g in V1_OTHER, f"{s}: {V1_OTHER.get(g, 'UNEXPECTED: no registered explanation for a V1 fixture')}")
    lines.append(f"# V1 fixture expectation: {'MET' if ok else 'NOT MET'} (verdict {report.get('verdict')}; {sum(s != OK for s in st.values())} of {len(st)} gates not PASS)")
    return ok, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    syn = sub.add_parser("synthetic")
    syn.add_argument("--out", required=True, type=Path)
    syn.add_argument("--keep-work", action="store_true")
    v1 = sub.add_parser("expect-v1")
    v1.add_argument("--gates", required=True, type=Path)
    v1.add_argument("--lifecycle", required=True, type=Path)
    v1.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.command == "synthetic":
        return synthetic(args.out, args.keep_work)
    ok, lines = expect_v1(json.loads(args.gates.read_text()), json.loads(args.lifecycle.read_text()))
    print("\n".join(lines))
    if args.out:
        args.out.write_text(json.dumps({"ok": ok, "lines": lines}, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
