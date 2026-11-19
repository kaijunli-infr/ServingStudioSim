#!/usr/bin/env python3
"""The V4 identity policy exercised against the ACTUAL generated V4 configuration, before any freeze (CPU-only; stdlib).

    python3 -I e1_v4scale_identity_rehearsal.py --reference <accepted e1_nsys_val run dir> --out DIR [--write-fixture]

V3 failed because its post-run identity step had only ever been run against a V2 configuration. This runs the V4 step on
the files a V4 launch would leave: the profile config is the generated configs_v4scale/e1_v4_scale/profile.json byte for
byte (the driver copies it into the run directory), the trace is traces/e1_v4_scale.csv at the path that config names,
and the launch record is the predicted one (tests/fixtures/e1_v4_scale/launch.predicted.json).

A real V4 launch record does not exist before a launch. The prediction is the V3 launch record with the pass name and the
run-root name substituted; the same substitution applied to the V2 launch record is checked here to reproduce the actual V3
launch record in every field, which is what makes it a prediction and not a guess. Gate G13 judges the real record.

Expectations (registered here, before the run):
  method     V2 launch record with V2 → V3 names = the actual V3 launch record, every field
  fixture    the committed predicted V4 launch record = the V3 record with V3 → V4 names
  v4_policy  V4 checker + V4 policy on (actual V4 config, predicted launch record): PASS; trace rule exercised; every trace
             check holds; the field counts are those of the accepted comparisons (182 launch, 74 profile)
  v1_policy  V1/V2 checker + policy on the same records: FAIL, on workload.frontend.path only (the V3 defect, detected)
  v3_policy  V3 checker + policy on the same records: FAIL (it applies to e1_v3_scale only)
  negatives  under the V4 policy each of these is FAIL: the V3 config and launch record; the V4 trace with two rows
             swapped; with one row removed; the V4 config with streaming switched on; with the hard deadline changed;
             the predicted launch record with a server argument changed; a config naming a copy of the trace elsewhere
Exit 0 only if every expectation holds.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

E1 = Path(__file__).resolve().parent
REPO = E1.parents[5]
FIXTURE = REPO / "tests" / "fixtures" / "e1_v4_scale" / "launch.predicted.json"
V2_LAUNCH = E1 / "runs/v2/run/evidence/dsv41_flash_1017_gb300_tp4_e1_v2_forced_release_launch.json"
V3_LAUNCH = E1 / "runs/v3scale/run/evidence/dsv41_flash_1017_gb300_tp4_e1_v3_scale_launch.json"
CONFIG = E1 / "configs_v4scale/e1_v4_scale/profile.json"
TRACE = E1 / "traces/e1_v4_scale.csv"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, E1 / "checks" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def substitute(text: str, pairs) -> str:
    for a, b in pairs:
        text = text.replace(a, b)
    return text


def predicted_v4_launch() -> str:
    return substitute(V3_LAUNCH.read_text(), (("e1_v3_scale", "e1_v4_scale"), ("e1_v3scale", "e1_v4scale")))


def run_dir(tmp: Path, config: bytes, launch: str) -> Path:
    b = tmp
    (b / "profile" / "vllm").mkdir(parents=True)
    (b / "profile.json").write_bytes(config)
    (b / "profile" / "vllm" / "dsv41_flash_1017_gb300_tp4_e1_v4_scale_launch.json").write_text(launch)
    return b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--write-fixture", action="store_true", help="(re)write the predicted launch record fixture, then check as usual")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    v4, v3, v2 = _load("e1_launch_identity_v4scale"), _load("e1_launch_identity_v3scale"), _load("e1_launch_identity_v2")
    policy4 = v2.load_policy(E1 / "checks/e1_identity_policy_v4scale.json")
    rows, ok_all = [], True

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal ok_all
        ok_all &= bool(ok)
        rows.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    pred_v3 = json.loads(substitute(V2_LAUNCH.read_text(), (("e1_v2_forced_release", "e1_v3_scale"), ("e1_v2fr", "e1_v3scale"))))
    act_v3 = json.loads(V3_LAUNCH.read_text())
    check("method.v2_record_with_v3_names_is_the_actual_v3_record", pred_v3 == act_v3, f"V2 launch record with the pass and run-root names substituted equals the actual V3 launch record in every field: {pred_v3 == act_v3} ({len(v2.flatten_launch(act_v3)) if hasattr(v2, 'flatten_launch') else '?'} compared fields)")
    launch = predicted_v4_launch()
    if a.write_fixture:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(launch)
    check("fixture.predicted_launch_record", FIXTURE.is_file() and FIXTURE.read_text() == launch and "e1_v3" not in launch and json.loads(launch)["name"] == "dsv41_flash_1017_gb300_tp4_e1_v4_scale", f"{FIXTURE.relative_to(REPO)} = the V3 launch record with V3 → V4 names; name {json.loads(launch)['name']}")
    config = CONFIG.read_bytes()
    cfg = json.loads(config)
    check("config.is_the_generated_v4_config", cfg["workload"]["frontend"]["path"] == str(TRACE) and cfg["name"] == "dsv41_flash_1017_gb300_tp4_e1_v4_scale" and TRACE.is_file(), f"profile config {CONFIG.relative_to(E1)} names the trace {cfg['workload']['frontend']['path']} (exists: {TRACE.is_file()})")

    def variant(name: str, expect: str, *, cfg_bytes: bytes = config, launch_text: str = launch, checker=v4, policy=policy4, reader=None, only_field=None, need=None) -> dict:
        with tempfile.TemporaryDirectory(prefix="v4scale-identity-") as tmp:
            b = run_dir(Path(tmp), cfg_bytes, launch_text)
            rep = checker.run(a.reference, b, json.loads(json.dumps(policy)), **({"reader": reader} if reader else {}))
        fails = [(r.get("scope"), r.get("field")) for r in rep["results"] if r["status"] == "FAIL"]
        ok = rep["verdict"] == expect and (only_field is None or fails == [only_field]) and (need is None or need(rep))
        check(name, ok, f"verdict {rep['verdict']} (expected {expect}); failed fields {fails[:5]}; normalized {len(rep.get('normalized', []))}; trace rule exercised {(rep.get('trace_identity') or {}).get('exercised')}")
        (a.out / f"{name}.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
        return rep

    def full_proof(rep: dict) -> bool:
        ti, info = rep["trace_identity"], {r["scope"]: r["reason"] for r in rep["results"] if r["status"] == "INFO"}
        return ti["exercised"] is True and len(ti["checks"]) == ti["expected_checks"] == 20 and all(c["ok"] for c in ti["checks"]) and info.get("launch", "").startswith("182 fields compared") and info.get("profile", "").startswith("74 fields compared")

    rep = variant("v4_policy.actual_config_predicted_launch", "PASS", need=full_proof)
    for c in rep["trace_identity"].get("checks", []):
        print(f"      TRACE {'PASS' if c['ok'] else 'FAIL'}  {c['check']}")
    variant("v1_policy.same_records", "FAIL", checker=v2, policy=v2.load_policy(E1 / "checks/e1_identity_policy_v1.json"), only_field=("profile", "workload.frontend.path"))
    variant("v3_policy.same_records", "FAIL", checker=v3, policy=v2.load_policy(E1 / "checks/e1_identity_policy_v3scale.json"))
    trace_rows = TRACE.read_text().split("\n")

    def reader_with(data: bytes):
        return lambda path: data if Path(path).name == TRACE.name else v4.read_file(path)

    swapped = trace_rows[:]
    swapped[130], swapped[131] = swapped[131], swapped[130]
    variant("negative.v3_config_and_launch_record", "FAIL", cfg_bytes=(E1 / "configs_v3scale/e1_v3_scale/profile.json").read_bytes(), launch_text=V3_LAUNCH.read_text())
    variant("negative.trace_two_rows_swapped", "FAIL", reader=reader_with("\n".join(swapped).encode()), only_field=("profile", "workload.frontend.path"))
    variant("negative.trace_one_row_removed", "FAIL", reader=reader_with("\n".join(trace_rows[:200] + trace_rows[201:]).encode()), only_field=("profile", "workload.frontend.path"))
    variant("negative.config_streaming_on", "FAIL", cfg_bytes=config.replace(b'"stream": false', b'"stream": true'))
    variant("negative.config_hard_deadline_changed", "FAIL", cfg_bytes=config.replace(b'"finalize_hard_deadline_s": 600.0', b'"finalize_hard_deadline_s": 900.0'))
    variant("negative.launch_server_argument_changed", "FAIL", launch_text=launch.replace('"--max-num-seqs"', '"--max-num-seqs-x"', 1) if '"--max-num-seqs"' in launch else launch.replace('"server_argv": [', '"server_argv": ["--unexpected", ', 1))
    elsewhere = json.loads(config)
    elsewhere["workload"]["frontend"]["path"] = "/tmp/e1_v4_scale.csv"
    variant("negative.trace_path_not_the_documented_location", "FAIL", cfg_bytes=json.dumps(elsewhere).encode(), reader=reader_with(TRACE.read_bytes()), only_field=("profile", "workload.frontend.path"))
    (a.out / "IDENTITY-REHEARSAL.json").write_text(json.dumps({"ok": ok_all, "checks": rows}, indent=1) + "\n")
    print(f"# V4 identity rehearsal on the actual V4 configuration: {sum(r['ok'] for r in rows)} of {len(rows)} expectations hold → {'MET' if ok_all else 'NOT MET'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
