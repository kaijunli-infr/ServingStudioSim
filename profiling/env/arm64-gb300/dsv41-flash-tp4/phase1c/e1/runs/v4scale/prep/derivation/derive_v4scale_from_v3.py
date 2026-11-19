#!/usr/bin/env python3
"""Derive the V4 scripts from the frozen V3 ones: a fixed renaming plus listed, asserted edits (each edit must match exactly
once, or the stated number of times). The V3 files are read only. Output: src/e1/*, src/tests/*; and, for review, one unified
diff per file against the V3 original after the renaming alone (so the diff shows only the listed edits)."""
import difflib, re, sys
from pathlib import Path

REF = Path("ref/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1")
REFT = Path("ref/tests")
OUT, OUTT, DIFF = Path("src/e1"), Path("src/tests"), Path("src/derivation")
DIFF.mkdir(parents=True, exist_ok=True)

def rename(s: str) -> str:
    for a, b in (("v3scale", "v4scale"), ("v3_scale", "v4_scale"), ("V3S", "V4S"), ("V3-SCALE", "V4-SCALE"), ("e1v3-scale", "e1v4-scale")):
        s = s.replace(a, b)
    return re.sub(r"\bV3\b", "V4", s)

def derive(src: Path, dst: Path, edits: list, keep=()):
    orig = src.read_text()
    base = rename(orig)
    for a, b in keep:  # places where the V3 name must stay
        assert a in base, (src.name, a[:60]); base = base.replace(a, b)
    s = base
    for e in edits:
        a, b, n = (*e, 1)[:3]
        assert s.count(a) == n, (src.name, a[:80], s.count(a))
        s = s.replace(a, b)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(s)
    d = "".join(difflib.unified_diff(base.splitlines(True), s.splitlines(True), f"{src.name} (V3, names changed to V4 only)", dst.name, n=0))
    (DIFF / (dst.name + ".diff")).write_text(d or "(no difference beyond the renaming)\n")
    print(f"{dst.name}: {len(edits)} edit(s), {sum(1 for l in d.splitlines() if l.startswith('+') and not l.startswith('+++'))} changed line(s)")

# ---- driver
derive(REF / "run_e1_v3scale.sh", OUT / "run_e1_v4scale.sh", [
    ("RSS_CEIL=128; TMP_CEIL=10; PERS_CEIL=8\n", "RSS_CEIL=128; TMP_CEIL=15; PERS_CEIL=18; RES_AGE=600; RES_GUARD=4680\n"),
    ("PERS_CEIL=${E1_V4S_REHEARSAL_PERSISTENT_GB:-$PERS_CEIL}\n", "PERS_CEIL=${E1_V4S_REHEARSAL_PERSISTENT_GB:-$PERS_CEIL}; RES_GUARD=${E1_V4S_REHEARSAL_RESERVATION_GUARD_S:-$RES_GUARD}\n"),
    ("ceilings importer $RSS_CEIL GiB / temporary $TMP_CEIL GB / persistent $PERS_CEIL GB\"", "ceilings importer $RSS_CEIL GiB / temporary $TMP_CEIL GB / persistent $PERS_CEIL GB; reservation guard $RES_GUARD s\""),
    ("frozen 112-row trace", "registered 224-row trace"),
    ('grep -c .)" = 112 ]', 'grep -c .)" = 224 ]'),
    ("# launch source record (gate G0)", "# reservation budget, first part: no launch later than $RES_AGE s after the reservation's creation (the end of the run would not fit the 90 min ceiling)\n"
     "python3 -I $E1/checks/e1_reservation_guard.py age --reservation $RUN/$PASS/reservation.json --max-age-s $RES_AGE | tee -a $RUN/$PASS/run.log\n"
     "[ \"${PIPESTATUS[0]}\" = 0 ] || { echo \"reservation too old (or its age unknown) for a launch; not starting\"; exit 7; }\n# launch source record (gate G0)"),
    ("--interval 1 --max-s 3400 --stop-file $STOPF; MPID=$(lastpid)", "--interval 1 --max-s 5400 --stop-file $STOPF; MPID=$(lastpid)"),
    ("--interval 2 --max-s 3400 --stop-file $STOPF; SPID=$(lastpid)", "--interval 2 --max-s 5400 --stop-file $STOPF; SPID=$(lastpid)"),
    ("# MEMORY-BREACH / STORAGE-BREACH and the samplers' heartbeats; one response; sends no signal itself", "# MEMORY-BREACH / STORAGE-BREACH and the samplers' heartbeats, and enforces the GPU-held ceiling before the stop request and the\n# reservation guard ($RES_GUARD s after the reservation's creation); one response; sends no signal itself"),
    ("python3 -I $E1/checks/e1_supervisor.py --run $RUN/$PASS --registry $REG --out $RUN/$PASS/supervisor.json", "python3 -I $E1/checks/e1_supervisor_v4.py --run $RUN/$PASS --registry $REG --out $RUN/$PASS/supervisor.json --reservation $RUN/$PASS/reservation.json --reservation-guard-s $RES_GUARD"),
    ("--cleanup-grace 30 --interval 1 --max-s 3400 --stop-file $STOPF | tee", "--cleanup-grace 30 --interval 1 --max-s 5400 --stop-file $STOPF | tee"),
    ("# stop the memory and storage samplers through their stop file", "# (the launcher's export and parse after the GPU release take about 20 min at this size: the memory and storage samplers and the\n# supervisor run for up to 5400 s, the reservation guard ends a run that is still going 4680 s after the reservation's creation)\n# stop the memory and storage samplers through their stop file"),
])
# ---- post-run
derive(REF / "postrun_e1_v3scale.sh", OUT / "postrun_e1_v4scale.sh", [
    ("--ceiling-gb temporary=10 --ceiling-gb persistent=8 --interval 2 --max-s 7200", "--ceiling-gb temporary=15 --ceiling-gb persistent=18 --interval 2 --max-s 7200"),
    ("# the normalization policy. (C) scientific checkers on the in-session profile: four-wave completeness", "# the V4-specific normalization policy (four-copy trace proof). (C) scientific checkers on the in-session profile: eight-wave completeness"),
    ('echo "## B. configuration identity under the normalization policy (launch record + profile config vs the accepted e1_nsys_val run)"', 'echo "## B. configuration identity under the V4 policy: four-copy trace proof; every other field as for V1/V2 (launch record + profile config vs the accepted e1_nsys_val run)"'),
    ("python3 -I $E1/checks/e1_launch_identity_v2.py --a $AR15 --b $RUN/$PASS --policy $E1/checks/e1_identity_policy_v1.json --out $CPU/identity_v2_$PASS.json > $CPU/identity_v2_$PASS.txt 2>&1; echo \"identity rc=$? ($(grep -c '^NORMALIZED' $CPU/identity_v2_$PASS.txt) normalized fields listed in cpu/identity_v2_$PASS.txt)\"",
     "python3 -I $E1/checks/e1_launch_identity_v4scale.py --a $AR15 --b $RUN/$PASS --policy $E1/checks/e1_identity_policy_v4scale.json --out $CPU/identity_v4scale_$PASS.json > $CPU/identity_v4scale_$PASS.txt 2>&1; echo \"identity rc=$? ($(grep -c '^NORMALIZED' $CPU/identity_v4scale_$PASS.txt) normalized fields, $(grep -c '^TRACE PASS' $CPU/identity_v4scale_$PASS.txt) of $(grep -c '^TRACE ' $CPU/identity_v4scale_$PASS.txt) trace checks, listed in cpu/identity_v4scale_$PASS.txt)\""),
    ('echo "completeness_v2 (four-wave plan, routing records, checksum) rc=$v2"', 'echo "completeness_v2 (eight-wave plan, routing records, checksum) rc=$v2"'),
    ("--identity $CPU/identity_v2_$PASS.json --postrun-storage", "--identity $CPU/identity_v4scale_$PASS.json --completeness $CPU/v2_$PASS.json --postrun-storage"),
    ("head -12 | cut -c1-200; echo \"completeness_v2", "head -16 | cut -c1-200; echo \"completeness_v2"),
])
# ---- monitor, closeout, archive
derive(REF / "monitor_e1_v3scale.sh", OUT / "monitor_e1_v4scale.sh", [
    ("[startup_limit_min=30] [capture_limit_min=10] [run_ceiling_GB=8] [tmp_floor_GB=50]", "[startup_limit_min=30] [capture_limit_min=10] [run_ceiling_GB=18] [tmp_floor_GB=50]"),
    ("CEIL=${5:-8}", "CEIL=${5:-18}"),
    ('s=load(run+"/storage_watch.json")\n', 'r=load(run+"/reservation.json") or {}\ntry:\n    import calendar\n    res_age = time.time() - calendar.timegm(time.strptime(r["creation_timestamp"], "%Y-%m-%dT%H:%M:%SZ"))\n    print(f"reservation: age {res_age / 60:.1f} min (supervisor guard 78 min, ceiling 90 min)")\n    if res_age > 4680 and not (u or {}).get("complete"): print("ABORT-CONDITION the reservation is older than 78 min and the run has not ended (the supervisor\'s reservation guard should have responded)")\nexcept (KeyError, ValueError, TypeError): print("reservation: age unknown (record missing or unreadable)")\ns=load(run+"/storage_watch.json")\n'),
])
derive(REF / "closeout_e1_v3scale.sh", OUT / "closeout_e1_v4scale.sh", [
    ('echo "INFO  reservation pod (from the laptop): delete only with the GPUs idle, then write', 'echo "INFO  reservation pod (from the laptop): delete immediately after a CLEAN closeout and only with the GPUs idle (budget: 12 min from the supervisor guard at 78 min to the 90 min ceiling), then write'),
])
derive(REF / "archive_e1_v3scale.sh", OUT / "archive_e1_v4scale.sh", [
    ("measures run root + archive against the 16 GB ceiling", "measures run root + archive against the 38 GB ceiling (checks/e1_archive_size.py)"),
    ('python3 -I - "$RUN" "$A" <<\'PY\' && ok "persistent storage including the archive within 16 GB" || bad "persistent storage including the archive exceeds 16 GB"\n', 'python3 -I $(dirname "${BASH_SOURCE[0]}")/checks/e1_archive_size.py --run "$RUN" --archive "$A" --ceiling-gb 38 && ok "persistent storage including the archive within 38 GB" || bad "persistent storage including the archive exceeds 38 GB (or could not be measured)"\n: <<\'PY\'\n'),
])
_a = OUT / "archive_e1_v4scale.sh"; _t = _a.read_text(); _n = re.sub(r": <<'PY'\n.*?\nPY\n", "", _t, flags=re.S); assert _n != _t and "import os, sys" not in _n; _a.write_text(_n)  # the inline size check is replaced by the checker
# ---- preflight
TRACE_BLOCK_OLD = '''python3 -I - "$TRACE" "$E1/traces/e1_dry.csv" <<'PY' && ok "trace = the 56 accepted rows twice: 112 requests, 4 waves (b=8 at 0 s, b=48 at 18.627 s, b=8 at 80 s, b=48 at 98.627 s), second pair = first pair with wave index 1 and +80 s" || bad "trace is not the accepted 56 rows twice"
import sys
v3 = [r.split(",") for r in open(sys.argv[1]).read().split()[1:]]; dry = [r.split(",") for r in open(sys.argv[2]).read().split()[1:]]
want = dry + [[r[0].replace("_w0_r", "_w1_r"), r[1], r[2], str(round(float(r[3]) + 80000.0, 1))] for r in dry]
waves = sorted({(r[0].rsplit("_r", 1)[0], float(r[3])) for r in v3}, key=lambda x: x[1])
sys.exit(0 if v3 == want and len(v3) == 112 and waves == [("e1_b8_kv32k_w0", 0.0), ("e1_b48_kv32k_w0", 18627.2), ("e1_b8_kv32k_w1", 80000.0), ("e1_b48_kv32k_w1", 98627.2)] else 1)
PY
'''
TRACE_BLOCK_NEW = '''python3 -I - "$TRACE" "$E1/traces/e1_dry.csv" <<'PY' && ok "trace = the 56 accepted rows four times: 224 requests, 8 waves (b=8 at 0 / 80 / 160 / 240 s, b=48 at 18.627 s after each), copy i = the accepted rows with wave index i and + i x 80 s" || bad "trace is not the accepted 56 rows four times"
import sys
v4 = [r.split(",") for r in open(sys.argv[1]).read().split()[1:]]; dry = [r.split(",") for r in open(sys.argv[2]).read().split()[1:]]
want = [[r[0].replace("_w0_r", f"_w{i}_r"), r[1], r[2], str(round(float(r[3]) + i * 80000.0, 1))] for i in range(4) for r in dry]
waves = sorted({(r[0].rsplit("_r", 1)[0], float(r[3])) for r in v4}, key=lambda x: x[1])
sys.exit(0 if v4 == want and len(v4) == 224 and len({r[0] for r in v4}) == 224 and waves == [(f"e1_b{b}_kv32k_w{i}", i * 80000.0 + (0.0 if b == 8 else 18627.2)) for i in range(4) for b in (8, 48)] else 1)
PY
python3 -I $E1/make_e1_inputs_v4scale.py --check >/dev/null 2>&1 && ok "the input generator reproduces the trace, config, inputs, manifest and identity policy on disk byte for byte (deterministic)" || bad "make_e1_inputs_v4scale.py --check: the generated inputs differ from the files on disk"
'''
INPUTS_OLD_START = "assert d['trace']['requests']==112 and len(d['trace']['waves'])==4 and e['iterations']==4320 and e['iterations_band']==[4200,4500] and e['nvtx_forward_ranges']==17280 and e['ranks']==4\n"
derive(REF / "preflight_e1_v3scale.sh", OUT / "preflight_e1_v4scale.sh", [
    ("tests/test_e1_v4scale_gates.py tests/test_e1_supervisor.py", "tests/test_e1_v3scale_gates.py tests/test_e1_v4scale_gates.py tests/test_e1_launch_identity_v4scale.py tests/test_e1_supervisor_v4.py tests/test_e1_reservation_guard.py tests/test_e1_reservation_release.py tests/test_e1_archive_size.py tests/test_e1_supervisor.py"),
    (TRACE_BLOCK_OLD.replace("v3", "v4") if False else rename(TRACE_BLOCK_OLD), TRACE_BLOCK_NEW),
    ("(all four waves in the capture)", "(all eight waves in the capture)"),
    (INPUTS_OLD_START, "assert d['trace']['requests']==224 and len(d['trace']['waves'])==8 and d['trace']['copies']==4 and e['iterations']==8640 and e['iterations_band']==[8400,9000] and e['nvtx_forward_ranges']==34560 and e['ranks']==4 and e['requests']==224\n"),
    ("assert e['kernel_rows']==24501664 and", "assert e['kernel_rows']==49003328==4*(e['kernel_rows_per_wave']['b8']+e['kernel_rows_per_wave']['b48']) and"),
    ("c['temporary_storage_gb']==10 and c['persistent_storage_gb']==8 and c['persistent_storage_with_archive_gb']==16 and", "c['temporary_storage_gb']==15 and c['persistent_storage_gb']==18 and c['persistent_storage_with_archive_gb']==38 and"),
    ("assert [(x['b'],x['wave']) for x in d['dry']['plan']]==[(8,0),(48,0),(8,1),(48,1)] and d['dry']['requests']==112 and d['config_sha256']==sys.argv[2]\"",
     "r=d['reservation']; assert r['prelaunch_max_age_s']==600 and r['supervisor_guard_s']==4680 and r['closeout_and_deletion_budget_s']==720 and r['prelaunch_max_age_s']+c['gpu_held_ceiling_s']+l['gpu_release_deadline_s']<=r['supervisor_guard_s'] and r['supervisor_guard_s']+r['closeout_and_deletion_budget_s']==c['reservation_hard_min']*60 and r['early_release_after_verified_gpu_release']['enabled'] is False\n"
     "assert d['identity']=={'policy':'checks/e1_identity_policy_v4scale.json','policy_version':'v4scale-1','checker':'e1_launch_identity_v4scale.py','reference_run':'e1_nsys_val (accepted, a62e40bd)'}\n"
     "assert [(x['b'],x['wave']) for x in d['dry']['plan']]==[(b,i) for i in range(4) for b in (8,48)] and d['dry']['requests']==224 and d['config_sha256']==sys.argv[2]\""),
    ('ok "E1-V4-SCALE-INPUTS.json: 112 requests, 4 waves, 4320 iterations (4200–4500), 17 280 NVTX ranges, 24 501 664 rows ± 5 %, phase deadlines 30/300/300/600/300 s, release 90 s, controller hard deadline = the 600 s total, ceilings 2700 s / 90 min / 10 / 8 / 16 GB / 64 GiB floor / 128 GiB importer, supervisor limits 10 / 60 / 30 s, four-cell plan"',
     'ok "E1-V4-SCALE-INPUTS.json: 224 requests, 8 waves, 8640 iterations (8400–9000), 34 560 NVTX ranges, 49 003 328 rows ± 5 %, phase deadlines 30/300/300/600/300 s, release 90 s, controller hard deadline = the 600 s total, ceilings 2700 s / 90 min / 15 / 18 / 38 GB / 64 GiB floor / 128 GiB importer, supervisor limits 10 / 60 / 30 s, reservation budget 600 + 4080 + 720 s = 90 min with early release disabled, V4 identity policy registered, eight-wave plan"'),
    ('[ "$(grep -c -- "--label supervisor:\\$PASS -- python3 -I \\$E1/checks/e1_supervisor.py" $E1/run_e1_v4scale.sh)" = 1 ]', '[ "$(grep -c -- "--label supervisor:\\$PASS -- python3 -I \\$E1/checks/e1_supervisor_v4.py" $E1/run_e1_v4scale.sh)" = 1 ] && grep -q -- \'--reservation \\$RUN/\\$PASS/reservation.json --reservation-guard-s \\$RES_GUARD\' $E1/run_e1_v4scale.sh && grep -q -- \'e1_reservation_guard.py age --reservation \\$RUN/\\$PASS/reservation.json --max-age-s \\$RES_AGE\' $E1/run_e1_v4scale.sh'),
    ("grep -q '^NAME=.*RSS_CEIL=128; TMP_CEIL=10; PERS_CEIL=8$' $E1/run_e1_v4scale.sh", "grep -q '^NAME=.*RSS_CEIL=128; TMP_CEIL=15; PERS_CEIL=18; RES_AGE=600; RES_GUARD=4680$' $E1/run_e1_v4scale.sh"),
    ('ok "the driver registers the run supervisor in its own registry and passes the registered ceilings (importer 128 GiB, temporary 10 GB, persistent 8 GB) to the samplers"', 'ok "the driver registers the V4 run supervisor in its own registry with the reservation guard (4680 s), refuses a launch later than 600 s after the reservation\'s creation, and passes the registered ceilings (importer 128 GiB, temporary 15 GB, persistent 18 GB) to the samplers"'),
    ("$E1/checks/e1_supervisor.py && ok \"the supervisor contains no signalling call; its only child is the pinned registry cleanup\"", "$E1/checks/e1_supervisor.py && python3 -I -c \"import sys; src=open(sys.argv[1]).read(); code=src.split('\\\"\\\"\\\"',2)[2]; sys.exit(0 if not any(x in code for x in ('os.kill','import signal','killpg','send_signal','.terminate(','.kill(','subprocess')) else 1)\" $E1/checks/e1_supervisor_v4.py && ok \"the supervisor and its V4 wrapper contain no signalling call; the only child is the pinned registry cleanup\""),
    ('python3 -I -c "import importlib.util,sys; s=importlib.util.spec_from_file_location(\'i\', sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); from pathlib import Path; p=m.load_policy(Path(sys.argv[2])); print(len(p[\'launch_rules\']), len(p[\'profile_rules\']))" $E1/checks/e1_launch_identity_v2.py $E1/checks/e1_identity_policy_v1.json >/dev/null 2>&1 && ok "identity normalization policy loads (no rule on a protected field)" || bad "identity policy does not load"',
     'python3 -I $E1/e1_v4scale_identity_rehearsal.py --reference $AR15 --out $OUT/identity_rehearsal_${TS} > $OUT/identity_rehearsal_${TS}.log 2>&1 && ok "V4 identity policy on the ACTUAL V4 config and trace with the predicted V4 launch record: PASS with the full trace proof; the V1/V2 policy on the same records FAILs on the trace (the V3 defect is detected); negative controls fail ($OUT/identity_rehearsal_${TS}.log)" || bad "V4 identity rehearsal on the actual V4 config (see $OUT/identity_rehearsal_${TS}.log)"'),
    ('python3 -I -c "import sys; sys.exit(0 if float(sys.argv[1]) + 0.71 + 4.7 <= 10 else 1)" "${tu:-99}" && ok "temporary storage now ${tu} GB (retained streams); with the expected stream 0.70 GB and post-run scratch 4.7 GB it stays within 10 GB" || bad "temporary storage now ${tu:-?} GB: expected use would exceed the 10 GB ceiling"',
     'python3 -I -c "import sys; sys.exit(0 if float(sys.argv[1]) + 1.47 + 8.6 <= 15 else 1)" "${tu:-99}" && ok "temporary storage now ${tu} GB (retained streams); with the predicted stream (at most 1.47 GB) and post-run scratch (at most 8.6 GB) it stays within 15 GB" || bad "temporary storage now ${tu:-?} GB: predicted use would exceed the 15 GB ceiling"'),
    ("temporary <= 10 GB; run root <= 8 GB (<= 16 GB with the archive)", "temporary <= 15 GB; run root <= 18 GB (<= 38 GB with the archive); no launch later than 600 s after the reservation's creation; supervisor reservation guard at 4680 s"),
    ('ok "reservation record: pod Running on this node ($(hostname)), creation time recorded" || bad "reservation record is not for this node, not Running, or has no creation time"; }', 'ok "reservation record: pod Running on this node ($(hostname)), creation time recorded" || bad "reservation record is not for this node, not Running, or has no creation time"; python3 -I $E1/checks/e1_reservation_guard.py age --reservation $RUNV4/$PASS/reservation.json --max-age-s 600 | sed "s/^PASS  /INFO  /; s/^FAIL  /INFO  (the driver will refuse) /"; }'),
], keep=[("RUNV3", "RUNV4")])
# ---- pins and manifests
derive(REF / "pin_v3scale.sh", OUT / "pin_v4scale.sh", [
    ("'tests/test_e1_v4scale_gates.py', 'tests/test_e1_supervisor.py'", "'tests/test_e1_v3scale_gates.py', 'tests/test_e1_v4scale_gates.py', 'tests/test_e1_launch_identity_v4scale.py', 'tests/test_e1_supervisor_v4.py', 'tests/test_e1_reservation_guard.py', 'tests/test_e1_reservation_release.py', 'tests/test_e1_archive_size.py', 'tests/test_e1_supervisor.py'"),
])
derive(REF / "hash_scripts_v3scale.sh", OUT / "hash_scripts_v4scale.sh", [
    ("rehearse_e1_v4scale.sh e1_v4scale_rehearsal.py e1_v4scale_fourwave.py rehearse_e2e_v4scale.sh e1_v4scale_mock_launcher.py e1_v4scale_e2e_check.py \\\n", "rehearse_e1_v4scale.sh e1_v4scale_rehearsal.py e1_v4scale_eightwave.py e1_v4scale_identity_rehearsal.py rehearse_e2e_v4scale.sh e1_v4scale_mock_launcher.py e1_v4scale_e2e_check.py \\\n"),
    (" checks/e1_v4scale_gates.py checks/e1_supervisor.py", " checks/e1_v4scale_gates.py checks/e1_v3scale_gates.py checks/e1_launch_identity_v4scale.py checks/e1_identity_policy_v4scale.json checks/e1_supervisor_v4.py checks/e1_reservation_guard.py checks/e1_reservation_release.py checks/e1_archive_size.py checks/e1_supervisor.py"),
    ("tests/test_e1_v4scale_gates.py tests/test_e1_supervisor.py", "tests/test_e1_v3scale_gates.py tests/test_e1_v4scale_gates.py tests/test_e1_launch_identity_v4scale.py tests/test_e1_supervisor_v4.py tests/test_e1_reservation_guard.py tests/test_e1_reservation_release.py tests/test_e1_archive_size.py tests/test_e1_supervisor.py"),
    ("tests/fixtures/e1_nsys_val/server_log_excerpt.log \\\n", "tests/fixtures/e1_nsys_val/server_log_excerpt.log tests/fixtures/e1_v4_scale/launch.predicted.json tests/fixtures/e1_v4_scale/reserve_pod.json \\\n"),
    ("# as the first argument it also writes FREEZE_v4scale.sha256", "# as the first argument it also writes FREEZE_v4scale.sha256 (PROPOSED until a review approves the freeze)"),
])
# ---- mock launcher (renaming only)
derive(REF / "e1_v3scale_mock_launcher.py", OUT / "e1_v4scale_mock_launcher.py", [])

# ---- rehearsal fixtures
def line_edits(path: Path, edits: list):
    lines = path.read_text().split("\n")
    for key, pairs in edits:
        idx = [i for i, l in enumerate(lines) if key in l]
        assert len(idx) == 1, (path.name, key, len(idx))
        for a, b in pairs:
            assert a in lines[idx[0]], (key, a)
            lines[idx[0]] = lines[idx[0]].replace(a, b)
    path.write_text("\n".join(lines))

NEW_HEAD = """TRACE_BYTES = (E1 / "traces" / "e1_v4_scale.csv").read_bytes()
CONFIG_BYTES = (E1 / "configs_v4scale" / "e1_v4_scale" / "profile.json").read_bytes()
V3_TRACE_BYTES = (E1 / "traces" / "e1_v3_scale.csv").read_bytes()


def identity_ok() -> dict:
    \"\"\"A complete, passing report of the V4 identity checker (the fields gate G13 reads).\"\"\"
    return {"verdict": "PASS", "checker": "e1_launch_identity_v4scale.py", "policy_version": "v4scale-1", "normalized": [], "results": [],
            "trace_identity": {"exercised": True, "expected_checks": 20, "checks": [{"check": f"check{i}", "ok": True} for i in range(20)]}}


def completeness_ok() -> dict:
    \"\"\"The completeness checker's record for a complete eight-wave capture (the fields gate G15 reads).\"\"\"
    d = json.loads(INPUTS.read_text())
    return {"fails": 0, "cells": {f"b{p['b']}_kv{p['kv']}_w{p['wave']}": {"pure": p["pure_decode_iters_min"], "ramp": 30} for p in d["dry"]["plan"]},
            "profile_result": {"request_timing_count": 224, "parsed_device_ids": [0, 1, 2, 3], "validation": {"nvtx_forward_ranges": 34560, "kernel_rows": 49003328, "ok": True}},
            "token_accounting": {"requests": 224}, "corpus": {"segments": 224, "bad_len": 0, "npy_mismatch": 0, "dup_topk": 0}, "join": {"duplicate": 0, "missing": 0, "non_monotone": 0}}


def rehearsal_inputs() -> dict:"""
NEW_FIX = """        # V4: boundaries of the reservation and the GPU-held ceiling
        Fixture("reservation_exactly_90_min", "boundary", "reservation held exactly 90.0 min", "PASS", {}, mutate=_set("deleted.deleted_observed_at", "2026-11-19T01:30:00Z")),
        Fixture("gpu_held_exactly_2700s", "boundary", "GPUs released exactly 2700.0 s after the launch (controller record); sampler upper edge 2699.5 s", "PASS", {}, clocks={"t_stop": 2535.5}),
        # V4: workload and recipe identity (G13)
        Fixture("identity_report_from_the_v1_policy", "negative", "a PASS identity report written by the V1/V2 checker and policy (the V3 defect)", F, np(G13=F), mutate=_both(_set("identity.checker", "e1_launch_identity_v2.py"), _set("identity.policy_version", "v1"))),
        Fixture("identity_trace_rule_not_exercised", "negative", "V4 report, verdict PASS, but the derived-trace rule was not exercised", F, np(G13=F), mutate=_set("identity.trace_identity.exercised", False)),
        Fixture("identity_trace_check_not_holding", "negative", "one trace-identity check does not hold although the report says PASS", F, np(G13=F), mutate=_set("identity.trace_identity.checks.7.ok", False)),
        Fixture("identity_trace_proof_incomplete", "negative", "19 of the 20 trace-identity checks are in the report", F, np(G13=F), mutate=lambda w: w.identity["trace_identity"]["checks"].pop()),
        Fixture("identity_report_inconclusive", "negative", "the identity checker could not read the launch record", INC, np(G9=INC, G13=INC), mutate=_set("identity.verdict", "INCONCLUSIVE")),
        # V4: the trace and config in the run directory (G14)
        Fixture("run_trace_is_the_v3_trace", "negative", "the run used the two-copy V3 trace (112 requests)", F, np(G14=F), mutate=_attr("trace_bytes", V3_TRACE_BYTES)),
        Fixture("run_trace_rows_reordered", "negative", "two rows of the third copy swapped in the run's trace", F, np(G14=F), mutate=lambda w: setattr(w, "trace_bytes", _swap_rows(w.trace_bytes, 130, 131))),
        Fixture("run_trace_missing", "negative", "no trace copy in the run directory", INC, np(G14=INC), mutate=_attr("trace_bytes", None)),
        Fixture("run_config_not_the_registered_one", "negative", "the run's profile config differs from the registered V4 config in one field", F, np(G14=F), mutate=lambda w: setattr(w, "config_bytes", w.config_bytes.replace(b'"stream": false', b'"stream": true'))),
        # V4: request, iteration, routing and rank coverage (G15)
        Fixture("coverage_cell_missing", "negative", "the b=8 cell of the fourth copy is absent from the completeness record", F, np(G15=F), mutate=lambda w: w.completeness["cells"].pop("b8_kv32768_w3")),
        Fixture("coverage_unplanned_cell", "negative", "a cell that is not in the plan appears", F, np(G15=F), mutate=_set("completeness.cells.b16_kv32768_w0", {"pure": 1000, "ramp": 60})),
        Fixture("coverage_pure_decode_below_floor", "negative", "one cell has 399 pure-decode iterations", F, np(G15=F), mutate=_set("completeness.cells.b48_kv32768_w2.pure", 399)),
        Fixture("coverage_one_request_short", "negative", "223 request timings", F, np(G15=F), mutate=_set("completeness.profile_result.request_timing_count", 223)),
        Fixture("coverage_three_ranks", "negative", "kernels from three ranks only", F, np(G15=F), mutate=_set("completeness.profile_result.parsed_device_ids", [0, 1, 2])),
        Fixture("coverage_routing_array_defect", "negative", "one routing array differs from its corpus segment", F, np(G15=F), mutate=_set("completeness.corpus.npy_mismatch", 1)),
        Fixture("coverage_record_missing", "negative", "the completeness checker wrote no record", INC, np(G15=INC), mutate=_attr("completeness", None)),
    ]
    return fx


def _swap_rows(data: bytes, i: int, j: int) -> bytes:
    lines = data.decode().split("\\n")
    lines[i], lines[j] = lines[j], lines[i]
    return "\\n".join(lines).encode()
"""
derive(REF / "e1_v3scale_rehearsal.py", OUT / "e1_v4scale_rehearsal.py", [
    ('"G8.offline_import", "G8b.recovered_trace_complete", "G9.scientific_completeness", "G10.bounds", "G11.storage_and_memory", "G12.closeout"]', '"G8.offline_import", "G8b.recovered_trace_complete", "G9.scientific_completeness", "G10.bounds", "G11.storage_and_memory", "G12.closeout",\n       "G13.workload_and_recipe_identity", "G14.four_copy_trace_in_run", "G15.request_iteration_routing_rank_coverage"]'),
    ("def rehearsal_inputs() -> dict:", NEW_HEAD),
    ("def fake_tools(tmp: Path, *, kernel_rows: int = 2450, nvtx: int = 17280, importer_rc: int = 0)", "def fake_tools(tmp: Path, *, kernel_rows: int = 4900, nvtx: int = 34560, importer_rc: int = 0)"),
    ('    identity: dict | None = field(default_factory=lambda: {"verdict": "PASS", "normalized": []})\n', '    identity: dict | None = field(default_factory=identity_ok)\n    completeness: dict | None = field(default_factory=completeness_ok)\n    trace_bytes: bytes | None = TRACE_BYTES\n    config_bytes: bytes | None = CONFIG_BYTES\n'),
    ('    in_session: dict = field(default_factory=lambda: {"kernel_rows": 2450, "nvtx": 17280})', '    in_session: dict = field(default_factory=lambda: {"kernel_rows": 4900, "nvtx": 34560})'),
    ('    tools: dict = field(default_factory=lambda: {"kernel_rows": 2450, "nvtx": 17280, "importer_rc": 0})', '    tools: dict = field(default_factory=lambda: {"kernel_rows": 4900, "nvtx": 34560, "importer_rc": 0})'),
    ("    temp_gb: float = 1.4\n    persistent_gb: float = 6.0\n    scratch_gb: float = 4.7\n    importer_rss_gb: float = 17.0\n", "    temp_gb: float = 2.7\n    persistent_gb: float = 12.6\n    scratch_gb: float = 8.6\n    importer_rss_gb: float = 31.0\n"),
    ("int((0.67 + (self.temp_gb - 0.67) *", "int((1.36 + (self.temp_gb - 1.36) *"),
    ("int((0.67 + 0.7 + self.scratch_gb *", "int((1.36 + 1.32 + self.scratch_gb *"),
    ('    (run / "run.log").write_text(f"[e1_v4_scale] start 2026-11-19T00:00:00Z on {HOST} x\\n")\n', '    (run / "run.log").write_text(f"[e1_v4_scale] start 2026-11-19T00:00:00Z on {HOST} x\\n")\n    for name, data in (("trace.csv", w.trace_bytes), ("profile.json", w.config_bytes)):\n        if data is not None:\n            (run / name).write_bytes(data)\n'),
    ('    for key, name, obj in (("identity", "identity.json", w.identity), ("deleted", "reservation_deleted.json", w.deleted)):', '    for key, name, obj in (("identity", "identity.json", w.identity), ("completeness", "completeness.json", w.completeness), ("deleted", "reservation_deleted.json", w.deleted)):'),
    ('(("--importer", "importer"), ("--nsys", "nsys"), ("--identity", "identity"),', '(("--importer", "importer"), ("--nsys", "nsys"), ("--identity", "identity"), ("--completeness", "completeness"),'),
    ('"temporary 10.0 GB, persistent 8.0 GB, importer 128 GiB, MemAvailable exactly at the floor"', '"temporary 15.0 GB, persistent 18.0 GB, importer 128 GiB, MemAvailable exactly at the floor"'),
    ('_scale_rows("post_storage", "temporary", 10.0), _scale_rows("storage", "persistent", 8.0), _scale_rows("post_storage", "persistent", 8.0)', '_scale_rows("post_storage", "temporary", 15.0), _scale_rows("storage", "persistent", 18.0), _scale_rows("post_storage", "persistent", 18.0)'),
    ('"post-run scratch peak makes temporary storage 10.05 GB", F, np(G11=F), mutate=_scale_rows("post_storage", "temporary", 10.05)', '"post-run scratch peak makes temporary storage 15.05 GB", F, np(G11=F), mutate=_scale_rows("post_storage", "temporary", 15.05)'),
    ('"run directory peak 8.05 GB", F, np(G11=F), mutate=_scale_rows("storage", "persistent", 8.05)', '"run directory peak 18.05 GB", F, np(G11=F), mutate=_scale_rows("storage", "persistent", 18.05)'),
    ('"4000 iterations recovered (band 4200–4500)", F, np(G8b=F), mutate=_both(_set("tools.nvtx", 16000), _set("in_session.nvtx", 16000))', '"8000 iterations recovered (band 8400–9000)", F, np(G8b=F), mutate=_both(_set("tools.nvtx", 32000), _set("in_session.nvtx", 32000))'),
    ('mutate=_both(_set("tools.kernel_rows", 2200), _set("in_session.kernel_rows", 2200))', 'mutate=_both(_set("tools.kernel_rows", 4400), _set("in_session.kernel_rows", 4400))'),
    ('mutate=_set("tools.kernel_rows", 2440))', 'mutate=_set("tools.kernel_rows", 4880))'),
    ('"four-wave completeness checker exit 1"', '"eight-wave completeness checker exit 1"'),
    ('"configuration identity report FAIL", F, np(G9=F), mutate=_set("identity.verdict", "FAIL")', '"configuration identity report FAIL", F, np(G9=F, G13=F), mutate=_set("identity.verdict", "FAIL")'),
    ('"no configuration identity report", INC, np(G9=INC), mutate=_attr("identity", None)', '"no configuration identity report", INC, np(G9=INC, G13=INC), mutate=_attr("identity", None)'),
    ('    ]\n    return fx\n', NEW_FIX),
    ("exactly at the ceilings, temporary lack of importer progress", "exactly at the ceilings (storage, memory, reservation 90 min, GPU-held 2700 s), temporary lack of importer progress"),
])
line_edits(OUT / "e1_v4scale_rehearsal.py", [
    ('Fixture("half_trace_two_waves_only"', [("only two of the four waves in the trace (2160 iterations)", "only four of the eight waves in the trace (4320 iterations)"), ('_set("tools.nvtx", 8640)', '_set("tools.nvtx", 17280)'), ('_set("in_session.nvtx", 8640)', '_set("in_session.nvtx", 17280)'), ('_set("tools.kernel_rows", 1225)', '_set("tools.kernel_rows", 2450)'), ('_set("in_session.kernel_rows", 1225)', '_set("in_session.kernel_rows", 2450)')]),
])
# ---- e2e check
derive(REF / "e1_v3scale_e2e_check.py", OUT / "e1_v4scale_e2e_check.py", [
    ('BASE = {"G0.source_and_executable": F, "G6.gpu_release_interval": INC, "G9.scientific_completeness": F, "G10.bounds": INC}  # a rehearsal launch; no GPU was held; no scientific profile',
     'BASE = {"G0.source_and_executable": F, "G6.gpu_release_interval": INC, "G9.scientific_completeness": F, "G10.bounds": INC,  # a rehearsal launch; no GPU was held; no scientific profile\n        "G13.workload_and_recipe_identity": INC, "G14.four_copy_trace_in_run": F, "G15.request_iteration_routing_rank_coverage": INC}  # no launch record (mock launcher); the rehearsal copy of the config is renamed; no completeness record'),
    ('choices=["success", "ambiguous", "memory_breach", "storage_breach"]', 'choices=["success", "ambiguous", "memory_breach", "storage_breach", "reservation_guard"]'),
    ('    if a.scenario == "storage_breach":  # the driver was terminated', '    if a.scenario in ("storage_breach", "reservation_guard"):  # the driver was terminated'),
    ('    if a.scenario == "storage_breach":\n        steps = (sup.get("response") or {}).get("steps") or [{}]\n        check("breach.marker_and_cause", (run / "STORAGE-BREACH").exists() and [c.get("kind") for c in sup.get("causes", [])] == ["persistent_storage_above_ceiling"],',
     '    if a.scenario == "reservation_guard":\n        c0 = (sup.get("causes") or [{}])[0]\n        check("guard.cause", [c.get("cause") for c in sup.get("causes", [])] == ["reservation_guard"] and c0.get("kind") == "run not ended when the reservation guard passed" and (c0.get("reservation_age_s") or 0) > c0.get("limit_s", 1e9) and not (run / "STORAGE-BREACH").exists() and not (run / "MEMORY-BREACH").exists(),\n              f"supervisor causes {[(c.get(\'cause\'), c.get(\'kind\'), c.get(\'reservation_age_s\'), c.get(\'limit_s\')) for c in sup.get(\'causes\', [])]}; no breach marker")\n    if a.scenario in ("storage_breach", "reservation_guard"):\n        steps = (sup.get("response") or {}).get("steps") or [{}]\n        check("breach.marker_and_cause", a.scenario == "reservation_guard" or (run / "STORAGE-BREACH").exists() and [c.get("kind") for c in sup.get("causes", [])] == ["persistent_storage_above_ceiling"],'),
])
# ---- e2e rehearsal script
derive(REF / "rehearse_e2e_v3scale.sh", OUT / "rehearse_e2e_v4scale.sh", [
    ("SCENARIOS=${*:-success ambiguous memory_breach storage_breach}", "SCENARIOS=${*:-success ambiguous memory_breach storage_breach reservation_guard}"),
    ("#   storage_breach  the run root exceeds the (rehearsal) ceiling during the capture: marker → supervisor → pinned registry cleanup\n", "#   storage_breach  the run root exceeds the (rehearsal) ceiling during the capture: marker → supervisor → pinned registry cleanup\n#   reservation_guard  the (rehearsal) reservation guard, 420 s after the reservation record's creation, passes during a 900 s mock\n#                   capture: supervisor → pinned registry cleanup (the driver's preflight must have finished within those 420 s)\n"),
    ('    storage_breach) envs="E1_V4S_MOCK_SCENARIO=storage_ballast E1_V4S_MOCK_STORAGE_BALLAST_MB=64 E1_V4S_MOCK_CAPTURE_S=40 E1_V4S_REHEARSAL_PERSISTENT_GB=0.032";;\n', '    storage_breach) envs="E1_V4S_MOCK_SCENARIO=storage_ballast E1_V4S_MOCK_STORAGE_BALLAST_MB=64 E1_V4S_MOCK_CAPTURE_S=40 E1_V4S_REHEARSAL_PERSISTENT_GB=0.032";;\n    reservation_guard) envs="E1_V4S_MOCK_SCENARIO=success E1_V4S_MOCK_CAPTURE_S=900 E1_V4S_REHEARSAL_RESERVATION_GUARD_S=420";;\n'),
    ("  if [ $sc != storage_breach ]; then\n", "  if [ $sc != storage_breach ] && [ $sc != reservation_guard ]; then\n"),
])
# ---- gate tests
derive(REFT / "test_e1_v3scale_gates.py", OUTT / "test_e1_v4scale_gates.py", [
    ('post_memory=a["post_memory"], reservation_deleted=a["deleted"], extras=a["extras"])', 'post_memory=a["post_memory"], reservation_deleted=a["deleted"], extras=a["extras"], completeness=a["completeness"])'),
    ('assert len(FIXTURES) == 99 and kinds.count("positive") == 3 and kinds.count("boundary") == 4 and kinds.count("negative") == 92', 'assert len(FIXTURES) == 117 and kinds.count("positive") == 3 and kinds.count("boundary") == 6 and kinds.count("negative") == 108'),
    ('assert report["verdict"] == "FAIL" and len(report["gates"]) == 1', 'assert report["verdict"] == "FAIL" and [g["gate"] for g in report["gates"]] == ["G3.phases_in_order", *G.NEW]'),
    ('assert d["expected"]["iterations"] == 4320 and d["trace"]["requests"] == 112 and len(d["dry"]["plan"]) == 4', 'assert d["expected"]["iterations"] == 8640 and d["trace"]["requests"] == 224 and len(d["dry"]["plan"]) == 8\n    assert d["ceilings"]["temporary_storage_gb"] == 15 and d["ceilings"]["persistent_storage_gb"] == 18 and d["ceilings"]["persistent_storage_with_archive_gb"] == 38 and d["ceilings"]["reservation_hard_min"] == 90 and d["ceilings"]["gpu_held_ceiling_s"] == 2700\n    assert d["limits"]["gpu_release_deadline_s"] == 90 and d["phase_deadlines_s"]["stop_accepted"] == 30 and d["phase_deadlines_s"]["session_close_after_server_sigterm"] == 300'),
])
