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

# ---- revision 2 (review decision of 2026-11-19 on 39561325): preflight memory 488 GiB; launch rule from the reservation budget;
# further rehearsal scenarios. These edits are applied after the first-revision edits of the same file, so the diffs written
# below still show every difference from the frozen V3 original.
NAME_LINE = "RSS_CEIL=128; TMP_CEIL=15; PERS_CEIL=18; RES_GUARD=4680; RES_RUN=4210; GPU_CEIL=2700"
ABORTED = '("storage_breach", "reservation_guard", "ceiling_before_stop", "export_near_guard", "guard_and_storage", "sampler_lost")'
REV2 = {
    "run_e1_v4scale.sh": [
        ("RSS_CEIL=128; TMP_CEIL=15; PERS_CEIL=18; RES_AGE=600; RES_GUARD=4680\n", NAME_LINE + "\n"),
        ("RES_GUARD=${E1_V4S_REHEARSAL_RESERVATION_GUARD_S:-$RES_GUARD}\n", "RES_GUARD=${E1_V4S_REHEARSAL_RESERVATION_GUARD_S:-$RES_GUARD}; RES_RUN=${E1_V4S_REHEARSAL_RUN_BUDGET_S:-$RES_RUN}; GPU_CEIL=${E1_V4S_REHEARSAL_SUPERVISOR_CEILING_S:-$GPU_CEIL}\n"),
        ("; reservation guard $RES_GUARD s\"", "; reservation guard $RES_GUARD s, run budget $RES_RUN s, supervisor GPU-held ceiling $GPU_CEIL s\""),
        ("# reservation budget, first part: no launch later than $RES_AGE s after the reservation's creation (the end of the run would not fit the 90 min ceiling)\n"
         "python3 -I $E1/checks/e1_reservation_guard.py age --reservation $RUN/$PASS/reservation.json --max-age-s $RES_AGE | tee -a $RUN/$PASS/run.log\n"
         "[ \"${PIPESTATUS[0]}\" = 0 ] || { echo \"reservation too old (or its age unknown) for a launch; not starting\"; exit 7; }\n",
         "# reservation budget, launch rule (E1-V4-TIMELINE.md): start nothing unless the time left until the supervisor's reservation guard\n"
         "# ($RES_GUARD s after the reservation's creation) is at least the registered run budget ($RES_RUN s = this check to the launcher's start\n"
         "# + GPU-held ceiling + release + CPU-only processing after the release). No observer and no launcher has been started at this point.\n"
         "python3 -I $E1/checks/e1_reservation_guard.py eligible --reservation $RUN/$PASS/reservation.json --guard-s $RES_GUARD --run-budget-s $RES_RUN | tee -a $RUN/$PASS/run.log\n"
         "[ \"${PIPESTATUS[0]}\" = 0 ] || { echo \"LAUNCH REFUSED: the remaining reservation budget is insufficient (or the reservation's age is unknown); nothing was started\" | tee -a $RUN/$PASS/run.log; exit 7; }\n"),
        ("--gpu-held-ceiling-s 2700 --ack-s 10", "--gpu-held-ceiling-s $GPU_CEIL --ack-s 10"),
    ],
    "preflight_e1_v4scale.sh": [
        ("tests/test_e1_archive_size.py tests/test_e1_supervisor.py", "tests/test_e1_archive_size.py tests/test_e1_preflight_memory.py tests/test_e1_v4scale_reservation_scenarios.py tests/test_e1_supervisor.py"),
        ("c['host_mem_available_preflight_min_gb']==448", "c['host_mem_available_preflight_min_gb']==488==c['importer_rss_ceiling_gb']+c['host_mem_available_floor_gb']+d['memory_budget']['server_allowance_gib']"),
        ("r=d['reservation']; assert r['prelaunch_max_age_s']==600 and r['supervisor_guard_s']==4680 and r['closeout_and_deletion_budget_s']==720 and r['prelaunch_max_age_s']+c['gpu_held_ceiling_s']+l['gpu_release_deadline_s']<=r['supervisor_guard_s'] and r['supervisor_guard_s']+r['closeout_and_deletion_budget_s']==c['reservation_hard_min']*60 and r['early_release_after_verified_gpu_release']['enabled'] is False\n",
         "r=d['reservation']; b=r['run_budget_s']; assert r['hard_ceiling_s']==5400==c['reservation_hard_min']*60 and r['supervisor_guard_s']==4680 and r['closeout_and_deletion_budget_s']==720==r['hard_ceiling_s']-r['supervisor_guard_s']==sum(r['closeout_and_deletion_parts_s'].values())\n"
         "assert b=={'eligibility_check_to_launcher_start':20,'gpu_held_ceiling':2700,'gpu_release_after_ceiling':90,'cpu_only_after_release':1400} and b['gpu_held_ceiling']==c['gpu_held_ceiling_s'] and b['gpu_release_after_ceiling']==l['gpu_release_deadline_s'] and sum(r['cpu_only_after_release_s'].values())==b['cpu_only_after_release']>=d['predictions']['export_and_parse_after_release_s'][2]\n"
         "assert r['run_budget_total_s']==sum(b.values())==4210<=r['supervisor_guard_s'] and r['launch_max_age_s']==r['supervisor_guard_s']-r['run_budget_total_s']==470 and r['early_release_after_verified_gpu_release']['enabled'] is False\n"),
        ("reservation budget 600 + 4080 + 720 s = 90 min with early release disabled", "preflight memory 488 GiB = 128 + 64 + 296, reservation budget: launch only while 4680 s - age >= run budget 4210 s (20 + 2700 + 90 + 1400; age <= 470 s), guard 4680 s + closeout and deletion 720 s = 90 min, early release disabled"),
        ("grep -q -- 'e1_reservation_guard.py age --reservation \\$RUN/\\$PASS/reservation.json --max-age-s \\$RES_AGE' $E1/run_e1_v4scale.sh", "grep -q -- 'e1_reservation_guard.py eligible --reservation \\$RUN/\\$PASS/reservation.json --guard-s \\$RES_GUARD --run-budget-s \\$RES_RUN' $E1/run_e1_v4scale.sh && grep -q -- '--gpu-held-ceiling-s \\$GPU_CEIL --ack-s 10' $E1/run_e1_v4scale.sh"),
        ("grep -q '^NAME=.*RSS_CEIL=128; TMP_CEIL=15; PERS_CEIL=18; RES_AGE=600; RES_GUARD=4680$' $E1/run_e1_v4scale.sh", "grep -q '^NAME=.*" + NAME_LINE + "$' $E1/run_e1_v4scale.sh"),
        ("refuses a launch later than 600 s after the reservation's creation,", "refuses a launch, before any observer or the launcher is started, unless 4680 s minus the reservation's age is at least the run budget of 4210 s, enforces the GPU-held ceiling of 2700 s before the stop request,"),
        ('[ "${ma:-0}" -ge 448 ] && ok "host MemAvailable ${ma} GiB >= 448 = importer ceiling 128 + floor 64 + 256 for the server; margin above that ${ma:+$((ma - 448))} GiB" || bad "host MemAvailable ${ma} GiB < 448 (importer ceiling 128 + floor 64 + 256 for the server)"\n'
         '{ [ "$cgmax" = max ] || [ "$((cgmax / 1073741824))" -ge 448 ]; } && ok "no cgroup memory limit below 448 GiB applies to this session (memory.max ${cgmax})" || bad "a cgroup memory limit of $((cgmax / 1073741824)) GiB applies: the importer ceiling of 128 GiB is not consistent with it"\n',
         '# exact check in kB against the registered minimum (488 GiB = importer ceiling 128 + runtime floor 64 + 296 for the run); two result lines\n'
         'mo=$(python3 -I $E1/checks/e1_preflight_memory.py --meminfo /proc/meminfo --inputs $INP --cgroup-max "$cgmax" 2>&1); mrc=$?\n'
         'for n in 1 2; do ml=$(echo "$mo" | sed -n "${n}p"); case "$ml" in "PASS  "*) ok "${ml#PASS  }";; *) bad "${ml:-host-memory check: no result line $n (exit code $mrc)}";; esac; done\n'
         '[ "$mrc" = 0 ] || bad "host-memory check exit code $mrc"\n'
         'mfx=$REPO/tests/fixtures/e1_v4_scale; mb=""; for f in 487GiB 488GiB_minus_1kB 488GiB 488GiB_plus_1kB 489GiB; do python3 -I $E1/checks/e1_preflight_memory.py --meminfo $mfx/meminfo_$f.txt --inputs $INP >/dev/null 2>&1; mb="$mb$?"; done\n'
         '[ "$mb" = 11000 ] && ok "host-memory boundary fixtures: 487 GiB and 488 GiB - 1 kB refused; 488 GiB, 488 GiB + 1 kB and 489 GiB accepted" || bad "host-memory boundary fixtures gave exit codes $mb (expected 11000)"\n'),
        ("no launch later than 600 s after the reservation's creation; supervisor reservation guard at 4680 s", "launch only while 4680 s - reservation age >= run budget 4210 s (age <= 470 s); supervisor reservation guard at 4680 s; reservation deleted by 5400 s; preflight MemAvailable >= 488 GiB"),
        ("python3 -I $E1/checks/e1_reservation_guard.py age --reservation $RUNV4/$PASS/reservation.json --max-age-s 600 |", "python3 -I $E1/checks/e1_reservation_guard.py eligible --reservation $RUNV4/$PASS/reservation.json --guard-s 4680 --run-budget-s 4210 |"),
    ],
    "monitor_e1_v4scale.sh": [
        ('    print(f"reservation: age {res_age / 60:.1f} min (supervisor guard 78 min, ceiling 90 min)")\n',
         '    print(f"reservation: age {res_age:.0f} s = {res_age / 60:.1f} min (launch permitted to 470 s; supervisor guard 4680 s; DELETE BY {f(time.time() - res_age + 5400)} = 90 min, {5400 - res_age:.0f} s left)")\n'
         '    if res_age > 5400 and not load(run+"/reservation_deleted.json"): print("ABORT-CONDITION the reservation is older than 90 min and no deletion record exists (gate G10 will FAIL): delete it now if the GPUs are idle")\n'),
    ],
    "closeout_e1_v4scale.sh": [
        ('echo "INFO  reservation pod (from the laptop): delete immediately after a CLEAN closeout', '[ -f $RUN/$PASS/reservation.json ] && python3 -I $E1/checks/e1_reservation_guard.py deadline --reservation $RUN/$PASS/reservation.json --ceiling-s 5400 | sed "s/^PASS  /INFO  /; s/^FAIL  /INFO  RESERVATION CEILING PASSED (gate G10 will FAIL): /; s/^INCONCLUSIVE/INFO /"\necho "INFO  reservation pod (from the laptop): delete immediately after a CLEAN closeout'),
    ],
    "pin_v4scale.sh": [
        ("'tests/test_e1_archive_size.py', 'tests/test_e1_supervisor.py'", "'tests/test_e1_archive_size.py', 'tests/test_e1_preflight_memory.py', 'tests/test_e1_v4scale_reservation_scenarios.py', 'tests/test_e1_supervisor.py'"),
    ],
    "hash_scripts_v4scale.sh": [
        (" checks/e1_reservation_guard.py", " checks/e1_reservation_guard.py checks/e1_preflight_memory.py"),
        ("tests/test_e1_archive_size.py tests/test_e1_supervisor.py", "tests/test_e1_archive_size.py tests/test_e1_preflight_memory.py tests/test_e1_v4scale_reservation_scenarios.py tests/test_e1_supervisor.py"),
        ("tests/fixtures/e1_v4_scale/reserve_pod.json \\\n", "tests/fixtures/e1_v4_scale/reserve_pod.json tests/fixtures/e1_v4_scale/meminfo_487GiB.txt tests/fixtures/e1_v4_scale/meminfo_488GiB_minus_1kB.txt tests/fixtures/e1_v4_scale/meminfo_488GiB.txt tests/fixtures/e1_v4_scale/meminfo_488GiB_plus_1kB.txt tests/fixtures/e1_v4_scale/meminfo_489GiB.txt \\\n"),
    ],
    "e1_v4scale_mock_launcher.py": [
        ("the capture). Timings: E1_V4S_MOCK_{STARTUP,CAPTURE,FLUSH,IMPORT}_S.\n", "the capture; with E1_V4S_MOCK_STORAGE_BALLAST_AT_RESERVATION_AGE_S not before the reservation record is that old).\nTimings: E1_V4S_MOCK_{STARTUP,CAPTURE,FLUSH,IMPORT}_S. E1_V4S_MOCK_EXPORT_UNTIL_RESERVATION_AGE_S: after the report, the launcher\nstays in its CPU-only export phase until the reservation record is that old (a long export and parse after the GPU release).\n"),
        ("import json\nimport os\n", "import calendar\nimport json\nimport os\n"),
        ("def launch(config: Path, provenance: Path) -> int:\n", "def reservation_created(config: Path) -> float:\n    \"\"\"Creation time in the reservation record next to the rehearsal copy of the profile config.\"\"\"\n    return float(calendar.timegm(time.strptime(json.loads((config.parent / \"reservation.json\").read_text())[\"creation_timestamp\"], \"%Y-%m-%dT%H:%M:%SZ\")))\n\n\ndef launch(config: Path, provenance: Path) -> int:\n"),
        ("        t_cap, end = time.time(), time.time() + CAPTURE_S\n", "        t_cap, end = time.time(), time.time() + CAPTURE_S\n        ballast_age = env_f(\"STORAGE_BALLAST_AT_RESERVATION_AGE_S\", 0)\n        ballast_from = reservation_created(config) + ballast_age if ballast_age > 0 else 0.0\n"),
        ("if SCENARIO == \"storage_ballast\" and time.time() - t_cap > 3 and not", "if SCENARIO == \"storage_ballast\" and time.time() - t_cap > 3 and time.time() >= ballast_from and not"),
        ("    after = frontend.verify_after_launch(log_dir)\n", "    until = env_f(\"EXPORT_UNTIL_RESERVATION_AGE_S\", 0)\n    if rc == 0 and until > 0:\n        created = reservation_created(config)\n        print(f\"[mock] export and parse (CPU only) continue until the reservation record is {until:.0f} s old\", flush=True)\n        while time.time() - created < until:\n            time.sleep(0.5)\n        print(\"[mock] export and parse ended\", flush=True)\n    after = frontend.verify_after_launch(log_dir)\n"),
    ],
    "e1_v4scale_rehearsal.py": [
        ('        Fixture("reservation_exactly_90_min", "boundary", "reservation held exactly 90.0 min", "PASS", {}, mutate=_set("deleted.deleted_observed_at", "2026-11-19T01:30:00Z")),\n',
         '        Fixture("reservation_exactly_90_min", "boundary", "reservation held exactly 90.0 min", "PASS", {}, mutate=_set("deleted.deleted_observed_at", "2026-11-19T01:30:00Z")),\n'
         '        Fixture("reservation_deleted_1s_late", "negative", "deletion observed 1 s after the 90 min ceiling (delayed closeout or deletion)", F, np(G10=F), mutate=_set("deleted.deleted_observed_at", "2026-11-19T01:30:01Z")),\n'),
    ],
    "test_e1_v4scale_gates.py": [
        ('assert len(FIXTURES) == 117 and kinds.count("positive") == 3 and kinds.count("boundary") == 6 and kinds.count("negative") == 108', 'assert len(FIXTURES) == 118 and kinds.count("positive") == 3 and kinds.count("boundary") == 6 and kinds.count("negative") == 109'),
        ('d["ceilings"]["host_mem_available_preflight_min_gb"] == 448', 'd["ceilings"]["host_mem_available_preflight_min_gb"] == 488'),
    ],
}
E2E_NEW = "launch_late_permitted launch_refused ceiling_before_stop complete_near_guard export_near_guard guard_and_storage sampler_lost supervisor_lost"
REV2["rehearse_e2e_v4scale.sh"] = [
    ("SCENARIOS=${*:-success ambiguous memory_breach storage_breach reservation_guard}", "SCENARIOS=${*:-success ambiguous memory_breach storage_breach reservation_guard " + E2E_NEW + "}\n"
     "# scenarios that end in a refusal or an abort: no post-run\nABORTED=\" storage_breach reservation_guard launch_refused ceiling_before_stop export_near_guard guard_and_storage sampler_lost \""),
    ("#                   capture: supervisor → pinned registry cleanup (the driver's preflight must have finished within those 420 s)\n",
     "#                   capture: supervisor → pinned registry cleanup (the driver's preflight must have finished within those 420 s)\n"
     "# Revision 2 (reservation budget; the rehearsal guard is 420 s and the rehearsal run budget 200 s unless stated):\n"
     "#   launch_late_permitted  the reservation record is 320 s old when the driver starts; with the REGISTERED guard and run budget the\n"
     "#                   launch rule still permits the launch (age at the check 400 – 470 s); the run then completes as in `success`\n"
     "#   launch_refused  the record is 380 s old when the driver starts; with the registered values the rule refuses (age > 470 s):\n"
     "#                   exit 7 before any observer, supervisor or launcher exists\n"
     "#   ceiling_before_stop  the supervisor's GPU-held ceiling (rehearsal value 45 s) passes during a 900 s mock capture: registry cleanup\n"
     "#   complete_near_guard  the mock launcher's CPU-only export phase lasts until the record is 375 s old; the run ends through its stop\n"
     "#                   file shortly before the guard: no response\n"
     "#   export_near_guard  the export phase would last until 900 s: the guard passes after the GPU release, during the export: registry\n"
     "#                   cleanup; the importer has finished and is not signalled; the preserved stream stays\n"
     "#   guard_and_storage  the run root exceeds its rehearsal ceiling at 419 s, one second before the guard: one response, whichever\n"
     "#                   cause the supervisor sees first, both recorded if they are present together\n"
     "#   sampler_lost    the storage sampler is ended 8 s after its start (verified cleanup of a registry holding only its record):\n"
     "#                   the supervisor's heartbeat rule → registry cleanup\n"
     "#   supervisor_lost the supervisor is ended 8 s after its start (verified cleanup of its own registry): the run completes under the\n"
     "#                   controller; the incomplete supervisor record makes gate G11 INCONCLUSIVE\n"),
    ('    reservation_guard) envs="E1_V4S_MOCK_SCENARIO=success E1_V4S_MOCK_CAPTURE_S=900 E1_V4S_REHEARSAL_RESERVATION_GUARD_S=420";;\n',
     '    reservation_guard) envs="E1_V4S_MOCK_SCENARIO=success E1_V4S_MOCK_CAPTURE_S=900 $RG";;\n'
     '    launch_late_permitted) envs="E1_V4S_MOCK_SCENARIO=success"; backdate 320;;\n'
     '    launch_refused) envs="E1_V4S_MOCK_SCENARIO=success"; backdate 380;;\n'
     '    ceiling_before_stop) envs="E1_V4S_MOCK_SCENARIO=success E1_V4S_MOCK_CAPTURE_S=900 E1_V4S_REHEARSAL_SUPERVISOR_CEILING_S=45";;\n'
     '    complete_near_guard) envs="E1_V4S_MOCK_SCENARIO=success E1_V4S_MOCK_EXPORT_UNTIL_RESERVATION_AGE_S=375 $RG";;\n'
     '    export_near_guard) envs="E1_V4S_MOCK_SCENARIO=success E1_V4S_MOCK_EXPORT_UNTIL_RESERVATION_AGE_S=900 $RG";;\n'
     '    guard_and_storage) envs="E1_V4S_MOCK_SCENARIO=storage_ballast E1_V4S_MOCK_STORAGE_BALLAST_MB=64 E1_V4S_MOCK_STORAGE_BALLAST_AT_RESERVATION_AGE_S=419 E1_V4S_MOCK_CAPTURE_S=900 E1_V4S_REHEARSAL_PERSISTENT_GB=0.032 $RG";;\n'
     '    sampler_lost)   envs="E1_V4S_MOCK_SCENARIO=success E1_V4S_MOCK_CAPTURE_S=900"; end_one owned.tsv storagewatch: $O/sampler_end & hp=$!;;\n'
     '    supervisor_lost) envs="E1_V4S_MOCK_SCENARIO=success E1_V4S_MOCK_CAPTURE_S=60"; end_one supervisor.tsv supervisor: $O/supervisor_end & hp=$!;;\n'),
    ("  case $sc in\n    success) ", "  RG=\"E1_V4S_REHEARSAL_RESERVATION_GUARD_S=420 E1_V4S_REHEARSAL_RUN_BUDGET_S=200\"; hp=\"\"\n"
     "  # the reservation record as if the pod had been created $1 seconds ago (launch-rule scenarios; the registered guard and budget apply)\n"
     "  backdate() { python3 -I -c \"import json,sys,time; p=sys.argv[1]; d=json.load(open(p)); d['creation_timestamp']=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time()-float(sys.argv[2]))); json.dump(d, open(p,'w'), indent=1)\" $R/$PASS/reservation.json $1; }\n"
     "  # end ONE registered process, 8 s after its record appears: a registry holding only that record, through the pinned verified cleanup\n"
     "  end_one() { for i in $(seq 1 300); do grep -q \"$2\" $R/pids/$1 2>/dev/null && break; sleep 1; done; sleep 8; { grep '^#' $R/pids/$1; grep \"$2\" $R/pids/$1; } > $3.tsv; python3 -I $E1/checks/e1_pids.py cleanup --registry $3.tsv --execute --grace 5 --kill9 --log $3.jsonl > $3.log 2>&1; }\n"
     "  case $sc in\n    success) "),
    ("bash $E1/run_e1_v4scale.sh > $O/driver.log 2>&1; drc=$?\n", "bash $E1/run_e1_v4scale.sh > $O/driver.log 2>&1; drc=$?\n  [ -n \"$hp\" ] && wait $hp\n"),
    ("  if [ $sc != storage_breach ] && [ $sc != reservation_guard ]; then\n", "  if [ \"${ABORTED#* $sc }\" = \"$ABORTED\" ]; then\n"),
]
REV2["e1_v4scale_e2e_check.py"] = [
    ("import argparse\n", "import argparse\nimport calendar\n"),
    ("import os\nimport sys\n", "import os\nimport sys\nimport time\n"),
    ('    "success": BASE,\n', '    "success": BASE,\n    "launch_late_permitted": BASE,\n    "complete_near_guard": BASE,\n    "supervisor_lost": {**BASE, "G11.storage_and_memory": INC},  # the supervisor\'s record is not complete\n'),
    ('LINE = "release-gpus V4 resource-limit abort (registered supervisor response)"\n', 'LINE = "release-gpus V4 resource-limit abort (registered supervisor response)"\nABORTED = ' + ABORTED + '  # ended by the supervisor\'s registry cleanup\nREHEARSAL_GUARD_S, REGISTERED_GUARD_S, REGISTERED_RUN_BUDGET_S = 420.0, 4680.0, 4210.0\n'),
    ('choices=["success", "ambiguous", "memory_breach", "storage_breach", "reservation_guard"]', 'choices=["success", "ambiguous", "memory_breach", "storage_breach", "reservation_guard", *"' + E2E_NEW + '".split()]'),
    ('    owned, supreg = registry(a.root / "pids" / "owned.tsv"), registry(a.root / "pids" / "supervisor.tsv")\n',
     '''    runlog = (run / "run.log").read_text(errors="replace") if (run / "run.log").exists() else ""
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
        (a.out / "E2E-CHECK.json").write_text(json.dumps({"scenario": a.scenario, "ok": ok, "checks": results, "pid_namespace_note": os.uname().nodename}, indent=1) + "\\n")
        print(f"# scenario {a.scenario}: {sum(r['ok'] for r in results)} of {len(results)} checks pass → {'MET' if ok else 'NOT MET'}")
        return 0 if ok else 1
    owned, supreg = registry(a.root / "pids" / "owned.tsv"), registry(a.root / "pids" / "supervisor.tsv")
'''),
    ('a.scenario in ("storage_breach", "reservation_guard")', 'a.scenario in ABORTED', 2),
    ('    if a.scenario in ("success", "ambiguous"):\n', '    if a.scenario in ("success", "ambiguous", "launch_late_permitted", "complete_near_guard"):\n'),
    ('    if a.scenario == "reservation_guard":\n        c0 = ', '    if a.scenario in ("reservation_guard", "export_near_guard"):\n        c0 = '),
    ('check("breach.marker_and_cause", a.scenario == "reservation_guard" or (run / "STORAGE-BREACH").exists()', 'check("breach.marker_and_cause", a.scenario != "storage_breach" or (run / "STORAGE-BREACH").exists()'),
    ('and lc.get("stop_requested_at") is None and not (run / "profile" / "nsys" / "lifecycle.control").exists(),', 'and (lc.get("stop_requested_at") is None) == (a.scenario != "export_near_guard") and not (run / "profile" / "nsys" / "lifecycle.control").exists(),'),
    ('        runlog = (run / "run.log").read_text(errors="replace") if (run / "run.log").exists() else ""\n        check("driver.did_not_finish"', '        check("driver.did_not_finish"'),
    ('    ok = all(r["ok"] for r in results)\n    a.out.mkdir(parents=True, exist_ok=True)\n',
     '''    causes = [(c.get("cause"), c.get("source"), c.get("kind")) for c in sup.get("causes", [])]
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
'''),
]

def rename(s: str) -> str:
    for a, b in (("v3scale", "v4scale"), ("v3_scale", "v4_scale"), ("V3S", "V4S"), ("V3-SCALE", "V4-SCALE"), ("e1v3-scale", "e1v4-scale")):
        s = s.replace(a, b)
    return re.sub(r"\bV3\b", "V4", s)

def derive(src: Path, dst: Path, edits: list, keep=()):
    orig = src.read_text()
    base = rename(orig)
    for a, b in keep:  # places where the V3 name must stay
        assert a in base, (src.name, a[:60]); base = base.replace(a, b)
    edits = list(edits) + REV2.get(dst.name, [])
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
