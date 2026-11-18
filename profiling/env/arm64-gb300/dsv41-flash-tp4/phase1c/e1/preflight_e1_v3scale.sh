#!/usr/bin/env bash
# V3 preflight, host side (read-only). Runs the Stage 1.5 preflight (incl. the Stage 1 host preflight) for the V3 pass, then the V3
# checks: freeze manifest, pinned inputs, pinned code tree and scripts, prebuilt executable contract, identity preflight, the V3 tool
# and gate test suites on this tray, observers present, process discovery, registry plan, GPU state, storage and host memory. ON the
# capture tray as dc, through a neutral-path wrapper (never put experiment words in the ssh command text):
#   E1_REPO=$S/wt-lifecycle bash $E1/preflight_e1_v3scale.sh
# Prints PASS/FAIL/INFO per check, exits non-zero on any FAIL. Nothing is started; the only files written are the identity-preflight
# record, the test log and the census record under $E1_V3S_OUT (default: <run root>/preflight).
set -u
export HOME=/mnt/shared/home/dc
S=$HOME/workspace/serving-studio
REPO=${E1_REPO:-$S/wt-lifecycle}
E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
RUNV3=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v3scale
OUT=${E1_V3S_OUT:-$RUNV3/preflight}; mkdir -p $OUT
PASS=e1_v3_scale; CFG=$E1/configs_v3scale/$PASS/profile.json; TRACE=$E1/traces/e1_v3_scale.csv; INP=$E1/E1-V3-SCALE-INPUTS.json
Q0=$S/artifacts/phase0-dsv41-flash-gb300-tp4-20261112/run/quadrant_nsys/profile.json
AR15=$S/artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115/run/e1_nsys_val
REF_LAUNCH=$AR15/profile/vllm/dsv41_flash_1017_gb300_tp4_e1_nsys_val_launch.json
REF_LOG=$AR15/profile/vllm/dsv41_flash_1017_gb300_tp4_e1_nsys_val_server.log
TESTS="tests/test_profiler_lifecycle.py tests/test_profiler_preflight.py tests/test_e1_lifecycle_gates.py tests/test_e1_pids_cleanup.py tests/test_cpu_only_guard.py tests/test_e1_lifecycle_gates_v2.py tests/test_e1_launch_identity_v2.py tests/test_profiler_lifecycle_stop_concurrency.py tests/test_profiler_lifecycle_host_stream.py tests/test_profiler_preflight_retained.py tests/test_e1_host_identity.py tests/test_e1_forced_release_gates.py tests/test_load_generator_prebuilt.py tests/test_profiler_lifecycle_release_evidence.py tests/test_e1_frontend_binary_gate.py tests/test_e1_import_probe.py tests/test_e1_mem_watch.py tests/test_e1_storage_watch.py tests/test_e1_v3scale_gates.py tests/test_e1_supervisor.py tests/test_profiler_lifecycle_v3_deadlines.py"
TS=$(date -u +%Y%m%dT%H%M%SZ)
fail=0
ok()   { echo "PASS  $*"; }
bad()  { echo "FAIL  $*"; fail=1; }
info() { echo "INFO  $*"; }
echo "# V3 preflight on $(hostname) at $(date -u +%FT%TZ); launcher tree $REPO"
# A. Stage 1.5 preflight (Stage 1 host checks, configs_v2 hashes, stream flags, binary provenance, launcher dry-runs)
E1_REPO=$REPO E1_PASS=$PASS bash $E1/preflight_e1_stage15.sh | sed 's/^/  /'
[ "${PIPESTATUS[0]}" = "0" ] && ok "Stage 1.5 preflight (incl. Stage 1 host preflight)" || bad "Stage 1.5 preflight (see above)"
# A2. freeze manifest: packet, manifests, inputs, pins, binary provenance
( cd $E1 && [ -f FREEZE_v3scale.sha256 ] && sha256sum -c FREEZE_v3scale.sha256 --quiet ) && ok "FREEZE_v3scale.sha256: all $(wc -l < $E1/FREEZE_v3scale.sha256) entries verify (manifest sha256 $(sha256sum $E1/FREEZE_v3scale.sha256 | cut -c1-16)…)" || bad "FREEZE_v3scale.sha256 missing or mismatch"
# B. V3 inputs
( cd $E1 && sha256sum -c INPUTS_v3scale.sha256 --quiet ) && ok "V3 trace and config, the V2 config and the accepted trace they derive from match INPUTS_v3scale.sha256" || bad "INPUTS_v3scale.sha256 mismatch"
[ "$(sha256sum $E1/traces/e1_dry.csv | cut -c1-64)" = "c6d8e13b7229ffd40cfede127bd57ad345aa17213b7590d39f9b2ef697f60158" ] && ok "source trace is the frozen dry-run trace e1_dry.csv (sha c6d8e13b…)" || bad "e1_dry.csv sha differs"
python3 -I - "$TRACE" "$E1/traces/e1_dry.csv" <<'PY' && ok "trace = the 56 accepted rows twice: 112 requests, 4 waves (b=8 at 0 s, b=48 at 18.627 s, b=8 at 80 s, b=48 at 98.627 s), second pair = first pair with wave index 1 and +80 s" || bad "trace is not the accepted 56 rows twice"
import sys
v3 = [r.split(",") for r in open(sys.argv[1]).read().split()[1:]]; dry = [r.split(",") for r in open(sys.argv[2]).read().split()[1:]]
want = dry + [[r[0].replace("_w0_r", "_w1_r"), r[1], r[2], str(round(float(r[3]) + 80000.0, 1))] for r in dry]
waves = sorted({(r[0].rsplit("_r", 1)[0], float(r[3])) for r in v3}, key=lambda x: x[1])
sys.exit(0 if v3 == want and len(v3) == 112 and waves == [("e1_b8_kv32k_w0", 0.0), ("e1_b48_kv32k_w0", 18627.2), ("e1_b8_kv32k_w1", 80000.0), ("e1_b48_kv32k_w1", 98627.2)] else 1)
PY
python3 -I - "$CFG" "$E1/configs_fr/e1_v2_forced_release/profile.json" "$TRACE" <<'PY' && ok "config = the V2 config with only name, log_dir and the trace path changed (backend openai, hook on, stream false, model, server arguments, Nsight flags and lifecycle bounds identical)" || bad "config differs from the V2 config in more than name, log_dir and trace path"
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ok = a["workload"]["frontend"]["path"] == sys.argv[3] and a["name"].endswith("e1_v3_scale") and "e1_v3scale/e1_v3_scale/profile" in a["log_dir"]
for d in (a, b):
    d.pop("name"); d.pop("log_dir"); d["workload"]["frontend"].pop("path")
sys.exit(0 if ok and a == b else 1)
PY
grep -q '"stream": false' $CFG && grep -q '"capture_routed_experts": true' $CFG && grep -q '"type": "openai"' $CFG && ok "stream=false, routing hook on, openai backend explicit" || bad "workload keys"
! grep -q capture_duration_seconds $CFG && ! grep -q capture_start_delay_seconds $CFG && ok "no bounded stop, no delayed start (all four waves in the capture)" || bad "capture_duration_seconds or capture_start_delay_seconds present"
grep -q '"gpu_held_ceiling_s": 2700.0' $CFG && grep -q '"finalize_hard_deadline_s": 600.0' $CFG && grep -q '"finalize_soft_deadline_s": 300.0' $CFG && grep -q '"finalize_session_close_timeout_s": 300.0' $CFG && grep -q '"preserve_raw_stream": true' $CFG && ok "lifecycle bounds explicit: soft 300 s, hard 600 s, session close 300 s, GPU-held ceiling 2700 s, stream preserved" || bad "lifecycle bounds in the config differ from the packet"
! grep -q '"tmp_dir"' $CFG && ! grep -q 'server_term_grace' $CFG && ok "no nsys.tmp_dir; server grace not overridden (controller default 60 s)" || bad "tmp_dir or server grace set"
python3 -I - "$CFG" "$Q0" <<'PY' && ok "nsys capture flags == Phase 0 quadrant_nsys nsys block (lifecycle keys excepted)" || bad "nsys capture flags differ from Phase 0 quadrant_nsys"
import json,sys
a=json.load(open(sys.argv[1]))["nsys"]; b=json.load(open(sys.argv[2]))["nsys"]; b.pop("capture_start_delay_seconds",None)
for k in list(a):
    if k.startswith("finalize_") or k in ("gpu_held_ceiling_s","preserve_raw_stream","raw_preserve_dir","min_free_bytes_stream","min_free_bytes_report","tmp_dir","qdstrm_importer"): a.pop(k)
sys.exit(0 if a==b else 1)
PY
( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && \
  uv run --no-sync python $E1/checks/e1_config_identity.py --a $E1/configs_v2/e1_nsys_val/profile.json --b $CFG \
    --allow name log_dir workload.frontend.path nsys.capture_start_delay_seconds nsys.finalize_soft_deadline_s nsys.finalize_hard_deadline_s nsys.finalize_stall_timeout_s \
            nsys.finalize_on_hard_deadline nsys.finalize_session_close_timeout_s nsys.finalize_max_extension_s nsys.gpu_held_ceiling_s nsys.preserve_raw_stream ) >/dev/null 2>&1 \
  && ok "config identity vs accepted e1_nsys_val: only name, log_dir, trace path, delayed start and the lifecycle bounds differ" || bad "config identity vs e1_nsys_val (checks/e1_config_identity.py)"
python3 -I -c "
import json,sys
d=json.load(open(sys.argv[1])); e=d['expected']; c=d['ceilings']; p=d['phase_deadlines_s']; l=d['limits']
assert d['trace']['requests']==112 and len(d['trace']['waves'])==4 and e['iterations']==4320 and e['iterations_band']==[4200,4500] and e['nvtx_forward_ranges']==17280 and e['ranks']==4
assert e['kernel_rows']==24501664 and e['kernel_rows_band'][0]<e['kernel_rows']<e['kernel_rows_band'][1]
assert c['gpu_held_ceiling_s']==d['lifecycle']['gpu_held_ceiling_s']==2700 and c['reservation_hard_min']==90 and c['temporary_storage_gb']==10 and c['persistent_storage_gb']==8 and c['persistent_storage_with_archive_gb']==16 and c['host_mem_available_floor_gb']==64 and c['importer_rss_ceiling_gb']==128 and c['host_mem_available_preflight_min_gb']==448
assert d['lifecycle']['finalize_soft_deadline_s']==300 and d['lifecycle']['finalize_hard_deadline_s']==p['stop_to_report_written']==600 and d['lifecycle']['finalize_max_extension_s']==120
assert d['supervisor']['control_line'].startswith('release-gpus ') and l['supervisor_ack_s']==10 and l['supervisor_stale_s']==60 and l['supervisor_cleanup_grace_s']==30
assert p=={'stop_accepted':30,'flush_stop_to_importing':300,'import_importer_first_seen_to_gone':300,'stop_to_report_written':600,'session_close_after_server_sigterm':300} and l['gpu_release_deadline_s']==90
assert [(x['b'],x['wave']) for x in d['dry']['plan']]==[(8,0),(48,0),(8,1),(48,1)] and d['dry']['requests']==112 and d['config_sha256']==sys.argv[2]" $INP "$(sha256sum $CFG | cut -d' ' -f1)" 2>/dev/null \
  && ok "E1-V3-SCALE-INPUTS.json: 112 requests, 4 waves, 4320 iterations (4200–4500), 17 280 NVTX ranges, 24 501 664 rows ± 5 %, phase deadlines 30/300/300/600/300 s, release 90 s, controller hard deadline = the 600 s total, ceilings 2700 s / 90 min / 10 / 8 / 16 GB / 64 GiB floor / 128 GiB importer, supervisor limits 10 / 60 / 30 s, four-cell plan" || bad "E1-V3-SCALE-INPUTS.json inconsistent with the packet or the config"
grep -q -- "--control" $E1/run_e1_v3scale.sh && bad "the driver passes a control file to an observer" || ok "the driver starts the import probe in observer mode (no control file, no trigger line)"
[ "$(grep -c "^spawn [a-z]*:\$PASS " $E1/run_e1_v3scale.sh)" = 4 ] && ok "the driver registers four observers: GPU sampler, import probe, host-memory sampler, storage sampler" || bad "observer registration lines in the driver != 4"
[ "$(grep -c -- "--label supervisor:\$PASS -- python3 -I \$E1/checks/e1_supervisor.py" $E1/run_e1_v3scale.sh)" = 1 ] && grep -q -- '--registry \$SREG --label supervisor' $E1/run_e1_v3scale.sh && grep -q -- '--max-importer-rss-gb \$RSS_CEIL' $E1/run_e1_v3scale.sh && grep -q '^NAME=.*RSS_CEIL=128; TMP_CEIL=10; PERS_CEIL=8$' $E1/run_e1_v3scale.sh \
  && ok "the driver registers the run supervisor in its own registry and passes the registered ceilings (importer 128 GiB, temporary 10 GB, persistent 8 GB) to the samplers" || bad "supervisor registration or sampler ceilings in the driver"
python3 -I -c "import ast,sys; src=open(sys.argv[1]).read(); code=src.split('\"\"\"',2)[2]; sys.exit(0 if not any(x in code for x in ('os.kill','import signal','killpg','send_signal','.terminate(','.kill(')) else 1)" $E1/checks/e1_supervisor.py && ok "the supervisor contains no signalling call; its only child is the pinned registry cleanup" || bad "the supervisor source contains a signalling call"
[ -z "${E1_V3S_MODE:-}" ] && ok "E1_V3S_MODE is not set: a launch from this context is a real launch" || info "E1_V3S_MODE=${E1_V3S_MODE}: rehearsal context (mock launcher; gate G0 fails on a rehearsal launch)"
# B2. prebuilt load-generator contract (fail-closed; this is the check the driver's launcher repeats before any engine start)
( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && export CUDA_VISIBLE_DEVICES="" && \
  uv run --no-sync python -m alignment.load_generator.runner verify-prebuilt --provenance $E1/BINARY-PROVENANCE.json ) 2>&1 | sed 's/^/  /'
[ "${PIPESTATUS[0]}" = "0" ] && ok "prebuilt session_runner contract: exact pinned sha256, pinned clean source, build-state record; cargo not consulted" || bad "prebuilt session_runner contract refused (see above): an authorized launch would refuse to start"
[ "$(grep -c "python -m launcher alignment profile" $E1/run_e1_v3scale.sh)" = 1 ] && grep -q -- '--frontend-provenance $E1/BINARY-PROVENANCE.json' $E1/run_e1_v3scale.sh && ok "the driver has exactly one launcher line and it passes --frontend-provenance (prebuilt path, build-tool guard armed)" || bad "driver launcher line"
python3 -I -c "import importlib.util,sys; s=importlib.util.spec_from_file_location('i', sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); from pathlib import Path; p=m.load_policy(Path(sys.argv[2])); print(len(p['launch_rules']), len(p['profile_rules']))" $E1/checks/e1_launch_identity_v2.py $E1/checks/e1_identity_policy_v1.json >/dev/null 2>&1 && ok "identity normalization policy loads (no rule on a protected field)" || bad "identity policy does not load"
( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && export CUDA_VISIBLE_DEVICES="" && \
  uv run --no-sync python -m launcher alignment profile --dry-run $CFG --frontend-provenance $E1/BINARY-PROVENANCE.json ) 2>&1 | grep -q "prebuilt session_runner verified" && ok "launcher dry-run $PASS with the prebuilt contract (CUDA hidden; nothing compiled)" || bad "launcher dry-run $PASS with --frontend-provenance"
[ ! -e "$RUNV3/$PASS/profile" ] && ok "no previous output dir for $PASS" || bad "output dir exists: $RUNV3/$PASS/profile"
# C. pinned code tree and scripts
PIN=$E1/V3S-PIN.json
for key in alignment launcher tests/fixtures tests/cpu_only_guard.py tests/cpu_only.sh $TESTS; do
  want=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['code']['tree'][sys.argv[2]])" $PIN $key 2>/dev/null)
  have=$(git -C $REPO rev-parse "HEAD:$key" 2>/dev/null)
  [ -n "$want" ] && [ "$want" = "$have" ] && ok "pinned tree $key = ${have:0:12}" || bad "tree $key ${have:0:12} != pinned ${want:0:12}"
done
for p in alignment/load_generator/req-frontend alignment/profiler/vllm; do
  want=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['code']['gitlinks'][sys.argv[2]])" $PIN $p 2>/dev/null); have=$(git -C $REPO ls-tree HEAD $p | awk '{print $3}'); co=$(git -C $REPO/$p rev-parse HEAD 2>/dev/null)
  [ -n "$want" ] && [ "$want" = "$have" ] && [ "$have" = "$co" ] && ok "submodule $p: gitlink = checkout = pinned ${have:0:12}" || bad "submodule $p: pinned ${want:0:12}, gitlink ${have:0:12}, checkout ${co:0:12}"
done
git -C $REPO merge-base --is-ancestor "$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['code']['commit'])" $PIN)" HEAD && ok "HEAD descends from the pinned code commit" || bad "HEAD does not descend from the pinned code commit"
[ -z "$(git -C $REPO status --short --untracked-files=no)" ] && ok "launcher tree clean (tracked files)" || bad "launcher tree has uncommitted tracked changes"
info "launcher tree HEAD $(git -C $REPO rev-parse HEAD) ($(git -C $REPO branch --show-current)); pinned code commit $(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['code']['commit'])" $PIN)"
( cd $E1 && sha256sum -c SCRIPTS_v3scale.sha256 --quiet ) && ok "V3 scripts, checks, tests and fixtures match SCRIPTS_v3scale.sha256 ($(wc -l < $E1/SCRIPTS_v3scale.sha256) files)" || bad "SCRIPTS_v3scale.sha256 mismatch (hash_scripts_v3scale.sh after a reviewed change)"
# D. identity preflight: profiler, importer, server, recipe, cleanup primitives, log parsers, locations (CPU; nothing started)
( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && unset CUDA_VISIBLE_DEVICES && \
  uv run --no-sync python -m alignment.profiler.preflight --config $CFG --pins $E1/V3S-PIN-identities.json --reference-launch $REF_LAUNCH \
    --reference-server-log $REF_LOG --json $OUT/identity_${TS}.json ) 2>&1 | sed 's/^/  /'
[ "${PIPESTATUS[0]}" = "0" ] && ok "identity preflight (alignment.profiler.preflight): all checks PASS → $OUT/identity_${TS}.json" || bad "identity preflight reported FAIL (see above)"
IMPORTER=/opt/nvidia/nsight-systems/2025.6.3/host-linux-armv8/QdstrmImporter
python3 -I -c "import json,sys,hashlib; d=json.load(open(sys.argv[1]))['importer']; sys.exit(0 if d['path']==sys.argv[2] and d['sha256']==hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest() else 1)" $E1/V3S-PIN-identities.json $IMPORTER 2>/dev/null && ok "the importer the observers are given is the pinned QdstrmImporter (path and sha256)" || bad "importer path / sha256 does not match V3S-PIN-identities.json"
# E. test suites on this tray, CPU-only: devices hidden, guard fails the run (exit 3) if a device node is opened
( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && \
  bash tests/cpu_only.sh -q -p no:cacheprovider $TESTS ) > $OUT/pytest_${TS}.log 2>&1
tr=$?; guard=$(grep -o "cpu_only_guard: [^(]*" $OUT/pytest_${TS}.log | head -1)
[ "$tr" = 0 ] && grep -q "cpu_only_guard: CUDA hidden" $OUT/pytest_${TS}.log && ok "controller / preflight / gates / observers / V3 gate test suites: $(tail -1 $OUT/pytest_${TS}.log); $guard" || bad "test suites rc $tr: $(tail -1 $OUT/pytest_${TS}.log); ${guard:-guard line missing}"
# F. discovery clean; registry plan clean; no Nsight process of this user
bash $E1/checks/e1_process_census.sh | sed 's/^/  /'; [ "${PIPESTATUS[0]}" = "0" ] && ok "process discovery: no candidate alive" || bad "process discovery found candidates (see above; review, never kill by pattern)"
REG=$RUNV3/pids/owned.tsv
python3 -I $E1/checks/e1_pids.py plan --registry $REG | sed 's/^/  /'; prc=${PIPESTATUS[0]}
[ "$prc" = 0 ] && ok "registry plan: no owned process alive, no lookalike, no refusal" || bad "registry plan rc $prc (owned/lookalike/refused; see above)"
python3 -I $E1/checks/e1_pids.py plan --registry $RUNV3/pids/supervisor.tsv | sed 's/^/  /'; prc=${PIPESTATUS[0]}
[ "$prc" = 0 ] && ok "supervisor registry plan: no supervisor alive, no refusal" || bad "supervisor registry plan rc $prc (see above)"
ns=$(for d in /proc/[0-9]*; do [ "$(stat -c %u $d 2>/dev/null)" = "$(id -u)" ] || continue; e=$(readlink $d/exe 2>/dev/null) || continue; case "$e" in /opt/nvidia/nsight-systems/*) echo "${d#/proc/}:$(basename $e)";; esac; done | tr '\n' ' ')
[ -z "${ns// /}" ] && ok "no Nsight process (CLI, agent, importer) of this user is running" || bad "Nsight processes of this user alive: $ns (by executable path; never killed by pattern)"
# F2. GPU state primitive (nvidia-smi compute-process census + memory state)
python3 -I $E1/checks/e1_gpu_release_census.py --label preflight --out $OUT/gpu_census_${TS}.json | sed 's/^/  /'; grc=${PIPESTATUS[0]}
[ "$grc" = 0 ] && ok "GPU state: no compute process, memory.used within the idle bound on every GPU" || bad "GPU state census rc $grc (compute process present or GPU memory held; see above)"
# G. storage, host memory and ceilings
fs=$(df -BG --output=avail /mnt/shared | tail -1 | tr -dc 0-9); [ "$fs" -gt 1000 ] && ok "CephFS free ${fs} GB (> 1000)" || bad "CephFS free ${fs} GB"
fr=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9); [ "$fr" -gt 100 ] && ok "/tmp (stream directory, tray root NVMe) free ${fr} GB (> 100)" || bad "/tmp free ${fr} GB"
mt=$(awk '/^MemTotal:/{printf "%d", $2/1048576}' /proc/meminfo); ma=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo); sw=$(awk '/^SwapTotal:/{printf "%d", $2/1048576}' /proc/meminfo)
cg=$(awk -F: '{print $3}' /proc/self/cgroup | head -1); cgmax=max; p=$cg; while [ -n "$p" ] && [ "$p" != "/" ]; do v=$(cat /sys/fs/cgroup$p/memory.max 2>/dev/null); [ -n "$v" ] && [ "$v" != max ] && { [ "$cgmax" = max ] || [ "$v" -lt "$cgmax" ]; } && cgmax=$v; p=$(dirname $p); done
info "host memory: MemTotal ${mt} GiB, MemAvailable ${ma} GiB, swap ${sw} GiB, cgroup memory.max on this session's path: ${cgmax}"
[ "${ma:-0}" -ge 448 ] && ok "host MemAvailable ${ma} GiB >= 448 = importer ceiling 128 + floor 64 + 256 for the server; margin above that ${ma:+$((ma - 448))} GiB" || bad "host MemAvailable ${ma} GiB < 448 (importer ceiling 128 + floor 64 + 256 for the server)"
{ [ "$cgmax" = max ] || [ "$((cgmax / 1073741824))" -ge 448 ]; } && ok "no cgroup memory limit below 448 GiB applies to this session (memory.max ${cgmax})" || bad "a cgroup memory limit of $((cgmax / 1073741824)) GiB applies: the importer ceiling of 128 GiB is not consistent with it"
tu=$(python3 -I $E1/checks/e1_storage_watch.py --out /dev/null --summary /dev/stdout --group "temporary=/tmp/nsys-report-*.qdstrm,/tmp/*.raw-preserve,/tmp/v3scale-recovery-*" --max-s 0 2>/dev/null | python3 -I -c "import json,sys; print(round(json.load(sys.stdin)['groups']['temporary']['peak_bytes']/1e9,3))" 2>/dev/null)
python3 -I -c "import sys; sys.exit(0 if float(sys.argv[1]) + 0.71 + 4.7 <= 10 else 1)" "${tu:-99}" && ok "temporary storage now ${tu} GB (retained streams); with the expected stream 0.70 GB and post-run scratch 4.7 GB it stays within 10 GB" || bad "temporary storage now ${tu:-?} GB: expected use would exceed the 10 GB ceiling"
info "ceilings: reservation 90 min hard; GPU-held 2700 s (controller); GPU release upper edge <= 90 s after the server SIGTERM; stop -> report 600 s hard; temporary <= 10 GB; run root <= 8 GB (<= 16 GB with the archive); MemAvailable floor 64 GiB; importer <= 128 GiB"
[ -f $RUNV3/$PASS/reservation.json ] && { python3 -I -c "import json,sys,socket; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get('node')==socket.gethostname() and r.get('phase')=='Running' and r.get('creation_timestamp') else 1)" $RUNV3/$PASS/reservation.json && ok "reservation record: pod Running on this node ($(hostname)), creation time recorded" || bad "reservation record is not for this node, not Running, or has no creation time"; } || info "reservation record $RUNV3/$PASS/reservation.json not present yet (written from kubectl before the launch; the driver refuses to start without it)"
echo "# result: $([ $fail = 0 ] && echo PREFLIGHT PASS || echo PREFLIGHT FAIL)"
exit $fail
