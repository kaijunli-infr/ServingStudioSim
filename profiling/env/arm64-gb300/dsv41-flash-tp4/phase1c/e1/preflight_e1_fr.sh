#!/usr/bin/env bash
# V2 forced-release preflight, host side (read-only). Runs the Stage 1.5 preflight (incl. the Stage 1 host preflight) for the V2 pass, then the V2
# checks: pinned inputs, pinned code tree, identity preflight (profiler, importer, server, cleanup primitives, log parsers), the
# lifecycle test suites on this tray, process discovery and the registry plan. ON the capture tray as dc, through a neutral-path
# wrapper (never put experiment words in the ssh command text):
#   E1_REPO=$S/wt-lifecycle bash $E1/preflight_e1_fr.sh
# Prints PASS/FAIL/INFO per check, exits non-zero on any FAIL. Nothing is started; the only files written are the identity-preflight
# record and this log under $E1_FR_OUT (default: <run root>/preflight).
set -u
export HOME=/mnt/shared/home/dc
S=$HOME/workspace/serving-studio
REPO=${E1_REPO:-$S/wt-lifecycle}
E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
RUNFR=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v2fr
OUT=${E1_FR_OUT:-$RUNFR/preflight}; mkdir -p $OUT
PASS=e1_v2_forced_release; CFG=$E1/configs_fr/$PASS/profile.json; TRACE=$E1/traces/e1_dry.csv
Q0=$S/artifacts/phase0-dsv41-flash-gb300-tp4-20261112/run/quadrant_nsys/profile.json
AR15=$S/artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115/run/e1_nsys_val
REF_LAUNCH=$AR15/profile/vllm/dsv41_flash_1017_gb300_tp4_e1_nsys_val_launch.json
REF_LOG=$AR15/profile/vllm/dsv41_flash_1017_gb300_tp4_e1_nsys_val_server.log
TS=$(date -u +%Y%m%dT%H%M%SZ)
fail=0
ok()   { echo "PASS  $*"; }
bad()  { echo "FAIL  $*"; fail=1; }
info() { echo "INFO  $*"; }
echo "# V2 forced-release preflight on $(hostname) at $(date -u +%FT%TZ); launcher tree $REPO"
# A. Stage 1.5 preflight (Stage 1 host checks, configs_v2 hashes, stream flags, binary provenance, launcher dry-runs)
E1_REPO=$REPO E1_PASS=$PASS bash $E1/preflight_e1_stage15.sh | sed 's/^/  /'
[ "${PIPESTATUS[0]}" = "0" ] && ok "Stage 1.5 preflight (incl. Stage 1 host preflight)" || bad "Stage 1.5 preflight (see above)"
# B. V2 inputs
( cd $E1 && sha256sum -c INPUTS_fr.sha256 --quiet ) && ok "V2 config, its accepted source and the trace match INPUTS_fr.sha256" || bad "INPUTS_fr.sha256 mismatch"
[ "$(sha256sum $TRACE | cut -c1-64)" = "c6d8e13b7229ffd40cfede127bd57ad345aa17213b7590d39f9b2ef697f60158" ] && ok "trace is the frozen dry-run trace e1_dry.csv (sha c6d8e13b…)" || bad "trace sha differs from e1_dry.csv"
n=$(tail -n +2 $TRACE | grep -c .); [ "$n" = 56 ] && ok "trace has 56 requests" || bad "trace has $n requests"
nw=$(tail -n +2 $TRACE | cut -d, -f1 | sed -E 's/_r[0-9]+$//' | sort -u | wc -l); [ "$nw" = 2 ] && ok "2 waves (b=8 at 0 s, b=48 at 18.6 s)" || bad "waves: $nw"
grep -q "\"path\": \"$TRACE\"" $CFG && ok "config points at the dry-run trace" || bad "config trace path"
grep -q '"stream": false' $CFG && ok "stream=false explicit" || bad "stream key"
grep -q '"capture_routed_experts": true' $CFG && ok "hook on (capture_routed_experts)" || bad "hook flag"
grep -q '"type": "openai"' $CFG && ok "openai backend (routes)" || bad "backend"
! grep -q capture_duration_seconds $CFG && ok "no bounded stop" || bad "capture_duration_seconds present"
! grep -q capture_start_delay_seconds $CFG && ok "no delayed start: profiler armed before the first request (both waves in the capture)" || bad "capture_start_delay_seconds present (would skip the b=8 wave: E1-NSYS-VAL-REPORT.md §6)"
grep -q '"gpu_held_ceiling_s": 2700.0' $CFG && grep -q '"finalize_hard_deadline_s": 600.0' $CFG && grep -q '"finalize_soft_deadline_s": 300.0' $CFG && grep -q '"preserve_raw_stream": true' $CFG && ok "lifecycle bounds explicit: soft 300 s, hard 600 s, GPU-held ceiling 2700 s, stream preserved" || bad "lifecycle bounds in the config differ from the packet"
! grep -q '"tmp_dir"' $CFG && ok "no nsys.tmp_dir: server environment identical to the accepted launch" || bad "tmp_dir set (environment change)"
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
  && ok "config identity vs accepted e1_nsys_val: only name, log_dir, trace path (same bytes), delayed start and the lifecycle bounds differ" || bad "config identity vs e1_nsys_val (checks/e1_config_identity.py)"
python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); e=d['expected']; assert d['trace']['requests']==56 and len(d['trace']['waves'])==2 and e['iterations']==2160 and e['ranks']==4 and e['kernel_rows_band'][0]<e['kernel_rows']<e['kernel_rows_band'][1] and d['ceilings']['gpu_held_ceiling_s']==d['lifecycle']['gpu_held_ceiling_s']==2700 and d['trigger']['control_line'].startswith('release-gpus ') and d['limits']['gpu_release_deadline_s']==90 and set(d['outcomes'])=={'A','B','neither'}" $E1/E1-V2-FR-INPUTS.json 2>/dev/null \
  && ok "E1-V2-FR-INPUTS.json: 56 requests, 2 waves, 2160 iterations, 4 ranks, ceilings consistent, one registered trigger line, release deadline 90 s, outcomes A/B/neither" || bad "E1-V2-FR-INPUTS.json inconsistent"
grep -qF "LINE='$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['trigger']['control_line'])" $E1/E1-V2-FR-INPUTS.json)'" $E1/v2_release_probe.sh && ok "the probe's trigger line is the registered one" || bad "probe trigger line differs from E1-V2-FR-INPUTS.json"
python3 -I -c "import importlib.util,sys; s=importlib.util.spec_from_file_location('i', sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); from pathlib import Path; p=m.load_policy(Path(sys.argv[2])); print(len(p['launch_rules']), len(p['profile_rules']))" $E1/checks/e1_launch_identity_v2.py $E1/checks/e1_identity_policy_v1.json >/dev/null 2>&1 && ok "identity normalization policy loads (no rule on a protected field)" || bad "identity policy does not load"
( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && export CUDA_VISIBLE_DEVICES="" && \
  uv run --no-sync python -m launcher alignment profile --dry-run $CFG ) >/dev/null 2>&1 && ok "launcher dry-run $PASS (CUDA hidden)" || bad "launcher dry-run $PASS"
[ ! -e "$RUNFR/$PASS/profile" ] && ok "no previous output dir for $PASS" || bad "output dir exists: $RUNFR/$PASS/profile"
# C. pinned code tree and scripts
PIN=$E1/V2-FR-PIN.json
for key in alignment launcher tests/fixtures tests/cpu_only_guard.py tests/cpu_only.sh tests/test_profiler_lifecycle.py tests/test_profiler_preflight.py tests/test_e1_lifecycle_gates.py tests/test_e1_pids_cleanup.py tests/test_cpu_only_guard.py tests/test_e1_lifecycle_gates_v2.py tests/test_e1_launch_identity_v2.py tests/test_profiler_lifecycle_stop_concurrency.py tests/test_profiler_lifecycle_host_stream.py tests/test_profiler_preflight_retained.py tests/test_e1_host_identity.py tests/test_e1_forced_release_gates.py; do
  want=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['code']['tree'][sys.argv[2]])" $PIN $key 2>/dev/null)
  have=$(git -C $REPO rev-parse "HEAD:$key" 2>/dev/null)
  [ -n "$want" ] && [ "$want" = "$have" ] && ok "pinned tree $key = ${have:0:12}" || bad "tree $key ${have:0:12} != pinned ${want:0:12}"
done
[ -z "$(git -C $REPO status --short --untracked-files=no)" ] && ok "launcher tree clean (tracked files)" || bad "launcher tree has uncommitted tracked changes"
info "launcher tree HEAD $(git -C $REPO rev-parse HEAD) ($(git -C $REPO branch --show-current)); pinned code commit $(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['code']['commit'])" $PIN)"
( cd $E1 && sha256sum -c SCRIPTS_fr.sha256 --quiet ) && ok "V2 scripts, checks, tests and fixtures match SCRIPTS_fr.sha256 ($(wc -l < $E1/SCRIPTS_fr.sha256) files)" || bad "SCRIPTS_fr.sha256 mismatch (hash_scripts_fr.sh after a reviewed change)"
# D. identity preflight: profiler, importer, server, recipe, cleanup primitives, log parsers, locations (CPU; nothing started)
( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && unset CUDA_VISIBLE_DEVICES && \
  uv run --no-sync python -m alignment.profiler.preflight --config $CFG --pins $E1/V2-FR-PIN-identities.json --reference-launch $REF_LAUNCH \
    --reference-server-log $REF_LOG --json $OUT/identity_${TS}.json ) 2>&1 | sed 's/^/  /'
[ "${PIPESTATUS[0]}" = "0" ] && ok "identity preflight (alignment.profiler.preflight): all checks PASS → $OUT/identity_${TS}.json" || bad "identity preflight reported FAIL (see above)"
# E. controller, preflight, gate, host-identity, cleanup and guard test suites on this tray, CPU-only: devices hidden, guard fails the run (exit 3) if a device node is opened
( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && \
  bash tests/cpu_only.sh -q -p no:cacheprovider tests/test_profiler_lifecycle.py tests/test_profiler_preflight.py tests/test_e1_lifecycle_gates.py tests/test_e1_pids_cleanup.py tests/test_cpu_only_guard.py tests/test_e1_lifecycle_gates_v2.py tests/test_e1_launch_identity_v2.py tests/test_profiler_lifecycle_stop_concurrency.py tests/test_profiler_lifecycle_host_stream.py tests/test_profiler_preflight_retained.py tests/test_e1_host_identity.py tests/test_e1_forced_release_gates.py ) > $OUT/pytest_${TS}.log 2>&1
tr=$?; guard=$(grep -o "cpu_only_guard: [^(]*" $OUT/pytest_${TS}.log | head -1)
[ "$tr" = 0 ] && grep -q "cpu_only_guard: CUDA hidden" $OUT/pytest_${TS}.log && ok "controller/preflight/gates/host/cleanup/guard test suites: $(tail -1 $OUT/pytest_${TS}.log); $guard" || bad "test suites rc $tr: $(tail -1 $OUT/pytest_${TS}.log); ${guard:-guard line missing}"
# F. discovery clean; registry plan clean
bash $E1/checks/e1_process_census.sh | sed 's/^/  /'; [ "${PIPESTATUS[0]}" = "0" ] && ok "process discovery: no candidate alive" || bad "process discovery found candidates (see above; review, never kill by pattern)"
REG=$RUNFR/pids/owned.tsv
python3 -I $E1/checks/e1_pids.py plan --registry $REG | sed 's/^/  /'; prc=${PIPESTATUS[0]}
[ "$prc" = 0 ] && ok "registry plan: no owned process alive, no lookalike, no refusal" || bad "registry plan rc $prc (owned/lookalike/refused; see above)"
# F2. GPU state primitive for gate L6c (nvidia-smi compute-process census + memory state; the controller's timestamps are not accepted alone)
python3 -I $E1/checks/e1_gpu_release_census.py --label preflight --out $OUT/gpu_census_${TS}.json | sed 's/^/  /'; grc=${PIPESTATUS[0]}
[ "$grc" = 0 ] && ok "GPU state: no compute process, memory.used within the idle bound on every GPU (census primitive works)" || bad "GPU state census rc $grc (compute process present or GPU memory held; see above)"
# G. storage and ceilings
fs=$(df -BG --output=avail /mnt/shared | tail -1 | tr -dc 0-9); [ "$fs" -gt 1000 ] && ok "CephFS free ${fs} GB (> 1000)" || bad "CephFS free ${fs} GB"
fr=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9); [ "$fr" -gt 100 ] && ok "/tmp (stream directory, tray root NVMe) free ${fr} GB (> 100)" || bad "/tmp free ${fr} GB"
info "ceilings: reservation 90 min hard; GPU-held 2700 s (controller-enforced from the launch); GPUs released ≤ 90 s after the forced-release request; temporary ≤ 5 GB in /tmp; persistent ≤ 5 GB run dir (≤ 10 GB with the archive copy); expected ≈ 2.4 GB raw, finalisation ≈ 90 s"
[ -f $RUNFR/$PASS/reservation.json ] && { python3 -I -c "import json,sys,socket; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get('node')==socket.gethostname() and r.get('phase')=='Running' else 1)" $RUNFR/$PASS/reservation.json && ok "reservation record: pod Running on this node ($(hostname))" || bad "reservation record is not for this node or not Running"; } || info "reservation record $RUNFR/$PASS/reservation.json not present yet (written from kubectl before the launch; the driver refuses to start without it)"
echo "# result: $([ $fail = 0 ] && echo PREFLIGHT PASS || echo PREFLIGHT FAIL)"
exit $fail
