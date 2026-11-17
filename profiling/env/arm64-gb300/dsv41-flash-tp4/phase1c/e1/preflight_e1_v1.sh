#!/usr/bin/env bash
# V1 preflight, host side (read-only). Runs the Stage 1.5 preflight (incl. the Stage 1 host preflight) for the V1 pass, then the V1
# checks: pinned inputs, pinned code tree, identity preflight (profiler, importer, server, cleanup primitives, log parsers), the
# lifecycle test suites on this tray, process discovery and the registry plan. ON the capture tray as dc, through a neutral-path
# wrapper (never put experiment words in the ssh command text):
#   E1_REPO=$S/wt-lifecycle bash $E1/preflight_e1_v1.sh
# Prints PASS/FAIL/INFO per check, exits non-zero on any FAIL. Nothing is started; the only files written are the identity-preflight
# record and this log under $E1_V1_OUT (default: <run root>/preflight).
set -u
export HOME=/mnt/shared/home/dc
S=$HOME/workspace/serving-studio
REPO=${E1_REPO:-$S/wt-lifecycle}
E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
RUNV1=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v1
OUT=${E1_V1_OUT:-$RUNV1/preflight}; mkdir -p $OUT
PASS=e1_v1_lifecycle; CFG=$E1/configs_v1/$PASS/profile.json; TRACE=$E1/traces/e1_dry.csv
Q0=$S/artifacts/phase0-dsv41-flash-gb300-tp4-20261112/run/quadrant_nsys/profile.json
AR15=$S/artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115/run/e1_nsys_val
REF_LAUNCH=$AR15/profile/vllm/dsv41_flash_1017_gb300_tp4_e1_nsys_val_launch.json
REF_LOG=$AR15/profile/vllm/dsv41_flash_1017_gb300_tp4_e1_nsys_val_server.log
TS=$(date -u +%Y%m%dT%H%M%SZ)
fail=0
ok()   { echo "PASS  $*"; }
bad()  { echo "FAIL  $*"; fail=1; }
info() { echo "INFO  $*"; }
echo "# V1 preflight on $(hostname) at $(date -u +%FT%TZ); launcher tree $REPO"
# A. Stage 1.5 preflight (Stage 1 host checks, configs_v2 hashes, stream flags, binary provenance, launcher dry-runs)
E1_REPO=$REPO E1_PASS=$PASS bash $E1/preflight_e1_stage15.sh | sed 's/^/  /'
[ "${PIPESTATUS[0]}" = "0" ] && ok "Stage 1.5 preflight (incl. Stage 1 host preflight)" || bad "Stage 1.5 preflight (see above)"
# B. V1 inputs
( cd $E1 && sha256sum -c INPUTS_v1.sha256 --quiet ) && ok "V1 config, its accepted source and the trace match INPUTS_v1.sha256" || bad "INPUTS_v1.sha256 mismatch"
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
python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); e=d['expected']; assert d['trace']['requests']==56 and len(d['trace']['waves'])==2 and e['iterations']==2160 and e['kernel_rows_band'][0]<e['kernel_rows']<e['kernel_rows_band'][1] and d['ceilings']['gpu_held_ceiling_s']==d['lifecycle']['gpu_held_ceiling_s']==2700 and len(d['control_probe']['lines'])==2" $E1/E1-V1-INPUTS.json 2>/dev/null \
  && ok "E1-V1-INPUTS.json: 56 requests, 2 waves, 2160 iterations expected, ceilings consistent, 2 probe lines" || bad "E1-V1-INPUTS.json inconsistent"
( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && unset CUDA_VISIBLE_DEVICES && \
  uv run --no-sync python -m launcher alignment profile --dry-run $CFG ) >/dev/null 2>&1 && ok "launcher dry-run $PASS" || bad "launcher dry-run $PASS"
[ ! -e "$RUNV1/$PASS/profile" ] && ok "no previous output dir for $PASS" || bad "output dir exists: $RUNV1/$PASS/profile"
# C. pinned code tree and scripts
PIN=$E1/V1-PIN.json
for key in alignment launcher tests/test_profiler_lifecycle.py tests/test_profiler_preflight.py tests/test_e1_lifecycle_gates.py tests/fixtures; do
  want=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['code']['tree'][sys.argv[2]])" $PIN $key 2>/dev/null)
  have=$(git -C $REPO rev-parse "HEAD:$key" 2>/dev/null)
  [ -n "$want" ] && [ "$want" = "$have" ] && ok "pinned tree $key = ${have:0:12}" || bad "tree $key ${have:0:12} != pinned ${want:0:12}"
done
[ -z "$(git -C $REPO status --short --untracked-files=no)" ] && ok "launcher tree clean (tracked files)" || bad "launcher tree has uncommitted tracked changes"
info "launcher tree HEAD $(git -C $REPO rev-parse HEAD) ($(git -C $REPO branch --show-current)); pinned code commit $(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['code']['commit'])" $PIN)"
( cd $E1 && sha256sum -c SCRIPTS_v1.sha256 --quiet ) && ok "V1 scripts, checks, tests and fixtures match SCRIPTS_v1.sha256 ($(wc -l < $E1/SCRIPTS_v1.sha256) files)" || bad "SCRIPTS_v1.sha256 mismatch (hash_scripts_v1.sh after a reviewed change)"
# D. identity preflight: profiler, importer, server, recipe, cleanup primitives, log parsers, locations (CPU; nothing started)
( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && unset CUDA_VISIBLE_DEVICES && \
  uv run --no-sync python -m alignment.profiler.preflight --config $CFG --pins $E1/V1-PIN-identities.json --reference-launch $REF_LAUNCH \
    --reference-server-log $REF_LOG --json $OUT/identity_${TS}.json ) 2>&1 | sed 's/^/  /'
[ "${PIPESTATUS[0]}" = "0" ] && ok "identity preflight (alignment.profiler.preflight): all checks PASS → $OUT/identity_${TS}.json" || bad "identity preflight reported FAIL (see above)"
# E. lifecycle, preflight and gate test suites on this tray (CPU)
( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && unset CUDA_VISIBLE_DEVICES && \
  uv run --no-sync python -m pytest -q tests/test_profiler_lifecycle.py tests/test_profiler_preflight.py tests/test_e1_lifecycle_gates.py tests/test_e1_pids_cleanup.py ) > $OUT/pytest_${TS}.log 2>&1 \
  && ok "lifecycle/preflight/gates/cleanup test suites: $(tail -1 $OUT/pytest_${TS}.log)" || bad "test suites: $(tail -1 $OUT/pytest_${TS}.log)"
# F. discovery clean; registry plan clean
bash $E1/checks/e1_process_census.sh | sed 's/^/  /'; [ "${PIPESTATUS[0]}" = "0" ] && ok "process discovery: no candidate alive" || bad "process discovery found candidates (see above; review, never kill by pattern)"
REG=$RUNV1/pids/owned.tsv
python3 -I $E1/checks/e1_pids.py plan --registry $REG | sed 's/^/  /'; prc=${PIPESTATUS[0]}
[ "$prc" = 0 ] && ok "registry plan: no owned process alive, no lookalike, no refusal" || bad "registry plan rc $prc (owned/lookalike/refused; see above)"
# G. storage and ceilings
fs=$(df -BG --output=avail /mnt/shared | tail -1 | tr -dc 0-9); [ "$fs" -gt 1000 ] && ok "CephFS free ${fs} GB (> 1000)" || bad "CephFS free ${fs} GB"
fr=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9); [ "$fr" -gt 100 ] && ok "/tmp (stream directory, tray root NVMe) free ${fr} GB (> 100)" || bad "/tmp free ${fr} GB"
info "ceilings: reservation 90 min hard; GPU-held 2700 s (controller-enforced from the launch); temporary ≤ 5 GB in /tmp; persistent ≤ 5 GB run dir (≤ 10 GB with the archive copy); expected ≈ 2.4 GB raw, finalisation ≈ 90 s"
info "reservation (from the laptop): pod dsv41-gb300-reserve-tray16 Running and the only GPU pod on the node before the go"
echo "# result: $([ $fail = 0 ] && echo PREFLIGHT PASS || echo PREFLIGHT FAIL)"
exit $fail
