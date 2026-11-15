#!/usr/bin/env bash
# b=8 recovery preflight, host side (read-only). Runs the Stage 1.5 preflight (which includes the Stage 1 host preflight) for
# the pass e1_b8_recovery, then the recovery-specific checks. ON the capture tray as dc:
#   ssh pod4-gb300-10-trayNN 'export HOME=/mnt/shared/home/dc; E1_REPO=/mnt/shared/home/dc/workspace/serving-studio/wt-e1stage1 bash $E1_REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/preflight_e1_b8.sh'
# Prints PASS/FAIL/INFO per check, exits non-zero on any FAIL. Nothing is started, written or deleted.
set -u
export HOME=/mnt/shared/home/dc
S=$HOME/workspace/serving-studio
REPO=${E1_REPO:-$S/ServingStudioSim}
E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
RUN15=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15
PASS=e1_b8_recovery; CFG=$E1/configs_v2/$PASS/profile.json; TRACE=$E1/traces/$PASS.csv
Q0=$S/artifacts/phase0-dsv41-flash-gb300-tp4-20261112/run/quadrant_nsys/profile.json
fail=0
ok()   { echo "PASS  $*"; }
bad()  { echo "FAIL  $*"; fail=1; }
info() { echo "INFO  $*"; }
echo "# b=8 recovery preflight on $(hostname) at $(date -u +%FT%TZ)"
# A. Stage 1.5 preflight (Stage 1 host checks, configs_v2 hashes, binary provenance, launcher dry-runs); output dir check for this pass
E1_REPO=$REPO E1_PASS=$PASS bash $E1/preflight_e1_stage15.sh | sed 's/^/  /'
[ "${PIPESTATUS[0]}" = "0" ] && ok "Stage 1.5 preflight (incl. Stage 1 host preflight)" || bad "Stage 1.5 preflight (see above)"
# B. recovery inputs
( cd $E1 && sha256sum -c INPUTS_b8.sha256 --quiet ) && ok "recovery trace/config and their Stage 1.5 sources match INPUTS_b8.sha256" || bad "INPUTS_b8.sha256 mismatch"
cmp -s <(head -9 $E1/traces/e1_dry.csv) $TRACE && ok "trace == header + first 8 rows of traces/e1_dry.csv (byte-identical)" || bad "trace differs from the first 8 rows of e1_dry.csv"
n=$(tail -n +2 $TRACE | grep -c .); [ "$n" = 8 ] && ok "trace has 8 requests" || bad "trace has $n requests"
[ "$(tail -n +2 $TRACE | grep -c '^e1_b8_kv32k_w0_r0[0-7],32768,1024,0.0$')" = 8 ] && ok "all 8 rows are e1_b8_kv32k_w0_r00..07, 32768/1024, arrival 0.0" || bad "unexpected trace rows"
! grep -q 'b48\|e1_main' $TRACE && ok "no b=48 request in the trace" || bad "b=48 ids present in the trace"
grep -q "\"path\": \"$TRACE\"" $CFG && ok "config points at the recovery trace" || bad "config trace path"
grep -q '"stream": false' $CFG && ok "stream=false explicit" || bad "stream key"
grep -q '"capture_routed_experts": true' $CFG && ok "hook on (capture_routed_experts)" || bad "hook flag"
! grep -q capture_duration_seconds $CFG && ok "no bounded stop" || bad "capture_duration_seconds present"
! grep -q capture_start_delay_seconds $CFG && ok "no delayed start: profiler armed before the first request (Phase 0 quadrant_nsys block)" || info "delayed start present: $(grep capture_start_delay_seconds $CFG | tr -d ' ,')"
python3 - "$CFG" "$Q0" <<'PY' && ok "nsys block == Phase 0 quadrant_nsys nsys block" || bad "nsys block differs from Phase 0 quadrant_nsys"
import json,sys
a=json.load(open(sys.argv[1]))["nsys"]; a.pop("capture_start_delay_seconds",None); b=json.load(open(sys.argv[2]))["nsys"]
sys.exit(0 if a==b else 1)
PY
( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && \
  uv run --no-sync python $E1/checks/e1_config_identity.py --a $E1/configs_v2/e1_nsys_val/profile.json --b $CFG \
    --allow name log_dir workload.frontend.path nsys.capture_start_delay_seconds ) >/dev/null 2>&1 \
  && ok "config identity vs e1_nsys_val: only name, log_dir, trace path, capture_start_delay differ" || bad "config identity vs e1_nsys_val (checks/e1_config_identity.py)"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); p=d['dry']['plan']; assert d['dry']['requests']==8 and len(p)==1 and p[0]['b']==8 and p[0]['kv']==32768 and d['run_order']==['e1_b8_recovery']" $E1/E1-B8-RECOVERY-INPUTS.json 2>/dev/null \
  && ok "E1-B8-RECOVERY-INPUTS.json: one plan cell (b=8, kv 32768), 8 requests, run_order [e1_b8_recovery]" || bad "recovery inputs json"
( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && unset CUDA_VISIBLE_DEVICES && \
  uv run --no-sync python -m launcher alignment profile --dry-run $CFG ) >/dev/null 2>&1 && ok "launcher dry-run $PASS" || bad "launcher dry-run $PASS"
# C. scripts pinned; no candidate process alive (discovery incl. watchers and tmux); registry plan clean; GPUs idle
( cd $E1 && sha256sum -c SCRIPTS_b8.sha256 --quiet ) && ok "recovery scripts, checks, analysis and tests match SCRIPTS_b8.sha256 ($(wc -l < $E1/SCRIPTS_b8.sha256) files)" || bad "SCRIPTS_b8.sha256 mismatch (hash_scripts_b8.sh after a reviewed change)"
bash $E1/checks/e1_process_census.sh | sed 's/^/  /'; [ "${PIPESTATUS[0]}" = "0" ] && ok "process discovery: no candidate alive" || bad "process discovery found candidates (see above; review, never kill by pattern)"
REG=$RUN15/pids/owned.tsv
python3 -I $E1/checks/e1_pids.py plan --registry $REG | sed 's/^/  /'; prc=${PIPESTATUS[0]}
[ "$prc" = 0 ] && ok "registry plan: no owned process alive, no lookalike, no refusal" || bad "registry plan rc $prc (owned/lookalike/refused; see above)"
# D. storage ceilings for this pass (packet): CephFS > 500 GB free, tray root > 100 GB free are checked by the Stage 1 preflight
info "expected raw for this pass: rep ~0.2 GB, sqlite ~1.0 GB (ceilings 0.6 / 3 GB), parquet/json ~0.1 GB; CephFS free $(df -BG --output=avail /mnt/shared | tail -1 | tr -dc 0-9) GB"
info "reservation (from the laptop): pod dsv41-gb300-reserve-tray16 must be Running and no other GPU pod on the node before the go"
echo "# result: $([ $fail = 0 ] && echo PREFLIGHT PASS || echo PREFLIGHT FAIL)"
exit $fail
