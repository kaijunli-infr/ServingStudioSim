#!/usr/bin/env bash
# Full-E1 v3 preflight, host side (read-only). Runs the Stage 1.5 preflight (incl. the Stage 1 host preflight), then the v3 checks.
# ON the capture tray as dc (through a neutral-path wrapper, see E1-AUTHORIZATION-PACKET-v2.md §7):
#   E1_REPO=$WT bash $E1/preflight_e1_main_v3.sh
# Prints PASS/FAIL/INFO per check, exits non-zero on any FAIL. Nothing is started, written or deleted.
set -u
export HOME=/mnt/shared/home/dc
S=$HOME/workspace/serving-studio
REPO=${E1_REPO:-$S/ServingStudioSim}
E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
RUNM=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_main
PASS=e1_main_v3; CFG=$E1/configs_v3/$PASS/profile.json; TRACE=$E1/traces/e1_main.csv
Q0=$S/artifacts/phase0-dsv41-flash-gb300-tp4-20261112/run/quadrant_nsys/profile.json
fail=0
ok()   { echo "PASS  $*"; }
bad()  { echo "FAIL  $*"; fail=1; }
info() { echo "INFO  $*"; }
echo "# full-E1 v3 preflight on $(hostname) at $(date -u +%FT%TZ)"
# A. Stage 1.5 preflight (Stage 1 host checks, configs_v2 hashes, stream flags, binary provenance, launcher dry-runs); E1_PASS names a
#    pass with no Stage 1.5 output dir, so its output-dir rule is satisfied; the v3 output dir is checked below
E1_REPO=$REPO E1_PASS=$PASS bash $E1/preflight_e1_stage15.sh | sed 's/^/  /'
[ "${PIPESTATUS[0]}" = "0" ] && ok "Stage 1.5 preflight (incl. Stage 1 host preflight)" || bad "Stage 1.5 preflight (see above)"
# B. v3 inputs
( cd $E1 && sha256sum -c INPUTS_v3.sha256 --quiet ) && ok "e1_main trace, v3 config and their accepted sources match INPUTS_v3.sha256" || bad "INPUTS_v3.sha256 mismatch"
[ "$(sha256sum $TRACE | cut -c1-64)" = "2e147402b5a24f9e2078bf12e3623691c2fd6d5c8378bab09d7e86ee23bcc2ef" ] && ok "trace is the frozen E1 trace (sha 2e147402…)" || bad "trace sha differs from the frozen E1 trace"
n=$(tail -n +2 $TRACE | grep -c .); [ "$n" = 450 ] && ok "trace has 450 requests" || bad "trace has $n requests"
nw=$(tail -n +2 $TRACE | cut -d, -f1 | sed -E 's/_r[0-9]+$//' | sort -u | wc -l); [ "$nw" = 36 ] && ok "36 waves (distinct e1_b*_kv*_w* prefixes)" || bad "waves: $nw"
[ "$(tail -n +2 $TRACE | grep -cE '^e1_b(1|2|4|8|16|24|32|48)_kv(4k|32k|96k)_w[01]_r[0-9]{2},(4096|32768|98304),1024,[0-9.]+$')" = 450 ] && ok "every row: E1 id, KV in {4096,32768,98304}, 1024 output tokens" || bad "unexpected trace rows"
grep -q "\"path\": \"$TRACE\"" $CFG && ok "config points at the E1 trace" || bad "config trace path"
grep -q '"stream": false' $CFG && ok "stream=false explicit" || bad "stream key"
grep -q '"capture_routed_experts": true' $CFG && ok "hook on (capture_routed_experts)" || bad "hook flag"
grep -q '"type": "openai"' $CFG && ok "openai backend (routes)" || bad "backend"
! grep -q capture_duration_seconds $CFG && ok "no bounded stop" || bad "capture_duration_seconds present"
! grep -q capture_start_delay_seconds $CFG && ok "no delayed start: profiler armed before the first request" || bad "capture_start_delay_seconds present (would miss the first wave: audit/schedule/E1-SCHEDULE-SIM.md)"
python3 - "$CFG" "$Q0" <<'PY' && ok "nsys block == Phase 0 quadrant_nsys nsys block" || bad "nsys block differs from Phase 0 quadrant_nsys"
import json,sys
a=json.load(open(sys.argv[1]))["nsys"]; b=json.load(open(sys.argv[2]))["nsys"]; b.pop("capture_start_delay_seconds",None)
sys.exit(0 if a==b else 1)
PY
( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && \
  uv run --no-sync python $E1/checks/e1_config_identity.py --a $E1/configs_v2/e1_nsys_val/profile.json --b $CFG \
    --allow name log_dir workload.frontend.path nsys.capture_start_delay_seconds ) >/dev/null 2>&1 \
  && ok "config identity vs accepted e1_nsys_val: only name, log_dir, trace path, capture_start_delay differ" || bad "config identity vs e1_nsys_val (checks/e1_config_identity.py)"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['trace']['requests']==450 and d['trace']['waves']==36 and len(d['plan'])==36 and d['pure_decode_iters_min']>=35000 and min(w['pure_decode_iters_min'] for w in d['plan'])>=400" $E1/E1-MAIN-V3-INPUTS.json 2>/dev/null \
  && ok "E1-MAIN-V3-INPUTS.json: 450 requests, 36 waves, every wave >= 400 pure-decode iterations by plan" || bad "v3 inputs json"
( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && unset CUDA_VISIBLE_DEVICES && \
  uv run --no-sync python -m launcher alignment profile --dry-run $CFG ) >/dev/null 2>&1 && ok "launcher dry-run $PASS" || bad "launcher dry-run $PASS"
[ ! -e "$RUNM/$PASS/profile" ] && ok "no previous output dir for $PASS" || bad "output dir exists: $RUNM/$PASS/profile"
# C. scripts pinned; discovery clean; registry plan clean
( cd $E1 && sha256sum -c SCRIPTS_v3.sha256 --quiet ) && ok "v3 scripts, checks, analysis and tests match SCRIPTS_v3.sha256 ($(wc -l < $E1/SCRIPTS_v3.sha256) files)" || bad "SCRIPTS_v3.sha256 mismatch (hash_scripts_v3.sh after a reviewed change)"
bash $E1/checks/e1_process_census.sh | sed 's/^/  /'; [ "${PIPESTATUS[0]}" = "0" ] && ok "process discovery: no candidate alive" || bad "process discovery found candidates (see above; review, never kill by pattern)"
REG=$RUNM/pids/owned.tsv
python3 -I $E1/checks/e1_pids.py plan --registry $REG | sed 's/^/  /'; prc=${PIPESTATUS[0]}
[ "$prc" = 0 ] && ok "registry plan: no owned process alive, no lookalike, no refusal" || bad "registry plan rc $prc (owned/lookalike/refused; see above)"
# D. storage for this pass: expected ~43 GB raw (rep 6.2, sqlite 32, derived 2.6, routes 0.5), ceiling 80 GB; Stage 1 thresholds (CephFS > 500 GB, root > 100 GB) are checked above
fs=$(df -BG --output=avail /mnt/shared | tail -1 | tr -dc 0-9); [ "$fs" -gt 1000 ] && ok "CephFS free ${fs} GB (> 1000 for an 80 GB ceiling with export temporaries)" || bad "CephFS free ${fs} GB"
info "expected raw ~43 GB (ceiling 80 GB; abort if the run directory exceeds it); export ~23 min, parse ~63 min after the GPUs are free"
info "reservation (from the laptop): pod dsv41-gb300-reserve-tray16 Running and the only GPU pod on the node before the go; hard limit 4 h"
echo "# result: $([ $fail = 0 ] && echo PREFLIGHT PASS || echo PREFLIGHT FAIL)"
exit $fail
