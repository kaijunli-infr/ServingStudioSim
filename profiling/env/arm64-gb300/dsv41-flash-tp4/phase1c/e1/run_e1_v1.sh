#!/usr/bin/env bash
# V1 driver (NOT to be run without Kevin's explicit go). Exactly one pass, e1_v1_lifecycle, on THIS tray (4 GPUs), as dc, from the
# pinned exp/profiler-lifecycle tree:
#   tmux new -d -s e1v1-lifecycle "E1_REPO=$S/wt-lifecycle bash $E1/run_e1_v1.sh"      (through the neutral-path launch wrapper)
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc GB300_ROOT=$S/gb300
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
unset CUDA_VISIBLE_DEVICES UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
PASS=e1_v1_lifecycle
[ "${1:-$PASS}" = "$PASS" ] || { echo "this driver runs only $PASS (got '$1')"; exit 2; }
REPO=${E1_REPO:-$S/wt-lifecycle}
E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v1
CFG=$E1/configs_v1/$PASS/profile.json; TRACE=$E1/traces/e1_dry.csv
[ -f "$CFG" ] || { echo "no config $CFG"; exit 2; }
mkdir -p $RUN/$PASS
E1_REPO=$REPO E1_V1_OUT=$RUN/preflight bash $E1/preflight_e1_v1.sh 2>&1 | tee $RUN/$PASS/preflight.log
[ "${PIPESTATUS[0]}" = "0" ] || { echo "preflight failed; not starting"; exit 3; }
# defence in depth, immediately before launch: frozen 56-row trace, no delayed start, explicit GPU-held ceiling
[ "$(tail -n +2 $TRACE | grep -c .)" = 56 ] && [ "$(sha256sum $TRACE | cut -c1-16)" = c6d8e13b7229ffd4 ] && grep -q "\"path\": \"$TRACE\"" $CFG && ! grep -q capture_start_delay_seconds $CFG && grep -q '"gpu_held_ceiling_s": 2700.0' $CFG || { echo "trace/config guard failed; not starting"; exit 4; }
cp $CFG $RUN/$PASS/profile.json; cp $TRACE $RUN/$PASS/trace.csv
REG=$RUN/pids/owned.tsv; python3 -I $E1/checks/e1_pids.py register --registry $REG --label driver:$PASS $$ | tee -a $RUN/$PASS/run.log
# control-file probe: registered background job (identity-verified cleanup), writes the two pre-registered lines once the stop is requested
python3 -I $E1/checks/e1_pids.py spawn --registry $REG --label probe:$PASS -- bash $E1/v1_control_probe.sh $RUN/$PASS/profile/nsys $RUN/$PASS/probe.log | tee -a $RUN/$PASS/run.log
cd $REPO
source profiling/env/arm64-gb300/gb300.env
unset CUDA_VISIBLE_DEVICES
echo "[$PASS] start $(date -u +%FT%TZ) on $(hostname) launcher_tree=$REPO repo=$(git rev-parse HEAD) tree_alignment=$(git rev-parse HEAD:alignment | cut -c1-12) cfg_sha=$(sha256sum $CFG | cut -c1-16) trace_sha=$(sha256sum $TRACE | cut -c1-16) session_runner=$(sha256sum alignment/load_generator/req-frontend/target/release/session_runner | cut -c1-16)" | tee -a $RUN/$PASS/run.log
uv run --no-sync python -m launcher alignment profile "$RUN/$PASS/profile.json" 2>&1 | tee -a $RUN/$PASS/run.log
rc=${PIPESTATUS[0]}
echo "[$PASS] rc=$rc end $(date -u +%FT%TZ)" | tee -a $RUN/$PASS/run.log
LC=$RUN/$PASS/profile/nsys/lifecycle.json
[ -f $LC ] && python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print('[lifecycle] phase', d['phase'], '| http', d['http'], '| stream', d['stream'], '| report', d['report'], '| gpu', d['gpu'], '| escalations', len(d['escalations']), '| control', [(c['command'],c['result']) for c in d['control_commands']], '| actions', [(a.get('pid'),a.get('action'),a.get('reason')) for a in d['actions']], '| warnings', [w['message'] for w in d['warnings']])" $LC | tee -a $RUN/$PASS/run.log
for f in $(grep -ao "Using FlashInfer autotune cache file: [^ ]*" $RUN/$PASS/profile/vllm/*_server.log 2>/dev/null | awk '{print $NF}' | sort -u); do
  [ -f "$f" ] && cp -p "$f" $RUN/$PASS/autotune_configs.$(basename $(dirname $f) | cut -c1-12).json
done
SL=$(ls $RUN/$PASS/profile/vllm/*_server.log 2>/dev/null | head -1)
if [ -n "$SL" ]; then
  sd=$(grep -an "\[shutdown\].*trigger received" "$SL" | head -1 | cut -d: -f1); sd=${sd:-0}
  ede=$(grep -an "EngineDeadError\|EngineCore encountered an issue\|output_handler failed" "$SL" | cut -d: -f1 | tr "\n" " ")
  before=0; for l in $ede; do [ "$sd" = 0 ] || [ "$l" -lt "$sd" ] && before=$((before+1)); done
  echo "[$PASS] shutdown trigger at server-log line ${sd}; EngineDeadError-family lines: ${ede:-none}; before shutdown: $before $([ $before = 0 ] && echo "(accepted)" || echo "(ABORT CONDITION)")" | tee -a $RUN/$PASS/run.log
  echo "[$PASS] stop timeline: $(grep -ao "[0-9:]* \[api_router.py:31\] Stopping profiler" "$SL" | head -1) → $(grep -ao "[0-9:]* \[api_router.py:33\] Profiler stopped" "$SL" | head -1); $(grep -ao "Generating '[^']*'" "$SL" | head -1)" | tee -a $RUN/$PASS/run.log
fi
sleep 20
E1_PID_REGISTRY=$REG bash $E1/checks/e1_process_census.sh 2>&1 | grep -v "tmux sessions" | tee -a $RUN/$PASS/run.log
exit $rc
