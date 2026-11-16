#!/usr/bin/env bash
# Full-E1 v3 driver (NOT to be run without Kevin's explicit go). Exactly one pass, e1_main_v3, on THIS tray (4 GPUs), as dc.
#   tmux new -d -s e1main-v3 "E1_REPO=$WT bash $E1/run_e1_main_v3.sh"      (through the neutral-path launch wrapper)
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc GB300_ROOT=$S/gb300
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
unset CUDA_VISIBLE_DEVICES UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
PASS=e1_main_v3
[ "${1:-$PASS}" = "$PASS" ] || { echo "this driver runs only $PASS (got '$1')"; exit 2; }
REPO=${E1_REPO:-$S/ServingStudioSim}
E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_main
CFG=$E1/configs_v3/$PASS/profile.json; TRACE=$E1/traces/e1_main.csv
[ -f "$CFG" ] || { echo "no config $CFG"; exit 2; }
mkdir -p $RUN/$PASS
E1_REPO=$REPO bash $E1/preflight_e1_main_v3.sh 2>&1 | tee $RUN/$PASS/preflight.log
[ "${PIPESTATUS[0]}" = "0" ] || { echo "preflight failed; not starting"; exit 3; }
# defence in depth, immediately before launch: the config's trace is the frozen 450-row E1 trace and the config has no delayed start
[ "$(tail -n +2 $TRACE | grep -c .)" = 450 ] && [ "$(sha256sum $TRACE | cut -c1-16)" = 2e147402b5a24f9e ] && grep -q "\"path\": \"$TRACE\"" $CFG && ! grep -q capture_start_delay_seconds $CFG || { echo "trace/config guard failed; not starting"; exit 4; }
cp $CFG $RUN/$PASS/profile.json; cp $TRACE $RUN/$PASS/trace.csv
# experiment-owned process record (pid + start time + boot id): the launcher and everything it starts are descendants of this shell
REG=$RUN/pids/owned.tsv; python3 -I $E1/checks/e1_pids.py register --registry $REG --label driver:$PASS $$ | tee -a $RUN/$PASS/run.log
cd $REPO
source profiling/env/arm64-gb300/gb300.env
unset CUDA_VISIBLE_DEVICES
echo "[$PASS] start $(date -u +%FT%TZ) on $(hostname) launcher_tree=$REPO repo=$(git rev-parse HEAD) cfg_sha=$(sha256sum $CFG | cut -c1-16) trace_sha=$(sha256sum $TRACE | cut -c1-16) session_runner=$(sha256sum alignment/load_generator/req-frontend/target/release/session_runner | cut -c1-16)" | tee -a $RUN/$PASS/run.log
uv run --no-sync python -m launcher alignment profile "$RUN/$PASS/profile.json" 2>&1 | tee -a $RUN/$PASS/run.log
rc=${PIPESTATUS[0]}
echo "[$PASS] rc=$rc end $(date -u +%FT%TZ)" | tee -a $RUN/$PASS/run.log
for f in $(grep -ao "Using FlashInfer autotune cache file: [^ ]*" $RUN/$PASS/profile/vllm/*_server.log 2>/dev/null | awk '{print $NF}' | sort -u); do
  [ -f "$f" ] && cp -p "$f" $RUN/$PASS/autotune_configs.$(basename $(dirname $f) | cut -c1-12).json
done
SL=$(ls $RUN/$PASS/profile/vllm/*_server.log 2>/dev/null | head -1)
if [ -n "$SL" ]; then
  sd=$(grep -an "\[shutdown\].*trigger received" "$SL" | head -1 | cut -d: -f1); sd=${sd:-0}
  ede=$(grep -an "EngineDeadError\|EngineCore encountered an issue\|output_handler failed" "$SL" | cut -d: -f1 | tr "\n" " ")
  before=0; for l in $ede; do [ "$sd" = 0 ] || [ "$l" -lt "$sd" ] && before=$((before+1)); done
  echo "[$PASS] shutdown trigger at server-log line ${sd}; EngineDeadError-family lines: ${ede:-none}; before shutdown: $before $([ $before = 0 ] && echo "(accepted)" || echo "(ABORT CONDITION)")" | tee -a $RUN/$PASS/run.log
fi
sleep 20
E1_PID_REGISTRY=$REG bash $E1/checks/e1_process_census.sh 2>&1 | grep -v "tmux sessions" | tee -a $RUN/$PASS/run.log
exit $rc
