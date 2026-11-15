#!/usr/bin/env bash
# b=8 recovery driver (NOT to be run without Kevin's explicit go). Exactly one pass, e1_b8_recovery, on THIS tray (4 GPUs), as dc.
# Refuses any other pass name: the trace holds the 8 b=8 requests only, so neither b=48 nor the full E1 workload can run from here.
#   tmux new -d -s e1b8-recovery "E1_REPO=$WT bash $E1/run_e1_b8.sh"
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc GB300_ROOT=$S/gb300
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
unset CUDA_VISIBLE_DEVICES UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
PASS=e1_b8_recovery
[ "${1:-$PASS}" = "$PASS" ] || { echo "this driver runs only $PASS (got '$1'); b=48 / e1_main are not authorized"; exit 2; }
REPO=${E1_REPO:-$S/ServingStudioSim}
E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15
CFG=$E1/configs_v2/$PASS/profile.json; TRACE=$E1/traces/$PASS.csv
[ -f "$CFG" ] || { echo "no config $CFG"; exit 2; }
mkdir -p $RUN/$PASS
E1_REPO=$REPO bash $E1/preflight_e1_b8.sh 2>&1 | tee $RUN/$PASS/preflight.log
[ "${PIPESTATUS[0]}" = "0" ] || { echo "preflight failed; not starting"; exit 3; }
# defence in depth, immediately before launch: the config's trace is the 8-row b=8 trace
[ "$(tail -n +2 $TRACE | grep -c .)" = 8 ] && ! grep -q b48 $TRACE && grep -q "\"path\": \"$TRACE\"" $CFG || { echo "trace/config guard failed; not starting"; exit 4; }
cp $CFG $RUN/$PASS/profile.json; cp $TRACE $RUN/$PASS/trace.csv
# experiment-owned process record: this driver shell (pid + start time + boot id); the launcher and everything it starts are its
# descendants, so `e1_pids.py cleanup` can terminate exactly this tree and nothing else (closeout_e1.sh / packet §5)
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
# closeout discovery + registry plan (this driver's own shell is still alive and is owned by design; it exits right after this)
E1_PID_REGISTRY=$REG bash $E1/checks/e1_process_census.sh 2>&1 | grep -v "tmux sessions" | tee -a $RUN/$PASS/run.log
exit $rc
