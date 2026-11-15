#!/usr/bin/env bash
# Stage 1.5 driver (NOT to be run without Kevin's explicit go). One pass per invocation, on THIS tray (4 GPUs), as dc:
#   run_e1_stage15.sh e1_dry_off_v2 | e1_dry_on_v2 | e1_dry_off2_v2 | e1_nsys_val
# Same shape as run_e1.sh (Stage 1, unchanged) with the configs_v2 directory, the stage15 run root and the Stage 1.5 preflight.
# Start detached in a tmux session on the tray (never `cd && setsid … &` over ssh):
#   tmux new -d -s e15-<pass> "E1_REPO=$WT bash $E1/run_e1_stage15.sh <pass>"
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc GB300_ROOT=$S/gb300
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
unset CUDA_VISIBLE_DEVICES UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
PASS=${1:?pass}
REPO=${E1_REPO:-$S/ServingStudioSim}      # launcher code tree with both stream patches applied (Stage 1.5: the scratch worktree)
E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15   # artifacts under the main clone's logs (git-ignored); Stage 1 dir untouched
CFG=$E1/configs_v2/$PASS/profile.json
[ -f "$CFG" ] || { echo "no config $CFG"; exit 2; }
mkdir -p $RUN/$PASS
E1_REPO=$REPO bash $E1/preflight_e1_stage15.sh 2>&1 | tee $RUN/$PASS/preflight.log
[ "${PIPESTATUS[0]}" = "0" ] || { echo "preflight failed; not starting"; exit 3; }
cp $CFG $RUN/$PASS/profile.json
cd $REPO
source profiling/env/arm64-gb300/gb300.env
unset CUDA_VISIBLE_DEVICES
echo "[$PASS] start $(date -u +%FT%TZ) on $(hostname) launcher_tree=$REPO repo=$(git rev-parse HEAD) cfg_sha=$(sha256sum $CFG | cut -c1-16) session_runner=$(sha256sum alignment/load_generator/req-frontend/target/release/session_runner | cut -c1-16)" | tee -a $RUN/$PASS/run.log
uv run --no-sync python -m launcher alignment profile "$RUN/$PASS/profile.json" 2>&1 | tee -a $RUN/$PASS/run.log
rc=${PIPESTATUS[0]}
echo "[$PASS] rc=$rc end $(date -u +%FT%TZ)" | tee -a $RUN/$PASS/run.log
# keep this launch's autotune cache file(s) beside the run (which hash dir was used is in the server log)
for f in $(grep -ao "Using FlashInfer autotune cache file: [^ ]*" $RUN/$PASS/profile/vllm/*_server.log 2>/dev/null | awk '{print $NF}' | sort -u); do
  [ -f "$f" ] && cp -p "$f" $RUN/$PASS/autotune_configs.$(basename $(dirname $f) | cut -c1-12).json
done
# EngineDeadError bookkeeping (runs/stage15/engine-dead-error.md): report where such lines sit relative to the shutdown trigger
SL=$(ls $RUN/$PASS/profile/vllm/*_server.log 2>/dev/null | head -1)
if [ -n "$SL" ]; then
  sd=$(grep -an "\[shutdown\].*trigger received" "$SL" | head -1 | cut -d: -f1); sd=${sd:-0}
  ede=$(grep -an "EngineDeadError\|EngineCore encountered an issue\|output_handler failed" "$SL" | cut -d: -f1 | tr "\n" " ")
  before=0; for l in $ede; do [ "$sd" = 0 ] || [ "$l" -lt "$sd" ] && before=$((before+1)); done
  echo "[$PASS] shutdown trigger at server-log line ${sd}; EngineDeadError-family lines: ${ede:-none}; before shutdown: $before $([ $before = 0 ] && echo "(accepted)" || echo "(ABORT CONDITION)")" | tee -a $RUN/$PASS/run.log
fi
sleep 20
left=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
if [ -n "$left" ]; then echo "[$PASS] WARNING compute apps still on the GPUs after exit:" | tee -a $RUN/$PASS/run.log; echo "$left" | tee -a $RUN/$PASS/run.log; fi
exit $rc
