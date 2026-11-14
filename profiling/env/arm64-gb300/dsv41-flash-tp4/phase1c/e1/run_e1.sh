#!/usr/bin/env bash
# E1 driver (NOT to be run without Kevin's explicit go). One pass per invocation, on THIS tray (4 GPUs), as dc:
#   run_e1.sh e1_dry_off | e1_dry_on | e1_main | e1_main_norouts
# Mirrors Phase 0 scripts/run_pass.sh (sha256 9b1fdf65...) with the E1 run root and the host preflight in front.
# Start it detached in a tmux session on the tray (never `cd && setsid … &` over ssh):
#   tmux new -d -s e1-<pass> "bash $E1/run_e1.sh <pass>"
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc GB300_ROOT=$S/gb300
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
unset CUDA_VISIBLE_DEVICES UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
PASS=${1:?pass}
REPO=${E1_REPO:-$S/ServingStudioSim}      # launcher code tree (Stage 1: the scratch worktree with the opt-in patch)
E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
RUN=$S/ServingStudioSim/logs/20261114_dsv41_flash_gb300_tp4_e1   # artifacts always under the main clone (git-ignored)
CFG=$E1/configs/$PASS/profile.json
[ -f "$CFG" ] || { echo "no config $CFG"; exit 2; }
mkdir -p $RUN/$PASS
bash $E1/preflight_e1_host.sh 2>&1 | tee $RUN/$PASS/preflight.log
[ "${PIPESTATUS[0]}" = "0" ] || { echo "preflight failed; not starting"; exit 3; }
cp $CFG $RUN/$PASS/profile.json
cd $REPO
source profiling/env/arm64-gb300/gb300.env       # Sim launcher venv, CUDA 13.1 on PATH
unset CUDA_VISIBLE_DEVICES                       # the profile config selects the devices
echo "[$PASS] start $(date -u +%FT%TZ) on $(hostname) launcher_tree=$REPO repo=$(git rev-parse HEAD) cfg_sha=$(sha256sum $CFG | cut -c1-16)" | tee -a $RUN/$PASS/run.log
uv run --no-sync python -m launcher alignment profile "$RUN/$PASS/profile.json" 2>&1 | tee -a $RUN/$PASS/run.log
rc=${PIPESTATUS[0]}
echo "[$PASS] rc=$rc end $(date -u +%FT%TZ)" | tee -a $RUN/$PASS/run.log
# teardown check (Phase 0: EngineCore + workers once survived rc=0): report, never kill blindly
sleep 20
left=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
if [ -n "$left" ]; then echo "[$PASS] WARNING compute apps still on the GPUs after exit:" | tee -a $RUN/$PASS/run.log; echo "$left" | tee -a $RUN/$PASS/run.log; fi
exit $rc
