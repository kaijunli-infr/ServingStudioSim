#!/usr/bin/env bash
# Run one Phase-0 profile pass on THIS tray (4 GPUs). Usage:
#   run_pass.sh <pass>   pass = quadrant_nsys | quadrant_workload | diverse_nsys | diverse_workload | diverse_corpus
# Preflight: refuses to start if any compute app is on the GPUs. Check the scheduler (0/4 GPUs on this node) first.
set -u
SS=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc GB300_ROOT=$SS/gb300
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
unset CUDA_VISIBLE_DEVICES UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
PASS=${1:?pass}
RUN=$SS/ServingStudioSim/logs/20261112_dsv41_flash_gb300_tp4
CFG=$RUN/$PASS/profile.json
[ -f "$CFG" ] || { echo "no config $CFG"; exit 2; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "GPUs busy on $(hostname):"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; exit 3
fi
cd $SS/ServingStudioSim
source profiling/env/arm64-gb300/gb300.env     # Sim launcher venv (UV_PROJECT_ENVIRONMENT), CUDA 13.1 on PATH
unset CUDA_VISIBLE_DEVICES                     # the profile config selects the devices
FORK_PY=$(python3 -c "import json;print(json.load(open('$CFG'))['fork_python'])")
echo "[$PASS] start $(date -u +%FT%TZ) on $(hostname) fork_python=$FORK_PY" | tee -a $RUN/$PASS/run.log
uv run --no-sync python -m launcher alignment profile "$CFG" 2>&1 | tee -a $RUN/$PASS/run.log
echo "[$PASS] rc=${PIPESTATUS[0]} end $(date -u +%FT%TZ)" | tee -a $RUN/$PASS/run.log
