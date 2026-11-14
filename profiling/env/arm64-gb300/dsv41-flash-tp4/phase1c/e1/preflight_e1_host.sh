#!/usr/bin/env bash
# E1 preflight, host side (read-only). Run ON the capture tray as dc:
#   ssh pod4-gb300-10-trayNN 'export HOME=/mnt/shared/home/dc; bash $HOME/workspace/serving-studio/ServingStudioSim/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/preflight_e1_host.sh'
# Prints PASS/FAIL per check and exits non-zero if any check fails. Nothing is started, written or deleted.
# The scheduler check (no GPU pod on this node, reservation pod Running) is done from the laptop:
#   HTTPS_PROXY=socks5://127.0.0.1:11080 kubectl get pods -A -o wide --field-selector spec.nodeName=$(hostname) | grep -v -e csi -e cilium -e monitoring -e nvidia -e prepull
set -u
export HOME=/mnt/shared/home/dc
S=$HOME/workspace/serving-studio
REPO=$S/ServingStudioSim
E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
fail=0
ok()   { echo "PASS  $*"; }
bad()  { echo "FAIL  $*"; fail=1; }
check() { local name=$1; shift; if "$@" >/dev/null 2>&1; then ok "$name"; else bad "$name"; fi; }

echo "# E1 host preflight on $(hostname) at $(date -u +%FT%TZ)"
[ "$(hostname)" != "pod4-gb300-3-tray06-f3" ] && [ "$(hostname)" != "pod4-gb300-3-tray10-f3" ] && ok "not an off-limits tray" || bad "off-limits tray"

# 1. GPUs idle: no compute apps, 0 MiB used on all four
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
[ "$apps" = "0" ] && ok "no compute apps on GPUs" || bad "compute apps present: $apps"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[ "$used" -lt 100 ] && ok "GPU memory used total ${used} MiB" || bad "GPU memory in use: ${used} MiB"
ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l); [ "$ngpu" = "4" ] && ok "4 GPUs visible" || bad "GPUs visible: $ngpu"
[ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" = "595.71.05" ] && ok "driver 595.71.05 (Phase 0)" || bad "driver differs from Phase 0"

# 2. Leftover servers / nsys from earlier runs (by name, pattern cannot match this script)
left=$(pgrep -c -f "vllm.entrypoints.cli.mai[n]|nsys profil[e]|session_runne[r]" || true)
[ "$left" = "0" ] && ok "no leftover vllm/nsys/req-frontend processes" || bad "leftover processes: $left"

# 3. Software identity = Phase 0 recipe
fork=$(git -C $REPO/alignment/profiler/vllm rev-parse HEAD 2>/dev/null); [ "$fork" = "0ab18711335780e062d65994a3243da0281d2081" ] && ok "fork submodule at 0ab18711" || bad "fork submodule HEAD $fork"
ver=$($S/gb300/fork-env-v0.31.0/venv/bin/python -c 'import vllm,torch;print(vllm.__version__, torch.__version__)' 2>/dev/null); [ "$ver" = "0.31.1.dev17+g0ab187113 2.13.0+cu130" ] && ok "venv $ver" || bad "venv versions: $ver"
[ -x /usr/local/bin/nsys ] && /usr/local/bin/nsys --version | grep -q "2025.6.3.541" && ok "nsys 2025.6.3.541" || bad "nsys version"
[ -f /mnt/shared/weights/DeepSeek-V4.1-Flash-1017/config.json ] && ok "weights present" || bad "weights missing"
[ "$(stat -c %s $S/corpus/enwik9 2>/dev/null)" = "1000000000" ] && ok "enwik9 corpus 1e9 bytes" || bad "corpus"
for d in deepgemm triton nv-compute flashinfer-autotune; do [ -n "$(ls -A $S/gb300/cache/$d 2>/dev/null)" ] && ok "cache warm: $d" || bad "cache empty: $d"; done
rh=$(git -C $REPO rev-parse HEAD); echo "INFO  ServingStudioSim HEAD $rh ($(git -C $REPO branch --show-current))"
[ -z "$(git -C $REPO status --short --untracked-files=no)" ] && ok "repo tree clean (tracked files)" || bad "repo has uncommitted tracked changes"

# 4. E1 inputs match the committed hashes
( cd $E1 && sha256sum -c INPUTS.sha256 --quiet ) && ok "E1 traces/configs match INPUTS.sha256" || bad "E1 inputs hash mismatch"
grep -q '"capture_start_delay_seconds": 10.0' $E1/configs/e1_main/profile.json && ok "nsys delayed start 10 s, no bounded stop" || bad "nsys block"
! grep -q capture_duration_seconds $E1/configs/e1_main/profile.json && ok "no capture_duration_seconds (Phase 0 lesson)" || bad "bounded stop present"

# 5. Disk: capture volume (CephFS) and the tray root (nsys temp)
free_shared=$(df -BG --output=avail /mnt/shared | tail -1 | tr -dc 0-9); [ "$free_shared" -gt 500 ] && ok "CephFS free ${free_shared} GB (> 500)" || bad "CephFS free ${free_shared} GB"
free_root=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); [ "$free_root" -gt 100 ] && ok "root disk free ${free_root} GB (> 100)" || bad "root free ${free_root} GB"
RUN=$REPO/logs/20261114_dsv41_flash_gb300_tp4_e1
[ ! -e "$RUN/e1_main/profile" ] && ok "no previous e1_main output dir" || bad "e1_main output dir exists: $RUN/e1_main/profile"

# 6. Launcher validates the configs (CPU only)
( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && unset CUDA_VISIBLE_DEVICES && \
  uv run --no-sync python -m launcher alignment profile --dry-run $E1/configs/e1_main/profile.json ) >/dev/null 2>&1 && ok "launcher dry-run e1_main" || bad "launcher dry-run e1_main (patch applied?)"

echo "# result: $([ $fail = 0 ] && echo PREFLIGHT PASS || echo PREFLIGHT FAIL)"
exit $fail
