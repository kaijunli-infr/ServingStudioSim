#!/usr/bin/env bash
# One read-only monitoring sample for a running E1 pass (packet v2 §7). Prints state lines and explicit ABORT-CONDITION lines; it
# never signals anything. Run every 60 s from the laptop through ssh (neutral-path), or in a tmux loop:
#   bash monitor_e1.sh <run root> <pass> [startup_limit_min=30] [capture_limit_min=25] [hold_limit_min=20] [raw_ceiling_GB=80]
set -u
RUN=$1; PASS=$2; SUL=${3:-30}; CL=${4:-25}; HL=${5:-20}; CEIL=${6:-80}
L=$RUN/$PASS/run.log; SL=$(ls $RUN/$PASS/profile/vllm/*_server.log 2>/dev/null | head -1)
drv=$(awk -F'\t' 'NR>1{print $1}' $RUN/pids/owned.tsv 2>/dev/null | head -1)
now=$(date -u +%s)
t0=$(grep -ao "^\[$PASS\] start [0-9T:-]*Z" $L 2>/dev/null | head -1 | awk '{print $3}'); t0s=$( [ -n "$t0" ] && date -u -d "$t0" +%s || echo "$now")
echo "== $(date -u +%FT%TZ) $PASS driver=$drv alive=$([ -n "$drv" ] && [ -d /proc/$drv ] && echo y || echo n) elapsed_min=$(( (now - t0s) / 60 )) tmux=$(tmux ls 2>/dev/null | cut -d: -f1 | tr '\n' ,)"
su=$(grep -a -c "Application startup complete" "$SL" 2>/dev/null || echo 0)
mb=$(grep -a -c "measurement begins" $L 2>/dev/null || echo 0); wd=$(grep -a -c "workload done" $L 2>/dev/null || echo 0); rc=$(grep -a -o "^\[$PASS\] rc=[0-9]*" $L 2>/dev/null | tail -1)
echo "phase: startup_complete=$su measurement_begun=$mb workload_done=$wd ${rc:-running}"
[ "$su" = 0 ] && [ $(( (now - t0s) / 60 )) -gt $SUL ] && echo "ABORT-CONDITION startup exceeds ${SUL} min"
if [ "$mb" != 0 ] && [ "$wd" = 0 ]; then
  mbt=$(stat -c %Y $L); echo "capture running for $(( (now - $(grep -a -n "measurement begins" $L >/dev/null && echo $t0s)) / 60 )) min since launcher start (limit from measurement begin: ${CL} min)"
fi
grep -a -q "EngineDeadError\|EngineCore encountered an issue\|output_handler failed" "$SL" 2>/dev/null && [ "$wd" = 0 ] && echo "ABORT-CONDITION engine error lines before shutdown"
gm=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'); echo "gpu: $gm"
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  p=$pid; own=n; while [ "$p" != 1 ] && [ -n "$p" ]; do [ "$p" = "$drv" ] && own=y && break; p=$(awk '/^PPid:/{print $2}' /proc/$p/status 2>/dev/null); done
  echo "gpuproc $pid own=$own $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-80)"; [ $own = n ] && echo "ABORT-CONDITION foreign GPU process $pid"
done
if [ "$wd" != 0 ] && [ -z "$rc" ]; then
  held=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
  wdt=$(date -u -d "$(grep -a -o "workload done.*" $L | head -1 | grep -o "20[0-9-]*T[0-9:]*Z" | head -1 || echo now)" +%s 2>/dev/null || echo $now)
  [ "$held" -gt 100 ] && [ $(( (now - wdt) / 60 )) -gt $HL ] && echo "ABORT-CONDITION GPUs still held ${held} MiB more than ${HL} min after the workload ended"
fi
raw=$(du -sBG $RUN/$PASS 2>/dev/null | cut -f1 | tr -dc 0-9); echo "pass dir ${raw:-0} GB (ceiling $CEIL); root free $(df -BG --output=avail / | tail -1 | tr -dc 0-9) GB; CephFS free $(df -BG --output=avail /mnt/shared | tail -1 | tr -dc 0-9) GB"
[ "${raw:-0}" -gt "$CEIL" ] && echo "ABORT-CONDITION raw artifacts ${raw} GB exceed ceiling ${CEIL} GB"
exit 0
