#!/usr/bin/env bash
# V1 control-file probe (packet item 8). Waits until the controller has sent the stop request, then writes the two pre-registered
# operator lines to lifecycle.control so the run records a `note` and a 60 s `extend` with their results. Started by run_e1_v1.sh
# through checks/e1_pids.py spawn (registered, identity-verified cleanup). Never signals anything; exits after writing or after 60 min.
#   bash v1_control_probe.sh <nsys dir> <log file>
set -u
NSYS_DIR=$1; LOG=$2
echo "[probe] start $(date -u +%FT%TZ) pid $$ watching $NSYS_DIR/lifecycle.json" >> $LOG
for i in $(seq 1 3600); do
  if [ -f $NSYS_DIR/lifecycle.json ]; then
    state=$(python3 -I -c "import json,sys
try:
    d=json.load(open(sys.argv[1])); print('terminal' if d.get('phase') in ('report_written','stop_failed') else ('stop' if d.get('stop_requested_at') else 'wait'))
except Exception: print('wait')" $NSYS_DIR/lifecycle.json)
    if [ "$state" = "stop" ]; then
      printf 'note V1 operator probe: control file read\n' >> $NSYS_DIR/lifecycle.control
      echo "[probe] wrote note at $(date -u +%FT%TZ)" >> $LOG
      sleep 2
      printf 'extend 60 V1 operator probe: extension logged\n' >> $NSYS_DIR/lifecycle.control
      echo "[probe] wrote extend at $(date -u +%FT%TZ); done" >> $LOG
      exit 0
    elif [ "$state" = "terminal" ]; then
      echo "[probe] stop already terminal before the probe could write (gate L7 will FAIL) $(date -u +%FT%TZ)" >> $LOG; exit 1
    fi
  fi
  sleep 1
done
echo "[probe] gave up after 60 min $(date -u +%FT%TZ)" >> $LOG; exit 2
