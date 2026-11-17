#!/usr/bin/env bash
# Closeout gate after the V1 pass, ON the capture tray as dc (read-only; prints the verified termination plan, signals nothing):
#   bash closeout_e1_v1.sh     -> PASS/FAIL per item, exit 1 on any FAIL
# Same items as closeout_e1.sh (which hard-codes the Stage 1.5 run root) plus the lifecycle record's final state and the preserved stream.
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc
E1=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v1; PASS=e1_v1_lifecycle
fail=0
echo "# closeout $PASS on $(hostname) at $(date -u +%FT%TZ)"
for i in 0 1 2 3; do m=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '); [ "${m:-9999}" -lt 100 ] && echo "PASS  GPU$i memory used ${m} MiB" || { echo "FAIL  GPU$i memory used ${m} MiB"; fail=1; }; done
REG=$RUN/pids/owned.tsv
E1_PID_REGISTRY=$REG bash $E1/checks/e1_process_census.sh; [ $? = 0 ] || fail=1
echo "## termination plan (dry-run; nothing is signalled here)"
python3 -I $E1/checks/e1_pids.py cleanup --registry $REG; prc=$?
case $prc in 0) echo "PASS  registry plan: nothing owned is alive";; 2) echo "FAIL  registry plan: owned processes alive or lookalikes present -> review; to terminate ONLY the verified owned tree: python3 -I $E1/checks/e1_pids.py cleanup --registry $REG --execute --grace 30 --log $RUN/cpu/cleanup_$(date -u +%Y%m%dT%H%M%SZ).jsonl"; fail=1;; *) echo "FAIL  registry plan refused (rc $prc): a registered pid is in this shell's ancestry or another user's -> stop and report"; fail=1;; esac
LC=$RUN/$PASS/profile/nsys/lifecycle.json
if [ -f $LC ]; then
  python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print('INFO  lifecycle final phase', d['phase'], 'nsys alive', (d.get('nsys') or {}).get('alive'), 'server alive', (d.get('server') or {}).get('alive'), 'gpu released', (d.get('gpu') or {}).get('released_at'), 'preserved', (d.get('stream') or {}).get('preserved_path'))" $LC
  [ "$(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['phase'])" $LC)" = report_written ] && echo "PASS  lifecycle phase report_written" || { echo "FAIL  lifecycle phase not report_written (stop_failed → preserved stream → profile --import-stream on CPU; do not relaunch)"; fail=1; }
else echo "INFO  no lifecycle.json for $PASS (nothing ran)"; fi
ls -d /tmp/*.raw-preserve /tmp/nsys-report-*.qdstrm 2>/dev/null | sed 's/^/INFO  stream artefact in \/tmp: /'
echo "INFO  the preserved stream is removed by the operator only after postrun_e1_v1.sh reports PASS (its L10 recovery proof needs it)"
if [ -f $RUN/$PASS/run.log ]; then
  echo "INFO  $(grep -a 'rc=' $RUN/$PASS/run.log | tail -1)"; echo "INFO  $(grep -a 'EngineDeadError-family' $RUN/$PASS/run.log | tail -1 | cut -c1-200)"
  echo "INFO  raw size $(du -sh $RUN/$PASS 2>/dev/null | cut -f1); nsys files: $(ls -l $RUN/$PASS/profile/nsys/ 2>/dev/null | awk 'NR>1{printf "%s %s  ", $5, $9}')"
else echo "INFO  no run.log for $PASS (nothing ran)"; fi
echo "INFO  reservation pod (from the laptop): HTTPS_PROXY=socks5://127.0.0.1:11080 kubectl --request-timeout=90s -n inference-bench delete pod dsv41-gb300-reserve-tray16 ; then kubectl get pods -A -o wide --field-selector spec.nodeName=pod4-gb300-10-tray16 (only daemonsets expected)"
echo "# closeout result: $([ $fail = 0 ] && echo CLEAN || echo NOT CLEAN)"
exit $fail
