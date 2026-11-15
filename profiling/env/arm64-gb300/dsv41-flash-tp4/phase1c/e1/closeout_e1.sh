#!/usr/bin/env bash
# Closeout gate after a pass, ON the capture tray as dc (read-only; prints the verified termination plan, signals nothing):
#   bash closeout_e1.sh [<pass>]     -> PASS/FAIL per item, exit 1 on any FAIL
# Uses the checks/ next to itself, whichever tree it is run from.
# Items: GPUs clear (0 compute apps, < 100 MiB per GPU), process discovery (checks/e1_process_census.sh:
# vllm/nsys/session_runner, driver shells, tail/less watchers, nvidia-smi loops, python monitors, tmux, cwd/cmdline refs),
# run directory present with rc line, raw sizes. The reservation pod is deleted from the laptop (kubectl); the command is printed.
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc
E1=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # the e1 directory this script lives in (no E1_REPO needed)
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15; PASS=${1:-e1_b8_recovery}
fail=0
echo "# closeout $PASS on $(hostname) at $(date -u +%FT%TZ)"
for i in 0 1 2 3; do m=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '); [ "${m:-9999}" -lt 100 ] && echo "PASS  GPU$i memory used ${m} MiB" || { echo "FAIL  GPU$i memory used ${m} MiB"; fail=1; }; done
REG=$RUN/pids/owned.tsv
E1_PID_REGISTRY=$REG bash $E1/checks/e1_process_census.sh; [ $? = 0 ] || fail=1
echo "## termination plan (dry-run; nothing is signalled here)"
python3 -I $E1/checks/e1_pids.py cleanup --registry $REG; prc=$?
case $prc in 0) echo "PASS  registry plan: nothing owned is alive";; 2) echo "FAIL  registry plan: owned processes alive or lookalikes present -> review; to terminate ONLY the verified owned tree: python3 -I $E1/checks/e1_pids.py cleanup --registry $REG --execute --grace 30 --log $RUN/cpu/cleanup_$(date -u +%Y%m%dT%H%M%SZ).jsonl"; fail=1;; *) echo "FAIL  registry plan refused (rc $prc): a registered pid is in this shell's ancestry or another user's -> stop and report"; fail=1;; esac
if [ -f $RUN/$PASS/run.log ]; then
  echo "INFO  $(grep -a 'rc=' $RUN/$PASS/run.log | tail -1)"; echo "INFO  $(grep -a 'EngineDeadError-family' $RUN/$PASS/run.log | tail -1 | cut -c1-200)"
  echo "INFO  raw size $(du -sh $RUN/$PASS 2>/dev/null | cut -f1); nsys files: $(ls -l $RUN/$PASS/profile/nsys/ 2>/dev/null | awk 'NR>1{printf "%s %s  ", $5, $9}')"
else echo "INFO  no run.log for $PASS (nothing ran)"; fi
echo "INFO  reservation pod (from the laptop): HTTPS_PROXY=socks5://127.0.0.1:11080 kubectl --request-timeout=90s -n inference-bench delete pod dsv41-gb300-reserve-tray16 ; then kubectl get pods -A -o wide --field-selector spec.nodeName=pod4-gb300-10-tray16 (only daemonsets expected)"
echo "# closeout result: $([ $fail = 0 ] && echo CLEAN || echo NOT CLEAN)"
exit $fail
