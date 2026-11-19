#!/usr/bin/env bash
# Closeout gate after the V4 pass, ON the capture tray as dc (read-only; prints the verified termination plan, signals nothing):
#   bash closeout_e1_v4scale.sh     -> PASS/FAIL per resource item, then four separate statements and one combined result
# Resource items as closeout_e1_fr.sh, plus: no Nsight CLI, agent or importer of this user left (found by executable path, never by name
# pattern), every registered observer, the supervisor and (once it has a verdict) the closeout watcher gone.
# Statements (checks/e1_closeout_verdict.py): resource cleanup; supervision validity; run response; reservation deletion; frozen scientific
# gates (reported from the post-run record if there is one; never evaluated here). The combined result is CLEAN (exit 0) only if the resources
# are clean, the supervision was valid (a lost supervisor or monitor is INVALID), no abort response or breach marker exists and the
# reservation deletion has not failed. It never deletes anything and never signals anything.
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc
E1=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN=${E1_V4S_RUN:-$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v4scale}; PASS=e1_v4_scale
fail=0
echo "# closeout $PASS on $(hostname) at $(date -u +%FT%TZ)"
for i in 0 1 2 3; do m=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '); [ "${m:-9999}" -lt 100 ] && echo "PASS  GPU$i memory used ${m} MiB" || { echo "FAIL  GPU$i memory used ${m} MiB"; fail=1; }; done
REG=$RUN/pids/owned.tsv
E1_PID_REGISTRY=$REG bash $E1/checks/e1_process_census.sh; [ $? = 0 ] || fail=1
echo "## termination plan (dry-run; nothing is signalled here)"
python3 -I $E1/checks/e1_pids.py cleanup --registry $REG; prc=$?
case $prc in 0) echo "PASS  registry plan: nothing owned is alive (driver, GPU sampler, import probe, memory sampler, storage sampler)";; 2) echo "FAIL  registry plan: owned processes alive or lookalikes present -> review; to terminate ONLY the verified owned tree: python3 -I $E1/checks/e1_pids.py cleanup --registry $REG --execute --grace 30 --log $RUN/cpu/cleanup_$(date -u +%Y%m%dT%H%M%SZ).jsonl"; fail=1;; *) echo "FAIL  registry plan refused (rc $prc): a registered pid is in this shell's ancestry or another user's -> stop and report"; fail=1;; esac
SREG=$RUN/pids/supervisor.tsv
python3 -I $E1/checks/e1_pids.py plan --registry $SREG | sed 's/^/  /'; src=${PIPESTATUS[0]}
[ "$src" = 0 ] && echo "PASS  supervisor registry plan: the supervisor is gone, no refusal" || { echo "FAIL  supervisor registry plan rc $src (the supervisor is still alive, or a refusal / lookalike; see above)"; fail=1; }
ns=$(for d in /proc/[0-9]*; do [ "$(stat -c %u $d 2>/dev/null)" = "$(id -u)" ] || continue; e=$(readlink $d/exe 2>/dev/null) || continue; case "$e" in /opt/nvidia/nsight-systems/*) echo "pid ${d#/proc/} $(basename $e) ppid $(awk '/^PPid:/{print $2}' $d/status 2>/dev/null): $(tr '\0' ' ' < $d/cmdline 2>/dev/null | cut -c1-160)";; esac; done)
if [ -z "$ns" ]; then echo "PASS  no Nsight CLI, agent or importer of this user is alive (by executable path)"; else echo "FAIL  Nsight processes of this user alive (the agent and the importer are outside the registered tree; they are never signalled by pattern; an importer is ended only by the packet's procedure, after the durable stream copy is verified):"; echo "$ns" | sed 's/^/      /'; fail=1; fi
LC=$RUN/$PASS/profile/nsys/lifecycle.json
if [ -f $LC ]; then
  python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print('INFO  lifecycle final phase', d['phase'], 'nsys alive', (d.get('nsys') or {}).get('alive'), 'server alive', (d.get('server') or {}).get('alive'), 'gpu released', (d.get('gpu') or {}).get('released_at'), 'preserved', (d.get('stream') or {}).get('preserved_path'))" $LC
  ph=$(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['phase'])" $LC); case "$ph" in report_written|stop_failed) echo "PASS  lifecycle phase $ph is terminal (acceptance is decided by postrun_e1_v4scale.sh; V4 requires report_written)";; *) echo "FAIL  lifecycle phase $ph is not terminal (importer running or hung: stream stays preserved; verified registry cleanup only)"; fail=1;; esac
else echo "INFO  no lifecycle.json for $PASS (nothing ran)"; fi
CREG=$RUN/pids/closeout.tsv
if [ -f $CREG ]; then
  python3 -I $E1/checks/e1_pids.py plan --registry $CREG | grep -E "summary|OWNED|GONE|REFUSED|STALE" | sed 's/^/  /'
  walive=$(awk -F'\t' '$1 ~ /^[0-9]+$/ {print $1}' $CREG | while read p; do [ -d /proc/$p ] && echo $p; done | wc -l)
  wdone=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1])).get('complete'))" $RUN/$PASS/reservation_closeout.json 2>/dev/null)
  if [ "$walive" = 0 ]; then echo "PASS  closeout watcher registry: the watcher is gone"; elif [ "$wdone" = True ]; then echo "FAIL  closeout watcher: its record is complete but $walive registered process(es) are alive"; fail=1; else echo "INFO  closeout watcher alive (pid in $CREG): it is waiting for the reservation's deletion; its verdict follows"; fi
else echo "INFO  no closeout watcher registry"; fi
for f in import_probe.json mem_watch.json storage_watch.json; do
  if [ -f $RUN/$PASS/$f ]; then python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print('INFO  ' + sys.argv[2] + ':', d.get('result') or ('complete=%s samples=%s breaches=%s' % (d.get('complete'), d.get('samples'), len(d.get('breaches', [])))))" $RUN/$PASS/$f $f; else echo "INFO  $f not present"; fi
done
ls -d /tmp/*.raw-preserve /tmp/nsys-report-*.qdstrm /tmp/v4scale-recovery-* 2>/dev/null | sed 's/^/INFO  stream artefact in \/tmp: /'
echo "INFO  the preserved stream is never removed by a script; postrun_e1_v4scale.sh copies it to durable storage and verifies the copy first"
if [ -f $RUN/$PASS/run.log ]; then
  echo "INFO  $(grep -a 'rc=' $RUN/$PASS/run.log | tail -1)"; echo "INFO  $(grep -a 'EngineDeadError-family' $RUN/$PASS/run.log | tail -1 | cut -c1-200)"
  echo "INFO  run root size $(du -sh $RUN 2>/dev/null | cut -f1); nsys files: $(ls -l $RUN/$PASS/profile/nsys/ 2>/dev/null | awk 'NR>1{printf "%s %s  ", $5, $9}')"
else echo "INFO  no run.log for $PASS (nothing ran)"; fi
[ -f $RUN/$PASS/reservation.json ] && python3 -I $E1/checks/e1_reservation_guard.py deadline --reservation $RUN/$PASS/reservation.json --ceiling-s 5400 | sed "s/^PASS  /INFO  /; s/^FAIL  /INFO  RESERVATION CEILING PASSED (gate G10 will FAIL): /; s/^INCONCLUSIVE/INFO /"
echo "INFO  the reservation pod is deleted only by the registered deleter (the deleting mode of checks/e1_reservation_closeout.py, on the operator's laptop) after the watcher's permit; never by hand before a permit, never early"
echo "## statements"
python3 -I $E1/checks/e1_closeout_verdict.py --run $RUN/$PASS --resource-failures $fail --gates $RUN/cpu/V4S-GATES.json --ceiling-s 5400
exit $?
