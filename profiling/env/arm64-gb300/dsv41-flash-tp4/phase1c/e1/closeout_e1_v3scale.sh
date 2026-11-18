#!/usr/bin/env bash
# Closeout gate after the V3 pass, ON the capture tray as dc (read-only; prints the verified termination plan, signals nothing):
#   bash closeout_e1_v3scale.sh     -> PASS/FAIL per item, exit 1 on any FAIL
# As closeout_e1_fr.sh, plus: no Nsight CLI, agent or importer of this user left (found by executable path, never by name pattern), every
# registered observer gone, and the observers' records complete.
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc
E1=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN=${E1_V3S_RUN:-$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v3scale}; PASS=e1_v3_scale
fail=0
echo "# closeout $PASS on $(hostname) at $(date -u +%FT%TZ)"
for i in 0 1 2 3; do m=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '); [ "${m:-9999}" -lt 100 ] && echo "PASS  GPU$i memory used ${m} MiB" || { echo "FAIL  GPU$i memory used ${m} MiB"; fail=1; }; done
REG=$RUN/pids/owned.tsv
E1_PID_REGISTRY=$REG bash $E1/checks/e1_process_census.sh; [ $? = 0 ] || fail=1
echo "## termination plan (dry-run; nothing is signalled here)"
python3 -I $E1/checks/e1_pids.py cleanup --registry $REG; prc=$?
case $prc in 0) echo "PASS  registry plan: nothing owned is alive (driver, GPU sampler, import probe, memory sampler, storage sampler)";; 2) echo "FAIL  registry plan: owned processes alive or lookalikes present -> review; to terminate ONLY the verified owned tree: python3 -I $E1/checks/e1_pids.py cleanup --registry $REG --execute --grace 30 --log $RUN/cpu/cleanup_$(date -u +%Y%m%dT%H%M%SZ).jsonl"; fail=1;; *) echo "FAIL  registry plan refused (rc $prc): a registered pid is in this shell's ancestry or another user's -> stop and report"; fail=1;; esac
ns=$(for d in /proc/[0-9]*; do [ "$(stat -c %u $d 2>/dev/null)" = "$(id -u)" ] || continue; e=$(readlink $d/exe 2>/dev/null) || continue; case "$e" in /opt/nvidia/nsight-systems/*) echo "pid ${d#/proc/} $(basename $e) ppid $(awk '/^PPid:/{print $2}' $d/status 2>/dev/null): $(tr '\0' ' ' < $d/cmdline 2>/dev/null | cut -c1-160)";; esac; done)
if [ -z "$ns" ]; then echo "PASS  no Nsight CLI, agent or importer of this user is alive (by executable path)"; else echo "FAIL  Nsight processes of this user alive (the agent and the importer are outside the registered tree; they are never signalled by pattern; an importer is ended only by the packet's procedure, after the durable stream copy is verified):"; echo "$ns" | sed 's/^/      /'; fail=1; fi
LC=$RUN/$PASS/profile/nsys/lifecycle.json
if [ -f $LC ]; then
  python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print('INFO  lifecycle final phase', d['phase'], 'nsys alive', (d.get('nsys') or {}).get('alive'), 'server alive', (d.get('server') or {}).get('alive'), 'gpu released', (d.get('gpu') or {}).get('released_at'), 'preserved', (d.get('stream') or {}).get('preserved_path'))" $LC
  ph=$(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['phase'])" $LC); case "$ph" in report_written|stop_failed) echo "PASS  lifecycle phase $ph is terminal (acceptance is decided by postrun_e1_v3scale.sh; V3 requires report_written)";; *) echo "FAIL  lifecycle phase $ph is not terminal (importer running or hung: stream stays preserved; verified registry cleanup only)"; fail=1;; esac
else echo "INFO  no lifecycle.json for $PASS (nothing ran)"; fi
for f in import_probe.json mem_watch.json storage_watch.json; do
  if [ -f $RUN/$PASS/$f ]; then python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print('INFO  ' + sys.argv[2] + ':', d.get('result') or ('complete=%s samples=%s breaches=%s' % (d.get('complete'), d.get('samples'), len(d.get('breaches', [])))))" $RUN/$PASS/$f $f; else echo "INFO  $f not present"; fi
done
for b in MEMORY-BREACH STORAGE-BREACH; do [ -f $RUN/$PASS/$b ] && { echo "FAIL  $b was written during the run: $(head -c 300 $RUN/$PASS/$b | tr '\n' ' ')"; fail=1; }; done
ls -d /tmp/*.raw-preserve /tmp/nsys-report-*.qdstrm /tmp/v3scale-recovery-* 2>/dev/null | sed 's/^/INFO  stream artefact in \/tmp: /'
echo "INFO  the preserved stream is never removed by a script; postrun_e1_v3scale.sh copies it to durable storage and verifies the copy first"
if [ -f $RUN/$PASS/run.log ]; then
  echo "INFO  $(grep -a 'rc=' $RUN/$PASS/run.log | tail -1)"; echo "INFO  $(grep -a 'EngineDeadError-family' $RUN/$PASS/run.log | tail -1 | cut -c1-200)"
  echo "INFO  run root size $(du -sh $RUN 2>/dev/null | cut -f1); nsys files: $(ls -l $RUN/$PASS/profile/nsys/ 2>/dev/null | awk 'NR>1{printf "%s %s  ", $5, $9}')"
else echo "INFO  no run.log for $PASS (nothing ran)"; fi
echo "INFO  reservation pod (from the laptop): delete only with the GPUs idle, then write $RUN/$PASS/reservation_deleted.json {pod, deleted_observed_at (UTC, %Y-%m-%dT%H:%M:%SZ), gpu_pods_on_node} from kubectl; gate G10/G12 read it"
echo "# closeout result: $([ $fail = 0 ] && echo CLEAN || echo NOT CLEAN)"
exit $fail
