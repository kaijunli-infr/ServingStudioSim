#!/usr/bin/env bash
# CPU-only end-to-end rehearsal of the V4 execution path (NOT V4 evidence; no GPU is used; nothing real is launched). ON the capture
# tray as dc, through a neutral-path wrapper:
#   E1_REPO=$S/wt-lifecycle bash $E1/rehearse_e2e_v4scale.sh [output dir, default runs/v4scale/e2e] [scenario ...]
# Runs the ACTUAL driver (run_e1_v4scale.sh, E1_V4S_MODE=rehearsal), closeout, post-run and archive scripts. In rehearsal mode the
# driver starts e1_v4scale_mock_launcher.py instead of the launcher: mock processes (Nsight CLI, API server, engine, workers,
# detached agent, importer) around the REAL stop controller, with the real observers, the real supervisor, the real registry and
# the real prebuilt-executable contract. Scenarios, each with expectations registered in e1_v4scale_e2e_check.py before the run:
#   success         launch → closeout → post-run → archive; then a second post-run with a failing importer (recovery failure)
#   ambiguous       two processes match the importer criteria: recorded INCONCLUSIVE, the run continues under the other monitors
#   memory_breach   the importer exceeds the (rehearsal) memory ceiling during the import: marker → supervisor → registered
#                   release line → the controller's identity-verified server-only stop; the importer is untouched and finishes
#   storage_breach  the run root exceeds the (rehearsal) ceiling during the capture: marker → supervisor → pinned registry cleanup
#   reservation_guard  the (rehearsal) reservation guard, 420 s after the reservation record's creation, passes during a 900 s mock
#                   capture: supervisor → pinned registry cleanup (the driver's preflight must have finished within those 420 s)
# Every output is kept, including failed attempts. The rehearsal run root and the mock stream are removed after each scenario.
set -u
export HOME=/mnt/shared/home/dc CUDA_VISIBLE_DEVICES=""; S=$HOME/workspace/serving-studio
export PATH=$HOME/.local/bin:$PATH GB300_ROOT=$S/gb300
REPO=${E1_REPO:-$S/wt-lifecycle}; E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1; PASS=e1_v4_scale
OUT=${1:-$E1/runs/v4scale/e2e}; shift 2>/dev/null || true
SCENARIOS=${*:-success ambiguous memory_breach storage_breach reservation_guard}
R=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v4scale_e2e; PRES=/tmp/dsv41_flash_1017_gb300_tp4_e1_v4_scale_e2e.raw-preserve
BIN=$REPO/alignment/load_generator/req-frontend/target/release/session_runner
PINSHA=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['sha256'])" $E1/BINARY-PROVENANCE.json)
mkdir -p $OUT; fail=0
ok()  { echo "PASS  $*"; }
bad() { echo "FAIL  $*"; fail=1; }
echo "# V4 end-to-end rehearsal $(date -u +%FT%TZ) on $(hostname); tree $REPO at $(git -C $REPO rev-parse HEAD) ($(git -C $REPO status --short --untracked-files=no | wc -l) modified tracked file(s)); scenarios: $SCENARIOS"
echo "gpu before (MiB / compute processes): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')/ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
b0=$(sha256sum $BIN | cut -d' ' -f1); st0=$(stat -c '%i %Y %s' $BIN); [ "$b0" = "$PINSHA" ] && ok "load-generator executable before the rehearsal = pinned $PINSHA" || bad "load-generator executable is $b0, pinned $PINSHA"
touch $OUT/.e2e_started
[ -e $R ] || [ -e $PRES ] && { bad "rehearsal run root $R or mock preserve directory $PRES exists: not touching it"; echo "# result: E2E REHEARSAL NOT RUN"; exit 2; }
T=$OUT/tools; rm -rf $T; PYEXE=$(cd $REPO && source profiling/env/arm64-gb300/gb300.env >/dev/null 2>&1 && uv run --no-sync python -c "import os,sys; print(os.path.realpath(sys.executable))")
[ -x "$PYEXE" ] && [ ! -e "$REPO/.venv" ] && ok "project interpreter $PYEXE (no virtual environment created in the worktree)" || { bad "could not resolve the project interpreter cleanly ('$PYEXE'; $REPO/.venv exists: $([ -e $REPO/.venv ] && echo yes || echo no))"; echo "# result: E2E REHEARSAL NOT RUN"; exit 2; }
python3 -I $E1/e1_v4scale_rehearsal.py tools --out $T --python $PYEXE > $OUT/tools.json && ok "stand-in importer / exporter and rehearsal inputs written (kernel-row band scaled to the stand-in exporter; every other limit is the registered one)" || bad "rehearsal tools"
for sc in $SCENARIOS; do
  O=$OUT/$sc; rm -rf $O; mkdir -p $O
  echo "## scenario $sc ($(date -u +%T))"
  mkdir -p $R/$PASS
  python3 -I -c "import json,socket,sys,time; json.dump({'pod': 'e2e-rehearsal-no-pod', 'node': socket.gethostname(), 'phase': 'Running', 'creation_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'note': 'REHEARSAL: no reservation pod exists; nothing uses a GPU'}, open(sys.argv[1], 'w'), indent=1)" $R/$PASS/reservation.json
  case $sc in
    success)        envs="E1_V4S_MOCK_SCENARIO=success";;
    ambiguous)      envs="E1_V4S_MOCK_SCENARIO=ambiguous";;
    memory_breach)  envs="E1_V4S_MOCK_SCENARIO=importer_ballast E1_V4S_MOCK_BALLAST_MB=160 E1_V4S_MOCK_IMPORT_S=20 E1_V4S_REHEARSAL_IMPORTER_RSS_GB=0.0625";;
    storage_breach) envs="E1_V4S_MOCK_SCENARIO=storage_ballast E1_V4S_MOCK_STORAGE_BALLAST_MB=64 E1_V4S_MOCK_CAPTURE_S=40 E1_V4S_REHEARSAL_PERSISTENT_GB=0.032";;
    reservation_guard) envs="E1_V4S_MOCK_SCENARIO=success E1_V4S_MOCK_CAPTURE_S=900 E1_V4S_REHEARSAL_RESERVATION_GUARD_S=420";;
    *) bad "unknown scenario $sc"; continue;;
  esac
  env $envs E1_V4S_MODE=rehearsal E1_V4S_MOCK_NSYS=$T/nsys E1_REPO=$REPO bash $E1/run_e1_v4scale.sh > $O/driver.log 2>&1; drc=$?
  echo "driver exit code $drc; $(grep -c '^PASS' $R/$PASS/preflight.log 2>/dev/null) PASS / $(grep -c '^FAIL' $R/$PASS/preflight.log 2>/dev/null) FAIL in the driver's preflight; $(grep -a 'REHEARSAL MODE\|\[mock\] stop ended\|rc=' $O/driver.log | tr '\n' ';' | cut -c1-300)"
  for i in $(seq 1 45); do [ -z "$(awk -F'\t' '$1 ~ /^[0-9]+$/ {print $1}' $R/pids/owned.tsv $R/pids/supervisor.tsv 2>/dev/null | while read p; do [ -d /proc/$p ] && echo $p; done)" ] && break; sleep 2; done
  E1_V4S_RUN=$R bash $E1/closeout_e1_v4scale.sh > $O/closeout.log 2>&1; crc=$?
  echo "closeout exit code $crc: $(grep '^FAIL\|# closeout result' $O/closeout.log | tr '\n' ';' | cut -c1-400)"
  prc=""; arc=""
  if [ $sc != storage_breach ] && [ $sc != reservation_guard ]; then
    python3 -I -c "import json,sys,time; json.dump({'pod': 'e2e-rehearsal-no-pod', 'deleted_observed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'gpu_pods_on_node': 0, 'note': 'REHEARSAL'}, open(sys.argv[1], 'w'), indent=1)" $R/$PASS/reservation_deleted.json
    E1_V4S_MODE=rehearsal E1_REPO=$REPO E1_V4S_RUN=$R E1_V4S_TABLES=$R/tables E1_V4S_INPUTS=$T/inputs.json E1_CLOSEOUT_RC=$crc E1_V4S_MOCK_IMPORTER=$T/QdstrmImporter E1_V4S_MOCK_NSYS=$T/nsys E1_V4S_MOCK_IMPORTER_EXE=$PYEXE bash $E1/postrun_e1_v4scale.sh > $O/postrun.log 2>&1; prc=$?
    echo "post-run exit code $prc: $(grep '^# V4 gates\|V4 ACCEPTANCE' $O/postrun.log | tr '\n' ';' | cut -c1-300)"
    cp $R/cpu/V4S-GATES.json $O/V4S-GATES.json 2>/dev/null; cp $R/cpu/V4S-GATES.txt $O/V4S-GATES.txt 2>/dev/null
  fi
  extra=""
  if [ $sc = success ]; then
    E1_V4S_RUN=$R bash $E1/archive_e1_v4scale.sh $O $R.archive > $O/archive.log 2>&1; arc=$?
    echo "archive exit code $arc: $(grep '^FAIL\|# archive result' $O/archive.log | tr '\n' ';' | cut -c1-300)"
    cp $R.archive/SHA256SUMS $O/ARCHIVE-SHA256SUMS 2>/dev/null; chmod -R u+w $R.archive 2>/dev/null; rm -rf $R.archive
    E1_V4S_MODE=rehearsal E1_REPO=$REPO E1_V4S_RUN=$R E1_V4S_TABLES=$R/tables E1_V4S_INPUTS=$T/inputs.json E1_CLOSEOUT_RC=$crc E1_V4S_MOCK_IMPORTER=$T/QdstrmImporter-failing E1_V4S_MOCK_NSYS=$T/nsys E1_V4S_MOCK_IMPORTER_EXE=$PYEXE bash $E1/postrun_e1_v4scale.sh > $O/postrun_failing_importer.log 2>&1; frc=$?
    echo "second post-run with a failing importer: exit code $frc: $(grep '^# V4 gates' $O/postrun_failing_importer.log | cut -c1-200)"
    cp $R/cpu/V4S-GATES.json $O/V4S-GATES-failing-importer.json 2>/dev/null; extra="--gates-failed-recovery $O/V4S-GATES-failing-importer.json --archive-rc $arc"
  fi
  python3 -I $E1/e1_v4scale_e2e_check.py --scenario $sc --root $R --out $O --driver-rc $drc --closeout-rc $crc ${prc:+--postrun-rc $prc --gates $O/V4S-GATES.json} $extra | tee $O/E2E-CHECK.txt
  [ "${PIPESTATUS[0]}" = 0 ] && ok "scenario $sc: every registered expectation met" || bad "scenario $sc: expectations not met (see $O/E2E-CHECK.txt)"
  # keep the small records; remove the rehearsal run root and the mock stream (both created by this scenario)
  mkdir -p $O/records; for f in run.log preflight.log launch_source.json binary_check.json reservation.json reservation_deleted.json import_probe.json import_probe.log mem_watch.json storage_watch.json supervisor.json MEMORY-BREACH STORAGE-BREACH gpu_release_census_t0.json gpu_release_census_t20.json profile.json; do [ -f $R/$PASS/$f ] && cp $R/$PASS/$f $O/records/; done
  for f in profile/nsys/lifecycle.json profile/nsys/lifecycle.control profile/load_generator/frontend_binary.json; do [ -f $R/$PASS/$f ] && cp $R/$PASS/$f $O/records/; done
  cp $R/$PASS/profile/vllm/*_server.log $O/records/mock_server.log 2>/dev/null; cp $R/pids/*.tsv $O/records/ 2>/dev/null; cp $R/cpu/supervisor_cleanup.jsonl $O/records/ 2>/dev/null
  for f in gpu_observations mem_observations storage_observations import_probe; do [ -f $R/$PASS/$f.jsonl ] && { head -c 200000 $R/$PASS/$f.jsonl > $O/records/$f.head.jsonl; wc -l < $R/$PASS/$f.jsonl > $O/records/$f.lines; }; done
  # never remove the run root while a process registered in it is alive: end it through the pinned verified cleanup of its own registry
  for reg in $R/pids/owned.tsv $R/pids/supervisor.tsv; do
    [ -f $reg ] || continue
    if [ -n "$(awk -F'\t' '$1 ~ /^[0-9]+$/ {print $1}' $reg | while read p; do [ -d /proc/$p ] && echo $p; done)" ]; then
      bad "scenario $sc: registered process(es) still alive in $(basename $reg) after the scenario; ending them through the verified registry cleanup"
      python3 -I $E1/checks/e1_pids.py cleanup --registry $reg --execute --grace 10 --kill9 --log $O/records/leftover_cleanup.jsonl 2>&1 | grep -E "ACTION|RESULT" | cut -c1-200
    fi
  done
  case "$R" in */logs/dsv41_flash_gb300_tp4_e1_v4scale_e2e) rm -rf $R;; esac
  case "$PRES" in /tmp/dsv41_flash_1017_gb300_tp4_e1_v4_scale_e2e.raw-preserve) rm -rf $PRES;; esac
  rm -f /tmp/nsys-report-e2e*.qdstrm; rm -rf /tmp/v4scale-recovery-$PASS
done
echo "## state after the rehearsal"
b1=$(sha256sum $BIN | cut -d' ' -f1); st1=$(stat -c '%i %Y %s' $BIN); [ "$b1" = "$PINSHA" ] && [ "$st0" = "$st1" ] && ok "load-generator executable after the rehearsal = pinned $PINSHA, same inode/mtime/size" || bad "load-generator executable changed ($b1; $st0 → $st1)"
nf=$(find $REPO/alignment/load_generator/req-frontend/target -newer $OUT/.e2e_started -type f 2>/dev/null | wc -l); [ "$nf" = 0 ] && ok "no file under the load generator's build directory was written during the rehearsal (no build tool ran)" || bad "$nf file(s) under the build directory are newer than the rehearsal start"
g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '); c=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); echo "gpu after (MiB / compute processes): $g/ $c"
[ "$c" = 0 ] && ok "no GPU compute process after the rehearsal" || bad "GPU compute processes present after the rehearsal"
lo=$(ps -u $(id -u) -o pid,args | grep "[e]1_v4scale_mock_launcher.py" | wc -l); [ "$lo" = 0 ] && ok "no mock process left" || bad "$lo mock process(es) left: $(ps -u $(id -u) -o pid,args | grep '[e]1_v4scale_mock_launcher.py' | cut -c1-200)"
ls -d $R $PRES /tmp/v4scale-recovery-* 2>/dev/null && bad "rehearsal artefacts left" || ok "rehearsal run root, mock preserve directory and scratch removed"
rm -f $OUT/.e2e_started
echo "# result: $([ $fail = 0 ] && echo E2E REHEARSAL EXPECTATIONS MET || echo E2E REHEARSAL EXPECTATIONS NOT MET)"
exit $fail
