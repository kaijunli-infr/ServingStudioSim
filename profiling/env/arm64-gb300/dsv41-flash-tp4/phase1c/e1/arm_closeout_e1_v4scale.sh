#!/usr/bin/env bash
# Arms the bounded closeout of the V4 reservation ON the capture tray as dc. Called by the driver (run_e1_v4scale.sh) after its preflight and
# before any observer or the launcher exists; it can also be run by hand through a neutral-path wrapper when a driver never got that far
# (then the watcher permits the deletion once no launch is possible any more):
#   bash arm_closeout_e1_v4scale.sh
# Needs <run root>/e1_v4_scale/reservation.json (written on the laptop by `checks/e1_reservation_closeout.py record`, copied here).
# It (1) verifies with this host's read-only credential that the live pod is exactly that record, (2) writes two neutral-path helper
# scripts for the deleting side (permit.py prints the watcher's record and touches the deleter heartbeat; publish.py stores the
# deleter's record as reservation_deleted.json), (3) starts the watcher detached, registered in its OWN registry pids/closeout.tsv.
# The watcher permits the deletion only after the run has ended (registered tree gone, two clean GPU censuses, ownership verified),
# observes the deletion and writes the verdict. It deletes nothing and signals nothing. Nothing here uses a GPU.
# E1_V4S_MODE=rehearsal: rehearsal run root, stand-in kubectl (E1_V4S_REHEARSAL_KUBECTL), rehearsal values for the ceiling and tail.
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc; export PATH=$HOME/.local/bin:$PATH
MODE=${E1_V4S_MODE:-real}
case "$MODE" in real|rehearsal) ;; *) echo "E1_V4S_MODE must be unset or 'rehearsal' (got '$MODE')"; exit 2;; esac
E1=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); PASS=e1_v4_scale; INP=$E1/E1-V4-SCALE-INPUTS.json
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v4scale; NEUTRAL=$S/scratch/lifecycle/v4closeout
val() { python3 -I -c "import json,sys; d=json.load(open(sys.argv[1]))['reservation']; print(eval(sys.argv[2], {}, d))" $INP "$1"; }
CEIL=$(val hard_ceiling_s); GUARD=$(val supervisor_guard_s); BUDGET=$(val run_budget_total_s); TAIL=$(val "closeout['watcher_tail_s']"); INTERVAL=$(val "closeout['interval_s']"); MARGIN=$(val "closeout['never_launched_margin_s']")
KUBECTL="kubectl --request-timeout=15s"; ATTR=""
if [ "$MODE" = rehearsal ]; then
  RUN=${RUN}_e2e; NEUTRAL=${NEUTRAL}_e2e
  KUBECTL=${E1_V4S_REHEARSAL_KUBECTL:-}; [ -n "$KUBECTL" ] || { echo "rehearsal: E1_V4S_REHEARSAL_KUBECTL is not set; a rehearsal never talks to the cluster"; exit 2; }
  CEIL=${E1_V4S_REHEARSAL_RESERVATION_CEILING_S:-$CEIL}; GUARD=${E1_V4S_REHEARSAL_RESERVATION_GUARD_S:-$GUARD}; BUDGET=${E1_V4S_REHEARSAL_RUN_BUDGET_S:-$BUDGET}; TAIL=${E1_V4S_REHEARSAL_CLOSEOUT_TAIL_S:-$TAIL}
  for r in ${E1_V4S_REHEARSAL_ATTRIBUTE_REGISTRIES:-}; do ATTR="$ATTR --attribute-registry $r"; done
  echo "REHEARSAL MODE: run root $RUN, stand-in kubectl, ceiling $CEIL s, guard $GUARD s, run budget $BUDGET s, tail $TAIL s"
fi
LATEST=$((GUARD - BUDGET)); D=$RUN/$PASS
[ -f $D/reservation.json ] || { echo "no reservation record $D/reservation.json; not arming"; exit 5; }
[ ! -e $RUN/pids/closeout.tsv ] && [ ! -e $D/reservation_closeout.json ] || { echo "a closeout watcher was already armed for this run root ($RUN/pids/closeout.tsv or $D/reservation_closeout.json exists); not arming a second one"; exit 8; }
python3 -I $E1/checks/e1_reservation_closeout.py verify --reservation $D/reservation.json --kubectl "$KUBECTL" || { echo "the live reservation pod is not the recorded one, or could not be read; not arming"; exit 5; }
mkdir -p $NEUTRAL $RUN/pids
cat > $NEUTRAL/permit.py <<PY
# neutral-path helper for the deleting side: deleter heartbeat + the closeout watcher's record (read-only otherwise)
import pathlib, sys
d = pathlib.Path("$D")
(d / "deleter.heartbeat").touch()
sys.stdout.write((d / "reservation_closeout.json").read_text())
PY
cat > $NEUTRAL/publish.py <<PY
# neutral-path helper for the deleting side: stores the deleter's record (stdin) as reservation_deleted.json, atomically
import json, os, pathlib, sys
d = pathlib.Path("$D")
rec = json.load(sys.stdin)
tmp = d / "reservation_deleted.json.tmp"
tmp.write_text(json.dumps(rec, indent=1) + "\n")
os.replace(tmp, d / "reservation_deleted.json")
PY
python3 -I $E1/checks/e1_pids.py spawn --registry $RUN/pids/closeout.tsv --label closeout:$PASS --log $D/closeout_watch.log -- python3 -I $E1/checks/e1_reservation_closeout.py watch --run $D --registry $RUN/pids/owned.tsv \
  --attribute-registry $RUN/pids/supervisor.tsv $ATTR --out $D/reservation_closeout.json --launch-latest-age-s $LATEST --never-launched-margin-s $MARGIN --ceiling-s $CEIL --tail-s $TAIL --interval $INTERVAL \
  --deleter-heartbeat $D/deleter.heartbeat --kubectl "$KUBECTL" || { echo "the watcher could not be started"; exit 1; }
echo "closeout watcher armed: registry $RUN/pids/closeout.tsv, record $D/reservation_closeout.json, launch possible until reservation age $LATEST s, deletion deadline $CEIL s after the pod's creation (+ $TAIL s observation tail)"
echo "deleting side (laptop): permit command 'python3 -I $NEUTRAL/permit.py', publish command 'python3 -I $NEUTRAL/publish.py' (through ssh to this host)"
