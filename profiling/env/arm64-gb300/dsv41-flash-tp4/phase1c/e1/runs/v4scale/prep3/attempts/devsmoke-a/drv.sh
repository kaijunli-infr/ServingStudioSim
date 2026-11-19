#!/usr/bin/env bash
# V4 driver (NOT to be run without Kevin's explicit go). Exactly one pass, e1_v4_scale, on THIS tray (4 GPUs), as dc, from the pinned
# exp/profiler-lifecycle tree:
#   tmux new -d -s e1v4-scale "E1_REPO=$S/wt-lifecycle bash $E1/run_e1_v4scale.sh"      (through the neutral-path launch wrapper)
# Normal stop. No trigger, no operator control line. The driver registers itself and four independent observers (GPU sampler,
# importer-specific probe in observer mode, host-memory sampler, storage sampler), none of which signals anything, and the run
# supervisor (own registry), which consumes the breach markers and the samplers' heartbeats and gives the one bounded response of
# the packet: the registered release line through the controller during finalisation, the pinned registry cleanup otherwise (latch:
# checks/e1_supervisor_v4.py; the reservation guard is acted on in every state).
# Before any observer or the launcher is started the driver requires: the reservation record written by `e1_reservation_closeout.py record`
# (with the deletion path proof) and the live pod = that record (this host's read-only credential); it then arms the closeout watcher
# (arm_closeout_e1_v4scale.sh: detached, own registry pids/closeout.tsv) and requires it to be evaluating with a fresh heartbeat of the
# deleting side; then the reservation budget (launch rule). Refusals: exit 5, 8, 7. After a refusal 8 or 7 the watcher stays armed and
# permits the reservation's deletion once this driver is gone.
#
# E1_V4S_MODE=rehearsal (CPU-only end-to-end rehearsal, rehearse_e2e_v4scale.sh): same script and same order, but the run root is
# the rehearsal root, CUDA stays hidden, the profile copy gets a rehearsal name and log_dir, and the launcher line starts
# e1_v4scale_mock_launcher.py (mock processes around the real stop controller). The launch record says mode=rehearsal and gate G0
# fails on it, so a rehearsal can never be V4 evidence. Without that variable nothing below reads any rehearsal setting.
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc GB300_ROOT=$S/gb300
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
unset CUDA_VISIBLE_DEVICES UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
MODE=${E1_V4S_MODE:-real}
case "$MODE" in real|rehearsal) ;; *) echo "E1_V4S_MODE must be unset or 'rehearsal' (got '$MODE')"; exit 2;; esac
PASS=e1_v4_scale
[ "${1:-$PASS}" = "$PASS" ] || { echo "this driver runs only $PASS (got '$1')"; exit 2; }
REPO=${E1_REPO:-$S/wt-lifecycle}
E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v4scale
CFG=$E1/configs_v4scale/$PASS/profile.json; TRACE=$E1/traces/e1_v4_scale.csv; INP=$E1/E1-V4-SCALE-INPUTS.json
IMPORTER=/opt/nvidia/nsight-systems/2025.6.3/host-linux-armv8/QdstrmImporter
NAME=dsv41_flash_1017_gb300_tp4_$PASS; RSS_CEIL=128; TMP_CEIL=15; PERS_CEIL=18; RES_GUARD=4680; RES_RUN=4210; GPU_CEIL=2700; KUBECTL="kubectl --request-timeout=15s"
if [ "$MODE" = rehearsal ]; then
  RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v4scale_e2e; NAME=${NAME}_e2e; export CUDA_VISIBLE_DEVICES=""
  # the mock importer is the interpreter the mock launcher runs under: resolve it inside the project environment (uv must not create a venv)
  IMPORTER=$(cd $REPO && source profiling/env/arm64-gb300/gb300.env >/dev/null 2>&1 && uv run --no-sync python -c "import os,sys; print(os.path.realpath(sys.executable))")
  [ -x "$IMPORTER" ] && [ ! -e "$REPO/.venv" ] || { echo "rehearsal: could not resolve the project interpreter cleanly (got '$IMPORTER'; $REPO/.venv exists: $([ -e $REPO/.venv ] && echo yes || echo no))"; exit 2; }
  RSS_CEIL=${E1_V4S_REHEARSAL_IMPORTER_RSS_GB:-$RSS_CEIL}; TMP_CEIL=${E1_V4S_REHEARSAL_TEMPORARY_GB:-$TMP_CEIL}; PERS_CEIL=${E1_V4S_REHEARSAL_PERSISTENT_GB:-$PERS_CEIL}; RES_GUARD=${E1_V4S_REHEARSAL_RESERVATION_GUARD_S:-$RES_GUARD}; RES_RUN=${E1_V4S_REHEARSAL_RUN_BUDGET_S:-$RES_RUN}; GPU_CEIL=${E1_V4S_REHEARSAL_SUPERVISOR_CEILING_S:-$GPU_CEIL}
  KUBECTL=${E1_V4S_REHEARSAL_KUBECTL:-}; [ -n "$KUBECTL" ] || { echo "rehearsal: E1_V4S_REHEARSAL_KUBECTL (the stand-in for kubectl) is not set; a rehearsal never talks to the cluster"; exit 2; }
  echo "REHEARSAL MODE: run root $RUN, CUDA hidden, mock launcher, importer executable $IMPORTER, ceilings importer $RSS_CEIL GiB / temporary $TMP_CEIL GB / persistent $PERS_CEIL GB; reservation guard $RES_GUARD s, run budget $RES_RUN s, supervisor GPU-held ceiling $GPU_CEIL s"
fi
[ -f "$CFG" ] || { echo "no config $CFG"; exit 2; }
mkdir -p $RUN/$PASS
# defence in depth, immediately before launch: pinned executable, registered 224-row trace, no delayed start, explicit GPU-held ceiling
[ "$(sha256sum $REPO/alignment/load_generator/req-frontend/target/release/session_runner 2>/dev/null | cut -d' ' -f1)" = "$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['sha256'])" $E1/BINARY-PROVENANCE.json)" ] || { echo "session_runner is not the pinned executable; not starting"; exit 6; }
[ "$(tail -n +2 $TRACE | grep -c .)" = 224 ] && [ "$(sha256sum $TRACE | cut -d' ' -f1)" = "$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['trace']['sha256'])" $INP)" ] && grep -q "\"path\": \"$TRACE\"" $CFG && ! grep -q capture_start_delay_seconds $CFG && grep -q '"gpu_held_ceiling_s": 2700.0' $CFG || { echo "trace/config guard failed; not starting"; exit 4; }
cp $CFG $RUN/$PASS/profile.json; cp $TRACE $RUN/$PASS/trace.csv
[ "$MODE" = rehearsal ] && python3 -I -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); d['name']=sys.argv[2]; d['log_dir']=sys.argv[3]; json.dump(d, open(p,'w'), indent=1)" $RUN/$PASS/profile.json $NAME $RUN/$PASS/profile
REG=$RUN/pids/owned.tsv; python3 -I $E1/checks/e1_pids.py register --registry $REG --label driver:$PASS $$ | tee -a $RUN/$PASS/run.log
# reservation record (written before the launch by `e1_reservation_closeout.py record` with the deleting credential): this node, Running, UID,
# creation time, deletion path proof; then the live pod must be exactly that record (this host's read-only credential)
python3 -I -c "import json,sys,socket; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get('node')==socket.gethostname() and r.get('phase')=='Running' and r.get('creation_timestamp') and r.get('uid') and r.get('pod') and r.get('namespace') and (r.get('deletion_path_proof') or {}).get('ok') is True else 1)" $RUN/$PASS/reservation.json 2>/dev/null || { echo "reservation.json missing, not Running, for another node, or without UID / creation time / deletion path proof; not starting" | tee -a $RUN/$PASS/run.log; exit 5; }
python3 -I $E1/checks/e1_reservation_closeout.py verify --reservation $RUN/$PASS/reservation.json --kubectl "$KUBECTL" | tee -a $RUN/$PASS/run.log
[ "${PIPESTATUS[0]}" = 0 ] || { echo "LAUNCH REFUSED: the live reservation pod is not the recorded one, or could not be read; nothing was started" | tee -a $RUN/$PASS/run.log; exit 5; }
# bounded closeout and deletion: arm the watcher (detached, own registry; it waits for the end of this run) and require it to be evaluating
# and the deleting side's heartbeat to be fresh; without both, a run could end with nobody able to delete the reservation inside its 90 minutes
E1_V4S_MODE=$MODE bash $E1/arm_closeout_e1_v4scale.sh 2>&1 | tee -a $RUN/$PASS/run.log
[ "${PIPESTATUS[0]}" = 0 ] || { echo "LAUNCH REFUSED: the closeout watcher could not be armed; no observer and no launcher was started" | tee -a $RUN/$PASS/run.log; exit 8; }
for i in $(seq 1 15); do python3 -I $E1/checks/e1_reservation_closeout.py armed --record $RUN/$PASS/reservation_closeout.json --registry $RUN/pids/closeout.tsv --max-age-s 30 --deleter-max-age-s 60 > $RUN/$PASS/closeout_armed.txt 2>&1 && break; sleep 3; done
cat $RUN/$PASS/closeout_armed.txt | tee -a $RUN/$PASS/run.log
grep -q "^PASS  closeout watcher armed" $RUN/$PASS/closeout_armed.txt || { echo "LAUNCH REFUSED: the closeout watcher is not evaluating or the deleter's heartbeat is missing (the deleting side must be started before the driver); no observer and no launcher was started; the watcher stays armed" | tee -a $RUN/$PASS/run.log; exit 8; }
# reservation budget, launch rule (E1-V4-TIMELINE.md): start nothing unless the time left until the supervisor's reservation guard
# ($RES_GUARD s after the reservation's creation) is at least the registered run budget ($RES_RUN s = this check to the launcher's start
# + GPU-held ceiling + release + CPU-only processing after the release). No observer and no launcher has been started at this point.
python3 -I $E1/checks/e1_reservation_guard.py eligible --reservation $RUN/$PASS/reservation.json --guard-s $RES_GUARD --run-budget-s $RES_RUN | tee -a $RUN/$PASS/run.log
[ "${PIPESTATUS[0]}" = 0 ] || { echo "LAUNCH REFUSED: the remaining reservation budget is insufficient (or the reservation's age is unknown); nothing was started" | tee -a $RUN/$PASS/run.log; exit 7; }
# launch source record (gate G0): commit, pinned trees, gitlinks, tracked changes, descent from the pinned code commit
python3 -I - "$REPO" "$E1/V4S-PIN.json" "$MODE" > $RUN/$PASS/launch_source.json <<'PY'
import json, subprocess, sys
repo, pin = sys.argv[1], json.load(open(sys.argv[2]))
g = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True)
print(json.dumps({"mode": sys.argv[3], "commit": g("rev-parse", "HEAD").stdout.strip(), "branch": g("branch", "--show-current").stdout.strip(),
    "trees": {k: g("rev-parse", f"HEAD:{k}").stdout.strip() for k in ("alignment", "launcher")},
    "gitlinks": {p: (g("ls-tree", "HEAD", p).stdout.split() + [None, None, None])[2] for p in pin["code"]["gitlinks"]},
    "tracked_changes": len([l for l in g("status", "--short", "--untracked-files=no").stdout.splitlines() if l.strip()]),
    "descends_from_pinned_commit": g("merge-base", "--is-ancestor", pin["code"]["commit"], "HEAD").returncode == 0}, indent=1))
PY
REP=$RUN/$PASS/profile/nsys/$NAME.nsys-rep; STOPF=$RUN/$PASS/samplers.stop; rm -f $STOPF
spawn() { python3 -I $E1/checks/e1_pids.py spawn --registry $REG --label "$1" -- "${@:2}" | tee -a $RUN/$PASS/run.log; }
lastpid() { grep -ao '"pid": [0-9]*' $RUN/$PASS/run.log | tail -1 | tr -dc 0-9; }
# independent GPU-state sampler (gate G6): every 2 s, exits by itself 60 s after it has seen the release, or, if the GPUs were never
# held, 10 samples after the stop file appears (same observation and record as the V2 sampler)
spawn gpuwatch:$PASS python3 -I $E1/checks/e1_gpu_watch_v3.py --out $RUN/$PASS/gpu_observations.jsonl --interval 2 --max-s 3300 --stop-after-released 30 --stop-file $STOPF --after-stop-samples 10; WPID=$(lastpid)
# importer-specific observer (gates G4, G5): observer mode, no control file, no trigger; exits when the importer is gone and the report has settled
spawn importprobe:$PASS python3 -I $E1/checks/e1_import_probe.py --importer-exe $IMPORTER --nsys-dir $RUN/$PASS/profile/nsys --report $REP --out $RUN/$PASS/import_probe.json --samples $RUN/$PASS/import_probe.jsonl --log $RUN/$PASS/import_probe.log --interval 0.5 --min-progress 2 --max-s 3300 --stop-file $STOPF; PPID_=$(lastpid)
# host-memory sampler with the bounded policy (gate G11): records and writes MEMORY-BREACH; signals nothing
spawn memwatch:$PASS python3 -I $E1/checks/e1_mem_watch.py --out $RUN/$PASS/mem_observations.jsonl --summary $RUN/$PASS/mem_watch.json --importer-exe $IMPORTER --nsys-dir $RUN/$PASS/profile/nsys --report $REP --min-available-gb 64 --max-importer-rss-gb $RSS_CEIL --breach-file $RUN/$PASS/MEMORY-BREACH --interval 1 --max-s 5400 --stop-file $STOPF; MPID=$(lastpid)
# storage sampler (gate G11): temporary = every stream artefact and recovery scratch in /tmp; persistent = the run root
spawn storagewatch:$PASS python3 -I $E1/checks/e1_storage_watch.py --out $RUN/$PASS/storage_observations.jsonl --summary $RUN/$PASS/storage_watch.json --group "temporary=/tmp/nsys-report-*.qdstrm,/tmp/*.raw-preserve,/tmp/v4scale-recovery-*" --group "persistent=$RUN" --ceiling-gb temporary=$TMP_CEIL --ceiling-gb persistent=$PERS_CEIL --breach-file $RUN/$PASS/STORAGE-BREACH --interval 2 --max-s 5400 --stop-file $STOPF; SPID=$(lastpid)
# run supervisor, in its OWN registry (so that the cleanup of $REG it may run is not a cleanup of its own ancestry): consumes
# MEMORY-BREACH / STORAGE-BREACH and the samplers' heartbeats, and enforces the GPU-held ceiling before the stop request and the
# reservation guard ($RES_GUARD s after the reservation's creation); one response; sends no signal itself
SREG=$RUN/pids/supervisor.tsv
python3 -I $E1/checks/e1_pids.py spawn --registry $SREG --label supervisor:$PASS -- python3 -I $E1/checks/e1_supervisor_v4.py --run $RUN/$PASS --registry $REG --out $RUN/$PASS/supervisor.json --reservation $RUN/$PASS/reservation.json --reservation-guard-s $RES_GUARD --control-line "$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['supervisor']['control_line'])" $INP)" --gpu-held-ceiling-s $GPU_CEIL --ack-s 10 --stale-s 60 --cleanup-grace 30 --interval 1 --max-s 5400 --stop-file $STOPF | tee -a $RUN/$PASS/run.log; UPID=$(lastpid)
cd $REPO
source profiling/env/arm64-gb300/gb300.env
if [ "$MODE" = real ]; then unset CUDA_VISIBLE_DEVICES; else export CUDA_VISIBLE_DEVICES=""; fi
echo "[$PASS] start $(date -u +%FT%TZ) on $(hostname) launcher_tree=$REPO repo=$(git rev-parse HEAD) tree_alignment=$(git rev-parse HEAD:alignment | cut -c1-12) cfg_sha=$(sha256sum $CFG | cut -c1-16) trace_sha=$(sha256sum $TRACE | cut -c1-16) session_runner=$(sha256sum alignment/load_generator/req-frontend/target/release/session_runner | cut -c1-16)" | tee -a $RUN/$PASS/run.log
BIN=alignment/load_generator/req-frontend/target/release/session_runner; BS0=$(sha256sum $BIN | cut -d' ' -f1); ST0=$(stat -c '%i %Y %s' $BIN)
# prebuilt load-generator contract, as V2: exact pinned bytes, never cargo, build-tool guard armed, executable hashed again at the end
if [ "$MODE" = real ]; then
  uv run --no-sync python -m launcher alignment profile "$RUN/$PASS/profile.json" --frontend-provenance $E1/BINARY-PROVENANCE.json 2>&1 | tee -a $RUN/$PASS/run.log
  rc=${PIPESTATUS[0]}
else
  export CUDA_VISIBLE_DEVICES="" E1_V4S_MODE=rehearsal
  uv run --no-sync python $E1/e1_v4scale_mock_launcher.py launch "$RUN/$PASS/profile.json" --frontend-provenance $E1/BINARY-PROVENANCE.json 2>&1 | tee -a $RUN/$PASS/run.log
  rc=${PIPESTATUS[0]}
fi
BS1=$(sha256sum $BIN | cut -d' ' -f1); ST1=$(stat -c '%i %Y %s' $BIN); nb=$(grep -ac "^ *Compiling " $RUN/$PASS/run.log)
printf '{"binary": "%s", "sha256_before_launcher": "%s", "sha256_after_launcher": "%s", "unchanged": %s, "cargo_compiling_lines_in_run_log": %s, "stat_before": "%s", "stat_after": "%s"}\n' "$BIN" "$BS0" "$BS1" "$([ "$BS0" = "$BS1" ] && echo true || echo false)" "$nb" "$ST0" "$ST1" > $RUN/$PASS/binary_check.json
echo "[$PASS] session_runner before/after the launcher: ${BS0:0:16} / ${BS1:0:16} $([ "$BS0" = "$BS1" ] && [ "$nb" = 0 ] && echo "(unchanged, no rebuild)" || echo "(REBUILT OR CHANGED DURING THE RUN: gate G0 will FAIL)")" | tee -a $RUN/$PASS/run.log
echo "[$PASS] rc=$rc end $(date -u +%FT%TZ)" | tee -a $RUN/$PASS/run.log
python3 -I $E1/checks/e1_gpu_release_census.py --registry $REG --label t0 --out $RUN/$PASS/gpu_release_census_t0.json | tee -a $RUN/$PASS/run.log
LC=$RUN/$PASS/profile/nsys/lifecycle.json
[ -f $LC ] && python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print('[lifecycle] phase', d['phase'], '| http', d['http'], '| report', d['report'], '| gpu', d['gpu'], '| escalations', len(d['escalations']), '| final stream', (d['stream'].get('final') or {}), '| control', [(c['command'],c['result'],c['phase']) for c in d['control_commands']], '| actions', [(a.get('pid'),a.get('action'),a.get('reason')) for a in d['actions']], '| warnings', [w['message'] for w in d['warnings']])" $LC | tee -a $RUN/$PASS/run.log
for f in $(grep -ao "Using FlashInfer autotune cache file: [^ ]*" $RUN/$PASS/profile/vllm/*_server.log 2>/dev/null | awk '{print $NF}' | sort -u); do
  [ -f "$f" ] && cp -p "$f" $RUN/$PASS/autotune_configs.$(basename $(dirname $f) | cut -c1-12).json
done
SLOG=$(ls $RUN/$PASS/profile/vllm/*_server.log 2>/dev/null | head -1)
if [ -n "$SLOG" ]; then
  sd=$(grep -an "\[shutdown\].*trigger received" "$SLOG" | head -1 | cut -d: -f1); sd=${sd:-0}
  ede=$(grep -an "EngineDeadError\|EngineCore encountered an issue\|output_handler failed" "$SLOG" | cut -d: -f1 | tr "\n" " ")
  before=0; for l in $ede; do [ "$sd" = 0 ] || [ "$l" -lt "$sd" ] && before=$((before+1)); done
  echo "[$PASS] shutdown trigger at server-log line ${sd}; EngineDeadError-family lines: ${ede:-none}; before shutdown: $before $([ $before = 0 ] && echo "(accepted)" || echo "(ABORT CONDITION)")" | tee -a $RUN/$PASS/run.log
fi
sleep 20
# (the launcher's export and parse after the GPU release take about 20 min at this size: the memory and storage samplers and the
# supervisor run for up to 5400 s, the reservation guard ends a run that is still going 4680 s after the reservation's creation)
# stop the memory and storage samplers through their stop file and let every registered observer finish (bounded), so the final
# census sees no owned process; nothing is signalled here
touch $STOPF
for p in "$WPID" "$PPID_" "$MPID" "$SPID" "$UPID"; do for i in $(seq 1 75); do [ -n "$p" ] && [ -d /proc/$p ] || break; sleep 2; done; done
python3 -I $E1/checks/e1_gpu_release_census.py --registry $REG --label t20 --out $RUN/$PASS/gpu_release_census_t20.json | tee -a $RUN/$PASS/run.log
E1_PID_REGISTRY=$REG bash $E1/checks/e1_process_census.sh 2>&1 | grep -v "tmux sessions" | tee -a $RUN/$PASS/run.log
python3 -I $E1/checks/e1_pids.py plan --registry $SREG 2>&1 | grep -E "summary|OWNED|GONE|REFUSED" | sed 's/^/[supervisor registry] /' | tee -a $RUN/$PASS/run.log
exit $rc
