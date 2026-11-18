#!/usr/bin/env bash
# V2 forced-release driver (NOT to be run without Kevin's explicit go). Exactly one pass, e1_v2_forced_release, on THIS tray (4 GPUs), as dc, from the
# pinned exp/profiler-lifecycle tree:
#   tmux new -d -s e1v2-forced "E1_REPO=$S/wt-lifecycle bash $E1/run_e1_fr.sh"      (through the neutral-path launch wrapper)
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc GB300_ROOT=$S/gb300
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
unset CUDA_VISIBLE_DEVICES UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
PASS=e1_v2_forced_release
[ "${1:-$PASS}" = "$PASS" ] || { echo "this driver runs only $PASS (got '$1')"; exit 2; }
REPO=${E1_REPO:-$S/wt-lifecycle}
E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v2fr
CFG=$E1/configs_fr/$PASS/profile.json; TRACE=$E1/traces/e1_dry.csv
[ -f "$CFG" ] || { echo "no config $CFG"; exit 2; }
mkdir -p $RUN/$PASS
E1_REPO=$REPO E1_FR_OUT=$RUN/preflight bash $E1/preflight_e1_fr.sh 2>&1 | tee $RUN/$PASS/preflight.log
[ "${PIPESTATUS[0]}" = "0" ] || { echo "preflight failed; not starting"; exit 3; }
# defence in depth, immediately before launch: frozen 56-row trace, no delayed start, explicit GPU-held ceiling
[ "$(sha256sum $REPO/alignment/load_generator/req-frontend/target/release/session_runner 2>/dev/null | cut -d' ' -f1)" = "$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['sha256'])" $E1/BINARY-PROVENANCE.json)" ] || { echo "session_runner is not the pinned executable; not starting"; exit 6; }
[ "$(tail -n +2 $TRACE | grep -c .)" = 56 ] && [ "$(sha256sum $TRACE | cut -c1-16)" = c6d8e13b7229ffd4 ] && grep -q "\"path\": \"$TRACE\"" $CFG && ! grep -q capture_start_delay_seconds $CFG && grep -q '"gpu_held_ceiling_s": 2700.0' $CFG || { echo "trace/config guard failed; not starting"; exit 4; }
cp $CFG $RUN/$PASS/profile.json; cp $TRACE $RUN/$PASS/trace.csv
REG=$RUN/pids/owned.tsv; python3 -I $E1/checks/e1_pids.py register --registry $REG --label driver:$PASS $$ | tee -a $RUN/$PASS/run.log
# reservation record (written by the operator from kubectl before the launch): node must be THIS host and the pod Running
python3 -I -c "import json,sys,socket; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get('node')==socket.gethostname() and r.get('phase')=='Running' else 1)" $RUN/$PASS/reservation.json 2>/dev/null || { echo "reservation.json missing, not Running, or for another node; not starting"; exit 5; }
# independent GPU-state sampler (gate FR5): registered, every 2 s, exits by itself 60 s after it has seen the release
python3 -I $E1/checks/e1_pids.py spawn --registry $REG --label gpuwatch:$PASS -- python3 -I $E1/checks/e1_gpu_watch.py --out $RUN/$PASS/gpu_observations.jsonl --interval 2 --max-s 3300 --stop-after-released 30 | tee -a $RUN/$PASS/run.log
WPID=$(grep -ao '"pid": [0-9]*' $RUN/$PASS/run.log | tail -1 | tr -dc 0-9)
# forced-release trigger and independent importer observer: registered probe; verifies the importer from /proc, confirms the import in
# progress, writes the one pre-registered control line, then samples the importer after the controller's SIGTERM
python3 -I $E1/checks/e1_pids.py spawn --registry $REG --label probe:$PASS -- python3 -I $E1/checks/e1_release_probe.py --nsys-dir $RUN/$PASS/profile/nsys --inputs $E1/E1-V2-FR-INPUTS.json --out $RUN/$PASS/probe.json --log $RUN/$PASS/probe.log | tee -a $RUN/$PASS/run.log
cd $REPO
source profiling/env/arm64-gb300/gb300.env
unset CUDA_VISIBLE_DEVICES
echo "[$PASS] start $(date -u +%FT%TZ) on $(hostname) launcher_tree=$REPO repo=$(git rev-parse HEAD) tree_alignment=$(git rev-parse HEAD:alignment | cut -c1-12) cfg_sha=$(sha256sum $CFG | cut -c1-16) trace_sha=$(sha256sum $TRACE | cut -c1-16) session_runner=$(sha256sum alignment/load_generator/req-frontend/target/release/session_runner | cut -c1-16)" | tee -a $RUN/$PASS/run.log
BIN=alignment/load_generator/req-frontend/target/release/session_runner; BS0=$(sha256sum $BIN | cut -d' ' -f1); ST0=$(stat -c '%i %Y %s' $BIN)
# prebuilt load-generator contract: exact pinned bytes, pinned source, recorded build state; the launcher refuses before any engine
# start on a mismatch, never runs cargo, refuses any build tool any helper asks for (audit-hook guard), and hashes the executable
# again when it ends
uv run --no-sync python -m launcher alignment profile "$RUN/$PASS/profile.json" --frontend-provenance $E1/BINARY-PROVENANCE.json 2>&1 | tee -a $RUN/$PASS/run.log
rc=${PIPESTATUS[0]}
# independent of the launcher's own record (load_generator/frontend_binary.json): the driver's before/after hash and the count of cargo
# "Compiling" lines in the run log (V1's launcher rebuilt the binary at launch). Both feed the post-run gate S.binary_unchanged.
BS1=$(sha256sum $BIN | cut -d' ' -f1); ST1=$(stat -c '%i %Y %s' $BIN); nb=$(grep -ac "^ *Compiling " $RUN/$PASS/run.log)
printf '{"binary": "%s", "sha256_before_launcher": "%s", "sha256_after_launcher": "%s", "unchanged": %s, "cargo_compiling_lines_in_run_log": %s, "stat_before": "%s", "stat_after": "%s"}\n' "$BIN" "$BS0" "$BS1" "$([ "$BS0" = "$BS1" ] && echo true || echo false)" "$nb" "$ST0" "$ST1" > $RUN/$PASS/binary_check.json
echo "[$PASS] session_runner before/after the launcher: ${BS0:0:16} / ${BS1:0:16} $([ "$BS0" = "$BS1" ] && [ "$nb" = 0 ] && echo "(unchanged, no rebuild)" || echo "(REBUILT OR CHANGED DURING THE RUN: gate S.binary_unchanged will FAIL)")" | tee -a $RUN/$PASS/run.log
echo "[$PASS] rc=$rc end $(date -u +%FT%TZ)" | tee -a $RUN/$PASS/run.log
# independent GPU-release verification (gate L6c): nvidia-smi compute-process census + memory state, outside the controller
python3 -I $E1/checks/e1_gpu_release_census.py --registry $REG --label t0 --out $RUN/$PASS/gpu_release_census_t0.json | tee -a $RUN/$PASS/run.log
LC=$RUN/$PASS/profile/nsys/lifecycle.json
[ -f $LC ] && python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print('[lifecycle] phase', d['phase'], '| http', d['http'], '| stream', d['stream'], '| report', d['report'], '| gpu', d['gpu'], '| escalations', len(d['escalations']), '| release_context', d['gpu'].get('release_context'), '| final stream', (d['stream'].get('final') or {}), '| control', [(c['command'],c['result'],c['phase']) for c in d['control_commands']], '| actions', [(a.get('pid'),a.get('action'),a.get('reason')) for a in d['actions']], '| warnings', [w['message'] for w in d['warnings']])" $LC | tee -a $RUN/$PASS/run.log
for f in $(grep -ao "Using FlashInfer autotune cache file: [^ ]*" $RUN/$PASS/profile/vllm/*_server.log 2>/dev/null | awk '{print $NF}' | sort -u); do
  [ -f "$f" ] && cp -p "$f" $RUN/$PASS/autotune_configs.$(basename $(dirname $f) | cut -c1-12).json
done
SL=$(ls $RUN/$PASS/profile/vllm/*_server.log 2>/dev/null | head -1)
if [ -n "$SL" ]; then
  sd=$(grep -an "\[shutdown\].*trigger received" "$SL" | head -1 | cut -d: -f1); sd=${sd:-0}
  ede=$(grep -an "EngineDeadError\|EngineCore encountered an issue\|output_handler failed" "$SL" | cut -d: -f1 | tr "\n" " ")
  before=0; for l in $ede; do [ "$sd" = 0 ] || [ "$l" -lt "$sd" ] && before=$((before+1)); done
  echo "[$PASS] shutdown trigger at server-log line ${sd}; EngineDeadError-family lines: ${ede:-none}; before shutdown: $before $([ $before = 0 ] && echo "(accepted)" || echo "(ABORT CONDITION)")" | tee -a $RUN/$PASS/run.log
  echo "[$PASS] stop timeline: $(grep -ao "[0-9:]* \[api_router.py:31\] Stopping profiler" "$SL" | head -1) → $(grep -ao "[0-9:]* \[api_router.py:33\] Profiler stopped" "$SL" | head -1); $(grep -ao "Generating '[^']*'" "$SL" | head -1)" | tee -a $RUN/$PASS/run.log
fi
sleep 20
# let the registered sampler finish its post-release samples (bounded) so the final census sees no owned process
for i in $(seq 1 75); do [ -n "$WPID" ] && [ -d /proc/$WPID ] || break; sleep 2; done
python3 -I $E1/checks/e1_gpu_release_census.py --registry $REG --label t20 --out $RUN/$PASS/gpu_release_census_t20.json | tee -a $RUN/$PASS/run.log
E1_PID_REGISTRY=$REG bash $E1/checks/e1_process_census.sh 2>&1 | grep -v "tmux sessions" | tee -a $RUN/$PASS/run.log
exit $rc
