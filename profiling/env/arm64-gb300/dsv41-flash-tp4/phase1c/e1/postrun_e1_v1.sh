#!/usr/bin/env bash
# Post-run CPU acceptance for the V1 pass (packet §7, §9): scientific checkers (completeness v2, Phase 1A extract, nsys validation,
# configuration identity of the launch record vs the accepted e1_nsys_val run) and the lifecycle gates (idempotence, phase order,
# report completeness, preserved stream + offline recovery proof, no signal to the importer, GPU-held within the ceiling, control
# probe recorded). ANY FAIL → STOP (exit 1): no advance to V2. Read-only on the run directory except the derived files it writes
# under <run root>/cpu. Run on any rack-10 tray as dc:
#   E1_REPO=$S/wt-lifecycle bash $E1/postrun_e1_v1.sh
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc; export PATH=$HOME/.local/bin:$PATH
REPO=${E1_REPO:-$S/wt-lifecycle}; E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
P1A=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1a
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v1; PASS=e1_v1_lifecycle; PROF=$RUN/$PASS/profile; CPU=$RUN/cpu; mkdir -p $CPU
TAB=$S/phase1a-tables/$PASS
PY=$S/gb300/venv/bin/python
AR15=$S/artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115/run/e1_nsys_val
IMPORTER=$(ls /opt/nvidia/nsight-systems/2025.6.3/host-linux-armv8/QdstrmImporter 2>/dev/null | head -1)
cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env
echo "# postrun $PASS $(date -u +%FT%TZ) on $(hostname)"
grep -a "rc=\|EngineDeadError-family\|stop timeline" $RUN/$PASS/run.log | tail -3
echo "## sizes"; ls -l $PROF/nsys/ $PROF/parsed.json $PROF/parsed.kernels.parquet $PROF/host_timeline.json 2>&1 | awk '{print $5, $9}'
$PY -c "import json,sys; r=json.load(open(sys.argv[1])); print('validation:', r.get('validation')); print('nsys_profiler:', r.get('nsys_profiler'))" $PROF/profile_result.json

echo "## 1. completeness v2 (routing-to-iteration joins, both waves; dry plan)"
uv run --no-sync python $E1/checks/e1_completeness_v2.py --profile $PROF --inputs $E1/E1-STAGE15-INPUTS.json --plan dry --checksum --out $CPU/v2_$PASS.json > $CPU/v2_$PASS.txt 2>&1; v2=$?
grep -E "^FAIL|^RESULT|requests|join|pure" $CPU/v2_$PASS.txt | head -10; echo "completeness_v2 rc=$v2"

echo "## 2. input hashes + Phase 1A tables (costmodel.extract)"
( cd $PROF && sha256sum parsed.json parsed.kernels.parquet host_timeline.json | sed "s# # run/$PASS/profile/#" ) > $CPU/${PASS}_extract_inputs.sha256
if [ ! -f $TAB/iter_rank.parquet ]; then
  ( cd $P1A && $PY -m costmodel.extract --profile $PROF --label $PASS --out $TAB --sha256sums $CPU/${PASS}_extract_inputs.sha256 --sha-prefix run/$PASS/profile/ ) > $CPU/extract_$PASS.log 2>&1; ex=$?
  tail -3 $CPU/extract_$PASS.log; echo "extract rc=$ex"
else echo "tables exist at $TAB (not re-extracted)"; ex=0; fi

echo "## 3. nsys validation checker (NVTX complete, GPU windows overlap, collective spread, JIT, joins)"
uv run --no-sync python $E1/checks/e1_nsys_validation.py --profile $PROF --tables $TAB --inputs $E1/E1-STAGE15-INPUTS.json --out $CPU/nsys_validation_$PASS.json > $CPU/nsys_validation_$PASS.txt 2>&1; nv=$?
grep -E "^(FAIL|WARN|RESULT)" $CPU/nsys_validation_$PASS.txt | head -20; grep -c "^PASS" $CPU/nsys_validation_$PASS.txt | sed 's/^/PASS lines: /'; echo "nsys_validation rc=$nv"

echo "## 4. configuration identity of the launch record vs the accepted e1_nsys_val run (argv, env, server config)"
uv run --no-sync python $E1/checks/e1_config_identity.py --a $AR15 --b $RUN/$PASS --ignore-env PATH \
  --allow name log_dir workload.frontend.path nsys.capture_start_delay_seconds nsys.finalize_soft_deadline_s nsys.finalize_hard_deadline_s nsys.finalize_stall_timeout_s \
          nsys.finalize_on_hard_deadline nsys.finalize_session_close_timeout_s nsys.finalize_max_extension_s nsys.gpu_held_ceiling_s nsys.preserve_raw_stream \
  --out $CPU/identity_nsys_val_vs_$PASS.json > $CPU/identity_nsys_val_vs_$PASS.txt 2>&1; ci=$?
grep -E "^FAIL|^RESULT" $CPU/identity_nsys_val_vs_$PASS.txt | head -12; echo "config_identity rc=$ci"

echo "## 5. recipe test against this run's launch record (argv/prefix/env unchanged by the lifecycle controller)"
LAUNCH=$(ls $PROF/vllm/*_launch.json | head -1)
uv run --no-sync python -m alignment.profiler.preflight --config $RUN/$PASS/profile.json --reference-launch $LAUNCH --json $CPU/identity_postrun_$PASS.json > $CPU/identity_postrun_$PASS.txt 2>&1
rp=$(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(0 if all(c['status']!='FAIL' for c in d['checks'] if c['name'].startswith('recipe.')) else 1)" $CPU/identity_postrun_$PASS.json 2>/dev/null || echo 1)
grep -E "recipe\." $CPU/identity_postrun_$PASS.txt; echo "recipe rc=$rp"

echo "## 6. lifecycle gates (+ offline recovery proof from the preserved stream)"
python3 -I $E1/checks/e1_lifecycle_gates.py --run $RUN/$PASS --inputs $E1/E1-V1-INPUTS.json ${IMPORTER:+--importer $IMPORTER} --scratch $CPU/recovery_$PASS \
  --extra completeness_v2=$v2 --extra extract=$ex --extra nsys_validation=$nv --extra config_identity=$ci --extra recipe=$rp --out $CPU/V1-GATES.json | tee $CPU/V1-GATES.txt; lg=${PIPESTATUS[0]}
echo "## verdict"
if [ $lg = 0 ]; then
  echo "V1 ACCEPTANCE: PASS (all lifecycle and scientific gates). Next step still needs its own GO. The preserved stream may now be removed by the operator:"
  python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); p=(d.get('stream') or {}).get('preserved_path'); print('  rm -rf', p.rsplit('/',1)[0] if p else '<none>')" $PROF/nsys/lifecycle.json
else
  echo "V1 ACCEPTANCE: STOP (a gate FAILED; see $CPU/V1-GATES.txt). Do not advance to V2; keep the preserved stream and the run directory for review."
fi
exit $lg
