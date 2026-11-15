#!/usr/bin/env bash
# Post-run CPU analysis for the e1_nsys_val pass (E1-NSYS-VALIDATION-PROTOCOL.md §4). Read-only on the run directory except
# for the derived files it writes next to it; nothing touches a GPU. Run on any rack-10 tray as dc:
#   E1_REPO=$WT bash $E1/postrun_e1_nsys_val.sh
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc; export PATH=$HOME/.local/bin:$PATH
REPO=${E1_REPO:-$S/ServingStudioSim}; E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
P1A=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1a
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15; PASS=e1_nsys_val; PROF=$RUN/$PASS/profile; CPU=$RUN/cpu; mkdir -p $CPU
TAB=$S/phase1a-tables/$PASS
PY=$S/gb300/venv/bin/python
QA=$S/artifacts/phase0-dsv41-flash-gb300-tp4-20261112/run/quadrant_nsys/profile; QT=$S/phase1a-tables/quadrant_nsys
cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env
echo "# postrun $PASS $(date -u +%FT%TZ) on $(hostname)"
grep -a "rc=\|EngineDeadError-family\|WARNING compute apps" $RUN/$PASS/run.log | tail -3
echo "## sizes"; ls -l $PROF/nsys/ $PROF/parsed.json $PROF/parsed.kernels.parquet $PROF/host_timeline.json 2>&1 | awk '{print $5, $9}'
$PY -c "import json,sys; r=json.load(open(sys.argv[1])); print('validation:', r.get('validation')); print('nsys_profiler:', r.get('nsys_profiler')); print('parsed_device_ids:', r.get('parsed_device_ids'))" $PROF/profile_result.json

echo "## 1. completeness v2 (routing-to-iteration joins, hook on)"
uv run --no-sync python $E1/checks/e1_completeness_v2.py --profile $PROF --inputs $E1/E1-STAGE15-INPUTS.json --checksum --out $CPU/v2_$PASS.json > $CPU/v2_$PASS.txt 2>&1; v2=$?
grep -E "^FAIL|^RESULT|requests|join" $CPU/v2_$PASS.txt | head -8; echo "completeness_v2 rc=$v2"

echo "## 2. input hashes + Phase 1A tables (costmodel.extract)"
( cd $PROF && sha256sum parsed.json parsed.kernels.parquet host_timeline.json | sed "s# # run/$PASS/profile/#" ) > $CPU/${PASS}_extract_inputs.sha256
cat $CPU/${PASS}_extract_inputs.sha256
if [ ! -f $TAB/iter_rank.parquet ]; then
  ( cd $P1A && $PY -m costmodel.extract --profile $PROF --label $PASS --out $TAB --sha256sums $CPU/${PASS}_extract_inputs.sha256 --sha-prefix run/$PASS/profile/ ) > $CPU/extract_$PASS.log 2>&1; ex=$?
  tail -5 $CPU/extract_$PASS.log; echo "extract rc=$ex"
else echo "tables exist at $TAB (not re-extracted)"; ex=0; fi

echo "## 3. nsys validation checker"
uv run --no-sync python $E1/checks/e1_nsys_validation.py --profile $PROF --tables $TAB --inputs $E1/E1-STAGE15-INPUTS.json --out $CPU/nsys_validation_$PASS.json > $CPU/nsys_validation_$PASS.txt 2>&1; nv=$?
grep -E "^(FAIL|WARN|RESULT)" $CPU/nsys_validation_$PASS.txt | head -20; grep -c "^PASS" $CPU/nsys_validation_$PASS.txt | sed 's/^/PASS lines: /'; echo "nsys_validation rc=$nv"

echo "## 4. per-rank execution, collective wait, kernel identities (+ cross-session reference quadrant_nsys)"
$PY $E1/analysis/e1_nsys_rank_kernels.py --profile $PROF --tables $TAB --label $PASS --cells 8,48 --top 25 \
  --ref-profile $QA --ref-tables $QT --ref-label quadrant_nsys --out $CPU/rank_kernels_$PASS.md --json $CPU/rank_kernels_$PASS.json > /dev/null 2> $CPU/rank_kernels_$PASS.err; rk=$?
echo "rank_kernels rc=$rk ($(wc -l < $CPU/rank_kernels_$PASS.md) lines)"; head -c 400 $CPU/rank_kernels_$PASS.err

echo "## 5. identify smoke test (estimator pipeline on a 2-cell pass; no estimate is used)"
MET=$(ls $PROF/vllm/*_metrics.jsonl 2>/dev/null | head -1)
uv run --no-sync python $E1/analysis/e1_identify.py --tables $TAB --label $PASS --out $CPU/identify_smoke_$PASS ${MET:+--metrics $MET} > $CPU/identify_smoke_$PASS.txt 2>&1; idf=$?
tail -3 $CPU/identify_smoke_$PASS.txt | cut -c1-200; echo "identify smoke rc=$idf (informational)"

echo "## 6. configuration identity of the launch record vs e1_dry_on_v2 (informational: profile_kind/nsys/name/log_dir differ by design)"
uv run --no-sync python $E1/checks/e1_config_identity.py --a $RUN/e1_dry_on_v2 --b $RUN/$PASS --ignore-env PATH --out $CPU/identity_on_v2_vs_$PASS.json > $CPU/identity_on_v2_vs_$PASS.txt 2>&1
grep -E "^FAIL|^RESULT" $CPU/identity_on_v2_vs_$PASS.txt | head -12

echo "GATES $PASS: completeness_v2 rc=$v2 extract rc=$ex nsys_validation rc=$nv -> $([ $v2 = 0 ] && [ $ex = 0 ] && [ $nv = 0 ] && echo PASS || echo FAIL)"
exit $(( v2 + ex + nv ))
