#!/usr/bin/env bash
# Post-run CPU analysis for the e1_b8_recovery pass (E1-B8-RECOVERY-PACKET.md §7). Read-only on the run directory except for
# the derived files it writes next to it; nothing touches a GPU. Run on any rack-10 tray as dc:
#   E1_REPO=$WT bash $E1/postrun_e1_b8.sh
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc; export PATH=$HOME/.local/bin:$PATH
REPO=${E1_REPO:-$S/ServingStudioSim}; E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
P1A=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1a
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15; PASS=e1_b8_recovery; PROF=$RUN/$PASS/profile; CPU=$RUN/cpu; mkdir -p $CPU
TAB=$S/phase1a-tables/$PASS; INP=$E1/E1-B8-RECOVERY-INPUTS.json
PY=$S/gb300/venv/bin/python
QA=$S/artifacts/phase0-dsv41-flash-gb300-tp4-20261112/run/quadrant_nsys/profile; QT=$S/phase1a-tables/quadrant_nsys
cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env
echo "# postrun $PASS $(date -u +%FT%TZ) on $(hostname)"
grep -a "rc=\|EngineDeadError-family\|process census" $RUN/$PASS/run.log | tail -3
echo "## sizes"; ls -l $PROF/nsys/ $PROF/parsed.json $PROF/parsed.kernels.parquet $PROF/host_timeline.json 2>&1 | awk '{print $5, $9}'
$PY -c "import json,sys; r=json.load(open(sys.argv[1])); print('validation:', r.get('validation')); print('parsed_device_ids:', r.get('parsed_device_ids'))" $PROF/profile_result.json

echo "## 0. capture-window coverage gate (analysis/e1_capture_window.py census): every pure b=8 iteration captured, none straddling"
$PY -I $E1/analysis/e1_capture_window.py census --run $RUN/$PASS --out $CPU/census_$PASS.md --json $CPU/census_$PASS.json > /dev/null 2> $CPU/census_$PASS.err; cw=$?
$PY - $CPU/census_$PASS.json <<'PY'; cwg=$?
import json,sys; c=json.load(open(sys.argv[1])); p=c.get("parsed") or {}; w=c["waves"].get("b8") or {}
cap=p.get("kinds",{}).get("pure_b8",0); tot=w.get("n_pure",0); own=p.get("owners",{})
ok = tot>0 and cap==tot and cap>=400 and p.get("contiguous") and not p.get("missing_in_metrics") and set(own)<= {"b8_kv32k","-"}
print(f"{'PASS' if ok else 'FAIL'}  pure b=8 captured {cap} of {tot} (need all, >= 400); parsed contiguous={p.get('contiguous')}; owners {own}; parsed {p.get('iterations')} of {c['metrics_iterations']} metrics iterations; first captured starts {p.get('first_parsed_start_s')} s")
sys.exit(0 if ok else 1)
PY
echo "capture_window census rc=$cw gate rc=$cwg"

echo "## 1. completeness v2 (routing-to-iteration joins, hook on; plan = the b=8 cell)"
uv run --no-sync python $E1/checks/e1_completeness_v2.py --profile $PROF --inputs $INP --checksum --out $CPU/v2_$PASS.json > $CPU/v2_$PASS.txt 2>&1; v2=$?
grep -E "^FAIL|^RESULT|requests seen|cell b=" $CPU/v2_$PASS.txt | head -8; echo "completeness_v2 rc=$v2"

echo "## 2. input hashes + Phase 1A tables (costmodel.extract)"
( cd $PROF && sha256sum parsed.json parsed.kernels.parquet host_timeline.json | sed "s# # run/$PASS/profile/#" ) > $CPU/${PASS}_extract_inputs.sha256
cat $CPU/${PASS}_extract_inputs.sha256
if [ ! -f $TAB/iter_rank.parquet ]; then
  ( cd $P1A && $PY -m costmodel.extract --profile $PROF --label $PASS --out $TAB --sha256sums $CPU/${PASS}_extract_inputs.sha256 --sha-prefix run/$PASS/profile/ ) > $CPU/extract_$PASS.log 2>&1; ex=$?
  tail -5 $CPU/extract_$PASS.log; echo "extract rc=$ex"
else echo "tables exist at $TAB (not re-extracted)"; ex=0; fi

echo "## 3. nsys validation checker (thresholds unchanged: spread median <= 0.02 ms, p90 <= 0.05 ms, JIT <= 20)"
uv run --no-sync python $E1/checks/e1_nsys_validation.py --profile $PROF --tables $TAB --inputs $INP --out $CPU/nsys_validation_$PASS.json > $CPU/nsys_validation_$PASS.txt 2>&1; nv=$?
grep -E "^(FAIL|WARN|RESULT)" $CPU/nsys_validation_$PASS.txt | head -20; grep -c "^PASS" $CPU/nsys_validation_$PASS.txt | sed 's/^/PASS lines: /'; echo "nsys_validation rc=$nv"

echo "## 4. workload identity of the 8 requests vs e1_dry_on_v2 and e1_nsys_val (replay records; routes informational)"
for ref in e1_dry_on_v2 e1_nsys_val; do
  $PY -I $E1/checks/e1_workload_identity.py --a $RUN/$ref --b $RUN/$PASS --out $CPU/workload_identity_${ref}_vs_$PASS.json > $CPU/workload_identity_${ref}_vs_$PASS.txt 2>&1; echo "vs $ref rc=$? : $(grep -E '^RESULT|^FAIL' $CPU/workload_identity_${ref}_vs_$PASS.txt | head -3 | tr '\n' ' ')"
done; wi=$(grep -L "^RESULT PASS" $CPU/workload_identity_*_vs_$PASS.txt | wc -l)

echo "## 5. per-rank execution, collective wait, kernel identities (b=8; reference quadrant_nsys, hypothesis material only)"
$PY $E1/analysis/e1_nsys_rank_kernels.py --profile $PROF --tables $TAB --label $PASS --cells 8 --top 25 \
  --ref-profile $QA --ref-tables $QT --ref-label quadrant_nsys --out $CPU/rank_kernels_$PASS.md --json $CPU/rank_kernels_$PASS.json > /dev/null 2> $CPU/rank_kernels_$PASS.err; rk=$?
echo "rank_kernels rc=$rk ($(wc -l < $CPU/rank_kernels_$PASS.md) lines)"; head -c 300 $CPU/rank_kernels_$PASS.err

echo "## 6. configuration identity of the launch record vs e1_nsys_val (informational: name/log_dir/trace path/delay differ by design)"
uv run --no-sync python $E1/checks/e1_config_identity.py --a $RUN/e1_nsys_val --b $RUN/$PASS --allow name log_dir workload.frontend.path nsys.capture_start_delay_seconds --ignore-env PATH --out $CPU/identity_nsys_val_vs_$PASS.json > $CPU/identity_nsys_val_vs_$PASS.txt 2>&1
grep -E "^FAIL|^RESULT" $CPU/identity_nsys_val_vs_$PASS.txt | head -12

echo "GATES $PASS: capture_window rc=$cwg completeness_v2 rc=$v2 extract rc=$ex nsys_validation rc=$nv workload_identity fails=$wi -> $([ $cwg = 0 ] && [ $v2 = 0 ] && [ $ex = 0 ] && [ $nv = 0 ] && [ $wi = 0 ] && echo PASS || echo FAIL)"
exit $(( cwg + v2 + ex + nv + wi ))
