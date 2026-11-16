#!/usr/bin/env bash
# Post-run CPU gates and analysis for the full-E1 v3 pass (E1-AUTHORIZATION-PACKET-v2.md §8). Read-only on the run directory except
# for the derived files it writes next to it; nothing touches a GPU. Any rack-10 tray as dc:   E1_REPO=$WT bash $E1/postrun_e1_main_v3.sh
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc; export PATH=$HOME/.local/bin:$PATH
REPO=${E1_REPO:-$S/ServingStudioSim}; E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
P1A=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1a
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_main; PASS=e1_main_v3; PROF=$RUN/$PASS/profile; CPU=$RUN/cpu; mkdir -p $CPU
RUN15=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15
TAB=$S/phase1a-tables/$PASS; INP=$E1/E1-INPUTS.json
PY=$S/gb300/venv/bin/python
Q0=$S/artifacts/phase0-dsv41-flash-gb300-tp4-20261112/run
cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env
echo "# postrun $PASS $(date -u +%FT%TZ) on $(hostname)"
grep -a "rc=\|EngineDeadError-family\|process discovery" $RUN/$PASS/run.log | tail -3
echo "## sizes"; ls -l $PROF/nsys/ $PROF/parsed.json $PROF/parsed.kernels.parquet $PROF/host_timeline.json 2>&1 | awk '{print $5, $9}'
$PY -c "import json,sys; r=json.load(open(sys.argv[1])); print('validation:', r.get('validation')); print('parsed_device_ids:', r.get('parsed_device_ids'))" $PROF/profile_result.json

echo "## 0. capture-window gate: every wave's pure-decode iterations captured (>= 400, >= plan minimum), none straddling, parsed contiguous"
$PY -I $E1/analysis/e1_capture_window.py census --run $RUN/$PASS --out $CPU/census_$PASS.md --json $CPU/census_$PASS.json > /dev/null 2> $CPU/census_$PASS.err; cw=$?
$PY - $CPU/census_$PASS.json $INP <<'PY'; cwg=$?
import json,sys; c=json.load(open(sys.argv[1])); plan=json.load(open(sys.argv[2]))["main"]["plan"]; p=c.get("parsed") or {}
ok = bool(p.get("contiguous")) and not p.get("missing_in_metrics") and p.get("iterations")==c["metrics_iterations"]
bad=[]
for w in plan:
    key=f"b{w['b']}"; wv=c["waves"].get(key) or {}
    # census pools waves per b: require at least the plan's summed minimum for that b and >= 400 per wave-equivalent
    need=sum(x["pure_decode_iters_min"] for x in plan if x["b"]==w["b"]); have=wv.get("n_pure",0)
    if have < need: bad.append((key, have, need))
bad=sorted(set(bad))
cap_total=sum(v for k,v in p.get("kinds",{}).items() if k.startswith("pure_")); tot=sum(v.get("n_pure",0) for v in c["waves"].values())
ok = ok and not bad and cap_total==tot
print(f"{'PASS' if ok else 'FAIL'}  pure-decode captured {cap_total} of {tot}; parsed {p.get('iterations')} of {c['metrics_iterations']} metrics iterations, contiguous={p.get('contiguous')}; per-b shortfalls {bad}; first captured starts {p.get('first_parsed_start_s')} s")
sys.exit(0 if ok else 1)
PY
echo "capture_window census rc=$cw gate rc=$cwg"

echo "## 1. completeness v2 (routing-to-iteration joins; plan = the 36 E1 cells)"
uv run --no-sync python $E1/checks/e1_completeness_v2.py --profile $PROF --inputs $INP --checksum --out $CPU/v2_$PASS.json > $CPU/v2_$PASS.txt 2>&1; v2=$?
grep -E "^FAIL|^RESULT|cell b=" $CPU/v2_$PASS.txt | head -40; echo "completeness_v2 rc=$v2"

echo "## 2. input hashes + Phase 1A tables (frozen extractor)"
( cd $PROF && sha256sum parsed.json parsed.kernels.parquet host_timeline.json | sed "s# # run/$PASS/profile/#" ) > $CPU/${PASS}_extract_inputs.sha256; cat $CPU/${PASS}_extract_inputs.sha256
if [ ! -f $TAB/iter_rank.parquet ]; then
  ( cd $P1A && $PY -m costmodel.extract --profile $PROF --label $PASS --out $TAB --sha256sums $CPU/${PASS}_extract_inputs.sha256 --sha-prefix run/$PASS/profile/ ) > $CPU/extract_$PASS.log 2>&1; ex=$?
  tail -3 $CPU/extract_$PASS.log; echo "extract rc=$ex"
else echo "tables exist at $TAB (not re-extracted)"; ex=0; fi

echo "## 3. nsys validation checker (frozen thresholds: spread median <= 0.02 ms, p90 <= 0.05 ms, JIT <= 20)"
uv run --no-sync python $E1/checks/e1_nsys_validation.py --profile $PROF --tables $TAB --inputs $INP --out $CPU/nsys_validation_$PASS.json > $CPU/nsys_validation_$PASS.txt 2>&1; nv=$?
grep -E "^(FAIL|WARN|RESULT)" $CPU/nsys_validation_$PASS.txt | head -30; grep -c "^PASS" $CPU/nsys_validation_$PASS.txt | sed 's/^/PASS lines: /'; echo "nsys_validation rc=$nv"

echo "## 4. workload identity of the shared waves vs the accepted passes (b=8 32k wave 0 vs e1_b8_recovery; b=48 32k wave 0 vs e1_nsys_val)"
$PY -I $E1/checks/e1_workload_identity.py --a $RUN15/e1_b8_recovery --b $RUN/$PASS --prefix e1_b8_kv32k_w0 --out $CPU/workload_identity_b8_vs_$PASS.json > $CPU/workload_identity_b8_vs_$PASS.txt 2>&1; echo "vs e1_b8_recovery rc=$? : $(grep RESULT $CPU/workload_identity_b8_vs_$PASS.txt)"
$PY -I $E1/checks/e1_workload_identity.py --a $RUN15/e1_nsys_val --b $RUN/$PASS --prefix e1_b48_kv32k_w0 --out $CPU/workload_identity_b48_vs_$PASS.json > $CPU/workload_identity_b48_vs_$PASS.txt 2>&1; echo "vs e1_nsys_val rc=$? : $(grep RESULT $CPU/workload_identity_b48_vs_$PASS.txt)"
wi=$(grep -L "^RESULT PASS" $CPU/workload_identity_b*_vs_$PASS.txt | wc -l)

echo "## 5. pre-registered identification estimators (frozen analysis/e1_identify.py; nothing fitted)"
MET=$(ls $PROF/vllm/*_metrics.jsonl 2>/dev/null | head -1)
uv run --no-sync python $E1/analysis/e1_identify.py --tables $TAB --label $PASS --out $CPU/identify_$PASS ${MET:+--metrics $MET} --corpus $PROF/token_corpus > $CPU/identify_$PASS.txt 2>&1; idf=$?
tail -4 $CPU/identify_$PASS.txt | cut -c1-200; echo "identify rc=$idf"

echo "## 6. per-rank execution, collective wait, kernel identities (refs: run2 for b<=16, quadrant for b>=24); rank-dependence audit per b"
$PY $E1/analysis/e1_nsys_rank_kernels.py --profile $PROF --tables $TAB --label $PASS --cells 1,2,4,8,16 --top 25 --ref-profile $Q0/diverse_nsys_run2/profile --ref-tables $S/phase1a-tables/diverse_nsys_run2 --ref-label diverse_nsys_run2 --out $CPU/rank_kernels_${PASS}_small_b.md --json $CPU/rank_kernels_${PASS}_small_b.json > /dev/null 2> $CPU/rank_kernels_${PASS}_small_b.err; echo "rank_kernels b<=16 rc=$?"
$PY $E1/analysis/e1_nsys_rank_kernels.py --profile $PROF --tables $TAB --label $PASS --cells 24,32,48 --top 25 --ref-profile $Q0/quadrant_nsys/profile --ref-tables $S/phase1a-tables/quadrant_nsys --ref-label quadrant_nsys --out $CPU/rank_kernels_${PASS}_large_b.md --json $CPU/rank_kernels_${PASS}_large_b.json > /dev/null 2> $CPU/rank_kernels_${PASS}_large_b.err; echo "rank_kernels b>=24 rc=$?"
$PY -I $E1/analysis/e1_rank_dependence.py --out $CPU/rank_dependence_$PASS --boot 100 $(for b in 1 2 4 8 16 24 32 48; do echo --pass ${PASS}:$TAB:$b; done) > $CPU/rank_dependence_$PASS.txt 2>&1; echo "rank_dependence rc=$?"

echo "## 7. configuration identity of the launch record vs accepted e1_nsys_val (informational: name/log_dir/trace path/delay/nsys output path differ by design)"
uv run --no-sync python $E1/checks/e1_config_identity.py --a $RUN15/e1_nsys_val --b $RUN/$PASS --allow name log_dir workload.frontend.path nsys.capture_start_delay_seconds --ignore-env PATH --out $CPU/identity_nsys_val_vs_$PASS.json > $CPU/identity_nsys_val_vs_$PASS.txt 2>&1
grep -E "^FAIL|^PASS|^RESULT" $CPU/identity_nsys_val_vs_$PASS.txt | head -12

echo "GATES $PASS: capture_window rc=$cwg completeness_v2 rc=$v2 extract rc=$ex nsys_validation rc=$nv workload_identity fails=$wi -> $([ $cwg = 0 ] && [ $v2 = 0 ] && [ $ex = 0 ] && [ $nv = 0 ] && [ $wi = 0 ] && echo PASS || echo FAIL)"
echo "next (only after the gates PASS and on Kevin's instruction): python3 $E1/make_partition_1c.py --tables $TAB --label $PASS --designation identification --out $E1/../partition/partition-1c.json"
exit $(( cwg + v2 + ex + nv + wi ))
