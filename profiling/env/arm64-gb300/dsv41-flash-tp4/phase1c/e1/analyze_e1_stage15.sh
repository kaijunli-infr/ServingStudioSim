#!/usr/bin/env bash
# Stage 1.5 analysis after passes 1-3 (CPU, read-only on the runs; pre-registered scripts only). Output: $RUN/cpu/final/
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc; export PATH=$HOME/.local/bin:$PATH
REPO=${E1_REPO:-$S/ServingStudioSim}; E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15; F=$RUN/cpu/final; mkdir -p $F
OFF=$RUN/e1_dry_off_v2; ON=$RUN/e1_dry_on_v2; OFF2=$RUN/e1_dry_off2_v2
cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env
echo "## pair comparison v2 (on vs off, control off2)"
uv run --no-sync python $E1/analysis/e1_pair_compare_v2.py --off $OFF/profile --on $ON/profile --control $OFF2/profile --inputs $E1/E1-STAGE15-INPUTS.json --out $F/pair_v2 > $F/pair_v2.stdout 2>&1; echo "rc=$?"
echo "## pair comparison v2 with the arms swapped (on vs off2, control off): symmetric view of the null"
uv run --no-sync python $E1/analysis/e1_pair_compare_v2.py --off $OFF2/profile --on $ON/profile --control $OFF/profile --inputs $E1/E1-STAGE15-INPUTS.json --out $F/pair_v2_swapped > $F/pair_v2_swapped.stdout 2>&1; echo "rc=$?"
echo "## v1 pair comparison (Stage 1 script, unchanged) on vs off, for continuity"
uv run --no-sync python $E1/analysis/e1_pair_compare.py --off $OFF/profile --on $ON/profile --inputs $E1/E1-STAGE15-INPUTS.json --out $F/pair_v1 > $F/pair_v1.stdout 2>&1; echo "rc=$?"
echo "## identity off vs off2 (launch records)"
uv run --no-sync python $E1/checks/e1_config_identity.py --a $OFF --b $OFF2 --ignore-env PATH --out $F/identity_off_vs_off2.json > $F/identity_off_vs_off2.txt 2>&1; tail -1 $F/identity_off_vs_off2.txt
echo "## autotune cache diffs"
A=$(ls $OFF/autotune_configs.*.json | head -1); B=$(ls $ON/autotune_configs.*.json | head -1); C=$(ls $OFF2/autotune_configs.*.json | head -1)
python3 $E1/analysis/autotune_cache_diff.py $A $C --label-a off --label-b off2 --out $F/autotune_diff_off_off2.md | grep "^\*\*"
python3 $E1/analysis/autotune_cache_diff.py $A $B --label-a off --label-b on --out $F/autotune_diff_off_on.md | grep "^\*\*"
python3 $E1/analysis/autotune_cache_diff.py $C $B --label-a off2 --label-b on --out $F/autotune_diff_off2_on.md | grep "^\*\*"
echo "## output differences"
python3 $E1/analysis/e1_output_diff.py --run off=$OFF/profile --run on=$ON/profile --run off2=$OFF2/profile --out $F/output_diff.md > /dev/null; sed -n 5,13p $F/output_diff.md
echo "## timeline"
for p in e1_dry_off_v2 e1_dry_on_v2 e1_dry_off2_v2; do SL=$(ls $RUN/$p/profile/vllm/*_server.log); st=$(grep -a "\] start " $RUN/$p/run.log | grep -o "20[0-9-]*T[0-9:]*Z" | head -1); en=$(grep -a "rc=" $RUN/$p/run.log | grep -o "20[0-9-]*T[0-9:]*Z" | tail -1); at0=$(grep -h "Autotuning process starts" $SL | head -1 | cut -c1-19); at1=$(grep -h "Autotuning process ends" $SL | head -1 | cut -c1-19); echo "$p start=$st rc_end=$en autotune=[$at0 -> $at1] iterations=$(grep -c VibeSimAlignmentIteration $SL) no_tuned_config=$(grep -c "No tuned config" $SL) cache_dir=$(grep -o "Using FlashInfer autotune cache file: .*" $SL | head -1 | sed "s|.*/flashinfer-autotune/||" | cut -c1-12) size=$(du -sh $RUN/$p | cut -f1)"; done | tee $F/timeline.txt
echo "## request-level engine timings (request_timings.jsonl): per pass, median engine_core_tpot_ms per cell"
for p in e1_dry_off_v2 e1_dry_on_v2 e1_dry_off2_v2; do python3 - $RUN/$p/profile/vllm/*_request_timings.jsonl $p <<'PY'
import json,sys,statistics,collections
recs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
c=collections.defaultdict(list)
for r in recs:
    if "e1_b" in r.get("request_id",""): c[r["request_id"].split("_r")[0].replace("independent_","")].append(r["engine_core_tpot_ms"])
print(sys.argv[2], {k:(len(v), round(statistics.median(v),4)) for k,v in sorted(c.items())})
PY
done | tee $F/request_tpot.txt
echo "done: $F"
