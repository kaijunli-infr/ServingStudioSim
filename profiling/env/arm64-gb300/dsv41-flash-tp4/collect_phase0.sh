#!/usr/bin/env bash
# Collect the small Phase-0 derived artifacts (configs, run logs, summaries, analysis outputs, scripts) into the repo tree for commit.
set -euo pipefail
S=/mnt/shared/home/dc/workspace/serving-studio
RUN=$S/ServingStudioSim/logs/20261112_dsv41_flash_gb300_tp4
DST=$S/ServingStudioSim/profiling/env/arm64-gb300/dsv41-flash-tp4
mkdir -p $DST/phase0
cp $RUN/progress.md $DST/progress.md
for p in quadrant_nsys_run1 quadrant_nsys quadrant_workload diverse_nsys diverse_nsys_run2 diverse_workload diverse_corpus; do
  [ -d $RUN/$p ] || continue
  d=$DST/phase0/$p; mkdir -p $d
  cp $RUN/$p/profile.json $d/
  [ -f $RUN/$p/run.log ] && grep -av "^\[=*[0-9]*%\|AutoTuner\|^requests [0-9]*/[0-9]* |" $RUN/$p/run.log | tr -d "\r" > $d/run.log || true
  for f in $RUN/$p/*.py $RUN/$p/*.out $RUN/$p/*.txt; do [ -f "$f" ] && cp "$f" $d/; done
  P=$RUN/$p/profile
  [ -f $P/profile_result.json ] && cp $P/profile_result.json $d/
  [ -f $P/salvage_result.json ] && cp $P/salvage_result.json $d/
  [ -f $P/load_generator/summary.json ] && cp $P/load_generator/summary.json $d/load_generator_summary.json
  for f in $P/vllm/*_launch.json $P/vllm/*_drive_summary.json; do [ -f "$f" ] && cp "$f" $d/; done
  if [ -d $P/token_corpus ]; then mkdir -p $d/token_corpus; cp $P/token_corpus/manifest.json $P/token_corpus/provenance.json $d/token_corpus/; fi
  # kernel_sequences.json is the folded per-stage sequence (≈9 MB) — keep only for the nsys passes
  [ -f $P/kernel_sequences.json ] && { cp $P/kernel_sequences.json $d/ && gzip -f $d/kernel_sequences.json; }
done
cp $S/scripts/{env.sh,run_pass.sh,build_fork_env_v0310.sh,bootstrap_sim_env.sh,build_req_frontend.sh} $DST/ 2>/dev/null || true
cp $S/dsv41-gb300/make_configs.py $DST/ 2>/dev/null || true
du -sh $DST; find $DST -type f | wc -l; find $DST -type f -size +2M -printf "%s %p\n" | sort -n | tail -5
