#!/usr/bin/env bash
# Collect the small E1 derived artifacts into the repo tree for commit and write SHA256SUMS of the raw run (after the capture).
# Raw captures stay outside Git: $S/artifacts/phase1c-e1-dsv41-flash-gb300-tp4-<date>/ (rsync of the run dir) + SHA256SUMS,
# compact tables in $S/phase1a-tables/e1_main/, second copy on rack 3 (/mnt/shared/home/kevin/dsv41-phase1-mirror/).
set -euo pipefail
S=/mnt/shared/home/dc/workspace/serving-studio
RUN=$S/ServingStudioSim/logs/20261114_dsv41_flash_gb300_tp4_e1
E1=${E1_REPO:-$S/ServingStudioSim}/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1   # derived files go to the launcher tree (Stage 1: the scratch worktree)
DATE=${DATE:-$(date -u +%Y%m%d)}
ART=$S/artifacts/phase1c-e1-dsv41-flash-gb300-tp4-$DATE
mkdir -p $E1/runs $ART
for p in e1_dry_off e1_dry_on e1_main e1_main_norouts; do
  [ -d $RUN/$p ] || continue
  d=$E1/runs/$p; mkdir -p $d
  cp $RUN/$p/profile.json $d/ 2>/dev/null || true
  [ -f $RUN/$p/preflight.log ] && cp $RUN/$p/preflight.log $d/
  [ -f $RUN/$p/run.log ] && grep -av "^\[=*[0-9]*%\|AutoTuner\|^requests [0-9]*/[0-9]* |" $RUN/$p/run.log | tr -d "\r" > $d/run.log || true
  P=$RUN/$p/profile
  [ -f $P/profile_result.json ] && cp $P/profile_result.json $d/
  [ -f $P/load_generator/summary.json ] && cp $P/load_generator/summary.json $d/load_generator_summary.json
  for f in $P/vllm/*_launch.json $P/vllm/*_drive_summary.json; do [ -f "$f" ] && cp "$f" $d/; done
  if [ -d $P/token_corpus ]; then mkdir -p $d/token_corpus; cp $P/token_corpus/manifest.json $P/token_corpus/provenance.json $d/token_corpus/; fi
  [ -f $P/kernel_sequences.json ] && { cp $P/kernel_sequences.json $d/ && gzip -f $d/kernel_sequences.json; }
done
# Stage 1 (dry-run pair) small reports: preflight logs, fingerprints, completeness and pair-comparison outputs
if [ -d $RUN/stage1 ]; then mkdir -p $E1/runs/stage1; rsync -a --exclude "*_profile/" $RUN/stage1/ $E1/runs/stage1/; fi
rsync -a $RUN/ $ART/run/
( cd $ART/run && find . -type f -print0 | sort -z | xargs -0 sha256sum > ../SHA256SUMS )
echo "raw: $ART ($(du -sh $ART | cut -f1), $(wc -l < $ART/SHA256SUMS) files)"; du -sh $E1/runs
