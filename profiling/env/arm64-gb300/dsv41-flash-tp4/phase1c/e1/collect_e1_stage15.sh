#!/usr/bin/env bash
# Collect the small Stage 1.5 derived artifacts into the repo tree and write SHA256SUMS of the raw stage15 run root.
# Same shape as collect_e1.sh (Stage 1, unchanged). Raw captures stay outside Git.
set -euo pipefail
S=/mnt/shared/home/dc/workspace/serving-studio
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15
E1=${E1_REPO:-$S/ServingStudioSim}/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
DATE=${DATE:-$(date -u +%Y%m%d)}
ART=$S/artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-$DATE
mkdir -p $E1/runs/stage15 $ART
for p in e1_dry_off_v2 e1_dry_on_v2 e1_dry_off2_v2 e1_nsys_val; do
  [ -d $RUN/$p ] || continue
  d=$E1/runs/stage15/$p; mkdir -p $d
  cp $RUN/$p/profile.json $d/ 2>/dev/null || true
  [ -f $RUN/$p/preflight.log ] && cp $RUN/$p/preflight.log $d/
  [ -f $RUN/$p/run.log ] && grep -av "^\[=*[0-9]*%\|AutoTuner\|^requests [0-9]*/[0-9]* |" $RUN/$p/run.log | tr -d "\r" > $d/run.log || true
  cp $RUN/$p/autotune_configs.*.json $d/ 2>/dev/null || true
  P=$RUN/$p/profile
  [ -f $P/profile_result.json ] && cp $P/profile_result.json $d/
  [ -f $P/load_generator/summary.json ] && cp $P/load_generator/summary.json $d/load_generator_summary.json
  for f in $P/vllm/*_launch.json $P/vllm/*_drive_summary.json; do [ -f "$f" ] && cp "$f" $d/; done
  if [ -d $P/token_corpus ]; then mkdir -p $d/token_corpus; cp $P/token_corpus/manifest.json $P/token_corpus/provenance.json $d/token_corpus/; fi
  [ -f $P/kernel_sequences.json ] && { cp $P/kernel_sequences.json $d/ && gzip -f $d/kernel_sequences.json; }
done
# CPU-side outputs (checkers, identity, pair comparison, nsys validation, autotune diffs) live in $RUN/cpu
if [ -d $RUN/cpu ]; then rsync -a --exclude "*.sqlite" $RUN/cpu/ $E1/runs/stage15/cpu/; fi
rsync -a $RUN/ $ART/run/
( cd $ART/run && find . -type f -print0 | sort -z | xargs -0 sha256sum > ../SHA256SUMS )
echo "raw: $ART ($(du -sh $ART | cut -f1), $(wc -l < $ART/SHA256SUMS) files)"; du -sh $E1/runs/stage15
