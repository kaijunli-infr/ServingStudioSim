#!/usr/bin/env bash
# Collect the small derived artifacts of the e1_b8_recovery pass into the repo tree (runs/b8_recovery/e1_b8_recovery/) and
# add the pass's raw files to the Stage 1.5 archive with their checksums (SHA256SUMS.b8, base run/). Raw captures stay outside Git.
# The Stage 1.5 collect script and its SHA256SUMS are not touched.
set -euo pipefail
S=/mnt/shared/home/dc/workspace/serving-studio
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15; PASS=e1_b8_recovery
E1=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ART=${ART:-$S/artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115}
d=$E1/runs/b8_recovery/$PASS; mkdir -p $d
cp $RUN/$PASS/profile.json $RUN/$PASS/trace.csv $d/ 2>/dev/null || true
[ -f $RUN/$PASS/preflight.log ] && cp $RUN/$PASS/preflight.log $d/
[ -f $RUN/$PASS/run.log ] && grep -av "^\[=*[0-9]*%\|AutoTuner\|^requests [0-9]*/[0-9]* |" $RUN/$PASS/run.log | tr -d "\r" > $d/run.log || true
cp $RUN/$PASS/autotune_configs.*.json $d/ 2>/dev/null || true
P=$RUN/$PASS/profile
[ -f $P/profile_result.json ] && cp $P/profile_result.json $d/
[ -f $P/load_generator/summary.json ] && cp $P/load_generator/summary.json $d/load_generator_summary.json
for f in $P/vllm/*_launch.json $P/vllm/*_drive_summary.json; do [ -f "$f" ] && cp "$f" $d/; done
if [ -d $P/token_corpus ]; then mkdir -p $d/token_corpus; cp $P/token_corpus/manifest.json $P/token_corpus/provenance.json $d/token_corpus/; fi
mkdir -p $d/cpu; for f in $RUN/cpu/*$PASS*; do [ -f "$f" ] && case "$f" in *.sqlite) ;; *) cp "$f" $d/cpu/;; esac; done
rsync -a $RUN/$PASS/ $ART/run/$PASS/
( cd $ART/run && find $PASS -type f -print0 | sort -z | xargs -0 sha256sum > ../SHA256SUMS.b8 )
echo "raw: $ART/run/$PASS ($(du -sh $ART/run/$PASS | cut -f1), $(wc -l < $ART/SHA256SUMS.b8) files in SHA256SUMS.b8)"; du -sh $d
