#!/usr/bin/env bash
# Copy the small e1_main_v3 artifacts into runs/e1_main_v3/ and the raw pass into a new archive with SHA256SUMS (raw stays out of Git).
set -u
S=/mnt/shared/home/dc/workspace/serving-studio; export HOME=/mnt/shared/home/dc
E1=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); PASS=e1_main_v3
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_main; P=$RUN/$PASS/profile
ART=$S/artifacts/phase1c-e1-main-dsv41-flash-gb300-tp4-$(date -u +%Y%m%d); d=$E1/runs/$PASS; mkdir -p $d $ART/run
cp $RUN/$PASS/profile.json $RUN/$PASS/trace.csv $RUN/$PASS/preflight.log $RUN/$PASS/run.log $d/ 2>/dev/null || true
cp $RUN/$PASS/autotune_configs.*.json $d/ 2>/dev/null || true
[ -f $P/profile_result.json ] && cp $P/profile_result.json $d/
[ -f $P/load_generator/summary.json ] && cp $P/load_generator/summary.json $d/load_generator_summary.json
for f in $P/vllm/*_launch.json $P/vllm/*_drive_summary.json; do [ -f "$f" ] && cp "$f" $d/; done
if [ -d $P/token_corpus ]; then mkdir -p $d/token_corpus; cp $P/token_corpus/manifest.json $P/token_corpus/provenance.json $d/token_corpus/; fi
mkdir -p $d/cpu; for f in $RUN/cpu/*$PASS*; do [ -e "$f" ] && case "$f" in *.sqlite|*.parquet) ;; *) cp -r "$f" $d/cpu/;; esac; done
rsync -a $RUN/$PASS/ $ART/run/$PASS/
( cd $ART/run && find $PASS -type f -print0 | sort -z | xargs -0 sha256sum > ../SHA256SUMS )
echo "raw: $ART/run/$PASS ($(du -sh $ART/run/$PASS | cut -f1), $(wc -l < $ART/SHA256SUMS) files in SHA256SUMS)"; du -sh $d
