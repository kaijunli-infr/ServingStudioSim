#!/usr/bin/env bash
# CPU-only evidence replay of the V1 run (remediation GO of 2026-11-18). Evaluates the PRESERVED V1 artifacts with the corrected
# validators (checks/e1_lifecycle_gates_v2.py, checks/e1_launch_identity_v2.py + checks/e1_identity_policy_v1.json). It never
# modifies the run directory, the archive, the preserved stream or the original acceptance record (V1-GATES.json: STOP), and it
# starts nothing on a GPU (CUDA hidden; only QdstrmImporter and `nsys export`, both CPU). Run on the capture tray as dc:
#   E1_REPO=$S/wt-lifecycle bash $E1/replay_e1_v1.sh        → runs/v1/replay/ (outputs), exit 0 PASS / 1 FAIL / 3 INCONCLUSIVE
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc CUDA_VISIBLE_DEVICES=""
REPO=${E1_REPO:-$S/wt-lifecycle}; E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v1; PASS=e1_v1_lifecycle
AR=$S/artifacts/phase1c-e1-v1-dsv41-flash-gb300-tp4-20261118
AR15=$S/artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115/run/e1_nsys_val
OUT=$E1/runs/v1/replay; mkdir -p $OUT
IMPORTER=/opt/nvidia/nsight-systems/2025.6.3/host-linux-armv8/QdstrmImporter
NSYS=/opt/nvidia/nsight-systems/2025.6.3/target-linux-sbsa-armv8/nsys
TMPQ=/tmp/dsv41_flash_1017_gb300_tp4_${PASS}.raw-preserve/dsv41_flash_1017_gb300_tp4_${PASS}.qdstrm
DURQ=$AR/preserved/dsv41_flash_1017_gb300_tp4_${PASS}.qdstrm
SCR=/tmp/v1-replay-recovery
echo "# V1 evidence replay $(date -u +%FT%TZ) on $(hostname); repo $(git -C $REPO rev-parse --short HEAD); CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'"
echo "## A. frozen evidence"
( cd $AR && sha256sum -c SHA256SUMS --quiet ) && echo "PASS  archive SHA256SUMS: all $(wc -l < $AR/SHA256SUMS) files verify" || { echo "FAIL  archive checksum mismatch"; exit 1; }
orig=$(sha256sum $RUN/cpu/V1-GATES.json | cut -d' ' -f1); arch=$(grep "  run/cpu/V1-GATES.json$" $AR/SHA256SUMS | cut -d' ' -f1)
[ "$orig" = "$arch" ] && echo "PASS  original acceptance record V1-GATES.json unchanged (sha256 $orig; verdict $(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['verdict'])" $RUN/cpu/V1-GATES.json))" || { echo "FAIL  original acceptance record changed"; exit 1; }
sd=$(sha256sum $DURQ | cut -d' ' -f1); echo "INFO  durable stream copy $DURQ: $(stat -c %s $DURQ) B sha256 $sd"
if [ -f $TMPQ ]; then st=$(sha256sum $TMPQ | cut -d' ' -f1); [ "$st" = "$sd" ] && echo "PASS  tray /tmp original retained and identical (sha256 $st)" || { echo "FAIL  tray original differs from the durable copy ($st)"; exit 1; }; else echo "INFO  tray /tmp original no longer present; the durable copy is used"; fi
before=$( (cd $RUN && find . -type f -newer $AR/SHA256SUMS | wc -l) )
echo "## D. configuration identity under the normalization policy"
python3 -I $E1/checks/e1_launch_identity_v2.py --a $AR15 --b $RUN/$PASS --policy $E1/checks/e1_identity_policy_v1.json --out $OUT/identity_v2.json | tee $OUT/identity_v2.txt; echo "identity_v2 rc=${PIPESTATUS[0]}"
echo "## corrected gates on the preserved artifacts (scientific checker exit codes are taken from the original post-run record)"
EXTRA=$(python3 -I -c "import json,sys
for g in json.load(open(sys.argv[1]))['gates']:
    if g['gate'].startswith('S.') and g['gate'] != 'S.config_identity': print('--extra', g['gate'][2:] + '=' + g['detail'].split()[-1], end=' ')" $RUN/cpu/V1-GATES.json)
python3 -I $E1/checks/e1_lifecycle_gates_v2.py --run $RUN/$PASS --inputs $E1/E1-V1-INPUTS.json --importer $IMPORTER --nsys $NSYS --scratch $SCR \
  --stream $DURQ --observations $OUT/operator_monitor_samples.jsonl --identity $OUT/identity_v2.json $EXTRA --original-gates $RUN/cpu/V1-GATES.json \
  --out $OUT/V1-RETRO-GATES.json | tee $OUT/V1-RETRO-GATES.txt; rc=${PIPESTATUS[0]}
rmdir $SCR 2>/dev/null
after=$( (cd $RUN && find . -type f -newer $AR/SHA256SUMS | wc -l) )
( cd $AR && sha256sum -c SHA256SUMS --quiet ) && [ "$before" = "$after" ] && echo "PASS  run directory and archive untouched by the replay (archive re-verified; files newer than the archive manifest: $after, unchanged)" || echo "FAIL  run directory or archive changed during the replay"
echo "## two separate verdicts"
echo "1. Original V1 (pinned gates, unchanged record): FAIL / STOP"
echo "2. Retrospective V1 evidence assessment (corrected gates): $(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['retrospective_assessment'])" $OUT/V1-RETRO-GATES.json)"
echo "gpu memory after the replay (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '); compute apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
exit $rc
