#!/usr/bin/env bash
# Post-run CPU acceptance for the V2 forced-release pass (packet sections 5-7). CPU only (CUDA hidden). ON the capture tray as dc,
# because the preserved stream is on its /tmp:
#   E1_REPO=$S/wt-lifecycle bash $E1/postrun_e1_fr.sh
# Order: (A) copy the preserved stream to durable storage and verify the copy; the tray original is never removed here.
# (B) configuration identity under the normalization policy. (C) if the importer wrote no report (outcome B), recover the profile
# from the durable stream through the launcher's --import-stream path. (D) scientific checkers on the profile. (E) the frozen V2
# gates. ANY FAIL or INCONCLUSIVE → STOP (exit non-zero): no retry, no advance. Nothing recoverable is deleted on any path.
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc CUDA_VISIBLE_DEVICES=""; export PATH=$HOME/.local/bin:$PATH
REPO=${E1_REPO:-$S/wt-lifecycle}; E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
P1A=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1a
RUN=${E1_FR_RUN:-$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v2fr}; PASS=e1_v2_forced_release; PROF=$RUN/$PASS/profile; CPU=$RUN/cpu; mkdir -p $CPU
TAB=${E1_FR_TABLES:-$S/phase1a-tables/$PASS}
PY=$S/gb300/venv/bin/python
AR15=$S/artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115/run/e1_nsys_val
IMPORTER=/opt/nvidia/nsight-systems/2025.6.3/host-linux-armv8/QdstrmImporter
NSYS=/opt/nvidia/nsight-systems/2025.6.3/target-linux-sbsa-armv8/nsys
LC=$PROF/nsys/lifecycle.json
cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env; export CUDA_VISIBLE_DEVICES=""
echo "# postrun $PASS $(date -u +%FT%TZ) on $(hostname); CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"
[ -f $LC ] || { echo "FAIL  no lifecycle.json: nothing to assess → STOP"; exit 1; }
PHASE=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['phase'])" $LC)
grep -a "rc=\|EngineDeadError-family\|stop timeline" $RUN/$PASS/run.log | tail -3
echo "## A. durable copy of the preserved stream (the tray original stays)"
SRC=$(python3 -I -c "import json,sys; print((json.load(open(sys.argv[1])).get('stream') or {}).get('preserved_path') or '')" $LC)
DST=$RUN/$PASS/preserved/$(basename "${SRC:-none.qdstrm}")
if [ -n "$SRC" ] && [ -f "$SRC" ]; then
  mkdir -p $RUN/$PASS/preserved; [ -f $DST ] || cp -p "$SRC" $DST
  a=$(sha256sum "$SRC" | cut -d' ' -f1); b=$(sha256sum $DST | cut -d' ' -f1)
  [ "$a" = "$b" ] && echo "PASS  durable copy $DST: $(stat -c %s $DST) B sha256 $b = tray original (retained)" || echo "FAIL  durable copy differs from the tray original ($a vs $b)"
  echo "$b  preserved/$(basename $DST)" > $RUN/$PASS/preserved/SHA256SUMS
elif [ -f $DST ]; then echo "INFO  tray original no longer present; using the existing durable copy $DST"
else echo "FAIL  no preserved stream at '${SRC}' and no durable copy"; fi
echo "## B. configuration identity under the normalization policy (launch record + profile config vs the accepted e1_nsys_val run)"
python3 -I $E1/checks/e1_launch_identity_v2.py --a $AR15 --b $RUN/$PASS --policy $E1/checks/e1_identity_policy_v1.json --out $CPU/identity_v2_$PASS.json > $CPU/identity_v2_$PASS.txt 2>&1; echo "identity rc=$? ($(grep -c '^NORMALIZED' $CPU/identity_v2_$PASS.txt) normalized fields listed in cpu/identity_v2_$PASS.txt)"
echo "## C. outcome: lifecycle phase $PHASE"
rr=0
if [ "$PHASE" = report_written ] && [ -f $PROF/parsed.json ]; then
  echo "outcome A: the importer wrote a report; the in-session profile is assessed and compared with the offline recovery"
elif [ -f $DST ]; then
  echo "outcome B (or an unfinished finalisation): recovering the profile from the durable stream through the launcher (CPU)"
  uv run --no-sync python -m launcher alignment profile $RUN/$PASS/profile.json --import-stream $DST > $CPU/recovery_launcher_$PASS.log 2>&1; rr=$?
  tail -4 $CPU/recovery_launcher_$PASS.log | cut -c1-200; echo "launcher recovery rc=$rr"
else rr=1; echo "no report and no stream: nothing to recover"; fi
echo "## D. scientific checkers on the profile"
uv run --no-sync python $E1/checks/e1_completeness_v2.py --profile $PROF --inputs $E1/E1-STAGE15-INPUTS.json --plan dry --checksum --out $CPU/v2_$PASS.json > $CPU/v2_$PASS.txt 2>&1; v2=$?
grep -E "^FAIL|^RESULT|pure" $CPU/v2_$PASS.txt | head -8; echo "completeness_v2 rc=$v2"
( cd $PROF && sha256sum parsed.json parsed.kernels.parquet host_timeline.json | sed "s# # run/$PASS/profile/#" ) > $CPU/${PASS}_extract_inputs.sha256 2>/dev/null
( cd $P1A && $PY -m costmodel.extract --profile $PROF --label $PASS --out $TAB --sha256sums $CPU/${PASS}_extract_inputs.sha256 --sha-prefix run/$PASS/profile/ ) > $CPU/extract_$PASS.log 2>&1; ex=$?
tail -2 $CPU/extract_$PASS.log | cut -c1-200; echo "extract rc=$ex"
uv run --no-sync python $E1/checks/e1_nsys_validation.py --profile $PROF --tables $TAB --inputs $E1/E1-STAGE15-INPUTS.json --out $CPU/nsys_validation_$PASS.json > $CPU/nsys_validation_$PASS.txt 2>&1; nv=$?
grep -E "^(FAIL|WARN|RESULT)" $CPU/nsys_validation_$PASS.txt | head -12; echo "nsys_validation rc=$nv"
echo "## E. frozen V2 gates"
python3 -I $E1/checks/e1_forced_release_gates.py --run $RUN/$PASS --inputs $E1/E1-V2-FR-INPUTS.json --importer $IMPORTER --nsys $NSYS --scratch /tmp/v2-recovery-$PASS \
  --durable-stream $DST --identity $CPU/identity_v2_$PASS.json --extra launcher_recovery=$rr --extra completeness_v2=$v2 --extra extract=$ex --extra nsys_validation=$nv \
  ${E1_CLOSEOUT_RC:+--extra closeout=$E1_CLOSEOUT_RC} --out $CPU/V2-GATES.json | tee $CPU/V2-GATES.txt; lg=${PIPESTATUS[0]}
rmdir /tmp/v2-recovery-$PASS 2>/dev/null
echo "## verdict"
case $lg in
  0) echo "V2 ACCEPTANCE: PASS (every frozen gate). Any next step needs its own GO. The preserved stream stays on the tray and in the durable copy.";;
  3) echo "V2 ACCEPTANCE: INCONCLUSIVE (a gate could not be evaluated; see $CPU/V2-GATES.txt). STOP: no retry, no advance; everything stays preserved.";;
  *) echo "V2 ACCEPTANCE: FAIL (see $CPU/V2-GATES.txt). STOP: no retry, no advance; everything stays preserved.";;
esac
echo "gpu memory after the post-run (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
exit $lg
