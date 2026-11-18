#!/usr/bin/env bash
# Post-run CPU acceptance for the V3 pass (packet sections 5-9). CPU only (CUDA hidden). ON the capture tray as dc, because the
# preserved stream is on its /tmp:
#   E1_REPO=$S/wt-lifecycle [E1_CLOSEOUT_RC=<closeout exit code>] bash $E1/postrun_e1_v3scale.sh
# Order: storage and host-memory samplers start (they measure this script's own scratch and the offline importer). (A) copy the
# preserved stream to durable storage and verify the copy; the tray original is never removed here. (B) configuration identity under
# the normalization policy. (C) scientific checkers on the in-session profile: four-wave completeness and routing records, extract,
# Nsight validation. (D) the V3 gates, which import the durable stream offline and compare it with the in-session export.
# ANY FAIL or INCONCLUSIVE → STOP (exit non-zero): no retry, no advance. Nothing recoverable is deleted on any path.
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc CUDA_VISIBLE_DEVICES=""; export PATH=$HOME/.local/bin:$PATH
REPO=${E1_REPO:-$S/wt-lifecycle}; E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
P1A=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1a
RUN=${E1_V3S_RUN:-$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v3scale}; PASS=e1_v3_scale; PROF=$RUN/$PASS/profile; CPU=$RUN/cpu; mkdir -p $CPU
TAB=${E1_V3S_TABLES:-$S/phase1a-tables/$PASS}
INP=${E1_V3S_INPUTS:-$E1/E1-V3-SCALE-INPUTS.json}
PY=$S/gb300/venv/bin/python
AR15=$S/artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115/run/e1_nsys_val
IMPORTER=/opt/nvidia/nsight-systems/2025.6.3/host-linux-armv8/QdstrmImporter
NSYS=/opt/nvidia/nsight-systems/2025.6.3/target-linux-sbsa-armv8/nsys
MEMEXE=$IMPORTER
if [ "${E1_V3S_MODE:-real}" = rehearsal ]; then   # end-to-end rehearsal only: stand-in importer / exporter; the verdict can never be PASS (gate G0)
  case "$RUN" in */dsv41_flash_gb300_tp4_e1_v3scale_e2e) ;; *) echo "rehearsal tools are accepted only on the rehearsal run root"; exit 2;; esac
  IMPORTER=${E1_V3S_MOCK_IMPORTER:?}; NSYS=${E1_V3S_MOCK_NSYS:?}; MEMEXE=${E1_V3S_MOCK_IMPORTER_EXE:?}
  echo "REHEARSAL MODE: stand-in importer $IMPORTER and exporter $NSYS"
fi
LC=$PROF/nsys/lifecycle.json; SCR=/tmp/v3scale-recovery-$PASS
cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env; export CUDA_VISIBLE_DEVICES=""
echo "# postrun $PASS $(date -u +%FT%TZ) on $(hostname); CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"
[ -f $LC ] || { echo "FAIL  no lifecycle.json: nothing to assess → STOP"; exit 1; }
[ -e $SCR ] && { echo "FAIL  scratch directory $SCR exists: not touching it → STOP"; exit 1; }
PHASE=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['phase'])" $LC)
grep -a "rc=\|EngineDeadError-family" $RUN/$PASS/run.log | tail -2
SRC=$(python3 -I -c "import json,sys; print((json.load(open(sys.argv[1])).get('stream') or {}).get('preserved_path') or '')" $LC)
DST=$RUN/$PASS/preserved/$(basename "${SRC:-none.qdstrm}")
echo "## samplers for this post-run (own children of this script; stopped through a stop file; they signal nothing)"
STOPF=$CPU/postrun_samplers.stop; rm -f $STOPF $CPU/storage_postrun.jsonl $CPU/mem_postrun.jsonl
python3 -I $E1/checks/e1_storage_watch.py --out $CPU/storage_postrun.jsonl --summary $CPU/storage_postrun.json --group "temporary=/tmp/nsys-report-*.qdstrm,/tmp/*.raw-preserve,/tmp/v3scale-recovery-*" --group "persistent=$RUN,$TAB" --ceiling-gb temporary=10 --ceiling-gb persistent=8 --interval 2 --max-s 7200 --stop-file $STOPF & SW=$!
python3 -I $E1/checks/e1_mem_watch.py --out $CPU/mem_postrun.jsonl --summary $CPU/mem_postrun.json --importer-exe $MEMEXE --stream $DST --report $SCR/recovery.nsys-rep --min-available-gb 64 --max-importer-rss-gb 128 --interval 1 --max-s 7200 --stop-file $STOPF & MW=$!
stop_samplers() { touch $STOPF; wait $SW $MW 2>/dev/null; }
trap stop_samplers EXIT
sleep 3
echo "## A. durable copy of the preserved stream (the tray original stays)"
if [ -n "$SRC" ] && [ -f "$SRC" ]; then
  mkdir -p $RUN/$PASS/preserved; [ -f $DST ] || cp -p "$SRC" $DST
  a=$(sha256sum "$SRC" | cut -d' ' -f1); b=$(sha256sum $DST | cut -d' ' -f1)
  [ "$a" = "$b" ] && echo "PASS  durable copy $DST: $(stat -c %s $DST) B sha256 $b = tray original (retained)" || echo "FAIL  durable copy differs from the tray original ($a vs $b)"
  echo "$b  preserved/$(basename $DST)" > $RUN/$PASS/preserved/SHA256SUMS
elif [ -f $DST ]; then echo "INFO  tray original no longer present; using the existing durable copy $DST"
else echo "FAIL  no preserved stream at '${SRC}' and no durable copy"; fi
echo "## B. configuration identity under the normalization policy (launch record + profile config vs the accepted e1_nsys_val run)"
python3 -I $E1/checks/e1_launch_identity_v3scale.py --a $AR15 --b $RUN/$PASS --policy $E1/checks/e1_identity_policy_v3scale.json --out $CPU/identity_v2_$PASS.json > $CPU/identity_v2_$PASS.txt 2>&1; echo "identity rc=$? ($(grep -c '^NORMALIZED' $CPU/identity_v2_$PASS.txt) normalized fields listed in cpu/identity_v2_$PASS.txt)"
echo "## C. scientific checkers on the in-session profile (lifecycle phase $PHASE)"
uv run --no-sync python $E1/checks/e1_completeness_v2.py --profile $PROF --inputs $INP --plan dry --checksum --out $CPU/v2_$PASS.json > $CPU/v2_$PASS.txt 2>&1; v2=$?
grep -E "^FAIL|^RESULT|cell b=|plan '" $CPU/v2_$PASS.txt | head -12 | cut -c1-200; echo "completeness_v2 (four-wave plan, routing records, checksum) rc=$v2"
( cd $PROF && sha256sum parsed.json parsed.kernels.parquet host_timeline.json | sed "s# # run/$PASS/profile/#" ) > $CPU/${PASS}_extract_inputs.sha256 2>/dev/null
( cd $P1A && $PY -m costmodel.extract --profile $PROF --label $PASS --out $TAB --sha256sums $CPU/${PASS}_extract_inputs.sha256 --sha-prefix run/$PASS/profile/ ) > $CPU/extract_$PASS.log 2>&1; ex=$?
tail -2 $CPU/extract_$PASS.log | cut -c1-200; echo "extract rc=$ex"
uv run --no-sync python $E1/checks/e1_nsys_validation.py --profile $PROF --tables $TAB --inputs $INP --out $CPU/nsys_validation_$PASS.json > $CPU/nsys_validation_$PASS.txt 2>&1; nv=$?
grep -E "^(FAIL|WARN|RESULT)" $CPU/nsys_validation_$PASS.txt | head -12 | cut -c1-200; echo "nsys_validation rc=$nv"
echo "## D. V3 gates"
python3 -I $E1/checks/e1_v3scale_gates.py --run $RUN/$PASS --inputs $INP --pin $E1/V3S-PIN.json --provenance $E1/BINARY-PROVENANCE.json \
  --binary $REPO/alignment/load_generator/req-frontend/target/release/session_runner --importer $IMPORTER --nsys $NSYS --scratch $SCR --durable-stream $DST \
  --identity $CPU/identity_v2_$PASS.json --postrun-storage $CPU/storage_postrun.jsonl --postrun-memory $CPU/mem_postrun.jsonl \
  $([ -f $RUN/$PASS/reservation_deleted.json ] && echo "--reservation-deleted $RUN/$PASS/reservation_deleted.json") \
  --extra completeness_v2=$v2 --extra extract=$ex --extra nsys_validation=$nv ${E1_CLOSEOUT_RC:+--extra closeout=$E1_CLOSEOUT_RC} --out $CPU/V3S-GATES.json | tee $CPU/V3S-GATES.txt; lg=${PIPESTATUS[0]}
rmdir $SCR 2>/dev/null
stop_samplers; trap - EXIT
python3 -I -c "
import json,sys
s=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2]))
print('post-run storage:', s['samples'], 'samples, max gap', s['max_gap_s'] and round(s['max_gap_s'],1), 's;', {g: 'peak %.3f GB' % ((v['peak_bytes'] or 0)/1e9) for g, v in s['groups'].items()})
print('post-run memory:', m['samples'], 'samples; min MemAvailable', m['min_available_kb'] and round(m['min_available_kb']/1048576,1), 'GB; offline importer peak', m['importer_peak_rss_kb'] and round(m['importer_peak_rss_kb']/1048576,2), 'GB over', m['importer_samples'], 'samples')" $CPU/storage_postrun.json $CPU/mem_postrun.json
echo "## verdict"
case $lg in
  0) echo "V3 ACCEPTANCE: PASS (every gate). Any next step needs its own GO. The preserved stream stays on the tray and in the durable copy.";;
  3) echo "V3 ACCEPTANCE: INCONCLUSIVE (a gate could not be evaluated; see $CPU/V3S-GATES.txt). STOP: no retry, no advance; everything stays preserved.";;
  *) echo "V3 ACCEPTANCE: FAIL (see $CPU/V3S-GATES.txt). STOP: no retry, no advance; everything stays preserved.";;
esac
echo "gpu memory after the post-run (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
exit $lg
