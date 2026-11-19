#!/usr/bin/env bash
# CPU-only rehearsal of the V4 acceptance path (NOT V4 evidence; no GPU, nothing launched). ON the capture tray as dc (the retained
# V3 stream is on its /tmp), through a neutral-path wrapper:
#   E1_REPO=$S/wt-lifecycle bash $E1/rehearse_e1_v4scale.sh [output dir, default runs/v4scale/rehearsal]
# 1. the V4 identity policy on the ACTUAL generated V4 config and trace with the predicted V4 launch record
#    (e1_v4scale_identity_rehearsal.py): PASS with the full four-copy trace proof; the V1/V2 policy FAILs on the same records.
# 2. the COMPLETE V4 post-run (postrun_e1_v4scale.sh: durable copy, V4 identity step, scientific checkers, real importer and exporter,
#    its own storage and memory samplers, the V4 gates G0-G16) on a V4-shaped run directory: a copy of the archived V3 run in which
#    the profile config is the actual V4 config byte for byte, the trace is the actual V4 trace and the launch record is the predicted
#    V4 launch record. The capture data are V3's (four waves, 24.5 M rows), so the registered expectation is FAIL with exactly:
#    G8b FAIL (4320 iterations and 24.5 M rows against the V4 bands), G9 FAIL (four of the eight planned cells), G15 FAIL (four of
#    eight cells, 112 of 224 requests), G16 FAIL (the V3 run has no watcher and no deleter record: its reservation was deleted by hand,
#    which V4 does not accept). Every other gate PASS, in particular G13 (V4 identity policy, trace proof complete), G14
#    (four-copy trace and V4 config in the run directory), G11 (storage and memory within the V4 ceilings) and G0-G7, G10, G12.
#    This is the step whose absence produced the V3 failure: the V4 post-run, including configuration identity, run against the V4
#    configuration.
# 3. the synthetic fixture matrix through the command-line validator (e1_v4scale_rehearsal.py synthetic): every ceiling and deadline
#    at its boundary, one registered defect per negative fixture.
# 4. eight-wave scientific completeness and routing records on a synthetic eight-wave structure (e1_v4scale_eightwave.py).
# The V3 run directory, archive and retained stream are read only; the fixture copy is removed at the end. Every output is kept.
set -u
export HOME=/mnt/shared/home/dc CUDA_VISIBLE_DEVICES=""; S=$HOME/workspace/serving-studio
export PATH=$HOME/.local/bin:$PATH GB300_ROOT=$S/gb300
REPO=${E1_REPO:-$S/wt-lifecycle}; E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1; PASS=e1_v4_scale
A=$S/artifacts/phase1c-e1-v3scale-dsv41-flash-gb300-tp4-20261118; AR=$A/run/e1_v3_scale
A2=$S/artifacts/phase1c-e1-v2fr-dsv41-flash-gb300-tp4-20261118/run/e1_v2_forced_release
AR15=$S/artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115/run/e1_nsys_val
V3Q=/tmp/dsv41_flash_1017_gb300_tp4_e1_v3_scale.raw-preserve; V3SHA=764d063d951d4c4683f1b4bea8336652846459de6b73ffb953a49d30744f8ba3
OUT=${1:-$E1/runs/v4scale/rehearsal}; mkdir -p $OUT/identity $OUT/v4_config_postrun $OUT/synthetic $OUT/eightwave
BIN=$REPO/alignment/load_generator/req-frontend/target/release/session_runner
PINSHA=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['sha256'])" $E1/BINARY-PROVENANCE.json)
EXPECT="G8b.recovered_trace_complete=FAIL,G9.scientific_completeness=FAIL,G15.request_iteration_routing_rank_coverage=FAIL,G16.reservation_deletion=FAIL"
fail=0
ok()  { echo "PASS  $*"; }
bad() { echo "FAIL  $*"; fail=1; }
echo "# V4 gate rehearsal $(date -u +%FT%TZ) on $(hostname); tree $REPO at $(git -C $REPO rev-parse HEAD) ($(git -C $REPO status --short --untracked-files=no | wc -l) modified tracked file(s)); CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"
echo "gpu before (MiB / compute processes): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')/ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
b0=$(sha256sum $BIN | cut -d' ' -f1); [ "$b0" = "$PINSHA" ] && ok "load-generator executable before the rehearsal = pinned $PINSHA" || bad "load-generator executable is $b0, pinned $PINSHA"
python3 -I $E1/make_e1_inputs_v4scale.py --check && ok "the V4 trace, config, inputs, manifest and identity policy on disk are what the generator produces" || bad "generated V4 inputs differ from the files on disk"
echo "## 1. V4 identity policy on the actual V4 configuration"
python3 -I $E1/e1_v4scale_identity_rehearsal.py --reference $AR15 --out $OUT/identity > $OUT/identity/run.log 2>&1; irc=$?
grep -E "^(PASS|FAIL) |^# " $OUT/identity/run.log | cut -c1-300
[ "$irc" = 0 ] && ok "identity rehearsal: every expectation met" || bad "identity rehearsal: exit code $irc (see $OUT/identity/run.log)"
echo "## 2. the complete V4 post-run on a V4-shaped run directory (V3 capture data, actual V4 config, V4 trace, predicted V4 launch record)"
q0=$(sha256sum $V3Q/*.qdstrm | cut -d' ' -f1)
[ "$q0" = "$V3SHA" ] && ok "retained V3 stream present (sha256 $q0)" || bad "retained V3 stream missing or changed ($q0)"
R=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v4scale_rehearsal
[ -e $R ] || [ -e /tmp/v4scale-recovery-$PASS ] && { bad "fixture directory $R or post-run scratch already exists: not touching it"; echo "# result: REHEARSAL NOT RUN"; exit 2; }
mkdir -p $R/$PASS $R/tables
rsync -a --exclude 'preserved' --exclude 'preflight.log' $AR/ $R/$PASS/
chmod -R u+w $R
cp $E1/configs_v4scale/$PASS/profile.json $R/$PASS/profile.json; cp $E1/traces/e1_v4_scale.csv $R/$PASS/trace.csv
rm -f $R/$PASS/profile/vllm/*_launch.json; cp $REPO/tests/fixtures/e1_v4_scale/launch.predicted.json $R/$PASS/profile/vllm/dsv41_flash_1017_gb300_tp4_${PASS}_launch.json
[ "$(sha256sum $R/$PASS/profile.json | cut -d' ' -f1)" = "$(sha256sum $E1/configs_v4scale/$PASS/profile.json | cut -d' ' -f1)" ] && [ "$(ls $R/$PASS/profile/vllm/*_launch.json | wc -l)" = 1 ] && ok "fixture: $(du -sh $R | cut -f1); profile.json = the actual V4 config (sha256 $(sha256sum $R/$PASS/profile.json | cut -c1-16)…), trace.csv = the V4 trace, one launch record = the predicted V4 one; lifecycle record, logs, in-session report and export are the V3 originals" || bad "fixture preparation"
CRC=$(cat $A/operator/closeout.rc 2>/dev/null)
echo "registered expectation: verdict FAIL; not PASS exactly: $EXPECT"
E1_REPO=$REPO E1_V4S_RUN=$R E1_V4S_TABLES=$R/tables E1_CLOSEOUT_RC=${CRC:-0} bash $E1/postrun_e1_v4scale.sh > $OUT/v4_config_postrun/postrun.log 2>&1; prc=$?
echo "postrun exit code $prc (0 = PASS, 1 = FAIL, 3 = INCONCLUSIVE)"
cp $R/cpu/* $OUT/v4_config_postrun/ 2>/dev/null; rm -f $OUT/v4_config_postrun/postrun_samplers.stop
grep -E "^## |^PASS|^FAIL|^INFO|rc=|RESULT|V4 ACCEPTANCE|gpu memory|post-run (storage|memory)|^(PASS|FAIL|INCONCLUSIVE) +G|^# V4" $OUT/v4_config_postrun/postrun.log | cut -c1-330
grep -E "^# configuration identity|^TRACE FAIL|^FAIL" $OUT/v4_config_postrun/identity_v4scale_$PASS.txt | cut -c1-260; echo "trace checks passing: $(grep -c '^TRACE PASS' $OUT/v4_config_postrun/identity_v4scale_$PASS.txt) of $(grep -c '^TRACE ' $OUT/v4_config_postrun/identity_v4scale_$PASS.txt)"
python3 -I $E1/e1_v4scale_rehearsal.py expect --gates $OUT/v4_config_postrun/V4S-GATES.json --verdict FAIL --expect-not-pass "$EXPECT" --out $OUT/v4_config_postrun/V4-CONFIG-EXPECTATION.json | tee $OUT/v4_config_postrun/V4-CONFIG-EXPECTATION.txt | cut -c1-400
[ "${PIPESTATUS[0]}" = 0 ] && [ "$prc" = 1 ] && grep -q "^# configuration identity v4scale: PASS" $OUT/v4_config_postrun/identity_v4scale_$PASS.txt && ok "V4 post-run on the V4 configuration: identity PASS under the V4 policy; verdict FAIL with exactly the registered size-dependent gates not passing" || bad "V4 configuration post-run expectation not met (see above)"
case "$R" in */logs/dsv41_flash_gb300_tp4_e1_v4scale_rehearsal) rm -rf $R; echo "fixture copy removed";; esac
ls -d /tmp/v4scale-recovery-* 2>/dev/null && bad "post-run scratch left in /tmp" || ok "no post-run scratch left in /tmp"
q1=$(sha256sum $V3Q/*.qdstrm | cut -d' ' -f1); [ "$q0" = "$q1" ] && ok "retained V3 stream unchanged (sha256 $q1)" || bad "retained V3 stream changed"
( cd $A && sha256sum -c SHA256SUMS --quiet ) && ok "V3 archive re-verified against its SHA256SUMS" || bad "V3 archive differs from its SHA256SUMS"
echo "## 3. synthetic fixtures through the command-line validator"
python3 -I $E1/e1_v4scale_rehearsal.py synthetic --out $OUT/synthetic > $OUT/synthetic/run.log 2>&1; src=$?
head -2 $OUT/synthetic/SUMMARY.txt 2>/dev/null | cut -c1-260; grep "^NOT MET" $OUT/synthetic/SUMMARY.txt 2>/dev/null | cut -c1-400
[ "$src" = 0 ] && ok "synthetic fixtures: every expectation met" || bad "synthetic fixtures: exit code $src (see $OUT/synthetic/run.log)"
echo "## 4. eight-wave completeness and routing records (synthetic eight-wave structure from the V2 capture)"
$S/gb300/venv/bin/python $E1/e1_v4scale_eightwave.py --v2-profile $A2/profile --out $OUT/eightwave > $OUT/eightwave/run.log 2>&1; frc=$?
cat $OUT/eightwave/SUMMARY.txt 2>/dev/null | cut -c1-330 || tail -5 $OUT/eightwave/run.log
[ "$frc" = 0 ] && ok "eight-wave fixtures: every expectation met" || bad "eight-wave fixtures: exit code $frc (see $OUT/eightwave/run.log)"
echo "## 5. state after the rehearsal"
b1=$(sha256sum $BIN | cut -d' ' -f1); [ "$b1" = "$PINSHA" ] && ok "load-generator executable after the rehearsal = pinned $PINSHA" || bad "load-generator executable is $b1, pinned $PINSHA"
echo "gpu after (MiB / compute processes): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')/ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
echo "# result: $([ $fail = 0 ] && echo REHEARSAL EXPECTATIONS MET || echo REHEARSAL EXPECTATIONS NOT MET)"
exit $fail
