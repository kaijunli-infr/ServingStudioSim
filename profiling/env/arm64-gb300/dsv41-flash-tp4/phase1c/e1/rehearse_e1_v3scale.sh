#!/usr/bin/env bash
# CPU-only rehearsal of the V3 acceptance (NOT V3 evidence; no GPU, nothing launched). ON the capture tray as dc (the retained V2 stream
# is on its /tmp), through a neutral-path wrapper:
#   E1_REPO=$S/wt-lifecycle bash $E1/rehearse_e1_v3scale.sh [output dir, default runs/v3scale/rehearsal]
# 1. the complete V3 post-run (postrun_e1_v3scale.sh, real importer and exporter, its own storage and memory samplers) on a copy of
#    the preserved V2 run. V2 was a forced release of half the trace without the V3 observers, so the registered expectation is FAIL
#    with exactly: G0 INCONCLUSIVE (no launch source record), G1 INCONCLUSIVE (no import-probe / memory / storage hosts), G2 FAIL (stop
#    request ended in an error), G3 FAIL (control line, forced release), G4 INCONCLUSIVE (importer boundaries not observed),
#    G5 INCONCLUSIVE (no import-probe record), G8b FAIL (2160 iterations, band 4200–4500), G9 FAIL (two of four cells), G10 INCONCLUSIVE
#    and G12 INCONCLUSIVE (no reservation-deleted record, no closeout code), G11 INCONCLUSIVE (no run-phase samples); G6, G7, G8 PASS.
# 2. the synthetic fixture matrix through the command-line validator (e1_v3scale_rehearsal.py synthetic).
# 3. four-wave scientific completeness and routing records on a synthetic four-wave structure (e1_v3scale_fourwave.py).
# The V2 run directory, archive and retained stream are read only; the fixture copy is removed at the end. Every output is kept.
set -u
export HOME=/mnt/shared/home/dc CUDA_VISIBLE_DEVICES=""; S=$HOME/workspace/serving-studio
export PATH=$HOME/.local/bin:$PATH GB300_ROOT=$S/gb300
REPO=${E1_REPO:-$S/wt-lifecycle}; E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
A=$S/artifacts/phase1c-e1-v2fr-dsv41-flash-gb300-tp4-20261118; AR=$A/run/e1_v2_forced_release
V2Q=/tmp/dsv41_flash_1017_gb300_tp4_e1_v2_forced_release.raw-preserve
OUT=${1:-$E1/runs/v3scale/rehearsal}; mkdir -p $OUT/v2_fixture $OUT/synthetic $OUT/fourwave
BIN=$REPO/alignment/load_generator/req-frontend/target/release/session_runner
PINSHA=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['sha256'])" $E1/BINARY-PROVENANCE.json)
EXPECT="G0.source_and_executable=INCONCLUSIVE,G1.host_identity=INCONCLUSIVE,G2.one_stop_request_returned_ok=FAIL,G3.phases_in_order=FAIL,G4.phase_deadlines=INCONCLUSIVE,G5.importer_observed=INCONCLUSIVE,G8b.recovered_trace_complete=FAIL,G9.scientific_completeness=FAIL,G10.bounds=INCONCLUSIVE,G11.storage_and_memory=INCONCLUSIVE,G12.closeout=INCONCLUSIVE"
fail=0
ok()  { echo "PASS  $*"; }
bad() { echo "FAIL  $*"; fail=1; }
echo "# V3 gate rehearsal $(date -u +%FT%TZ) on $(hostname); tree $REPO at $(git -C $REPO rev-parse HEAD) ($(git -C $REPO status --short --untracked-files=no | wc -l) modified tracked file(s)); CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"
echo "gpu memory before (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
b0=$(sha256sum $BIN | cut -d' ' -f1); [ "$b0" = "$PINSHA" ] && ok "load-generator executable before the rehearsal = pinned $PINSHA" || bad "load-generator executable is $b0, pinned $PINSHA"
echo "## 1. V2 fixture: the retained V2 stream and the archived V2 run (read only)"
q0=$(sha256sum $V2Q/*.qdstrm | cut -d' ' -f1)
[ "$q0" = 4739d6581d1b035bae1f1a0f15a3163bf8e708dc022527fb64228f1143db839c ] && ok "retained V2 stream present (sha256 $q0)" || bad "retained V2 stream missing or changed ($q0)"
R=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v3scale_rehearsal
[ -e $R ] && { bad "fixture directory $R already exists: not touching it"; echo "# result: REHEARSAL NOT RUN"; exit 2; }
mkdir -p $R/e1_v3_scale $R/tables
rsync -a --exclude 'preserved' --exclude 'preflight.log' $AR/ $R/e1_v3_scale/
chmod -R u+w $R
echo "fixture: $(du -sh $R | cut -f1), pass directory renamed to e1_v3_scale; lifecycle record, logs, in-session report and export are the V2 originals, unmodified"
echo "## 2. the complete V3 post-run on the V2 fixture (registered expectation: FAIL; not PASS exactly: $EXPECT)"
E1_REPO=$REPO E1_V3S_RUN=$R E1_V3S_TABLES=$R/tables bash $E1/postrun_e1_v3scale.sh > $OUT/v2_fixture/postrun.log 2>&1; prc=$?
echo "postrun exit code $prc (0 = PASS, 1 = FAIL, 3 = INCONCLUSIVE)"
cp $R/cpu/* $OUT/v2_fixture/ 2>/dev/null; rm -f $OUT/v2_fixture/postrun_samplers.stop
grep -E "^## |^PASS|^FAIL|^INFO|rc=|RESULT|V3 ACCEPTANCE|gpu memory|post-run (storage|memory)|^(PASS|FAIL|INCONCLUSIVE) +G|^# V3" $OUT/v2_fixture/postrun.log | cut -c1-330
python3 -I $E1/e1_v3scale_rehearsal.py expect --gates $OUT/v2_fixture/V3S-GATES.json --verdict FAIL --expect-not-pass "$EXPECT" --out $OUT/v2_fixture/V2-FIXTURE-EXPECTATION.json | tee $OUT/v2_fixture/V2-FIXTURE-EXPECTATION.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ "$prc" = 1 ] && ok "V2 fixture: verdict FAIL with exactly the registered gates not passing" || bad "V2 fixture expectation not met (see above)"
case "$R" in */logs/dsv41_flash_gb300_tp4_e1_v3scale_rehearsal) rm -rf $R; echo "fixture copy removed";; esac
ls -d /tmp/v3scale-recovery-* 2>/dev/null && bad "post-run scratch left in /tmp" || ok "no post-run scratch left in /tmp"
q1=$(sha256sum $V2Q/*.qdstrm | cut -d' ' -f1); [ "$q0" = "$q1" ] && ok "retained V2 stream unchanged (sha256 $q1)" || bad "retained V2 stream changed"
( cd $A && sha256sum -c SHA256SUMS --quiet ) && ok "V2 archive re-verified against its SHA256SUMS" || bad "V2 archive differs from its SHA256SUMS"
echo "## 3. synthetic fixtures through the command-line validator"
python3 -I $E1/e1_v3scale_rehearsal.py synthetic --out $OUT/synthetic > $OUT/synthetic/run.log 2>&1; src=$?
head -2 $OUT/synthetic/SUMMARY.txt 2>/dev/null | cut -c1-260; grep "^NOT MET" $OUT/synthetic/SUMMARY.txt 2>/dev/null | cut -c1-400
[ "$src" = 0 ] && ok "synthetic fixtures: every expectation met" || bad "synthetic fixtures: exit code $src (see $OUT/synthetic/run.log)"
echo "## 4. four-wave completeness and routing records (synthetic four-wave structure from the V2 capture)"
$S/gb300/venv/bin/python $E1/e1_v3scale_fourwave.py --v2-profile $AR/profile --out $OUT/fourwave > $OUT/fourwave/run.log 2>&1; frc=$?
cat $OUT/fourwave/SUMMARY.txt 2>/dev/null | cut -c1-330 || tail -5 $OUT/fourwave/run.log
[ "$frc" = 0 ] && ok "four-wave fixtures: every expectation met" || bad "four-wave fixtures: exit code $frc (see $OUT/fourwave/run.log)"
echo "## 5. state after the rehearsal"
b1=$(sha256sum $BIN | cut -d' ' -f1); [ "$b1" = "$PINSHA" ] && ok "load-generator executable after the rehearsal = pinned $PINSHA" || bad "load-generator executable is $b1, pinned $PINSHA"
echo "gpu memory after (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
echo "# result: $([ $fail = 0 ] && echo REHEARSAL EXPECTATIONS MET || echo REHEARSAL EXPECTATIONS NOT MET)"
exit $fail
