#!/usr/bin/env bash
# Retrospective, CPU-only replay of the V3 post-run on PRESERVED EVIDENCE ONLY (no GPU, nothing launched, no new capture). ON the
# capture tray as dc (the tray-preserved V3 stream is on its /tmp), through a neutral-path wrapper:
#   E1_REPO=$S/wt-lifecycle bash $E1/retro_e1_v3scale.sh [output dir, default runs/v3scale/retro]
# It does not touch the original V3 verdict (FAIL under the frozen gates), its report, its gate outputs or the frozen packet; it checks
# that they are unchanged. Steps:
#   1. freeze check: archive against its SHA256SUMS; stream hash in every copy; committed V3 report, evidence, packet and manifests
#      unchanged since the run's commit; frozen V1/V2/V3 manifests verify
#   2. independent verification of the trace and of the launch / profile fields (e1_v3scale_identity_independent.py)
#   3. reproduction: the FROZEN post-run, unchanged, on a copy of the archived run directory. Registered expectation: FAIL with
#      exactly G9 not passing, as in the original
#   4. corrected assessment: the same post-run script with ONE line changed (the identity step uses e1_launch_identity_v3scale.py
#      with e1_identity_policy_v3scale.json); the one-line difference is checked and recorded. Its verdict is whatever the gates give.
# The replay root (a copy of the archive) is removed at the end; the archive and the tray stream are read only. Every output is kept.
set -u
export HOME=/mnt/shared/home/dc CUDA_VISIBLE_DEVICES=""; S=$HOME/workspace/serving-studio
export PATH=$HOME/.local/bin:$PATH GB300_ROOT=$S/gb300
REPO=${E1_REPO:-$S/wt-lifecycle}; E1R=profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1; E1=$REPO/$E1R; PASS=e1_v3_scale
A=$S/artifacts/phase1c-e1-v3scale-dsv41-flash-gb300-tp4-20261118; AR15=$S/artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115/run/e1_nsys_val
TP=/tmp/dsv41_flash_1017_gb300_tp4_e1_v3_scale.raw-preserve; STREAM_SHA=764d063d951d4c4683f1b4bea8336652846459de6b73ffb953a49d30744f8ba3; RUNCOMMIT=5a01f642
OUT=${1:-$E1/runs/v3scale/retro}; mkdir -p $OUT/reproduction $OUT/corrected $OUT/independent
R=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v3scale_replay
BIN=$REPO/alignment/load_generator/req-frontend/target/release/session_runner
fail=0
ok()  { echo "PASS  $*"; }
bad() { echo "FAIL  $*"; fail=1; }
echo "# V3 retrospective replay $(date -u +%FT%TZ) on $(hostname); tree $REPO at $(git -C $REPO rev-parse HEAD) ($(git -C $REPO status --short --untracked-files=no | wc -l) modified tracked file(s)); CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"
echo "gpu before (MiB / compute processes): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')/ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
echo "## 1. the incident as preserved"
( cd $A && sha256sum -c SHA256SUMS --quiet ) && ok "archive: all $(wc -l < $A/SHA256SUMS) files match SHA256SUMS (its own sha256 $(sha256sum $A/SHA256SUMS | cut -d' ' -f1))" || bad "archive differs from its SHA256SUMS"
[ "$(sha256sum $A/SHA256SUMS | cut -d' ' -f1)" = "$(sha256sum $E1/runs/v3scale/run/evidence/ARCHIVE-SHA256SUMS | cut -d' ' -f1)" ] && ok "the committed ARCHIVE-SHA256SUMS is the archive's SHA256SUMS" || bad "committed ARCHIVE-SHA256SUMS differs from the archive's"
n=0; for f in $TP/*.qdstrm $A/preserved/*.qdstrm $A/run/$PASS/preserved/*.qdstrm; do h=$(sha256sum $f | cut -d' ' -f1); n=$((n+1)); [ "$h" = "$STREAM_SHA" ] || bad "stream $f is $h"; done; [ $n = 3 ] && [ $fail = 0 ] && ok "raw stream sha256 $STREAM_SHA in all three copies (tray original, archive, archived durable copy) = the controller's final record $(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['stream']['final']['sha256'][:16])" $A/run/$PASS/profile/nsys/lifecycle.json)…" || bad "stream copies: $n found"
[ -z "$(git -C $REPO diff --stat $RUNCOMMIT HEAD -- $E1R/runs/v3scale/run $E1R/E1-V3-SCALE-PACKET.md $E1R/FREEZE_v3scale.sha256 $E1R/SCRIPTS_v3scale.sha256 $E1R/INPUTS_v3scale.sha256 $E1R/E1-V3-SCALE-INPUTS.json $E1R/V3S-PIN.json $E1R/V3S-PIN-identities.json $E1R/checks/e1_v3scale_gates.py $E1R/postrun_e1_v3scale.sh $E1R/checks/e1_launch_identity_v2.py $E1R/checks/e1_identity_policy_v1.json | grep -v "E1-V3-REVIEW-ADDENDUM.md")" ] && ok "original V3 report, evidence (gate outputs, identity report), frozen packet, manifests, gate code, post-run script and the V1/V2 identity checker and policy: unchanged since the run's commit $RUNCOMMIT (only the separate addendum was added)" || { bad "a preserved V3 file changed since $RUNCOMMIT:"; git -C $REPO diff --stat $RUNCOMMIT HEAD -- $E1R/runs/v3scale/run $E1R/E1-V3-SCALE-PACKET.md | tail -5; }
for f in V3S-GATES.json V3S-GATES.txt identity_v2_${PASS}.json identity_v2_${PASS}.txt; do [ "$(sha256sum $A/run/cpu/$f | cut -d' ' -f1)" = "$(sha256sum $E1/runs/v3scale/run/evidence/cpu/$f | cut -d' ' -f1)" ] || bad "committed $f differs from the archived original"; done; [ $fail = 0 ] && ok "committed original gate outputs and identity report are byte-identical to the archived ones (V3S-GATES.json sha256 $(sha256sum $A/run/cpu/V3S-GATES.json | cut -c1-16)…)"
( cd $E1 && sha256sum -c FREEZE_v3scale.sha256 --quiet && sha256sum -c SCRIPTS_v3scale.sha256 --quiet && sha256sum -c INPUTS_v3scale.sha256 --quiet && sha256sum -c FREEZE_fr.sha256 --quiet && sha256sum -c SCRIPTS_fr.sha256 --quiet && sha256sum -c SCRIPTS_v1.sha256 --quiet ) && ok "frozen V3 manifests (freeze, 66 scripts, inputs) and the V2 / V1 manifests verify" || bad "a frozen manifest does not verify"
python3 -I -c "
import json,sys
d=json.load(open(sys.argv[1])); g={x['gate']: x['status'] for x in d['gates']}
print('  original gate outputs:', d['verdict'], {k: v for k, v in g.items() if v != 'PASS'}, '|', sum(v == 'PASS' for v in g.values()), 'of', len(g), 'PASS')
sys.exit(0 if d['verdict'] == 'FAIL' and {k: v for k, v in g.items() if v != 'PASS'} == {'G9.scientific_completeness': 'FAIL'} else 1)" $A/run/cpu/V3S-GATES.json && ok "original V3 verdict as archived: FAIL, G9 only" || bad "archived original verdict is not FAIL on G9 only"
b0=$(sha256sum $BIN | cut -d' ' -f1); [ "$b0" = 7f6eb81d45e6b98f3b6ebb56ab20b60ce8aaf4584aa3d4f25e3449b2d05a00c2 ] && ok "load-generator executable = pinned" || bad "executable $b0"
echo "## 2. independent verification (no code shared with the identity checkers)"
python3 -I $E1/e1_v3scale_identity_independent.py --reference $AR15 --run $A/run/$PASS --original-report $A/run/cpu/identity_v2_$PASS.json --out $OUT/independent | cut -c1-330
[ "${PIPESTATUS[0]}" = 0 ] && ok "independent verification: everything agrees" || bad "independent verification (see $OUT/independent)"
echo "## 3. reproduction: the frozen post-run, unchanged, on a copy of the archived run directory"
[ -e $R ] || [ -e /tmp/v3scale-recovery-$PASS ] && { bad "replay root $R or post-run scratch exists: not touching it"; echo "# result: RETROSPECTIVE REPLAY NOT RUN"; exit 2; }
mkdir -p $R && cp -a $A/run/. $R/ && chmod -R u+w $R && mv $R/cpu $R/cpu_original && rm -rf $R/preflight
CRC=$(cat $A/operator/closeout.rc)
echo "replay root: copy of the archive ($(du -sh $R | cut -f1)); original closeout exit code $CRC (from the archived operator record)"
env -u E1_V3S_MODE E1_REPO=$REPO E1_V3S_RUN=$R E1_V3S_TABLES=$R/tables E1_CLOSEOUT_RC=$CRC bash $E1/postrun_e1_v3scale.sh > $OUT/reproduction/postrun.log 2>&1; prc=$?
cp $R/cpu/* $OUT/reproduction/ 2>/dev/null; rm -f $OUT/reproduction/postrun_samplers.stop
grep -E "identity rc|rc=|^# V3 gates|V3 ACCEPTANCE|post-run (storage|memory)" $OUT/reproduction/postrun.log | cut -c1-260
python3 -I $E1/e1_v3scale_rehearsal.py expect --gates $OUT/reproduction/V3S-GATES.json --verdict FAIL --expect-not-pass "G9.scientific_completeness=FAIL" --out $OUT/reproduction/REPRODUCTION-EXPECTATION.json | tee $OUT/reproduction/REPRODUCTION-EXPECTATION.txt | cut -c1-260
[ "${PIPESTATUS[0]}" = 0 ] && [ "$prc" = 1 ] && grep -q "FAIL         profile.workload.frontend.path: rule condition not met: sha256 a=c6d8e13b7229ffd4 b=d9d2dc77b316b786 pinned=c6d8e13b7229ffd4" $OUT/reproduction/identity_v2_$PASS.txt && ok "the original result is reproducible: frozen post-run exit 1, verdict FAIL, exactly G9 not passing, on the same trace-hash rule" || bad "the original result was not reproduced (post-run exit $prc)"
mv $R/cpu $R/cpu_reproduction; rm -rf $R/tables
echo "## 4. corrected assessment: the same post-run with the identity step replaced"
C=$OUT/corrected/postrun_e1_v3scale_corrected.sh
sed -e 's#\$E1/checks/e1_launch_identity_v2.py --a \$AR15 --b \$RUN/\$PASS --policy \$E1/checks/e1_identity_policy_v1.json#$E1/checks/e1_launch_identity_v3scale.py --a $AR15 --b $RUN/$PASS --policy $E1/checks/e1_identity_policy_v3scale.json#' $E1/postrun_e1_v3scale.sh > $C
diff $E1/postrun_e1_v3scale.sh $C > $OUT/corrected/postrun_one_line_difference.diff; nd=$(grep -c '^[<>]' $OUT/corrected/postrun_one_line_difference.diff)
[ "$nd" = 2 ] && grep -q "^> .*e1_launch_identity_v3scale.py --a \$AR15 --b \$RUN/\$PASS --policy \$E1/checks/e1_identity_policy_v3scale.json" $OUT/corrected/postrun_one_line_difference.diff && ok "the corrected post-run differs from the frozen one in exactly one line: the identity checker and its policy (postrun_one_line_difference.diff); gate code, checkers, inputs and thresholds are the frozen ones" || bad "the corrected post-run differs in $nd lines"
env -u E1_V3S_MODE E1_REPO=$REPO E1_V3S_RUN=$R E1_V3S_TABLES=$R/tables E1_CLOSEOUT_RC=$CRC bash $C > $OUT/corrected/postrun.log 2>&1; qrc=$?
cp $R/cpu/* $OUT/corrected/ 2>/dev/null; rm -f $OUT/corrected/postrun_samplers.stop
grep -E "identity rc|completeness_v2|extract rc|nsys_validation rc|^(PASS|FAIL|INCONCLUSIVE) +G|^# V3 gates|post-run (storage|memory)" $OUT/corrected/postrun.log | cut -c1-230
grep -E "^# configuration identity|^TRACE FAIL|^FAIL" $OUT/corrected/identity_v2_$PASS.txt | cut -c1-260; echo "trace checks passing: $(grep -c '^TRACE PASS' $OUT/corrected/identity_v2_$PASS.txt) of $(grep -c '^TRACE ' $OUT/corrected/identity_v2_$PASS.txt)"
case $qrc in 0) v=PASS;; 3) v=INCONCLUSIVE;; *) v=FAIL;; esac
python3 -I -c "
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); ga={x['gate']: x['status'] for x in a['gates']}; gb={x['gate']: x['status'] for x in b['gates']}
print('  gates whose status differs between the reproduction and the corrected assessment:', {k: (ga.get(k), gb.get(k)) for k in sorted(set(ga) | set(gb)) if ga.get(k) != gb.get(k)})
print('  corrected assessment: verdict', b['verdict'], '| not PASS', {k: v for k, v in gb.items() if v != 'PASS'}, '|', sum(v == 'PASS' for v in gb.values()), 'of', len(gb), 'PASS')" $OUT/reproduction/V3S-GATES.json $OUT/corrected/V3S-GATES.json
echo "RETROSPECTIVE V3 EVIDENCE ASSESSMENT under the corrected identity policy: $v (post-run exit $qrc). The original V3 verdict stays FAIL under the frozen gates."
printf '{"original_v3_verdict_under_frozen_gates": "FAIL", "original_reproduced": %s, "retrospective_assessment_under_corrected_identity_policy": "%s", "corrected_postrun_exit": %s, "note": "two separate verdicts; the retrospective assessment does not replace the original"}\n' "$([ $prc = 1 ] && echo true || echo false)" "$v" "$qrc" > $OUT/VERDICTS.json
case "$R" in */logs/dsv41_flash_gb300_tp4_e1_v3scale_replay) rm -rf $R; echo "replay root removed";; esac
echo "## 5. state after the replay"
ls -d /tmp/v3scale-recovery-* 2>/dev/null && bad "post-run scratch left in /tmp" || ok "no post-run scratch left in /tmp"
[ "$(sha256sum $TP/*.qdstrm | cut -d' ' -f1)" = "$STREAM_SHA" ] && ( cd $A && sha256sum -c SHA256SUMS --quiet ) && ok "tray stream and archive unchanged after the replay" || bad "tray stream or archive changed"
[ "$(sha256sum $BIN | cut -d' ' -f1)" = "$b0" ] && ok "load-generator executable unchanged" || bad "executable changed"
echo "gpu after (MiB / compute processes): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')/ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
echo "# result: $([ $fail = 0 ] && echo "REPLAY CHECKS MET (retrospective assessment: $v)" || echo "REPLAY CHECKS NOT MET")"
exit $fail
