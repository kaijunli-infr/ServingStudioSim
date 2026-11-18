#!/usr/bin/env bash
# Final CPU-only rehearsal of the V2 forced-release acceptance (NOT V2 evidence; no GPU, nothing launched). ON the capture tray as dc
# (the retained V1 stream is on its /tmp), through a neutral-path wrapper:
#   E1_REPO=$S/wt-lifecycle bash $E1/rehearse_e1_fr.sh [output dir, default runs/v2/rehearsal_final]
# 1. the complete pinned V2 post-run (postrun_e1_fr.sh) on a fixture built from the preserved V1 run. Expected: FAIL or INCONCLUSIVE,
#    because V1 forced no release during its import (e1_fr_rehearsal.py expect-v1 checks exactly that).
# 2. the synthetic fixture matrix through the same command-line validators (e1_fr_rehearsal.py synthetic): the positive fixtures
#    pass every gate; every negative fixture stops at its registered gate.
# The V1 run directory, archive and retained stream are read only; the fixture copy is removed at the end. Every output is kept.
set -u
export HOME=/mnt/shared/home/dc CUDA_VISIBLE_DEVICES=""; S=$HOME/workspace/serving-studio
export PATH=$HOME/.local/bin:$PATH GB300_ROOT=$S/gb300
REPO=${E1_REPO:-$S/wt-lifecycle}; E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
A=$S/artifacts/phase1c-e1-v1-dsv41-flash-gb300-tp4-20261118; AR=$A/run/e1_v1_lifecycle
V1Q=/tmp/dsv41_flash_1017_gb300_tp4_e1_v1_lifecycle.raw-preserve
OUT=${1:-$E1/runs/v2/rehearsal_final}; mkdir -p $OUT/v1_fixture $OUT/synthetic
BIN=$REPO/alignment/load_generator/req-frontend/target/release/session_runner
PINSHA=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['sha256'])" $E1/BINARY-PROVENANCE.json)
fail=0
ok()  { echo "PASS  $*"; }
bad() { echo "FAIL  $*"; fail=1; }
echo "# final V2 gate rehearsal $(date -u +%FT%TZ) on $(hostname); tree $REPO at $(git -C $REPO rev-parse HEAD) ($(git -C $REPO status --short --untracked-files=no | wc -l) modified tracked file(s)); CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"
echo "gpu memory before (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
b0=$(sha256sum $BIN | cut -d' ' -f1); [ "$b0" = "$PINSHA" ] && ok "load-generator executable before the rehearsal = pinned $PINSHA" || bad "load-generator executable is $b0, pinned $PINSHA"
echo "## 1. V1 fixture: the retained V1 stream and the archived V1 run (read only)"
q0=$(sha256sum $V1Q/*.qdstrm | cut -d' ' -f1)
[ -f $V1Q/RETENTION.json ] && ok "retained V1 stream present with its retention record (sha256 $q0)" || bad "retained V1 stream or its retention record missing in $V1Q"
R=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v2fr_rehearsal
[ -e $R ] && { bad "fixture directory $R already exists: not touching it"; echo "# result: REHEARSAL NOT RUN"; exit 2; }
mkdir -p $R/e1_v2_forced_release $R/tables
rsync -a --exclude '*.nsys-rep' --exclude '*.sqlite' --exclude 'parsed.json' --exclude 'parsed.kernels.parquet' --exclude 'profile_result.json' --exclude 'preflight.log' $AR/ $R/e1_v2_forced_release/
chmod -R u+w $R
python3 -I - $R/e1_v2_forced_release $V1Q <<'PY'
import glob, hashlib, json, os, sys
run, q = sys.argv[1:3]
cfg = json.load(open(run + "/profile.json")); cfg["log_dir"] = run + "/profile"; json.dump(cfg, open(run + "/profile.json", "w"), indent=1)
p = run + "/profile/nsys/lifecycle.json"; d = json.load(open(p))
stream = glob.glob(q + "/*.qdstrm")[0]
digest = hashlib.sha256()
with open(stream, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        digest.update(chunk)
# outcome B shape: the report and export are withheld from the fixture, so the profile must be recovered from the stream
d["phase"] = "stop_failed"; d["phase_history"] = [h for h in d["phase_history"] if h["phase"] != "report_written"] + [{"phase": "stop_failed", "at": d["session_closed_at"]}]
d["report"] = {"path": None, "size": None, "written_at": None}
d["stream"]["final"] = {"available": True, "size": os.path.getsize(stream), "sha256": digest.hexdigest(), "taken_at": d["session_closed_at"] + 1, "session_closed_at": d["session_closed_at"], "note": "REHEARSAL FIXTURE: injected, the V1 controller (schema 1) wrote no final record"}
d["rehearsal_fixture"] = "built from the V1 record; NOT a V2 run. Nothing about a release was added: V1 forced none."
json.dump(d, open(p, "w"), indent=1)
print("fixture lifecycle: phase", d["phase"], "| release requested", (d.get("gpu") or {}).get("release_requested_at"), "| control commands", [c.get("command") for c in d.get("control_commands", [])])
PY
echo "## 2. the complete pinned V2 post-run on the V1 fixture"
E1_REPO=$REPO E1_FR_RUN=$R E1_FR_TABLES=$R/tables bash $E1/postrun_e1_fr.sh > $OUT/v1_fixture/postrun.log 2>&1; prc=$?
echo "postrun exit code $prc (0 = PASS, 1 = FAIL, 3 = INCONCLUSIVE)"
cp $R/cpu/* $OUT/v1_fixture/ 2>/dev/null; cp $R/e1_v2_forced_release/profile/nsys/lifecycle.json $OUT/v1_fixture/fixture_lifecycle.json
grep -E "^## |^PASS|^FAIL|^INFO|rc=|RESULT|V2 ACCEPTANCE|gpu memory|^(PASS|FAIL|INCONCLUSIVE) +(FR|H\.|S\.)|^# V2" $OUT/v1_fixture/postrun.log | cut -c1-300
[ "$prc" = 1 ] || [ "$prc" = 3 ] && ok "the V2 post-run does not pass on V1 (exit code $prc)" || bad "the V2 post-run exit code on the V1 fixture is $prc"
echo "## 3. why it does not pass (e1_fr_rehearsal.py expect-v1)"
python3 -I $E1/e1_fr_rehearsal.py expect-v1 --gates $OUT/v1_fixture/V2-GATES.json --lifecycle $OUT/v1_fixture/fixture_lifecycle.json --out $OUT/v1_fixture/V1-FIXTURE-EXPECTATION.json | tee $OUT/v1_fixture/V1-FIXTURE-EXPECTATION.txt
[ "${PIPESTATUS[0]}" = 0 ] && ok "V1 fixture: FAIL/INCONCLUSIVE specifically because no release was forced during the import" || bad "V1 fixture expectation not met (see above)"
case "$R" in */logs/dsv41_flash_gb300_tp4_e1_v2fr_rehearsal) du -sh $R | cut -f1 | sed 's/^/fixture size /'; rm -rf $R; echo "fixture copy removed";; esac
q1=$(sha256sum $V1Q/*.qdstrm | cut -d' ' -f1); [ "$q0" = "$q1" ] && ok "retained V1 stream unchanged (sha256 $q1)" || bad "retained V1 stream changed"
( cd $A && sha256sum -c SHA256SUMS --quiet ) && ok "V1 archive re-verified against its SHA256SUMS" || bad "V1 archive differs from its SHA256SUMS"
echo "## 4. synthetic fixtures through the command-line validators"
( cd $REPO && source profiling/env/arm64-gb300/gb300.env >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""; uv run --no-sync python $E1/e1_fr_rehearsal.py synthetic --out $OUT/synthetic ) > $OUT/synthetic/run.log 2>&1; src=$?
cat $OUT/synthetic/SUMMARY.txt 2>/dev/null | cut -c1-260 || tail -5 $OUT/synthetic/run.log
[ "$src" = 0 ] && ok "synthetic fixtures: every expectation met" || bad "synthetic fixtures: exit code $src (see $OUT/synthetic/run.log)"
echo "## 5. state after the rehearsal"
b1=$(sha256sum $BIN | cut -d' ' -f1); [ "$b1" = "$PINSHA" ] && ok "load-generator executable after the rehearsal = pinned $PINSHA" || bad "load-generator executable is $b1, pinned $PINSHA"
echo "gpu memory after (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
echo "# result: $([ $fail = 0 ] && echo REHEARSAL EXPECTATIONS MET || echo REHEARSAL EXPECTATIONS NOT MET)"
exit $fail
