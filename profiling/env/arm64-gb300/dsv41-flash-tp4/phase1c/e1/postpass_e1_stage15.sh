#!/usr/bin/env bash
# Stage 1.5 gate after each pass (CPU, read-only on the run): corrected completeness checker (PASS required to continue),
# configuration identity of the launch records against pass 1, EngineDeadError position. Usage: postpass_e1_stage15.sh <pass>
set -u
S=/mnt/shared/home/dc/workspace/serving-studio
export HOME=/mnt/shared/home/dc; export PATH=$HOME/.local/bin:$PATH
REPO=${E1_REPO:-$S/ServingStudioSim}; E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
RUN=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15; PASS=${1:?pass}; CPU=$RUN/cpu; mkdir -p $CPU
cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env
rc_line=$(grep -a "rc=" $RUN/$PASS/run.log | tail -1); echo "$rc_line"
cmp=""; [ "$PASS" != e1_dry_off_v2 ] && cmp="--compare $RUN/e1_dry_off_v2/profile"
uv run --no-sync python $E1/checks/e1_completeness_v2.py --profile $RUN/$PASS/profile --inputs $E1/E1-STAGE15-INPUTS.json $cmp --checksum --out $CPU/v2_$PASS.json > $CPU/v2_$PASS.txt 2>&1; v2=$?
grep -E "^FAIL|^INCONCLUSIVE|overhead|prospective|^RESULT" $CPU/v2_$PASS.txt
if [ "$PASS" != e1_dry_off_v2 ]; then
  uv run --no-sync python $E1/checks/e1_config_identity.py --a $RUN/e1_dry_off_v2 --b $RUN/$PASS --ignore-env PATH --out $CPU/identity_off_v2_vs_$PASS.json > $CPU/identity_off_v2_vs_$PASS.txt 2>&1; id=$?
  grep -E "^FAIL|^RESULT" $CPU/identity_off_v2_vs_$PASS.txt
else id=0; fi
grep -a "EngineDeadError-family" $RUN/$PASS/run.log | tail -1
echo "GATE $PASS: completeness_v2 rc=$v2 identity rc=$id -> $([ $v2 = 0 ] && [ $id = 0 ] && echo CONTINUE || echo STOP)"
exit $(( v2 + id ))
