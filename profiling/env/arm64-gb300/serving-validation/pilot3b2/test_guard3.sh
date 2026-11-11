#!/usr/bin/env bash
# CPU-only tests of run_pilot3.sh (TEST_MODE=1: no GPU query, passes are named
# `sleep` stand-ins; four passes: timing, control2048, control8192, identity).
#   test_guard3.sh BIN_DIR TEST_DIR
# test_guard2.sh's A-H, with the continuation's prior occupancy (50.3 s), plus
# I a CUDA compiler (a copy of `sleep` named nvcc) among the run's processes
# stops the run, and J a ptxas under a triton/ directory does not.
# A-D as pilot3b/test_guard.sh (accounting and cleanup); E-H the byte limits:
# E the source cache already at the cap (stop before launch, nothing occupied),
# F the private cache growing past its cap during a pass, G the artifacts
# growing past theirs, H temporary files (local TMPDIR) counted with the artifacts.
set -u
BIN=$1; T=$2
rm -rf "$T"; mkdir -p "$T/src" "$T/localtmp"
for i in 1 2 3; do head -c 3000000 /dev/urandom >"$T/src/f$i"; done
SRC_LIST=$(cd "$T/src" && find . -type f -printf '%P %s %T@\n' | sort | sha256sum)
SRC_BYTES=$(du -sb "$T/src" | cut -f1)
pass=0; failn=0
check() { # check NAME LO HI EXPECT_STOP [PATTERN]  (LO = -1: no pass may have run)
  local name=$1 lo=$2 hi=$3 stop=$4 pat=${5:-} run occ ok=1
  run=$(ls -dt "$T"/run-* | head -1)
  occ=$(python3 -c "import json;print(json.load(open('$run/occupancy.json'))['rerun_s'])" 2>/dev/null || echo none)
  if [ "$lo" = -1 ]; then [ "$occ" = none ] || ok=0
  else python3 -c "import sys; sys.exit(0 if '$occ' != 'none' and $lo <= float('$occ') <= $hi else 1)" || ok=0; fi
  ls -d "$T"/scratch-* >/dev/null 2>&1 && ok=0
  [ -z "$(ls -A "$T/localtmp")" ] || ok=0
  if [ "$stop" = 1 ]; then grep -q STOP "$run/steps.log" || ok=0; else grep -q " done$" "$run/steps.log" || ok=0; fi
  [ -n "$pat" ] && { grep -q "$pat" "$run/steps.log" || ok=0; }
  [ "$(cd "$T/src" && find . -type f -printf '%P %s %T@\n' | sort | sha256sum)" = "$SRC_LIST" ] || ok=0
  if [ $ok = 1 ]; then pass=$((pass+1)); echo "PASS $name occupancy=$occ"; else failn=$((failn+1)); echo "FAIL $name occupancy=$occ"; sed 's/^/    /' "$run/steps.log"; fi
}
export TEST_MODE=1 PILOT_DIR=$T SRC_CACHE=$T/src LOCALTMP_ROOT=$T/localtmp MIN_FREE_BYTES=1
# A: normal, 3 s + 2 s; sizes sampled during the run
setsid -w bash "$BIN/run_pilot3.sh" >/dev/null 2>&1; check A_normal 6.8 13 0
grep -q '"prior_s": 50.3' "$(ls -dt "$T"/run-* | head -1)/occupancy.json" && echo "PASS A_prior_counted" && pass=$((pass+1)) || { echo "FAIL A_prior_counted"; failn=$((failn+1)); }
n=$(wc -l <"$(ls -dt "$T"/run-* | head -1)/sizes.log"); [ "$n" -ge 1 ] && echo "PASS A_sizes_sampled ($n samples)" && pass=$((pass+1)) || { echo "FAIL A_sizes_sampled"; failn=$((failn+1)); }
sleep 1
# B: timing pass times out at 2 s
TIMING_CAP_S=2 TEST_CMD_T='exec -a pilot_test_pass sleep 30' setsid -w bash "$BIN/run_pilot3.sh" >/dev/null 2>&1; check B_timeout 1.9 6 1; sleep 1
# C: external stop (USR1 = foreign process) 4 s into a 30 s pass
TEST_CMD_T='exec -a pilot_test_pass sleep 30' setsid -w bash "$BIN/run_pilot3.sh" >/dev/null 2>&1 &
sleep 5; G=$(sed -n 's/.*root_pid=\([0-9]*\).*/\1/p' "$(ls -dt "$T"/run-* | head -1)/steps.log"); kill -USR1 "$G"; wait; check C_interrupt 3 8 1; sleep 1
# D: pass exits at once but leaves a 4 s child; occupancy runs until the child is gone
TEST_CMD_T='(exec -a pilot_test_pass sleep 4) & exit 0' TEST_CMD_C='exit 0' TEST_CMD_I='exit 0' setsid -w bash "$BIN/run_pilot3.sh" >/dev/null 2>&1; check D_leftover 3.5 10 0; sleep 1
# E: source cache not below the cap -> stop before any pass
CACHE_BYTES=$SRC_BYTES setsid -w bash "$BIN/run_pilot3.sh" >/dev/null 2>&1; check E_source_at_cap -1 0 1 "not below"; sleep 1
# F: private cache grows past cap (source + 2 MB) during the pass
CACHE_BYTES=$((SRC_BYTES + 2000000)) TEST_CMD_T='head -c 4000000 /dev/urandom >"$PILOT_NV/new"; exec -a pilot_test_pass sleep 30' \
  setsid -w bash "$BIN/run_pilot3.sh" >/dev/null 2>&1; check F_cache_growth 0 8 1 "private CUDA cache [0-9]* B above"; sleep 1
grep -h "private CUDA cache grew" "$(ls -dt "$T"/run-* | head -1)/steps.log"
# G: artifacts grow past 3 MB during the pass
DISK_BYTES=3000000 TEST_CMD_T='head -c 4000000 /dev/urandom >"$PILOT_OUT/blob"; exec -a pilot_test_pass sleep 30' \
  setsid -w bash "$BIN/run_pilot3.sh" >/dev/null 2>&1; check G_artifact_growth 0 8 1 "artifacts [0-9]* B above"; sleep 1
# H: local temporary files count with the artifacts
DISK_BYTES=3000000 TEST_CMD_T='head -c 4000000 /dev/urandom >"$TMPDIR/blob"; exec -a pilot_test_pass sleep 30' \
  setsid -w bash "$BIN/run_pilot3.sh" >/dev/null 2>&1; check H_tmp_counted 0 8 1 "artifacts [0-9]* B above"
sleep 1
# I: a CUDA compiler process among the run's processes
mkdir -p "$T/fakebin/triton"; cp "$(command -v sleep)" "$T/fakebin/nvcc"; cp "$(command -v sleep)" "$T/fakebin/triton/ptxas"
TEST_CMD_T="\"$T/fakebin/nvcc\" 30" setsid -w bash "$BIN/run_pilot3.sh" >/dev/null 2>&1; check I_compiler_stops 0 8 1 "unexpected CUDA compilation"; sleep 1
# J: Triton's own ptxas is not a stop
TEST_CMD_T="\"$T/fakebin/triton/ptxas\" 3" setsid -w bash "$BIN/run_pilot3.sh" >/dev/null 2>&1; check J_triton_ptxas_ok 6.8 13 0
echo "guard tests: $pass passed, $failn failed"
[ $failn = 0 ]
