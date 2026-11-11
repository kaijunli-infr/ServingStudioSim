#!/usr/bin/env bash
# CPU-only tests of run_pilot.sh's accounting, cleanup and cache handling (TEST_MODE=1:
# no GPU query, passes are named `sleep` stand-ins).  test_guard.sh BIN_DIR TEST_DIR
set -u
BIN=$1; T=$2
rm -rf "$T"; mkdir -p "$T/src"
for i in 1 2 3; do head -c 3000000 /dev/urandom >"$T/src/f$i"; done
SRC_LIST=$(cd "$T/src" && find . -type f -printf '%P %s %T@\n' | sort | sha256sum)
pass=0; failn=0
check() { # check NAME LO HI EXPECT_STOP
  local name=$1 lo=$2 hi=$3 stop=$4 run occ ok=1
  run=$(ls -dt "$T"/run-* | head -1)
  occ=$(python3 -c "import json;print(json.load(open('$run/occupancy.json'))['rerun_s'])" 2>/dev/null || echo none)
  python3 -c "import sys; sys.exit(0 if '$occ' != 'none' and $lo <= float('$occ') <= $hi else 1)" || ok=0
  ls -d "$T"/scratch-* >/dev/null 2>&1 && ok=0
  if [ "$stop" = 1 ]; then grep -q STOP "$run/steps.log" || ok=0; else grep -q " done$" "$run/steps.log" || ok=0; fi
  [ "$(cd "$T/src" && find . -type f -printf '%P %s %T@\n' | sort | sha256sum)" = "$SRC_LIST" ] || ok=0
  grep -q "source cache .* unchanged" "$run/steps.log" 2>/dev/null; # (not checked in TEST_MODE)
  if [ $ok = 1 ]; then pass=$((pass+1)); echo "PASS $name occupancy=$occ in [$lo,$hi]"; else failn=$((failn+1)); echo "FAIL $name occupancy=$occ"; sed 's/^/    /' "$run/steps.log"; fi
}
export TEST_MODE=1 PILOT_DIR=$T SRC_CACHE=$T/src
# A: normal, 3 s + 2 s
setsid -w bash "$BIN/run_pilot.sh" >/dev/null 2>&1; check A_normal 4.8 9 0; sleep 1
# B: timing pass times out at 2 s
TIMING_CAP_S=2 TEST_CMD_T='exec -a pilot_test_pass sleep 30' setsid -w bash "$BIN/run_pilot.sh" >/dev/null 2>&1; check B_timeout 1.9 6 1; sleep 1
# C: external stop (USR1 = foreign process) 4 s into a 30 s pass
TEST_CMD_T='exec -a pilot_test_pass sleep 30' setsid -w bash "$BIN/run_pilot.sh" >/dev/null 2>&1 &
sleep 5; G=$(sed -n 's/.*root_pid=\([0-9]*\).*/\1/p' "$(ls -dt "$T"/run-* | head -1)/steps.log"); kill -USR1 "$G"; wait; check C_interrupt 3 8 1; sleep 1
# D: pass exits at once but leaves a 4 s child; occupancy runs until the child is gone
TEST_CMD_T='(exec -a pilot_test_pass sleep 4) & exit 0' TEST_CMD_I='exit 0' setsid -w bash "$BIN/run_pilot.sh" >/dev/null 2>&1; check D_leftover 3.5 9 0; sleep 1
# E: private cache above its 1 MB limit -> stop, private copy deleted, source unchanged
CACHE_MB=1 TEST_CMD_T='exec -a pilot_test_pass sleep 30' setsid -w bash "$BIN/run_pilot.sh" >/dev/null 2>&1; check E_cache_budget 0 8 1
grep -h "private CUDA cache\|deleted private" "$(ls -dt "$T"/run-* | head -1)/steps.log"
echo "guard tests: $pass passed, $failn failed"
[ $failn = 0 ]
