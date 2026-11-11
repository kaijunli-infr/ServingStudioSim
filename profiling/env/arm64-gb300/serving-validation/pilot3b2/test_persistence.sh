#!/usr/bin/env bash
# CPU-only Gate 1 tests of pilot_driver3.py's persistence (--selftest: stand-in
# cases through the same run loop and Recorder as the GPU modes).
#   test_persistence.sh DRIVER_DIR TEST_DIR      (PY = interpreter, default python3)
# The signal's send time is recorded and checked against the journal's event
# times, so each scenario proves where the signal landed:
# A  no signal: all 5 cases persisted, status done, exit 0
# B  SIGINT inside a case (after its start, before it could finish): the case
#    is not persisted; stop record: it is in progress, the previous one last completed
# C  SIGINT in the gap after a persisted case: nothing in progress
# D  SIGINT inside the persistence window (write widened to 2 s): the case being
#    written is durable and is the last completed; nothing in progress
# E  SIGTERM inside a case: as B, exit 143
# Every run: persisted cases are a prefix of the plan, every case file parses
# with status pass, no *.tmp left, journal lines parse and its done events equal
# the case files, the summary holds exactly the persisted cases.
set -u
D=$1; T=$2; PY=${PY:-python3}
rm -rf "$T"; mkdir -p "$T"
pass=0; fail=0
verify() { # verify NAME EXPECT_RC RC WHERE CASE_S  (WHERE: none|case|gap|persist)
  "$PY" - "$T/$1" "$2" "$3" "$4" "$5" "$T/$1.sigtime" <<'PY'
import json, sys
from pathlib import Path
d, want_rc, rc, where, case_s, sigfile = sys.argv[1:7]
d, case_s = Path(d), float(case_s)
ok, why = True, []
def bad(msg):
    global ok
    ok = False; why.append(msg)
if rc != want_rc: bad(f"rc {rc} != {want_rc}")
plan = [f"st_{i:02d}" for i in range(5)]
cases = [json.loads(f.read_text()) for f in sorted(d.glob("cases/[0-9][0-9][0-9]-*.json"))]
done = [c["case"] for c in cases]
if done != plan[:len(done)]: bad(f"persisted {done} is not a prefix of the plan")
if any(c.get("status") != "pass" for c in cases): bad("a case file is not pass")
if list(d.glob("**/*.tmp")): bad("tmp file left")
events = [json.loads(l) for l in (d / "journal.jsonl").read_text().splitlines()]
if [e["case"] for e in events if e["event"] == "done"] != done: bad("journal done events differ from case files")
summary = json.loads((d / "selftest.json").read_text())
if sorted(summary["cases"]) != sorted(done): bad("summary differs from case files")
stop = json.loads((d / "stop.json").read_text()) if (d / "stop.json").exists() else None
start = {e["case"]: e["t"] for e in events if e["event"] == "start"}
finish = {e["case"]: e["t"] for e in events if e["event"] == "done"}
if where == "none":
    if stop or len(done) != 5: bad("expected a complete run")
else:
    sig = float(Path(sigfile).read_text())
    if not stop: bad("no stop record")
    else:
        last, prog = stop["last_completed"], stop["in_progress"]
        if last != (done[-1] if done else None): bad(f"last_completed {last} != last persisted")
        if where == "case":
            nxt = plan[len(done)]
            if prog != nxt: bad(f"in_progress {prog} != {nxt}")
            if not (start.get(nxt, 1e18) <= sig < start.get(nxt, 0) + case_s): bad("signal not inside the case")
        elif where == "gap":
            if prog is not None: bad("in_progress set")
            if not (finish.get(last, 1e18) <= sig): bad("signal before the last done")
            if plan[len(done)] in start and start[plan[len(done)]] <= sig: bad("signal after the next start")
        elif where == "persist":
            if prog is not None: bad("in_progress set")
            if not (start[last] + case_s <= sig <= finish[last]): bad("signal not inside the persistence window")
print(("PASS " if ok else "FAIL ") + d.name, "persisted", done, "stop", stop, "; ".join(why))
sys.exit(0 if ok else 1)
PY
}
run() { # run NAME SIGNAL DELAY ARGS...  (DELAY after the driver's begin event; SIGNAL - for none)
  local name=$1 sig=$2 delay=$3; shift 3
  "$PY" "$D/pilot_driver3.py" --selftest 5 --out "$T/$name" "$@" >"$T/$name.log" 2>&1 &
  local pid=$!
  if [ "$sig" != - ]; then
    until grep -q '"begin"' "$T/$name/journal.jsonl" 2>/dev/null; do sleep 0.05; done
    sleep "$delay"; date +%s.%N >"$T/$name.sigtime"; kill -"$sig" "$pid"
  fi
  wait "$pid"; RC=$?
}
run A_normal - 0 --selftest-case-s 0.3
verify A_normal 0 $RC none 0.3 && pass=$((pass+1)) || fail=$((fail+1))
run B_int_in_case INT 2.5 --selftest-case-s 1.0
verify B_int_in_case 130 $RC case 1.0 && pass=$((pass+1)) || fail=$((fail+1))
run C_int_between INT 1.2 --selftest-case-s 0.2 --selftest-gap-s 2.0
verify C_int_between 130 $RC gap 0.2 && pass=$((pass+1)) || fail=$((fail+1))
run D_int_in_persist INT 1.0 --selftest-case-s 0.2 --selftest-persist-s 2.0
verify D_int_in_persist 130 $RC persist 0.2 && pass=$((pass+1)) || fail=$((fail+1))
run E_term_in_case TERM 2.5 --selftest-case-s 1.0
verify E_term_in_case 143 $RC case 1.0 && pass=$((pass+1)) || fail=$((fail+1))
echo "persistence tests: $pass passed, $fail failed"
[ $fail = 0 ]
