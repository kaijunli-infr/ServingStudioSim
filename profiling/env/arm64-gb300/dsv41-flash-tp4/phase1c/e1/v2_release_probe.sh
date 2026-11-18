#!/usr/bin/env bash
# V2 forced-release trigger (packet section 3). Polls lifecycle.json; on the FIRST observation of phase `importing` it checks, from
# /proc and independently of the controller, that the recorded Nsight process is alive, writes the one pre-registered line to
# lifecycle.control and records what it saw in <probe.json>. Started by run_e1_fr.sh through checks/e1_pids.py spawn (registered,
# identity-verified cleanup). It signals nothing. Exits 0 after writing, 1 if the stop became terminal first, 2 after 60 min.
#   bash v2_release_probe.sh <nsys dir> <probe.log> <probe.json>
set -u
NSYS_DIR=$1; LOG=$2; OUT=$3
LINE='release-gpus V2 forced release during import (pre-registered trigger: first observation of phase importing)'
echo "[probe] start $(date -u +%FT%TZ) pid $$ watching $NSYS_DIR/lifecycle.json" >> $LOG
for i in $(seq 1 7200); do
  if [ -f $NSYS_DIR/lifecycle.json ]; then
    state=$(python3 -I - "$NSYS_DIR/lifecycle.json" "$OUT" "$NSYS_DIR/lifecycle.control" "$LINE" <<'PY'
import json, os, sys, time
status, out, control, line = sys.argv[1:5]
try:
    d = json.load(open(status))
except Exception:
    print("wait"); sys.exit(0)
phase = d.get("phase")
if phase in ("report_written", "stop_failed"):
    print("terminal"); sys.exit(0)
if phase != "importing":
    print("wait"); sys.exit(0)
ident = (d.get("nsys") or {}).get("identity") or {}
pid = (d.get("nsys") or {}).get("pid")
alive, starttime = False, None
try:
    stat = open(f"/proc/{pid}/stat").read()
    starttime = int(stat[stat.rindex(")") + 2:].split()[19])
    alive = starttime == ident.get("starttime")
except Exception:
    pass
children = []
try:
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                st = open(f"/proc/{p}/stat").read()
                if int(st[st.rindex(")") + 2:].split()[1]) == pid:
                    children.append([int(p), st[st.index("(") + 1:st.rindex(")")]])
            except Exception:
                pass
except Exception:
    pass
record = {"saw_phase": phase, "saw_at": time.time(), "import_progress_pct": d.get("import_progress_pct"), "nsys_pid": pid,
          "nsys_alive": alive, "nsys_starttime_proc": starttime, "nsys_children": children, "stream": (d.get("stream") or {}).get("path"),
          "source": "v2_release_probe.sh: /proc of the recorded Nsight pid, read by the probe"}
if alive:
    with open(control, "a") as fh:
        fh.write(line + "\n")
    record["wrote_trigger_at"] = time.time()
    record["control_line"] = line
json.dump(record, open(out, "w"), indent=1)
print("fired" if alive else "dead")
PY
)
    case "$state" in
      fired) echo "[probe] phase importing observed, Nsight alive; trigger line written at $(date -u +%FT%TZ); done" >> $LOG; exit 0;;
      dead) echo "[probe] phase importing observed but the recorded Nsight process is not alive; NO trigger written $(date -u +%FT%TZ)" >> $LOG; exit 1;;
      terminal) echo "[probe] stop already terminal before phase importing was observed; NO trigger written (gate FR2 will FAIL) $(date -u +%FT%TZ)" >> $LOG; exit 1;;
    esac
  fi
  sleep 0.5
done
echo "[probe] gave up after 60 min $(date -u +%FT%TZ)" >> $LOG; exit 2
