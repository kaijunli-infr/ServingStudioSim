#!/usr/bin/env bash
# One read-only monitoring sample for the running V3 pass. Prints state lines from lifecycle.json, the observers' records and explicit
# ABORT-CONDITION lines; it never signals anything. Run every 60 s from the laptop through ssh (neutral path):
#   bash monitor_e1_v3scale.sh [run root] [pass] [startup_limit_min=30] [capture_limit_min=10] [run_ceiling_GB=8] [tmp_floor_GB=50]
set -u
RUN=${1:-/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v3scale}; PASS=${2:-e1_v3_scale}
SUL=${3:-30}; CL=${4:-10}; CEIL=${5:-8}; TFLOOR=${6:-50}
L=$RUN/$PASS/run.log; SL=$(ls $RUN/$PASS/profile/vllm/*_server.log 2>/dev/null | head -1); LC=$RUN/$PASS/profile/nsys/lifecycle.json
drv=$(awk -F'\t' 'NR>1 && $0 ~ /driver:/ {print $1}' $RUN/pids/owned.tsv 2>/dev/null | head -1)
now=$(date -u +%s)
t0=$(grep -ao "^\[$PASS\] start [0-9T:-]*Z" $L 2>/dev/null | head -1 | awk '{print $3}'); t0s=$( [ -n "$t0" ] && date -u -d "$t0" +%s || echo "$now")
echo "== $(date -u +%FT%TZ) $PASS driver=${drv:-?} alive=$([ -n "$drv" ] && [ -d /proc/$drv ] && echo y || echo n) elapsed_min=$(( (now - t0s) / 60 )) tmux=$(tmux ls 2>/dev/null | cut -d: -f1 | tr '\n' ,)"
su=$(grep -a -c "Application startup complete" "$SL" 2>/dev/null || true); su=${su:-0}
mb=$(grep -a -c "measurement begins" $L 2>/dev/null || true); mb=${mb:-0}
rc=$(grep -a -o "^\[$PASS\] rc=[0-9]*" $L 2>/dev/null | tail -1)
echo "launcher: startup_complete=$su measurement_begun=$mb ${rc:-running}"
[ "$su" = 0 ] && [ -z "$rc" ] && [ $(( (now - t0s) / 60 )) -gt $SUL ] && echo "ABORT-CONDITION startup exceeds ${SUL} min (operator: registry cleanup, never by pattern)"
if [ -f "$LC" ]; then
  python3 -I - "$LC" "$now" "$CL" <<'PY'
import json,sys,time
d=json.load(open(sys.argv[1])); now=float(sys.argv[2]); cl=float(sys.argv[3])
dl=d.get("deadlines") or {}; gpu=d.get("gpu") or {}; st=d.get("stream") or {}
f=lambda t: "-" if not t else time.strftime("%H:%M:%SZ", time.gmtime(t))
print(f"lifecycle: phase={d['phase']} stop_requested={f(d.get('stop_requested_at'))} hard_at={f(dl.get('hard_at'))} ceiling_at={f(dl.get('gpu_held_ceiling_at'))} ext={dl.get('hard_extension_s',0)} s (reported progress {d.get('import_progress_pct')}%: not evidence of importer activity)")
print(f"lifecycle: stream={st.get('path')} size={st.get('size')} preserved={st.get('preserved_path')} ({st.get('preserved_size')}) report={d.get('report',{}).get('size')} gpu_released={f(gpu.get('released_at'))} nsys_alive={(d.get('nsys') or {}).get('alive')} server_alive={(d.get('server') or {}).get('alive')}")
print(f"lifecycle: control={[(c['command'][:40], c['result'][:40]) for c in d.get('control_commands',[])]} warnings={len(d.get('warnings',[]))} last={(d.get('warnings') or [{}])[-1].get('message')}")
for e in d.get("escalations", []): print(("NOTE soft-deadline warning" if e.get("kind") == "soft_deadline" else f"ABORT-CONDITION escalation {e.get('kind')}") + f" after {e.get('elapsed_s',0):.0f} s in phase {e.get('phase')} (the controller acts on hard deadline and ceiling only; operator reviews)")
if d.get("control_commands"): print("ABORT-CONDITION a control line was accepted (V3 registers none)")
if d["phase"] == "stop_failed": print("ABORT-CONDITION stop_failed: preserved stream → offline recovery (CPU) after closeout")
if d.get("stop_requested_at") is None and d.get("launch_started_at") and now - d["launch_started_at"] > (cl + 30) * 60:
    print(f"ABORT-CONDITION no stop request {int((now - d['launch_started_at'])/60)} min after the launch (startup limit 30 + capture limit {cl:.0f} min)")
PY
fi
python3 -I - "$RUN/$PASS" <<'PY'
import json,sys,time
run=sys.argv[1]; f=lambda t: "-" if not t else time.strftime("%H:%M:%SZ", time.gmtime(t))
def load(p):
    try: return json.load(open(p))
    except Exception: return None
p=load(run+"/import_probe.json")
if p: print(f"import probe: {p.get('result')} | importer {(p.get('importer') or {}).get('pid')} first seen {f((p.get('importer') or {}).get('first_seen_at'))} | states {[(t['state'], f(t['at'])) for t in p.get('transitions', [])][-5:]}")
m=load(run+"/mem_watch.json")
if m: print(f"memory: samples {m['samples']} last {f(m['last_at'])} min MemAvailable {m['min_available_kb'] and round(m['min_available_kb']/1048576,1)} GB (floor {m['policy']['min_available_gb']}) importer peak {m['importer_peak_rss_kb'] and round(m['importer_peak_rss_kb']/1048576,2)} GB (ceiling {m['policy']['max_importer_rss_gb']}) breaches {len(m['breaches'])}")
s=load(run+"/storage_watch.json")
if s: print("storage: samples", s["samples"], "last", f(s["last_at"]), {g: f"peak {round((v['peak_bytes'] or 0)/1e9,3)} GB (ceiling {v['ceiling_gb']})" for g, v in s["groups"].items()}, "breaches", len(s["breaches"]))
PY
for b in MEMORY-BREACH STORAGE-BREACH; do [ -f $RUN/$PASS/$b ] && echo "ABORT-CONDITION $b: $(head -c 300 $RUN/$PASS/$b | tr '\n' ' ')"; done
grep -a -q "EngineDeadError\|EngineCore encountered an issue\|output_handler failed" "$SL" 2>/dev/null && [ -z "$rc" ] && echo "ABORT-CONDITION engine error lines in the server log"
gm=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'); echo "gpu: $gm"
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  p=$pid; own=n; while [ "$p" != 1 ] && [ -n "$p" ]; do [ "$p" = "$drv" ] && own=y && break; p=$(awk '/^PPid:/{print $2}' /proc/$p/status 2>/dev/null); done
  echo "gpuproc $pid own=$own $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-80)"; [ $own = n ] && echo "ABORT-CONDITION foreign GPU process $pid"
done
raw=$(du -sBG $RUN 2>/dev/null | cut -f1 | tr -dc 0-9); tf=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)
echo "run root ${raw:-0} GB (ceiling $CEIL); /tmp free ${tf} GB (floor $TFLOOR); CephFS free $(df -BG --output=avail /mnt/shared | tail -1 | tr -dc 0-9) GB; MemAvailable $(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo) GB"
[ "${raw:-0}" -gt "$CEIL" ] && echo "ABORT-CONDITION run root ${raw} GB exceeds ceiling ${CEIL} GB"
[ "${tf:-999}" -lt "$TFLOOR" ] && echo "ABORT-CONDITION /tmp free ${tf} GB below floor ${TFLOOR} GB"
exit 0
