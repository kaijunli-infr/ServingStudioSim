#!/usr/bin/env bash
# CPU-only: offline import of the preserved V2 stream (archive copy, read-only) under the new importer-specific probe.
# $1 = label, $2 = seconds after which this script's OWN importer child is killed (0 = never)
set -u
export HOME=/mnt/shared/home/dc; S=$HOME/workspace/serving-studio; SL=$S/scratch/lifecycle; O=$SL/v2close; WL=$S/wt-lifecycle; E1=$WL/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
export CUDA_VISIBLE_DEVICES=""
IMP=/opt/nvidia/nsight-systems/2025.6.3/host-linux-armv8/QdstrmImporter
SRC=$S/artifacts/phase1c-e1-v2fr-dsv41-flash-gb300-tp4-20261118/preserved/dsv41_flash_1017_gb300_tp4_e1_v2_forced_release.qdstrm
D=/tmp/v2close-import-$1; rm -rf $D; mkdir -p $D
echo "# offline import '$1' on $(hostname) $(date -u +%FT%TZ); kill own importer after: $2 s; source sha256 $(sha256sum $SRC | cut -c1-64)"
echo "gpu before (MiB / compute apps): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ') / $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
python3 -I $SL/procwatch.py --out $O/import_procwatch_$1.jsonl --interval 0.25 --max-s 300 --stop-file $D/stop &
W=$!
python3 -I $E1/checks/e1_import_probe.py --importer-exe $IMP --stream $SRC --report $D/out.nsys-rep --out $O/import_probe_$1.json --samples $O/import_probe_$1.jsonl --log $O/import_probe_$1.log --interval 0.5 --max-s 300 &
P=$!
sleep 1.5
( while [ ! -e $D/stop ]; do echo "$(date +%s.%N) $(du -sk $D | cut -f1)"; sleep 0.5; done ) > $O/import_scratch_$1.txt &
python3 -I - "$IMP" "$SRC" "$D/out.nsys-rep" "$2" > $O/import_run_$1.log 2>&1 <<'PY'
import subprocess, sys, time
imp, src, out, kill_after = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
t0 = time.time(); p = subprocess.Popen([imp, "--input-file", src, "--output-file", out], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
print(f"importer child pid {p.pid} started {t0:.3f}")
if kill_after > 0:
    try: p.wait(timeout=kill_after)
    except subprocess.TimeoutExpired:
        p.kill(); print(f"SIGKILL sent to own child pid {p.pid} at {time.time():.3f} (+{time.time()-t0:.2f} s)")
rc = p.wait(); print(f"importer rc {rc} after {time.time()-t0:.2f} s")
PY
wait $P; prc=$?; sleep 0.5; touch $D/stop; wait $W; wait
cat $O/import_run_$1.log; echo "probe rc=$prc"; cat $O/import_probe_$1.log
python3 -I - $O/import_probe_$1.json $O/import_probe_$1.jsonl $O/import_scratch_$1.txt <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); rows = [json.loads(l) for l in open(sys.argv[2])]
print("result:", r["result"]); print("importer:", {k: v for k, v in r.get("importer", {}).items() if k != "agent"}); print("agent/parent:", r.get("importer", {}).get("agent"))
print("transitions:", [(round(t["at"] - r["started_at"], 2), t["state"]) for t in r["transitions"]])
imp = [x for x in rows if x.get("importer")]; step = max(1, len(imp) // 12)
for x in imp[::step] + imp[-1:]: print(f"  +{x['at']-r['started_at']:6.2f} s state {x['importer']['state']} ticks {x['importer']['ticks']} rchar {x['importer']['io']['rchar']} wchar {x['importer']['io']['wchar']} report_size {x['report_size']}")
print("  after importer gone:", [(round(x["at"] - r["started_at"], 2), x["report_size"]) for x in rows if not x.get("importer")][-3:])
sc = [int(l.split()[1]) for l in open(sys.argv[3]) if l.strip()]; print(f"scratch dir peak {max(sc)/1e6:.3f} GB over {len(sc)} samples (0.5 s interval)")
PY
python3 -I - $O/import_procwatch_$1.jsonl <<'PY'
import json, sys
hw = 0; rss = 0
for l in open(sys.argv[1]):
    for p in json.loads(l)["procs"]:
        if p["comm"] == "QdstrmImporter" and p.get("mem_kb"): hw = max(hw, p["mem_kb"].get("VmHWM", 0)); rss = max(rss, p["mem_kb"].get("VmRSS", 0))
print(f"importer peak RSS (sampled VmRSS max / VmHWM): {rss/1e6:.2f} GB / {hw/1e6:.2f} GB")
PY
ls -la $D | awk 'NR>1{print $5, $9}' | tr '\n' ';'; echo
echo "gpu after (MiB / compute apps): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ') / $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); importer processes left: $(ps -u $(id -u) -o comm | grep -c Qdstrm)"
rm -rf $D; echo "IMPDONE $1"
