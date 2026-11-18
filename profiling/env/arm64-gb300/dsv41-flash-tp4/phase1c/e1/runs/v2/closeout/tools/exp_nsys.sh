#!/usr/bin/env bash
# CPU-only synthetic Nsight session (no CUDA: devices hidden, target is a pure-CPU python loop) to observe which process does the import.
set -u
export HOME=/mnt/shared/home/dc; S=$HOME/workspace/serving-studio; SL=$S/scratch/lifecycle; O=$SL/v2close; mkdir -p $O
export CUDA_VISIBLE_DEVICES=""
NS=/opt/nvidia/nsight-systems/2025.6.3/target-linux-sbsa-armv8/nsys
D=/tmp/v2close-synth-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $D; cd $D
echo "# synthetic CPU-only session on $(hostname) $(date -u +%FT%TZ); CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"
echo "gpu before (MiB / compute apps): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ') / $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
cat > target.py <<'PY'
import os, time
fd = os.open("/dev/null", os.O_WRONLY); t = time.time(); n = 0
while time.time() - t < float(os.environ.get("SYN_S", "6")):
    for _ in range(20000): os.write(fd, b"x")
    n += 20000
print("target done, writes:", n, flush=True)
if os.environ.get("SYN_LINGER"): time.sleep(float(os.environ["SYN_LINGER"]))
PY
python3 -I $SL/procwatch.py --out $O/synth_procwatch_$1.jsonl --interval 0.05 --max-s 180 --stop-file $D/stop &
W=$!
$NS profile --trace=osrt --sample=none --cpuctxsw=none --force-overwrite=true -o $D/synth python3 -I target.py > $O/synth_nsys_$1.log 2>&1; rc=$?
sleep 1; touch $D/stop; wait $W
echo "session rc=$rc; files: $(ls -la $D | awk 'NR>1{print $5, $9}' | tr '\n' ';')"
tr '\r' '\n' < $O/synth_nsys_$1.log | grep -v "^\[1/1\] \[=*[0-9]*%" | tail -12
python3 -I $SL/summarize.py $O/synth_procwatch_$1.jsonl Qdstrm
echo "gpu after (MiB / compute apps): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ') / $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
echo "leftover nsight processes: $(ps -u $(id -u) -o pid,comm | grep -ci 'nsys\|qdstrm')"; rm -rf $D; echo SYNDONE
