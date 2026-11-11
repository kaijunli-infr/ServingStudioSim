#!/usr/bin/env bash
# Gate 0 (CPU only): prebuild FlashInfer's trtllm-gen FMHA module in place, at the
# path the GPU continuation will load it from. Rack 3 tray12, no GPU visible.
#
#   bash prebuild_flashinfer.sh            (writes $G/phase3b2/cont/prebuild-<stamp>/)
#
# - FLASHINFER_WORKSPACE_BASE=$FI_BASE (fresh, never copied or moved afterwards);
#   FLASHINFER_CUDA_ARCH_LIST=10.3 (GB300's sm_103a, the arch the serving cache
#   built for), so nothing queries a device. CUDA_VISIBLE_DEVICES is empty.
# - Importing FlashInfer initializes the CUDA driver (torch.cuda.is_available()
#   in tvm_ffi's optional DLPack loader, then cuda.tile via flashinfer.cutile),
#   which opens every /dev/nvidia* node even with no visible device: attempts
#   prebuild-20261111-180507 and -180858 were refused by this script for that.
#   Every step therefore runs with nogpu_shim.so preloaded (built here from
#   nogpu_shim.c): any open of a /dev/nvidia* path fails with EACCES before
#   reaching the kernel and is logged to nogpu.log, so no process of the build
#   can initialize a GPU. strace then must show no /dev/nvidia* syscall at all.
#   TVM_FFI_DISABLE_TORCH_C_DLPACK=1 is kept (not a build input).
# - build.ninja is compared with the first attempt's (built with the driver
#   reachable): equal means the build's inputs do not depend on driver state.
# - Every step runs under `strace -f` (all child processes: ninja, nvcc, cicc,
#   ptxas, ...); a step fails if any process opened a /dev/nvidia* node.
# - Temporary files go to a local TMPDIR sampled every second against
#   TMP_CAP_BYTES (350,000,000); a breach kills the build. The peak is recorded;
#   the TMPDIR is deleted at the end.
# - The shared caches ($G/cache/flashinfer, $G/cache/nv-compute) are
#   fingerprinted (path, size, mtime of every file) before and after.
# - Steps: build, check (second build must be a ninja no-op), load (module load
#   under strace; if it touches a device node it is reported, and the load
#   verification is left to the GPU window).
set -u
S=/mnt/shared/home/kevin/workspace/serving-studio
G=$S/gb300/serving-validation
PY=$G/venv/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FI_BASE=${FI_BASE:-$G/phase3b2/cont/flashinfer}
TMP_CAP_BYTES=${TMP_CAP_BYTES:-350000000}
STAMP=$(date -u +%Y%m%d-%H%M%S)
OUT=$G/phase3b2/cont/prebuild-$STAMP
TMPD=/tmp/fi-prebuild-$STAMP
mkdir -p "$OUT" "$TMPD"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/steps.log"; }
listing() { (cd "$1" && find . -type f -printf '%P %s %T@\n' | sort | sha256sum | cut -d' ' -f1); }
log "prebuild out=$OUT fi_base=$FI_BASE tmp=$TMPD tmp_cap=$TMP_CAP_BYTES helper=$(sha256sum "$HERE/fi_prebuild.py" | cut -d' ' -f1) script=$(sha256sum "$0" | cut -d' ' -f1)"
[ -e "$FI_BASE" ] && { log "STOP: $FI_BASE already exists; the prebuild starts from an empty workspace"; exit 2; }
for d in "$G/cache/flashinfer" "$G/cache/nv-compute"; do echo "$d $(listing "$d")" >>"$OUT/shared-caches-pre.txt"; done
free_shared=$(df -B1 --output=avail "$G" | tail -1); free_tmp=$(df -B1 --output=avail /tmp | tail -1)
log "capacity: shared fs free ${free_shared} B, /tmp free ${free_tmp} B"
[ "$free_shared" -ge 2000000000 ] && [ "$free_tmp" -ge $((TMP_CAP_BYTES * 2)) ] || { log "STOP: not enough free space"; exit 2; }
gcc -shared -fPIC -O2 -Wall -o "$OUT/nogpu_shim.so" "$HERE/nogpu_shim.c" -ldl || { log "STOP: shim build failed"; exit 2; }
log "shim $(sha256sum "$HERE/nogpu_shim.c" | cut -d' ' -f1) built"
mkdir -p "$FI_BASE"
export LD_PRELOAD="$OUT/nogpu_shim.so" NOGPU_LOG="$OUT/nogpu.log"
export CUDA_VISIBLE_DEVICES= FLASHINFER_WORKSPACE_BASE="$FI_BASE" FLASHINFER_CUDA_ARCH_LIST=10.3 \
  TMPDIR="$TMPD" MAX_JOBS=1 HF_HUB_OFFLINE=1 PATH="$G/venv/bin:$PATH" TVM_FFI_DISABLE_TORCH_C_DLPACK=1
PEAK=0
step() { # step NAME : strace the helper's mode NAME, sampling TMPDIR
  local name=$1 pid rc b
  log "step $name start"
  # Own session, so a breach stops strace and every traced process together.
  setsid strace -f -qq -e trace=open,openat,openat2 -o "$OUT/strace-$name.log" \
    "$PY" -P "$HERE/fi_prebuild.py" "$name" --out "$OUT/$name.json" >"$OUT/$name.out" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    b=$(du -sb "$TMPD" 2>/dev/null | cut -f1)
    echo "$(date -u +%T) step=$name tmp_bytes=$b fi_bytes=$(du -sb "$FI_BASE" | cut -f1)" >>"$OUT/sizes.log"
    [ "${b:-0}" -gt "$PEAK" ] && PEAK=$b
    if [ "${b:-0}" -gt "$TMP_CAP_BYTES" ]; then
      log "STOP: temporary files ${b} B above ${TMP_CAP_BYTES} B"; kill -KILL -- "-$pid"; wait "$pid"; return 9
    fi
    sleep 1
  done
  wait "$pid"; rc=$?
  grep '"/dev/nvidia' "$OUT/strace-$name.log" | head -5 >"$OUT/strace-$name-devices.txt"
  log "step $name rc=$rc device-node syscalls=$(grep -c '"/dev/nvidia' "$OUT/strace-$name.log") refused-by-shim so far=$(wc -l <"$OUT/nogpu.log" 2>/dev/null || echo 0) tmp peak so far ${PEAK} B"
  return $rc
}
step build || { log "STOP: build failed or breached (see build.out)"; rm -rf "$TMPD"; exit 3; }
[ -s "$OUT/strace-build-devices.txt" ] && { log "STOP: the build opened a GPU device node"; rm -rf "$TMPD"; exit 4; }
step check || { log "STOP: second build was not a no-op (see check.json)"; rm -rf "$TMPD"; exit 5; }
[ -s "$OUT/strace-check-devices.txt" ] && { log "STOP: the no-op check opened a GPU device node"; rm -rf "$TMPD"; exit 4; }
step load; lrc=$?
if [ -s "$OUT/strace-load-devices.txt" ] || [ $lrc -ne 0 ]; then
  log "load: touched a GPU device node or failed (rc=$lrc); load verification deferred to the GPU window"
  echo deferred >"$OUT/load-verdict.txt"
else
  log "load: loaded with no GPU device-node access"
  echo cpu-verified >"$OUT/load-verdict.txt"
fi
log "temporary files at the end: $(du -sb "$TMPD" | cut -f1) B; peak sampled ${PEAK} B"
rm -rf "$TMPD" && log "deleted $TMPD"
log "retained module output: $(du -sb "$FI_BASE" | cut -f1) B in $FI_BASE"
REF=${REF_NINJA_SHA:-b484f12fb7ebaa373befcc58d295b566933a55474e073e26ca9c5a8209669cf1}
got=$(python3 -c "import json;print(json.load(open('$OUT/build.json'))['build_ninja_sha256'])")
[ "$got" = "$REF" ] && log "build.ninja $got equals the driver-reachable attempt's" || log "build.ninja $got DIFFERS from $REF"
log "shim refusals: $(cut -d' ' -f2 "$OUT/nogpu.log" 2>/dev/null | sort | uniq -c | tr '\n' ';')"
for d in "$G/cache/flashinfer" "$G/cache/nv-compute"; do echo "$d $(listing "$d")" >>"$OUT/shared-caches-post.txt"; done
cmp -s "$OUT/shared-caches-pre.txt" "$OUT/shared-caches-post.txt" && log "shared caches unchanged" || log "a shared cache CHANGED"
log "done"
