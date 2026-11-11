#!/usr/bin/env bash
# Phase 3B measurement-validation pilot (P5-P10), guarded. Rack 3 tray12, GPU 1 only.
# A copy of pilot3b/run_pilot.sh (the Phase 3B pilot's guard) whose limits are
# decimal byte counts (`du -sb`), sampled during the run.
#
#   setsid -f bash run_pilot2.sh           (writes $P/run-<stamp>/)
#   TEST_MODE=1 bash run_pilot2.sh ...     (CPU only: passes are `sleep`, see below)
#
# GPU occupancy: each pass (timing, then the Nsight identity pass) is timed from
# its launch to the moment it and every process it started are gone, in every
# exit path (normal end, stop, timeout, signal; `account` runs from `fail` and
# from the EXIT trap). The aggregate cap is 600 s (PRIOR_S = 0: no earlier
# attempt of this pilot). Each pass's hard timeout is the remaining budget minus
# the 15 s kill grace and a 5 s margin. A timeout stops the run; no retry.
#
# CUDA driver cache: a private copy of the warm cache ($G/cache/nv-compute,
# untouched; its listing is fingerprinted before and after) in
# $P/scratch-<stamp>/nv, limited to CACHE_BYTES (and CUDA_CACHE_MAXSIZE set to
# it); its initial and final size and the new bytes are recorded, then only that
# copy is deleted. All other new artifacts (traces, Nsight temporary files,
# Inductor/Triton output, logs) are limited to DISK_BYTES, and an Nsight report
# to NSYS_BYTES. Every limit is checked every 2 s during the run, and each check
# appends both sizes to sizes.log.
#
# before: GPU 1 has the expected UUID, utilisation <= 5 %, < 1 GiB used and no
#   compute process; every compute process on the tray is recorded (GPU 2's
#   SGLang server included); the serving venv's package fingerprint equals the
#   committed one; both profile databases are checksummed; commits recorded.
# during (every 2 s): a foreign compute process on GPU 1, a pilot process on any
#   other GPU, or a budget breach stops the run. GPU 2 is recorded, never acted on.
# after: this run's leftover processes only are stopped; GPU 1 must return to
#   < 1 GiB with no compute process; fingerprints and checksums re-checked.
set -u
# Become a child subreaper (preserved across exec): a process a pass orphans is
# re-parented to this guard instead of init, so it stays in the guard's tree,
# is counted in the pass's occupancy and is stopped by cleanup.
if [ -z "${PILOT_SUBREAPER:-}" ]; then
  export PILOT_SUBREAPER=1
  exec python3 -c 'import ctypes, os, sys
if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
    sys.exit("prctl(PR_SET_CHILD_SUBREAPER) failed")
os.execvp("bash", ["bash", *sys.argv[1:]])' "$0" "$@"
fi
TEST_MODE=${TEST_MODE:-0}
TOTAL_S=600
PRIOR_S=${PRIOR_S:-0}
RERUN_CAP_S=${RERUN_CAP_S:-600}
TIMING_CAP_S=${TIMING_CAP_S:-430}
DISK_BYTES=${DISK_BYTES:-300000000}
CACHE_BYTES=${CACHE_BYTES:-700000000}
NSYS_BYTES=${NSYS_BYTES:-200000000}
# Free space the shared filesystem must have before launch (cache + artifacts + margin).
MIN_FREE_BYTES=${MIN_FREE_BYTES:-5000000000}
S=/mnt/shared/home/kevin/workspace/serving-studio
G=$S/gb300/serving-validation
P=${PILOT_DIR:-$G/phase3b2/pilot}
WT=$S/ServingStudioSim-gb300
PY=$G/venv/bin/python
NSYS=/usr/local/bin/nsys
SRC_CACHE=${SRC_CACHE:-$G/cache/nv-compute}
UUID1=GPU-9574ced0-46be-d5b7-0098-01fa1e84dccb
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP=$(date -u +%Y%m%d-%H%M%S)
OUT=$P/run-$STAMP
SCRATCH=$P/scratch-$STAMP
NVPRIV=$SCRATCH/nv
mkdir -p "$OUT"/{timing,identity,cache/inductor,cache/triton,cache/vllm,cache/cupti} "$SCRATCH"
# Temporary files of the passes (Nsight's included) on local disk, counted with
# the artifacts and deleted at the end.
LOCALTMP=${LOCALTMP_ROOT:-/tmp}/pilot3b2-$STAMP
mkdir -p "$LOCALTMP"
export TMPDIR="$LOCALTMP"
export PILOT_OUT="$OUT" PILOT_NV="$NVPRIV"   # read by TEST_MODE stand-in passes only
OWN_ROOT=$$
OWN_FILE="$OUT/own-pids.txt"
: >"$OWN_FILE"
# shellcheck source=/dev/null
source $S/logs/proc_own.sh
BUDGET_S=$(python3 -c "print(min($RERUN_CAP_S, $TOTAL_S - $PRIOR_S))")
USED_S=0
PASS_T0=
PASS_NAME=

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/steps.log"; }
now() { date +%s.%N; }
account() { # close the open pass's interval (after its processes are gone)
  [ -z "$PASS_T0" ] && return 0
  local dt
  dt=$(python3 -c "print(round($(now) - $PASS_T0, 1))")
  USED_S=$(python3 -c "print(round($USED_S + $dt, 1))")
  log "pass $PASS_NAME occupancy ${dt} s; rerun total ${USED_S} s of ${BUDGET_S}; aggregate $(python3 -c "print(round($PRIOR_S + $USED_S, 1))") s of ${TOTAL_S}"
  printf '{"prior_s": %s, "rerun_s": %s, "rerun_budget_s": %s, "aggregate_s": %s, "aggregate_cap_s": %s}\n' \
    "$PRIOR_S" "$USED_S" "$BUDGET_S" "$(python3 -c "print(round($PRIOR_S + $USED_S, 1))")" "$TOTAL_S" >"$OUT/occupancy.json"
  PASS_T0=
}
cleanup() {
  local pids
  # Never `kill 0` (the whole process group): only helpers that were started.
  [ -n "${WATCH:-}" ] && kill "$WATCH" 2>/dev/null
  [ -n "${SAMPLER:-}" ] && kill "$SAMPLER" 2>/dev/null
  WATCH= SAMPLER=
  record_own
  [ -n "${PASS:-}" ] && kill -INT "$PASS" 2>/dev/null
  for _ in $(seq 1 20); do [ -z "$(live_own | grep -vx "$$")" ] && break; sleep 1; done
  pids="$(live_own | grep -vx "$$" | sort -u | tr '\n' ' ')"
  # shellcheck disable=SC2086
  [ -n "${pids// /}" ] && { log "cleanup INT $pids"; kill -INT $pids 2>/dev/null; sleep 10; }
  pids="$(live_own | grep -vx "$$" | sort -u | tr '\n' ' ')"
  # shellcheck disable=SC2086
  [ -n "${pids// /}" ] && { log "cleanup KILL $pids"; kill -KILL $pids 2>/dev/null; sleep 2; }
  PASS=
  return 0
}
cache_listing() { # cache_listing DIR -> sha256 of (relative path, size, mtime) of every file
  (cd "$1" && find . -type f -printf '%P %s %T@\n' | sort | sha256sum | cut -d' ' -f1)
}
cache_report() { # cache_report LABEL
  local bytes files
  bytes=$(du -sb "$NVPRIV" 2>/dev/null | cut -f1)
  files=$(find "$NVPRIV" -type f 2>/dev/null | wc -l)
  echo "$1 $bytes $files" >>"$OUT/private-cache.txt"
  log "private CUDA cache $1: $bytes bytes, $files files"
}
FINALIZED=0
finalize() { # the `finally` of the run: account, cache report, delete only the private cache
  [ $FINALIZED -eq 1 ] && return 0
  FINALIZED=1
  account
  if [ -d "$NVPRIV" ]; then
    cache_report final
    python3 - "$OUT/private-cache.txt" <<'PY' | tee -a "$OUT/steps.log"
import sys
rows = {l.split()[0]: (int(l.split()[1]), int(l.split()[2])) for l in open(sys.argv[1])}
i, f = rows.get("initial", (0, 0)), rows.get("final", (0, 0))
print(f"private CUDA cache grew by {f[0] - i[0]} bytes, {f[1] - i[1]} files")
PY
  fi
  [ "$TEST_MODE" = 1 ] || { cache_listing "$SRC_CACHE" >"$OUT/source-cache-listing-post.txt"
    cmp -s "$OUT/source-cache-listing-pre.txt" "$OUT/source-cache-listing-post.txt" \
      && log "source cache $SRC_CACHE unchanged" || log "source cache listing CHANGED"; }
  rm -rf "$SCRATCH" && log "deleted private cache $SCRATCH"
  log "local temporary files at the end: $(du -sb "$LOCALTMP" 2>/dev/null | cut -f1) bytes"
  rm -rf "$LOCALTMP" && log "deleted $LOCALTMP"
}
trap finalize EXIT
gpu1_baseline() { # gpu1_baseline LABEL -> fails unless GPU 1 is idle and empty
  [ "$TEST_MODE" = 1 ] && { echo "TEST_MODE: no GPU query"; return 0; }
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu \
    --format=csv >"$OUT/gpus-$1.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader >"$OUT/apps-$1.csv"
  local u util mem
  u="$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)"
  [ "$u" = "$UUID1" ] || { echo "GPU 1 is $u"; return 1; }
  grep -q "$UUID1" "$OUT/apps-$1.csv" && { echo "compute process on GPU 1: $(grep "$UUID1" "$OUT/apps-$1.csv")"; return 1; }
  util="$(nvidia-smi -i 1 --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
  mem="$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)"
  echo "GPU 1 util ${util} % mem ${mem} MiB"
  [ "$util" -le 5 ] && [ "$mem" -lt 1024 ]
}
fingerprints() { # fingerprints LABEL
  [ "$TEST_MODE" = 1 ] && return 0
  (cd /tmp && CUDA_VISIBLE_DEVICES= "$PY" -I - "$WT/profiling/env/arm64-gb300/serving-validation/env-fingerprint.json" <<'PY') >"$OUT/serving-env-$1.txt" 2>&1
import hashlib, importlib.metadata as md, json, sys
want = json.load(open(sys.argv[1]))["packages_sha256"]
pk = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions())
got = hashlib.sha256("\n".join(pk).encode()).hexdigest()
print(got, "==" if got == want else "!=", want)
sys.exit(0 if got == want else 1)
PY
  local rc=$?
  sha256sum $S/ServingStudioSim/profiling/profile.db $S/gb300/db/profile-gb300.db >"$OUT/db-sha256-$1.txt"
  return $rc
}
POSTED=0
post_checks() {
  [ $POSTED -eq 1 ] && return 0
  POSTED=1
  local ok=0
  for _ in $(seq 1 30); do gpu1_baseline post >"$OUT/gpu1-post.txt" 2>&1 && { ok=1; break; }; sleep 1; done
  [ $ok -eq 1 ] && log "GPU 1 back to baseline: $(cat "$OUT/gpu1-post.txt")" || log "GPU 1 NOT at baseline after run: $(cat "$OUT/gpu1-post.txt")"
  [ "$TEST_MODE" = 1 ] && return 0
  fingerprints post && log "serving env fingerprint unchanged" || log "serving env fingerprint CHANGED"
  cmp -s "$OUT/db-sha256-pre.txt" "$OUT/db-sha256-post.txt" && log "databases unchanged" || log "a profile database CHANGED"
  log "artifacts (excluding the private cache) $(du -scb "$OUT" "$LOCALTMP" | tail -1 | cut -f1) bytes; peak sampled $(awk -F'[= ]' '{if ($3>m) m=$3} END {print m+0}' "$OUT/sizes.log") bytes"
}
fail() { # stop: processes first, then account the pass up to their actual end
  trap - USR1 USR2 HUP
  log "STOP: $1"
  [ "$TEST_MODE" = 1 ] || nvidia-smi >"$OUT/nvidia-smi-at-stop.txt" 2>&1
  cleanup
  account
  post_checks
  exit "${2:-1}"
}

log "pilot out=$OUT test_mode=$TEST_MODE subreaper=${PILOT_SUBREAPER:-0} prior_s=$PRIOR_S rerun_budget_s=$BUDGET_S root_pid=$$"
if [ "$TEST_MODE" != 1 ]; then
  {
    echo "worktree $(git -C "$WT" rev-parse HEAD) dirty=$(git -C "$WT" status --porcelain | wc -l)"
    echo "vllm-fork $(git -C $G/vllm-fork rev-parse HEAD 2>/dev/null)"
    echo "driver $(sha256sum "$HERE/pilot_driver2.py" | cut -d' ' -f1)"
    echo "guard $(sha256sum "$HERE/run_pilot2.sh" | cut -d' ' -f1)"
    echo "nsys $($NSYS --version 2>&1 | tail -1)"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 1 | sed 's/^/driver /'
  } >"$OUT/commits.txt"
  fingerprints pre || fail "serving environment differs from the committed fingerprint" 2
fi
cache_listing "$SRC_CACHE" >"$OUT/source-cache-listing-pre.txt"
SRC_BYTES=$(du -sb "$SRC_CACHE" | cut -f1)
FREE_BYTES=$(( $(df -B1 --output=avail "$P" | tail -1) ))
log "capacity: source cache ${SRC_BYTES} B (cap ${CACHE_BYTES} B), free ${FREE_BYTES} B (need ${MIN_FREE_BYTES} B)"
[ "$SRC_BYTES" -lt "$CACHE_BYTES" ] || fail "source cache ${SRC_BYTES} B is not below the ${CACHE_BYTES} B cap" 2
[ "$FREE_BYTES" -ge "$MIN_FREE_BYTES" ] || fail "only ${FREE_BYTES} B free" 2
cp -a "$SRC_CACHE" "$NVPRIV" || fail "could not copy $SRC_CACHE" 2
cache_report initial
if [ "$TEST_MODE" != 1 ]; then
  cp -r $G/cache/flashinfer "$OUT/cache/flashinfer"
  # The venv is called by path, not activated: its bin (ninja for the CUPTI build) goes on PATH.
  export PATH="$G/venv/bin:$PATH" CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$WT:$G/vllm-fork" HF_HUB_OFFLINE=1 \
    TORCHINDUCTOR_CACHE_DIR="$OUT/cache/inductor" TRITON_CACHE_DIR="$OUT/cache/triton" \
    CUDA_CACHE_PATH="$NVPRIV" CUDA_CACHE_MAXSIZE=$CACHE_BYTES VLLM_CACHE_ROOT="$OUT/cache/vllm" \
    VIBESIM_PROFILE_ENERGY=0 \
    FLASHINFER_WORKSPACE_BASE="$OUT/cache/flashinfer" MOESIM_CUPTI_BUILD_DIR="$OUT/cache/cupti"
  # CUPTI extension: C++ only, built with no device visible, before the GPU window.
  CUDA_VISIBLE_DEVICES= "$PY" -c "from profiling.profilers.cupti_kernel_profiler import _get_extension; _get_extension(); print('cupti extension ready')" \
    >"$OUT/cupti-build.log" 2>&1 || fail "CUPTI extension build failed (CPU)" 2
  log "$(tail -1 "$OUT/cupti-build.log")"
fi
# Immediately before launch.
gpu1_baseline pre >"$OUT/gpu1-pre.txt" 2>&1 || fail "GPU 1 not idle: $(cat "$OUT/gpu1-pre.txt")" 4
log "pre: $(cat "$OUT/gpu1-pre.txt"); other compute processes: $(tr '\n' ';' <"$OUT/apps-pre.csv" 2>/dev/null)"

watch_run() {
  local pid u
  while sleep 2; do
    record_own
    if [ "$TEST_MODE" != 1 ]; then
      nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader >"$OUT/apps-watch.csv" 2>/dev/null
      while IFS=, read -r u pid _; do
        pid="${pid// /}"
        [ -z "$pid" ] && continue
        if [ "$u" = "$UUID1" ]; then
          is_own "$pid" && continue
          echo "$(date -u +%FT%TZ) foreign pid $pid on GPU 1: $(ps -o user=,args= -p "$pid" | cut -c1-160)" >>"$OUT/steps.log"
          kill -USR1 $$; return
        elif is_own "$pid"; then
          echo "$(date -u +%FT%TZ) pilot pid $pid on another GPU $u" >>"$OUT/steps.log"
          kill -USR2 $$; return
        fi
      done <"$OUT/apps-watch.csv"
      grep -v "$UUID1" "$OUT/apps-watch.csv" | cut -d, -f1,2,4 | tr '\n' ';' | sed "s/^/$(date -u +%T) /" >>"$OUT/other-gpus.log"
      echo >>"$OUT/other-gpus.log"
    fi
    local ob cb
    ob=$(du -scb "$OUT" "$LOCALTMP" 2>/dev/null | tail -1 | cut -f1); cb=$(du -sb "$NVPRIV" 2>/dev/null | cut -f1)
    echo "$(date -u +%T) artifacts_bytes=$ob private_cache_bytes=$cb" >>"$OUT/sizes.log"
    [ "${ob:-0}" -gt "$DISK_BYTES" ] && { echo "$(date -u +%FT%TZ) artifacts $ob B above ${DISK_BYTES} B" >>"$OUT/steps.log"; kill -HUP $$; return; }
    [ "${cb:-0}" -gt "$CACHE_BYTES" ] && { echo "$(date -u +%FT%TZ) private CUDA cache $cb B above ${CACHE_BYTES} B" >>"$OUT/steps.log"; kill -HUP $$; return; }
    for f in $(find "$OUT" "$LOCALTMP" -name '*.nsys-rep' -o -name '*.qdstrm' 2>/dev/null); do
      [ -f "$f" ] && [ "$(du -sb "$f" | cut -f1)" -gt "$NSYS_BYTES" ] && { echo "$(date -u +%FT%TZ) $f above ${NSYS_BYTES} B" >>"$OUT/steps.log"; kill -HUP $$; return; }
    done
  done
}
trap 'fail "foreign compute process on GPU 1" 4' USR1
trap 'fail "pilot process on a GPU other than GPU 1" 4' USR2
trap 'fail "disk, cache or trace budget exceeded" 6' HUP
watch_run &
WATCH=$!
if [ "$TEST_MODE" != 1 ]; then
  nvidia-smi -i 1,2 --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm \
    --format=csv -lms 1000 >"$OUT/gpu-samples.csv" 2>/dev/null &
  SAMPLER=$!
fi

pass_leftovers() { # recorded processes of this run that belong to a pass (by command line)
  record_own
  local pid
  for pid in $(live_own); do
    tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | grep -qE 'pilot_driver2|nsys|_inductor|compile_worker|pilot_test_pass' && echo "$pid"
  done
}
remaining() { python3 -c "print(int($BUDGET_S - $USED_S - 20))"; }
run_pass() { # run_pass NAME TIMEOUT CMD...  (returns the command's rc)
  local limit=$2 rc
  PASS_NAME=$1
  shift 2
  log "pass $PASS_NAME start (timeout ${limit} s)"
  PASS_T0=$(now)
  timeout --signal=INT --kill-after=15 "$limit" "$@" >"$OUT/$PASS_NAME.out" 2>"$OUT/$PASS_NAME.err" &
  PASS=$!
  wait "$PASS"
  rc=$?
  PASS=
  # The pass's own descendants (Inductor compile workers, the Nsight agent) must be gone too.
  for _ in $(seq 1 30); do [ -z "$(pass_leftovers)" ] && break; sleep 1; done
  [ -n "$(pass_leftovers)" ] && log "pass $PASS_NAME leftovers still alive: $(pass_leftovers | tr '\n' ' ')"
  account
  log "pass $PASS_NAME rc=$rc"
  return $rc
}
# The driver stops starting new cases 60 s before the timing pass's hard limit.
SOFT_T=$(python3 -c "print(max(30, min($TIMING_CAP_S, $(remaining)) - 60))")
if [ "$TEST_MODE" = 1 ]; then
  # `exec -a` names the stand-in process so pass_leftovers recognises it.
  CMD_T=(bash -c "${TEST_CMD_T:-exec -a pilot_test_pass sleep 3}")
  CMD_I=(bash -c "${TEST_CMD_I:-exec -a pilot_test_pass sleep 2}")
else
  CMD_T=("$PY" "$HERE/pilot_driver2.py" --mode timing --out "$OUT/timing" --expect-uuid "$UUID1"
         --soft-deadline-s "$SOFT_T")
  CMD_I=("$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none
         --force-overwrite=true -o "$OUT/identity/identity"
         "$PY" "$HERE/pilot_driver2.py" --mode identity --out "$OUT/identity" --expect-uuid "$UUID1"
         --soft-deadline-s 110)
fi

LIMIT=$(python3 -c "print(min($TIMING_CAP_S, $(remaining)))")
run_pass timing "$LIMIT" "${CMD_T[@]}"
rc=$?
[ $rc -eq 124 ] && fail "timing pass reached its ${LIMIT} s limit" 5
[ $rc -eq 3 ] && fail "oracle failure in the timing pass (see timing/timing.json)" 3
[ $rc -ne 0 ] && fail "timing pass returned $rc (see timing.err)" 3
LIMIT=$(remaining)
[ "$LIMIT" -ge 30 ] || fail "only ${LIMIT} s of GPU budget left for the identity pass" 5
run_pass identity "$LIMIT" "${CMD_I[@]}"
rc=$?
[ $rc -eq 124 ] && fail "identity pass reached its ${LIMIT} s limit" 5
[ $rc -ne 0 ] && fail "identity pass returned $rc (see identity.err)" 3
trap - USR1 USR2 HUP
cleanup
post_checks
if [ "$TEST_MODE" != 1 ]; then
  "$NSYS" export --type sqlite --force-overwrite true -o "$OUT/identity/identity.sqlite" "$OUT/identity/identity.nsys-rep" \
    >"$OUT/nsys-export.log" 2>&1 && log "sqlite exported (CPU)" || log "sqlite export failed"
fi
log "done"
