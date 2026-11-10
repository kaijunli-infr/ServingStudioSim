#!/usr/bin/env bash
# GB300 Phase 2D: diagnostic re-measurement of the small-message all-reduce rows.
#
#   run_allreduce_diag.sh NUM_GPUS GPU_INDICES EXPECTED_UUIDS
#   run_allreduce_diag.sh 2 2,3 GPU-aaa,GPU-bbb
#
# One stage per call: environment guard, GPU preflight, then GROUPS (5) fresh
# NCCL rank groups, each running allreduce_diag.py over the 12 message sizes
# below 16 MiB (SUM check, warm-up, three batches of 100 collectives). Rank r
# runs on the r-th GPU of GPU_INDICES. No profile database is written: the
# private and the tracked database must have the same sha256 before and after.
#
# Fail closed: an environment mismatch, a busy or foreign-occupied GPU, a
# correctness failure or a timeout stops the stage; completed groups stay on
# disk. Nothing is repaired. On a stop only processes of this driver's own
# session are killed. BUDGET_S (default 480 s) bounds the stage.
set -u
SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$SIM_ROOT"
# shellcheck disable=SC1091
source profiling/env/arm64-gb300/gb300.env
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PATH="$GB300_ROOT/venv/bin:$PATH"

NUM_GPUS="${1:?NUM_GPUS}"
GPU_INDICES="${2:?GPU_INDICES}"
EXPECT_UUIDS="${3:?EXPECTED_UUIDS}"
BUDGET_S="${BUDGET_S:-480}"
GROUPS_N=5
T_START=$(date +%s)
STAMP="$(date +%Y%m%d-%H%M%S)"
CAMP="$GB300_ROOT/artifacts/phase2d"
OUT="$CAMP/tp$NUM_GPUS-run-$STAMP"
mkdir -p "$OUT"
ln -sfn "$OUT" "$CAMP/tp$NUM_GPUS-latest"
ENVDIR=profiling/env/arm64-gb300
SID="$(ps -o sid= -p $$ | tr -d ' ')"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDICES"
export PYTHONUNBUFFERED=1
RUN=(uv run --no-sync --project "$ENVDIR" python)

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/steps.log"; }
elapsed() { echo $(( $(date +%s) - T_START )); }
step_time() { # step_time NAME TIMEOUT_S cmd...
  local name="$1" to="$2"; shift 2
  local t0 t1 rc left
  left=$(( BUDGET_S - $(elapsed) ))
  [ "$left" -lt 20 ] && fail "stage budget exhausted before $name" 5
  [ "$to" -gt "$left" ] && to="$left"
  t0=$(date +%s.%N)
  timeout --signal=INT --kill-after=20 "$to" "$@" >"$OUT/$name.out" 2>"$OUT/$name.err"
  rc=$?
  t1=$(date +%s.%N)
  printf '{"step":"%s","rc":%d,"wall_s":%.3f,"timeout_s":%s}\n' "$name" "$rc" "$(echo "$t1 - $t0" | bc)" "$to" >>"$OUT/steps.jsonl"
  log "step $name rc=$rc wall=$(printf '%.1f' "$(echo "$t1 - $t0" | bc)")s"
  return $rc
}
# Kill what this driver started and nothing else: members of its own session.
cleanup() {
  local pids
  kill "${SAMPLER:-0}" 2>/dev/null
  pids="$(ps -s "$SID" -o pid= | grep -vw "$$" | tr '\n' ' ')"
  # shellcheck disable=SC2086
  [ -n "${pids// /}" ] && kill -KILL $pids 2>/dev/null
  return 0
}
fail() { log "STOP: $1"; nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv >"$OUT/gpu-at-stop.csv"; cleanup; exit "${2:-1}"; }
# Preflight / boundary check: identity of the stage's GPUs, low utilisation on
# them, and no compute process on any GPU of the tray (ours have exited).
preflight() { # preflight LABEL
  local apps got util idx
  got="$(for idx in ${GPU_INDICES//,/ }; do nvidia-smi -i "$idx" --query-gpu=uuid --format=csv,noheader; done | paste -sd,)"
  [ "$got" = "$EXPECT_UUIDS" ] || fail "GPU identity at $1: indices $GPU_INDICES are $got, expected $EXPECT_UUIDS" 4
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader)"
  util="$(nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';')"
  printf '%s %s apps=%s gpus=%s\n' "$(date -u +%FT%TZ)" "$1" "${apps:-none}" "$util" >>"$OUT/isolation.log"
  [ -n "$apps" ] && fail "compute process on a tray GPU at $1: $apps" 4
  for idx in ${GPU_INDICES//,/ }; do
    util="$(nvidia-smi -i "$idx" --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
    [ "$util" -le 5 ] || fail "GPU $idx utilisation $util % at $1" 4
  done
  return 0
}

log "phase2d stage start num_gpus=$NUM_GPUS gpus=$GPU_INDICES uuids=$EXPECT_UUIDS out=$OUT db=$VIBESIM_PROFILE_DB budget_s=$BUDGET_S sid=$SID"
log "source commit $(git rev-parse HEAD) dirty=$([ -n "$(git status --porcelain)" ] && echo yes || echo no)"
[ "${UV_NO_SYNC:-}" = 1 ] || fail "UV_NO_SYNC is not set" 2
sha256sum profiling/profile.db "$VIBESIM_PROFILE_DB" >"$OUT/db-sha256-pre.txt"
du -sb "$GB300_ROOT/cache" | cut -f1 >"$OUT/cache-bytes-pre.txt"
ulimit -s >"$OUT/ulimit-stack.txt"

# Gate 0: environment, fail closed.
step_time env-guard-pre 120 "${RUN[@]}" $ENVDIR/env_guard.py --out "$OUT/env-guard-pre.json" \
  || fail "environment mismatch; see env-guard-pre.out (nothing was repaired)" 2
preflight start

# shellcheck disable=SC2086
nvidia-smi -i "$GPU_INDICES" --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu \
  --format=csv -lms 500 >"$OUT/gpu-samples.csv" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

step_time manifest-pre 120 "${RUN[@]}" $ENVDIR/collect_manifest.py --out "$OUT/manifest-pre.json" \
  --run-config phase=2d-allreduce-diagnostic "gpus=tray12:$GPU_INDICES" arch=llama3_dense_tp "tp_size=$NUM_GPUS" || fail "manifest-pre" 2

for g in $(seq 1 $GROUPS_N); do
  # shellcheck disable=SC2086
  step_time "group$g" 150 "${RUN[@]}" $ENVDIR/allreduce_diag.py --num-gpus "$NUM_GPUS" \
    --expect-uuids ${EXPECT_UUIDS//,/ } --group "$g" --out-dir "$OUT/raw" \
    || fail "diagnostic group $g failed or timed out; see group$g.out/err" 3
  grep -E "^DIAG GROUP" "$OUT/group$g.out" | tee -a "$OUT/steps.log"
  preflight "after group $g"
done

kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null; trap - EXIT
step_time env-guard-post 120 "${RUN[@]}" $ENVDIR/env_guard.py --out "$OUT/env-guard-post.json" --same-as "$OUT/env-guard-pre.json" \
  || fail "environment changed during the stage; see env-guard-post.out" 2
sha256sum profiling/profile.db "$VIBESIM_PROFILE_DB" >"$OUT/db-sha256-post.txt"
cmp -s "$OUT/db-sha256-pre.txt" "$OUT/db-sha256-post.txt" || fail "a profile database changed during the stage" 7
log "private and tracked profile databases unchanged"
log "wall used: $(elapsed) s of $BUDGET_S"
log "phase2d stage tp$NUM_GPUS done"
