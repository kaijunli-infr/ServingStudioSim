#!/usr/bin/env bash
# GB300 TP2/TP4 all-reduce probe and fill (TP2_TP4_PROPOSAL.md phase B).
#
#   run_allreduce_fill.sh NUM_GPUS GPU_INDICES EXPECTED_UUIDS
#   run_allreduce_fill.sh 2 2,3 GPU-aaa,GPU-bbb
#
# One stage per call: environment guard, GPU preflight, NCCL probe through the
# profiler's own launcher (nccl_probe.py), then the `all_reduce`/nccl rows for
# NUM_GPUS ranks that the private DB lacks, validation, and a --fresh repeat of
# the whole stage (REPEATS times). Rank r runs on the r-th GPU of GPU_INDICES.
#
# Fail closed: an environment mismatch, a busy or foreign-occupied GPU, a probe
# failure, a timeout or a validator anomaly stops the stage. Nothing is
# reinstalled or repaired here. Every step has its own timeout; on a stop only
# processes of this driver's own session are killed. BUDGET_S (default 780 s)
# bounds the stage.
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
BUDGET_S="${BUDGET_S:-780}"
T_START=$(date +%s)
STAMP="$(date +%Y%m%d-%H%M%S)"
CAMP="$GB300_ROOT/artifacts/allreduce"
OUT="$CAMP/tp$NUM_GPUS-run-$STAMP"
SPECS_ALL="$GB300_ROOT/artifacts/tp24/specs/all_reduce.json"
mkdir -p "$OUT"
ln -sfn "$OUT" "$CAMP/tp$NUM_GPUS-latest"
ENVDIR=profiling/env/arm64-gb300
SID="$(ps -o sid= -p $$ | tr -d ' ')"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDICES"
export PYTHONUNBUFFERED=1
RUN=(uv run --no-sync --project "$ENVDIR" python)
PROFILE=(uv run --no-sync --project "$ENVDIR" python -m profiling.cli run)

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

log "allreduce stage start num_gpus=$NUM_GPUS gpus=$GPU_INDICES uuids=$EXPECT_UUIDS out=$OUT db=$VIBESIM_PROFILE_DB budget_s=$BUDGET_S sid=$SID"
log "source commit $(git rev-parse HEAD) dirty=$([ -n "$(git status --porcelain)" ] && echo yes || echo no)"
[ "${UV_NO_SYNC:-}" = 1 ] || fail "UV_NO_SYNC is not set" 2
[ -f "$SPECS_ALL" ] || fail "no spec file $SPECS_ALL" 2
sha256sum profiling/profile.db "$VIBESIM_PROFILE_DB" >"$OUT/db-sha256-pre.txt"
du -sb "$GB300_ROOT/cache" | cut -f1 >"$OUT/cache-bytes-pre.txt"
ulimit -s >"$OUT/ulimit-stack.txt"

# Gate 0: environment, fail closed.
step_time env-guard-pre 120 "${RUN[@]}" $ENVDIR/env_guard.py --out "$OUT/env-guard-pre.json" \
  || fail "environment mismatch; see env-guard-pre.out (nothing was repaired)" 2
preflight start

python3 - "$SPECS_ALL" "$NUM_GPUS" "$OUT/specs.json" <<'PY'
import json, sys
specs = [s for s in json.load(open(sys.argv[1]))["specs"] if s["num_gpus"] == int(sys.argv[2])]
specs.sort(key=lambda s: s["message_size_bytes"])
json.dump({"specs": specs}, open(sys.argv[3], "w"), indent=1)
print(len(specs))
PY
[ "$(grep -c message_size_bytes "$OUT/specs.json")" -eq 19 ] || fail "expected 19 specs for $NUM_GPUS GPUs" 2

# shellcheck disable=SC2086
nvidia-smi -i "$GPU_INDICES" --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu \
  --format=csv -lms 500 >"$OUT/gpu-samples.csv" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

step_time manifest-pre 120 "${RUN[@]}" $ENVDIR/collect_manifest.py --out "$OUT/manifest-pre.json" \
  --run-config phase=allreduce-fill "gpus=tray12:$GPU_INDICES" arch=llama3_dense_tp "tp_size=$NUM_GPUS" || fail "manifest-pre" 2

# Gate 1: rank group, rank-to-GPU mapping, SUM correctness, init vs steady state.
# shellcheck disable=SC2086
NCCL_DEBUG=INFO step_time probe 240 "${RUN[@]}" $ENVDIR/nccl_probe.py --num-gpus "$NUM_GPUS" \
  --expect-uuids ${EXPECT_UUIDS//,/ } --out-dir "$OUT/probe" \
  || fail "NCCL probe failed or timed out; see probe.out/err" 3
grep -E "^rank|^steady|^NCCL PROBE" "$OUT/probe.out" | tee -a "$OUT/steps.log"
preflight "after probe"

# Reviewed re-measurement: FORCE_SPECS is the list allreduce_spikes.py wrote
# (stored rows that fresh repeats did not reproduce); --force replaces them.
if [ -n "${FORCE_SPECS:-}" ]; then
  cp "$FORCE_SPECS" "$OUT/force-specs.json"
  step_time force 300 "${PROFILE[@]}" all_reduce --backend nccl --gpu-name "NVIDIA GB300" --json --force \
    --output-dir "$OUT/force" --specs "$OUT/force-specs.json" || fail "forced re-measurement failed" 6
  preflight "after force"
fi

# Fill: the profiler measures only the specs the DB lacks.
step_time fill 420 "${PROFILE[@]}" all_reduce --backend nccl --gpu-name "NVIDIA GB300" --json \
  --output-dir "$OUT/fill" --specs "$OUT/specs.json"
rc=$?
tail -c 600 "$OUT/fill.err" | tee -a "$OUT/steps.log"
step_time validate 120 "${RUN[@]}" $ENVDIR/tp1_validate.py batch --db "$VIBESIM_PROFILE_DB" \
  --kind all_reduce --backend nccl --specs "$OUT/specs.json" \
  --results "$OUT/fill/results.json" --stderr "$OUT/fill.err" --report "$OUT/validate.json"
vrc=$?
tee -a "$OUT/steps.log" <"$OUT/validate.out"
# rc 1 = anomaly (rows complete, a ratio or shape outside the validator's
# thresholds): stops the stage unless a reviewed rerun sets ACCEPT_ANOMALY to
# the written reason. rc 2 = missing or invalid rows: always stops.
if [ $vrc -eq 1 ] && [ -n "${ACCEPT_ANOMALY:-}" ]; then
  log "validator anomaly accepted after review: $ACCEPT_ANOMALY"
elif [ $vrc -ne 0 ]; then
  fail "validation returned $vrc (fill rc=$rc)" "$vrc"
fi
[ $rc -ne 0 ] && fail "fill returned $rc although validation passed" "$rc"
preflight "after fill"

# Repeatability: the whole stage again in fresh rank groups (REPEATS, default
# 2), no DB write.
rrc=0
for i in $(seq 1 "${REPEATS:-2}"); do
  step_time "repeat$i" 420 "${PROFILE[@]}" all_reduce --backend nccl --gpu-name "NVIDIA GB300" --json --fresh \
    --output-dir "$OUT/repeat$i" --specs "$OUT/specs.json" || fail "repeat run $i failed" 6
  step_time "validate-repeat$i" 120 "${RUN[@]}" $ENVDIR/tp1_validate.py repeat --db "$VIBESIM_PROFILE_DB" \
    --kind all_reduce --backend nccl --results "$OUT/repeat$i/results.json" --report "$OUT/validate-repeat$i.json" || rrc=1
  tee -a "$OUT/steps.log" <"$OUT/validate-repeat$i.out"
done
preflight "after repeat"

kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null; trap - EXIT
step_time manifest-post 120 "${RUN[@]}" $ENVDIR/collect_manifest.py --out "$OUT/manifest-post.json" \
  --run-config phase=allreduce-fill "gpus=tray12:$GPU_INDICES" arch=llama3_dense_tp "tp_size=$NUM_GPUS"
step_time env-guard-post 120 "${RUN[@]}" $ENVDIR/env_guard.py --out "$OUT/env-guard-post.json" --same-as "$OUT/env-guard-pre.json" \
  || log "WARNING: environment changed during the stage; see env-guard-post.out"
sha256sum profiling/profile.db "$VIBESIM_PROFILE_DB" >"$OUT/db-sha256-post.txt"
du -sb "$GB300_ROOT/cache" | cut -f1 >"$OUT/cache-bytes-post.txt"
log "cache delta bytes: $(( $(cat "$OUT/cache-bytes-post.txt") - $(cat "$OUT/cache-bytes-pre.txt") ))"
if [ "$(head -1 "$OUT/db-sha256-pre.txt")" = "$(head -1 "$OUT/db-sha256-post.txt")" ]; then log "tracked profiling/profile.db unchanged"; else log "WARNING: tracked profile.db changed"; fi
log "wall used: $(elapsed) s of $BUDGET_S"
if [ $rrc -ne 0 ] && [ -n "${ACCEPT_ANOMALY:-}" ]; then
  log "repeatability anomaly accepted after review: $ACCEPT_ANOMALY"
elif [ $rrc -ne 0 ]; then
  fail "a repeatability check did not pass" 6
fi
log "allreduce stage tp$NUM_GPUS done"
