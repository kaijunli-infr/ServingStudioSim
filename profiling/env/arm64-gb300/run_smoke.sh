#!/usr/bin/env bash
# GB300 Phase 2B smoke test (SMOKE_PLAN.md), one GPU, no profile-db writes.
#
#   run_smoke.sh [GPU_INDEX] [EXPECTED_UUID]
#
# Runs from the Sim repo root on a GB300 tray. Everything it writes goes to
# $GB300_ROOT/{artifacts/smoke-<stamp>,cache,logs}; `--fresh` means no rows.
# Overall bound: 45 min wall (outer `timeout`), per-step bounds below.
set -u
SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$SIM_ROOT"
# shellcheck disable=SC1091
source profiling/env/arm64-gb300/gb300.env
# Non-interactive shells (ssh, setsid) do not read ~/.bashrc, where uv's dir is added.
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

GPU_INDEX="${1:-1}"
EXPECT_UUID="${2:-}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$GB300_ROOT/artifacts/smoke-$STAMP"
mkdir -p "$OUT"
ln -sfn "$OUT" "$GB300_ROOT/artifacts/smoke-latest"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONUNBUFFERED=1
RUN=(uv run --no-sync --project profiling/env/arm64-gb300 python)
PROFILE=(uv run --no-sync --project profiling/env/arm64-gb300 python -m profiling.cli run)
COMMON=(--gpu-name "NVIDIA GB300" --fresh --json)

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/steps.log"; }
step_time() { # step_time NAME TIMEOUT_S cmd...
  local name="$1" to="$2"; shift 2
  local t0 t1 rc
  t0=$(date +%s.%N)
  timeout --signal=INT --kill-after=30 "$to" "$@" >"$OUT/$name.out" 2>"$OUT/$name.err"
  rc=$?
  t1=$(date +%s.%N)
  printf '{"step":"%s","rc":%d,"wall_s":%.3f,"timeout_s":%s}\n' "$name" "$rc" "$(echo "$t1 - $t0" | bc)" "$to" >>"$OUT/steps.jsonl"
  log "step $name rc=$rc wall=$(printf '%.1f' "$(echo "$t1 - $t0" | bc)")s"
  return $rc
}
db_sums() { sha256sum profiling/profile.db "$VIBESIM_PROFILE_DB" 2>/dev/null; }
cache_du() { du -sb "$GB300_ROOT/cache" | cut -f1; }

log "smoke start out=$OUT gpu_index=$GPU_INDEX expect_uuid=${EXPECT_UUID:-none}"
db_sums >"$OUT/db-sha256-pre.txt"
CACHE_PRE=$(cache_du); echo "$CACHE_PRE" >"$OUT/cache-bytes-pre.txt"
nvidia-smi -i "$GPU_INDEX" --query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu --format=csv >"$OUT/gpu-pre.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv >"$OUT/compute-apps-pre.csv"
if [ "$(wc -l <"$OUT/compute-apps-pre.csv")" -gt 1 ]; then
  log "ABORT: compute processes present before start"; cat "$OUT/compute-apps-pre.csv" | tee -a "$OUT/steps.log"; exit 2
fi

# Utilization sampler on the one GPU, 2 Hz, for the whole run.
nvidia-smi -i "$GPU_INDEX" --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu \
  --format=csv -lms 500 >"$OUT/gpu-samples.csv" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

# 0. pre manifest (never touches CUDA)
step_time manifest-pre 120 "${RUN[@]}" profiling/env/arm64-gb300/collect_manifest.py \
  --out "$OUT/manifest-pre.json" --run-config phase=2B-smoke "gpu=tray12:$GPU_INDEX" "expect_uuid=${EXPECT_UUID:-none}" || { log "ABORT: manifest-pre failed"; exit 2; }

# 1. identity gate: name, cc, UUID, ones-matmul. Aborts the run on mismatch.
step_time identity 300 "${RUN[@]}" profiling/env/arm64-gb300/smoke_checks.py \
  ${EXPECT_UUID:+--expect-uuid "$EXPECT_UUID"} --only identity --out "$OUT/identity.json" \
  || { log "ABORT: identity check failed (see identity.out/err)"; exit 3; }
cat "$OUT/identity.out" | tee -a "$OUT/steps.log"

# 2-5. profiler runs (same code path as the TP1 fill), each kind twice:
#      first run carries JIT compile, second is warm-cache.
ATTN='"num_qo_heads":32,"num_kv_heads":8,"head_dim":128,"q_dtype":"bf16","kv_dtype":"bf16","o_dtype":"bf16"'
run_kind() { # run_kind NAME TIMEOUT table backend spec...
  local name="$1" to="$2" table="$3" backend="$4"; shift 4
  local specs=()
  for s in "$@"; do specs+=(--spec "$s"); done
  step_time "$name" "$to" "${PROFILE[@]}" "$table" --backend "$backend" "${COMMON[@]}" --output-dir "$OUT/$name" "${specs[@]}"
}
run_kind gemm-torch_linear 600 single_gemm torch_linear \
  '{"m":64,"n":6144,"k":4096,"dtype":"bf16"}' '{"m":2048,"n":6144,"k":4096,"dtype":"bf16"}' '{"m":4096,"n":6144,"k":4096,"dtype":"bf16"}'
run_kind rmsnorm-cold 900 rms_norm flashinfer '{"m":64,"hidden":4096,"dtype":"bf16"}' '{"m":2048,"hidden":4096,"dtype":"bf16"}'
run_kind rmsnorm-warm 300 rms_norm flashinfer '{"m":64,"hidden":4096,"dtype":"bf16"}' '{"m":2048,"hidden":4096,"dtype":"bf16"}'
run_kind prefill-fa2-cold 900 flashinfer_attn_prefill fa2 "{$ATTN,\"prefix_len\":0,\"append_len\":2048}" "{$ATTN,\"prefix_len\":2048,\"append_len\":2048}"
run_kind prefill-fa2-warm 300 flashinfer_attn_prefill fa2 "{$ATTN,\"prefix_len\":0,\"append_len\":2048}" "{$ATTN,\"prefix_len\":2048,\"append_len\":2048}"
run_kind decode-fa2-cold 900 flashinfer_attn_decode fa2 "{$ATTN,\"batch_size\":1,\"total_tokens\":4096}" "{$ATTN,\"batch_size\":64,\"total_tokens\":38400}"
run_kind decode-fa2-warm 300 flashinfer_attn_decode fa2 "{$ATTN,\"batch_size\":1,\"total_tokens\":4096}" "{$ATTN,\"batch_size\":64,\"total_tokens\":38400}"

# Correctness vs fp32 references + CUDA-event latency distributions (own process).
step_time checks 900 "${RUN[@]}" profiling/env/arm64-gb300/smoke_checks.py \
  ${EXPECT_UUID:+--expect-uuid "$EXPECT_UUID"} --out "$OUT/checks.json"

# 6. FA3 probe: separate process, strict timeout, failure expected.
run_kind fa3-probe 300 flashinfer_attn_prefill fa3 "{$ATTN,\"prefix_len\":0,\"append_len\":2048}"
log "fa3-probe exit recorded (failure is the expected outcome)"

# 7. post manifest, process check, deltas
kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null; trap - EXIT
step_time manifest-post 120 "${RUN[@]}" profiling/env/arm64-gb300/collect_manifest.py \
  --out "$OUT/manifest-post.json" --run-config phase=2B-smoke "gpu=tray12:$GPU_INDEX"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv >"$OUT/compute-apps-post.csv"
nvidia-smi -i "$GPU_INDEX" --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv >"$OUT/gpu-post.csv"
db_sums >"$OUT/db-sha256-post.txt"
CACHE_POST=$(cache_du); echo "$CACHE_POST" >"$OUT/cache-bytes-post.txt"
du -sb "$GB300_ROOT"/cache/* >"$OUT/cache-du-post.txt"
log "cache delta bytes: $((CACHE_POST - CACHE_PRE))"
if cmp -s "$OUT/db-sha256-pre.txt" "$OUT/db-sha256-post.txt"; then log "profile DBs unchanged"; else log "WARNING: a profile DB checksum changed"; fi
log "compute apps after run: $(( $(wc -l <"$OUT/compute-apps-post.csv") - 1 ))"
log "smoke done"
