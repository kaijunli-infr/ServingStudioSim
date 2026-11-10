#!/usr/bin/env bash
# GB300 Llama 3.1 8B BF16 TP2/TP4 single-GPU fill (TP2_TP4_PROPOSAL.md phase A).
#
#   run_tp24_fill.sh [GPU_INDEX] [EXPECTED_UUID]
#
# Measures the per-rank kernels a TP2 or TP4 replica needs that the TP1 fill
# does not cover (spec files prepared on CPU by tp_new_specs.py under
# $GB300_ROOT/artifacts/tp24/specs). No communication kernel is run here.
#
# One batch per (kind, backend): fill, validate, mark done. Restartable twice
# over: a validated batch leaves $STATE/<batch>.done and is skipped on the next
# run, and inside a batch the profiler measures only the specs the DB lacks.
# A wall-clock budget (BUDGET_S, default 3300 s) is checked before each batch
# and caps its timeout; when it runs out the driver stops between batches.
set -u
SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$SIM_ROOT"
# shellcheck disable=SC1091
source profiling/env/arm64-gb300/gb300.env
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# ninja for the CUPTI timer extension and the shim's interpreter.
export PATH="$GB300_ROOT/venv/bin:$PATH"

GPU_INDEX="${1:-1}"
EXPECT_UUID="${2:-}"
BUDGET_S="${BUDGET_S:-3300}"
T_START=$(date +%s)
STAMP="$(date +%Y%m%d-%H%M%S)"
CAMP="$GB300_ROOT/artifacts/tp24"
SPECS="$CAMP/specs"
STATE="$CAMP/state"
OUT="$CAMP/run-$STAMP"
mkdir -p "$OUT" "$STATE"
ln -sfn "$OUT" "$CAMP/run-latest"
ENVDIR=profiling/env/arm64-gb300
SHIM="$VIBESIM_VLLM_FORK_ROOT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONUNBUFFERED=1
RUN=(uv run --no-sync --project "$ENVDIR" python)
PROFILE=(uv run --no-sync --project "$ENVDIR" python -m profiling.cli run)

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/steps.log"; }
elapsed() { echo $(( $(date +%s) - T_START )); }
step_time() { # step_time NAME TIMEOUT_S cmd...
  local name="$1" to="$2"; shift 2
  local t0 t1 rc
  t0=$(date +%s.%N)
  timeout --signal=INT --kill-after=60 "$to" "$@" >"$OUT/$name.out" 2>"$OUT/$name.err"
  rc=$?
  t1=$(date +%s.%N)
  printf '{"step":"%s","rc":%d,"wall_s":%.3f,"timeout_s":%s}\n' "$name" "$rc" "$(echo "$t1 - $t0" | bc)" "$to" >>"$OUT/steps.jsonl"
  log "step $name rc=$rc wall=$(printf '%.1f' "$(echo "$t1 - $t0" | bc)")s"
  return $rc
}
fail() { log "STOP: $*"; kill "${SAMPLER:-0}" 2>/dev/null; exit "${2:-1}"; }
# Isolation: between our own steps no compute process may exist on this tray's
# GPUs. One on our GPU stops the run; one elsewhere is recorded.
isolation() { # isolation LABEL
  local apps mine
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader)"
  printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$1" "${apps:-none}" >>"$OUT/isolation.log"
  mine="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader)"
  if [ -n "$apps" ] && printf '%s\n' "$apps" | grep -q "$mine"; then
    fail "foreign compute process on GPU $GPU_INDEX at $1: $apps" 4
  fi
  [ -n "$apps" ] && log "note: compute process on another GPU at $1: $apps"
  return 0
}

log "tp24 fill start out=$OUT gpu_index=$GPU_INDEX expect_uuid=${EXPECT_UUID:-none} db=$VIBESIM_PROFILE_DB budget_s=$BUDGET_S"
log "source commit $(git rev-parse HEAD) dirty=$([ -n "$(git status --porcelain)" ] && echo yes || echo no)"
[ -f "$SPECS/summary.json" ] || fail "no spec files at $SPECS (run tp_new_specs.py on CPU first)" 2
[ -f "$SHIM/manifest.json" ] || fail "no shim manifest at $SHIM" 2
cp "$SHIM/manifest.json" "$OUT/shim-manifest.json"
cp "$SPECS/summary.json" "$OUT/specs-summary.json"
sha256sum profiling/profile.db "$VIBESIM_PROFILE_DB" >"$OUT/db-sha256-pre.txt"
du -sb "$GB300_ROOT/cache" | cut -f1 >"$OUT/cache-bytes-pre.txt"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv >"$OUT/gpu-pre.csv"
isolation start

nvidia-smi -i "$GPU_INDEX" --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active \
  --format=csv -lms 500 >"$OUT/gpu-samples.csv" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

step_time manifest-pre 120 "${RUN[@]}" $ENVDIR/collect_manifest.py --out "$OUT/manifest-pre.json" \
  --run-config phase=tp24-single-gpu-fill "gpu=tray12:$GPU_INDEX" arch=llama3_dense_tp tp_size=2,4 || fail "manifest-pre" 2

# Gate 1: identity + Triton probe (as for TP1).
step_time probe 600 "${RUN[@]}" $ENVDIR/tp1_probe.py ${EXPECT_UUID:+--expect-uuid "$EXPECT_UUID"} --out "$OUT/probe.json" \
  || fail "probe failed (identity or Triton on SM103); see probe.out/err" 3
grep -E "^identity|^TRITON PROBE" "$OUT/probe.out" | tee -a "$OUT/steps.log"

# Gate 2: numerical correctness at a rank's shapes, once per campaign.
# TP2: 16 / 4 heads, up_gate shard (n 14336, k 4096). TP4: 8 / 2 heads, down_proj shard (n 4096, k 3584).
check_shapes() { # check_shapes TAG HEADS KV_HEADS GEMM_N GEMM_K
  local tag="$1"
  [ -f "$STATE/checks-$tag.done" ] && { log "skip checks-$tag (done)"; return 0; }
  GB300_CHECK_HEADS="$2" GB300_CHECK_KV_HEADS="$3" GB300_CHECK_GEMM_N="$4" GB300_CHECK_GEMM_K="$5" \
    step_time "checks-$tag" 600 "${RUN[@]}" $ENVDIR/smoke_checks.py ${EXPECT_UUID:+--expect-uuid "$EXPECT_UUID"} --out "$OUT/checks-$tag.json" \
    || fail "correctness checks ($tag) did not run" 3
  grep -E "pass=|ERROR" "$OUT/checks-$tag.out" | cut -c1-260 | tee -a "$OUT/steps.log"
  if grep -qE "pass=False|ERROR" "$OUT/checks-$tag.out"; then fail "correctness mismatch at $tag shapes" 3; fi
  GB300_PROBE_KV_HEADS="$3" step_time "probe-shim-$tag" 300 env PYTHONPATH="$SHIM:$SIM_ROOT" "$SHIM/.venv/bin/python" -P \
    $ENVDIR/vllm-shim/probe_shim.py ${EXPECT_UUID:+--expect-uuid "$EXPECT_UUID"} --num-tokens 64 8192 --out "$OUT/probe-shim-$tag.json" \
    || fail "shim probe ($tag) failed; see probe-shim-$tag.out/err" 3
  grep -E "^num_tokens|^fp8" "$OUT/probe-shim-$tag.out" | cut -c1-120 | tee -a "$OUT/steps.log"
  cp "$OUT/checks-$tag.json" "$OUT/probe-shim-$tag.json" "$STATE/" && touch "$STATE/checks-$tag.done"
  isolation "after checks-$tag"
}
check_shapes tp2 16 4 14336 4096
check_shapes tp4 8 2 4096 3584

run_batch() { # run_batch KIND BACKEND CAP_S
  local kind="$1" backend="$2" cap="$3" name="fill-$1-$2" left to
  [ -f "$STATE/$name.done" ] && { log "skip $name (done)"; return 0; }
  left=$(( BUDGET_S - $(elapsed) ))
  [ "$left" -lt 120 ] && fail "wall budget exhausted before $name ($(elapsed) s used); rerun to resume" 5
  to=$(( cap < left ? cap : left ))
  step_time "$name" "$to" "${PROFILE[@]}" "$kind" --backend "$backend" --gpu-name "NVIDIA GB300" --json \
    --output-dir "$OUT/$name" --specs "$SPECS/$kind.json"
  local rc=$?
  tail -c 600 "$OUT/$name.err" | tee -a "$OUT/steps.log"
  step_time "validate-$kind-$backend" 300 "${RUN[@]}" $ENVDIR/tp1_validate.py batch --db "$VIBESIM_PROFILE_DB" \
    --kind "$kind" --backend "$backend" --specs "$SPECS/$kind.json" \
    --results "$OUT/$name/results.json" --stderr "$OUT/$name.err" --report "$OUT/validate-$kind-$backend.json"
  local vrc=$?
  cat "$OUT/validate-$kind-$backend.out" | tee -a "$OUT/steps.log"
  if [ $vrc -ne 0 ]; then fail "validation of $kind/$backend returned $vrc (run rc=$rc)" $vrc; fi
  [ $rc -ne 0 ] && fail "$name returned $rc although validation passed" "$rc"
  cp "$OUT/validate-$kind-$backend.json" "$STATE/" && touch "$STATE/$name.done"
  isolation "after $name"
}

# Short batches first, the two GEMM batches last.
run_batch kv_cache_append vllm_cuda 300
run_batch elementwise triton 400
run_batch flashinfer_attn_decode fa2 700
run_batch flashinfer_attn_prefill fa2 900
run_batch single_gemm torch_linear 1200
run_batch single_gemm torch 1200

# Repeatability: a stride sample per kind, re-measured with --fresh (no DB write).
mkdir -p "$OUT/repeat"
"${RUN[@]}" - "$SPECS" "$OUT/repeat" <<'PY'
import json, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
stride = {"single_gemm": 16, "elementwise": 4, "flashinfer_attn_prefill": 12,
          "flashinfer_attn_decode": 12, "kv_cache_append": 7}
for kind, n in stride.items():
    specs = json.loads((src / f"{kind}.json").read_text())["specs"]
    sample = specs[n // 2 :: n]
    (dst / f"{kind}.json").write_text(json.dumps({"specs": sample}) + "\n")
    print(kind, len(sample))
PY
for kb in "kv_cache_append vllm_cuda" "elementwise triton" "flashinfer_attn_decode fa2" "flashinfer_attn_prefill fa2" "single_gemm torch_linear"; do
  set -- $kb
  [ -f "$STATE/repeat-$1-$2.done" ] && { log "skip repeat-$1-$2 (done)"; continue; }
  left=$(( BUDGET_S + 240 - $(elapsed) ))
  [ "$left" -lt 60 ] && { log "budget: repeat-$1-$2 skipped"; continue; }
  step_time "repeat-$1-$2" "$(( left < 400 ? left : 400 ))" "${PROFILE[@]}" "$1" --backend "$2" --gpu-name "NVIDIA GB300" --json --fresh \
    --output-dir "$OUT/repeat-$1-$2" --specs "$OUT/repeat/$1.json"
  step_time "validate-repeat-$1-$2" 120 "${RUN[@]}" $ENVDIR/tp1_validate.py repeat --db "$VIBESIM_PROFILE_DB" \
    --kind "$1" --backend "$2" --results "$OUT/repeat-$1-$2/results.json" --report "$OUT/validate-repeat-$1-$2.json" \
    && cp "$OUT/validate-repeat-$1-$2.json" "$STATE/" && touch "$STATE/repeat-$1-$2.done"
  cat "$OUT/validate-repeat-$1-$2.out" | tee -a "$OUT/steps.log"
done
isolation "after repeats"

kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null; trap - EXIT
step_time manifest-post 120 "${RUN[@]}" $ENVDIR/collect_manifest.py --out "$OUT/manifest-post.json" \
  --run-config phase=tp24-single-gpu-fill "gpu=tray12:$GPU_INDEX" arch=llama3_dense_tp tp_size=2,4
sha256sum profiling/profile.db "$VIBESIM_PROFILE_DB" >"$OUT/db-sha256-post.txt"
du -sb "$GB300_ROOT/cache" | cut -f1 >"$OUT/cache-bytes-post.txt"
log "cache delta bytes: $(( $(cat "$OUT/cache-bytes-post.txt") - $(cat "$OUT/cache-bytes-pre.txt") ))"
if [ "$(head -1 "$OUT/db-sha256-pre.txt")" = "$(head -1 "$OUT/db-sha256-post.txt")" ]; then log "tracked profiling/profile.db unchanged"; else log "WARNING: tracked profile.db changed"; fi
log "wall used: $(elapsed) s of $BUDGET_S"
log "tp24 fill done"
