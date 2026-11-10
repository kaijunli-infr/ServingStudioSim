#!/usr/bin/env bash
# GB300 Llama 3.1 8B BF16 TP1 fill, first batch (PROPOSALS.md §C), one GPU.
#
#   run_tp1_fill.sh [GPU_INDEX] [EXPECTED_UUID]
#
# Fills the private profile DB ($VIBESIM_PROFILE_DB) kind by kind, validating
# after each batch and stopping on a FAIL/ANOMALY verdict. Re-running resumes:
# a fill without --fresh/--force measures only specs the DB lacks.
set -u
SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$SIM_ROOT"
# shellcheck disable=SC1091
source profiling/env/arm64-gb300/gb300.env
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

GPU_INDEX="${1:-1}"
EXPECT_UUID="${2:-}"
STAMP="$(date +%Y%m%d-%H%M%S)"
TP1="$GB300_ROOT/artifacts/tp1"
OUT="$TP1/run-$STAMP"
mkdir -p "$OUT"
ln -sfn "$OUT" "$TP1/run-latest"
ENVDIR=profiling/env/arm64-gb300

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONUNBUFFERED=1
RUN=(uv run --no-sync --project "$ENVDIR" python)
PROFILE=(uv run --no-sync --project "$ENVDIR" python -m profiling.cli run)

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/steps.log"; }
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

log "tp1 fill start out=$OUT gpu_index=$GPU_INDEX expect_uuid=${EXPECT_UUID:-none} db=$VIBESIM_PROFILE_DB"
sha256sum profiling/profile.db "$VIBESIM_PROFILE_DB" >"$OUT/db-sha256-pre.txt"
du -sb "$GB300_ROOT/cache" | cut -f1 >"$OUT/cache-bytes-pre.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv >"$OUT/compute-apps-pre.csv"
[ "$(wc -l <"$OUT/compute-apps-pre.csv")" -gt 1 ] && fail "compute processes present before start" 2

nvidia-smi -i "$GPU_INDEX" --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active \
  --format=csv -lms 500 >"$OUT/gpu-samples.csv" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

step_time manifest-pre 120 "${RUN[@]}" $ENVDIR/collect_manifest.py --out "$OUT/manifest-pre.json" \
  --run-config phase=2C-tp1-fill "gpu=tray12:$GPU_INDEX" preset=predict_llama3_8b_gb300_iter.json || fail "manifest-pre" 2

# Identity gate + Triton probe + Kineto cross-check (one process).
step_time probe 600 "${RUN[@]}" $ENVDIR/tp1_probe.py ${EXPECT_UUID:+--expect-uuid "$EXPECT_UUID"} --out "$OUT/probe.json" \
  || fail "probe failed (identity or Triton on SM103); see probe.out/err" 3
grep -E "^identity|^TRITON PROBE" "$OUT/probe.out" | tee -a "$OUT/steps.log"

# Spec lists from the simulator's kernel-config export (CPU only, already exported).
step_time specs 120 "${RUN[@]}" $ENVDIR/tp1_specs.py "$TP1/cost_trees.json" "$OUT/specs" || fail "spec export" 2
cat "$OUT/specs.out" | tee -a "$OUT/steps.log"

run_batch() { # run_batch KIND BACKEND TIMEOUT_S
  local kind="$1" backend="$2" to="$3" name="fill-$1-$2"
  step_time "$name" "$to" "${PROFILE[@]}" "$kind" --backend "$backend" --gpu-name "NVIDIA GB300" --json \
    --output-dir "$OUT/$name" --specs "$OUT/specs/$kind.json"
  local rc=$?
  tail -c 600 "$OUT/$name.err" | tee -a "$OUT/steps.log"
  step_time "validate-$kind-$backend" 300 "${RUN[@]}" $ENVDIR/tp1_validate.py batch --db "$VIBESIM_PROFILE_DB" \
    --kind "$kind" --backend "$backend" --specs "$OUT/specs/$kind.json" \
    --results "$OUT/$name/results.json" --stderr "$OUT/$name.err" --report "$OUT/validate-$kind-$backend.json"
  local vrc=$?
  cat "$OUT/validate-$kind-$backend.out" | tee -a "$OUT/steps.log"
  if [ $vrc -ne 0 ]; then fail "validation of $kind/$backend returned $vrc (run rc=$rc)" $vrc; fi
}

# Batches in the approved order; each is restartable.
run_batch single_gemm torch_linear 1500
run_batch single_gemm torch 1500
run_batch rms_norm flashinfer 400
run_batch elementwise triton 600
run_batch flashinfer_attn_prefill fa2 900
run_batch flashinfer_attn_decode fa2 900

# Repeatability: every Nth spec per kind, re-measured with --fresh (no DB write).
mkdir -p "$OUT/repeat"
"${RUN[@]}" - "$OUT/specs" "$OUT/repeat" <<'PY'
import json, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
stride = {"single_gemm": 15, "rms_norm": 12, "elementwise": 12, "flashinfer_attn_prefill": 12, "flashinfer_attn_decode": 12}
for kind, n in stride.items():
    specs = json.loads((src / f"{kind}.json").read_text())["specs"]
    sample = specs[n // 2 :: n]
    (dst / f"{kind}.json").write_text(json.dumps({"specs": sample}) + "\n")
    print(kind, len(sample))
PY
for kb in "single_gemm torch_linear" "rms_norm flashinfer" "elementwise triton" "flashinfer_attn_prefill fa2" "flashinfer_attn_decode fa2"; do
  set -- $kb
  step_time "repeat-$1-$2" 600 "${PROFILE[@]}" "$1" --backend "$2" --gpu-name "NVIDIA GB300" --json --fresh \
    --output-dir "$OUT/repeat-$1-$2" --specs "$OUT/repeat/$1.json"
  step_time "validate-repeat-$1-$2" 120 "${RUN[@]}" $ENVDIR/tp1_validate.py repeat --db "$VIBESIM_PROFILE_DB" \
    --kind "$1" --backend "$2" --results "$OUT/repeat-$1-$2/results.json" --report "$OUT/validate-repeat-$1-$2.json"
  cat "$OUT/validate-repeat-$1-$2.out" | tee -a "$OUT/steps.log"
done

kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null; trap - EXIT
step_time manifest-post 120 "${RUN[@]}" $ENVDIR/collect_manifest.py --out "$OUT/manifest-post.json" \
  --run-config phase=2C-tp1-fill "gpu=tray12:$GPU_INDEX" preset=predict_llama3_8b_gb300_iter.json
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv >"$OUT/compute-apps-post.csv"
sha256sum profiling/profile.db "$VIBESIM_PROFILE_DB" >"$OUT/db-sha256-post.txt"
du -sb "$GB300_ROOT/cache" | cut -f1 >"$OUT/cache-bytes-post.txt"
du -sb "$GB300_ROOT"/cache/* >"$OUT/cache-du-post.txt"
log "cache delta bytes: $(( $(cat "$OUT/cache-bytes-post.txt") - $(cat "$OUT/cache-bytes-pre.txt") ))"
if [ "$(head -1 "$OUT/db-sha256-pre.txt")" = "$(head -1 "$OUT/db-sha256-post.txt")" ]; then log "tracked profiling/profile.db unchanged"; else log "WARNING: tracked profile.db changed"; fi
log "compute apps after run: $(( $(wc -l <"$OUT/compute-apps-post.csv") - 1 ))"
log "tp1 fill done"
