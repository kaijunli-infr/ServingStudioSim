#!/usr/bin/env bash
# GB300 Llama 3.1 8B BF16 TP1: the 70 kv_cache_append/vllm_cuda rows through
# the vllm shim (Phase 2C, workstream B), one GPU. Same shape as
# run_tp1_fill.sh: gates, one batch, validation, a repeat sample, manifests.
#
#   run_kv_fill.sh [GPU_INDEX] [EXPECTED_UUID]
#
# Gates before the fill: no compute process on the tray; the shim's two-spec
# probe (bitwise correctness vs the Torch reference, kernel identity, CUPTI
# capture, explicit fp8 refusal) on the approved GPU; then a --fresh run of
# the same two specs through the profiler's own worker (vllm_env) so the
# measurement path the fill uses is the one that was probed.
set -u
SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$SIM_ROOT"
# shellcheck disable=SC1091
source profiling/env/arm64-gb300/gb300.env
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# The CUPTI timer builds its extension per interpreter path with ninja, which lives
# in the venv's bin: both the probe and the vllm_env workers run the shim's
# `.venv/bin/python`, a path the earlier TP1 fill never used.
export PATH="$GB300_ROOT/venv/bin:$PATH"

GPU_INDEX="${1:-1}"
EXPECT_UUID="${2:-}"
STAMP="$(date +%Y%m%d-%H%M%S)"
TP1="$GB300_ROOT/artifacts/tp1"
OUT="$TP1/run-$STAMP-kv"
mkdir -p "$OUT"
ln -sfn "$OUT" "$TP1/run-latest-kv"
ENVDIR=profiling/env/arm64-gb300
SHIM="$VIBESIM_VLLM_FORK_ROOT"
KIND=kv_cache_append
BACKEND=vllm_cuda

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

log "kv fill start out=$OUT gpu_index=$GPU_INDEX expect_uuid=${EXPECT_UUID:-none} db=$VIBESIM_PROFILE_DB shim=$SHIM"
[ -f "$SHIM/manifest.json" ] || fail "no shim manifest at $SHIM (run vllm-shim/install_shim.sh)" 2
cp "$SHIM/manifest.json" "$OUT/shim-manifest.json"
sha256sum profiling/profile.db "$VIBESIM_PROFILE_DB" >"$OUT/db-sha256-pre.txt"
du -sb "$GB300_ROOT/cache" | cut -f1 >"$OUT/cache-bytes-pre.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv >"$OUT/compute-apps-pre.csv"
[ "$(wc -l <"$OUT/compute-apps-pre.csv")" -gt 1 ] && fail "compute processes present before start" 2
nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid,memory.used,utilization.gpu --format=csv >"$OUT/gpu-pre.csv"

nvidia-smi -i "$GPU_INDEX" --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active \
  --format=csv -lms 500 >"$OUT/gpu-samples.csv" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

step_time manifest-pre 120 "${RUN[@]}" $ENVDIR/collect_manifest.py --out "$OUT/manifest-pre.json" \
  --run-config phase=2C-kv-fill "gpu=tray12:$GPU_INDEX" preset=predict_llama3_8b_gb300_iter.json || fail "manifest-pre" 2

# Gate 1: the shim probe on the shim's interpreter (what vllm_env workers run).
# -P: do not put the script's directory first on sys.path, or the repository's copy
# of the shim package (no manifest, no library) would shadow the installed one.
step_time probe-shim 900 env PYTHONPATH="$SHIM:$SIM_ROOT" "$SHIM/.venv/bin/python" -P $ENVDIR/vllm-shim/probe_shim.py \
  --out "$OUT/probe-shim.json" ${EXPECT_UUID:+--expect-uuid "$EXPECT_UUID"} --num-tokens 64 8192 \
  || fail "shim probe failed; see probe-shim.out/err" 3
cat "$OUT/probe-shim.out" | tee -a "$OUT/steps.log"

# Spec list from the simulator's kernel-config export (CPU only, already exported).
step_time specs 120 "${RUN[@]}" $ENVDIR/tp1_specs.py "$TP1/cost_trees.json" "$OUT/specs" || fail "spec export" 2
grep "^$KIND" "$OUT/specs.out" | tee -a "$OUT/steps.log"
"${RUN[@]}" - "$OUT/specs/$KIND.json" "$OUT/specs/$KIND-probe.json" <<'PY'
import json, sys
specs = json.loads(open(sys.argv[1]).read())["specs"]
two = [s for s in specs if s["num_tokens"] in (64, 8192)]
assert len(two) == 2, [s["num_tokens"] for s in specs]
open(sys.argv[2], "w").write(json.dumps({"specs": two}) + "\n")
PY

# Gate 2: the same two specs through the profiler's worker path, no DB write.
step_time probe-worker 900 "${PROFILE[@]}" "$KIND" --backend "$BACKEND" --gpu-name "NVIDIA GB300" --json --fresh \
  --output-dir "$OUT/probe-worker" --specs "$OUT/specs/$KIND-probe.json" || fail "worker probe failed; see probe-worker.err" 3
tail -c 400 "$OUT/probe-worker.err" | tee -a "$OUT/steps.log"
"${RUN[@]}" - "$OUT/probe-worker/results.json" "$OUT/probe-shim.json" <<'PY' | tee -a "$OUT/steps.log"
import json, sys
res = json.load(open(sys.argv[1])); probe = json.load(open(sys.argv[2]))
by_tokens = {}
for spec, r in zip(res.get("specs", []), res.get("results", [])):
    by_tokens[spec.get("num_tokens")] = (r.get("status"), (r.get("metrics") or {}).get("time_ms"))
bad = 0
for s in probe["specs"]:
    st, t = by_tokens.get(s["num_tokens"], (None, None))
    ratio = (t * 1e3 / s["cupti"]["mean_us"]) if t else None
    ok = st == "ok" and ratio is not None and 0.8 <= ratio <= 1.25
    bad += not ok
    print(f"worker vs probe num_tokens={s['num_tokens']}: status={st} worker_us={t and round(t*1e3,2)} "
          f"probe_us={round(s['cupti']['mean_us'],2)} ratio={ratio and round(ratio,3)} {'OK' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "worker measurement disagrees with the probe" 3

# The fill: one batch, inserted only when the whole batch finishes.
step_time "fill-$KIND-$BACKEND" 1800 "${PROFILE[@]}" "$KIND" --backend "$BACKEND" --gpu-name "NVIDIA GB300" --json \
  --output-dir "$OUT/fill-$KIND-$BACKEND" --specs "$OUT/specs/$KIND.json"
rc=$?
tail -c 600 "$OUT/fill-$KIND-$BACKEND.err" | tee -a "$OUT/steps.log"
step_time "validate-$KIND-$BACKEND" 300 "${RUN[@]}" $ENVDIR/tp1_validate.py batch --db "$VIBESIM_PROFILE_DB" \
  --kind "$KIND" --backend "$BACKEND" --specs "$OUT/specs/$KIND.json" \
  --results "$OUT/fill-$KIND-$BACKEND/results.json" --stderr "$OUT/fill-$KIND-$BACKEND.err" --report "$OUT/validate-$KIND-$BACKEND.json"
vrc=$?
cat "$OUT/validate-$KIND-$BACKEND.out" | tee -a "$OUT/steps.log"
[ $vrc -ne 0 ] && fail "validation returned $vrc (run rc=$rc)" $vrc

# Repeatability: every 7th spec, re-measured with --fresh (no DB write).
mkdir -p "$OUT/repeat"
"${RUN[@]}" - "$OUT/specs/$KIND.json" "$OUT/repeat/$KIND.json" <<'PY'
import json, sys
specs = json.loads(open(sys.argv[1]).read())["specs"]
sample = specs[3::7]
open(sys.argv[2], "w").write(json.dumps({"specs": sample}) + "\n")
print("repeat sample", len(sample))
PY
step_time "repeat-$KIND-$BACKEND" 600 "${PROFILE[@]}" "$KIND" --backend "$BACKEND" --gpu-name "NVIDIA GB300" --json --fresh \
  --output-dir "$OUT/repeat-$KIND-$BACKEND" --specs "$OUT/repeat/$KIND.json"
step_time "validate-repeat-$KIND-$BACKEND" 120 "${RUN[@]}" $ENVDIR/tp1_validate.py repeat --db "$VIBESIM_PROFILE_DB" \
  --kind "$KIND" --backend "$BACKEND" --results "$OUT/repeat-$KIND-$BACKEND/results.json" --report "$OUT/validate-repeat-$KIND-$BACKEND.json"
cat "$OUT/validate-repeat-$KIND-$BACKEND.out" | tee -a "$OUT/steps.log"

kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null; trap - EXIT
step_time manifest-post 120 "${RUN[@]}" $ENVDIR/collect_manifest.py --out "$OUT/manifest-post.json" \
  --run-config phase=2C-kv-fill "gpu=tray12:$GPU_INDEX" preset=predict_llama3_8b_gb300_iter.json
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv >"$OUT/compute-apps-post.csv"
sha256sum profiling/profile.db "$VIBESIM_PROFILE_DB" >"$OUT/db-sha256-post.txt"
du -sb "$GB300_ROOT/cache" | cut -f1 >"$OUT/cache-bytes-post.txt"
log "cache delta bytes: $(( $(cat "$OUT/cache-bytes-post.txt") - $(cat "$OUT/cache-bytes-pre.txt") ))"
if [ "$(head -1 "$OUT/db-sha256-pre.txt")" = "$(head -1 "$OUT/db-sha256-post.txt")" ]; then log "tracked profiling/profile.db unchanged"; else log "WARNING: tracked profile.db changed"; fi
log "compute apps after run: $(( $(wc -l <"$OUT/compute-apps-post.csv") - 1 ))"
log "kv fill done"
