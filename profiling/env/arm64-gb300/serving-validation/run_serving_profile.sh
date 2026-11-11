#!/usr/bin/env bash
# One guarded alignment profile of real vLLM serving on tray12 (needs GPUs).
#
#   run_serving_profile.sh RUN        RUN = tp1-smoke | tp2-default | tp2-nccl
#
# Runs `python -m launcher alignment profile $ROOT/configs/RUN.yaml` (rendered by
# make_configs.py) from the serving worktree, under an outer wall-clock limit
# (LIMIT_S, default 900 s), with these guards:
#
# before: the run's GPUs have the expected UUIDs, utilisation <= 5 %, memory
#   used < 1 GiB and no compute process; every compute process on the tray is
#   recorded; the serving venv's package fingerprint equals the committed one;
#   the profiling env guard passes unchanged; the pinned model snapshot is
#   complete; at least 60 GB is free; the server port is free; both profile
#   databases are checksummed.
# during: every 10 s, a compute process on one of the run's GPUs that is not
#   in this run's process session stops the run.
# after: the same environment, fingerprint and database checks; leftover
#   processes of this run's own session are killed, nothing else.
#
# It never resets a GPU and never signals a process outside its own session.
set -u
RUN="${1:?RUN}"
LIMIT_S="${LIMIT_S:-900}"
S=/mnt/shared/home/kevin/workspace/serving-studio
G=$S/gb300
ROOT=$G/serving-validation
W=$S/ServingStudioSim-serving
CFG=$ROOT/configs/$RUN.yaml
SIM_PY=$S/ServingStudioSim/.venv/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP=$(date -u +%Y%m%d-%H%M%S)
OUT=$ROOT/guard/$RUN-$STAMP
mkdir -p "$OUT"
SID="$(ps -o sid= -p $$ | tr -d ' ')"
declare -A UUID=([0]=GPU-0294efb9-8850-f68b-5cdd-9d4e92e2cb9c [1]=GPU-9574ced0-46be-d5b7-0098-01fa1e84dccb
                 [2]=GPU-d2cef419-102a-b229-7832-70f157a02dc1 [3]=GPU-098db872-19a4-a98b-2793-9ea752f88553)

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/steps.log"; }
cleanup() {
  local pids
  kill "${WATCH:-0}" 2>/dev/null
  pids="$(ps -s "$SID" -o pid= | grep -vw "$$" | tr '\n' ' ')"
  # shellcheck disable=SC2086
  [ -n "${pids// /}" ] && kill -INT $pids 2>/dev/null && sleep 20
  pids="$(ps -s "$SID" -o pid= | grep -vw "$$" | tr '\n' ' ')"
  # shellcheck disable=SC2086
  [ -n "${pids// /}" ] && kill -KILL $pids 2>/dev/null
  return 0
}
fail() { log "STOP: $1"; nvidia-smi >"$OUT/nvidia-smi-at-stop.txt" 2>&1; cleanup; exit "${2:-1}"; }

[ -f "$CFG" ] || fail "no config $CFG" 2
DEVICES="$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['cuda_visible_devices'])" "$CFG")"
log "run $RUN devices=$DEVICES limit_s=$LIMIT_S sid=$SID out=$OUT"

device_state() { # device_state LABEL
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu,fabric.state \
    --format=csv >"$OUT/gpus-$1.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader >"$OUT/apps-$1.csv"
  local idx u util mem
  for idx in ${DEVICES//,/ }; do
    u="$(nvidia-smi -i "$idx" --query-gpu=uuid --format=csv,noheader)"
    [ "$u" = "${UUID[$idx]}" ] || fail "GPU $idx is $u, expected ${UUID[$idx]}" 4
    grep -q "$u" "$OUT/apps-$1.csv" && fail "compute process on GPU $idx at $1: $(grep "$u" "$OUT/apps-$1.csv")" 4
    util="$(nvidia-smi -i "$idx" --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
    mem="$(nvidia-smi -i "$idx" --query-gpu=memory.used --format=csv,noheader,nounits)"
    [ "$util" -le 5 ] || fail "GPU $idx utilisation $util % at $1" 4
    [ "$mem" -lt 1024 ] || fail "GPU $idx has $mem MiB in use at $1" 4
  done
  [ -s "$OUT/apps-$1.csv" ] && log "other compute processes on the tray at $1: $(tr '\n' ';' <"$OUT/apps-$1.csv")"
  return 0
}

fingerprints() { # fingerprints LABEL
  (cd /tmp && CUDA_VISIBLE_DEVICES= "$ROOT/venv/bin/python" -I - "$HERE/env-fingerprint.json" <<'PY') >"$OUT/serving-env-$1.txt" 2>&1 \
    || fail "serving environment differs from the committed fingerprint at $1" 2
import hashlib, importlib.metadata as md, json, sys
want = json.load(open(sys.argv[1]))["packages_sha256"]
pk = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions())
got = hashlib.sha256("\n".join(pk).encode()).hexdigest()
print(got)
sys.exit(0 if got == want else 1)
PY
  (cd $S/ServingStudioSim-gb300 && source profiling/env/arm64-gb300/gb300.env \
     && CUDA_VISIBLE_DEVICES= PATH="$HOME/.local/bin:$PATH" uv run --no-sync --project profiling/env/arm64-gb300 python \
        profiling/env/arm64-gb300/env_guard.py --out "$OUT/profiling-env-$1.json" \
        --same-as $G/artifacts/phase2f/tp2-latest/env-guard-pre.json >/dev/null) \
    || fail "profiling environment guard failed at $1" 2
  sha256sum $S/ServingStudioSim/profiling/profile.db $G/db/profile-gb300.db >"$OUT/db-sha256-$1.txt"
}

# Model snapshot: every weight file present with the size the hub reports.
MODEL="$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['server']['model_path'])" "$CFG")"
python3 -I - "$MODEL" "$HERE/hf-access.json" <<'PY' || fail "model snapshot incomplete: $MODEL" 2
import json, os, sys
model, ref = sys.argv[1], json.load(open(sys.argv[2]))
bad = [f for f, size in ref["weight_files"].items() if "/" not in f
       and (not os.path.isfile(os.path.join(model, f)) or os.path.getsize(os.path.join(model, f)) != size)]
bad += [f for f in ("config.json", "tokenizer.json") if not os.path.isfile(os.path.join(model, f))]
print("missing or wrong size:", bad)
sys.exit(1 if bad else 0)
PY
FREE_GB=$(( $(df --output=avail -B1G $ROOT | tail -1) ))
[ "$FREE_GB" -ge 60 ] || fail "only $FREE_GB GB free" 2

PORT="$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['server'].get('port', 8000))" "$CFG")"
ss -ltn | awk 'NR>1{print $4}' | grep -q ":$PORT\$" && fail "port $PORT already in use on this tray" 2

fingerprints pre
device_state pre

watch_gpus() {
  local idx u pid
  while sleep 10; do
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader >"$OUT/apps-watch.csv" 2>/dev/null
    for idx in ${DEVICES//,/ }; do
      u="${UUID[$idx]}"
      for pid in $(grep "$u" "$OUT/apps-watch.csv" | cut -d, -f2 | tr -d ' '); do
        [ "$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$SID" ] && continue
        echo "$(date -u +%FT%TZ) foreign pid $pid on GPU $idx" >>"$OUT/steps.log"
        kill -USR1 $$
        return
      done
    done
  done
}
trap 'fail "foreign compute process appeared on a run GPU" 4' USR1
watch_gpus &
WATCH=$!

log "launch"
(cd "$W" && CUDA_VISIBLE_DEVICES= timeout --signal=INT --kill-after=120 "$LIMIT_S" \
   "$SIM_PY" -m launcher alignment profile "$CFG") >"$OUT/launcher.out" 2>"$OUT/launcher.err"
rc=$?
log "launcher rc=$rc"
kill "$WATCH" 2>/dev/null; trap - USR1
cleanup
fingerprints post
cmp -s "$OUT/db-sha256-pre.txt" "$OUT/db-sha256-post.txt" || fail "a profile database changed" 7
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader >"$OUT/apps-post.csv"
[ $rc -eq 124 ] && fail "outer limit of $LIMIT_S s reached" 5
[ $rc -ne 0 ] && fail "launcher returned $rc; see launcher.err" 3
log "done; artifacts in $(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['log_dir'])" "$CFG")"
