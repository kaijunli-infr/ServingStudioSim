#!/usr/bin/env bash
# Pin the V2 forced-release source: code tree ids of the launcher tree (stable across documentation-only commits), submodule gitlinks, the
# session_runner binary, and the identities the preflight module observes on this tray (Nsight CLI + QdstrmImporter sha256/version,
# server interpreter, vLLM fork head/version, expanded recipe). Read-only on the system; writes V2-FR-PIN.json and V2-FR-PIN-identities.json
# next to itself. Run ON the capture tray as dc (identities are per-host installs):
#   E1_REPO=$S/wt-lifecycle bash $E1/pin_fr.sh
set -euo pipefail
export HOME=/mnt/shared/home/dc; S=$HOME/workspace/serving-studio
E1=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=${E1_REPO:-$(cd "$E1/../../../../../.." && pwd)}
PASS=e1_v2_forced_release; CFG=$E1/configs_fr/$PASS/profile.json
AR15=$S/artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115/run/e1_nsys_val
cd $REPO && source $S/scripts/env.sh >/dev/null 2>&1 && source profiling/env/arm64-gb300/gb300.env >/dev/null 2>&1; unset CUDA_VISIBLE_DEVICES
uv run --no-sync python -m alignment.profiler.preflight --config $CFG --reference-launch $AR15/profile/vllm/dsv41_flash_1017_gb300_tp4_e1_nsys_val_launch.json \
  --reference-server-log $AR15/profile/vllm/dsv41_flash_1017_gb300_tp4_e1_nsys_val_server.log --write-pins $E1/V2-FR-PIN-identities.json --json $E1/V2-FR-PIN-identities.preflight.json | tail -3
python3 -I - "$REPO" "$E1/V2-FR-PIN.json" "$(hostname)" <<'PY'
import json, subprocess, sys, hashlib, datetime
repo, out, host = sys.argv[1:4]
g = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, check=True).stdout.strip()
keys = ['alignment', 'launcher', 'tests/fixtures', 'tests/cpu_only_guard.py', 'tests/cpu_only.sh', 'tests/test_profiler_lifecycle.py', 'tests/test_profiler_preflight.py', 'tests/test_e1_lifecycle_gates.py', 'tests/test_e1_pids_cleanup.py', 'tests/test_cpu_only_guard.py', 'tests/test_e1_lifecycle_gates_v2.py', 'tests/test_e1_launch_identity_v2.py', 'tests/test_profiler_lifecycle_stop_concurrency.py', 'tests/test_profiler_lifecycle_host_stream.py', 'tests/test_profiler_preflight_retained.py', 'tests/test_e1_host_identity.py', 'tests/test_e1_forced_release_gates.py', 'tests/test_load_generator_prebuilt.py', 'tests/test_e1_release_probe.py', 'tests/test_profiler_lifecycle_release_evidence.py', 'tests/test_e1_frontend_binary_gate.py', 'tests/test_e1_fr_rehearsal.py']
pin = {
  "pass": "e1_v2_forced_release",
  "pinned_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "host": host,
  "code": {
    "commit": g("rev-parse", "HEAD"), "branch": g("branch", "--show-current"),
    "tree": {k: g("rev-parse", f"HEAD:{k}") for k in keys},
    "gitlinks": {p: g("ls-tree", "HEAD", p).split()[2] for p in ("alignment/load_generator/req-frontend", "alignment/profiler/vllm")},
    "note": "tree ids are what preflight_e1_fr.sh compares (a later documentation-only commit keeps them); the launching commit must be a descendant of `commit` with these trees",
  },
  "session_runner_sha256": hashlib.sha256(open(f"{repo}/alignment/load_generator/req-frontend/target/release/session_runner", "rb").read()).hexdigest(),
  "identities_file": "V2-FR-PIN-identities.json",
}
json.dump(pin, open(out, "w"), indent=1); open(out, "a").write("\n")
print(json.dumps(pin["code"], indent=1))
PY
