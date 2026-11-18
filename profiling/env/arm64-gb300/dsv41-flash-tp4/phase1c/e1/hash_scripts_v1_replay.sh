#!/usr/bin/env bash
# Manifest of the remediation validators, tests and replay script: SCRIPTS_v1_replay.sha256. The pinned SCRIPTS_v1.sha256 is not touched.
set -euo pipefail
E1=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$E1/../../../../../.." && pwd)
cd $E1
{ sha256sum replay_e1_v1.sh hash_scripts_v1_replay.sh checks/e1_lifecycle_gates_v2.py checks/e1_launch_identity_v2.py checks/e1_identity_policy_v1.json checks/e1_gpu_watch.py
  ( cd $REPO && sha256sum tests/test_e1_lifecycle_gates_v2.py tests/test_e1_launch_identity_v2.py tests/test_profiler_lifecycle_stop_concurrency.py | sed "s#  tests/#  ../../../../../../tests/#" ); } > SCRIPTS_v1_replay.sha256
sha256sum -c SCRIPTS_v1_replay.sha256 --quiet && echo "SCRIPTS_v1_replay.sha256: $(wc -l < SCRIPTS_v1_replay.sha256) files"
sha256sum -c SCRIPTS_v1.sha256 --quiet && echo "pinned SCRIPTS_v1.sha256 still verifies ($(wc -l < SCRIPTS_v1.sha256) files unchanged)"
