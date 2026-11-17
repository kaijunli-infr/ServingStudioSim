#!/usr/bin/env bash
# Pin the V1 scripts, checks, tests and fixtures: writes SCRIPTS_v1.sha256 (verified by preflight_e1_v1.sh).
set -euo pipefail
E1=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$E1/../../../../../.." && pwd)
cd $E1
FILES="make_e1_inputs_v1.py preflight_e1_v1.sh run_e1_v1.sh v1_control_probe.sh postrun_e1_v1.sh monitor_e1_v1.sh closeout_e1_v1.sh hash_scripts_v1.sh pin_v1.sh \
 preflight_e1_stage15.sh preflight_e1_host.sh \
 checks/e1_lifecycle_gates.py checks/e1_pids.py checks/e1_process_census.sh checks/e1_nsys_validation.py checks/e1_completeness_v2.py checks/e1_config_identity.py"
{ for f in $FILES; do sha256sum $f; done
  ( cd $REPO && sha256sum tests/test_profiler_lifecycle.py tests/test_profiler_preflight.py tests/test_e1_lifecycle_gates.py tests/test_e1_pids_cleanup.py \
      tests/fixtures/e1_main_v3/profile.json tests/fixtures/e1_main_v3/launch.json tests/fixtures/e1_nsys_val/profile.json tests/fixtures/e1_nsys_val/launch.json tests/fixtures/e1_nsys_val/server_log_excerpt.log \
      | sed "s#  tests/#  ../../../../../../tests/#" ); } > SCRIPTS_v1.sha256
sha256sum -c SCRIPTS_v1.sha256 --quiet && echo "SCRIPTS_v1.sha256: $(wc -l < SCRIPTS_v1.sha256) files, sha256 $(sha256sum SCRIPTS_v1.sha256 | cut -c1-16)"
