#!/usr/bin/env bash
# Pin the V2 forced-release scripts, checks, tests and fixtures: writes SCRIPTS_fr.sha256 (verified by preflight_e1_fr.sh).
set -euo pipefail
E1=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$E1/../../../../../.." && pwd)
cd $E1
FILES="make_e1_inputs_fr.py preflight_e1_fr.sh run_e1_fr.sh postrun_e1_fr.sh monitor_e1_fr.sh closeout_e1_fr.sh hash_scripts_fr.sh pin_fr.sh rehearse_e1_fr.sh e1_fr_rehearsal.py \
 preflight_e1_stage15.sh preflight_e1_host.sh \
 checks/e1_forced_release_gates.py checks/e1_lifecycle_gates_v2.py checks/e1_lifecycle_gates.py checks/e1_host_identity.py checks/e1_launch_identity_v2.py checks/e1_identity_policy_v1.json \
 checks/e1_release_probe.py checks/e1_frontend_binary_gate.py checks/e1_gpu_watch.py checks/e1_gpu_release_census.py checks/e1_pids.py checks/e1_process_census.sh checks/e1_nsys_validation.py checks/e1_completeness_v2.py checks/e1_config_identity.py"
{ for f in $FILES; do sha256sum $f; done
  ( cd $REPO && sha256sum tests/cpu_only_guard.py tests/cpu_only.sh tests/test_profiler_lifecycle.py tests/test_profiler_preflight.py tests/test_e1_lifecycle_gates.py tests/test_e1_pids_cleanup.py tests/test_cpu_only_guard.py tests/test_e1_lifecycle_gates_v2.py tests/test_e1_launch_identity_v2.py tests/test_profiler_lifecycle_stop_concurrency.py tests/test_profiler_lifecycle_host_stream.py tests/test_profiler_preflight_retained.py tests/test_e1_host_identity.py tests/test_e1_forced_release_gates.py tests/test_load_generator_prebuilt.py tests/test_e1_release_probe.py tests/test_profiler_lifecycle_release_evidence.py tests/test_e1_frontend_binary_gate.py tests/test_e1_fr_rehearsal.py \
      tests/fixtures/e1_main_v3/profile.json tests/fixtures/e1_main_v3/launch.json tests/fixtures/e1_nsys_val/profile.json tests/fixtures/e1_nsys_val/launch.json tests/fixtures/e1_nsys_val/server_log_excerpt.log \
      | sed "s#  tests/#  ../../../../../../tests/#" ); } > SCRIPTS_fr.sha256
sha256sum -c SCRIPTS_fr.sha256 --quiet && echo "SCRIPTS_fr.sha256: $(wc -l < SCRIPTS_fr.sha256) files, sha256 $(sha256sum SCRIPTS_fr.sha256 | cut -c1-16)"
