#!/usr/bin/env bash
# Pin the V4 scripts, checks, tests and fixtures: writes SCRIPTS_v4scale.sha256 (verified by preflight_e1_v4scale.sh). With `freeze`
# as the first argument it also writes FREEZE_v4scale.sha256 (PROPOSED until a review approves the freeze) over the packet, the manifests, the inputs, the pins and the binary
# provenance (run after pin_v4scale.sh and make_e1_inputs_v4scale.py).
set -euo pipefail
E1=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$E1/../../../../../.." && pwd)
cd $E1
FILES="make_e1_inputs_v4scale.py preflight_e1_v4scale.sh run_e1_v4scale.sh postrun_e1_v4scale.sh monitor_e1_v4scale.sh closeout_e1_v4scale.sh archive_e1_v4scale.sh hash_scripts_v4scale.sh pin_v4scale.sh rehearse_e1_v4scale.sh e1_v4scale_rehearsal.py e1_v4scale_eightwave.py e1_v4scale_identity_rehearsal.py rehearse_e2e_v4scale.sh e1_v4scale_mock_launcher.py e1_v4scale_e2e_check.py \
 preflight_e1_stage15.sh preflight_e1_host.sh \
 checks/e1_v4scale_gates.py checks/e1_v3scale_gates.py checks/e1_launch_identity_v4scale.py checks/e1_identity_policy_v4scale.json checks/e1_supervisor_v4.py checks/e1_reservation_guard.py checks/e1_reservation_release.py checks/e1_archive_size.py checks/e1_supervisor.py checks/e1_import_probe.py checks/e1_mem_watch.py checks/e1_storage_watch.py checks/e1_forced_release_gates.py checks/e1_lifecycle_gates_v2.py checks/e1_lifecycle_gates.py checks/e1_host_identity.py checks/e1_launch_identity_v2.py checks/e1_identity_policy_v1.json \
 checks/e1_frontend_binary_gate.py checks/e1_gpu_watch_v3.py checks/e1_gpu_watch.py checks/e1_gpu_release_census.py checks/e1_pids.py checks/e1_process_census.sh checks/e1_nsys_validation.py checks/e1_completeness_v2.py checks/e1_config_identity.py"
{ for f in $FILES; do sha256sum $f; done
  ( cd $REPO && sha256sum tests/cpu_only_guard.py tests/cpu_only.sh tests/test_profiler_lifecycle.py tests/test_profiler_preflight.py tests/test_e1_lifecycle_gates.py tests/test_e1_pids_cleanup.py tests/test_cpu_only_guard.py tests/test_e1_lifecycle_gates_v2.py tests/test_e1_launch_identity_v2.py tests/test_profiler_lifecycle_stop_concurrency.py tests/test_profiler_lifecycle_host_stream.py tests/test_profiler_preflight_retained.py tests/test_e1_host_identity.py tests/test_e1_forced_release_gates.py tests/test_load_generator_prebuilt.py tests/test_profiler_lifecycle_release_evidence.py tests/test_e1_frontend_binary_gate.py tests/test_e1_import_probe.py tests/test_e1_mem_watch.py tests/test_e1_storage_watch.py tests/test_e1_v3scale_gates.py tests/test_e1_v4scale_gates.py tests/test_e1_launch_identity_v4scale.py tests/test_e1_supervisor_v4.py tests/test_e1_reservation_guard.py tests/test_e1_reservation_release.py tests/test_e1_archive_size.py tests/test_e1_supervisor.py tests/test_profiler_lifecycle_v3_deadlines.py tests/test_e1_gpu_watch_v3.py \
      tests/fixtures/e1_main_v3/profile.json tests/fixtures/e1_main_v3/launch.json tests/fixtures/e1_nsys_val/profile.json tests/fixtures/e1_nsys_val/launch.json tests/fixtures/e1_nsys_val/server_log_excerpt.log tests/fixtures/e1_v4_scale/launch.predicted.json tests/fixtures/e1_v4_scale/reserve_pod.json \
      | sed "s#  tests/#  ../../../../../../tests/#" ); } > SCRIPTS_v4scale.sha256
sha256sum -c SCRIPTS_v4scale.sha256 --quiet && echo "SCRIPTS_v4scale.sha256: $(wc -l < SCRIPTS_v4scale.sha256) files, sha256 $(sha256sum SCRIPTS_v4scale.sha256 | cut -c1-16)"
if [ "${1:-}" = freeze ]; then
  sha256sum E1-V4-SCALE-PACKET.md SCRIPTS_v4scale.sha256 INPUTS_v4scale.sha256 E1-V4-SCALE-INPUTS.json V4S-PIN.json V4S-PIN-identities.json BINARY-PROVENANCE.json > FREEZE_v4scale.sha256
  sha256sum -c FREEZE_v4scale.sha256 --quiet && echo "FREEZE_v4scale.sha256: $(wc -l < FREEZE_v4scale.sha256) entries, sha256 $(sha256sum FREEZE_v4scale.sha256 | cut -c1-16)"
fi
