#!/usr/bin/env bash
# Pin the b=8 recovery scripts, checks, analysis and tests: writes SCRIPTS_b8.sha256 (verified by preflight_e1_b8.sh).
# Re-run only after a reviewed change; the packet records the manifest's own sha256.
set -euo pipefail
E1=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$E1/../../../../../.." && pwd)
cd $E1
FILES="make_e1_inputs_b8.py preflight_e1_b8.sh run_e1_b8.sh postrun_e1_b8.sh closeout_e1.sh collect_e1_b8.sh hash_scripts_b8.sh \
 checks/e1_pids.py checks/e1_process_census.sh checks/e1_workload_identity.py checks/e1_nsys_validation.py checks/e1_completeness_v2.py checks/e1_config_identity.py \
 analysis/e1_capture_window.py analysis/e1_nsys_rank_kernels.py preflight_e1_stage15.sh preflight_e1_host.sh"
{ for f in $FILES; do sha256sum $f; done; ( cd $REPO && sha256sum tests/test_e1_capture_window.py tests/test_e1_pids_cleanup.py | sed "s#  tests/#  ../../../../../../tests/#" ); } > SCRIPTS_b8.sha256
sha256sum -c SCRIPTS_b8.sha256 --quiet && echo "SCRIPTS_b8.sha256: $(wc -l < SCRIPTS_b8.sha256) files, sha256 $(sha256sum SCRIPTS_b8.sha256 | cut -c1-16)"
