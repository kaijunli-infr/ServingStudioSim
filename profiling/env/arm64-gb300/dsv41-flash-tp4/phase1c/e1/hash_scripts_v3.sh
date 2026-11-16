#!/usr/bin/env bash
# Pin the full-E1 v3 scripts, checks, analysis (incl. the audit tools) and tests: writes SCRIPTS_v3.sha256 (verified by preflight_e1_main_v3.sh).
set -euo pipefail
E1=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$E1/../../../../../.." && pwd)
cd $E1
FILES="make_e1_inputs_v3.py preflight_e1_main_v3.sh run_e1_main_v3.sh postrun_e1_main_v3.sh monitor_e1.sh closeout_e1.sh collect_e1_main_v3.sh hash_scripts_v3.sh make_partition_1c.py \
 preflight_e1_stage15.sh preflight_e1_host.sh \
 checks/e1_pids.py checks/e1_process_census.sh checks/e1_workload_identity.py checks/e1_nsys_validation.py checks/e1_completeness_v2.py checks/e1_config_identity.py \
 analysis/e1_capture_window.py analysis/e1_nsys_rank_kernels.py analysis/e1_identify.py analysis/e1_rank_dependence.py analysis/e1_autotune_tactics.py analysis/e1_schedule_sim.py"
{ for f in $FILES; do sha256sum $f; done; ( cd $REPO && sha256sum tests/test_e1_capture_window.py tests/test_e1_pids_cleanup.py | sed "s#  tests/#  ../../../../../../tests/#" ); } > SCRIPTS_v3.sha256
sha256sum -c SCRIPTS_v3.sha256 --quiet && echo "SCRIPTS_v3.sha256: $(wc -l < SCRIPTS_v3.sha256) files, sha256 $(sha256sum SCRIPTS_v3.sha256 | cut -c1-16)"
