#!/usr/bin/env bash
# Stage 1.5 preflight, host side (read-only). Extends preflight_e1_host.sh (unchanged, Stage 1 evidence) for the configs_v2 passes.
# Run ON the capture tray as dc:
#   ssh pod4-gb300-10-trayNN 'export HOME=/mnt/shared/home/dc; E1_REPO=/mnt/shared/home/dc/workspace/serving-studio/wt-e1stage1 bash $E1_REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/preflight_e1_stage15.sh'
# Prints PASS/FAIL/INFO per check, exits non-zero on any FAIL. Nothing is started, written or deleted.
set -u
export HOME=/mnt/shared/home/dc
S=$HOME/workspace/serving-studio
REPO=${E1_REPO:-$S/ServingStudioSim}
E1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
RUN15=$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15
fail=0
ok()   { echo "PASS  $*"; }
bad()  { echo "FAIL  $*"; fail=1; }
info() { echo "INFO  $*"; }

echo "# Stage 1.5 preflight on $(hostname) at $(date -u +%FT%TZ)"
# A. everything Stage 1 checked (GPUs idle, leftovers, software identity, Phase 0 recipe, disk, e1_main config)
E1_REPO=$REPO bash $E1/preflight_e1_host.sh | sed 's/^/  /'
[ "${PIPESTATUS[0]}" = "0" ] && ok "Stage 1 host preflight" || bad "Stage 1 host preflight (see above)"

# B. Stage 1.5 inputs: configs_v2 match the committed hashes; every config says stream=false explicitly
( cd $E1 && sha256sum -c INPUTS_v2.sha256 --quiet ) && ok "configs_v2 / trace match INPUTS_v2.sha256" || bad "INPUTS_v2.sha256 mismatch"
for c in e1_dry_off_v2 e1_dry_on_v2 e1_dry_off2_v2 e1_nsys_val; do
  grep -q '"stream": false' $E1/configs_v2/$c/profile.json && ok "$c: stream=false explicit" || bad "$c: stream key missing"
  [ ! -e "$RUN15/$c/profile" ] && ok "no previous output dir for $c" || bad "output dir exists: $RUN15/$c/profile"
done
grep -q '"capture_start_delay_seconds": 10.0' $E1/configs_v2/e1_nsys_val/profile.json && ok "e1_nsys_val: nsys delayed start 10 s" || bad "e1_nsys_val nsys block"
! grep -q capture_duration_seconds $E1/configs_v2/e1_nsys_val/profile.json && ok "e1_nsys_val: no bounded stop" || bad "bounded stop present"

# C. Configuration identity between the arms (CPU)
( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && \
  uv run --no-sync python $E1/checks/e1_config_identity.py --a $E1/configs_v2/e1_dry_off_v2/profile.json --b $E1/configs_v2/e1_dry_on_v2/profile.json && \
  uv run --no-sync python $E1/checks/e1_config_identity.py --a $E1/configs_v2/e1_dry_off_v2/profile.json --b $E1/configs_v2/e1_dry_off2_v2/profile.json ) >/dev/null 2>&1 \
  && ok "config identity off_v2/on_v2 and off_v2/off2_v2" || bad "config identity (checks/e1_config_identity.py)"

# D. Launcher tree supports stream=false: Python opt-out present, Rust binary lists --stream
grep -q "effective_stream" $REPO/alignment/load_generator/config.py && ok "launcher: workload.stream opt-out applied" || bad "launcher lacks workload.stream (patch/apply_stream_optin.py)"
BIN=$REPO/alignment/load_generator/req-frontend/target/release/session_runner
if [ -x "$BIN" ] && $BIN --help 2>/dev/null | grep -q -- "--stream <STREAM>"; then ok "session_runner supports --stream ($(sha256sum $BIN | cut -c1-16))"
else bad "session_runner lacks --stream: apply patch/apply_stream_flag_rs.py + cargo build, or install \$S/bundles/session_runner.stream-flag.2cd7990+patch"; fi

# E. Launcher validates every v2 config (CPU only)
for c in e1_dry_off_v2 e1_dry_on_v2 e1_dry_off2_v2 e1_nsys_val; do
  ( cd $REPO && source $S/scripts/env.sh && source profiling/env/arm64-gb300/gb300.env && unset CUDA_VISIBLE_DEVICES && \
    uv run --no-sync python -m launcher alignment profile --dry-run $E1/configs_v2/$c/profile.json ) >/dev/null 2>&1 && ok "launcher dry-run $c" || bad "launcher dry-run $c"
done

# F. Autotune cache: informational only (the autotuner re-tunes every launch; AUTOTUNER-INVESTIGATION.md)
C=$S/gb300/cache/flashinfer-autotune
info "flashinfer-autotune cache: $(ls -d $C/*/ 2>/dev/null | wc -l) hash dirs, newest $(ls -t $C/*/autotune_configs.json 2>/dev/null | head -1 | xargs -r stat -c %y | cut -c1-19); expect ~19 min of tuning per launch regardless"
echo "# result: $([ $fail = 0 ] && echo PREFLIGHT PASS || echo PREFLIGHT FAIL)"
exit $fail
