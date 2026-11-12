#!/usr/bin/env bash
# CPU-only: the Sim launcher env for aarch64 (profiling/env/arm64-gb300), under GB300_ROOT.
set -u
source "$(dirname "$0")/env.sh"
cd $SS/ServingStudioSim
echo "start $(date -u +%FT%TZ)"
profiling/env/arm64-gb300/bootstrap.sh > $GB300_ROOT/bootstrap.log 2>&1; echo "bootstrap rc=$? $(date -u +%FT%TZ)"
echo BUILD_DONE
