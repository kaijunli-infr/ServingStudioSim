#!/usr/bin/env bash
set -u
source "$(dirname "$0")/env.sh"
cd $SS/ServingStudioSim/alignment/load_generator/req-frontend
echo "start $(date -u +%FT%TZ) rev $(git rev-parse HEAD)"
cargo build --locked --release --bin session_runner > $SS/gb300/req-frontend-build.log 2>&1; echo "build rc=$? $(date -u +%FT%TZ)"
echo BUILD_DONE
