#!/usr/bin/env bash
# CPU-only: build the alignment load generator (req-frontend session_runner) at
# the revision the repository pins, into a separate target directory.
set -u
export PATH="$HOME/.cargo/bin:$PATH"
V=/mnt/shared/home/kevin/workspace/serving-studio/gb300/serving-validation
cd $V/req-frontend
[ "$(git rev-parse HEAD)" = 2cd7990c527f9750d2bda1fb1dfe35e77000a5f8 ] || { echo "wrong revision"; exit 2; }
echo "start $(date -u +%FT%TZ)"
CARGO_TARGET_DIR=$V/req-frontend-target CARGO_HOME=$HOME/.cargo \
  cargo build --locked --release --bin session_runner > $V/req-frontend-build.log 2>&1
echo "build rc=$? $(date -u +%FT%TZ)"
echo BUILD_DONE
