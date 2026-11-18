#!/usr/bin/env bash
# CPU-only test command: hides CUDA devices and loads the guard that fails the run (exit 3) if the pytest
# process opened a /dev/nvidia<N> device node or initialised torch.cuda anyway.
#   bash tests/cpu_only.sh [pytest args]            (e.g. -q tests/test_profiler_lifecycle.py; never with -n)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=""
exec uv run --no-sync python -m pytest -p cpu_only_guard "$@"
