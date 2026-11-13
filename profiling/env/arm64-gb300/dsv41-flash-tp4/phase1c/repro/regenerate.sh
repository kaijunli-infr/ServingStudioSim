#!/bin/bash
# Regenerate the Phase 1B outputs and verify them against the manifest (CPU-only, ~4 minutes on one core).
# Usage: TABLES=/path/to/phase1a-tables PY=/path/to/python repro/regenerate.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
TABLES=${TABLES:?set TABLES to the directory holding diverse_nsys_run2/ quadrant_nsys/ diverse_nsys_salvaged/}
"$PY" "$HERE/verify.py" --tables "$TABLES" "$@"
