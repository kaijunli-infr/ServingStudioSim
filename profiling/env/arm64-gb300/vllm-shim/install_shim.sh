#!/usr/bin/env bash
# Create (or rebuild) the GB300 vllm shim at $GB300_ROOT/vllm-shim, the directory
# profiling/exec/env.py's `vllm_env` points at through VIBESIM_VLLM_FORK_ROOT:
#   .venv      -> $GB300_ROOT/venv (the arm64 env; the worker runs its python,
#                 and torch's CUDA runtime comes from its torch/lib)
#   vllm/      the shim package (this directory's copy)
#   build/     the extension; manifest.json its provenance
# Needs: `source profiling/env/arm64-gb300/gb300.env` (GB300_ROOT, CUDA_HOME,
# TORCH_CUDA_ARCH_LIST) and the fork sparse clone at the pinned commit
# ($VLLM_FORK_SRC, default $GB300_ROOT/src/vllm-fork). CPU only.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${GB300_ROOT:?source profiling/env/arm64-gb300/gb300.env first}"
FORK=${VLLM_FORK_SRC:-$GB300_ROOT/src/vllm-fork}
DEST=$GB300_ROOT/vllm-shim
mkdir -p "$DEST"
ln -sfn "$GB300_ROOT/venv" "$DEST/.venv"
rm -rf "$DEST/vllm"
cp -r "$HERE/vllm" "$DEST/vllm"
cp "$HERE/shim_bindings.cpp" "$HERE/build_shim.py" "$HERE/probe_shim.py" "$DEST/"
"$GB300_ROOT/venv/bin/python" "$DEST/build_shim.py" --fork "$FORK" --out "$DEST" "$@"
echo "shim ready at $DEST; export VIBESIM_VLLM_FORK_ROOT=$DEST (gb300.env does)"
