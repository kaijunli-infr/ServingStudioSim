#!/usr/bin/env bash
# Create (or reproduce) the isolated ARM64 + CUDA 13 env. Run from the
# repository root on a GB300 tray; CPU-only (no CUDA context is created).
#
#   CUDA_VISIBLE_DEVICES= profiling/env/arm64-gb300/bootstrap.sh
#
# Steps, in the order DeepGEMM's un-isolated build forces:
#   1. venv + torch (cu130 arm64 wheel) + the setuptools stack, so DeepGEMM's
#      setup.py can import torch when uv resolves its metadata;
#   2. `uv lock` (no-op when uv.lock is present and current: reproduction);
#   3. `uv sync` of everything but DeepGEMM, then of DeepGEMM against that torch.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=gb300.env
source "$here/gb300.env"
export DG_USE_LOCAL_VERSION=0   # DeepGEMM: JIT-only build, no git-clean assert (see justfile)
project=(--project "$here")

echo "[bootstrap] venv at $UV_PROJECT_ENVIRONMENT"
[ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ] || uv venv --python 3.12 "$UV_PROJECT_ENVIRONMENT"
echo "[bootstrap] torch for DeepGEMM's metadata build"
# torch alone from the cu130 index: with PyPI as an extra index uv prefers it
# and installs the CPU-only aarch64 wheel PyPI serves under the same version.
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
    --index-url https://download.pytorch.org/whl/cu130 "torch==2.10.0"
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
    "setuptools>=81.0.0" "wheel>=0.46.0" "packaging>=26.0" "numpy>=2.0.0"
"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
import torch
assert torch.version.cuda, f"CPU-only torch {torch.__version__}: the cu130 index was not used"
print("[bootstrap] torch", torch.__version__, "cuda", torch.version.cuda)
PY
echo "[bootstrap] lock"
uv lock "${project[@]}"
echo "[bootstrap] sync (all but DeepGEMM)"
uv sync "${project[@]}" --no-install-package deep-gemm
echo "[bootstrap] sync (DeepGEMM against the env torch)"
uv sync "${project[@]}"
echo "[bootstrap] done: $("$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)')"
