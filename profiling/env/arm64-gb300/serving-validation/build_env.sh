#!/usr/bin/env bash
# CPU-only: build the isolated serving environment for the pinned vLLM fork.
#   1. the fork's CUDA requirements into $V/venv (torch 2.13.0+cu130, aarch64);
#   2. the fork's Python source, editable, over the native extensions of the
#      precompiled upstream wheel of its base commit 04730e82 (cu130), the
#      install the fork documents in ALIGNMENT.md. The fork changes no file
#      under csrc/, cmake/, CMakeLists.txt or setup.py relative to that commit.
# Never touches $GB300_ROOT/venv or the repository's own .venv.
set -u
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
V=/mnt/shared/home/kevin/workspace/serving-studio/gb300/serving-validation
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
export UV_CACHE_DIR=$V/uv-cache CUDA_VISIBLE_DEVICES=
IDX="--extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match"
FORK_COMMIT=e1423a4d7496dc6535862ae11bab8cb4d87b155f
BASE_COMMIT=04730e82700d0c9b957cda188dd1bbc8073e1fcd
cd $V/vllm-fork
[ "$(git rev-parse HEAD)" = "$FORK_COMMIT" ] || { echo "fork checkout is not $FORK_COMMIT"; exit 2; }
[ -x $V/venv/bin/python ] || uv venv -q --python 3.12 $V/venv
echo "deps start $(date -u +%FT%TZ)"
uv pip install --python $V/venv/bin/python -r requirements/cuda.txt $IDX > $V/install-deps.log 2>&1; echo "deps rc=$? $(date -u +%FT%TZ)"
echo "vllm start $(date -u +%FT%TZ)"
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=$BASE_COMMIT VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
  uv pip install --python $V/venv/bin/python -e . --no-deps $IDX > $V/install-vllm.log 2>&1; echo "vllm rc=$? $(date -u +%FT%TZ)"
uv pip install --python $V/venv/bin/python "nvtx==0.2.15" > $V/install-extra.log 2>&1; echo "extra rc=$?"
echo BUILD_DONE
