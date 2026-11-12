#!/usr/bin/env bash
# CPU-only. The serving env for the fork rebased onto vLLM v0.31.0: its Python source
# (editable) over the precompiled cu130 aarch64 wheel of the v0.31.0 release commit,
# torch 2.13.0+cu130 from requirements/cuda.txt. Mirrors build_fork_env_base.sh.
set -u
source "$(dirname "$0")/env.sh"
V=$GB300_ROOT/fork-env-v0.31.0; mkdir -p $V
FORK=$SS/ServingStudioSim/alignment/profiler/vllm   # the path the launcher puts on PYTHONPATH; detached at the rebased commit
BASE_COMMIT=db9527a46873454610df6dbedf79a36d6bf1a7f6   # upstream commit of tag v0.31.0
IDX="--extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match"
cd $FORK
[ "$(git rev-parse v0.31.0^{commit})" = "$BASE_COMMIT" ] || { echo "v0.31.0 is not $BASE_COMMIT"; exit 2; }
git merge-base --is-ancestor $BASE_COMMIT HEAD || { echo "HEAD is not on top of v0.31.0"; exit 2; }
[ -z "$(git status --porcelain)" ] || { echo "fork worktree is dirty"; exit 2; }
if git diff --name-only $BASE_COMMIT HEAD | grep -E "\.(c|cc|cpp|cu|cuh|h|hpp|rs)$|^csrc/|^cmake/|CMakeLists.txt|^setup.py"; then echo "native changes: precompiled base invalid"; exit 2; fi
echo "fork HEAD $(git rev-parse HEAD)"
[ -x $V/venv/bin/python ] || uv venv -q --python 3.12 $V/venv
echo "deps start $(date -u +%FT%TZ)"
uv pip install --python $V/venv/bin/python -r requirements/cuda.txt $IDX > $V/install-deps.log 2>&1; echo "deps rc=$? $(date -u +%FT%TZ)"
echo "vllm start $(date -u +%FT%TZ)"
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=$BASE_COMMIT VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
  uv pip install --python $V/venv/bin/python -e . --no-deps $IDX > $V/install-vllm.log 2>&1; echo "vllm rc=$? $(date -u +%FT%TZ)"
uv pip install --python $V/venv/bin/python "nvtx==0.2.15" > $V/install-extra.log 2>&1; echo "extra rc=$?"
$V/venv/bin/python -c "import vllm,torch,flashinfer,platform;print(platform.machine(),vllm.__version__,torch.__version__,torch.version.cuda,flashinfer.__version__)" 2>&1 | tail -1
uv pip check --python $V/venv/bin/python 2>&1 | tail -2
echo BUILD_DONE
