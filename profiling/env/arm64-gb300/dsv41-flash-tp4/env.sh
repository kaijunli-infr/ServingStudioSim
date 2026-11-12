export HOME=/mnt/shared/home/dc
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
export SS=/mnt/shared/home/dc/workspace/serving-studio
export GB300_ROOT=$SS/gb300
export UV_CACHE_DIR=$GB300_ROOT/cache/uv
export CUDA_VISIBLE_DEVICES=
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
export UV_PYTHON_PREFERENCE=only-managed   # /usr/bin/python3.12 has no Python.h on rack 10; the managed build ships headers
