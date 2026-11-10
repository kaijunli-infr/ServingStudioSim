# ARM64 + CUDA 13 profiling environment for GB300 trays

Why this exists: the repository's `uv.lock` resolves a CPU-only torch on
aarch64 (its CUDA torch and `nvidia-*` runtime packages are pinned for
x86_64), and the profiler images (`profiling/container/*.Dockerfile`) install
that lock plus x86_64-only wheels. A GB300 tray is a Grace (aarch64) host, so
neither can run a kernel there. This directory is a separate uv project with
the same dependency set as the repository's default groups and an arm64 CUDA
stack. It is isolated: its own venv, its own uv cache, its own JIT caches and
its own copy of the profile database, all under one directory on the shared
home (`GB300_ROOT`, default `/mnt/shared/home/kevin/workspace/serving-studio/gb300`).

| file | role |
| --- | --- |
| `pyproject.toml` | the dependency set; torch pinned to the PyTorch cu130 index |
| `uv.lock` | the record: 107 packages, every wheel URL and hash, DeepGEMM's commit |
| `bootstrap.sh` | creates or reproduces the venv (the one supported procedure) |
| `gb300.env` | `source` it: venv path, caches, CUDA toolkit, profile DB copy |
| `collect_manifest.py` | provenance manifest of a run, collected without CUDA |

## What is pinned (first bootstrap, tray12, 2026-11-10)

| component | version | source |
| --- | --- | --- |
| Python | 3.12.3 (`/usr/bin/python3.12`, Ubuntu 24.04) | host |
| torch | 2.10.0+cu130 (cuDNN 9.15.1) | `https://download.pytorch.org/whl/cu130`, `manylinux_2_28_aarch64` |
| triton | 3.6.0 | PyPI |
| flashinfer-python / flashinfer-cubin | 0.6.11.post3 (the repository's locked version) | PyPI |
| nvshmem4py-cu13 | 0.4.1 (the repository uses `-cu12`) | PyPI |
| deep-gemm | 2.5.0 at `891d57b4` (the repository's pin), built against the env torch | git, `serendipity-zk/DeepGEMM` |
| transformers | 5.22.2 | PyPI |
| CUDA toolkit for JIT | 13.1 (`/usr/local/cuda-13.1`) | host; driver 610.57.04 = CUDA 13.3 user mode |

Deviations from the x86 reference stack, all deliberate: cu130 torch instead of
cu128 (no cu128 arm64 wheel of 2.10 fits the trays' CUDA 13 driver as cleanly),
`nvshmem4py-cu13` instead of `-cu12`, no `causal_conv1d` (x86_64 wheel only;
only the `causal_conv1d_env` kernels need it), and `TORCH_CUDA_ARCH_LIST=10.3`
so JIT builds compile for the GB300 only. `flashinfer-jit-cache` is not
installed: FlashInfer compiles its kernels with nvcc into
`FLASHINFER_WORKSPACE_BASE` on first use.

## Reproduce

```bash
cd <repository root>
CUDA_VISIBLE_DEVICES= profiling/env/arm64-gb300/bootstrap.sh   # CPU only
```

`bootstrap.sh` pre-installs torch from the cu130 index (DeepGEMM's `setup.py`
imports torch while uv resolves its metadata, and build isolation is off for
it as in the repository), then `uv lock` (a no-op when `uv.lock` is current),
then `uv sync` without and with DeepGEMM. It asserts the torch it got has
CUDA before locking: with PyPI as an extra index uv prefers PyPI's CPU-only
aarch64 wheel of the same version. The venv, uv cache and DeepGEMM build
take about 8 GB under `GB300_ROOT`.

## Run the Sim on it

```bash
source profiling/env/arm64-gb300/gb300.env
uv run --no-sync python -m launcher timing-predict presets/predict_llama3_8b_gb300_iter.json --dry-run --no-gpu
uv run --no-sync python -m profiling list
uv run --no-sync python profiling/env/arm64-gb300/collect_manifest.py --out "$GB300_ROOT/artifacts/manifest.json" --run-config preset=...
```

`gb300.env` sets `UV_PROJECT_ENVIRONMENT`, so `uv run --no-sync` from the
repository root uses this venv, not `.venv`. It also sets
`VIBESIM_PROFILE_RUNTIME=host` (workers run in this venv; the x86 images are
not usable here) and `VIBESIM_PROFILE_DB` to a copy of `profiling/profile.db`
under `GB300_ROOT/db`, so no run writes the tracked database.

## Isolation

| what | where | how |
| --- | --- | --- |
| venv, uv cache | `$GB300_ROOT/venv`, `$GB300_ROOT/cache/uv` | `UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR` |
| Triton, FlashInfer, torch extension, DeepGEMM, driver JIT caches | `$GB300_ROOT/cache/{triton,flashinfer,torch-extensions,deepgemm,nv-compute}` | `TRITON_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`, `TORCH_EXTENSIONS_DIR`, `DG_JIT_CACHE_DIR`, `CUDA_CACHE_PATH` |
| Hugging Face hub | `$GB300_ROOT/cache/huggingface` | `HF_HOME` |
| profiler worker cache | `$GB300_ROOT/cache/vibesim-profiler` | `VIBESIM_PROFILE_CACHE_DIR` |
| profile rows | `$GB300_ROOT/db/profile-gb300.db` | `VIBESIM_PROFILE_DB` (a copy; merge is a separate step) |
| artifacts, manifests, logs | `$GB300_ROOT/artifacts`, `$GB300_ROOT/logs` | `--output-dir`, `collect_manifest.py --out` |
| GPUs | the one GPU a run is approved for | `CUDA_VISIBLE_DEVICES=<index>` and `VIBESIM_PROFILE_GPUS=<index>`; `CUDA_VISIBLE_DEVICES=` (empty) for CPU-only work |

Nothing here reads or writes `~/.cache`, the repository's `.venv`, or
`profiling/profile.db`.
