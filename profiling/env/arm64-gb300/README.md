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
| `SMOKE_PLAN.md`, `run_smoke.sh`, `smoke_checks.py` | the Phase 2B GPU smoke test: plan, orchestrator (one GPU, `--fresh`, manifests, sampler, deltas), correctness and latency checks |
| `SMOKE_RESULTS.md` | what the smoke test measured on tray12 GPU 1 (2026-11-10) and what broke on the way |
| `PROPOSALS.md` | fa3 SM90 gate (implemented in Phase 2C, with the `Kernel::build` device gate), kv_cache_append torch-vs-vLLM equivalence, the first TP1 fill batch |
| `run_tp1_fill.sh`, `tp1_probe.py`, `tp1_specs.py`, `tp1_validate.py`, `tp1_report.py` | the TP1 fill campaign: probe gate, spec export from `simulator cost-trees --kernel-configs`, per-batch validation, repeatability, summary |
| `TP1_RESULTS.md` | the TP1 fill: first batch (1166 rows), Phase 2C (fa3 gate, the 70 vLLM kv_cache_append rows), final validation; 0 / 1236 specs missing in the private DB (2026-11-10) |
| `vllm-shim/` | the fork's `reshape_and_cache_flash` kernel built unmodified as a torch extension, with a `vllm` package shim the `vllm_cuda` runner imports; `build_shim.py` pins the fork commit and file hashes, `probe_shim.py` checks correctness, kernel identity and CUPTI timing on the GPU |
| `run_kv_fill.sh` | Phase 2C driver: shim probe gate, worker-vs-probe gate, the 70-row fill, validation, repeat sample, manifests |
| `MERGE_RESULTS.md`, `merge/` | the merge of the 1236 GB300 rows into the tracked `profiling/profile.db` (2026-11-10): checksums, row counts, checks, regression results, prediction parity, measurement provenance; the profiler's merge report and the check report |
| `TP2_TP4_PROPOSAL.md` | TP2 / TP4 plan: missing configurations, the all-reduce route (torch's NCCL backend is available), topology, GPU allocation, estimates, and the idle-GPU-memory investigation |
| `run_tp24_fill.sh`, `tp_new_specs.py` | the TP2/TP4 single-GPU fill: spec export (TP cells minus TP1's), checkpointed validated batches, budget and isolation guards |
| `TP24_RESULTS.md` | that fill (1854 rows in the private DB, 2026-11-10): coverage, gates, latency distributions, repeatability, provenance |
| `run_allreduce_fill.sh`, `nccl_probe.py`, `env_guard.py`, `allreduce_spikes.py` | the TP2/TP4 all-reduce stages: fail-closed environment guard, GPU identity and isolation preflight, NCCL rank-group probe (rank-to-GPU mapping, SUM check, init vs steady state), fill, validation, fresh repeats, rule-based re-measurement of non-reproduced rows |
| `ALLREDUCE_RESULTS.md` | those stages (38 rows in the private DB, 2026-11-10): probe, rows, scaling, repeatability and its limits, TP1/TP2/TP4 predictions, provenance |
| `merge_check.py` | verifies a `profiling merge-db` output against the DB it started from: prior rows byte-for-byte present, added rows all one GPU, no duplicate keys, valid `time_ms`, run records consistent |
| `public_api_check.py` | local public API query over a profile DB (GB300 catalog entry, peaks, per-kind rows); CPU only, nothing deployed |

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
repository root uses this venv, not `.venv`. The profiler's *workers* are a
different matter: `profiling.exec.env` makes `default_env` the checkout's
`.venv/bin/python` whenever that file exists (CPU-only torch here), so run
profiling from a checkout **without** a `.venv`; on rack 3 that is the worktree
`ServingStudioSim-gb300` next to the main checkout (`git worktree add --detach
… gb300-support`, re-detach after each push). The CUPTI timer builds its
extension against the `nvidia-cuda-cupti` 13 wheel (`nvidia/cu13/`), into
`MOESIM_CUPTI_BUILD_DIR` (set by `gb300.env`). It also sets
`VIBESIM_PROFILE_RUNTIME=host` (workers run in this venv; the x86 images are
not usable here) and `VIBESIM_PROFILE_DB` to a copy of `profiling/profile.db`
under `GB300_ROOT/db`, so no run writes the tracked database.

## Environment incident, 2026-11-10 (found and repaired the same day)

At 10:06:35 UTC a full `timing-predict` was run with `gb300.env` sourced. Its
analyze stage (Rust) spawns plain `uv run python …` commands; with
`UV_PROJECT_ENVIRONMENT` pointing at this venv, uv synced the **repository's
root lock** into it: `torch 2.10.0+cpu` over `2.10.0+cu130`, `transformers`
5.13.0 over 5.22.2, and the cu12 NVSHMEM packages added. It went unnoticed
for five hours because everything run in between was CPU-only.

- **Measurements are not affected.** Every GPU row of the day was measured
  before 10:04:47 UTC (the last fill's manifests record `torch 2.10.0+cu130`);
  the database merge, the predictions and the public API checks do not use
  torch's CUDA side.
- **One wrong conclusion came out of it:** a CPU-only check during those hours
  read `USE_NCCL=OFF` and was reported as "the arm64 torch wheel has no NCCL".
  The real wheel has it (`TP2_TP4_PROPOSAL.md`).
- **Found** when the TP2/TP4 driver's identity probe stopped with `Torch not
  compiled with CUDA enabled` before any measurement.
- **Repaired** with `bootstrap.sh` (now pinning `torch==2.10.0+cu130`, so a
  same-version CPU wheel is replaced) plus a reinstall of
  `nvidia-nvshmem-cu13` / `nvshmem4py-cu13`, whose files the cu12 packages
  had overwritten. Afterwards: every file of every installed distribution
  present (45,107 checked), `uv sync --locked --dry-run` proposes nothing but a
  same-version reinstall of `nvidia-cusparselt-cu13`, torch is
  `2.10.0+cu130` with CUDA 13.0 and NCCL, the shim library hash is unchanged,
  and the identity, Triton, correctness and shim probes pass on the GPU.
  Log: `$GB300_ROOT/logs/bootstrap-restore-20261110.log`.
- **Prevented** by `export UV_NO_SYNC=1` in `gb300.env`: no `uv run` may sync
  this venv; only `bootstrap.sh` does.

## Isolation

| what | where | how |
| --- | --- | --- |
| venv, uv cache | `$GB300_ROOT/venv`, `$GB300_ROOT/cache/uv` | `UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR` |
| Triton, FlashInfer, torch extension, DeepGEMM, driver JIT caches | `$GB300_ROOT/cache/{triton,flashinfer,torch-extensions,deepgemm,nv-compute}` | `TRITON_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`, `TORCH_EXTENSIONS_DIR`, `DG_JIT_CACHE_DIR`, `CUDA_CACHE_PATH` |
| Hugging Face hub | `$GB300_ROOT/cache/huggingface` | `HF_HOME` |
| profiler worker cache | `$GB300_ROOT/cache/vibesim-profiler` | `VIBESIM_PROFILE_CACHE_DIR` |
| CUPTI timer extension build | `$GB300_ROOT/cache/cupti-ext` | `MOESIM_CUPTI_BUILD_DIR` (default would be `/tmp`) |
| profile rows | `$GB300_ROOT/db/profile-gb300.db` | `VIBESIM_PROFILE_DB` (a copy; merge is a separate step) |
| artifacts, manifests, logs | `$GB300_ROOT/artifacts`, `$GB300_ROOT/logs` | `--output-dir`, `collect_manifest.py --out` |
| GPUs | the one GPU a run is approved for | `CUDA_VISIBLE_DEVICES=<index>` and `VIBESIM_PROFILE_GPUS=<index>`; `CUDA_VISIBLE_DEVICES=` (empty) for CPU-only work |

Nothing here reads or writes `~/.cache`, the repository's `.venv`, or
`profiling/profile.db`.
