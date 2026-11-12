# Phase 0 evidence manifest — DeepSeek-V4.1-Flash-1017, GB300 TP4 (rack 10 tray16), 2026-11-12

## Source revisions (exact)
| component | revision | where |
|---|---|---|
| ServingStudioSim (campaign scripts, configs, progress.md, analysis scripts) | branch `dsv41-gb300-tp4` (rack 10) = `test-ss` (GitHub kaijunli-infr/ServingStudioSim) | `/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim` |
| vLLM fork (submodule `alignment/profiler/vllm`) | `0ab18711335780e062d65994a3243da0281d2081`, branch `servingstudio-alignment-v0.31.0` (17 commits rebased onto v0.31.0 db9527a4) | GitHub kaijunli-infr/vllm (serendipity-zk/vllm does NOT have this branch); worktree `/mnt/shared/home/dc/workspace/serving-studio/vllm-v0.31.0` |
| engine version string | `0.31.1.dev17+g0ab187113`, torch `2.13.0+cu130` | venv `gb300/fork-env-v0.31.0/venv` (uv-managed CPython 3.12) |
| model weights | `deepseek-ai/DeepSeek-V4.1-Flash-1017`, HF revision `08c8657f` (476 GB) | `/mnt/shared/weights/DeepSeek-V4.1-Flash-1017` |
| Nsight Systems | `2025.6.3.541-256337736014v0` | tray16 system install |
| host | Ubuntu 24.04.4, kernel `6.14.0-1013-nvidia-64k`, driver `595.71.05`, 4x NVIDIA GB300 (Grace aarch64) | pod4-gb300-10-tray16 |
| server flags | vLLM recipe TP4 flags verbatim (see each pass `profile.json` → `server`), chunk 16384, FULL_AND_PIECEWISE graphs (cap 256) | pass configs |

## Raw artifacts (outside Git)
Durable copy: `/mnt/shared/home/dc/workspace/serving-studio/artifacts/phase0-dsv41-flash-gb300-tp4-20261112/` (CephFS, shared volume, 22 TB free at write time).
- `run/<pass>/` — full rsync of each pass directory of `ServingStudioSim/logs/20261112_dsv41_flash_gb300_tp4/` (profile.json, run.log, profile/{nsys,vllm,load_generator,parsed.json,parsed.kernels.parquet,kernel_sequences.json,host_timeline.json}, analysis scripts and outputs).
- `run/progress.md` — the log of record (identical to the committed `profiling/env/arm64-gb300/dsv41-flash-tp4/progress.md`).
- `SHA256SUMS` — sha256 of every file under `run/` (verify: `cd run && sha256sum -c ../SHA256SUMS`).
Retrieval: `ssh pod4-gb300-10-trayNN` as `dc`, `export HOME=/mnt/shared/home/dc`; the path is identical on all 18 rack-10 trays (shared CephFS). Large files: `*.nsys-rep` (open with Nsight Systems ≥ 2025.6) and `*.sqlite` (`nsys export` output, readable with sqlite3 / the launcher's parser via `--resume`).
Hashes of the raw Nsight captures (full list of all 311 files in `SHA256SUMS`):
| file | sha256 | size |
|---|---|---|
| run/quadrant_nsys_run1/…/quadrant_nsys.nsys-rep (pass 1 run 1, cold caches) | 369a91eb9fe694f3473cddbc9677f0c926f83efbfd46f32b58ada91b9bf8e481 | see SHA256SUMS dir |
| run/quadrant_nsys_run1/…/quadrant_nsys.sqlite | a1cd3d1c58e55444db5ef8765641edb1669521258c4846d207e387865c86d47e | |
| run/quadrant_nsys/…/quadrant_nsys.nsys-rep (pass 1 run 2, warm — the pass-1 capture of record) | 4cbc9b8a74f4545d4c3d71935b6bc5520cf01b0f1119f2b771ffefd96b07b279 | |
| run/quadrant_nsys/…/quadrant_nsys.sqlite | 744eef870abf1b7b19115d5662771650c174e75599771bbf5e83c33a59be3fc9 | |
| run/diverse_nsys/…/diverse_nsys.nsys-rep (pass 3 original, bounded 60 s window, engine died — salvaged) | da559e3fb60ae6178ef5d058371c15a8e967172b200ca17610a34b69dc32e059 | 767 MB |
| run/diverse_nsys/…/diverse_nsys.sqlite | 31de9e83f5131c51d0b8cd5343e52d55a8a68e4373bd08c88e3a5ff323323414 | 3.7 GB |
| run/diverse_nsys_run2/…/diverse_nsys_run2.nsys-rep (pass 3 recovery — the pass-3 capture of record) | 4a2a9451a2970541412d23fa5744469a4712bee321ca457f49892d241d7aea73 | 2.83 GB |
| run/diverse_nsys_run2/…/diverse_nsys_run2.sqlite | c80243a84ca8bc1a5bc41eba1979b29c0e3aeae24f5a582ee546d56459b29562 | 14.5 GB |

## Passes (all on tray16, TP4, same server flags)
| pass | dir | kind | status | clean-sample exclusions (iteration ids) |
|---|---|---|---|---|
| 1 quadrant_nsys run 1 | `run/quadrant_nsys_run1` | nsys, quadrant trace, cold JIT/autotune caches | complete; superseded by run 2 for timing | JIT-compile iterations (see progress.md) |
| 1 quadrant_nsys run 2 | `run/quadrant_nsys` | nsys, quadrant trace, warm caches | complete, 6 clean 16,384-token chunks | 245, 505 |
| 2 quadrant_workload | `run/quadrant_workload` | workload_metrics (no nsys) | complete | 507 |
| 3 diverse_nsys (original) | `run/diverse_nsys` | nsys, diverse_100, bounded 60 s window | engine died at cudaProfilerStop; 4,687 iterations salvaged (`salvage_parse.py`) | 1744, 1755, 1923, 5488 |
| 3 diverse_nsys_run2 (recovery) | `run/diverse_nsys_run2` | nsys, diverse_100, 10 s delayed start, no bounded stop | **complete**: 17,686 iterations, 100 % NVTX attribution, 100/100 requests | 49 module-load iterations (`jit_excl.txt`) |
| 4 diverse_workload | `run/diverse_workload` | workload_metrics (no nsys) | complete | 3861, 12868 |
| 5 diverse_corpus | `run/diverse_corpus` | token_corpus (per-token expert routes) | complete, 135,144 tokens × 40 layers × top-6, completeness validated | — |

## Committed (Git) derived artifacts
`profiling/env/arm64-gb300/dsv41-flash-tp4/` on branch `dsv41-gb300-tp4` (rack 10) = `test-ss` (GitHub kaijunli-infr/ServingStudioSim):
`progress.md` (log of record), `phase0/<pass>/` (profile.json, filtered run.log, profile_result.json, salvage_result.json, load_generator_summary.json,
launch/drive_summary json, token_corpus manifest/provenance, analysis scripts `*.py` + their outputs `*.out`, `jit_excl.txt`, `kernel_sequences.json.gz`),
`collect_phase0.sh`, `env.sh`, `run_pass.sh`, `build_fork_env_v0310.sh`, `bootstrap_sim_env.sh`, `make_configs.py`, this `MANIFEST.md`, `SHA256SUMS`.
Submodule `alignment/profiler/vllm` is pinned to 0ab18711 (only reachable on kaijunli-infr/vllm; `.gitmodules` still points at serendipity-zk/vllm,
so `git submodule update` needs `git -C alignment/profiler/vllm fetch https://github.com/kaijunli-infr/vllm servingstudio-alignment-v0.31.0`).

## Reproduce a pass
```
ssh pod4-gb300-10-tray16; export HOME=/mnt/shared/home/dc; cd $HOME/workspace/serving-studio
source scripts/env.sh                      # GB300_ROOT, uv-managed python, caches under gb300/cache
scripts/run_pass.sh <pass>                 # refuses if GPUs busy; runs: uv run --no-sync python -m launcher alignment profile <pass>/profile.json
```
Pass configs: `ServingStudioSim/logs/20261112_dsv41_flash_gb300_tp4/<pass>/profile.json` (generated by `dsv41-gb300/make_configs.py`;
copies in `phase0/<pass>/profile.json`). Tray reservation during the campaign: pause pod `dsv41-gb300-reserve-tray16` (ns `inference-bench`).

## Known limitations carried into Phase 1 review
- DeepGEMM nvcc JIT (~3.3 s per first-seen GEMM M) hits every pass; handled by iteration exclusion lists, never by config changes.
- Prefill chunks > 256 tokens run eager (FULL_AND_PIECEWISE, default `max_cudagraph_capture_size`); small eager mixed forwards are host-launch-bound,
  so twoshot all-reduce durations there are mostly rank-arrival skew (see progress.md, closeout item 2).
- Pass 3 original failed (bounded capture); its salvaged data is kept for provenance only. Recovery window starts 10 s into the trace (requests 0–8 outside).
- DSpark fields (layers 37–39) are inactive draft-model metadata in this configuration (closeout item 3); no speculative decoding was active.
- No push rights to SyFI-ServingStudio/ServingStudioSim or serendipity-zk/vllm; everything lives on kaijunli-infr forks.
