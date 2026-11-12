# Phase 0 evidence freeze — DeepSeek-V4.1-Flash-1017 on GB300, vLLM TP4 (rack 10 tray16)

Status: **frozen 2026-11-12**. Baseline = kaijunli-infr/ServingStudioSim branch `test-ss` at **`ca9f8ed06bd97606ae7864740fa6d166935f1f3e`**
(identical to rack-10 branch `dsv41-gb300-tp4`). Nothing under `profiling/env/arm64-gb300/dsv41-flash-tp4/{progress.md,MANIFEST.md,SHA256SUMS,phase0/}`
or under the CephFS artifact directory is to be modified; Phase 1 work lives in `phase1/` and in new branches only.

## 1. Source, model and environment revisions
| component | revision | notes |
|---|---|---|
| ServingStudioSim | `ca9f8ed0` (`test-ss`); campaign scripts/configs first committed at `7ce99e55`; delayed nsys start at `839c3617` | base: `gb300-support` 73e23c2e |
| vLLM fork (submodule `alignment/profiler/vllm`) | `0ab18711335780e062d65994a3243da0281d2081`, branch `servingstudio-alignment-v0.31.0` = 17 alignment commits rebased onto upstream v0.31.0 (`db9527a4`) | hosted only on kaijunli-infr/vllm; `.gitmodules` still names serendipity-zk/vllm |
| engine version string | `0.31.1.dev17+g0ab187113` | torch `2.13.0+cu130`, flashinfer per fork lock |
| Python | uv-managed CPython 3.12 (`UV_PYTHON_PREFERENCE=only-managed`) | system python lacks headers |
| model weights | `deepseek-ai/DeepSeek-V4.1-Flash-1017`, HF revision `08c8657f` | `/mnt/shared/weights/DeepSeek-V4.1-Flash-1017` (476 GB) |
| model config | 40 layers, 384 routed experts top-6 + 1 shared, 8 index-source layers, engram layers [1, 14], `num_nextn_predict_layers` 3, dspark_* fields (draft model, inactive) | see progress.md "Closeout item 3" |
| Nsight Systems | `2025.6.3.541-256337736014v0` | `/usr/local/bin/nsys` |
| host | pod4-gb300-10-tray16: Ubuntu 24.04.4, kernel `6.14.0-1013-nvidia-64k`, driver `595.71.05`, 4× NVIDIA GB300 (Grace aarch64) | shared CephFS home `/mnt/shared/home/dc` |
| caches | DeepGEMM / FlashInfer autotune / Triton / Inductor / nv-compute under `gb300/cache/` (persistent, warm after pass 1) | JIT behaviour below |

## 2. Workload and capture configurations
Server (identical in all passes, `phase0/<pass>/profile.json` → `server`): TP 4, DP 1, `chunk_size` 16384, `gpu_memory_utilization` 0.9,
`enforce_eager` false (FULL_AND_PIECEWISE CUDA graphs, default capture cap 256 tokens → every prefill chunk > 256 tokens runs eager), 1 API server,
`--max-num-seqs 128`, `--max-model-len 131072`, `--tokenizer-mode deepseek_v41`, `--attention-config {"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}`,
`--kv-cache-dtype fp8`, `--engram-config {"cpu_offload":true}`, `--language-model-only`. **No `--enable-expert-parallel`**: experts are TP-sharded across the 4 ranks
(structural difference from the B200 preset `deepseek_v41_flash_b200_vllm_tp4_ep4`, which uses EP4 and chunk 2048).

| pass | trace | arrivals | nsys |
|---|---|---|---|
| quadrant_nsys (run1 cold, run2 warm) | `trace/quadrant_c48_96.csv` (96 requests: 128/128, 128/4096, 4096/128, 3072/3072 × 24) | saturated, max_concurrency 48 | cuda_profiler_api, graph trace `node`, no start delay |
| quadrant_workload | same | same | none (workload_metrics) |
| diverse_nsys (original) | `trace/diverse_100.csv` (100 requests, inputs 256 … 131,070, outputs ≤ 8192, total output 135,244, last arrival 114.9 s) | trace-timed Poisson | start delay 10 s **and** `capture_duration_seconds` 60 → failed (see §4) |
| diverse_nsys_run2 (recovery) | same | same | start delay 10 s, **no bounded stop**; profiler stopped after the workload with the server idle |
| diverse_workload | same | same | none |
| diverse_corpus | same | same | token_corpus (per-token expert routes of accepted tokens) |

Text source `corpus/enwik9`; `warmup: true`; nsys `cuda_event_trace` off.

## 3. Valid and excluded iteration ids
Exclusion rule (unchanged from Phase 0): any iteration with `jit_module_loads > 0` or `jit_stall_ns > 0` is excluded from clean timing samples.
DeepGEMM compiles a new `sm100_fp8_gemm_1d1d` cubin (~3.3 s, nvcc) for every first-seen GEMM M; cached cubins still cost a module load (≤ 32 ms).
| pass | iterations (clean) | excluded ids |
|---|---|---|
| quadrant_nsys run 2 (record) | 4,750 (index 0 … 4749) | 245, 505 |
| quadrant_nsys run 1 (cold) | superseded; keep for provenance | compile-stall iterations per progress.md |
| diverse_nsys original (salvaged) | 4,687 (60 s window) | 1744, 1755, 1923, 5488 |
| **diverse_nsys_run2 (record)** | **17,686 (index 1614 … 19299; 168 mixed, 17,518 decode)** | 49 module-load iterations, full list in `phase0/diverse_nsys_run2/jit_excl.txt` (2294, 2302, 2305, 2931, 3121, 3137, 3192, 3333, 3405, 3409, 3411, 3494, 3961, 3971, 4169, 4285, 4694, 5289, 7475, 7477, 7588, 7736, 7965, 8645, 8654, 8780, 8782, 8787, 8891, 9094, 9482, 9486, 9765, 10148, 10468, 10807, 10984, 11101, 11814, 11817, …) |
| quadrant_workload / diverse_workload | request-level metrics only | JIT iterations 507 / 3861, 12868 noted; no kernel data |

## 4. Original failed capture vs recovered capture (kept separately)
| | diverse_nsys (original) | diverse_nsys_run2 (recovery) |
|---|---|---|
| directory | `run/diverse_nsys/` (artifacts), `phase0/diverse_nsys/` (repo) | `run/diverse_nsys_run2/`, `phase0/diverse_nsys_run2/` |
| nsys-rep sha256 | `da559e3fb60ae6178ef5d058371c15a8e967172b200ca17610a34b69dc32e059` (767 MB) | `4a2a9451a2970541412d23fa5744469a4712bee321ca457f49892d241d7aea73` (2.83 GB) |
| sqlite sha256 | `31de9e83f5131c51d0b8cd5343e52d55a8a68e4373bd08c88e3a5ff323323414` (3.7 GB) | `c80243a84ca8bc1a5bc41eba1979b29c0e3aeae24f5a582ee546d56459b29562` (14.5 GB) |
| outcome | bounded `cudaProfilerStop` at +70 s froze the engine 162 s while nsys wrote the report; workers hung; EngineDeadError; 36/100 requests truncated; launcher aborted at the request-record join | 100/100 requests, clean shutdown, `rc=0` |
| data | `salvage_parse.py` re-ran the parser on the sqlite: 4,687 iterations, 100 % attribution | launcher pipeline: 17,686 iterations, 100 % attribution |
| use in Phase 1 | cross-check only (same trace, independent run): long-context slope 0.256 ms/1k kv vs 0.260 in the recovery | **capture of record** for diverse_100 |

## 5. Kernel-to-NVTX attribution rules (parser `alignment/nsys/parse.py`)
- The fork emits one NVTX range per phase per iteration on each worker thread: `vllm_iteration(N): {preprocess, forward, postprocess, sample, bookkeep, eplb}` (28 ranges per iteration over 4 devices).
- **Ownership by correlationId**: for each range, the CUDA runtime/driver calls issued on the emitting `globalTid` inside `[start, end)` are collected, their `correlationId`s mapped to `CUPTI_ACTIVITY_KIND_KERNEL` rows of the same `globalPid`. A kernel belongs to the range that launched it, not to the range it overlaps in time. This is what makes CUDA-graph decode iterations attributable (graph kernels execute after the `forward` range has closed; graph node tracing = `cuda_graph_trace: node`).
- Per range the parser records `kernel_count`, `kernel_sum_duration_ns` (sum over streams) and `kernel_busy_union_ns` (union of kernel intervals). GPU idle is measured against the kernel span, not the NVTX window. Multi-stream overlap = sum − union.
- JIT: module loads (`cuModuleLoadData*`/`cuLibraryLoad*`) inside a range are counted into `jit_module_loads`, their host time into `jit_stall_ns`.
- Phase 1 kernel groups are defined on kernel names (regexes in `phase1/longctx_chains.py::group` and `phase1/critpath.py::is_comm`); the per-kernel inventory is in `phase0/<pass>/gate5b.out`.

## 6. Per-rank timestamp alignment assumptions
- DP = 1, so the 4 TP ranks share `iteration_index`; iterations are paired across devices by that index (`_pair_by_iteration_index`), reference device = device 0, window selected on the reference rank; `devices_absent`/`unpaired_steps_by_device` were empty in every pass.
- All ranks are processes on one host captured in one nsys session; CUPTI kernel timestamps are on the session's common timebase, so cross-rank start/end comparisons are made directly, without offset correction. Empirical check: in graph-launched decode iterations the spread of the first kernel start across ranks is 4–13 µs (median), i.e. any clock offset is below that.
- Host events (NVTX ranges, API calls) are on the same timebase; "host launch skew" = spread of the `forward` range start across ranks.
- Critical-path quantities use the slowest rank (last kernel end − first kernel start over all ranks), never the sum over ranks.

## 7. Checksums and retrieval
- Raw copy: `/mnt/shared/home/dc/workspace/serving-studio/artifacts/phase0-dsv41-flash-gb300-tp4-20261112/` (CephFS, reachable from any rack-10 tray as `dc`): `run/<pass>/`, `run/progress.md`, `SHA256SUMS` (311 files, verified `sha256sum -c` → all OK on 2026-11-12), `MANIFEST.md`, `meta/`.
- Repo copies (committed at `ca9f8ed0`): `profiling/env/arm64-gb300/dsv41-flash-tp4/{progress.md,MANIFEST.md,SHA256SUMS,collect_phase0.sh,phase0/<pass>/…}`.
- Verify: `cd …/phase0-dsv41-flash-gb300-tp4-20261112 && sha256sum -c --quiet SHA256SUMS`.
- Re-parse without GPUs: `nsys export --type sqlite <rep>` then the launcher's parser (`salvage_parse.py` pattern, `PYTHONPATH=$PWD`) or `--resume`.
