# GB300 smoke test results (Phase 2B, 2026-11-10)

Executed `SMOKE_PLAN.md` on rack 3 `pod4-gb300-3-tray12-f3`, physical GPU 1
(`GPU-9574ced0-46be-d5b7-0098-01fa1e84dccb`, bus `00000009:06:00.0`), from the
isolated arm64 env (`torch 2.10.0+cu130`, `flashinfer 0.6.11.post3`, CUPTI
13.0.85, driver 610.57.04, CUDA 13.1 toolkit). Three attempts; the first two
exercised no profiler kernel (see "What went wrong"), the third is the result.
Artifacts: `$GB300_ROOT/artifacts/smoke-20261110-064049/` (also `smoke-latest`):
`manifest-{pre,post}.json`, `identity.json`, `checks.json`, one directory with
`results.json` per profiler step, `gpu-samples.csv` (2 Hz), `steps.jsonl`.

## Conditions that were verified

| condition | evidence |
| --- | --- |
| one GPU visible, the right one | `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=1`; torch reports `device_count 1`, `NVIDIA GB300`, cc `(10, 3)`, UUID equal to the expected one (`identity.json`) |
| no foreign process on GPU 1 before, during, after | compute-apps query empty before start and after the run; only our worker PIDs appeared on GPU 1 in between |
| no profile rows written | sha256 of `profiling/profile.db` and `$GB300_ROOT/db/profile-gb300.db` identical before and after each run; every `results.json` says `"persisted": false` |
| disk | shared home 25 TB free; cache growth 586 MB (run 1) + 24 MB + 24 MB (details below), limit 3 GB |
| time | run 3: 2 min 22 s wall, GPU busy well under 1 min; all three runs together under 8 min wall; limit 45 min |
| clocks / power | max SM 2070 MHz, HBM 3996 MHz, power limit 1400 W, persistence on, ECC on, MIG off, compute mode Default; the GPU-state fields of the pre and post manifests are identical |

## Profiler-path measurements (CUPTI kernel time, `profiling run --fresh`)

`B200` and `H200` are the tracked `profile.db` rows at the same spec.

| kernel / backend | spec | GB300 | plan bound | B200 | H200 |
| --- | --- | --- | --- | --- | --- |
| single_gemm torch_linear bf16, n 6144, k 4096 | m 64 | 11.5 µs, 279 TFLOPS | 8–16 µs | 12.5 µs | 15.8 µs |
| | m 2048 | 53.1 µs, 1942 TFLOPS | 50–85 µs | 74.3 µs | 144.6 µs |
| | m 4096 | 109.9 µs, 1876 TFLOPS | 2× m 2048 ± 15 % (+3.5 %) | 147.6 µs | 293.9 µs |
| rms_norm flashinfer bf16, hidden 4096 | m 64 | 3.1 µs, 337 GB/s | 2–4 µs | 2.64 µs | 2.82 µs |
| | m 2048 | 7.0 µs, 4776 GB/s | 7–11 µs | 9.03 µs | 9.92 µs |
| flashinfer_attn_prefill fa2 bf16, 32/8/128 | prefix 0, append 2048 | 121.2 µs, 284 TFLOPS | 25–150 µs | – | fa2 144.8 µs (fa3 77.6 µs) |
| | prefix 2048, append 2048 | 311.0 µs, 331 TFLOPS | < 300 µs: **missed by 3.7 %** | – | – |
| flashinfer_attn_decode fa2 bf16, 32/8/128 | batch 1, 4096 tokens | 13.7 µs, 1229 GB/s | < 20 µs | – | fa2 17.1 µs |
| | batch 64, 38400 tokens | 35.7 µs, 4498 GB/s | < 120 µs | – | – |
| flashinfer_attn_prefill fa3 (diagnostic) | prefix 0, append 2048 | failed cleanly after 68 s (JIT of the sm90a kernel, then launch refused) | clean failure ≤ 5 min | – | – |

FA3 error text, verbatim from the worker:
`Check failed: (status == cudaSuccess) is false: BatchPrefillWithRaggedKVCacheSM90Run failed with error: no kernel image is available for execution on the device`.

The one missed bound is a planning guess, not a defect: FA2 is a pre-Blackwell
kernel and reaches 13 % of the GB300 dense BF16 peak here (H200 fa2 reaches
24 % of its peak at prefix 0). Everything else landed at or below the B200
time, as the plan expected. The profiler's energy column is 0 (energy
measurement off by default).

### Compile time vs steady state

Wall time of each profiler step includes about 4 s of worker start-up.

| step | cold (empty FlashInfer cache) | warm | JIT |
| --- | --- | --- | --- |
| rms_norm flashinfer | 4.3 s | 4.2 s | none: ships ahead-of-time |
| flashinfer_attn_prefill fa2 | 23.7 s | 5.0 s | ≈ 18.7 s (first `plan+run` measured in-process: 18.5 s), 14.4 MB of cubins |
| flashinfer_attn_decode fa2 | 4.2 s | 4.2 s | none: ships ahead-of-time |
| single_gemm torch_linear | 5.7 s | – | none from FlashInfer; the CUDA driver JIT-compiled PTX for cuBLAS/torch kernels into `CUDA_CACHE_PATH` on first use (571 MB, run 1 only) |
| flashinfer_attn_prefill fa3 | 68 s | – | ≈ 60 s to build an sm90a cubin (9.4 MB) that then cannot launch |

Steady-state kernel times above are unaffected by the cold/warm split (cold and
warm runs agree to 0.3 %).

## Correctness against fp32 references (`smoke_checks.py`, same shapes)

| kernel | reference | max abs err | tolerance | result |
| --- | --- | --- | --- | --- |
| GEMM m 64 / 2048 / 4096 | fp32 `a @ w.T` | 0.94 / 1.00 / 1.00 (bf16 output rounding; |ref| up to ≈ 300) | atol 1.0, rtol 2e-2 | 0 mismatches of 393k / 12.6M / 25.2M |
| RMSNorm m 64 / 2048 | fp32 `x * rsqrt(mean(x²) + 1e-6) * w` | 0.031 / 0.031 | atol 5e-2, rtol 2e-2 | 0 mismatches |
| FA2 prefill (0, 2048) / (2048, 2048) | fp32 causal GQA softmax attention with prefix offset | 8.1e-3 / 7.8e-4 | atol 2e-2, rtol 2e-2 | 0 mismatches |
| FA2 decode batch 1 / 64 | fp32 per-request attention over the paged cache | 4.1e-4 / 1.1e-3 | atol 2e-2, rtol 2e-2 | 0 mismatches |

Also exact: `device_count == 1`, 1024² bf16 ones-matmul equals 1024 in every
element and sums to 1024³.

## Latency distributions (in-stream CUDA events, 200 back-to-back launches)

These are a spread check beside the CUPTI numbers, not a replacement: the loop
has no L2 flush and no host sync, so for kernels shorter than the host's launch
path (about 19 µs for a FlashInfer Python call on this Grace CPU, about 30 µs
for `F.linear` with m 64) the interval measures the host, not the kernel.

| kernel | p50 | p90 | p99 | CUPTI | note |
| --- | --- | --- | --- | --- | --- |
| GEMM m 64 | 29.8 µs | 34.1 | 46.6 | 11.5 µs | host-bound |
| GEMM m 2048 | 59.4 | 60.4 | 61.5 | 53.1 | p99/p50 1.04 |
| GEMM m 4096 | 111.6 | 113.7 | 114.7 | 109.9 | p99/p50 1.03 |
| RMSNorm m 64 / 2048 | 19.3 / 19.8 | 21.1 / 22.1 | 33.2 / 32.4 | 3.1 / 7.0 | host-bound |
| FA2 prefill (0, 2048) | 124.9 | 126.1 | 135.1 | 121.2 | p99/p50 1.08 |
| FA2 prefill (2048, 2048) | 313.3 | 318.2 | 325.6 | 311.0 | p99/p50 1.04 |
| FA2 decode batch 1 | 35.6 | 40.4 | 48.4 | 13.7 | host-bound |
| FA2 decode batch 64 | 39.9 | 40.0 | 42.0 | 35.7 | p99/p50 1.05 |

For every kernel long enough to measure this way, p50 is within 4–12 % of the
CUPTI time (CUPTI flushes L2 before each launch, so it should be the larger
number for memory-bound kernels, and it is for decode).

## GPU utilization, power, clocks (nvidia-smi at 2 Hz, run 3)

271 samples over 2 min 22 s; 19 samples with non-zero utilization. Peaks:
100 % utilization with 1370 W (of the 1400 W limit) and the SM clock dropping
from 2070 to 1890 MHz during the m 4096 GEMM; 95–100 % during prefill; peak
memory 3.5 GB. The GEMM numbers above were therefore taken at a power-limited
clock, which is the realistic serving condition. Idle between steps is the
worker start-up and the fp32 reference computation.

## Cache growth and leftovers

| cache (`$GB300_ROOT/cache/…`) | after run 3 | grew in |
| --- | --- | --- |
| `nv-compute` (CUDA driver PTX JIT cache) | 571 MB | run 1, first torch/cuBLAS use on sm_103 |
| `flashinfer` | 23.8 MB | every run (cleared before runs 2 and 3): fa2 prefill 14.4 MB + fa3 sm90a 9.4 MB |
| `cupti-ext` (profiler's CUPTI extension) | 0.75 MB | run 3 |
| `triton`, `torch-extensions`, `deepgemm`, `vibesim-profiler` | 0 | – |

No process was left on the GPU after any run. One stale build of the CUPTI
extension that run 2 had put under `/tmp/moesim_cupti_ext` on tray12's root
disk (3 MB, linked against the wrong CUPTI) was removed; `gb300.env` now points
`MOESIM_CUPTI_BUILD_DIR` at the private cache.

## What went wrong, and what changed because of it

1. **Run 1: the profiler never reached the GPU.** `profiling.exec.env`
   makes `default_env` the checkout's `.venv/bin/python` whenever that file
   exists, and the rack-3 checkout keeps the CPU-only test venv there, so every
   worker failed with `CUDA is required for …`. Workaround, no code change: run
   from a second worktree of the same branch without a `.venv`
   (`/mnt/shared/home/kevin/workspace/serving-studio/ServingStudioSim-gb300`,
   detached at the branch head); there `default_env` resolves to the arm64 venv.
   A cleaner fix would be an explicit interpreter override; not done.
2. **Run 2: CUPTI refused the device.** `_resolve_cupti_paths` only knew the
   CUDA 12 wheel layout (`nvidia/cuda_cupti/`), so on the cu13 stack it fell
   through to Triton's bundled CUPTI 12.8, which failed
   `cuptiActivityRegisterCallbacks` with `CUPTI_ERROR_INVALID_DEVICE` on this
   13.x driver. Fixed in commit `ff734683` (profiler harness, not kernel code):
   the cu13 layout (`nvidia/cu13/`) is recognised and ranked by Torch's CUDA
   major, the extension selects `CUpti_ActivityKernel10` when
   `CUPTI_API_VERSION >= 130000` (CUPTI 13 removed `Kernel9`; `Kernel10` is a
   superset), and four CPU unit tests cover the ranking. The x86 cu12 path is
   unchanged.
3. **Run 1 timing method.** The first `smoke_checks.py` synchronised after
   each launch, which made every kernel read as its launch latency; replaced by
   in-stream events (commit `8fb4936c`). The correctness checks were unaffected.
4. **One bound missed** (prefill with a 2048-token prefix, 311 vs 300 µs), see
   above.
