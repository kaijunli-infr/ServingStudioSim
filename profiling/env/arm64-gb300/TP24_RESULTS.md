# GB300 Llama 3.1 8B BF16 TP2/TP4 single-GPU fill (2026-11-10)

The per-rank kernels a TP2 or TP4 replica needs beyond the TP1 rows: **1854
rows, all measured, all batches PASS**. Private database only
(`$GB300_ROOT/db/profile-gb300.db`); the tracked `profiling/profile.db` is
unchanged (`41c9cbe5…`). No communication kernel was run.

| | |
| --- | --- |
| run directory | `$GB300_ROOT/artifacts/tp24/run-20261110-150233` (`run-latest`), log `$GB300_ROOT/logs/tp24-fill-run2.log`, checkpoints `$GB300_ROOT/artifacts/tp24/state/` |
| source | worktree `ServingStudioSim-gb300` at `2a295264`, **clean**; the run records carry that hash without `-dirty` |
| device | rack 3 tray12 physical GPU 1, `GPU-9574ced0-46be-d5b7-0098-01fa1e84dccb` (asserted), cc 10.3, driver 610.57.04 |
| software | torch 2.10.0+cu130, triton 3.6.0, flashinfer 0.6.11.post3, the vLLM shim library `ca75929f…` (unchanged) |
| wall | 15:02:33 to 15:26:46 UTC, **24 min 13 s** of the 60 min allowed (budget guard 55 min) |
| disk | 100 MB added under `$GB300_ROOT` (74 MB of it the pre-run database backup, 8.1 MB JIT caches, 0.3 MB database growth), limit 0.5 GB |
| private DB | before `f2dae6e9…` 74,207,232 B (kept as `artifacts/tp24/profile-gb300.db.pre-tp24`), after `dd50ceeb…` 74,502,144 B, `integrity_check` ok |

A first attempt at 14:56 UTC stopped at the identity probe before measuring
anything: the environment's torch had been replaced by a CPU-only build
earlier in the day (README, "Environment incident"). The environment was
restored with `bootstrap.sh` and the run restarted from a clean worktree.

## Coverage

| kind / backend | specs | rows | TP2 | TP4 |
| --- | --- | --- | --- | --- |
| single_gemm torch_linear | 544 | 544 | 272 | 272 |
| single_gemm torch | 544 | 544 | 272 | 272 |
| flashinfer_attn_prefill fa2 | 270 | 270 | 135 | 135 |
| flashinfer_attn_decode fa2 | 324 | 324 | 162 | 162 |
| kv_cache_append vllm_cuda | 140 | 140 | 70 | 70 |
| elementwise triton | 32 | 32 | 16 | 32 (contains TP2's 16) |
| **total** | **1854** | **1854** | 927 | 943 |

0 failed specs, 0 retries, 0 rows flagged `is_outlier`. GB300 rows in the
private database: 3090 (1236 TP1 + 1854).

Dry runs against the private database after the fill
(`artifacts/tp24/dryrun-after-tp{1,2,4}.log`):

| arch | missing |
| --- | --- |
| TP1 (`llama3_dense`) | 0 / 1236 |
| TP2 (`llama3_dense_tp`, 2) | **19 / 1255**, all `attn_block.tp_allreduce` (`all_reduce`, nccl, 2 GPUs, nvlink) |
| TP4 (`llama3_dense_tp`, 4) | **19 / 1255**, all `attn_block.tp_allreduce` (4 GPUs) |

**Remaining: 38 all-reduce rows** (19 sizes from 4 KiB to 1 GiB at 2 GPUs
and at 4 GPUs). TP2 / TP4 prediction stays blocked until they exist.

## Gates before the fill

- Identity: `NVIDIA GB300`, UUID match, 152 SMs, exact ones-matmul; Triton
  probe PASS.
- Numerical correctness at a rank's shapes (`smoke_checks.py` with the head
  counts and GEMM shape overridden), against fp32 references, 0 mismatches
  in every case:
  - TP2: `F.linear` n 14336, k 4096 at m 64 / 2048 / 4096; RMSNorm; fa2
    prefill and decode at 16 query / 4 KV heads.
  - TP4: `F.linear` n 4096, k 3584; RMSNorm; fa2 prefill and decode at
    8 / 2 heads.
- vLLM shim at 4 and at 2 KV heads, 64 and 8192 tokens: caches bitwise equal
  to the torch reference, one `reshape_and_cache_flash_kernel` per launch
  seen by CUPTI, fp8 refused explicitly.
- Elementwise has no separate reference check here; the Triton probe and the
  TP1 smoke test cover the kernel, and only its width differs.

## Latency distributions (CUPTI kernel time)

| kind / backend | time µs min / p50 / p90 / max | peak | vs H200 p50 (rows) | vs B200 p50 (rows) | monotonic violations |
| --- | --- | --- | --- | --- | --- |
| single_gemm torch_linear | 4.7 / 50.0 / 365 / 4422 | 2124 TFLOPS | 0.39 (544) | 0.77 (68) | 0 |
| single_gemm torch | 4.8 / 50.5 / 364 / 4420 | 2118 TFLOPS | 0.40 (544) | – | 0 |
| flashinfer_attn_prefill fa2 | 8.1 / 170 / 2176 / 37751 | 466 TFLOPS | 0.87 (270) | – | 0 |
| flashinfer_attn_decode fa2 | 5.9 / 16.1 / 313 / 1272 | 7.1 TB/s | 0.76 (324) | – | 0 |
| kv_cache_append vllm_cuda | 2.7 / 13.9 / 51 / 312 | 1.1 TB/s | 0.68 (140) | 1.21 (70) | 0 |
| elementwise triton | 2.4 / 9.6 / 18 / 23 | 7.6 TB/s | 0.55 (32) | 1.02 (9) | 0 |

Ratios are GB300 time over the tracked row at identical arguments. They sit
where the TP1 rows did (GEMM 0.40, prefill 0.88, decode 0.71, elementwise
0.57 against H200). One thing to know, not an anomaly by the validator's
thresholds: the KV append is 21 % slower than B200 at the median over the 70
rows both have (range 0.79–1.39), while faster than H200; these are 3–300 µs
launches of a small copy kernel, and TP1 had no B200 rows to compare. The
GEMM peak, 2124 TFLOPS, is 85 % of the 2500 TFLOPS BF16 spec peak.

## Repeatability (`--fresh` re-measurement of a stride sample, not inserted)

| kind / backend | pairs | \|Δ\| p50 | \|Δ\| max | verdict |
| --- | --- | --- | --- | --- |
| single_gemm torch_linear | 34 | 0.22 % | 5.3 % | PASS |
| flashinfer_attn_prefill fa2 | 22 | 0.08 % | 1.2 % | PASS |
| flashinfer_attn_decode fa2 | 27 | 0.33 % | 2.1 % | PASS |
| kv_cache_append vllm_cuda | 20 | 1.2 % | 5.0 % | PASS |
| elementwise triton | 8 | 0.09 % | 0.9 % | PASS |

## GPU state and isolation

2743 samples at 2 Hz: utilization above 0 in 95 %, at least 50 % in 91 %;
power peak 1410 W, 1192 W mean while busy; SM clock 2070 MHz, down to
1342 MHz under the large GEMMs (software power cap, as in TP1); memory peak
15.5 GB; 65 °C peak. No compute process other than the profiler's own was on
any of the tray's GPUs at the start or at any of the 9 batch boundaries
(`isolation.log`), and none after. No step timed out; the budget guard never
triggered.

## Provenance

Three run records, all `profiler_git_hash = 2a295264ec584e522b22778cc7f3aa5a16587119`,
cuda 13.0, driver 610.57.04: `backend_version` `2.10.0+cu130` (single_gemm
torch), `3.6.0` (elementwise triton), NULL (torch_linear, fa2, vllm_cuda; the
shim's identity is `vllm-shim/BUILD_MANIFEST.tray12-20261110.json`).
Manifests before and after the run are in the run directory.

## Restart

`run_tp24_fill.sh` skips every batch that has a `.done` marker in
`artifacts/tp24/state/`; all six fills, both correctness gates and the five
repeat samples have one. Running it again would only redo the identity gates.
