# GB300 TP1 rows merged into the tracked profile database (2026-11-10)

The 1236 `NVIDIA GB300` rows of the Llama 3.1 8B BF16 TP1 dataset
(`TP1_RESULTS.md`) are now in `profiling/profile.db` on the fork branch
`gb300-support`, commit `251258c8`. Nothing else in the database changed.
Everything below ran on rack 3 tray12 with `CUDA_VISIBLE_DEVICES=` (no GPU).
Working files: `$GB300_ROOT/artifacts/merge/`.

## Files and checksums

| file | bytes | sha256 |
| --- | --- | --- |
| tracked `profiling/profile.db` before (commit `a28e8f4d` and earlier) | 74,006,528 | `e2d80ff89da29c5a4e85288009724969a198c0bd0ddd3b47e970a24f24b41431` |
| backup of it, `$GB300_ROOT/artifacts/merge/profile.db.pre-merge-backup` | 74,006,528 | same |
| private GB300 database, `$GB300_ROOT/db/profile-gb300.db` (unchanged, kept) | 74,207,232 | `f2dae6e9e2b40113869491742ddb5ec6935949f521e91fc0d0903a2a1700365b` |
| tracked `profiling/profile.db` after (commit `251258c8`) | 74,207,232 | `41c9cbe58d571e61fc7b1a7a0e86ae5e493aef0487001d944a270afc87bd82af` |

The merged file and the private file hold the same rows but are not
byte-identical: the merge assigns surrogate `id`s in its own order.
`git checkout a28e8f4d -- profiling/profile.db` restores the previous
database.

## How

The profiler's own merge, with the tracked database as the left input:

```bash
python -m profiling.cli merge-db profiling/profile.db "$GB300_ROOT/db/profile-gb300.db" \
    --output "$GB300_ROOT/artifacts/merge/profile-merged.db" --json
```

It merges by each table's unique `(gpu_name, backend, …)` key and publishes
the output only when schemas agree and no key maps to differing data. Report
(`merge/merge-report.json`): 108 tables, 185,027 rows on the left, 186,267 on
the right, **1240 inserted** (1236 measurement rows + 4 `_profile_run`
records), 185,027 duplicates, **0 conflicts**. The output was then copied over
`profiling/profile.db` and committed alone.

## Row counts

Measurement rows per GPU (all tables with a `gpu_name`), from
`merge_check.py` (`merge/merge-check.json`):

| GPU | before | after |
| --- | --- | --- |
| NVIDIA B200 | 129,826 | 129,826 |
| NVIDIA H200 | 54,857 | 54,857 |
| NVIDIA GB300 | 0 | 1,236 |

| table | before | after | added (all GB300) | backend |
| --- | --- | --- | --- | --- |
| single_gemm | 15,677 | 16,357 | 680 | torch 340, torch_linear 340 |
| rms_norm | 756 | 819 | 63 | flashinfer |
| elementwise | 9,154 | 9,280 | 126 | triton |
| flashinfer_attn_prefill | 2,835 | 2,970 | 135 | fa2 |
| flashinfer_attn_decode | 3,888 | 4,050 | 162 | fa2 |
| kv_cache_append | 1,131 | 1,201 | 70 | vllm_cuda |
| `_profile_run` | 344 | 348 | 4 | – |

The public API's own B200 count is 128,274, lower than the database total
above; it is the same before and after the merge, and the difference was not
investigated.

## Checks (`merge_check.py`, verdict PASS)

- **B200 and H200 untouched:** every row of the previous database is present
  in the merged one with identical content in every column except the
  surrogate `id` (multiset comparison per table, 108 tables); 0 lost or
  changed.
- **Only GB300 added:** all 1236 added measurement rows have
  `gpu_name = 'NVIDIA GB300'`; no row was added to any table without a GPU.
- **No duplicate measurement keys:** 0 repeated `(gpu_name, backend, …)` keys
  in every table (also enforced by the unique index).
- **No missing GB300 configurations:** the simulator's dry run over the merged
  tracked database reports `0 / 1236 specs missing`.
- **Valid latencies:** every added `time_ms` is finite and positive (2.3 µs to
  74.7 ms), 0 rows flagged `is_outlier`, every added row cites a run record,
  and the 4 added run records are exactly the ones cited.
- `PRAGMA integrity_check` is `ok` on both files.

## Regression tests on the merged commit (`251258c8`, tray12, CPU-only torch)

| suite | result |
| --- | --- |
| `cargo build --release -p simulator -p analyzer --bin analyze` | ok |
| `cargo test -p simulator --lib` | **1583 passed**, 0 failed, 8 ignored |
| `pytest -m "not gpu and not agent and not bench and not presets" -n 16` | **4989 passed**, 36 skipped, 8 failed (all environmental, below) |
| `pytest -m presets` (every public preset built through the binary) | **9 passed** |

The 8 failures need a CUDA build of torch, which the repository's lock does
not give on aarch64 (`Torch not compiled with CUDA enabled`, or an energy
counter that reads 0 without a GPU): 4 in `test_l1_single_gemm` (energy), 2
in `test_dsa_sparse_mla_attention`, 1 in `test_gdn_spec_decode`, 1 in
`test_kda_chunk_prefill_flashinfer`. Seven fail identically on untouched
upstream `5795c5b2` in this venv; the eighth
(`test_trtllm_fp8_rope_dim_selects_the_accepted_layout[overrides0]`) fails
identically on a pristine export of the pre-merge commit `74c5e7d7` with the
old database, and this branch touches neither that test nor its runner. None
reads GB300 rows. The suites cover the GPU catalog (`gpu/spec.json`
resolution), the profiling registry and database layer including
`profiling.db.merge`, backend capability filtering (the fa3 gate tests), the
launcher and timing prediction, and the public API.

## Prediction parity (merged vs private database)

`python -m launcher timing-predict presets/predict_llama3_8b_gb300_iter.json
--no-gpu`, once against each database, outputs kept under
`$GB300_ROOT/artifacts/merge/predict-{private,merged}`:

- `payloads/iter_breakdown.json`, `raw/cost_manifest/worker_predict_0.json`,
  `raw/kernel_grid_peaks.json`, `raw/achievable_peaks.json`, `raw/params.json`,
  `reports/kernel_time_share_report.json`,
  `payloads/kernel_time_share_composition.json`: **byte-identical**.
- `raw/cost_log/worker_predict_0.parquet`: tables equal.
- Totals: prefill iteration (2048 tokens) 21,180 µs; decode iteration
  (64 requests, 38,400 cached tokens) 4,169 µs, in both.

## Public API over the merged database (local, nothing deployed)

`public_api_check.py --db profiling/profile.db`: health ok at `251258c8`;
`NVIDIA GB300` in the catalog with **1236 rows**; peaks from the
`B300-SXM-288GB` catalog entry: bf16 and fp16 2500 TFLOPS dense, fp8 5000,
nvfp4 / mxfp4 13,125, fp32 80, HBM 8000 GB/s, NVLink 5.0 900 GB/s one way
(1800 both). Per kind for GB300: single_gemm 680 (torch, torch_linear),
rms_norm 63 (flashinfer), elementwise 126 (triton), flashinfer_attn_prefill
135 (fa2), flashinfer_attn_decode 162 (fa2), kv_cache_append 70 (vllm_cuda).
B200 128,274 and H200 54,857, as before.

## Measurement provenance, as recorded (not rewritten)

The four run records came over unchanged:

| rows | `profiler_git_hash` | cuda | driver | `backend_version` |
| --- | --- | --- | --- | --- |
| first batch (1166 rows), 3 run records | `e897db56971c5f4f0813fb916475ac6d8477e2ce` (clean worktree) | 13.0 | 610.57.04 | `2.10.0+cu130` (single_gemm torch, 340), `3.6.0` (elementwise triton, 126), NULL (single_gemm torch_linear 340, rms_norm 63, attention fa2 297) |
| kv_cache_append / vllm_cuda (70 rows) | `3b17315d3b77d6e4acdc8f647ca8162be5377815-dirty` | 13.0 | 610.57.04 | NULL |

**The `-dirty` suffix is accurate and stays.** When the 70 kv_cache_append
rows were measured (2026-11-10 10:04 UTC, run
`run-20261110-100407-kv`), the profiling worktree `ServingStudioSim-gb300`
was at commit `3b17315d` with two files modified in place, copied there
while fixing the driver between attempts:

- `profiling/env/arm64-gb300/run_kv_fill.sh` (the orchestration script),
- `profiling/env/arm64-gb300/vllm-shim/probe_shim.py` (the pre-fill probe).

What is and is not established about that state:

- The run's own manifest (`manifest-pre.json`) records `commit 3b17315d`,
  `dirty: true`; it does not record a diff. The modified bytes were not
  archived, and the worktree was reset after the changes were committed.
- Before the reset, `git diff` of the worktree against the follow-up commits
  showed `run_kv_fill.sh` identical to its committed version (`acc821e7`) and
  `probe_shim.py` differing from it only in formatting (`git diff -w` listed
  line-wrapping changes only). That comparison was made by eye from the diff
  output, not stored.
- Neither file is on the measurement path of a row. The rows were produced by
  `python -m profiling.cli run kv_cache_append --backend vllm_cuda`, whose
  profiler, runner and timer code were the committed `3b17315d` files, and by
  the shim library whose inputs are hashed in
  `vllm-shim/BUILD_MANIFEST.tray12-20261110.json` (the installed
  `build_shim.py` hash `7fa04931…` equals the `3b17315d` file; the library is
  `ca75929f…`). The two modified files decide whether the fill starts and
  which gates run before it.
- `backend_version` is NULL for these rows because the shim is not an
  installed distribution; the kernel's identity is the fork commit
  `e1423a4d` and the `cache_kernels.cu` hash in the build manifest.

Anyone who wants rows with a clean hash can re-measure the 70 specs from a
clean worktree (10 s of GPU time, needs a new approval); the merge would then
report 70 conflicts on `run_key` and the timings, which is the mechanism
working as designed. That has not been done.
