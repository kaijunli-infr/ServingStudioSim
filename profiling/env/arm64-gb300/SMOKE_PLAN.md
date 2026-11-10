# GB300 smoke test plan (Phase 2B, needs approval before any step runs)

Goal: prove that the arm64 environment of this directory runs real kernels on
one GB300 before spending GPU time on the 1,533-row Llama 3.1 8B BF16 TP1
fill. Four representative kernels, one GPU, no database writes.

## Device and isolation

| item | value |
| --- | --- |
| host | rack 3 `pod4-gb300-3-tray12-f3` (10.13.84.3), the tray that holds this env and the toolchain |
| GPU | physical index 1 (`nvidia-smi -i 1`), re-checked idle immediately before start; GPU 0 carries a 123 MiB foreign context, so it is not used |
| exposure | `CUDA_VISIBLE_DEVICES=1` for every command, so the run sees exactly one device; no other GPU is opened |
| env | `source profiling/env/arm64-gb300/gb300.env` (venv, caches, `VIBESIM_PROFILE_RUNTIME=host`, DB copy) |
| writes | only `$GB300_ROOT/cache/*` (JIT caches), `$GB300_ROOT/artifacts/smoke/*`, `$GB300_ROOT/logs/*`; `--fresh` inserts no profile rows, so neither `profile-gb300.db` nor the tracked `profiling/profile.db` changes |
| pre-flight | `nvidia-smi --query-compute-apps=pid,process_name --format=csv` empty for GPU 1; `RmProfilingAdminOnly` must be 0 or CUPTI refuses non-root profiling; tray12 root disk and `/mnt/shared` free space recorded |
| abort | kill the `uv run` process; nothing persists but caches and artifacts, both deletable; confirm `nvidia-smi` shows no leftover process on GPU 1 |

## Steps

Every step runs from the repository root as
`source profiling/env/arm64-gb300/gb300.env; export CUDA_VISIBLE_DEVICES=1`,
and every `profiling run` uses `--gpu-name "NVIDIA GB300" --fresh --json
--output-dir "$GB300_ROOT/artifacts/smoke/<step>"`.

| # | step | command (abridged) | success criterion | expected time |
| --- | --- | --- | --- | --- |
| 0 | manifest before | `collect_manifest.py --out .../smoke/manifest-pre.json --run-config phase=2B-smoke gpu=tray12:1` | file written; `nvidia_smi.gpus` has 1 entry named `NVIDIA GB300`, cc 10.3 | 1 min |
| 1 | CUDA init | `python -c` : `get_device_name(0)`, `get_device_capability(0)`, `mem_get_info`, one 1024² bf16 matmul of ones | name `NVIDIA GB300`, capability `(10, 3)`, free memory > 270 GiB, matmul sum = 1024³ exactly, no error | 1 min |
| 2 | BF16 GEMM | `profiling run single_gemm --backend torch_linear` with `{"m":M,"n":6144,"k":4096,"dtype":"bf16"}` for M = 64, 2048, 4096 (Llama 8B QKV projection) | 3 results; M=2048 between 50 and 85 µs (1200–2050 TFLOPS); M=4096 within ±15 % of 2× M=2048; M=64 between 8 and 16 µs | 3 min, no JIT |
| 3 | RMSNorm | `profiling run rms_norm --backend flashinfer` with `{"m":M,"hidden":4096,"dtype":"bf16"}` for M = 64, 2048 | M=2048 between 7 and 11 µs (≥ 3 TB/s effective); M=64 between 2 and 4 µs | 3–6 min incl. first FlashInfer JIT |
| 4 | BF16 prefill attention | `profiling run flashinfer_attn_prefill --backend fa2` with `{"num_qo_heads":32,"num_kv_heads":8,"head_dim":128,"q_dtype":"bf16","kv_dtype":"bf16","o_dtype":"bf16","prefix_len":P,"append_len":2048}` for P = 0, 2048 | both complete; P=0 between 25 and 150 µs; P=2048 slower than P=0 and under 300 µs | 5–12 min incl. fa2 JIT |
| 5 | BF16 decode attention | `profiling run flashinfer_attn_decode --backend fa2` with the same heads and `{"batch_size":1,"total_tokens":4096}`, `{"batch_size":64,"total_tokens":38400}` | both complete; batch 1 under 20 µs; batch 64 under 120 µs; reported bandwidth > 0 | 5–12 min incl. fa2 decode JIT |
| 6 | FA3 probe (diagnostic) | step 4's first spec with `--backend fa3` | expected to **fail**: FlashInfer builds FA3 only for SM90. Pass = a clean, reported failure within 5 min and no hang; the error text is the evidence for the FP8-attention report | ≤ 5 min |
| 7 | manifest after, cleanup check | `collect_manifest.py --out .../smoke/manifest-post.json`; `nvidia-smi` compute-apps empty; `du -sh $GB300_ROOT/cache` | no leftover process; cache growth recorded | 1 min |

Reference points for the bounds (profile.db rows at the same shapes): B200
`torch_linear` GEMM 12.5 / 74.3 / 147.6 µs for M = 64 / 2048 / 4096; B200
RMSNorm 2.64 / 9.03 µs; H200 fa2 prefill (0, 2048) 144.8 µs; H200 fa2 decode
batch 1 × 4096 tokens 17.1 µs. GB300 has the B200's 8 TB/s HBM and a higher
dense BF16 peak (2500 vs 2250 TFLOPS), so memory-bound kernels should land
near B200 and GEMMs at or below B200 time.

Totals: about 25–45 min wall clock, under 10 min of GPU activity, one GPU.
Disk: JIT caches 0.5–3 GB under `$GB300_ROOT/cache`, artifacts under 50 MB.
Shared home at planning time: 25 TB free of 50 TB; `$GB300_ROOT` 8.2 GB.

## What the smoke test does not cover, and why it matters for the TP1 fill

- `kv_cache_append`'s only backend in the Llama arch is `vllm_cuda`, which runs
  in `vllm_env`: the vLLM fork's own venv (`alignment/profiler/vllm/.venv`),
  which does not exist on arm64. 70 of the 1,533 TP1 rows need it. Options,
  none taken yet: build the fork's env for arm64, or let Llama use the
  `torch` backend of `kv_cache_append` (an arch change).
- The Llama arch lists `fa2` and `fa3` for attention. On GB300 every `fa3`
  spec will fail (step 6 shows how). The batch path counts failures per
  backend, inserts the successful `fa2` rows and logs the `fa3` total failure
  as an error, so the fill proceeds, but 594 failing attempts cost time. The
  clean fix is `sm_targets={"sm_90a"}` on the fa3 backends, a compatibility
  change that is not approved yet.
- Only `default_env` kernels run on the host runtime here (`single_gemm`
  torch/torch_linear, `rms_norm` flashinfer, `elementwise` triton, the
  FlashInfer attention kinds). `sglang_env` and `vllm_env` backends are out of
  reach until their envs exist on arm64; the TP1 fill needs none but
  `kv_cache_append`.
