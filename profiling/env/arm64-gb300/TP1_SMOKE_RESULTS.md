# TP1 serving smoke test on GB300, 2026-11-11

Real vLLM serving of Llama 3.1 8B Instruct (BF16) on tray12 GPU 1 under the
alignment pipeline and Nsight Systems: 3 prefills of 256 tokens and 3 batches
of 8 decodes. **It completed.** All 27 requests returned exactly the requested
token counts. All 201 iterations have their engine record and a traced
forward range. Every kernel is identified.

The measurement infrastructure works for a larger TP1 experiment. No
simulator prediction error is computed or claimed from this run.

## Versions

| item | value |
| --- | --- |
| engine | vLLM fork `e1423a4d` (`v0.1.dev21921+ge1423a4d7`), native extensions from the precompiled upstream wheel of `04730e82` (cu130, aarch64) |
| torch | 2.13.0+cu130 |
| CUDA | runtime 13.0 (torch build); driver 610.57.04 |
| NCCL | 2.29.7 linked by torch; no collective at TP1 |
| FlashInfer | 0.7.0 |
| Nsight Systems | 2026.3.2 (`/opt/nvidia/nsight-systems/2026.3.2/target-linux-sbsa-armv8/nsys`) |
| model | `meta-llama/Llama-3.1-8B-Instruct` at `0e9e39f249a16976918f6564b8830bc894c89659`, local snapshot, offline |
| tokenizer | same revision |
| environment | serving venv fingerprint `43672e8a…` before and after, equal to the committed one |

**Model integrity** (`tp1-smoke/model-verify.json`):

- 4 weight shards, 16,060,556,376 bytes, each matching the hub's LFS sha256.
- `config.json`, `generation_config.json`, the safetensors index and the 3
  tokenizer files each match the hub's git blob id.
- Only these 10 files were downloaded, to
  `$GB300_ROOT/serving-validation/hf/hub`. No other cache was touched.
- Access was verified first with `hf_access_check.py` under the run user's
  stored credential, which I did not read.

## Attempts and GPU time

| attempt | start (UTC) | outcome | GPU 1 occupied |
| --- | --- | --- | --- |
| 1 | 09:07:21 | launcher exited after 1 s: `cargo` not on the non-interactive `PATH`; the engine never started | 0 s |
| 2 | 09:09:07 | the guard stopped the run at 30 s: it took the run's own engine for a foreign process; I then stopped that run's processes by hand | about 115 s (09:09:36 to the end of cleanup, about 09:11:30) |
| 3 | 09:13:51 | completed, launcher rc 0 at 09:18:03 | 227 s (09:14:11 to 09:17:58, memory above 1 GiB) |
| total | | | **about 342 s of the 900 s allowed** |

**Attempt 2 in detail.**

- The launcher starts Nsight Systems and vLLM in new sessions. The wrapper
  counted only its own session as the run, so it flagged the engine at 30 s
  and left that engine on GPU 1.
- I traced the leftover processes' ancestry: `EngineCore` ← `vllm serve` ←
  `nsys-launcher` ← `nsys profile`, all from the run.
- I stopped exactly those process groups with SIGINT. All exited; GPU 1 went
  back to 93 MiB.

**Fix before attempt 3** (`serving-validation/proc_own.sh`):

- A process belongs to the run when it descends from the wrapper, recorded
  with its start time so it stays identified after re-parenting. The detached
  Nsight agent is recognised by its session name.
- Cleanup signals the launcher first, then the remaining recorded processes.
- Tested on CPU with dummy process trees: a new-session grandchild, an
  orphaned grandchild and a detached agent were all identified and stopped,
  and an unrelated process was left alone.
- Attempt 3 ran under a 600 s limit so the total stayed under 900 s.

## Guards (attempt 3)

| guard | result |
| --- | --- |
| GPU 1 UUID | `GPU-9574ced0-46be-d5b7-0098-01fa1e84dccb`, as expected |
| GPU 1 before | 0 % utilisation, 93 MiB, no compute process |
| rest of the tray | GPU 2: another user's SGLang server (`sglang::scheduler` and a `python3`), recorded, not touched; GPUs 0 and 3 idle |
| port 8000 | free |
| serving environment | fingerprint unchanged before and after |
| profiling environment | guard passed before and after, same fingerprint |
| model snapshot | complete, sizes as on the hub |
| profile databases | sha256 unchanged (tracked `3ec95dcd…`, private `9474a224…`) |
| isolation during the run | no foreign process on GPU 1; 85 processes recorded as the run's own |
| cleanup | the launcher shut everything down itself; no process of the run left; GPU 1 back to 93 MiB; port 8000 free |

## Requests and token counts

- **Requests:** 27 of 27 succeeded, 0 failed: 3 with 256 prompt tokens and 1
  output token, 24 with 568 and 64.
- **Per-request check:** for every request, the prompt tokens the server
  reported equal the trace's input length, the output equals the target, and
  the finish reason is `length`. Output tokens: 1,539 target, 1,539 actual,
  0 mismatches.
- **Content:** token-in, token-out with `ignore_eos`, so the content is not
  judged. The check is exact lengths and successful generation.
- **Engine log:** no error. The only traceback is a warning from the API
  server, "Tokenizer not available when `skip_tokenizer_init=True`", expected
  with `--tokens-only`.

## Iteration records

- 201 `VibeSimAlignmentIteration` records (schema 4), 201 traced
  `vllm_iteration(N): forward` ranges, and 201 parsed iterations.
- 87,036 kernels were attached to them by correlation ID.

| iteration shape | count | CUDA-graph replay of forward kernels |
| --- | --- | --- |
| prefill 256 tokens | 3 | piecewise: 73 % of kernels from a graph, attention outside |
| prefill 568 tokens (first request of a decode batch) | 3 | piecewise, 73 % |
| mixed: 2,047 prefill + 1 decode | 3 | none (above the 1,024-token capture limit) |
| mixed: 1,929 prefill + 4 decodes | 3 | none |
| decode, 4 and 7 requests | 3 + 3 | full graph, 100 % |
| **decode, 8 requests** | **183** | **full graph, 100 %** |

- **Decode batches:** the 8 requests of a batch reach the engine over the
  first few iterations, so each batch starts with a 568-token prefill and two
  mixed chunks before its 61 steps of 8 pure decodes. Selection by
  `decode_requests == 8` isolates those steps.
- **Count check:** 3 + 3 × (1 + 1 + 1 + 1 + 1 + 61) = 201.
- **Graph settings:** the engine log shows `FULL_AND_PIECEWISE`,
  `max_cudagraph_capture_size` 1,024 and `VLLM_COMPILE`, as Stage A read
  from source. `fuse_allreduce_rms` is false at TP1, since it applies to
  TP > 1 only.

## Kernels and their classes

`serving-validation/kernel_classes.py` on the trace (`tp1-smoke/kernel_classes.json`):

| class | kernels | GPU ms | from a graph | names |
| --- | --- | --- | --- | --- |
| gemm_cublas | 38,025 | 638.3 | 97 % | `nvjet_sm103_tst_*`, `cublasLt::splitKreduce_kernel` |
| inductor_triton | 25,929 | 67.6 | 97 % | `triton_red_fused_fused_add_rms_norm_*`, `triton_poi_fused_mul_silu_slice_1`, `triton_red_fused__to_copy_embedding_rms_norm_0`, `triton_poi_fused_*` |
| attention_trtllm | 6,624 | 62.7 | 91 % | decode `fmhaSm100fKernel_…MultiCtasKvCgaVarSeqQ8Kv256StaticGroupedSwapsAbForGen`; prefill `fmhaSm103aKernel_…VarSeqQ256Kv128PersistentContext` |
| kv_append | 6,432 | 16.0 | 94 % | `vllm::reshape_and_cache_flash_kernel<bf16, bf16, kAuto>` |
| torch_aten | 8,523 | 9.3 | 71 % | `at::native::*` fill, index, arange, argmax, gather; `cub` scan |
| vllm_runtime_triton | 1,503 | 5.2 | 0 % | `_compute_slot_mappings_kernel`, `_gumbel_sample_kernel`, `_gather_block_tables_kernel` and 7 more |
| unknown | 0 | | | |

**What this shows against the simulator's TP1 kernel choices** (no
error figure):

- GEMMs are cuBLAS, as the `torch` / `torch_linear` rows assume.
- The KV write is the kernel of the `vllm_cuda` rows.
- Attention runs TRT-LLM FMHA kernels, not the FlashInfer FA2 kernels behind
  the simulator's `fa2` rows. vLLM logs "Using TRTLLM prefill attention
  (auto-detected)".
- RMSNorm, residual add, SiLU-and-mul and the embedding are fused by
  torch.compile into Triton kernels. The simulator's separate `rms_norm`
  (flashinfer) and `elementwise` (triton) rows do not describe them.
- A fill kernel per layer (`FillFunctor<unsigned char>`, about 6,600 calls)
  and the model runner's bookkeeping kernels are not in the simulator.

**Classifier corrections in this commit:**

- New classes `gemm_cublas`, `inductor_triton`, `vllm_runtime_triton` and
  `torch_aten`, which leave no unknown name in this trace.
- Real names from the trace added to the self-test, which now passes 22 of
  22.
- No communication sample could be checked, since TP1 runs no collective.

## Resources

| item | value |
| --- | --- |
| GPU time | about 342 s in total (227 s in the completed run) |
| peak GPU 1 memory | 243,345 MiB: weights 15.0 GiB plus the KV cache at `gpu_memory_utilization` 0.85 |
| trace | `.nsys-rep` 8.2 MB, `.sqlite` 26.4 MB; run directory 39 MB in total (budget 1 GB) |
| model load / compile | weights loaded in 2 s; torch.compile 0.26 s from the cache the aborted attempt 2 had filled; piecewise graph capture about 60 s |
| launcher wall time | 4 min 10 s |
| disk, serving workspace | 32 GB (environment 13, model 15, corpus 1, rest builds, caches and runs) |

## Is TP1 ready for a larger experiment?

**Yes, for the infrastructure:** engine, model, load generator, Nsight
capture, iteration records, kernel attribution, graph-mode detection and the
guard all worked. Three harness defects were found and fixed:

- `cargo` missing from `PATH`;
- session-based process ownership;
- warm-up requests removed from the smoke config, to keep the request set
  exact.

**Before predictions are compared**, the simulator side needs decisions, since
three kernel families differ from what `llama3_dense` charges:

- TRT-LLM attention instead of FA2;
- Inductor-fused norm and activation;
- per-layer fills.

Only GEMM and KV append are like for like. A TP1 comparison is therefore
mainly a test of those mismatches. The per-kernel table will show them by
name.

## Artifacts

- `$GB300_ROOT/serving-validation/runs/tp1-smoke/`: Nsight report and sqlite,
  parsed iterations, kernel table, kernel sequences, host timeline, engine and
  load-generator logs.
- `$GB300_ROOT/serving-validation/guard/tp1-smoke-20261111-091351/`: guard
  logs, GPU samples, fingerprints, checksums. The earlier attempts' directories
  sit beside it.
- Committed copies in `serving-validation/tp1-smoke/`: `kernel_classes.json`,
  `summary.json`, `profile_result.json`, `model-verify.json`,
  `guard-steps.log`.
