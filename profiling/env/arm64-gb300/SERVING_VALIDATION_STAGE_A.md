# Serving validation, Stage A: feasibility and preparation (CPU-only), 2026-11-11

Supersedes `SERVING_VALIDATION_PROPOSAL.md` where they differ. Nothing here ran
on a GPU and no model weights were downloaded.

## Findings in brief

1. **vLLM does not run NCCL for Llama TP2/TP4 on GB300 by default.** With the
   default optimisation level, each tensor-parallel all-reduce is fused with
   the following residual add and RMSNorm into one FlashInfer kernel
   (`allreduce_fusion`, pattern `kARResidualRMSNorm`) inside the compiled
   graph. Any all-reduce left unfused goes to FlashInfer's standalone
   all-reduce first. NCCL is the last fallback.
2. **The `nccl_graph` rows apply only to an NCCL-forced engine variant**, which
   four switches produce. They do not describe default serving. The kernel
   kind that matches the default path exists in the repository
   (`all_reduce_residual_rms_norm`, backends `flashinfer_trtllm` and
   `flashinfer_mnnvl`), but `llama3_dense_tp` does not use it and GB300 has
   no rows for it.
3. **Attention will most likely not be the FlashInfer FA2 kernels the
   simulator's rows measure.** On the SM100 family with FlashInfer cubins
   installed, vLLM selects TRT-LLM attention kernels for prefill and decode.
4. **Decode replays a full CUDA graph; prefill does not, or only partly.**
5. **The engine environment is built and imports on aarch64**, isolated from the
   profiling environment, which is unchanged.
6. **Two prerequisites are open:** access to the gated model (no credentials
   on the rack), and the alignment load generator (`req-frontend`), whose
   clone and build were blocked by this session's permission policy.

## 1. Engine revision

| item | value |
| --- | --- |
| fork | `github.com/serendipity-zk/vllm`, branch `servingstudio-alignment` |
| commit | `e1423a4d7496dc6535862ae11bab8cb4d87b155f` (2026-10-13), the commit `alignment/profiler/vllm` pins |
| upstream base | vLLM main `04730e82700d0c9b957cda188dd1bbc8073e1fcd`, 17 fork commits on top |
| fork changes | 31 files, Python only. None under `csrc/`, `cmake/`, `CMakeLists.txt` or `setup.py`, none in `vllm/distributed/device_communicators/`, `vllm/compilation/` or `vllm/config/{compilation,vllm}.py` |
| version string | `0.1.dev21921+ge1423a4d7` (base corresponds to `0.30.1rc1.dev97`) |

The one fork change under `vllm/config/` (`parallel.py`) does not touch
all-reduce or CUDA-graph settings. The selection logic below is therefore
upstream vLLM's at `04730e82`.

Source inspected in a separate full checkout:
`$GB300_ROOT/serving-validation/vllm-fork`. The sparse checkout used for the
KV shim was not changed.

## 2. All-reduce: implementation and selection

All references are to the fork at `e1423a4d`.

**Fused path (default for TP > 1 on GB300).**

- `enable_allreduce_rms_fusion` returns true when TP > 1, the platform is CUDA,
  FlashInfer is installed, and the device is in capability family 100 or is
  9.0. GB300 is 10.3, family 100. (`vllm/config/vllm.py:177-201`)
- The default optimisation level O2 sets
  `pass_config.fuse_allreduce_rms = enable_allreduce_rms_fusion`, as does O3.
  O0 and O1 set it false. (`vllm/config/vllm.py:300-331`, `:258`, `:281`;
  default level `O2` at `:439`)
- `AllReduceFusionPass` replaces all-reduce followed by `fused_add_rms_norm`
  with one `flashinfer_comm.allreduce_fusion` call, pattern
  `kARResidualRMSNorm`.
  (`vllm/compilation/passes/fusion/allreduce_rms_fusion.py:410-458`, pass at
  `:984`)
- Llama's decoder layer is exactly that shape: `post_attention_layernorm` and
  the next `input_layernorm` each take `(hidden_states, residual)`.
  (`vllm/model_executor/models/llama.py:322-328`)
- The fusion size limit at capability 103 is 64 MB for both 2 and 4 ranks,
  which is 8,192 tokens at hidden 4096 in bf16. Above it, the compile range
  splits and the all-reduce stays unfused.
  (`allreduce_rms_fusion.py:104-128`; `vllm/config/vllm.py:2627-2652`)
- Every planned workload stays at or below 2,048 tokens per iteration, so
  every all-reduce is expected to be fused.
- FlashInfer's all-reduce backend is `auto`, which selects `mnnvl` (NVLink
  multicast) and falls back to `trtllm` on a single node without multicast.
  (`vllm/envs.py:1752-1756`; `flashinfer_all_reduce.py:137-161`, `:207-212`)
  NCCL reported NVLS multicast available on tray12 in Phase 2A, so `mnnvl` is
  expected.

**Unfused all-reduce: dispatch order.**

`CudaCommunicator.all_reduce` tries backends in this order
(`cuda_communicator.py:335-408`):

| order | backend | on GB300 at TP2/TP4 by default |
| --- | --- | --- |
| 1 | NCCL symmetric memory | off, `VLLM_USE_NCCL_SYMM_MEM=0` (`envs.py:2016`; `pynccl_allocator.py:48-50`) |
| 2 | quick reduce | ROCm only |
| 3 | FlashInfer PCIe IPC | off by default (`envs.py:1897`) |
| 4 | **FlashInfer standalone** | **on** (`VLLM_ALLREDUCE_USE_FLASHINFER=1`, `envs.py:1891`). Takes 2-D bf16 tensors up to 64 MB at TP2 and 80 MB at TP4 (`flashinfer_all_reduce.py:370-401`, `:420-470`; `all_reduce_utils.py:87-92`) |
| 5 | AITER custom | ROCm only |
| 6 | vLLM custom all-reduce | on unless `--disable-custom-all-reduce`; below 4 MB at 10.3 when symmetric memory is on, else 8 MB (`custom_all_reduce.py:118-126`, `:205-220`, `:404-419`; `all_reduce_utils.py:44-49`) |
| 7 | torch symmetric memory | on (`VLLM_ALLREDUCE_USE_SYMM_MEM=1`); below 4 MB at TP2 and 32 MB at TP4 on 10.3 (`all_reduce_utils.py:71-76`) |
| 8 | PyNCCL | `ncclAllReduce` on the current stream (`pynccl.py:268-296`) |

**NCCL-forced variant.** All four switches together make every TP all-reduce
an unfused PyNCCL `ncclAllReduce`:

```
--disable-custom-all-reduce
--compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'
VLLM_ALLREDUCE_USE_FLASHINFER=0
VLLM_ALLREDUCE_USE_SYMM_MEM=0
```

This is the same NCCL kernel the `nccl` and `nccl_graph` rows time, with
three differences:

- PyNCCL calls NCCL directly rather than through `torch.distributed`.
- The engine environment links NCCL 2.29.7; the rows used 2.28.9.
  `VLLM_NCCL_SO_PATH` can point PyNCCL at the 2.28.9 library
  (`vllm/utils/nccl.py:18-36`).
- Decode replays it inside a full CUDA graph, together with the other kernels.

**Applicability of `nccl_graph`.**

| engine configuration | matching rows |
| --- | --- |
| default (fused FlashInfer) | none on GB300. The repository kind `all_reduce_residual_rms_norm` / `flashinfer_mnnvl` models it, but `llama3_dense_tp` charges a separate `all_reduce` and `rms_norm` |
| NCCL-forced, decode (graph-replayed) | `nccl_graph` |
| NCCL-forced, prefill of 2,048 tokens (eager, see section 3) | eager `nccl` is closer, since calls are issued from the host. Prefill all-reduces are 16 MiB, where the two backends agree within 7 % |

Modelling the default path for Llama TP would need the arch or its `backends`
to use the fused kind, plus GB300 rows for it. That is a simulator and
profiling change outside this approval.

## 3. CUDA graphs and compiled kernels

**Graph mode.** O2 sets `cudagraph_mode = FULL_AND_PIECEWISE`
(`vllm/config/vllm.py:315`; enum at `vllm/config/compilation.py:53-75`):

- uniform decode batches replay a full graph;
- mixed and prefill batches run piecewise, with attention outside the graph
  and the rest captured.

**Capture sizes.** `[1, 2, 4, 8, 16, …, 248]` in steps of 8, then steps of 16
up to the maximum (`vllm/config/vllm.py:2352-2362`, `:2404-2420`).

- The maximum is min(2 × `max_num_seqs`, 1,024 on family 100), capped by
  `max_num_batched_tokens`.
- `max_num_seqs` defaults to 1,024 on GPUs with at least 160 GB
  (`vllm/engine/arg_utils.py:2789-2799`).
- The alignment launcher sets `--max-num-batched-tokens` from `chunk_size`,
  default 2,048.

The cap is therefore 1,024 tokens. Expected per case:

| case | tokens per iteration | graph |
| --- | --- | --- |
| decode 1, 8, 64, 256 | 1, 8, 64, 256 | full graph replay |
| prefill 256 | 256 | piecewise, with attention eager |
| prefill 2048 | 2,048 | above the cap: no graph, eager |

**Attention.**

- On capability major 10, causal attention, vLLM ranks FLASHINFER first
  (`vllm/platforms/cuda.py:153-161`).
- Inside FlashInfer, TRT-LLM attention is used on family 100 when cubins are
  available (`vllm/utils/flashinfer.py:592-614`).
- `flashinfer-cubin==0.7.0` is installed, which `has_nvidia_artifactory`
  accepts as available (`:564-572`).
- Expected result: TRT-LLM-generated FMHA kernels for prefill and decode, where
  the simulator's rows are FlashInfer FA2. This is to be confirmed in the TP1
  smoke test.

**Norms and activations.**

- Under the default compile mode they are likely Inductor-generated Triton
  kernels rather than separate custom ops.
- `enable_sp` and `fuse_gemm_comms` are off, because `IS_DENSE = False`
  (`vllm/config/vllm.py:145-152`, `:306-307`).
- GEMMs go through torch, matching the `torch` and `torch_linear` rows.
- The KV write is vLLM's `reshape_and_cache_flash` family, matching
  `vllm_cuda`.

These are source readings. The trace decides.

## 4. Environment

**Isolated environment.**

| item | value |
| --- | --- |
| location | `$GB300_ROOT/serving-validation/venv`, Python 3.12.3, aarch64 |
| build script | `serving-validation/build_env.sh` |
| fingerprint | `serving-validation/env-fingerprint.json` (200 packages, sha256 `43672e8a…`) |
| torch | 2.13.0+cu130, CUDA 13.0, NCCL 2.29.7 |
| FlashInfer | 0.7.0 (python and cubin), `flashinfer.comm.allreduce_fusion` present |
| triton | 3.7.1 |
| transformers | 5.22.2 |
| nvtx | 0.2.15 |
| vLLM | the fork, editable, over the native extensions of the precompiled upstream wheel for `04730e82` |
| precompiled wheel | variant `cu130`, `manylinux_2_28_aarch64`, from `wheels.vllm.ai`; this is the install method the fork documents in `ALIGNMENT.md` |
| import check, CPU only | `vllm`, `_C_stable_libtorch`, `_moe_C_stable_libtorch`, `vllm_flash_attn._vllm_fa2_C`, the communicator modules: all import |
| `uv pip check` | one finding: `nvidia-cusparselt-cu13` reported as built for another platform; unused by this model, to be watched in the smoke test |
| disk | 14 GB (venv 13 GB, fork checkout 0.9 GB) |

**Source build.** A full build of the fork failed at one target: DeepGEMM's
extension needs `elfutils/libdwfl.h` (package `libdw-dev`), which is not
installed on tray12. Installing it needs root and is not needed with the
precompiled wheel. The failed build tree was deleted.

**Profiling environment unchanged.** The environment guard on
`$GB300_ROOT/venv` passes with the same package fingerprint as in Phase 2F
(torch 2.10.0+cu130, NCCL 2.28.9). `UV_NO_SYNC` was not involved, since the
new venv was built with `uv pip --python` and no project.

**Nsight Systems.** 2026.3.2 at `/usr/local/bin/nsys`, platform SBSA. The
kernel paranoid level is 4, so CPU sampling is unavailable. The pipeline
already runs with `--sample=none --cpuctxsw=none` and needs only CUDA and NVTX
tracing.

**Alignment pipeline requirements** (from `alignment/` and
`launcher/alignment*.py`):

- **Load generator not built.** The runner builds `req-frontend`
  (`alignment/load_generator/req-frontend`, Rust, `uw-syfi/request-factory`
  at `2cd7990c`) with `cargo build --release --bin session_runner` before each
  profile (`alignment/runner.py:398`). The submodule is not initialised, and
  my clone and build of it on tray12 was refused by this session's permission
  policy. **Unresolved.**
- **Submodule path forced.** The server's `PYTHONPATH` is forced to
  `alignment/profiler/vllm` (`alignment/profiler/vllm_server.py:33`, `:178`),
  and its git provenance is read there. That submodule is also uninitialised.
  For the GPU stages it should be initialised at `e1423a4d` in the GPU
  worktree, with `fork_python` set to this venv's interpreter. **Not done**,
  because it adds a third-party checkout to the profiling worktree.
- **Environment variables.** They reach the server only through
  `python_runtime.environment` in `profile.yaml`, which requires a non-empty
  `packages` list (pin `nvtx==0.2.15`), or through the parent environment,
  which `launch.json` does not record. The four NCCL-forcing settings
  therefore go into `server.extra_args` and `python_runtime.environment`.
- **Device order.** `CUDA_DEVICE_ORDER` is not set by the launcher; the
  configs must set it to keep indices equal to the profiling runs.

## 5. Model

| item | value |
| --- | --- |
| repository | `meta-llama/Llama-3.1-8B-Instruct`, gated (`manual`) |
| revision | `0e9e39f249a16976918f6564b8830bc894c89659` (2024-09-25) |
| weights | 4 safetensors files, 16,060,556,376 bytes |
| tokenizer | same revision: `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` |
| access from tray12 | **none**. No Hugging Face credential is configured for this user, and `auth_check` returns `GatedRepoError` (`serving-validation/hf-access.json`) |
| local copy | none in the shared Hugging Face caches |

The check reads metadata only. It never reads, prints or stores a token
(`serving-validation/hf_access_check.py`). The architecture matches
`model/config/llama3_8b.json`.

## 6. Comparing GPU iteration latency

**Iteration boundary.** An iteration is the fork's NVTX range
`vllm_iteration(N): forward` on each rank (`alignment/nsys/parse.py:61`).

- Kernels belong to it by CUDA correlation ID from launches on the range's
  thread, not by time overlap (`parse.py:579-654`).
- Graph-replayed kernels are resolved per node, because capture uses
  `--cuda-graph-trace=node`.
- Its batch comes from the fork's `VibeSimAlignmentIteration` record
  (schema 4: `prefill_chunk_pairs`, `decode_kv_lens`, `decode_requests`, …).

**Measured quantities per iteration** (analyzer `alignment_iteration`):

| quantity | meaning |
| --- | --- |
| `measured_ms` | critical-path GPU time across ranks: busy time plus collective time |
| `measured_kernel_sum_ms` | sum of kernel durations on one rank |
| `wall_ms` | first kernel to last kernel |
| `measured_gpu_cycle_ms` | this iteration's first kernel to the next iteration's first kernel |

**Predicted quantity.** `timing-predict` on the measured batch of the same
iteration: the CostTree total, which is the kernel sum at
`gpu_time_multiplier` 1.0.

**Errors.**

- Per iteration: `(predicted − measured_ms) / measured_ms`.
- Per case, over its selected iterations: signed error `Σ(sim − meas) / Σ meas`
  and absolute error `Σ|sim − meas| / Σ meas`.
- Per kernel role, with mapped labels, so a wrong attention backend or a
  missing fused all-reduce shows up by name.

**Kept separate.** Host-facing figures are reported, never mixed into the
error: `wall_ms / measured_ms`, and `measured_gpu_cycle_ms / measured_ms`,
which is the analyzer's recommended `gpu_time_multiplier`. Client-facing time
to first token, time per output token and throughput come from a separate pass
without Nsight Systems and are compared with the predicted iteration rate,
never with kernel time.

## 7. Workloads and how prefill and decode are isolated

**Traces.** `serving-validation/make_traces.py` writes one `independent` trace
per case (`id,input_len,output_len,arrival_time`).

- Each case's batch is repeated 5 times, 1 s apart.
- Requests use the native token endpoint (`backend: vllm_tokens`), with
  `arrival_mode: saturated`, `max_concurrency` equal to the batch size, and
  prefix caching off.

| case | batch | prompt tokens | output tokens | iterations selected |
| --- | --- | --- | --- | --- |
| prefill 256 | 1 | 256 | 1 | kind `prefill`, `prefill_chunk_pairs = [[0, 256]]` |
| prefill 2048 | 1 | 2,048 | 1 | kind `prefill`, `[[0, 2048]]` (one chunk at `chunk_size` 2,048) |
| decode 1 / 8 / 64 / 256 | N | 568 | 64 | kind `decode` with `decode_requests == N` |

- **Decode.** After the prompts are prefilled, in chunks of at most 2,048
  tokens and possibly mixed with first decode steps, each batch runs about 63
  pure decode steps with cached lengths 568 to 631. Only those are selected,
  so the decode statistic contains no prefill work.
- **Prefill.** Each prefill request is alone in the engine, so the selected
  iteration is pure prefill.
- **Prediction shapes.** Predictions use each iteration's measured shapes, not
  the averaged case: `alignment timing-predict` builds `decode_kv_lens` per
  request.
- **Graph-mode check.** Whether each selected iteration replayed a CUDA graph
  is checked against section 3 from the trace.

## 8. Revised GPU experiment

Each step needs its own approval and ends with a report.

**Step 0, CPU-only prerequisites:**

1. You arrange read access to the gated model for the session's user on the
   rack. The credential never passes through me.
2. Then download the pinned revision: 16.1 GB to
   `$GB300_ROOT/serving-validation/hf`.
3. Permission to clone and build `req-frontend` at `2cd7990c`.
4. Initialise `alignment/profiler/vllm` at `e1423a4d` in the GPU worktree.

**Step 1, TP1 smoke test.** tray12 GPU 1.

- One launch, prefill 256 and decode 8, 3 batches each, under Nsight Systems.
- Establish: the engine starts; the attention, norm, activation, GEMM and KV
  kernels by name; graph mode per iteration; iteration records present; label
  coverage.
- No error figure is trusted yet.
- About 15 minutes of GPU time.

**Step 2, TP2 collective identification.** GPUs 2 and 3.

- Two launches: default, and NCCL-forced (section 2).
- Cases: decode 8, decode 64, prefill 256.
- Establish per launch: which all-reduce kernel runs and how often per layer;
  whether it is fused with RMSNorm; its GPU time per call at 16 KiB, 128 KiB
  and 2 MiB.
- Compare the NCCL-forced per-call time with `nccl_graph` and eager `nccl`.
- About 30 minutes.

**Step 3, TP4 benchmark.** GPUs 0 to 3, only after step 2.

- All six cases.
- Under Nsight Systems for the iteration comparison, for the variant(s)
  step 2 shows to be comparable.
- Then a pass without Nsight Systems for client latency and throughput.
- TP1 and TP2 get the same full pass at that point.
- About 60 minutes for three TP sizes.

**Resources.**

| | estimate |
| --- | --- |
| model load, compile and capture per launch | 3 to 6 min (compile cache in `$GB300_ROOT/serving-validation/cache`) |
| GPU time, steps 1 to 3 | about 105 min, limit 120 |
| disk | weights 16 GB + environment 14 GB + traces 10 to 20 GB + caches 5 GB; limit 60 GB |

**Isolation and stop conditions.**

- Before each launch: expected UUIDs, utilisation at most 5 %, no compute
  process on any tray12 GPU.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`, rank r on the r-th listed GPU.
- Timeouts:
  - 15 min to server readiness;
  - per case, 10 min;
  - Nsight `stop_profile` 10 min (the launcher's own limits);
  - an outer `timeout` on each launch.
- Stop on any of:
  - an environment guard or UUID mismatch;
  - a foreign GPU process;
  - startup failure;
  - a request error or wrong token count;
  - unexpected kernels in step 1;
  - a timeout;
  - the budget.
- Cleanup kills only the engine's process group started by the run.
- No profile database, simulator file or preset is written.

**What the result can show.** With the current arch, prediction error is
meaningful for the NCCL-forced variant. For the default variant, the error will
also contain the fused all-reduce plus RMSNorm and the attention kernel
mismatch. Step 2 quantifies both and tells whether to propose GB300 rows for
`all_reduce_residual_rms_norm` and for TRT-LLM attention.

## Artifacts

- In `serving-validation/`: `build_env.sh`, `env-fingerprint.json`,
  `hf_access_check.py`, `hf-access.json`, `make_traces.py`.
- On the rack, `$GB300_ROOT/serving-validation/`: `vllm-fork/` (full checkout
  at `e1423a4d`), `venv/`, build and install logs, `profiling-env-guard.json`.
