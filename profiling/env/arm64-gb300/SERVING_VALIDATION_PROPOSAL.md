# Proposal: validate GB300 predictions against real Llama 3.1 8B serving

Prepared 2026-11-11 on CPU only (repository and rack inspection, no GPU, no
download, no engine start). Nothing here runs without a new approval.

**Question.** Do ServingStudio Sim's TP1, TP2 and TP4 iteration predictions
for Llama 3.1 8B BF16 on GB300 match a real serving engine, and which
all-reduce backend (`nccl` eager or `nccl_graph`) describes what the engine
does?

**What is not known yet and decides the design.** Whether the engine replays
CUDA graphs for the measured steps, and which collective it actually runs
under TP. The local copy of the pinned engine source is a sparse checkout
(C++ kernels only), so neither could be read from source in this phase. Stage
1 below establishes both from a trace before any number is compared.

## 1. Serving engine

| item | proposal |
| --- | --- |
| engine | vLLM, the repository's instrumented fork (`alignment/profiler/vllm` submodule) |
| repository, branch | `github.com/serendipity-zk/vllm`, `servingstudio-alignment` |
| revision | `e1423a4d7496dc6535862ae11bab8cb4d87b155f` (2026-10-13), the commit this repository pins and the one whose `cache_kernels.cu` the GB300 KV-append rows were built from |
| why this one | it emits the `vllm_iteration(N)` NVTX scopes and `VibeSimAlignmentIteration` records that the repository's alignment pipeline needs to cut a trace into iterations |
| harness | `python -m launcher alignment profile` (engine lifecycle, load generator, Nsight Systems capture), then `alignment timing-predict` and `alignment analyze` |
| profiler | Nsight Systems 2026.3.2 at `/usr/local/bin/nsys` on tray12, `capture_mode: cuda_profiler_api` |

Open prerequisite: the fork needs its own Python environment on aarch64 with a
CUDA 13 torch. Whether it builds there is unverified; the official aarch64 vLLM
wheels were rejected earlier for their torch pins. Building it is CPU-only
work on shared storage (estimate 10 to 15 GB, 30 to 60 minutes) and is the
first step to approve.

Fallback if the fork cannot be built: a stock aarch64 `vllm/vllm-openai`
image. tray12 has such images, but they belong to other sessions and lack the
instrumentation, so only client-side latency and throughput could be compared,
not iterations or kernels. That would be a weaker experiment and a separate
decision.

## 2. Model and tokenizer

| item | proposal |
| --- | --- |
| model | `meta-llama/Llama-3.1-8B-Instruct`, BF16 safetensors, no quantisation |
| match with the simulator | `model/config/llama3_8b.json` (32 layers, hidden 4096, 32/8 heads, intermediate 14336, vocab 128256, `llama3` RoPE scaling) |
| revision | the snapshot hash Hugging Face resolves at download; recorded in the run manifest and passed as `--revision` on every launch |
| tokenizer | from the same repository and revision |
| size | about 16 GB |

The model repository is gated and is not in any cache on the rack. Access needs
your Hugging Face account and licence acceptance. The token is yours to
provide to the session's environment; I will not print or copy it.

## 3. Configurations

| | TP1 | TP2 | TP4 |
| --- | --- | --- | --- |
| tray12 GPUs | 1 | 2, 3 | 0, 1, 2, 3 |
| `--tensor-parallel-size` | 1 | 2 | 4 |
| simulator arch | `llama3_dense` | `llama3_dense_tp`, 2 | `llama3_dense_tp`, 4 |
| prediction modes compared | default | eager `nccl`, `nccl_graph` | eager `nccl`, `nccl_graph` |

Common settings: `--dtype bfloat16`, one replica, prefix caching off, no
speculative decoding, default scheduler, `--max-model-len 8192`, the same
GPU order (`CUDA_DEVICE_ORDER=PCI_BUS_ID`) and rank-to-GPU mapping as the
profiling runs. Attention backend and KV layout are whatever the fork selects
by default; stage 1 checks that they are the kernels the simulator's arch
assumes (FlashInfer attention, `reshape_and_cache_flash`).

## 4. CUDA graph and collective behaviour

To record per configuration, from the trace and the server log, before any
comparison:

1. **Graph replay.** Whether decode steps and prefill steps run from a CUDA
   graph, and for which batch sizes (the engine's capture-size list).
2. **Which collective runs.** The kernel names of the TP all-reduce at each
   message size: NCCL (`ncclDevKernel_AllReduce_*`) or an engine-specific
   all-reduce. To my knowledge vLLM prefers its own shared-memory all-reduce
   for small tensors on a full NVLink mesh and falls back to NCCL otherwise;
   this is unverified for the pinned revision and for GB300.
3. **Per-collective GPU time** at 8 KiB, 64 KiB, 512 KiB, 2 MiB and 16 MiB,
   against the stored `nccl` and `nccl_graph` rows.

Two engine variants, if the engine offers the switch: its default collective
path, and NCCL forced (vLLM's option to disable its custom all-reduce). The
first is what production does; the second is what the simulator's rows model.
If the default path is not NCCL, the finding is that neither stored backend
describes it, and a matching kernel kind would be a later proposal.

## 5. Workloads

Fixed shapes equal to the prediction cases already reported, sent token-in
and token-out (`backend: vllm_tokens`) so lengths are exact.

| case | definition | bytes per all-reduce |
| --- | --- | --- |
| prefill 256 | one request, 256 prompt tokens, 1 output token | 2 MiB |
| prefill 2048 | one request, 2048 prompt tokens, 1 output token | 16 MiB |
| decode 1 | 1 request, 568 prompt tokens, 64 output tokens | 8 KiB |
| decode 8 | 8 concurrent requests, same lengths | 64 KiB |
| decode 64 | 64 concurrent requests, same lengths | 512 KiB |
| decode 256 | 256 concurrent requests, same lengths | 2 MiB |

A decode case is measured only on iterations where all N requests are
decoding; with 568 prompt and 64 output tokens the cached length averages 600,
as in the prediction. Prompts are fixed pseudo-random token ids. Each case is
preceded by the pipeline's warm-up of the same shape. An optional seventh case,
a short mixed trace, would exercise the scheduler; it is not needed for the
iteration comparison and is left out by default.

## 6. GPU allocation and isolation

- tray12, one configuration at a time; TP4 needs the whole tray.
- Before each launch: expected UUIDs, utilisation at most 5 %, no compute
  process on any tray GPU, as in the profiling drivers. A foreign process
  appearing during a run stops it.
- Other sessions' containers exist on tray12 (`inkling-rl-vllm`, and since
  2026-11-11 `grtest-*` with an SGLang image). They must be idle on the GPUs
  for the window; I will not stop them. If the tray is not free, wait or ask.
- The engine runs under this session's processes only; cleanup kills only
  those.

## 7. Method and runtime

Per configuration, in this order:

1. **Stage 1, identity (about 5 minutes).** One launch, decode 8 and prefill
   256 under Nsight Systems. Establish section 4, and that the kernels in an
   iteration are the ones the arch lists. Stop here if they are not.
2. **Stage 2, kernel-level pass.** All six cases under Nsight Systems, at
   least 50 measured iterations per case. Per iteration: sum of GPU kernel
   time, GPU busy span, wall time; per kernel role: time per call.
3. **Stage 3, undisturbed pass.** The same cases without the profiler,
   three launches, for client-side latency and throughput and to size the
   profiler's overhead.
4. **CPU-only:** `alignment timing-predict` on the measured iteration shapes
   (so batch composition is the engine's, not assumed), then `alignment
   analyze`.

Order of configurations: TP1 first as a pilot, then TP2, then TP4, each
needing a pass of the previous one.

| | estimate |
| --- | --- |
| model load per launch | 1 to 3 minutes |
| GPU window per configuration | 20 to 30 minutes |
| total GPU window, three configurations | 60 to 90 minutes |
| analysis, CPU-only | 30 minutes |

## 8. Metrics

- **Iteration latency** per case: measured kernel-sum GPU time (the quantity
  the simulator predicts), GPU busy span, wall time; median and spread over
  iterations and launches.
- **Prediction error** per case and mode: (predicted − measured kernel sum) /
  measured, for TP1, and for TP2 and TP4 in both `nccl` and `nccl_graph`
  modes. Proposed reading: within ±15 % is agreement; beyond that the
  per-role table says which kernel is off.
- **Per-role error:** predicted against measured time for each kernel role
  (norms, GEMMs, attention, KV append, activation, all-reduce).
- **All-reduce:** measured GPU time per collective against both stored rows,
  and the collective's share of the iteration.
- **Wall against kernel time:** the ratio the simulator calls
  `gpu_time_multiplier` (it assumes 1.0), per case.
- **Throughput and client latency** from stage 3: output tokens per second,
  time to first token, time per output token, against the predicted iteration
  rate.
- **TP ranking:** whether measured TP1, TP2, TP4 order per case matches the
  prediction in each mode.

## 9. Resource limits and stopping conditions

| limit | value |
| --- | --- |
| GPU execution, all configurations | 120 minutes |
| disk on shared storage | 60 GB (weights 16, engine environment 15, traces and logs up to 29) |
| per launch | 10-minute timeout to readiness, 15-minute timeout per case |
| network | one model download, engine dependencies |

Stop and report when: the environment guard or a UUID check fails; a foreign
GPU process appears; the engine fails to start or to reach readiness; a
request errors or returns a wrong token count; stage 1 shows kernels the arch
does not model; a case exceeds its timeout; or the budget is reached. A
stopped configuration is reported as incomplete; later ones do not start.

Nothing is written to any profile database. No GPU is reset and no unrelated
process is terminated. The tracked database, fork main and upstream stay
untouched.

## 10. Approvals this would need, in order

1. CPU-only: build the fork's environment on aarch64 and download the weights
   (needs your Hugging Face access). No GPU.
2. GPU: TP1 pilot (stages 1 to 3), about 25 minutes.
3. GPU: TP2, then TP4, each after the previous one's report.

A result that the engine's default collective is not NCCL, or that steps are
not graph-replayed, changes what the later steps should measure, so each step
ends with a report and a decision.
