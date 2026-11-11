# Serving validation: readiness for the TP1 smoke test, 2026-11-11

CPU-only preparation after Stage A (`SERVING_VALIDATION_STAGE_A.md`). Nothing
ran on a GPU, no weights were downloaded, and no credential was read.

**Ready:** engine environment, load generator, serving worktree, text corpus,
validated run configs, engine command lines, launch guard, kernel classifier.

**Blocked on:**

1. Gated-model access for the run user. Once access exists:
   - check it with `serving-validation/hf_access_check.py`;
   - then a 16.06 GB download, which needs your confirmation.
2. Approval of the TP1 smoke test, and a GPU 1 that passes the guard at
   launch time.
3. For TP2, GPUs 2 and 3, or another pair you choose. GPU 2 currently holds
   another user's SGLang server (249.5 GB, `sglang::scheduler`, since about
   06:40 UTC).

## 1. Load generator (`req-frontend`)

| item | state |
| --- | --- |
| source | `uw-syfi/request-factory` at `2cd7990c527f9750d2bda1fb1dfe35e77000a5f8` (the submodule pin), cloned to `$GB300_ROOT/serving-validation/req-frontend` |
| build | `cargo build --locked --release --bin session_runner` (rustc 1.99.1, aarch64), 41 s, rc 0; script `serving-validation/build_req_frontend.sh` |
| binary | `req-frontend-target/release/session_runner`, sha256 `e71c8062…a568ea` |
| exact lengths | requests carry `max_tokens` and `ignore_eos: true` (`src/backend/wire/vllm.rs:28`), and the pipeline audits actual against target output lengths |
| inputs it needs | a tokenizer (`tokenizer.json`, from the gated snapshot) and a text corpus |
| corpus | `enwik9`, the corpus the pipeline's host template names, downloaded from `mattmahoney.net`, 1,000,000,000 bytes, sha256 `159b8535…3744bc` |

On the earlier refusal: in Stage A one command that cloned, built and ran
the binary was refused. This time I ran the clone and the build as separate
commands and both were allowed. I did not change any permission and did not
work around a refusal.

**Can the pipeline run without it? No.**

- For every non-resume profile, `alignment profile` builds `session_runner`
  before starting the engine (`alignment/runner.py:393-398`).
- It runs it from the fixed path `alignment/load_generator/req-frontend/target/release/session_runner`
  (`alignment/load_generator/runner.py:14-15`, `:36-51`).
- A hand-driven smoke test (engine under `nsys`, requests sent by hand) would
  bypass the launch records the iteration parser and request audit expect. It
  would test less and is not recommended.

## 2. Serving worktree

The pipeline imports the engine from `alignment/profiler/vllm` and the load
generator from `alignment/load_generator/req-frontend` of the checkout it runs
from. It sets `PYTHONPATH` to the former (`alignment/profiler/vllm_server.py:33`,
`:178`). So it runs from its own worktree:

- `ServingStudioSim-serving`, detached at `5f018cb6`, beside the main checkout
  and the profiling worktree.
- Its two submodule paths are symbolic links to the pinned checkouts in
  `$GB300_ROOT/serving-validation`, and `req-frontend/target` links to the
  built target. `git status` there shows the two links as type changes; they
  are not committed.
- The profiling worktree `ServingStudioSim-gb300` and the main checkout are
  untouched.

## 3. Run configs and exact commands

`serving-validation/make_configs.py --root $GB300_ROOT/serving-validation
--out-dir $GB300_ROOT/serving-validation/configs` renders three `profile.yaml`
files and their traces. All three load without error through the launcher's
own `load_profile_config`.

| run | GPUs | requests | engine |
| --- | --- | --- | --- |
| `tp1-smoke` | 1 | 3 single prefills of 256 tokens; 3 batches of 8 decodes (568 prompt, 64 output) | defaults |
| `tp2-default` | 2, 3 | the same, plus 3 batches of 64 decodes | defaults |
| `tp2-nccl` | 2, 3 | as `tp2-default` | NCCL forced |

**Common settings.**

- Batches start 1 s apart (`arrival_mode: trace-timed`).
- Warm-up on; `HF_HUB_OFFLINE=1`; `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- All caches under `$GB300_ROOT/serving-validation/cache`.
- Model path: the local snapshot directory of revision
  `0e9e39f249a16976918f6564b8830bc894c89659`.
- TP2 runs add `NCCL_DEBUG=INFO` for initialisation only
  (`NCCL_DEBUG_SUBSYS=INIT,ENV`).

**Engine command lines**, rendered by the launcher's own `build_server_argv`
plus the vLLM token backend and Nsight Systems arguments:

```
tp1-smoke: <venv>/bin/python -m vllm.entrypoints.cli.main serve <snapshot> --api-server-count 1
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 --data-parallel-size 1
  --max-num-batched-tokens 2048 --enable-chunked-prefill --enable-prompt-tokens-details
  --gpu-memory-utilization 0.85 --enable-server-load-tracking --enable-logging-iteration-details
  --served-model-name llama31-8b-instruct --dtype bfloat16 --max-model-len 8192
  --no-enable-prefix-caching --seed 0 --tokens-only --profiler-config.profiler=cuda

tp2-nccl:  the same with --tensor-parallel-size 2, plus
  --disable-custom-all-reduce --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'
  and in the environment VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0
```

`tp2-default` is `tp2-nccl` without the four switches. Port 8000 is free on
tray12 at present; the launch guard checks it again.

**Launch** (needs approval):

```
serving-validation/run_serving_profile.sh tp1-smoke     # LIMIT_S=900
```

which runs, from the serving worktree:

```
CUDA_VISIBLE_DEVICES= timeout --signal=INT --kill-after=120 900 \
  <main checkout>/.venv/bin/python -m launcher alignment profile <root>/configs/tp1-smoke.yaml
```

## 4. Are the four switches sufficient? Yes, by source

Checked against the fork at `e1423a4d`:

- **Arguments parse as intended.** The fork's own argument parser, given the
  planned arguments on CPU, yields `disable_custom_all_reduce = True` and
  `pass_config.fuse_allreduce_rms = False`.
- **A user value survives the optimisation-level defaults.** Those defaults
  are applied only to fields the user left unset
  (`vllm/config/vllm.py:1027-1055`), so O2 cannot turn the fusion back on.
- **Each switch removes one path in the dispatch chain**
  (`cuda_communicator.py:335-408`):
  - FlashInfer standalone: `fi_ar_comm` is never created
    (`cuda_communicator.py:64-66`, `:120-124`).
  - Custom all-reduce: `_ENABLE_CUSTOM_ALL_REDUCE` false (`gpu_worker.py:1559`)
    means no `ca_comm` (`cuda_communicator.py:143-151`).
  - torch symmetric memory: `symm_mem_comm` is never created (`:114-118`).
  - Fusion: `AllReduceFusionPass` is not installed.
- **The rest of the chain is off without a switch:**
  - NCCL symmetric memory needs `VLLM_USE_NCCL_SYMM_MEM=1` (default 0);
  - FlashInfer PCIe IPC needs `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`
    (default 0);
  - quick reduce and AITER are ROCm only;
  - sequence parallelism and async TP are off for every model (`IS_DENSE = False`).
- **Result:** every all-reduce reaches PyNCCL's `ncclAllReduce`
  (`pynccl.py:268-296`).

**Collectives the simulator does not model**, in both variants:

- `VocabParallelEmbedding` all-reduces its output once per iteration
  (`vocab_parallel_embedding.py:556`).
- The logits are gathered across ranks (`logits_processor.py:130-133`).
- The engine shards `lm_head` across ranks; `llama3_dense_tp` keeps it
  replicated at full size.

Per iteration the engine therefore runs 65 all-reduces and one gather where
the simulator charges 64 all-reduces. Its `lm_head` GEMM is 1/TP of the
simulated one, worth about 70 µs at TP2 decode. These are expected,
attributable differences, and the per-kernel comparison names them.

## 5. NCCL library: pinning 2.28.9 does not work

Checked on CPU only, without touching either environment's files:

- A process can map both libraries side by side. Loading the profiling
  environment's `libnccl.so.2` (2.28.9) by full path inside the serving
  environment, after torch has loaded its own 2.29.7, reports version 22809.
- Both have SONAME `libnccl.so.2`.
- PyNCCL resolves 20 entry points. Three are absent from 2.28.9:
  `ncclCommQueryProperties`, `ncclCommSuspend` and `ncclCommResume`.
  - The first is optional.
  - On CUDA, a missing `ncclCommSuspend` or `ncclCommResume` raises during
    PyNCCL initialisation; only ROCm skips them
    (`pynccl_wrapper.py:400-433`, comment: "vLLM's CUDA path already requires
    that version").

So `VLLM_NCCL_SO_PATH` pointed at 2.28.9 would fail at engine start. The
engine has to run NCCL 2.29.7, while the stored `nccl` and `nccl_graph` rows
used 2.28.9. That version difference stays a documented confound of the
TP2 comparison. Removing it would mean re-measuring the rows with 2.29.7,
which is GPU profiling and not approved.

## 6. Telling the collectives apart in a trace

`serving-validation/kernel_classes.py` classifies kernel names. Its self-test
passes 14 of 14 sample names, which were written from the kernel templates,
not taken from a trace.

| class | identified by |
| --- | --- |
| NCCL all-reduce | `ncclDevKernel_AllReduce_*` |
| other NCCL (logits gather) | other `ncclDevKernel_*` |
| FlashInfer fused all-reduce + residual + RMSNorm | `allreduce_fusion_kernel_{oneshot_lamport,twoshot_sync}` with template pattern 1 (`kARResidualRMSNorm`); `oneshotAllreduceFusionKernel` with `RMSNormFusion = true`; `rmsNormLamport` after `twoshotAllreduceKernel` |
| FlashInfer standalone all-reduce | the same kernels with pattern 0 or `RMSNormFusion = false`; `twoshotAllreduceKernel` |
| vLLM custom all-reduce | `cross_device_reduce_1stage` / `_2stage` |
| torch symmetric memory | `multimem_all_reduce`, `two_shot_all_reduce` |
| attention, TRT-LLM / FlashInfer | `fmha`/`trtllm` / `BatchPrefill`, `BatchDecode` |
| KV write | `reshape_and_cache_flash_kernel` |

Sources: FlashInfer 0.7.0 `flashinfer/comm/trtllm_allreduce_fusion.cuh`
(pattern enum at line 823), `trtllm_mnnvl_allreduce.cuh` (lines 913-916,
1268-1269, 1340-1342); vLLM `csrc/custom_all_reduce.cuh:6-34`.

Two independent confirmations per run:

- the engine logs its active all-reduce backends in dispatch order
  (`cuda_communicator.py:326-333`);
- `NCCL_DEBUG=INFO` logs communicator initialisation in TP2 runs.

The classifier also reports, per class, the share of kernels launched from a
CUDA graph (`graphNodeId`). Its sqlite path has not run yet, because no
serving trace exists.

## 7. The three kinds of latency stay separate

| level | quantity | source | compared with |
| --- | --- | --- | --- |
| GPU iteration | `measured_ms` per `vllm_iteration(N): forward`: critical-path busy time plus collectives across ranks; also kernel sum and first-to-last-kernel span | Nsight trace, analyzer `alignment_iteration` | the CostTree total of the same measured batch (`timing-predict`), error `(sim − meas) / meas` |
| kernel | duration per kernel, grouped by role and by the classes above; per-call all-reduce time at each message size | Nsight trace, `kernel_classes.py`, label rules | the simulator's leaf for that role; for the NCCL-forced all-reduce, the `nccl_graph` and eager `nccl` rows |
| client | time to first token, time per output token, request throughput | load generator summary and the engine's request timing records, from a pass without Nsight | the predicted iteration rate; never mixed into the GPU error |

Host gaps between iterations (`measured_gpu_cycle_ms / measured_ms`) are
reported as their own figure, not folded into the GPU error.

## 8. TP1 smoke test, ready to approve

| item | value |
| --- | --- |
| GPU | tray12 GPU 1, `GPU-9574ced0…` |
| GPU time | at most 15 min, wall limit 900 s including load, compile and graph capture |
| engine | fork `e1423a4d` over precompiled `04730e82`, `serving-validation/env-fingerprint.json` |
| model | snapshot `0e9e39f2…`, local, offline |
| requests | 3 prefills of 256 tokens, 3 batches of 8 decodes of 64 tokens; warm-up first |
| tracing | CUDA and NVTX only, CUDA-graph node trace, capture limited to the measurement window by the profiler API |
| artifacts | `.nsys-rep` and `.sqlite` (expected under 1 GB), parsed iterations, kernel sequences, `kernel_classes.json`, guard logs |
| what it must establish | engine starts and serves; token counts correct; iteration records present; kernel names per role; graph mode of each selected iteration (decode full graph, prefill 256 piecewise) |
| error figures | not interpreted yet |

**Guards** (`run_serving_profile.sh`):

- **Before launch:**
  - expected UUID on GPU 1;
  - utilisation at most 5 %, memory below 1 GiB, no compute process on it;
  - every compute process on the tray recorded;
  - serving environment equal to the committed fingerprint;
  - profiling environment guard unchanged;
  - model snapshot complete;
  - at least 60 GB free;
  - port free;
  - both profile databases checksummed.
- **During:** a foreign compute process on GPU 1 stops the run.
- **After:** the same checks, and the databases unchanged.

**Stop conditions:**

- any guard failure;
- startup failure, or a server not ready within 10 minutes;
- a request error or wrong token count;
- the outer limit.

**Cleanup.** Only processes of the run's own session are signalled. No GPU
reset; the SGLang server on GPU 2 is not touched.

`run_serving_profile.sh` has been syntax-checked, not run. Its first real run
is the smoke test itself.

## 9. Disk

| item | size |
| --- | --- |
| serving workspace now | 16 GB (engine environment 13 GB, corpus 1 GB, fork 0.9 GB, load-generator build 0.7 GB) |
| still needed | weights 16.1 GB, caches up to 5 GB, smoke-test traces under 1 GB |
| free on `/mnt/shared` | 23 TB |

## TP1 smoke test: stopped at the access check, 2026-11-11

The smoke test was approved on the condition that model access verifies first.
It did not:

- `hf_access_check.py` in the serving environment, with the default credential
  lookup, reports `authenticated: false` (`LocalTokenNotFoundError`) and
  `download_access: false` (`GatedRepoError`).
- The pinned revision `0e9e39f249a16976918f6564b8830bc894c89659` still
  resolves, 16,060,556,376 bytes of weights.

Per the stop conditions, nothing further ran:

- no download;
- no GPU guard check or launch;
- no change to any cache or database.

To proceed, an authorized user stores a credential for the run user `kevin`
on the rack with an account that has accepted the Llama 3.1 licence, for
example with the serving environment's `hf auth login`. The credential lands
in `/mnt/shared/home/kevin/.cache/huggingface/`, which the check reads. The
access check is then repeated before anything else.
