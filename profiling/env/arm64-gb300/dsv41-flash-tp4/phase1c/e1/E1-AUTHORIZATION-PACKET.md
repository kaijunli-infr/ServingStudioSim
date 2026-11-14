# E1 authorization packet — decode routing and rank-arrival capture (preflight review; nothing executed)

Prepared 2026-11-14 for Kevin's E1 preflight review, after the acceptance of Phase 1C-0 (`phase1c-prep` 2e53072d).
Status: **proposal; no GPU process, no reservation pod, no launcher change applied, no fit, no model change**. The
host preflight was run read-only on tray16 (section 7); the CPU checks of every script were run on frozen Phase 0
data without opening a held-out row (section 10). E1 starts only after an explicit go on this packet.

Governing documents (verbatim excerpts in sections A and B at the end): `GPU-EXPERIMENT-MATRIX.md` E1 section and
common setup (file sha256 869f0d06171a5b62e9c0fe673546921b3329cb37f2fdbe05c9c431591394079e) and the frozen
`VALIDATION-PROTOCOL.md` v1.0 (sha256 07ab7d7171ae3185d44bcc92c5a5b6ef0f03c3bb8d82eec27461d429e80327be, recorded
in `PROTOCOL.sha256`). Where this packet deviates from the matrix text, the deviation is listed in section 11.

Modelling restrictions acknowledged: the Phase 1A and Phase 1B fitted models stay frozen; the observed 70–80 % and
25–30 % wait positions are reported, not converted into fitted correlation weights; the Monte Carlo dispersion
factor and the interval widths are untouched; `uncertainty/RELIABILITY.md` and `FINDINGS.md` are preserved. Every E1
script in this directory is an estimator or a check; none writes a coefficient, a parameter or a preset.

## 0. Summary of what E1 is and how it will run

| item | value |
|---|---|
| question | is the per-rank variation of decode segments common-mode (routing, same kernels) or independent (host arrival), and does the wait position change with batch size, with KV length, or between workloads/passes; what is the routed-expert load per iteration and how much wall variance it explains; what is the b = 1–2 rank-arrival pattern |
| design | one vLLM launch, one nsys session; 24 cells = b ∈ {1, 2, 4, 8, 16, 24, 32, 48} × KV ∈ {4k, 32k, 96k} prompt tokens, 1024 generated tokens per request, 2 waves for b ≤ 8 and 1 wave for b ≥ 16 (36 waves, 450 requests, trace span 940.6 s) |
| data | ≥ 35,624 pure-decode iterations at the target b (≥ 736 per cell, ≥ 1,000 for b ≤ 8), 1,240 prefill-chunk iterations, 460,800 generated tokens with per-token routed experts |
| designation | **identification** (protocol section 2); its rows become `identification` in `partition-1c.json`; it validates nothing; E2 is the validation |
| hardware | rack 10 tray16, 4 × GB300 (TP4), reservation pod `dsv41-gb300-reserve-tray16` |
| GPU time | estimated 1 h 45 min GPU-held (dry-run pair ≈ 55 min + capture ≈ 50 min), tray wall ≈ 3 h 15 min including CPU-only export and parse; reservation 5 h maximum |
| storage | expected ≈ 42 GB raw on CephFS; budget 80 GB |
| code | fork unchanged (0ab18711); launcher patch of 46 lines in 3 files + 5 CPU tests (section 4.4) to record routes beside a timing pass; Option B (no routes, no patch) is available |
| decision enabled | choose the per-rank sampling structure of the decode band for model v1.1 (common-mode / independent / mixture indexed by b and/or KV); decide whether a routing-conditional MoE term is a candidate; decide whether b1-2 gets its own arrival model — all subject to E2 |

## 1. Exact workload configuration, batch regimes, context lengths and repetitions

**Engine and recipe** — identical to every Phase 0 pass (the server block, environment and nsys block of the E1
configs are copied verbatim from `make_configs.py`, sha256 92b3f80750f8daa9742e55b7ed3af520e3e8d0f476b3036b5d5f0f4c2e097366):

| item | value |
|---|---|
| model | `/mnt/shared/weights/DeepSeek-V4.1-Flash-1017` (HF 08c8657f, 476 GB) |
| engine | vLLM fork `servingstudio-alignment-v0.31.0` = 0ab18711 (`0.31.1.dev17+g0ab187113`), torch 2.13.0+cu130, uv-managed CPython 3.12, venv `gb300/fork-env-v0.31.0/venv` |
| parallelism | TP4, DP1, no EP, no speculative decoding |
| server flags | recipe TP4 verbatim: `--tokenizer-mode deepseek_v41 --attention-config {backend FLASHINFER_MLA_SPARSE_DSV41, indexer_kv_dtype mxfp4, indexer_sparse_logits} --kv-cache-dtype fp8 --max-num-seqs 128 --engram-config {cpu_offload} --max-model-len 131072 --language-model-only`; chunk (max-num-batched-tokens) 16384; gpu_memory_utilization 0.9; CUDA graphs FULL_AND_PIECEWISE with capture sizes 1…256 (decode at every E1 b runs in graphs; prefill chunks run eager) |
| caches | warm, persistent, under `gb300/cache/` (deepgemm, triton, nv-compute, flashinfer-autotune; preflight checks they are non-empty) |
| request backend | `openai` (the only backend that returns routed experts) with `temperature 0.0` (req-frontend default), `ignore_eos true`, `max_tokens = output_len`; Option B uses `vllm_tokens` as the Phase 0 timing passes did |
| prompts | enwik9 token slices, one distinct pool offset per request (no shared prefix; the prefix cache stays enabled as in Phase 0 and has nothing to hit) |

**Cells and repetitions** (`make_e1_inputs.py`, deterministic; the trace `traces/e1_main.csv` has sha256
2e147402b5a24f9e2078bf12e3623691c2fd6d5c8378bab09d7e86ee23bcc2ef):

| parameter | value |
|---|---|
| batch sizes b | 1, 2, 4, 8, 16, 24, 32, 48 |
| per-request prompt length (KV level) | 4096, 32768, 98304 tokens (the matrix's 4k / 32k / 96k) |
| generated tokens per request | 1024 (every request; a wave therefore decodes at exactly b requests once its prefills are done and all b requests finish within a few iterations of each other) |
| waves (repetitions) per cell | 2 for b ≤ 8, 1 for b ≥ 16 — 36 waves, 450 requests |
| arrival | trace-timed: the b requests of a wave arrive together; the next wave arrives after the previous wave's estimated duration × 1.5 + 5 s (schedule below); one launch, one trace, one nsys session |
| pure-decode iterations per wave (lower bound) | 1024 − (prefill-chunk iterations of the wave): minimum 736 (b = 48, 96k), ≥ 1,000 for every b ≤ 8 cell; matrix requirement ≥ 400 met everywhere |
| expected totals | prefill-chunk iterations 1,240; decode iterations 36,864; pure-decode ≥ 35,624; generated tokens 460,800 |
| KV residency | max 4,767,744 tokens (b = 48 × (98304 + 1024)) of 24,088,361 (Phase 0 server log: "GPU KV cache size: 24,088,361 tokens") = 19.8 %; max_model_len 131072 ≥ 99,328 |
| warm-up exclusion | the first 50 iterations after engine start and JIT-affected iterations (Phase 0 rule); the first iterations of a wave are mixed (prefill + decode) and are classified `mixed`, not excluded — they are an E1 contrast (section 8) |

Schedule (arrival of each wave; durations from the Phase 1A medians 4.65 / 4.41 / 4.60 / 5.20 / 6.30 / 7.73 / 8.54 /
9.93 ms for b = 1 … 48 and 0.235 s per 16k chunk; `E1-INPUTS.json`):

| b | KV | wave | arrival s | prefill iters | est. duration s | pure decode ≥ |
|---|---|---|---|---|---|---|
| 1 | 4096 | 0 | 0.0 | 1 | 5.0 | 1023 |
| 1 | 4096 | 1 | 12.5 | 1 | 5.0 | 1023 |
| 2 | 4096 | 0 | 25.0 | 1 | 4.8 | 1023 |
| 2 | 4096 | 1 | 37.1 | 1 | 4.8 | 1023 |
| 4 | 4096 | 0 | 49.2 | 1 | 4.9 | 1023 |
| 4 | 4096 | 1 | 61.7 | 1 | 4.9 | 1023 |
| 8 | 4096 | 0 | 74.1 | 2 | 5.8 | 1022 |
| 8 | 4096 | 1 | 87.8 | 2 | 5.8 | 1022 |
| 16 | 4096 | 0 | 101.5 | 4 | 7.4 | 1020 |
| 24 | 4096 | 0 | 117.5 | 6 | 9.3 | 1018 |
| 32 | 4096 | 0 | 136.5 | 8 | 10.6 | 1016 |
| 48 | 4096 | 0 | 157.5 | 12 | 13.0 | 1012 |
| 1 | 32768 | 0 | 182.0 | 2 | 5.2 | 1022 |
| 1 | 32768 | 1 | 194.8 | 2 | 5.2 | 1022 |
| 2 | 32768 | 0 | 207.7 | 4 | 5.5 | 1020 |
| 2 | 32768 | 1 | 220.8 | 4 | 5.5 | 1020 |
| 4 | 32768 | 0 | 234.0 | 8 | 6.6 | 1016 |
| 4 | 32768 | 1 | 248.9 | 8 | 6.6 | 1016 |
| 8 | 32768 | 0 | 263.8 | 16 | 9.1 | 1008 |
| 8 | 32768 | 1 | 282.4 | 16 | 9.1 | 1008 |
| 16 | 32768 | 0 | 301.0 | 32 | 14.0 | 992 |
| 24 | 32768 | 0 | 327.0 | 48 | 19.2 | 976 |
| 32 | 32768 | 0 | 360.8 | 64 | 23.8 | 960 |
| 48 | 32768 | 0 | 401.5 | 96 | 32.7 | 928 |
| 1 | 98304 | 0 | 455.6 | 6 | 6.2 | 1018 |
| 1 | 98304 | 1 | 469.8 | 6 | 6.2 | 1018 |
| 2 | 98304 | 0 | 484.1 | 12 | 7.3 | 1012 |
| 2 | 98304 | 1 | 500.1 | 12 | 7.3 | 1012 |
| 4 | 98304 | 0 | 516.1 | 24 | 10.4 | 1000 |
| 4 | 98304 | 1 | 536.6 | 24 | 10.4 | 1000 |
| 8 | 98304 | 0 | 557.1 | 48 | 16.6 | 976 |
| 8 | 98304 | 1 | 587.0 | 48 | 16.6 | 976 |
| 16 | 98304 | 0 | 617.0 | 96 | 29.0 | 928 |
| 24 | 98304 | 0 | 665.5 | 144 | 41.8 | 880 |
| 32 | 98304 | 0 | 733.1 | 192 | 53.9 | 832 |
| 48 | 98304 | 0 | 818.9 | 288 | 77.8 | 736 |

**Dry-run pair** (`traces/e1_dry.csv`, sha256 c6d8e13b7229ffd40cfede127bd57ad345aa17213b7590d39f9b2ef697f60158; span 72.7 s; `profile_kind
workload_metrics`, no nsys): cells (b = 8, 32k) and (b = 48, 32k), one wave each, 56 requests, run twice in two launches
— `e1_dry_off` (openai backend, no routes) and `e1_dry_on` (identical plus `capture_routed_experts: true`). Purpose:
the matrix's hook stop conditions (overhead ≤ 2 % on the engine's own iteration time, token identity) and an
informational comparison of the openai path against the Phase 0 b = 8 median (5.20 ms, `vllm_tokens`).

| b | KV | wave | arrival s | prefill iters | est. duration s | pure decode ≥ |
|---|---|---|---|---|---|---|
| 8 | 32768 | 0 | 0.0 | 16 | 9.1 | 1008 |
| 48 | 32768 | 0 | 18.6 | 96 | 32.7 | 928 |

Configs (`configs/<pass>/profile.json`; all four pass `launcher alignment profile --dry-run`, the two with routes only
with the patch of section 4.4 applied):

| pass | kind | backend | routes | sha256 |
|---|---|---|---|---|
| `e1_dry_off` | workload_metrics | openai | no | 9567412994b78332742b6077c323847d6ef59c10bd516e642e4e8d2bcb4b0683 |
| `e1_dry_on` | workload_metrics | openai | yes | c7630db64831b326170e86765818228a97e5eeea5121cde80c71ae6c1232a371 |
| `e1_main` | nsys (delay 10 s, no bounded stop) | openai | yes | 87577adcf4313733aa80c1a708b5c5a4dd32bb3015bc52fba2b7b4315a786b49 |
| `e1_main_norouts` (Option B) | nsys (delay 10 s, no bounded stop) | vllm_tokens | no | a852cfd4ab480ed3c9e4be899be6290d8606bb93111c7f7b0523b9fde7585fcc |

## 2. GPU tray, device count, reservation method and isolation

| item | value |
|---|---|
| tray | rack 10 `pod4-gb300-10-tray16-f3` (the Phase 0 tray: driver 595.71.05, kernel 6.14.0-1013-nvidia-64k, nsys 2025.6.3.541, Ubuntu 24.04.4); user `dc`, `HOME=/mnt/shared/home/dc` |
| devices | all 4 GB300 of the tray (`cuda_visible_devices: 0,1,2,3`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`), TP4; no other process may use any of them |
| reservation | Kubernetes pause pod `dsv41-gb300-reserve-tray16` (namespace `inference-bench`, requests/limits `nvidia.com/gpu: 4`, nodeSelector tray16) from the laptop file `k8s/dsv41-gb300/reserve-tray16.yaml` (content below), applied through the Rancher SOCKS proxy before the first launch and deleted after the last run; it holds the scheduler's view of the GPUs while the capture runs host-level |
| isolation | host-level run as dc with no GPU pod on the node (checked from the laptop before each launch and every 5 min during runs); the host preflight refuses to start if `nvidia-smi` shows any compute app or any GPU memory in use, if leftover `vllm`/`nsys`/`session_runner` processes exist, or if the recipe identity differs; the tray can still be pre-empted by the scheduler (stop condition); no CPU pinning (as in Phase 0); other sessions use tray13/tray14, none uses tray16 |
| state on 2026-11-14 17:47Z | node `nvidia.com/gpu` allocated 0 / 0; no reservation pod exists (the Phase 0 pod was deleted 2026-11-12 19:40Z); only DaemonSet and system pods on the node; all four GPUs 0 MiB, no compute apps (host preflight log in section 7) |
| alternative | any other rack-10 tray with 0 GPU pods (edit the nodeSelector, rerun both preflights); never rack 3 tray06 or tray10; the CephFS caches and weights are tray-independent |

```yaml
# Placeholder that holds all 4 GPUs of rack-10 tray16 in the scheduler while the
# DSV41 GB300 capture runs host-level on that tray (dc), so no GPU pod lands there.
# It runs nothing on the GPUs. Delete it when the capture campaign is over:
#   HTTPS_PROXY=socks5://127.0.0.1:11080 kubectl -n inference-bench delete pod dsv41-gb300-reserve-tray16
apiVersion: v1
kind: Pod
metadata:
  name: dsv41-gb300-reserve-tray16
  namespace: inference-bench
  labels: {app: dsv41-gb300-reserve, owner: kevin}
spec:
  restartPolicy: Never
  runtimeClassName: nvidia     # same runtime class as the other GPU pods on this cluster
  nodeSelector: {kubernetes.io/hostname: pod4-gb300-10-tray16-f3}
  tolerations:
    - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
  containers:
    - name: hold
      image: registry.k8s.io/pause:3.9
      resources:
        requests: {nvidia.com/gpu: 4}
        limits: {nvidia.com/gpu: 4}
```

Laptop commands (proxy as in CLAUDE.md):

```
ssh -fN -D 127.0.0.1:11080 pod4-gb300-10-tray14
HTTPS_PROXY=socks5://127.0.0.1:11080 kubectl get pods -A -o wide --field-selector spec.nodeName=pod4-gb300-10-tray16-f3
HTTPS_PROXY=socks5://127.0.0.1:11080 kubectl apply -f k8s/dsv41-gb300/reserve-tray16.yaml
HTTPS_PROXY=socks5://127.0.0.1:11080 kubectl -n inference-bench get pod dsv41-gb300-reserve-tray16 -o wide   # Running on tray16
HTTPS_PROXY=socks5://127.0.0.1:11080 kubectl -n inference-bench delete pod dsv41-gb300-reserve-tray16          # after the run
```

## 3. Estimated and maximum GPU occupancy

Basis: Phase 0 on the same tray with warm caches (quadrant run 2: start 13:15:32Z → measurement 13:38:40Z; run2:
16:50:42Z → first measured iteration 17:14:48Z; model load 75–82 s, CUDA-graph capture ≈ 40 s, req-frontend warm-up
and FlashInfer autotune the rest). Export and parse of run2 (97.2 M kernel rows): report write ≈ 10 min, sqlite export
≈ 9 min, parse ≈ 29 min, host timeline ≈ 1 min. E1 has ≈ 38 k iterations ≈ 2.2 × run2.

| phase | GPU-held | CPU-only after GPUs are free | basis |
|---|---|---|---|
| `e1_dry_off` | ≈ 27 min (25 startup + 1.3 workload + 1 teardown) | 1 min (metrics extraction) | Phase 0 startup |
| `e1_dry_on` | ≈ 27 min | 1 min | same |
| `e1_main` | ≈ 50 min (25 startup + 17 capture + up to 8 report finalisation) | ≈ 20 min sqlite export + ≈ 65 min parse + 2 min host timeline + ≈ 10 min compact-table extraction | run2 × 2.2 |
| **estimate** | **≈ 1 h 45 min GPU-held (≈ 7 GPU-hours)** | ≈ 1 h 40 min | |
| **maximum** | reservation **5 h**; abort if startup > 60 min, if the capture workload has not finished 40 min after measurement began, or if the GPUs are still held 15 min after the workload ended | export > 60 min or parse > 3 h → stop and report (the capture is kept) | |

The matrix estimated 60 min + 20 min export with a 2 h reservation; the revision is due to the dry-run pair (two
extra launches) and the larger trace. The reservation pod stays until the launcher exits (export and parse are
CPU-only but run on the reserved tray).

## 4. Nsight / NVTX capture settings and per-rank timestamp alignment

### 4.1 nsys (identical to Phase 0, delayed start, no bounded stop)

Launcher-built command line (Phase 0 `progress.md`, run.log of every nsys pass):

```
/opt/nvidia/nsight-systems/2025.6.3/target-linux-sbsa-armv8/nsys profile --trace-fork-before-exec=true --trace=cuda,nvtx
  --sample=none --cpuctxsw=none --cuda-graph-trace=node --cuda-event-trace=false
  --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o <log_dir>/nsys/<name> <vllm serve …>
```

| setting | value | reason |
|---|---|---|
| `capture_mode` | `cuda_profiler_api`: the launcher calls the server's `/start_profile` (VLLM_SERVER_DEV_MODE=1) which arms CUPTI from the CUDA-owning workers; server flag `--profiler-config.profiler=cuda` | Phase 0 |
| `capture_start_delay_seconds` | 10.0 (armed 10 s after "measurement begins", i.e. after warm-up drained and the prefix cache reset) | Phase 0 run2 (839c3617) |
| `capture_duration_seconds` | **absent**: the capture ends when the server exits | Phase 0 lesson: an in-inference `cudaProfilerStop` froze the engine for minutes and killed pass 3 |
| `cuda_graph_trace` | `node`: every kernel replayed inside a CUDA graph is a row (decode runs in graphs) | per-segment kernel attribution |
| `--trace=cuda,nvtx` | CUDA runtime/driver API rows (`cuLaunchKernel`, `cudaGraphLaunch`, syncs, memcpys → `host_timeline.json`), kernel rows, NVTX ranges; `osrt` **not** traced (the matrix's "OS-runtime" wording is not adopted: the launch timestamps come from the CUDA API trace, and Phase 0 comparability is kept) | Phase 0 |
| export | `nsys export --type sqlite --force-overwrite`, then `validate_export` (`nvtx_forward_ranges`, `kernel_rows`, `kernel_span_ms`, `ok`), then the Phase 0 parser → `parsed.json`, `parsed.kernels.parquet`, `kernel_sequences.json`, `host_timeline.json` | launcher |

### 4.2 NVTX and iteration identity

The fork (0ab18711, unchanged) pushes per rank and per iteration the ranges `vllm_iteration(N): preprocess | forward |
postprocess | sample | bookkeep | eplb` plus the iteration range (`alignment_trace.py`; `VLLM_NVTX_SCOPES_FOR_PROFILING`
on by default) — 7 ranges per device, 28 per iteration. `N` is the EngineCore's `alignment_iteration_index`, the same
number on all four ranks and in the per-iteration metrics record (`--enable-logging-iteration-details`, schema 4,
with `decode_request_progress` per request: `request_id`, `kv_len`, `output_tokens_before`, `emitted_tokens`). The
parser pairs ranges by `N` and device and reports `step_alignment.unpaired_steps_by_device` (must be `{}`) and
`devices_absent` per iteration (must be empty). Phase 0: 100 % attribution on all passes.

### 4.3 Per-rank timestamp alignment

All four TP workers are child processes of one EngineCore on one host and are traced in one nsys session
(`--trace-fork-before-exec=true`). nsys converts the CUPTI timestamps of every device into the single session
timeline, so kernel start/end times of different ranks are directly comparable; no per-rank offset is estimated or
applied anywhere in Phase 1A/1B/1C. Pre-registered verification on the E1 tables (`analysis/e1_identify.py`,
`collective_end_spread`): the end times of one collective kernel across the four ranks must agree, because the
all-reduce completes simultaneously. Phase 0 quadrant calibration rows (smoke output): median 0.0002–0.003 ms, p90
≤ 0.009 ms per cell. **Acceptance: median ≤ 0.02 ms and p90 ≤ 0.05 ms in every cell**; otherwise the arrival and
wait statistics are reported with the spread as a systematic uncertainty and no correction is applied.

Iteration origin: `t0_ns` = earliest kernel start across ranks (Phase 1A extractor); per rank `first_start_ns`,
collective windows `start_rel_ns`/`end_rel_ns`, segment lengths `seg_len_ns`; intrinsic collective time = minimum
duration across ranks; wait = duration − intrinsic (Phase 1A definitions, `costmodel/data.py::segments`, unchanged).

### 4.4 Routing hook (no fork change) and the launcher patch

The fork already contains the capturer used by Phase 0 pass 5: `RoutedExpertsCapturer` writes each layer's
`topk_ids` into a preallocated device buffer inside the forward (so inside the CUDA graphs) and the OpenAI API
returns a request's routes when the server runs with `--enable-return-routed-experts`; req-frontend persists one
`.npy` per request when given `--routed-experts-dir`, and `alignment/profiler/token_corpus.py` packs them into
`routes.u16` (`[token, layer, top_k]`, little-endian u16, 40 layers, 384 experts, top-6; rows = generated tokens
except the last one per request, "accepted generated tokens; no prompt"). Today the launcher offers this only in a
`token_corpus` pass, which switches the timing instrumentation **off**. The patch `patch/launcher-routes-optin.patch`
(3 files, 46 insertions, 5 deletions; applied by `patch/apply_routes_optin.py`; tests
`patch/test_alignment_routes_optin.py`) adds the opt-in key `workload.capture_routed_experts` for a timing pass:

- `alignment/load_generator/config.py`: the boolean key, refused unless the backend returns routes;
- `alignment/runner.py`: the routes directory is prepared for the opt-in; the server argv gets
  `--enable-return-routed-experts` exactly once and **never** the EPLB flags; after the run the routes are coverage-
  checked (`_check_routes_cover_replay`) and packed into `<log_dir>/token_corpus` and the result records
  `token_corpus_*`; timing instrumentation, NSYS and request timing records are untouched because the pass kind is
  still `nsys` / `workload_metrics`;
- `launcher/alignment_config.py`: the key requires `engine: vllm`.

CPU results (scratch worktree at 2e53072d on tray14, removed afterwards): the 5 new tests pass; the existing launcher
tests pass 134 / 135 with the single pre-existing environment failure (`test_launcher_corpus::test_emit_backends…`,
`cargo` not on PATH in a non-login ssh shell, documented in Phase 1B); the four E1 configs validate under
`launcher alignment profile --dry-run`. Without the patch, the two configs with the key are rejected by the launcher
with `LoadGeneratorConfig.__init__() got an unexpected keyword argument 'capture_routed_experts'`, i.e. they cannot
run by accident.

Join of routes to iterations (pre-registered, checked by `checks/e1_completeness.py`): generated token j (1-based)
of request r is the query of the pure-decode iteration in which `decode_request_progress` lists r with
`output_tokens_before == j` and `emitted_tokens == 1`; its routes are row `segment_start(r) + j − 1` of
`routes.u16` (`provenance.json::request_segments`). Each request therefore has exactly 1023 rows and 1023 such
iterations; the checker fails on any mismatch. Per-iteration routing statistics (per layer: max expert load,
number of non-empty experts, padded-M ratio at block 64) are computed by `analysis/e1_identify.py`.

Overhead: the capturer's device writes are inside the graphs; the D2H copy happens per request at completion. The
dry-run pair measures the hook on the engine's own `observed_elapsed_ms` (median per cell, hook-on / hook-off);
stop condition > 1.02. The `token_corpus` pass of Phase 0 ran the same capturer at the full diverse workload.

## 5. Artifact, temporary-file and disk-space budgets

Expected sizes (run2 at 17,686 iterations: `.nsys-rep` 2.7 GB, `.sqlite` 14 GB, `parsed.json` 399 MB,
`parsed.kernels.parquet` 604 MB, `host_timeline.json` 565 MB, metrics jsonl 52 MB; Phase 0 routes 62 MB for 135 k
tokens; scaled × 2.2 for ≈ 38 k iterations, routes × 3.4 for 460 k tokens):

| artifact | expected | where |
|---|---|---|
| `e1_main` `.nsys-rep` | ≈ 6 GB | run dir on CephFS → raw artifact copy |
| `e1_main` `.sqlite` export | ≈ 31 GB | same |
| `parsed.json` + `parsed.kernels.parquet` + `host_timeline.json` + `kernel_sequences.json` | ≈ 3.4 GB | same |
| metrics jsonl (per-request progress per iteration), request timings, server/replay logs | ≈ 0.5 GB | same |
| `routed_experts/*.npy` + `token_corpus/` | ≈ 0.45 GB | same |
| dry-run pair (no nsys): logs, metrics, routes of 56 requests | ≈ 0.2 GB | same |
| compact tables `phase1a-tables/e1_main/` (frozen extractor) | ≈ 0.3 GB | `phase1a-tables/` + rack 3 mirror |
| **total raw** | **≈ 42 GB** | **budget 80 GB** (2× margin; the matrix's 40 GB under-estimated the sqlite export) |

Locations: run dir `ServingStudioSim/logs/20261114_dsv41_flash_gb300_tp4_e1/<pass>/` (git-ignored); raw copy
`artifacts/phase1c-e1-dsv41-flash-gb300-tp4-<date>/run/` with `SHA256SUMS` (`collect_e1.sh`); compact tables
`phase1a-tables/e1_main/` with `meta.json` input hashes; second copy of the tables, the corpus and the small
derived files on rack 3 `/mnt/shared/home/kevin/dsv41-phase1-mirror/` (the 42 GB raw copy is mirrored only if Kevin
asks). Committed to git: configs, traces, scripts, filtered run logs, `profile_result.json`, launch/drive JSON,
corpus `manifest.json`/`provenance.json`, preflight logs, the checker and estimator outputs (`runs/`, `out/`); not
the parquet, sqlite, nsys-rep, metrics jsonl or routes.

Free space on 2026-11-14: CephFS `/mnt/shared` 21.1 TB free; tray16 root disk (nsys temporary files under `/tmp`)
1.17 TB free. Stop conditions: CephFS < 500 GB free, root disk < 100 GB free (preflight thresholds; the matrix's
100 GB rule is kept for the root disk and tightened for the capture volume).

## 6. Explicit correctness and trace-completeness checks

Pre-launch (`preflight_e1_host.sh`, read-only, 24 checks; run before every launch; the 2026-11-14 output is in
section 7): off-limits tray; no compute apps and 0 MiB on all 4 GPUs; 4 GPUs visible; driver 595.71.05; no leftover
vllm/nsys/req-frontend processes; fork submodule 0ab18711; venv `0.31.1.dev17+g0ab187113` / torch `2.13.0+cu130`;
nsys 2025.6.3.541; weights and enwik9 present; the four JIT/autotune caches non-empty; repo tree clean; E1 traces and
configs match `INPUTS.sha256`; nsys block has the 10 s delay and no bounded stop; CephFS and root disk free space;
no previous `e1_main` output directory; `launcher alignment profile --dry-run` of `e1_main` passes.

Dry-run pair (`checks/e1_completeness.py --compare`):

| check | criterion |
|---|---|
| hook overhead | per cell, median `observed_elapsed_ms` hook-on / hook-off ≤ 1.02 (E1 stop condition) |
| token count | every request emitted exactly 1024 tokens in both runs |
| token identity | greedy decoding, identical prompts: fraction of requests whose generated ids are identical hook-on vs hook-off, and the first divergence position; **reported**; divergence alone is not a stop (batch-dependent kernel numerics can differ between launches); contingency if divergence is widespread: a second hook-off launch to measure the baseline divergence rate before deciding |
| routes present and complete | `routed_experts/*.npy` count = 56; corpus 56 × 1023 rows; join check (below) |
| openai path vs Phase 0 | informational: b = 8 cell median vs Phase 0 b = 8 median 5.20 ms (`vllm_tokens`); a difference > 5 % is reported before `e1_main` |

Post-capture (`checks/e1_completeness.py --inputs E1-INPUTS.json --parsed`), each PASS/FAIL:

| check | criterion |
|---|---|
| export validation | `profile_result.validation.ok`; `nvtx_forward_ranges` = 4 × iterations; parsed devices `[0, 1, 2, 3]` |
| NVTX completeness | every iteration has 28 ranges (7 per device); `devices_absent` empty everywhere; `step_alignment.unpaired_steps_by_device == {}` |
| JIT | iterations with `jit_module_loads` after the first 50: ≤ 20, listed in `jit_excl.txt` (Phase 0 exclusion rule); > 20 means the caches were cold → the capture is not comparable |
| cells | pure-decode iterations per (b, KV, wave) ≥ min(planned lower bound, 400); no iteration carrying requests of two cells (overrun); ramp iterations counted |
| requests | 450 requests seen, each emitted exactly 1024 tokens |
| routes | `.npy` count = 450; corpus `num_tokens` = 450 × 1023 = 460,350; every request segment length 1023; every expert id < 384; for every request, the number of pure-decode iterations with 1 ≤ `output_tokens_before` ≤ 1023 equals its segment length (join check) |
| collective sequence | per cell the modal collective count K (Phase 0 decode: 163) and the number of iterations with another K (`n_other_K`, reported; those iterations are excluded from the cell) |
| clock | collective end spread across ranks: median ≤ 0.02 ms, p90 ≤ 0.05 ms per cell (section 4.3) |
| extraction | `costmodel.extract` (frozen, sha256 369a1bb9…) with `--sha256sums` against the raw `SHA256SUMS`; `meta.json` records the input hashes and `verified: true` |

## 7. Preflight, monitoring, cleanup and abort conditions

Sequence (every step is logged; the run stops at the first failed step):

1. Kevin's go on this packet (Option A or B; full or staged, section 12).
2. Branch `phase1c-e1` from the packet commit; apply `patch/apply_routes_optin.py` (Option A only); run
   `pytest tests/test_alignment_routes_optin.py tests/test_alignment_launcher.py` on tray14 (CPU); commit.
3. Laptop: scheduler check (no GPU pod on tray16), apply the reservation pod, wait for `Running`.
4. Tray16 (`tmux new -d -s e1-dry_off "bash …/run_e1.sh e1_dry_off"`): `run_e1.sh` runs the host preflight (must
   PASS), copies the config into the run dir, launches `uv run --no-sync python -m launcher alignment profile`, logs rc,
   and reports any compute app still on the GPUs 20 s after exit.
5. `e1_dry_on` the same way after the GPUs are clear; then `checks/e1_completeness.py --compare`. Gate: all dry-run
   checks PASS, overhead ≤ 1.02.
6. `e1_main` in tmux `e1-main` (staged mode: only after a second go with the dry-run report).
7. After rc = 0: completeness checks, `costmodel.extract`, `make_partition_1c.py --designation identification`,
   `analysis/e1_identify.py`, `collect_e1.sh`, `SHA256SUMS`, rack-3 mirror, delete the reservation pod, commit, report.

Monitoring during a run (laptop and tray, every 5 min or continuously):

```
tmux attach -r -t e1-<pass>                                   # launcher + req-frontend progress ("requests x/450 | … | elapsed")
watch -n 30 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
grep -c "nvcc\|DeepGEMM.*JIT\|No tuned config" <profile>/vllm/*_server.log          # JIT burst (stop if > 20 modules after warm-up)
grep -i "out of memory\|NCCL\|Traceback" <profile>/vllm/*_server.log                 # OOM / NCCL / crash
df -BG --output=avail /mnt/shared /                                                  # disk
HTTPS_PROXY=socks5://127.0.0.1:11080 kubectl get pods -A -o wide --field-selector spec.nodeName=pod4-gb300-10-tray16-f3   # pre-emption
```

Abort conditions (common, from the matrix): a GPU pod scheduled on the tray or scheduler pre-emption; JIT compile
burst > 20 modules after warm-up (cache cold or wrong cache dir); engine stall > 60 s (no iteration log line);
nsys report write > 10 min **during** inference (cannot happen without a bounded stop, kept as a guard); free disk
below the thresholds; any OOM / NCCL error; configuration drift (fork commit, flags, versions — the preflight and
the launch JSON `fork_git.head`). E1-specific: hook overhead > 2 % in the dry-run pair; a request with a wrong token
count in the dry-run pair. Occupancy maxima of section 3.

Abort procedure: `tmux send-keys -t e1-<pass> C-c` (the launcher terminates the server and nsys); after 60 s check
`nvidia-smi --query-compute-apps=pid,process_name`; kill only the listed PIDs (never `pkill -f` with a pattern that
matches the operator's own command line); keep every partial artifact, append `ABORTED <reason> <time>` to
`run.log`, report. A partial `e1_main` is never analysed as E1; it is kept for provenance only.

Cleanup: verify no compute apps and 0 MiB; `collect_e1.sh`; delete the reservation pod; nothing in the run dir is
deleted; the raw artifact gets `SHA256SUMS`; the compact tables and small files are mirrored to rack 3.

Host preflight output of 2026-11-14 (read-only; the last line fails as expected because the patch is not applied):

```
# E1 host preflight on pod4-gb300-10-tray16-f3 at 2026-11-14T17:56:33Z
PASS  not an off-limits tray
PASS  no compute apps on GPUs
PASS  GPU memory used total 0 MiB
PASS  4 GPUs visible
PASS  driver 595.71.05 (Phase 0)
PASS  no leftover vllm/nsys/req-frontend processes
PASS  fork submodule at 0ab18711
PASS  venv 0.31.1.dev17+g0ab187113 2.13.0+cu130
PASS  nsys 2025.6.3.541
PASS  weights present
PASS  enwik9 corpus 1e9 bytes
PASS  cache warm: deepgemm
PASS  cache warm: triton
PASS  cache warm: nv-compute
PASS  cache warm: flashinfer-autotune
INFO  ServingStudioSim HEAD 2e53072d3206c7e5a557d74451d18e1de01ae0d7 (phase1c-prep)
PASS  repo tree clean (tracked files)
PASS  E1 traces/configs match INPUTS.sha256
PASS  nsys delayed start 10 s, no bounded stop
PASS  no capture_duration_seconds (Phase 0 lesson)
PASS  CephFS free 21100 GB (> 500)
PASS  root disk free 1165 GB (> 100)
PASS  no previous e1_main output dir
FAIL  launcher dry-run e1_main (patch applied?)
# result: PREFLIGHT FAIL
```

## 8. Measurements that separate batch-regime effects from workload/pass effects

What the frozen captures confound. The decode tables with b ≤ 16 come from `diverse_nsys_run2` (Poisson arrivals,
prefill chunks interleaved with decode, KV up to 131 k) and the tables with b ≥ 24 come from `quadrant_nsys`
(saturated 48 in flight, four request shapes with 128–4096 prompt tokens, KV < 8 k). Batch size, KV length, workload
type (interleaving, carry-over of rank skew from a preceding prefill chunk) and pass (launch, day) change together,
so the wait position (70–80 % at b ≤ 16, 25–30 % at b ≥ 24) cannot be attributed. E1 varies b and KV factorially
inside one launch and one workload type, and contains both pure and mixed phases at every (b, KV).

Estimators (`analysis/e1_identify.py`, frozen, per cell = (b, KV, phase), iterations with the modal K):

| statistic | definition | what it separates |
|---|---|---|
| `W_meas[r]` | mean over iterations of Σ_k (duration_rk − min_r' duration_r'k) | measured per-rank wait (Phase 1A definition) |
| `W_med`, `W_ind`, `W_cm` | wait produced by `critical_path_batch` from the cell's own empirical per-(rank, segment) distributions: median path; independent draws per (rank, segment); common-mode draws (same quantile on every rank) | the three composition references of the Phase 1C-0 audit, now per cell |
| `f_ind` | (W_meas − W_cm) / (W_ind − W_cm), mean over ranks, bootstrap 90 % CI over iterations (200 resamples) | the wait position; the quantity whose b dependence is in question |
| `rho`, `c_common` | variance-weighted mean pairwise cross-rank correlation of per-segment deviations; normalized common-mode variance fraction | direct measure of common-mode vs independent segment variation, independent of the barrier model |
| `first_arrival_spread`, `last_arriving_rank_hist`, `mean_lateness_ms_per_rank` | spread across ranks of the arrival at the first collective (median, p90); which rank arrives last; mean lateness per rank | host-arrival contribution; the b = 1–2 pattern (Phase 1A: 0.45–0.9 ms, pass-dependent) |
| `collective_end_spread` | max − min across ranks of each collective's end | clock/alignment check (section 4.3) |
| `moe_cross_rank_corr`, `r2_wall_on_routing`, `r2_moe_on_routing` | cross-rank correlation of the MoE group time; R² of wall and MoE time on per-iteration routing statistics | routing is common-mode by construction under TP without EP (every rank runs every expert on its shard): a high MoE cross-rank correlation and R² identify routing as the common-mode source |

Pre-registered contrasts and thresholds (identification outcomes; they choose the structure of model v1.1, which
E2 validates; nothing is fitted in E1):

| contrast | computed as | threshold |
|---|---|---|
| batch regime | at each KV: mean f_ind over b ∈ {24, 32, 48} minus mean over b ∈ {1, 2, 4, 8, 16} (pure phase) | regime effect if Δ < −0.2 at every KV with bootstrap CIs excluding 0 |
| KV | at each b: range of f_ind across the three KV levels | KV effect if range > 0.2 at two or more b |
| workload / phase | at each (b, KV): f_ind(mixed) − f_ind(pure) | interleaving effect if \|Δ\| > 0.2 at two or more cells with n ≥ 50 (mixed phase) |
| pass | at matched (b, KV bucket): E1 f_ind minus the frozen calibration-row value (b ≤ 16 against run2 by KV bucket, b ≥ 24 at KV 4k against quadrant kv < 8k) | pass effect if \|Δ\| > 0.15 beyond the CI at two or more b, after regime, KV and phase are accounted for |

Interpretation (all subject to E2): regime effect only → mixture indexed by b; KV effect → mixture indexed by KV;
phase effect → carry-over term from the preceding iteration (already a sequence-simulator concept); pass effect
without the others → the decode band must carry a pass-level dispersion that the production model reports as a
lower-confidence status rather than models. If f_ind is flat at a single value, the Phase 1B structure is kept with
that value as a candidate parameter for E2.

Smoke evidence that the estimator reproduces the frozen audit on calibration rows (frozen tables, no held-out row
read; `out/smoke_quadrant/`, `out/smoke_run2/`): quadrant b = 24 / 25 / 48 → f_ind 0.35 / 0.36 / 0.24 against the
frozen 0.32 / 0.32 / 0.22; run2 b = 6 / 7 / 8 / 12 → 0.76 / 0.75 / 0.78 / 0.72 against 0.76 / 0.79 / 0.78 / 0.68;
cross-rank ρ 0.78–0.83 at quadrant b = 24–25 versus 0.2–0.5 at b ≤ 13 in both passes; first-collective arrival
spread 0.016–0.021 ms for b ≥ 3 and 0.48–0.9 ms for b = 1–2 with rank 0 or rank 2 arriving last (pass-dependent).
These are smoke outputs, not findings; E1 is designed to measure them under controlled conditions.

## 9. Independence of E1 from the frozen validation partition

- The Phase 1A partition (`phase1a/partition/partition.json`, sha256 288e193f…) is not edited (pinned by
  `phase1c/tests/test_invariants.py`); its held-out rows are not read by any E1 script. `analysis/e1_identify.py`
  refuses to open a frozen table without `--calibration-only <partition.json>` and then reads calibration-role rows
  only; the frozen reference values it compares against come from the committed coefficients, which were fitted on
  calibration rows. The smoke tests in `out/` were run that way.
- E1 is a new capture: a different launch, day, trace, prompts and request mix from every frozen capture; no E1
  iteration is a row of any frozen table.
- E1 carries the designation **identification** (protocol section 2), fixed here before the capture. Its iterations
  are written once by `make_partition_1c.py --designation identification` into `phase1c/partition/partition-1c.json`
  (Phase 1A layout: `passes.e1_main.{role, cls, exclusions, counts}`, protocol hash recorded), so the frozen
  `Partition.calibration_only` leakage guard applies unchanged and a row's role can never be changed afterwards.
- Nothing estimated from E1 is validated by E1 (protocol sections 2 and 4). The validation is E2: a separate launch
  on a different day, a different request-order seed and prompt subset, the production backend (`vllm_tokens`), no
  routes, designated `validation`, compared one-shot against a model version whose hash is recorded before E2 runs.
  A model revision informed by E1 is a new version with a new hash; the Phase 1A/1B coefficients, parameters and
  predictions stay frozen and reproducible (`phase1c/repro/verify.py`).
- Per Kevin's restrictions, E1's f_ind values are reported, not converted into fitted correlation weights, the
  Monte Carlo dispersion factor is not adjusted, and the Phase 1C-0 uncertainty tables are preserved.

## 10. Exact scripts, hashes and commits

Frozen inputs the capture and its analysis depend on:

| artefact | identity |
|---|---|
| ServingStudioSim | branch `phase1c-prep` = 2e53072d3206c7e5a557d74451d18e1de01ae0d7 (Phase 1C-0 accepted); this packet is committed on `phase1c-e1-packet` on top of it (commit id in the report message); Phase 0 `test-ss` ca9f8ed0 |
| vLLM fork | `servingstudio-alignment-v0.31.0` = 0ab18711335780e062d65994a3243da0281d2081 (kaijunli-infr/vllm), submodule pin unchanged; engine `0.31.1.dev17+g0ab187113` |
| Phase 0 scripts | `scripts/run_pass.sh` 9b1fdf65113f6b8ca93c256af4d26a4cf2acbb8c5c25e85c069b591a4f508b31; `dsv41-gb300/make_configs.py` 92b3f80750f8daa9742e55b7ed3af520e3e8d0f476b3036b5d5f0f4c2e097366; `scripts/env.sh` bcf84d6fabace0014ce7846e3b23071c34055d8089cec887aacc4fc3ec722919 |
| Phase 1A extractor and model code (frozen) | `costmodel/extract.py` 369a1bb9f59a3e6a33462c2ecf520f30c8c82018a73f93e4c4b025808539668b; `data.py` 204dbcdb134579ab1f87e9bda60bd3faab220a88958bf1df9df212b0312e10d9; `compose.py` d07c41685718e8df9c9166522632dd9129b1bc062f823c528a73a25ef4dd914d; `decode.py` 790fcee5661128a9eca4db8d737b39032cd2219939282bbd8e2d6ee37f7a216a; `predict.py` 5da68e39f5a1f960de92d59e7ca4e5c8f89c766820413e9436146e7b0c1126c7 |
| Phase 1A / 1B artefacts (frozen) | `coefficients.json` 78f937c5…; `partition.json` 288e193f…; `params.json` 8dc79573… (full hashes in `phase1c/repro/MANIFEST.json`) |
| protocol / matrix | `VALIDATION-PROTOCOL.md` 07ab7d7171ae3185d44bcc92c5a5b6ef0f03c3bb8d82eec27461d429e80327be; `GPU-EXPERIMENT-MATRIX.md` 869f0d06171a5b62e9c0fe673546921b3329cb37f2fdbe05c9c431591394079e |
| software on tray16 | nsys 2025.6.3.541-256337736014v0; driver 595.71.05; kernel 6.14.0-1013-nvidia-64k; Python 3.12 (uv-managed); torch 2.13.0+cu130 |
| weights / corpus | DeepSeek-V4.1-Flash-1017 HF 08c8657f; enwik9 1,000,000,000 bytes |

E1 files in this directory (`phase1c/e1/`; every hash in `SHA256SUMS`, inputs also in `INPUTS.sha256`):

| file | role |
|---|---|
| `make_e1_inputs.py` | writes `traces/e1_main.csv` (2e147402b5a24f9e…), `traces/e1_dry.csv` (c6d8e13b7229ffd4…), the four `configs/*/profile.json`, `E1-INPUTS.json`, `INPUTS.sha256` (deterministic) |
| `preflight_e1_host.sh` | read-only host preflight, 24 checks (section 6) |
| `run_e1.sh` | the driver (one pass per invocation; preflight gate; Phase 0 launcher invocation; teardown report) — **not run** |
| `checks/e1_completeness.py` | correctness and completeness checks, dry-run overhead comparison, `jit_excl.txt` |
| `analysis/e1_identify.py` | pre-registered estimators and decision statistics (section 8) |
| `make_partition_1c.py` | the Phase 1C partition file (section 9) — run once per capture |
| `collect_e1.sh` | small derived files into `runs/`, raw copy + `SHA256SUMS` |
| `patch/apply_routes_optin.py`, `patch/launcher-routes-optin.patch`, `patch/test_alignment_routes_optin.py` | the launcher change (section 4.4) — **not applied** to any branch |
| `out/` | CPU smoke outputs on frozen calibration rows and the tray16 preflight log |

Post-capture commands (exact; `$S=/mnt/shared/home/dc/workspace/serving-studio`, `$D=$S/ServingStudioSim/profiling/env/arm64-gb300/dsv41-flash-tp4`, `PY=$S/gb300/venv/bin/python`, `P=$S/ServingStudioSim/logs/20261114_dsv41_flash_gb300_tp4_e1/e1_main/profile`):

```
$PY $D/phase1c/e1/checks/e1_completeness.py --profile $P --inputs $D/phase1c/e1/E1-INPUTS.json --parsed --out $D/phase1c/e1/out/e1_main/completeness.json
cd $D/phase1a && $PY -m costmodel.extract --profile $P --label e1_main --out $S/phase1a-tables/e1_main --excl $D/phase1c/e1/out/e1_main/jit_excl.txt \
    --sha256sums $S/artifacts/phase1c-e1-dsv41-flash-gb300-tp4-<date>/SHA256SUMS --sha-prefix run/e1_main/profile
$PY $D/phase1c/e1/make_partition_1c.py --tables $S/phase1a-tables/e1_main --label e1_main --designation identification --excl $D/phase1c/e1/out/e1_main/jit_excl.txt --out $D/phase1c/partition/partition-1c.json
$PY $D/phase1c/e1/analysis/e1_identify.py --tables $S/phase1a-tables/e1_main --label e1_main --metrics $P/vllm/*_metrics.jsonl --corpus $P/token_corpus --out $D/phase1c/e1/out/e1_main
bash $D/phase1c/e1/collect_e1.sh
```

CPU test results of this packet (tray14, 2026-11-14): `make_e1_inputs.py` reproduces the committed traces and
configs byte-for-byte (`INPUTS.sha256`); `launcher alignment profile --dry-run` validates `e1_dry_off` and
`e1_main_norouts` on the unpatched tree and all four configs with the patch; patch tests 5 passed; launcher tests
134 passed, 1 pre-existing environment failure; `checks/e1_completeness.py --parsed` on the Phase 0 quadrant profile:
PASS (28 ranges on all 4,750 iterations, no unpaired steps, 10 JIT iterations → `jit_excl.txt`);
`analysis/e1_identify.py --calibration-only` on quadrant (9 s) and run2 (30 s): outputs in `out/smoke_*`; host
preflight on tray16: 23 PASS, 1 expected FAIL (patch not applied).

## 11. Deviations from the matrix text (for the record)

| matrix | this packet | reason |
|---|---|---|
| "≥ 400 iterations each" | ≥ 736 per cell, ≥ 1,000 for b ≤ 8 (2 waves) | more repetitions at low b where the variance is largest; cost is small |
| "24 short nsys runs or 3 runs" | one launch, one nsys session, trace-timed cells | startup is ≈ 25 min per launch; one session keeps every cell in one timeline |
| "routing hook: a reviewed vLLM-fork patch" | no fork change; existing capturer + 46-line launcher patch | the fork already has the capturer; the launcher only lacked the opt-in for a timing pass |
| "OS-runtime (launch API) tracing" (common setup) | `--trace=cuda,nvtx` as Phase 0 | launch timestamps come from the CUDA API trace; `osrt` adds overhead and breaks comparability |
| "≈ 60 min wall + 20 min export; reservation 2 h" | ≈ 1 h 45 min GPU-held, ≈ 3 h 15 min wall; reservation 5 h | dry-run pair (two launches) and larger trace |
| "budget 40 GB" | expected 42 GB, budget 80 GB | the sqlite export is ≈ 5 × the `.nsys-rep` |
| "any divergence of the generated tokens … (hook must be observation-only)" | token identity is measured and reported; a stop only on wrong token counts or widespread divergence after a baseline launch | greedy outputs can differ between launches through batch-dependent kernel numerics regardless of the hook; the hook's observation-only nature is established by code (unchanged fork) and by the overhead measurement |
| request backend | `openai` (Option A) instead of `vllm_tokens` | routes are returned only through the OpenAI protocol; Option B keeps `vllm_tokens` and drops routes |

## 12. Decisions requested with the go

1. **Option A** (routes, openai backend, launcher patch applied on the E1 branch; recommended: it answers the full E1
   question) or **Option B** (`e1_main_norouts`: timing only, `vllm_tokens`, no code change; answers the common-mode
   / independent / b / KV questions but not routing).
2. **Staged** (stop after the dry-run pair and post its report for a second go before `e1_main`; recommended) or
   **full** (continue automatically when every dry-run check passes).
3. Tray16 (default) or another free rack-10 tray.
4. Whether the 42 GB raw capture is mirrored to rack 3 (the tables, corpus and derived files are mirrored in any case).
5. Reservation maximum 5 h.

---

## A. Verbatim: `GPU-EXPERIMENT-MATRIX.md` — common setup and E1 (sha256 869f0d06…)

## Common setup

| item | value |
|---|---|
| hardware | one rack-10 GB300 tray (4 GPUs, TP4), reserved through `k8s/dsv41-gb300/reserve-tray16.yaml` (or another free tray) for the whole experiment; host-level GPU use only on a tray with no GPU pods, and still pre-emptible by the scheduler (stop condition) |
| software | the Phase 0 recipe exactly: vLLM fork `servingstudio-alignment-v0.31.0` (0ab18711), uv Python 3.12, TP4, chunk 16384, CUDA-graph capture up to 256 tokens, warm JIT/autotune caches under `gb300/cache/`; weights `/mnt/shared/weights/DeepSeek-V4.1-Flash-1017` (HF 08c8657f) |
| capture | nsys with CUDA, NVTX and OS-runtime (launch API) tracing, delayed start 10 s, **no bounded stop** (Phase 0 lesson: an in-inference `cudaProfilerStop` froze the engine); per-rank host NVTX ranges as in Phase 0 (preprocess / forward / postprocess / sample / bookkeep / eplb) plus the vLLM scheduler step range; `parsed.json` from the Phase 0 parser; compact tables from `costmodel.extract` (hash-recorded) |
| attribution | kernel → segment via the Phase 0 collective-kind sequence (K collectives per iteration), kernel names → groups (fmha / indexer / sparse_meta / moe / comm / other) via `costmodel.groups`; host launches via the CUDA API trace (`cuLaunchKernel`, `cudaGraphLaunch`) per rank; per-iteration ids = `iteration_index` |
| storage | nsys `.nsys-rep` + exported sqlite + `parsed.json` + compact tables; Phase 0 averaged ≈ 5 GB per nsys pass; each budget below includes a 2× margin; raw captures to `artifacts/<capture>/` with `SHA256SUMS` + `MANIFEST.md`; tables to `phase1a-tables/<label>/`; mirror on rack 3 |
| stop conditions (all experiments) | tray not free or a GPU pod scheduled on the tray; scheduler pre-emption; JIT compile burst > 20 modules after warm-up (cache cold = wrong cache dir); engine stall > 60 s; nsys report write > 10 min; free disk on the capture volume < 100 GB; any OOM / NCCL error; a configuration drift (vLLM commit, flags) from the recipe |
| exclusions (pre-registered) | JIT-affected iterations by the Phase 0 rule; the first 50 iterations after engine start (warm-up); iterations with `_kind_mismatch` in the collective sequence |

Expected-uncertainty-reduction ranking uses the Phase 1B run2 figures: share of the workload Σ walls per class × the
current spread between what the model can and cannot pin down for that class (median-path vs stochastic-mean bias,
p90 APE, or extrapolation status). Workload shares (run2): decode b3-16 and b≥25 ≈ 79 % of Σ walls, 16k chunks ≈ 19 %,
eager mixed ≈ 2.6 %, decode b1-2 ≈ 7 % of iterations but ≈ 6 % of walls, graph-kind mixed < 0.5 %.

## E1 — decode routing and rank-arrival capture (identification)

| field | content |
|---|---|
| unresolved question | Why do the median path and the stochastic mean of decode walls bracket the measurement by ≈ 3 % (run2 b3-16: −2.8 % / +0.5 %; quadrant: −1 % / +2 %)? Is the per-rank segment variation common-mode (same kernels, routing-driven) or independent (host-arrival driven)? What is the per-iteration routed-expert distribution and how much wall variance does it explain? |
| workload and configuration | synthetic decode-only runs at fixed batch b ∈ {1, 2, 4, 8, 16, 24, 32, 48} (≥ 400 iterations each) at three KV levels (≈ 4k, 32k, 96k tokens per request), the diverse corpus prompts; same recipe; plus the **routing hook**: a reviewed vLLM-fork patch that logs per-layer routed expert ids (or expert load counts) per iteration to a side file with the iteration index (patch reviewed and merged on the fork branch before the capture; no behaviour change) |
| why existing traces cannot answer | the Phase 0 traces have no routing information (gap G3); per-rank arrival at the first collective is measured, but with mixed batch sizes and interleaved prefill the common-mode vs independent question is confounded by the preceding iteration's skew carry-over; b=1–2 at kv<32k shows pass-dependent rank-lateness patterns (Phase 1A) |
| expected measurements and attribution | per iteration and rank: segment windows (R, K+1), collective intrinsic and wait, first-collective arrival spread, host launch times (graph launch), scheduler gap; routing: expert ids / loads per layer; derived: correlation of per-rank segment deviations across ranks (common-mode fraction), wall variance explained by routed-expert load, per-b dispersion quantiles by KV |
| designation | **identification** (new parameters: common-mode fraction of decode dispersion, routing-conditional MoE time, b1-2 arrival pattern). Results are never used for validation; E2 validates. |
| GPU occupancy and reservation | 1 tray × 4 GPUs, ≈ 60 min wall (8 b × 3 KV × ≈ 1.5 min + model load + warm-up) + 20 min nsys export; reservation pod for 2 h |
| storage and profiling budget | 24 short nsys runs or 3 runs (one per KV level) with the batch sweep inside: ≈ 3 × 6 GB nsys + sqlite ≈ 25 GB raw, ≈ 1 GB tables, routing log ≈ 200 MB; budget 40 GB |
| stop conditions (specific) | routing hook adds > 2 % to iteration wall in a dry run without nsys (measured against Phase 0 b=8 median 5.2 ms); any divergence of the generated tokens from a no-hook run on the same prompts (hook must be observation-only) |
| decision enabled | choose between common-mode and independent per-rank sampling (or a fitted mixture) for the decode band; decide whether a routing-conditional MoE term replaces the corpus band; decide whether b1-2 needs its own arrival model or stays lower-confidence |
| priority | **1** — decode is ≈ 79 % of the workload total; the 3 % median/mean spread is the largest single uncertainty in the full-workload prediction (≈ 2 % of span) |

## B. Verbatim: `VALIDATION-PROTOCOL.md` v1.0 — provisions that govern E1 (sha256 07ab7d71…)

## 1. Frozen artefacts

| artefact | identity | rule |
|---|---|---|
| Phase 1A coefficients | `phase1a/out/coefficients.json`, sha256 78f937c5… | never modified; a revised model writes a new file with a new hash |
| Phase 1A partition | `phase1a/partition/partition.json`, sha256 288e193f… | never modified; its held-out rows remain held-out for every future model version (they were looked at once for Phase 1A and once for Phase 1B and are therefore *reported*, not *validation*, from now on) |
| Phase 1B parameters and predictions | `phase1b/out/params.json` (8dc79573…), `results.json`, per-iteration parquet (canonical checksums in `phase1c/repro/MANIFEST.json`) | never modified; reproducible through `phase1c/repro/verify.py` |
| the three Phase 0 captures | run2, quadrant, salvaged compact tables (hashes in the manifest) | may be used for identification of new parameters only in their calibration-role rows; their held-out rows are not used for anything new |

## 2. Designations

Every new capture receives exactly one designation **before** it runs, written in the approved experiment matrix:

- **identification**: its iterations may be used to estimate parameters, choose model structure, or set bands. Nothing
  estimated from it is validated by it. Its rows get role `identification` in the Phase 1C partition.
- **validation**: its iterations are used once, for a pre-registered comparison, after the model version under test
  is frozen (hash recorded). Role `validation`. No parameter, threshold, band, exclusion rule or class definition may
  be changed in response to a validation result; a change produces a new model version that needs a *new* validation
  capture. A validation capture is never re-used for a later model version.
- **identification + validation half**: two separate vLLM launches (separate runs, different request-order seeds); the
  halves are assigned at capture time by run, never by picking iterations after looking at them.

A capture is called *held-out* only if it carries the validation designation and has not been opened for any purpose
other than the one-shot comparison. Describing an identification capture as held-out, or using a validation capture's
results to refit, is a protocol violation and the affected results are reported as "calibration".

## 3. Partition

New captures are recorded in a **new** partition file `phase1c/partition/partition-1c.json` (role per iteration:
`identification` / `validation` / `excluded`), hash-recorded, with the same leakage guard as Phase 1A
(`Partition.calibration_only` refuses fits on non-identification rows). The Phase 1A partition file is not edited.

## 4. Independence requirement

When a capture identifies a missing or previously unidentifiable parameter (c_kv, rank-2 offset, common-mode fraction,
routing-conditional MoE term, first-chunk profile, Q cap, indexer slope beyond 115k), its validation must come from a
**different** capture: a separate vLLM launch, a different request-order seed and prompt subset, and — for E1/E2 — a
different day. Validation of a parameter on the capture that identified it is reported as in-sample and does not count.

## 5. Pre-registered comparison (one shot per model version)

Metrics, computed by the frozen scripts (`phase1b/seqsim/evaluate.py`, `phase1c/uncertainty/reliability.py`, hashes
recorded) in the four Phase 1B modes with 32 draws, seed 0, plus seeds 1–4 for seed variation:

per class (decode b1-2, b3-16, b17-24, b≥25; prefill 16k chunk by kv0 band; mixed-eager host-bound; mixed-ring other;
mixed-graph) and role: n, signed bias (ms, %), median and p90 absolute error, p90 APE, worst case, p10–p90 coverage,
mean and median interval width, fraction below/above; full-workload span error, Σ walls signed and absolute error,
throughput; lead and host-skew tables by transition.

Acceptance for a model version to be eligible for Phase 1C-2 integration (all on validation captures):

| class | acceptance |
|---|---|
| decode b3-16, b17-24, b≥25 | p90 APE ≤ 5 %, |signed bias| ≤ 2 %, coverage p10–p90 within 70–90 % |
| decode b1-2 | p90 APE ≤ 10 % or explicitly flagged low-confidence in the output (status propagated) |
| prefill 16k chunk, kv0 ≤ 160k | p90 APE ≤ 3 %, coverage 70–90 % |
| mixed-eager host-bound | p90 APE ≤ 10 % or flagged low-confidence |
| mixed-ring other / mixed-graph | p90 APE ≤ 8 % or flagged |
| full workload (span and Σ walls) | |error| ≤ 1 % on every validation pass |
| lead on chains | p90 abs error ≤ 20 ms |

Classes with n < 30 on a validation capture are reported but cannot pass or fail; they remain flagged.

## 7. Prohibited

Tuning on validation rows; widening bands to raise coverage without an identified source of dispersion; global
multipliers; re-running a validation comparison with changed exclusions; editing this file for an in-flight capture.

