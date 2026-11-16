# E1 b=8 Nsight recovery pass `e1_b8_recovery`: validation report (2026-11-16, GO on commit b21a52fb)

**Result: all scientific acceptance gates PASS.** One launch, rc=0, 1008/1008 pure b=8 decode iterations captured with no
truncated or straddling iteration, four-rank NVTX and iteration alignment complete, collective end-time spread far inside
the frozen thresholds, routing joins complete, 0 JIT-affected iterations, export validated and checksummed. No deviation
from the approved capture protocol. GPU-held 28 min 01 s, reservation 31 min 50 s (limit 60), raw artifacts 1.1 GB (limit 5).

Commit under test: `phase1c-e1-stage15` b21a52fb (worktree `wt-e1stage1`, clean). Config `configs_v2/e1_b8_recovery/profile.json`
sha 1f8f9738…, trace `traces/e1_b8_recovery.csv` sha 398c317e… (`INPUTS_b8.sha256` f51e5792…, `SCRIPTS_b8.sha256` 6ac02dc3…),
req-frontend gitlink/checkout 5c20a259 tree 06a4a61e, `session_runner` sha 7f6eb81d… (== BINARY-PROVENANCE.json, mtime unchanged
after the run), nsys 2025.6.3.541. Thresholds, model, kernels, serving configuration: unchanged (checked, §6 below).

## 1. Required preflight (GO items 1–8), all before GPU use

| # | item | evidence | result |
|---|---|---|---|
| 1 | HEAD exactly b21a52fb | `git rev-parse HEAD` = b21a52fb1816fb599755910897e093ba2ed2a4c9, 0 dirty files, branch phase1c-e1-stage15; remote head identical | PASS |
| 2 | `SCRIPTS_b8.sha256` and frozen input manifest | 19/19 OK; `INPUTS_b8.sha256` 4/4 OK (config, trace, nsys_val config, e1_dry.csv); `INPUTS_v2.sha256` 5/5 OK | PASS |
| 3 | pinned req-frontend source 5c20a259 and binary provenance | gitlink 5c20a259; checkout HEAD 5c20a259 on `e1-stage15-stream-flag`, tree 06a4a61e, clean; `target/release/session_runner` 28759968 B sha 7f6eb81d45e6… == BINARY-PROVENANCE.json; `.gitmodules` URL still uw-syfi | PASS |
| 4 | four GPUs available, reservation exclusive | tray16 before: 4 × 0 MiB, 0 compute apps, no tmux server, 0 logins; node `nvidia.com/gpu` 0/4 allocated, no GPU pod; reservation `dsv41-gb300-reserve-tray16` applied 05:33:37Z, Running 05:34:16Z on `pod4-gb300-10-tray16-f3`, afterwards the only GPU-requesting pod on the node | PASS |
| 5 | preflight in the exact tmux + `tee` context | `preflight_tray16_driver_context_20261116T053256Z.log`: tmux `e1b8-ctxtest`, `bash preflight_e1_b8.sh 2>&1 \| tee <run root>/ctxtest/preflight.log`: 18 PASS, 0 FAIL | PASS |
| 6 | invocation pipeline is INFO, not a lookalike | same log: `INFO own invocation pipeline (excluded from candidates): [bash] [bash] [tee] [sed]`, `INFO own tmux session (excluded): e1b8-ctxtest`, process discovery CLEAN, registry plan clean | PASS |
| 7 | genuine independent lookalikes still cause STOP | `negative_control_tray16_20261116T053230Z.log`: an independent `setsid`'d bash whose argv names the run root (registered only in a scratch registry) → `FOUND … pids 302158`, `LOOKALIKE pid 302158 … NOT owned, not touched`, `PREFLIGHT FAIL`, rc 1, nothing signalled; then removed only by `e1_pids.py cleanup --execute` on the scratch registry (identity-verified, 2 SIGTERM, survivors 0); the own pipeline stayed INFO throughout | PASS |
| 8 | registry, cleanup protections, b=8-only trace guard | preflight: trace == `head -9 e1_dry.csv`, 8 rows, 0 b48 rows, config nsys block == Phase 0 quadrant, no `capture_start_delay_seconds`; registry path absent (empty) before launch; driver guard line (8 rows, no b48, config path) and registration line present and unchanged | PASS |

Capacity: CephFS 20.66 TB free, tray16 root 1165 GB free (≥ 5 GB).

## 2. Timeline (UTC, 2026-11-16)

| time | event |
|---|---|
| 05:33:37 | reservation pod applied; Running 05:34:16 |
| 05:35:02 | launch: `tmux new -d -s e1b8-recovery "E1_REPO=$WT bash $E1/run_e1_b8.sh"` (packet §7 line, unchanged; typed through a neutral-path wrapper that exits immediately) |
| 05:35:18 | driver's own preflight PASS; driver pid 332134 registered (`pids/owned.tsv`, start time 280129747); launcher started on b21a52fb |
| 05:36 | four `VLLM::Worker_TP0..3` on the four GPUs, all descendants of the registered driver; monitored every 60 s: never a foreign GPU process, reservation always the only GPU pod |
| ≤ 05:58:35 | `Application startup complete` (≤ 23 min 17 s after the launcher start; limit 30 min) |
| 05:59–06:00 | warmup (serial + saturated), prefix cache reset, external nsys launched, frontend ready, measurement (replay 9.613 s) |
| 06:01 | engine shut down (shutdown trigger at server-log line 3831; EngineDeadError-family lines: none), GPUs 0 MiB; sqlite export ok; 1040 iterations / 5 887 244 kernel rows parsed |
| 06:03:03 | driver `rc=0` (GPU-held 28 min 01 s from launch) |
| 06:04:39 | closeout (`closeout_tray16_20261116T060439Z.log`): GPUs 4 × 0 MiB, 16 PASS process checks, discovery CLEAN, registry: driver GONE, dry-run plan 0 processes, 0 lookalikes; no tmux server |
| 06:05:08 | reservation pod deleted; gone 06:05:27; node 0/4 GPUs allocated, no GPU pods (reservation 31 min 50 s) |
| 06:06:29 | post-run CPU gates on tray14 (`postrun_20261116T060629Z.log`): GATES PASS |

## 3. Profiler armed before the first request

`capture_start_delay_seconds: null` in the launch record (`*_launch.json`, nsys_config), capture mode `cuda_profiler_api`,
`cuda_graph_trace node`, `cuda_event_trace false`, identical to the Phase 0 quadrant block. With no delay the launcher calls
`/start_profile` synchronously before `replay_start_monotonic_ns` and before the first request (`alignment/runner.py`). Evidence
in the data: the first EngineCore metrics iteration of the measurement (index 85, starting 0.0384 s after replay start) is the
first parsed iteration; parsed = metrics = 1040 iterations, contiguous, none missing; kernel span 8985.8 ms vs replay 9.613 s.

## 4. Scientific acceptance (frozen thresholds; checker outputs in `e1_b8_recovery/cpu/`)

| # | requirement | observed | result |
|---|---|---|---|
| 1 | b=8 pure-decode coverage 1008/1008 | census (`analysis/e1_capture_window.py`): captured pure_b8 1008 of 1008; wave 85 (0.038 s) → first pure 102 (3.615 s) → last pure 1109 (ends 8.948 s), wave end 9.022 s; completeness v2: cell b=8 kv=32768 pure-decode iterations 1008 (need ≥ 400), ramp 30 | PASS |
| 2 | no truncated or straddling iterations | parsed iterations 85..1124 contiguous, `missing in metrics=[]`, 1040 of 1040 metrics iterations captured; captured kinds prefill_only 2 + mixed_b8 15 + decode_ramp_b8 15 + pure_b8 1008; overrun iterations 0 | PASS |
| 3 | complete four-rank NVTX ownership and iteration alignment | `nvtx_forward_ranges` 4160 == 4 × 1040; iterations without 28 NVTX ranges 0; devices_absent 0; NVTX iteration index differing between devices 0; unpaired steps {}; start sequence strictly increasing on all devices; GPU kernel windows overlap on 1040/1040 iterations (first-kernel start spread median 0.0042 ms, p90 0.0090 ms, max 0.355 ms) | PASS |
| 4 | collective end-time spread within frozen thresholds (median ≤ 0.02 ms, p90 ≤ 0.05 ms) | b=8 (n=164 304 collectives, 1008 iterations): median 0.0010 ms, p90 0.0080 ms; modal K 163 in all 1008 iterations, n_other_K 0; ramp cells b=1..7 also PASS (medians 0.0009–0.0011 ms); pure-decode iterations without rank-aligned collectives 0 of 1023 | PASS |
| 5 | routing-to-iteration joins complete | completeness v2.0.0: duplicate joins 0, missing joins 0, every request's position→iteration map strictly increasing; 8 routed_experts files shape (1024, 40, 6) uint16, expert ids < 384; corpus 8184 tokens = 8 × 1023, segments == npy[1:], every segment row maps to exactly one iteration (8 of 8) | PASS |
| 6 | JIT exclusions | JIT-affected iterations after warm-up 0 (limit 20); clean pure b=8 iterations remaining 1008 | PASS |
| 7 | valid, checksummed trace export | launcher `validation.ok True` (kernel_rows 5 887 244, kernel_span_ms 8985.76); `.nsys-rep` 170 126 141 B sha d239986d19124c7d…, `.sqlite` 874 708 992 B sha 2c43621e41d64df4…, parsed.json sha b3801caaa505e014…; archive `artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115/run/e1_b8_recovery/` with `SHA256SUMS.b8` (40 files) | PASS |
| — | workload identity vs `e1_dry_on_v2` and `e1_nsys_val` (8 requests: prompts, output lengths) | RESULT PASS for both references | PASS |
| — | configuration identity vs `e1_nsys_val` | profile configs identical apart from the allowed `log_dir`, `name`, `nsys.capture_start_delay_seconds`, `workload.frontend.path`; server argv identical; server env identical; the nsys argv differs only in the output path. The checker's three FAIL lines (argv output path, `capture_start_delay_seconds`, derived summary) are exactly these by-design differences; informational, as the postrun script states | OK |

## 5. Per-rank execution, collective wait, critical path, kernels (cell b=8, 1008 pure-decode iterations; `cpu/rank_kernels_e1_b8_recovery.md`)

| rank | GPU span ms p50 / p90 | busy p50 | compute union p50 | comm union p50 | idle p50 | collective wait sum p50 / p90 ms | wait share of comm p50 | kernels / iter |
|---|---|---|---|---|---|---|---|---|
| 0 | 5.2789 / 5.3531 | 5.1305 | 4.4999 | 0.6944 | 0.1489 | 0.2196 / 0.2799 | 0.309 | 1410 |
| 1 | 5.2791 / 5.3526 | 5.1286 | 4.5538 | 0.6394 | 0.1490 | 0.1637 / 0.2166 | 0.249 | 1409 |
| 2 | 5.2778 / 5.3531 | 5.1255 | 4.5616 | 0.6287 | 0.1512 | 0.1535 / 0.1959 | 0.238 | 1409 |
| 3 | 5.2785 / 5.3532 | 5.1268 | 4.5496 | 0.6448 | 0.1509 | 0.1695 / 0.2192 | 0.256 | 1409 |

* Critical path: rank 2 is the least-waiting (straggler) rank, rank 0 the most-waiting; latest-ending rank counts {2: 575, 3: 362, 0: 67, 1: 4}; earliest-starting {1: 777, 0: 174, 3: 40, 2: 17}. 163 collectives per iteration; collective end spread p50/p90/max 0.00096 / 0.00795 / 0.0249 ms. Rank 0 carries ~0.05 ms more comm per iteration than ranks 1–3 (its compute union is ~0.05 ms shorter), i.e. it waits for the others.
* Kernel identities: 95 distinct names, ~1410 kernels and 46–47 graph launches per rank per iteration. Top GPU-time kernels: MXFP4 block-scaled MoE `bmm_MxE4m3_MxE2m1MxE4m3_…_t128x8x256_s6` (148/iter, 27.2 µs, 14.1 % of GPU time), FlashInfer dense block-scaled GEMM sm100 (520/iter, 5.1 µs, 9.3 %), `bmm_Bfloat16_MxE2m1MxE4m3_…_t128x8x256_s6` (160/iter, 16.3 µs, 9.2 %), deep_gemm tf32 hc prenorm (316/iter, 6.0 µs, 6.7 %), splitk block-scaled GEMM (320/iter, 5.6 µs, 6.3 %), cute-DSL all-reduce mhc Lamport (308/iter, 5.7 µs, 6.1 %), mhc_pre_big_fuse tilelang (332/iter, 5.2 µs, 6.1 %), fmha sm100 paged-KV token-sparse (152/iter, 10.4 µs, 5.5 %).
* Tactic mix: the MoE bmm runs almost entirely on the `t128x8x256_s6` tactic (148/iter) with a `t128x8x512_s3` variant on 12/iter (28.1 µs); the dense GEMM splits 520 non-splitk vs 320 splitk launches per iteration; two all-reduce mhc variants (Lamport 308/iter, Quad 148/iter, Shar 160/iter). Full table in the md/json.
* The tool's cross-session reference section against Phase 0 `quadrant_nsys` is empty (`ref_n_iterations 0`: no quadrant iterations matched the b=8 cell under the tool's selection). Not investigated further (stop boundary); hypothesis material only, as the postrun header says.

## 6. Deviations from the approved capture protocol

None in workload, model, serving configuration, Nsight settings, thresholds or cleanup policy. Operating notes:
* The attempt-1 `preflight.log` left in the pass directory was renamed to `preflight.attempt1_20261115T182633Z.log` (sha 6703c25f…, identical to the committed copy) before launch so the new run's preflight log is unambiguous; it is in the archive.
* The §7 tmux launch line, the closeout and the post-run were typed through small wrapper scripts at neutral paths (`$S/scratch/`), so no ssh shell carrying experiment paths lingered during discovery; the lines themselves are unchanged.
* The driver's own closing registry plan reported its own pid as REFUSED (self-ancestry) and signalled nothing, as designed; the driver then exited by itself and the closeout saw it GONE. Nothing was terminated by anyone during this pass.
* Cross-session reference section empty (§5).

## 7. Files

Repo (`runs/b8_recovery/`): this report; `e1_b8_recovery/` (preflight.log, run.log, profile.json, trace.csv, profile_result.json, launch json, drive summary, load-generator summary, token_corpus manifest + provenance, autotune_configs copy, `cpu/` checker outputs: census, completeness v2, nsys validation, workload identity, rank_kernels md/json, config identity, extract log/sha256); `preflight_tray16_driver_context_20261116T053256Z.log`, `negative_control_tray16_20261116T053230Z.log`, `closeout_tray16_20261116T060439Z.log`, `postrun_20261116T060629Z.log`.
Outside Git: raw pass `$S/artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115/run/e1_b8_recovery/` (1.1 GB, `SHA256SUMS.b8`), Phase 1A tables `$S/phase1a-tables/e1_b8_recovery/`, run directory `$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15/e1_b8_recovery/`.

## 8. Stop boundary

This was the single authorized attempt. Nothing further was run: no b=48, no full E1, no additional profiling, no model fitting,
no Rust integration, no database writes, no preset changes, no E2–E6. Returned for review.
