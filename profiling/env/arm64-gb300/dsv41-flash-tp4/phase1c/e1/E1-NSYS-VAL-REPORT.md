# E1 Stage 1.5 pass 4: `e1_nsys_val` trace-validation report (2026-11-15, rack 10 tray16)

Authorization: Kevin's GO of 2026-11-15 for exactly one bounded `e1_nsys_val` pass (90 min reservation maximum, no retries, no
additional cells, no change to model, coefficients, kernels, hook or thresholds, no continuation into full E1). Everything else on
the NO GO list (full E1 capture, more hook-on/off comparisons, fitting, Rust integration, database writes, presets, E2-E6) was not run.

**Outcome in one paragraph.** The pass ran once, rc=0, inside every limit (reservation 31 min 53 s of 90; startup 22 min 22 s of
30). Every pre-registered validation gate passed on the captured data: export valid, NVTX complete on all four devices, iteration
indices aligned, GPU windows of the four ranks overlap in every iteration, collective end spread at b=48 median 0.0001 ms / p90
0.0003 ms (thresholds 0.02 / 0.05), JIT-affected iterations 3 (limit 20), routing-to-iteration joins complete for all 56 requests.
**One material limitation, discovered after the run:** the pre-registered 10 s delayed capture start (the Phase 0 lesson, carried
over into the protocol) placed the capture window entirely inside the b=48 wave. The b=8 wave (trace arrival 0 s) had already finished
at 8.4 s of engine time, so the trace contains 0 of the 1008 pure b=8 iterations and 928 of 928 pure b=48 iterations. The
protocol's "~2160 iterations expected" was wrong and I did not simulate the delay against the trace timeline during the CPU design;
the b=8 cell therefore has no nsys-level evidence from this pass. See §6.

## 1. Pre-GPU verification (Kevin's items 1-7)

All seven passed before the launch; evidence in `runs/stage15/e1_nsys_val/PREGPU-VERIFICATION.md`. In short: the eight frozen heads
are unchanged on the remote (test-ss ca9f8ed0, phase1-planning cc06d9c2, phase1a-costmodel a7725255, phase1b-seqsim 29082793,
phase1c-prep 2e53072d, phase1c-e1-packet 5c86ae1c, phase1c-e1-stage1 c7e8ed9b, kevin/dsv41-gb300-main 58f5d779); the req-frontend
bundle verifies and carries 5c20a259 (6 objects on 2cd7990; bundle sha256 2d6e197c…, recorded for the first time); the gitlink,
the submodule checkout (clean, tree 06a4a61e) and the binary (sha256 7f6eb81d…, `--stream` present) match BINARY-PROVENANCE.json;
preflight on tray16 44 PASS / 0 FAIL; tray16 idle with the reservation pod Running (12:45:30Z) and no GPU pods; the config hash matches
INPUTS_v2, differs from `e1_dry_on_v2` only in `profile_kind`, `nsys.*`, `name`, `log_dir`, and its `nsys` block equals the Phase 0
`diverse_nsys_run2` block; CephFS 21 TB and tray root 1.2 TB free against an expected ~2.4 GB.

## 2. Execution timeline (UTC, 2026-11-15)

| event | time | note |
|---|---|---|
| reservation pod Running | 12:45:30 | `dsv41-gb300-reserve-tray16`, 4 GPUs, pause image |
| preflight on tray16 (standalone) | 12:46:08 | 44 PASS; repeated by the driver at 12:46:38, PASS |
| driver start (tmux `e15-e1_nsys_val`) | 12:46:41 | launcher tree worktree HEAD 92716018, cfg df949648…, session_runner 7f6eb81d… |
| engine init / workers hold the GPUs | 12:47:04 / 12:49:17 | 4 x 260,928 MiB, PIDs 33007-33010, one launch only |
| FlashInfer autotune | 12:49:36 to 13:08:37 | 19 min 01 s (passes 1-3: 18:53, 19:00, 19:05) |
| server ready | 13:09:03 | 22 min 22 s after driver start (limit 30) |
| warm-up, `/start_profile`, measured replay | 13:09 to 13:11:54 | replay 51.2 s wall, 56/56 requests OK, reached idle |
| `/stop_profile`, trace flushed (.nsys-rep written) | 13:12:57 | 223 MB |
| engine shutdown trigger | 13:12:59 | no EngineDeadError-family line anywhere in the log (0 before, 0 after) |
| sqlite export done | 13:13:41 | 1.09 GB, 44 s |
| parse (parsed.json, kernels parquet, host timeline) | 13:15:36 to 13:15:44 | launcher CPU work, GPUs already free |
| driver rc=0 | 13:15:44 | leftover-compute-app check: none |
| GPUs verified clear | 13:16:38 | 4 x 0 MiB, 0 %, no vllm/nsys/session_runner/tail processes, no tmux |
| reservation pod deleted | 13:17:23 | reservation held 31 min 53 s of 90 |

GPU occupancy: worker processes held the four GPUs from about 12:49 to 13:13 (about 24 min), the capture itself covered 32.0 s of
kernel span. Startup limit: respected. No unrelated GPU process appeared (the driver's post-exit check and the 13:16 check were empty;
no Kubernetes GPU pod landed on the node).

Storage: run directory 1.6 GB (nsys-rep 223 MB, sqlite 1.09 GB, host_timeline.json 115 MB, parsed.json 58 MB, parsed.kernels.parquet
41 MB); Phase 1A tables 6.1 MB; archive `$S/artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115/` now 302 run files
(1.8 GB, `SHA256SUMS`, base `run/`; the 196 Stage 1.5 entries are present unchanged) plus `tables/e1_nsys_val` (`SHA256SUMS.tables`);
both verified. The sqlite is kept (protocol §4.5 allowed deleting it; nothing was deleted).

## 3. Validation gates (all PASS; full text `runs/stage15/e1_nsys_val/cpu/nsys_validation_e1_nsys_val.txt`, 90 PASS lines)

| requirement | observed | result |
|---|---|---|
| trace completeness and export | `validation.ok`, 6,363,588 kernel rows, span 32,039 ms, parsed devices [0,1,2,3], nvtx_forward_ranges 4480 = 4 x 1120, rep and sqlite present | PASS |
| NVTX completeness and correlation ownership | 1120 iterations, every one with 28 NVTX ranges (7 phases x 4 devices); devices_absent 0; iteration index identical across devices 0 mismatches; `step_alignment.unpaired_steps_by_device` {} | PASS |
| four-rank iteration alignment | GPU kernel windows of the four ranks overlap in 1120 of 1120 iterations; first-kernel start spread median 0.0074 ms, p90 0.0132 ms, max 2.15 ms; host-side forward NVTX start spread median 0.35 ms (launch skew, INFO) | PASS |
| timestamp consistency | per device the iteration start sequence is strictly increasing (0 violations); one nsys session; tables iterations 1120 == parsed 1120 | PASS |
| collective end-time spread | b=48: median 0.0001 ms, p90 0.0003 ms, max 0.0053 ms over 79,808 collectives in 928 iterations, modal K 86, n_other_K 0; every pure-decode iteration rank-aligned (1023 of 1023, including the ramp iterations b=1..47 with K 163 for b<=16 and 86 above) | PASS (<= 0.02 / 0.05) |
| routing-to-iteration joins | completeness v2 on the same profile: 56 requests, 1024 tokens each, 0 duplicate and 0 missing (request, position) joins, maps strictly increasing, 56 routed-experts files, corpus 57,288 tokens, segments == npy[1:] for all 56; cells b=8 1008 and b=48 928 pure iterations in the metrics | PASS |
| JIT / compilation exclusions | JIT-affected iterations after warm-up 3 (limit 20); `excluded` flag set in the tables | PASS |
| configuration identity vs `e1_dry_on_v2` (informational) | profile.json differs only in profile_kind/nsys/name/log_dir; the launch record adds the nsys wrapper argv and `--profiler-config.profiler=cuda`, nothing else; wire mode non-streaming on both | expected |

Phase 1A extraction (`costmodel.extract`, 4 s): 1120 iterations, 4480 rank rows, 395,136 collective rows, 0 unaligned; input hashes
recorded in `cpu/e1_nsys_val_extract_inputs.sha256`. The estimator smoke test (`analysis/e1_identify.py` with the pass metrics) ran
end to end on the one captured cell and produced no decision (expected for a 1-cell table); nothing from it is used.

## 4. Observed measurements at b=48 (hook ON under nsys; `cpu/rank_kernels_e1_nsys_val.md`)

928 pure-decode iterations, KV 32,768 per request. These describe what the four ranks did in this pass. They do not measure the
hook's effect: there is no hook-off nsys pass under identical conditions (none was authorized).

**Per-rank GPU execution (ms per iteration, p50 / p90)**

| rank | GPU span | busy | compute union | comm union | idle | collective wait (sum) | launch API time | kernels | launches |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 10.296 / 10.421 | 10.132 / 10.254 | 9.694 / 9.803 | 0.776 / 0.877 | 0.161 / 0.192 | 0.193 / 0.289 | 0.582 | 1386 | 48 |
| 1 | 10.296 / 10.422 | 10.134 / 10.257 | 9.760 / 9.864 | 0.677 / 0.743 | 0.161 / 0.175 | 0.094 / 0.158 | 0.552 | 1385 | 47 |
| 2 | 10.295 / 10.419 | 10.130 / 10.251 | 9.766 / 9.874 | 0.666 / 0.730 | 0.163 / 0.191 | 0.083 / 0.145 | 0.607 | 1385 | 47 |
| 3 | 10.294 / 10.420 | 10.127 / 10.254 | 9.762 / 9.876 | 0.664 / 0.735 | 0.162 / 0.186 | 0.081 / 0.149 | 0.641 | 1385 | 47 |

* The four GPU spans agree to 2 us at the median; rank 1 starts first in 742 of 928 iterations, ranks 2 and 3 end last (466 / 440).
* Collective wait (a rank's time inside a collective beyond the fastest rank's, summed over the 86 collectives of an iteration): rank 0
  waits most (0.19 ms p50, 25 % of its comm time), ranks 1-3 wait 0.08-0.09 ms (12-14 %). Rank 3 is the usual straggler (waits least).
  Rank 0 also has about 0.07 ms less compute union than the others and 0.1 ms more comm union: it finishes its compute earlier and
  waits in the collectives. Per collective kind: kind 2 (n 300,672) wait p50 0.0008 ms, p90 0.0034 ms; kind 4 (n 18,560) wait p50
  0.0004 ms. Collective end spread across ranks p50 0.00014 ms, p90 0.0003 ms, max 0.0053 ms.
* `api_sync` time is 0 on every rank (no host synchronisation inside the forward); `comp_in_comm` 0.27-0.31 ms; `seg_idle` about
  0.98 ms per iteration on every rank.
* Metrics-side iteration time for the same iterations (engine `observed_elapsed_ms`, median): 8.527 ms. The GPU span above (10.3 ms)
  is the per-iteration kernel window from the Phase 1A tables and is not the same quantity (consecutive iterations' GPU windows overlap
  under async scheduling; Phase 0 quadrant_nsys had span 9.92 ms at b=48 with the same gap). Not interpreted further here.

**Kernel identities and launch characteristics (b=48, top of 97 distinct names, 1385-1386 kernels and 47-48 launch API calls per rank
per iteration, 5-7 streams)**

| # | kernel (truncated) | count / iteration (4 ranks) | mean us | share of GPU time |
|---|---|---|---|---|
| 1 | `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_…_t128x16x256u2_s6_et128x16_m256x16x32` (MoE expert GEMM, FlashInfer-tuned) | 160 | 77.8 | 26.4 % |
| 2 | `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_…_t128x16x256_s6_et128x16_m256x16x32_c2x1` (MoE expert GEMM) | 160 | 41.6 | 14.1 % |
| 3 | `deep_gemm::sm100_mega_mhc_impl<5120, 40, …>` | 308 | 14.2 | 9.3 % |
| 4 | `flashinfer … dense_blockscaled_gemm_sm100 … Per…` (dense GEMM, tactic A) | 480 | 7.9 | 8.1 % |
| 5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H512PagedKvDenseDynamicTokenSparse…` (sparse MLA attention) | 152 | 18.8 | 6.1 % |
| 6 | `flashinfer … dense_blockscaled_gemm_sm100 … ` (dense GEMM, tactic B) | 360 | 7.3 | 5.6 % |
| 7 | `flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceKernel` | 324 | 7.3 | 5.0 % |
| 8-25 | cutlass bf16 GEMM, `nvjet_sm103 … splitK`, topk, MoE routing/finalize, cublasLt splitK reduce, mxfp8 quantize, fused DSV4 cache ops, NCCL AllGather (20 / iteration, 23 us) | | | 0.5-2.1 % each |

## 5. Cross-session comparison and hypotheses about the b=48 discrepancy (clearly separated from §4)

The only hook-off nsys data of this recipe at b=48 is the Phase 0 `quadrant_nsys` pass (2026-11-12, `vllm_tokens` backend, prompts of
4096 tokens instead of KV 32,768, 3032 pure iterations). Differences against it mix session-to-session drift (Stage 1.5 showed the same
arm moving about 5 % between sessions at b=48), workload differences and any hook effect, and cannot separate them. Observed:

* Kernel set: identical except two PyTorch elementwise/copy kernels present only in this pass (the routed-experts capturer's copies,
  0.6 % of GPU time together). No MoE, GEMM, attention or collective kernel appears or disappears with the hook.
* Dense GEMM tactic mix: the two `dense_blockscaled_gemm_sm100` variants run 480 + 360 times per iteration here versus 320 + 520 in
  quadrant_nsys (same total 840), i.e. FlashInfer's autotuner chose the other tactic for some shapes; their summed time is 6.42 ms vs
  5.87 ms per iteration (4 ranks). The autotune cache of this launch differs from `e1_dry_on_v2`'s in 102 of 159 tactics (64 %) although
  both are hook-on launches of the same hash dir (`cpu/autotune_diff_on_v2_vs_nsys_val.md`), consistent with Stage 1.5's finding that
  tactics are re-chosen every launch.
* Per-rank busy time p50 10.13 ms here vs 9.76 ms in quadrant_nsys; attention/indexer kernels (`sm100_paged_mqa_logits`, topk) take
  2-3x longer here, as expected from the 8x longer KV.

Hypotheses (not measured by this pass): (H1) the Stage 1.5 b=48 differences between launches (hook-on 5.9 % faster than two hook-off
launches that agree to 0.04 %; the same arm moved about 5 % between Stage 1 and Stage 1.5) are driven by per-launch FlashInfer tactic
selection in the dense GEMMs and expert `bmm` kernels, which this pass shows to be of the right size (about 0.5 ms per iteration between
tactic mixes, against a 0.5 ms discrepancy). (H2) The hook's own kernels cost well under 1 % of GPU time at b=48 (the two extra copy
kernels), so a hook effect of the Stage 1 size (+1.5 to +2 %) would have to come from host-side or scheduling paths, not from GPU work.
Testing either needs a hook-off nsys pass in the same session, or a run with the autotuner's tactic choice pinned, neither of which is
authorized or requested here. Metrics-side medians for the record (engine `observed_elapsed_ms`, pure b=48): off_v2 9.023, on_v2 8.488,
off2_v2 9.027, nsys_val (hook on) 8.527 ms; the two hook-on launches agree to 0.5 % and the two hook-off launches to 0.04 %, in two
separate pairs of sessions. This is reported, not attributed.

## 6. Deviations and limitations

1. **b=8 cell not captured (material).** Pre-registered `capture_start_delay_seconds: 10` after measurement start; the dry trace puts
   the 8 b=8 requests at arrival 0 s and the 48 b=48 requests at 18.6 s; the b=8 wave's 1008 pure-decode iterations ended at 8.4 s of
   engine time. Captured iterations are 1223..2342 (ramp into b=48 and the b=48 wave); 0 of 1008 b=8 iterations and 928 of 928 b=48
   iterations are in the trace (`cpu/nsys_capture_window.md`). The protocol's expected size (~2160 iterations) was wrong and the design
   review did not simulate the delay against the trace. The two "b=8" iterations in the rank/kernel report are ramp transients inside
   the b=48 wave, not the b=8 cell. Consequence: NVTX, alignment, collective-spread, JIT and kernel checks are validated on the b=48 cell
   and on the ramp iterations b=1..47; the b=8 cell at KV 32k has no nsys evidence. Remedy if wanted (not run): one more pass with the
   delay reduced to 2-3 s (warm-up is drained before measurement, so the Phase 0 reason for the delay is a cold-start artefact that a
   short delay still covers), or the b=48 wave first in the trace. Needs a new go.
2. **Leftover Stage 1.5 processes.** Three `tail -F` log watchers I started during passes 1-3 were still running on tray16 (no GPU use);
   killed by PID before this pass. The Stage 1.5 closeout statement "all experiment-owned processes stopped" was wrong on that point.
3. **Config identity check is FAIL by construction for an nsys pass** (profile_kind, nsys block, nsys wrapper argv,
   `--profiler-config.profiler=cuda`); treated as informational, listed in §3.
4. **Archive checksum rebuild.** The first merge of the new entries into the archive's `SHA256SUMS` mixed path bases and failed
   verification; the file was regenerated from the files (base `run/`), the 196 Stage 1.5 entries are present with unchanged hashes, and
   both sum files verify.
5. Monitoring only: my first log monitor was restarted once with a tighter filter (no effect on the run).

## 7. Source hosting proposal for req-frontend 5c20a259 (nothing pushed)

The bundle `patch/request-factory-e1-stage15-stream-flag.bundle` (sha256 2d6e197c…) and the local branches are preserved. Proposal:
fork `uw-syfi/request-factory` to `github.com/kaijunli-infr/request-factory` (the account that already pushes ServingStudioSim) and push
branch `e1-stage15-stream-flag` (= 5c20a259, one commit on upstream 2cd7990) there; then, on `phase1c-e1-stage15` only, point
`.gitmodules` for `alignment/load_generator/req-frontend` at that fork so the gitlink becomes fetchable. Alternative if preferred:
a branch of the same name on `uw-syfi/request-factory` itself (needs write rights there) and no `.gitmodules` change. Fork creation,
push and the `.gitmodules` edit each need Kevin's permission; none was done.

## 8. Files

`runs/stage15/e1_nsys_val/`: `PREGPU-VERIFICATION.md`, `preflight.log`, `run.log`, `profile.json`, `profile_result.json`,
`autotune_configs.6adeb850201b.json`, `load_generator_summary.json`, `tables_meta.json`, `RAW-FILES.tsv`, `RAW-SHA256SUMS`,
`ARCHIVE-SHA256SUMS` (+ `.tables`), `cpu/` (nsys validation txt/json, completeness v2 txt/json, rank/kernel report md/json, autotune
diff, capture-window report, identify smoke, extract log and input hashes, identity vs on_v2, postrun log, preflight log, quadrant
smoke of the rank/kernel script). New scripts: `postrun_e1_nsys_val.sh`, `analysis/e1_nsys_rank_kernels.py` (numpy/pyarrow),
`analysis/e1_nsys_capture_window.py`. Raw data: `$S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15/e1_nsys_val/` and the
archive above; Phase 1A tables `$S/phase1a-tables/e1_nsys_val/`.
