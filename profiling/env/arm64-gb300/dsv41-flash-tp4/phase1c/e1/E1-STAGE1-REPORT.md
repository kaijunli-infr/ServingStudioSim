# E1 Stage 1 report: pre-registered two-cell dry-run pair (routing hook OFF vs ON)

Date: 2026-11-14. Tray: rack 10 `pod4-gb300-10-tray16-f3`, GPUs 0-3 (4x GB300), reservation pod `dsv41-gb300-reserve-tray16`.
Authorization: Stage 1 only (Kevin, 2026-11-14): "GO: Option A, staged dry-run pair. NO GO: Full E1 capture, model fitting,
Rust integration, database writes or preset changes." Nothing beyond the two dry runs was executed. **Stage 2 (full E1) was not
started and is not started by this report.** Phase 0 / 1A / 1B / 1C-0 branches and model parameters are untouched.

Launcher tree: scratch worktree `$S/wt-e1stage1`, branch `phase1c-e1-stage1` at 7e1bbbf0 (packet 5c86ae1c + opt-in patch, default
off). Artifacts under the main clone's git-ignored `logs/20261114_dsv41_flash_gb300_tp4_e1/` (= `$RUN`), raw copy in
`$S/artifacts/phase1c-e1-dsv41-flash-gb300-tp4-20261114/run/` + `SHA256SUMS`. Small derived files: `e1/runs/{e1_dry_off,e1_dry_on,stage1}/`.

## 1. Result in one paragraph

Both dry runs completed: 56/56 requests each, every request exactly 1024 output tokens (57,344 per arm), all HTTP 200, no failures,
identical engine iteration structure (2160 recorded iterations, 1008 pure-decode iterations in cell b=8/KV 32k and 928 in cell
b=48/KV 32k on both sides). Routing-hook overhead on the pre-registered metric (per-cell median engine `observed_elapsed_ms`,
hook-on / hook-off): **1.0086 at b=8, 1.0195 at b=48**. Both point estimates are within the pre-registered 1.02 limit, but at b=48
the bootstrap 95 % interval [1.0180, 1.0209] straddles the limit, so I report that cell as **marginal / inconclusive**, not PASS.
The overhead grows with batch size (0.9 % -> 2.0 %). Routing records are complete and the token-to-iteration join is unique
(56 x 1023 rows, 40 layers, no duplicates, no gaps). Two acceptance gates (NVTX ownership / step alignment, collective end-time
spread) are not measurable in the dry-run pair because, by the pre-registered design, the dry runs use `workload_metrics` without
nsys. Greedy outputs are not bit-reproducible between the two runs (35/56 previews differ); token counts are identical.

## 2. Timeline, occupancy, storage

| event | UTC | note |
|---|---|---|
| reservation pod created | 18:13:12 | 4 GPUs, nodeSelector tray16 |
| preflight (worktree) 24 PASS + 1 INFO | 18:14:07 | `stage1/preflight_pre_dry_off.log` |
| launch attempt 1 (hook off) | 18:14:36 -> rc=1 at 18:14:37 | no GPU use; worktree lacked the req-frontend submodule checkout (Cargo.toml missing) |
| launch attempt 2 (hook off) | 18:22:46 -> rc=1 at 18:23:10 | no GPU use; vLLM import failed (`_vllm_fa2_C/_vllm_fa3_C` missing): worktree's vllm checkout had no compiled extensions |
| preflight re-runs | 18:20:39 (wrong tree, by-design FAIL, kept), 18:22 (PASS), 18:27 (PASS) | see §5 |
| **e1_dry_off** (hook off) | 18:27:40 -> rc=0 18:53:03 | 25 min 23 s GPU-held; workload 50.4 s; metrics 2160 iterations |
| preflight (pre hook-on) 24 PASS | 18:54 | `stage1/preflight_pre_dry_on.log` |
| **e1_dry_on** (hook on) | 18:54:49 -> rc=0 19:20:10 | 25 min 21 s GPU-held; workload 50.8 s; corpus + routed_experts written |
| cleanup verified | 19:25:38 | 0 compute apps, 0 MiB on all 4 GPUs, 0 experiment processes, no tmux server |
| reservation pod deleted | 19:27:25 | held 1 h 14 min of the 5 h maximum; no extension |

Per run: ~5 s weight load, CUDA-graph capture, JIT kernel warmup and **flashinfer AutoTuner tuning (mxfp8_gemm 22 profiles,
trtllm_fp4_block_scale_moe 22 + 5 profiles) ~20 min**, then the 51 s workload. Startup is ~22.5 min of each 25.4 min run.
Peak GPU memory 256,849 MiB per GPU (all four, every sample; only the four vLLM ranks were ever listed by nvidia-smi).
Storage: `$RUN` 169 MB total (hook off 58 MB; hook on 111 MB of which routes.u16 27 MB + 56 npy 27 MB; stage1 1 MB); raw copy
169 MB; budget was 80 GB. CephFS free 21.1 TB, tray16 root free 1.16 TB throughout.

## 3. Preflight

Script: worktree `e1/preflight_e1_host.sh` (sha256 1db667d8...), run with `E1_REPO=$S/wt-e1stage1`. Results: 24 PASS + 1 INFO
(launcher tree 7e1bbbf0) at 18:14:07, 18:22 (`preflight_pre_dry_off_3.log`), 18:27 (`e1_dry_off/preflight.log`) and 18:54
(`preflight_pre_dry_on.log`, `e1_dry_on/preflight.log`). The "previously expected patch-related failure" is the launcher dry-run
check on the unpatched main tree: it fails there by design (unknown key `capture_routed_experts`) and passes from the worktree.
The 18:20:39 run (`preflight_pre_dry_off_2_WRONG_TREE_main_clone_by_design_fail.log`) is that failure: I had invoked the packet
branch's copy of the script, which hard-codes the main clone as launcher tree. Scheduler side: `kubectl` showed the reservation pod
as the only GPU pod on tray16 at 18:23:45, 19:22:49 and before deletion; no other pod was scheduled to the node in the window.
Fingerprints: `stage1/fingerprint_20261114T181357Z.txt`, `stage1/fingerprint_addendum_20261114T183130Z.txt`.

## 4. Acceptance gates

| # | gate | verdict | evidence |
|---|---|---|---|
| 1 | all requests / tokens complete | **PASS** | 56/56 both arms; `output_len_actual` = `completion_tokens` = 1024 for all 112; `finish_reason` length; 0 failed; 113 POST /v1/completions access-log lines per arm, all 200, none non-200; `output_mismatch_steps` 0 |
| 2 | hook overhead <= 2 % (pre-registered: per-cell median engine iteration ratio) | **b=8 PASS 1.0086; b=48 MARGINAL/INCONCLUSIVE 1.0195** | bootstrap 95 % CI b=8 [1.0075, 1.0097], b=48 [1.0180, 1.0209]; frozen checker (point estimate only) prints PASS for both. Threshold not changed; the interval criterion is an addition, reported separately |
| 3 | paired latency / throughput / iteration comparison with variability | reported | §4.1 |
| 4 | routing records cover all requests, iterations (tokens), layers, ranks | **PASS** | 56 `routed_experts/*.npy`, each (1024, 40, 6) uint16; corpus 57,288 rows = 56 x 1023; 40 layers, zero all-zero rows in every layer; 368-384 distinct experts per layer; max expert id 383 < 384; TP4 capture is rank-independent (routing is replicated) |
| 5 | no ambiguous / duplicate route-to-iteration joins | **PASS (unambiguous join); frozen checker sub-check FAIL, see §5(b)** | for every request, `output_tokens_before` 1..1023 appears exactly once with `emitted_tokens` 1 (0 duplicates, 0 missing); `provenance.request_segments` 56 unique, contiguous, non-overlapping, lengths all 1023, names == metrics request ids |
| 6 | NVTX ownership and step alignment | **NOT MEASURABLE in Stage 1** | dry runs are `workload_metrics` without nsys by pre-registered design; no NVTX trace exists |
| 7 | collective end-time spread | **NOT MEASURABLE in Stage 1** | same reason |
| 8 | no unexpected JIT / compilation contamination of the comparison | **PASS (symmetric)** | startup compile + autotune finished before the first request in both arms; during the workload both arms show the same events: one Triton JIT (`_ring_slot_mapping_kernel`, TP0) during the b=48 ramp and flashinfer "No tuned config covers mxfp8_gemm" warnings for prefill shapes 16384 / 1480 / 2257 tokens; no pure-decode iteration > 1.5x cell median except the first decode iteration of b=8 (index 198, 7.16 vs 7.25 ms) on both sides; no iteration > 250 ms |
| 9 | no unexpected device activity / foreign processes | **PASS** | nvidia-smi listed only the four vLLM ranks at every 2-3 min sample, 0 apps before and after each run; reservation pod the only GPU pod on the node |
| 10 | no artifact / reservation limit violation | **PASS** | 1 h 14 min of 5 h; 169 MB of 80 GB; only GPUs 0-3 used (4 visible, `CUDA_VISIBLE_DEVICES` unset, TP4) |

### 4.1 Paired comparison (full tables: `stage1/pair/E1-PAIR-COMPARE.md`, `e1_pair_compare.json`)

Engine iteration time, pure-decode iterations (`observed_elapsed_ms`), per cell:

| cell | n | median off | median on | ratio | mean off / on | std off / on | p10/p90 off | p10/p90 on |
|---|---|---|---|---|---|---|---|---|
| b8 KV32k | 1008 / 1008 | 4.711 | 4.751 | 1.0086 | 4.711 / 4.738 | 0.135 / 0.193 | 4.654 / 4.771 | 4.632 / 4.815 |
| b48 KV32k | 928 / 928 | 8.627 | 8.795 | 1.0195 | 8.634 / 8.796 | 0.164 / 0.159 | 8.496 / 8.771 | 8.657 / 8.921 |

Throughput (emitted tokens / summed pure-decode time): b8 1698.1 -> 1688.6 tok/s (0.9944); b48 5559.1 -> 5456.8 tok/s (0.9816).
Replay-level: run duration 50.45 s (off) vs 50.83 s (on); output-token throughput 1136.7 vs 1128.1 tok/s (-0.76 %).

Request level (replay.jsonl, 56 pairs): total duration median b48 31,526 -> 31,907 ms (1.0121), b8 8,868 -> 9,001 ms (1.0150).
**TTFT and TPOT are not comparable across arms**: the hook-on arm's requests were served non-streaming (`chunk_count` 1,
`first_token_ms` == `total_duration_ms`, `tpot_measured_steps` 0), the hook-off arm streamed (1024 chunks). So the two arms differ
in two respects, routed-experts return and streaming mode, not one. The pre-registered gate metric is engine-side and unaffected
by client streaming, but non-streaming removes per-token SSE work from the API server, which can only make the hook-on arm look
faster, not slower, so it does not mask hook overhead.

Token identity (reported, not gated, per packet §11): `output_preview` (first 100 characters) identical for 21/56 pairs (b8 6/8,
b48 15/48); the other 35 diverge at character positions 1-100. Token counts identical (1024/1024). Greedy decoding on this stack is
not bit-reproducible run to run; whether the hook contributes cannot be told from one pair (needs an off/off repeat).

### 4.2 Pure-decode counts vs plan

b8: 1008 / 1008 (plan 1008, floor 400); b48: 928 / 928 (plan 928, floor 400). PASS. Every request emits 15-95 tokens (median 95)
in ramp (mixed prefill+decode) iterations before its cell reaches steady state; those tokens have routing rows but are outside
the pure-decode analysis set, as the design intends.

## 5. Deviations and findings

a. **Two failed launches before any GPU use** (attempts 1 and 2 above). Fixes, confined to the worktree: (1) `git clone --shared`
   of the main clone's `alignment/load_generator/req-frontend` checkout at gitlink 2cd7990c, then `cargo build --release --bin
   session_runner` (47 s; sha256 557bd272..., differs from the main clone's 98973f7c... only by embedded build path); (2) rsync of
   the 2005 git-ignored build outputs (`*.abi3.so`, `_version.py`, `third_party/*`, `vllm_flash_attn/{cute,layers,ops}`, ...) from the
   main clone's vllm checkout into the worktree's checkout at the same commit 0ab18711, verified byte-identical
   (`stage1/vllm_build_artifacts_copied.sha256`). Worktree `git status` clean before every launch. Lesson: a shared-clone worktree is
   not a complete launcher tree; the packet should list these two steps.
b. **Frozen checker `checks/e1_completeness.py` has two defects** (left unmodified; reported): (i) `cell_counts` parses the trace tag
   KV as "32" while the plan keys use 32768, so the per-cell pure-decode-minimum sub-check silently never ran (covered instead by
   `analysis/e1_pair_compare.py`, §4.2); (ii) the join sub-check requires "pure-decode iteration count == corpus rows", which
   ignores tokens emitted in ramp iterations, so it reports FAIL for 54/56 requests on correct data. The pre-registered join rule
   itself (token j <-> iteration with `output_tokens_before == j`) holds uniquely for every token. Decision needed: accept a
   corrected checker for Stage 2.
c. **Preflight 18:20:39 FAIL** was an invocation error (packet-branch copy of the script), not a tray/environment failure; log kept.
d. **Hook-on arm non-streaming** (§4.1); it follows from the packet's openai-backend routed-experts path.
e. **flashinfer AutoTuner re-tunes at every start (~20 min)** although the preflight "cache warm: flashinfer-autotune" check passes;
   the check only tests that the cache directory is non-empty. For E1 main this adds ~20 min per launch to the occupancy estimate
   (one launch planned) and the check should be tightened.
f. **Hook-on shutdown logged `EngineDeadError` in `AsyncLLM.output_handler`** at 19:20:01 during the launcher's abort-mode teardown
   (force-kill of EngineCore), after the workload and corpus packing had completed; hook-off shut down cleanly. Post-measurement,
   no effect on data; worth a look before Stage 2.
g. Added (small, scripts only): `analysis/e1_pair_compare.py`, `analysis/e1_stage1_outliers.py`; `collect_e1.sh` gained
   `E1_REPO` and the `runs/stage1/` copy. No change to configs, traces, hashes in `INPUTS.sha256`, the patch, or frozen branches.

## 6. Cleanup evidence

tray16 at 19:25:38Z: nvidia-smi 0 compute apps, 0 MiB / 0 % on GPUs 0-3; `pgrep` for vllm / nsys / session_runner / launcher: 0;
tmux: no server; seven experiment-owned temp entries in /tmp (4-12 KB each, mtimes 18:28-19:17, owner dc) removed; nothing else
touched. Reservation pod deleted 19:27:25Z (NotFound confirmed). Main clone `git status` clean; frozen branches unchanged
(phase1-planning cc06d9c2, phase1a-costmodel a7725255, phase1b-seqsim 29082793, phase1c-prep 2e53072d, phase1c-e1-packet 5c86ae1c).

## 7. Stage 2

Not started. Items for the Stage 2 decision: (1) the b=48 overhead at 1.95 % with an interval reaching 2.09 %, and its growth with
batch size (E1 main's largest batch is 48; its decode cost per iteration at KV 98k is higher, so the relative overhead there is
likely smaller, but this is an inference, not a measurement); (2) checker corrections (§5b); (3) the streaming confound (§5d);
(4) autotune startup cost (§5e); (5) run-to-run token non-determinism (§4.1); (6) gates 6 and 7 can only be tested with nsys, i.e.
in the full capture or in an explicitly authorized nsys dry run.
