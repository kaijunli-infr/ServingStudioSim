# E1 Stage 1.5 report — passes 1-3 executed on rack 10 tray16, 2026-11-15 (hook off / hook on / off-off control)

Authorization: Kevin's conditional GO for passes 1-3 only (reservation max 2 h 30 min); NO GO for `e1_nsys_val`, the full E1
capture, model fitting, Rust production integration, profiling-database writes and preset changes — none of those was done.
Branch `phase1c-e1-stage15`: prerequisites e52929d5, driver fix 05d9eefe, this report on top. Stage 1 evidence untouched.

## 1. Result in one paragraph

All three passes ran to completion (rc=0) on the pinned, provenance-checked launcher; every pre-registered gate passed:
completeness v2 PASS on all three runs (56/56 requests x 1024 tokens, 2337 iterations each, pure-decode 1008 (b=8) and
928 (b=48) in every run, routing records complete and uniquely joined in the hook-on run), configuration identity PASS
between all arms (only the hook flag differs; same wire mode, `chunk_count` 1 for all 168 requests), the original 2 % point
gate PASS and the prospective block-bootstrap gate PASS in both cells. The **off/off control is quiet** (control/off 0.9971
[0.9956, 0.9988] at b=8, 1.0004 [0.9980, 1.0030] at b=48). The hook-on arm is **1.5 % slower at b=8** (1.0149 [1.0136,
1.0162]; excess over the control 1.0178 [1.0158, 1.0200]) and **5.9 % faster at b=48** (0.9406 [0.9379, 0.9432]). The b=48
result is not an "overhead" and is not explained by the within-session control; set against Stage 1 (hook-off b=48 median
8.627 ms yesterday vs 9.023/9.027 ms today, hook-on 8.795 vs 8.488), the same arm moves by 4-5 % between sessions, so a
single on/off pair cannot resolve a 2 % hook effect at b=48 even when the same-session off/off pair agrees to 0.04 %.
FlashInfer winning tactics differed in 55-58 % of entries between every pair of launches, including off vs off2.

## 2. Timeline, occupancy, storage

| event | UTC | note |
|---|---|---|
| prerequisites committed (e52929d5), preflight tray16 49 PASS / 0 FAIL | 08:15:53 | `runs/stage15/cpu/preflight_tray16_pre_pass1.log` |
| reservation pod `dsv41-gb300-reserve-tray16` Running | 08:19:19 | only GPU pod on the node, 4/4 GPUs allocated |
| **pass 1 `e1_dry_off_v2`** | 08:20:47 -> rc=0 08:45:52 | 25 min 05 s; autotune 08:23:37 -> 08:42:30 (18 min 53 s); cache dir c6867df7 |
| pass 2 attempt 1 refused by preflight (no GPU use) | 08:47:15 | check "no previous output dir" also covered the finished pass 1 (preflight design error) |
| pass 2 attempt 2 refused by preflight (no GPU use) | 08:51:12 | the fix left tracked changes uncommitted -> "repo has uncommitted tracked changes" |
| driver fix committed 05d9eefe | 08:58 | preflight now checks the output dir of the pass about to run (`E1_PASS`) |
| **pass 2 `e1_dry_on_v2`** | 08:58:28 -> rc=0 09:23:34 | 25 min 06 s; autotune 09:01:12 -> 09:20:12 (19 min 00 s); cache dir 6adeb850 |
| **pass 3 `e1_dry_off2_v2`** | 09:25:45 -> rc=0 09:50:53 | 25 min 08 s; autotune 09:28:31 -> 09:47:36 (19 min 05 s); cache dir c6867df7 |
| GPUs clear (0 compute apps, 0 MiB x4, no vllm/nsys/session_runner, no tmux) | 09:51:08 | checked after every pass |
| reservation pod deleted | 09:53:02 | **held 1 h 34 min of the 2 h 30 min maximum** |

Each pass: 56 requests, trace-timed, ~51 s of workload after ~22.5 min startup (of which ~19 min autotune). Peak memory as
Stage 1 (all four GPUs). Storage: run root 227 MB (off 58 MB, on 111 MB, off2 58 MB); raw copy
`$S/artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115/run/` with `SHA256SUMS` (196 files, 226 MB).
Pinned launcher: req-frontend 5c20a259, `session_runner` sha256 7f6eb81d… (BINARY-PROVENANCE.json), logged by the driver at
every start.

## 3. Prerequisites (all met before the first launch)

1. Rust `--stream` fix committed as req-frontend branch `e1-stage15-stream-flag` = 5c20a259 (on 2cd7990), gitlink pinned in
   the worktree (e52929d5). No hosted copy: the GitHub account has pull-only rights on uw-syfi/request-factory and creating a
   fork was not permitted in this session; the commit is in the committed bundle `patch/request-factory-e1-stage15-stream-flag.bundle`
   and in the shared local repos on rack 10 and rack 3. Frozen branches unchanged (verified on the remote).
2. `session_runner` rebuilt from the pinned source (10 s); sha256 7f6eb81d45e6b98f… (reproducible: identical to the
   2026-11-14 build), `--help` lists `--stream`, cargo tests 102+7+32+39 green; wire mode verified against a mock server
   (`checks/stream_wire_check.py`: `--stream false` -> `"stream": false` bodies and `chunk_count` 1; `--stream true` ->
   `"stream": true` and 13 chunks).
3. The untracked binary from 2026-11-14 was not used; the driver records the executed binary's hash.
4. `EngineDeadError`: `runs/stage15/engine-dead-error.md` — teardown race in the launcher's abort-mode shutdown (SIGTERM +
   immediate force kill of EngineCore while the API server's output handler still awaits the engine socket), after all
   requests complete; abort conditions defined. In Stage 1.5 it appeared after the shutdown trigger in passes 1 and 2 (lines
   7673/7680 after 7658; 7680/7687 after 7665) and not at all in pass 3 — consistent with a race, 0 occurrences before
   shutdown, all runs accepted.
5. Preflight on tray16: 49 PASS, 0 FAIL, including gitlink == provenance == checkout HEAD, clean checkout, binary sha256.
6. Tray: GPUs 0 MiB/0 apps, no leftovers, no GPU pod on the node, CephFS 21 TB free, root 1.2 TB free; reservation Running
   before launch and the only GPU pod throughout.

## 4. Gates per pass (`postpass_e1_stage15.sh`; outputs `runs/stage15/cpu/v2_<pass>.txt`, `identity_off_v2_vs_<pass>.txt`)

| pass | rc | completeness v2 | identity vs pass 1 (launch records, `--ignore-env PATH`) | EngineDeadError before shutdown | wire mode |
|---|---|---|---|---|---|
| e1_dry_off_v2 | 0 | PASS (13 checks) | - | 0 | chunk_count 1 x56 |
| e1_dry_on_v2 | 0 | PASS (32 checks incl. 56 npy, corpus, join, checksum) | PASS (argv differ only by `--enable-return-routed-experts`) | 0 | chunk_count 1 x56 |
| e1_dry_off2_v2 | 0 | PASS (13) | PASS (identical) | 0 (none at all) | chunk_count 1 x56 |

## 5. Analysis (pre-registered scripts; `runs/stage15/cpu/final/`)

### 5.1 Pure-decode iteration time (`observed_elapsed_ms`), `analysis/e1_pair_compare_v2.py --off 1 --on 2 --control 3`

| cell | arm | n | median ms | mean | std | acf1 | acf2 | acf10 | tau_int | block L | 1st/2nd half median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b8 | off | 1008 | 4.718 | 4.717 | 0.167 | -0.128 | 0.015 | 0.010 | 1.00 | 10 | 4.723 / 4.712 |
| b8 | on | 1008 | 4.788 | 4.788 | 0.093 | 0.016 | 0.035 | 0.041 | 1.83 | 10 | 4.779 / 4.795 |
| b8 | control (off2) | 1008 | 4.704 | 4.704 | 0.195 | -0.233 | 0.034 | -0.001 | 1.00 | 10 | 4.697 / 4.712 |
| b48 | off | 928 | 9.023 | 9.027 | 0.155 | 0.130 | 0.272 | 0.149 | 6.24 | 32 | 9.066 / 8.983 |
| b48 | on | 928 | 8.488 | 8.489 | 0.144 | 0.153 | 0.143 | 0.038 | 2.20 | 11 | 8.504 / 8.470 |
| b48 | control (off2) | 928 | 9.027 | 9.032 | 0.142 | 0.021 | 0.150 | 0.108 | 1.34 | 10 | 9.018 / 9.042 |

| cell | on/off | original 2 % point gate | block-bootstrap 95 % CI | prospective gate | control/off | control CI (noise flag) | excess on vs control [CI] |
|---|---|---|---|---|---|---|---|
| b8_kv32768_w0 | **1.0149** | PASS | [1.0136, 1.0162] | PASS | 0.9971 | [0.9956, 0.9988] (quiet) | 1.0178 [1.0158, 1.0200] |
| b48_kv32768_w0 | **0.9406** | PASS | [0.9379, 0.9432] | PASS | 1.0004 | [0.9980, 1.0030] (quiet) | 0.9402 [0.9365, 0.9436] |

Swapped orientation (on vs off2, control = off): b8 1.0178 [1.0162, 1.0194] PASS/PASS, control 1.0029 [1.0011, 1.0043] quiet;
b48 0.9402 [0.9383, 0.9420], control 0.9996 [0.9971, 1.0020] quiet. The Stage 1 script (`e1_pair_compare.py`, unchanged) gives
the same ratios (1.0149 / 0.9406, gate PASS). Request-level engine TPOT medians (ms, `request_timings.jsonl`): b8 6.46 / 6.62
/ 6.48, b48 20.23 / 19.98 / 20.28 for off / on / off2 (these include ramp and prefill-overlapped iterations).

### 5.2 Reading, per E1-OVERHEAD-METHODOLOGY.md (no threshold changed)

* **b=8:** the hook costs 1.5 % (interval 1.4-1.6 %); the off/off control is quiet and the excess over the control is 1.8 %
  [1.6 %, 2.0 %]. This is attributable to the hook-on launch: both hook-off runs agree (0.3 % apart) and the hook-on arm is
  above both. Consistent in sign with Stage 1 (0.9 %), larger in size. Both gates PASS.
* **b=48:** the hook-on launch was **5.9 % faster** than both hook-off launches, which agree with each other to 0.04 %. The
  within-session control therefore does not exhibit comparable variability, and by the pre-registered rule the difference is
  associated with the hook-on launch — but it has the wrong sign for an overhead and the opposite sign of Stage 1 (+1.95 %).
  Between sessions the *same* arm moved by 4.6 % (hook-off 8.627 -> 9.023 ms) and -3.5 % (hook-on 8.795 -> 8.488 ms). Two
  readings are compatible with the data: (i) a launch-level state (kernel tactic set, clocks, placement) that happened to
  favour the hook-on launch today and the hook-off launch yesterday, with the two hook-off launches today landing in the same
  state; (ii) a genuine hook-dependent effect at b=48 that changes sign between sessions, which is implausible mechanically.
  Neither is established. **Conclusion: at b=48 the hook overhead is not resolvable at the 2 % level from one on/off pair,
  even with a quiet same-session control**; the between-session spread of a single arm (~5 %) is the relevant noise scale,
  and it is larger than the gate. This is reported, not adjudicated; thresholds and coefficients are unchanged.
* **Serial correlation:** b48 hook-off shows tau_int 6.2 (block 32); the block bootstrap handled it; intervals are 2-3x wider
  than a naive bootstrap would give, as anticipated.

### 5.3 FlashInfer winning tactics (`autotune_cache_diff.py`, the three `autotune_configs.json` saved beside the runs)

| pair | metadata | common keys | differing tactics | MoE (27) | mxfp8_gemm (132) |
|---|---|---|---|---|---|
| off vs off2 (same cache dir c6867df7) | identical | 159 | **92 (58 %)** | 14 | 78 |
| off vs on | identical | 159 | 87 (55 %) | 11 | 76 |
| off2 vs on | identical | 159 | 93 (58 %) | 15 | 78 |

Tactic re-selection between launches is as large between two hook-off launches as between arms; it did not move the b=48
median between off and off2 (0.04 %), so tactic drift alone does not explain the 5.9 % difference of the hook-on launch
either. Every launch re-tuned (~19 min) and logged 48 "No tuned config covers mxfp8_gemm" fallbacks (AUTOTUNER-INVESTIGATION.md).

### 5.4 Output-token differences (`e1_output_diff.py`; finest available = 100-char `output_preview` + counts)

All 168 requests: `output_len_actual` 1024, `finish_reason` length, `completion_tokens` 1024, SUCCESS, `chunk_count` 1.
Preview identical: off vs on 17/56 (b8 4/8, b48 13/48); **off vs off2 22/56** (3/8, 19/48); on vs off2 25/56 (4/8, 21/48).
Two hook-off launches disagree on 34 of 56 greedy outputs — run-to-run non-determinism of the engine at this level, not a
hook effect. Full token sequences are not logged by the pre-registered load-generator configuration; adding that would be a
config/code change for a later stage.

### 5.5 Collective-wait and rank-timing information

Not available from workload_metrics passes: the per-iteration records carry one `observed_elapsed_ms` per iteration and a
`dp_rank` (0) only; TP-rank kernel windows and collective end spread need the nsys pass (`e1_nsys_val`, NO GO here;
checker `checks/e1_nsys_validation.py` is ready, validated on Phase 0 data).

## 6. Deviations

a. Two refused pass-2 launches (08:47, 08:51), both stopped by the preflight before any GPU use: a Stage 1.5 preflight design
   error (output-dir check over all passes) and then my uncommitted fix. Fixed and committed (05d9eefe); logs kept
   (`runs/stage15/cpu/preflight_tray16_pre_pass2_attempt*.log`). Pass 2 started 13 min later than planned; total within budget.
b. No hosted copy of the req-frontend branch (pull-only rights; fork creation not permitted) — see §3.1.
c. Scripts added during the session (not frozen, not configs): `postpass_e1_stage15.sh`, `analyze_e1_stage15.sh`,
   `analysis/e1_output_diff.py`; the preflight/driver fix in (a). Configs, trace, hashes (`INPUTS_v2.sha256`), checkers,
   `e1_pair_compare_v2.py` and thresholds unchanged since 5a3f5f33.

## 7. Cleanup evidence

After pass 3 (09:51:08Z): `nvidia-smi` 0 compute apps, 0 MiB on all four GPUs; 0 vllm/nsys/session_runner processes; no tmux
server; reservation pod deleted 09:53:02Z and confirmed NotFound. Raw artifacts checksummed (§2).

## 8. Open for review (no action taken)

1. Whether the b=48 between-session spread (~5 % in the same arm) changes the E1 main plan: the hook-overhead question at
   large batch needs either more launches per arm (each ~26 min) or a same-process comparison that does not exist.
2. `e1_nsys_val` (NO GO today) remains the only way to test gates 6/7 (NVTX, collective spread) before Stage 2.
3. Hosting of req-frontend branch 5c20a259 (fork or upstream branch) — Kevin's call.
