# E1 readiness audit (CPU only, 2026-11-16): frozen evidence, rank dependence, outstanding discrepancies

Authorized by Kevin's GO "CPU-only E1 readiness audit" after acceptance of the b=8 recovery (223aff6d) and completion of Stage 1.5
trace validation. No GPU was used. No frozen measurement, coefficient, threshold or held-out row was modified. Branch
`phase1c-e1-audit` from 223aff6d; outputs under `audit/`; new scripts in `analysis/` (hashes in `audit/AUDIT-SCRIPTS.sha256`).

## A. Accepted evidence frozen and checksummed

| what | where | result |
|---|---|---|
| repo evidence of both accepted passes: `runs/stage15/**` (b=48 `e1_nsys_val`), `runs/b8_recovery/**` (b=8 `e1_b8_recovery`), configs_v2, traces, checks, analysis, manifests, provenance, reports, drivers; plus the Phase 1C protocol, matrix and the Phase 1C-0 uncertainty audit (FINDINGS, RELIABILITY, reliability.json, wait_decomposition.py) | `audit/FROZEN-EVIDENCE.sha256` (239 + 7 files, paths relative to `phase1c/e1`) | written |
| raw archive copies `artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115/run/{e1_nsys_val,e1_b8_recovery}` | verified against the recorded `SHA256SUMS` (88 files of e1_nsys_val) and `SHA256SUMS.b8` (40 files): **all OK** | verified |
| the two SHA256SUMS files and the Phase 1A compact tables `phase1a-tables/{e1_nsys_val,e1_b8_recovery}` (parquet + meta.json) | `audit/FROZEN-EVIDENCE-EXTERNAL.sha256` (11 entries, absolute paths) | written |
| write protection | `chmod -R a-w` on the two raw run copies and the two table directories (dc-owned; reversible); 0 writable files left | applied |

Historical acceptance thresholds (collective end spread median ≤ 0.02 ms, p90 ≤ 0.05 ms; JIT ≤ 20; coverage rules) are
untouched: they live in `checks/e1_nsys_validation.py` defaults and the packets, both pinned above.

## B. Rank dependence from the existing traces (`analysis/e1_rank_dependence.py`; frozen estimator `analysis/e1_identify.py`)

Passes and conditions (all TP4, same model and server recipe):

| pass | b | KV | hook | backend / stream | session | capture | n pure (modal K) | K |
|---|---|---|---|---|---|---|---|---|
| `e1_b8_recovery` | 8 | 32 k | on | openai / off | 2026-11-16 05:35Z | whole wave, no delay | 1008 (1008) | 163 |
| `e1_nsys_val` | 48 | 32 k | on | openai / off | 2026-11-15 12:46Z | started mid-wave (10 s delay): the first 80 pure iterations of the wave are not in the trace | 928 (928) | 86 |
| `diverse_nsys_run2` (Phase 0 reference) | 8 | ≤ 131 k, Poisson | off | vllm_tokens | 2026-11-12 | delayed | 1675 | 163 |
| `quadrant_nsys` (Phase 0 reference) | 48 / 24 | < 8 k | off | vllm_tokens | 2026-11-12 | delayed | 3032 / 238 | 86 |

The two Stage 1.5 passes share workload type, KV, output length, hook, backend and recipe; they differ in b (hence K: the
b ≤ 16 and b ≥ 17 collective paths) and in session. The Phase 0 confound (b changes together with workload type and pass) is
therefore reduced to "b or session" here, and the Phase 0 references bracket the session axis.

### B.1 Reconstruction of per-rank arrival, start and completion (directly measured)

Per (iteration, rank, collective k) the Phase 1A `collectives` table holds the rank's arrival (`start_rel_ns`: its collective
kernel starts when it reaches the barrier), completion (`end_rel_ns`) and in-collective time (`comm_sum_ns`). All four ranks
have identical kind sequences at the modal K in every pass (`kind_sequence_identical_across_ranks` true).

| pass | collective end spread p50 / p90 / max ms | arrival spread (all k) p50 / p90 ms | first-collective arrival spread p50 / p90 ms |
|---|---|---|---|
| b=8 recovery | 0.0010 / 0.0080 / 0.0249 | 0.0028 / 0.0098 | 0.0149 / 0.0211 |
| b=48 nsys_val | 0.0001 / 0.0003 / 0.0053 | 0.0023 / 0.0048 | 0.0142 / 0.0202 |
| run2 b=8 | 0.0010 / 0.0088 / 0.0385 | 0.0028 / 0.0106 | 0.0167 / 0.0228 |
| quadrant b=48 | 0.0001 / 0.0003 / 0.0083 | 0.0017 / 0.0043 | 0.0181 / 0.0247 |

The first-collective (host-arrival) spread, 0.014–0.018 ms, equals the frozen Phase 1C-0 value (0.016–0.02 ms for b ≥ 3). The
end spread is at the nsys timestamp resolution. **Neither number says anything about rank independence**: both passes have
tiny end spreads and very different wait structure (below). End spread is a clock/alignment check only.

### B.2 Intrinsic execution vs synchronisation waiting: where the trace supports the distinction

Two definitions were computed. *Proxy* (Phase 1A, used by the frozen model): wait_r,k = comm_r,k − min_r' comm_r',k, intrinsic_k =
min_r comm; it assumes the fastest rank's in-collective time is the execution time. *Direct*: wait_r,k = latest arrival − arrival_r,
intrinsic_k = completion − latest arrival; it assumes simultaneous completion on all ranks and needs no proxy.

| pass | intrinsic per iteration (sum), direct / proxy ms | direct − proxy per collective p50 / p90 ms | wait sum per rank, direct vs proxy |
|---|---|---|---|
| b=48 nsys_val | 0.581 / 0.583 | −0.0000 / 0.0001 | agree within 2 % (r0 0.1975 vs 0.1928; r3 0.0821 vs 0.0809) |
| quadrant b=48 | 0.578 / 0.580 | −0.0000 / 0.0001 | agree |
| b=8 recovery | 0.270 / 0.492 | −0.0007 / −0.0002 | **direct ≈ 2 × proxy** (r0 0.458 vs 0.220) |
| run2 b=8 | 0.234 / 0.491 | −0.0007 / −0.0002 | direct ≈ 2 × proxy |

On the b ≥ 17 path (K = 86, collectives ≈ 6.8 µs, end spread 0.1 µs) the distinction is supported directly: two independent
definitions agree to 2 %, so per-rank wait and intrinsic time at b=48 are measured, not modelled. On the b ≤ 16 path (K = 163,
mhc / Lamport all-reduce kernels of ≈ 3 µs, end spread p50 1 µs, p90 8 µs) the ranks do **not** complete simultaneously at the
sub-microsecond level, the direct definition double-counts and only the proxy definition is usable. Every b=8 wait number below
is therefore the Phase 1A proxy and carries the "fastest rank = intrinsic" assumption; every b=48 number is corroborated by both.

### B.3 Per-rank waiting distributions, tails, critical-path rank frequencies

Per-rank wait per iteration (Phase 1A proxy, sum over the K collectives, ms):

| pass | rank | p50 | p90 | p99 | max | share of comm p50 | share of wall p50 | first-collective lateness mean ms |
|---|---|---|---|---|---|---|---|---|
| b=8 recovery | 0 | 0.220 | 0.280 | 0.336 | 1.03 | 0.31 | 4.2 % | 0.0004 |
|  | 1 | 0.164 | 0.217 | 0.269 | 1.04 | 0.25 | 3.1 % | 0.0070 |
|  | 2 | 0.154 | 0.196 | 0.256 | 0.82 | 0.24 | 2.9 % | 0.0146 |
|  | 3 | 0.170 | 0.219 | 0.287 | 1.13 | 0.26 | 3.2 % | 0.0130 |
| b=48 nsys_val | 0 | 0.193 | 0.289 | 0.382 | 5.43 | 0.25 | 1.9 % | 0.0007 |
|  | 1 | 0.094 | 0.158 | 0.245 | 5.34 | 0.14 | 0.9 % | 0.0036 |
|  | 2 | 0.083 | 0.145 | 0.218 | 5.33 | 0.12 | 0.8 % | 0.0129 |
|  | 3 | 0.081 | 0.149 | 0.244 | 0.28 | 0.12 | 0.8 % | 0.0178 |

The b=8 values reproduce the frozen Phase 1C-0 measured waits (0.225 / 0.188 / 0.182 / 0.183 ms from run2 at b=8) within 0.02 ms
with the same rank-0 excess; b=48 reproduces the quadrant pattern (rank 0 ≈ 0.14–0.19 ms, ranks 2–3 ≈ 0.08 ms). Tails: p99 is
1.5–1.7 × p50 in both cells; the b=48 maxima (5.3–5.4 ms on ranks 0–2 in one iteration where rank 3 arrived 5.3 ms late) are a
single event out of 928 and a candidate stall, not a distribution feature (rank 3's max is 0.28 ms).

Which rank is last, and which is on the critical path:

| pass | last-arriving share per collective r0 r1 r2 r3 | at the first collective | at the final collective | least-waiting (critical-path) rank share | lag-1 persistence of the final-collective last rank (chance) |
|---|---|---|---|---|---|
| b=8 recovery | .16 .28 .30 .26 | .01 .05 .54 .40 | .11 .12 .32 .46 | .05 .29 .42 .24 | 0.37 (0.33) |
| b=48 nsys_val | .08 .32 .29 .30 | .01 .01 .56 .42 | .12 .13 .29 .46 | .05 .26 .34 .36 | 0.37 (0.32) |
| run2 b=8 | .20 .29 .26 .25 | .00 .06 .54 .40 | .16 .14 .32 .38 | .11 .26 .33 .29 | 0.35 (0.29) |
| quadrant b=48 | .16 .23 .30 .31 | .00 .00 .61 .38 | .11 .09 .39 .41 | .09 .11 .40 .40 | 0.36 (0.34) |

Directly measured: rank 2 or 3 arrives last at the first collective in ≈ 95 % of iterations in every pass (host-side launch order,
the Phase 1C-0 "rank-lateness pattern"); inside the forward the last-arriving rank is spread over ranks 1–3 almost evenly and rank 0
is rarely last (8–20 %); the critical-path rank changes from iteration to iteration with no memory (persistence equals chance).
Rank 0 waits most in every pass. These frequencies are the same at b=8 and b=48 and the same as in Phase 0.

### B.4 b=8 versus b=48 under identified conditions

| quantity | b=8 (32 k, hook on, 11-16) | b=48 (32 k, hook on, 11-15) | Phase 0 b=8 (run2) | Phase 0 b=48 (quadrant) | status |
|---|---|---|---|---|---|
| wait share of wall, rank 0 / others | 4.2 % / 2.9–3.2 % | 1.9 % / 0.8–0.9 % | 4.3 % / 3.5 % | 1.4 % / 0.8–1.2 % | measured |
| segment cross-rank correlation ρ (variance-weighted) | 0.30 | 0.58 | 0.42 | 0.73 | measured |
| ρ p10 / p50 / p90 over segments | −0.01 / 0.03 / 0.37 | 0.04 / 0.29 / 0.97 | −0.01 / 0.01 / 0.36 | 0.35 / 0.70 / 0.98 | measured |
| normalised common-mode variance c_common | 0.23 | 0.21 | 0.34 | 0.66 | measured |
| per-iteration busy cross-rank corr (min off-diagonal) | 0.84 | 0.51 | 0.43 | 0.62 | measured |
| MoE group time cross-rank corr (min) | 0.61 | 0.95 | 0.68 | 0.99 | measured |
| f_ind, permutation / quantile composition (this audit) | 0.835 [0.828, 0.840] | 0.332 [0.322, 0.333] | 0.796 [0.776, 0.812] | 0.226 [0.218, 0.230] | model-dependent (barrier chain) |
| f_ind, frozen estimator `e1_identify.py` | 0.79 [0.77, 0.80] | 0.33 [0.31, 0.37] | — | — | model-dependent (quantile draws) |
| frozen Phase 1C-0 value (FINDINGS §3) | ≈ 0.80 (run2 b=8) | ≈ 0.25 (quadrant b=48) | | | frozen reference |

Reading, without fitting anything:
* The **regime difference persists** with workload type, KV, hook, backend and recipe held constant: f_ind ≈ 0.8 at b=8 and ≈ 0.33 at
  b=48, by both estimators. In Phase 1C-0 this contrast was confounded with workload and pass; here the only remaining confound is
  the session (two launches one day apart). The pre-registered "regime effect" threshold of the E1 packet (Δ < −0.2 between
  b ≥ 24 and b ≤ 16) is met on this pair (Δ ≈ −0.47); E1 is still needed to see it at every KV and inside one launch.
* What drives the difference is visible directly: at b=48 the high-variance segments are strongly correlated across ranks (ρ p90
  0.97, MoE group correlation 0.95), i.e. the per-iteration variation is dominated by a common-mode component (routing load is
  common to the four TP ranks), while at b=8 most segments are essentially uncorrelated (ρ median 0.03) and the common-mode part is
  small. c_common is similar (0.21–0.23) because it is unweighted; ρ (variance-weighted) is the quantity that matches f_ind.
* Pass effect: the Stage 1.5 b=48 value (0.33) is above the quadrant value (0.23) by 0.10 with non-overlapping CIs; the b=8 value
  (0.79–0.84) is within 0.04 of run2. Under the packet's "pass effect" rule (|Δ| > 0.15 at two or more b) this is not a pass effect,
  but it is a reminder that quadrant (KV < 8 k, vllm_tokens, saturated mix) is not the same workload as E1 at b=48.
* Temporal structure (measured, new): at b=8 the iteration wall has long memory inside the wave (ACF 0.57 at lag 1, 0.24 at lag
  10, 0.09 at lag 100) and so do both the common-mode and the rank-independent components (0.50 / 0.53 at lag 1); at b=48 the wall
  ACF is 0.14 at lag 1 but the rank-independent component still has 0.42 at lag 1 and 0.33 at lag 5. Phase 0 run2 (Poisson
  arrivals) shows 0.09 at lag 1 for the wall. The frozen model draws iterations independently; the trace-timed, single-wave E1
  workload has serially correlated iterations, which matters for how the per-cell bootstrap CIs are formed (block bootstrap) and
  is listed as a risk in the packet.
* The straggler identity is memoryless (persistence = chance), so the correlation is in magnitudes, not in which rank is late.

### B.5 Directly measured vs model-dependent

| directly measured from timestamps | depends on a modelling assumption |
|---|---|
| arrival / completion per rank and collective; end spread; arrival spread; first-collective lateness and last-rank frequencies; per-segment and per-iteration cross-rank correlations (ρ, c_common, busy / MoE / wait correlations); autocorrelations; critical-path rank frequencies; b=48 wait and intrinsic (two definitions agree) | b=8 wait and intrinsic (fastest rank = intrinsic; the direct definition is invalid on the K = 163 path); W_ind, W_cm and therefore f_ind (counterfactual barrier-chain compositions of the measured segments; the frozen estimator additionally uses per-(rank, segment) quantile draws); any statement about "what E1 will show at other KV" |

### B.6 Against the frozen Phase 1C uncertainty hypotheses (FINDINGS §3–5), no coefficients fitted

| frozen statement | this audit |
|---|---|
| measured wait sits between common-mode and independent: ≈ 0.80 of the way at b=8 (run2), ≈ 0.25 at b=48 (quadrant) | reproduced on the frozen passes by the new coefficient-free estimator (0.796 / 0.226) and found at 0.79–0.84 / 0.33 on the Stage 1.5 passes |
| first-collective arrival spread 0.016–0.02 ms for b ≥ 3, host arrival not the source of the skew | 0.014–0.018 ms in all four passes; lateness ≤ 0.018 ms per rank against waits of 0.08–0.22 ms |
| whether the fraction depends on b (K path) or on the pass cannot be separated with three passes | separated one step further: at fixed workload type and KV the fraction still changes with b (0.8 → 0.33); the residual confound is the session; E1 (one launch) removes it |
| the skew arises inside the forward, likely from per-rank routed-expert load | consistent: at b=48 the MoE group time is 95 % correlated across ranks and the correlated segments carry the variance; at b=8 the MoE correlation is 0.61 and segments are mostly uncorrelated. Not a proof of mechanism: routing-conditional per-rank MoE time is E1's measurement (R² wall~routing is 0.06 at b=8 and 0.17 at b=48 on the routing statistics available here) |
| "the natural revision is a per-segment common-mode / independent mixture" | nothing fitted; the data say the mixture weight is b-dependent and that iterations are serially correlated within a wave, both to be carried into the E1 analysis plan |

## C. Outstanding discrepancies

### C.1 Empty Phase 0 cross-session reference section (b=8 report §5)

`analysis/e1_nsys_rank_kernels.py` builds section E from the reference pass's pure-decode iterations at the requested b. The
reference given to the b=8 post-run was `quadrant_nsys`, which contains **no** pure b=8 decode iteration (its b histogram: 1: 258,
2: 3, 4: 1, 24: 238, 32: 2, 48: 3032; the frozen RELIABILITY table shows `quad_n = 0` at b=8). The section was empty by
construction (`ref_n_iterations 0`), not because of a tool fault. The post-run script inherited `--ref-profile quadrant_nsys`
from the b=48 pass. Re-run against the Phase 0 pass that has b=8 (`diverse_nsys_run2`, 1675 iterations): `audit/rank_kernels_b8_vs_run2.md`.
Kernel set identical except the autotuner-selected `bmm` tactic variants (4 names only in the recovery pass, 3 only in run2:
`t128x8x256_s6` / `t128x8x512_s3` / `t128x8x256_s5` / `t128x8x512u2_s3`) and the hook's two PyTorch copy kernels; per-rank busy
5.13 vs 5.04 ms (+1.8 %), span 5.28 vs 5.19 ms. Hypothesis material only (different session, backend, KV distribution, hook).

### C.2 The three configuration-identity "FAIL" lines (b=8 report §4, `cpu/identity_nsys_val_vs_e1_b8_recovery.txt`)

`checks/e1_config_identity.py` compares the launch records of `e1_nsys_val` and `e1_b8_recovery`. The `--allow` list
(`name log_dir workload.frontend.path nsys.capture_start_delay_seconds`) applies to the *profile config* comparison, which PASSES
("identical apart from" exactly those four keys). The three FAIL lines are in the *launch record* comparison, which has no allow
list:
1. `argv: only in a / only in b` — the nsys `-o` output path, which contains the pass name (`…/e1_nsys_val/profile/nsys/dsv41_flash_1017_gb300_tp4_e1_nsys_val` vs `…/e1_b8_recovery/…`). Consequence of `name`/`log_dir`.
2. `launch nsys_config differs: ['nsys_config.capture_start_delay_seconds']` — 10.0 vs null: the one intended protocol change.
3. `launch records identical apart from the hook flag` — the summary line, which fails because of 1 and 2.
Server argv, server environment (PATH ignored) and the effective wire mode are identical (PASS). Nothing else differs. The checker
should accept an allow list for the launch-record comparison too; this is a tooling note, not a change made here.

### C.3 Can kernel-tactic differences account for the Stage 1.5 b=48 timing shift? (`analysis/e1_autotune_tactics.py`, `audit/tactics/`)

The shift: at b=48 the hook-on launch `e1_dry_on_v2` was 5.9 % faster (8.49 vs 9.02 ms engine median) than two hook-off launches
that agree to 0.04 %; the hook-on nsys launch agreed with the hook-on metrics launch to 0.5 %. FlashInfer re-tunes at every
launch (AUTOTUNER-INVESTIGATION.md).

*Autotune caches* (seven launches: Stage 1 off/on, Stage 1.5 off_v2/on_v2/off2_v2/nsys_val, b8_recovery; 160 keys each: 132
`mxfp8_gemm`, 27 `trtllm_fp4_block_scale_moe`, 1 generation): every pair of launches disagrees on 51–67 % of the chosen tactics.
`off_v2` vs `off2_v2` (timings equal to 0.04 %) differ on 93/160 (58 %); `off_v2` vs `on_v2` (timings 5.9 % apart) differ on
88/160 (55 %); `on_v2` vs `nsys_val` (timings 0.5 % apart) differ on 103/160 (64 %). The *number* of differing tactics carries no
timing information; the two launches that agree best in timing are not the two that agree best in tactics.

*Kernel traces* (the only launches with per-kernel time at b=48 are `e1_nsys_val` and Phase 0 `quadrant_nsys`; per iteration, four
ranks summed): tactic-dependent families (MoE `bmm` + dense block-scaled GEMM + quantize) 32.84 vs 31.67 ms (+1.17 ms, +3.7 % of
the family, ≈ +0.29 ms per rank per iteration ≈ 2.8 % of a 10.3 ms iteration), of which dense GEMM +0.88, MoE bmm +0.28; total GPU
time 47.16 vs 45.27 ms (attention +0.13 for KV 32 k vs < 8 k, other +0.41). At b=8 the same comparison (recovery vs run2) gives
−1 % in the tactic-dependent families (16.23 vs 16.39 ms) and +0.4 % in total.

*Conclusion.* Tactic re-selection between launches moves GPU time by about 1–3 % of the iteration at b=48 in the one cross-session
pair that has traces; that is the right order of magnitude for roughly half of the 5.9 % shift, and it moved in the *opposite*
direction there (the hook-on nsys launch is the slower one in GEMM time). With the existing data the shift **cannot** be attributed
to tactics: the hook-off launches have no kernel traces, the hook-on launches with traces also carry nsys overhead, and tactic-set
disagreement is uniform across all launch pairs. What would identify it (not requested, not authorized): a hook-off nsys pass in the
same session as a hook-on one, and/or two launches with the autotune cache pinned (the saved `autotune_configs.*.json` files make
pinning feasible); only then can the per-launch GPU time be decomposed by tactic.

### C.4 What cannot be identified without further matched experiments

* Hook overhead at b=48 at the 2 % level (Stage 1.5 §5): needs same-session on/off pairs, ideally with pinned tactics.
* Attribution of the between-launch 5–6 % drift at b=48 to tactic selection versus other launch-state (clocks, memory layout).
* Session versus batch-regime origin of the f_ind difference (0.8 → 0.33): E1's single launch with all b at every KV.
* Routing-conditional per-rank MoE time (mechanism of the rank-independent part): E1's hook-on capture with the corpus.
* Whether the b=48 single 5.3 ms stall is a workload-independent event: more pure b=48 iterations in one launch (E1 has ≥ 928 × 3).
* b=8 intrinsic/wait separation beyond the fastest-rank proxy: a different measurement (per-rank collective-kernel internal timing) that nsys does not provide.
