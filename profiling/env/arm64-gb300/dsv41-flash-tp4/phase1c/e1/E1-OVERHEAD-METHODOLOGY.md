# E1 overhead methodology (Phase 1.5 item E) — pre-registered before any Stage 1.5 GPU run

Scope: the hook-on vs hook-off comparison of pure-decode iteration time (`observed_elapsed_ms` of iterations with no prefill
and exactly b decode requests of one cell), per cell, as in the packet (§4) and Stage 1. Implemented in
`analysis/e1_pair_compare_v2.py` and (gate A + interval) `checks/e1_completeness_v2.py --compare`. Nothing below was chosen
by looking at which rule Stage 1 would pass: the gate value (1.02) is the packet's, the block-length rule and the verdict
labels were fixed before the Stage 1.5 configs were generated, and Stage 1 numbers appear only as illustration (§5).

## 1. Gate A — the pre-registered point gate (unchanged)

Per cell: `median(on) / median(off) <= 1.02` -> PASS, else FAIL. This remains the acceptance gate for the hook overhead.
The Stage 1 result (b8 1.0086 PASS, b48 1.0195 PASS) stands as reported.

## 2. Gate B — prospective uncertainty-aware criterion (applies to runs after 2026-11-14)

Per cell, on the same two series:

1. **Serial correlation is measured, not assumed away.** ACF at lags 1..20; integrated autocorrelation time
   `tau_int = 1 + 2 * sum_{k>=1} acf(k)` truncated at the first non-positive lag (Geyer's initial positive sequence).
2. **Block length.** `L = clip(ceil(5 * tau_int), 10, n // 20)` per arm (minimum 10 so that drift at the scale of tens of
   iterations is inside a block; maximum n/20 so that at least 20 blocks are resampled).
3. **Moving-block bootstrap of the ratio of medians.** 2000 draws, seed 20261114; in each draw both arms are resampled
   independently with overlapping blocks of their own L, the medians are taken and divided. Report the 2.5 % and 97.5 %
   quantiles as the 95 % interval.
4. **Verdict.** FAIL if the point ratio > 1.02 (Gate A fails); PASS if the upper bound <= 1.02; otherwise INCONCLUSIVE.
   INCONCLUSIVE is not a failure of the hook; it says the run did not have the power to show overhead <= 2 % at 95 %.
5. **Drift diagnostics (reported, no gate):** medians of the first and second half and per 100 iterations, so that a
   monotone drift (thermal, autotuner-induced, scheduler) is visible rather than folded into "noise".

Why blocks: Stage 1 series are not independent observations (b48 off acf lag 2 = 0.25, b48 on acf lag 1 = 0.17 with
tau_int 5.3; per-100-iteration medians move by 0.1 ms). A naive bootstrap of the b48 hook-on median gives SD 0.0044 ms; with
blocks of 10/25/50 iterations it gives 0.0075/0.0097/0.0110 ms. Independence would overstate the precision by 2x.

## 3. Repeated-run variability: the off/off control

Two hook-off launches (`e1_dry_off_v2`, `e1_dry_off2_v2`) with byte-identical configuration apart from `name`/`log_dir`
(checked by `checks/e1_config_identity.py`) give the run-to-run null `control/off`, computed exactly as in §2 (point, interval,
tau, L). It contains autotuner tactic re-selection (AUTOTUNER-INVESTIGATION.md), clocks/thermals and host scheduling.

Rules fixed now:

* **Noise flag.** If the control/off 95 % interval reaches outside `[1/1.02, 1.02]` the cell is flagged NOISY: the between-launch
  variation alone is of the size of the gate, and Gate B on on/off is reported as "INCONCLUSIVE (noise)" whatever its own
  interval says. If the interval stays inside, the flag is "quiet" and Gate B is read as is.
* **Excess.** `excess = (on/off) / (control/off)` with its interval from the same bootstrap draws (joint resampling). It is
  reported for interpretation (how much of on/off is beyond one off/off replicate); it carries no gate, because one control
  pair is one realization of the between-launch difference, not an estimate of its distribution.
* **Replicates.** One control pair is the minimum that can show the problem; it cannot bound it. If the control is NOISY and
  the question matters for Stage 2, more off/off replicates (each ~25 min of GPU) are the remedy, to be requested separately.
* **Token determinism** (`output_preview` identical per request) is reported for control vs off as well; Stage 1 showed 21/56
  identical between hook-off and hook-on, so a hook-independent baseline is needed before reading that number.

## 4. What is not done

No threshold is tuned on Stage 1 outcomes; no model is fitted; no re-analysis replaces the Stage 1 report. The prospective
criterion and the control are added for future runs; Stage 1 remains judged by Gate A as pre-registered.

## 5. Illustration on Stage 1 (not a re-adjudication)

`runs/stage15/pair_v2/E1-PAIR-COMPARE-V2.md`: b8 ratio 1.0086, L = 10/10, interval [1.0066, 1.0100] -> would be PASS;
b48 ratio 1.0195, L = 10/27, interval [1.0168, 1.0221] -> would be INCONCLUSIVE. No control existed in Stage 1; a smoke run
with the hook-off run as its own control returns control/off = 1.0000 with interval [0.9979, 1.0020] (quiet) as expected
(`runs/stage15/pair_v2_selfcontrol_smoke/`).
