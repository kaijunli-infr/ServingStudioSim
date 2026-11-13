# Proposed Phase 1C validation plan (not authorised; for review)

Goal: validate the Phase 1B sequence simulator on data it has never seen, and close the identified
gaps with the smallest set of GPU captures. Every item below needs a separate go; nothing in Phase 1B
runs on a GPU.

## 1C-0 CPU-only, no new data (can start after this review)

| item | what | pass criterion |
|---|---|---|
| sequence replay of the salvaged pass with the Phase 1B parameters refitted on run2 + quadrant only (already the case) | cross-check of lead, skew, span on a pass never used for any fit | span error ≤ 1 %, lead p90 abs error ≤ 20 ms on chains |
| schedule perturbation tests | re-order / drop iterations of the replayed schedule and check that the engine's invariants hold (no negative gaps, queue depth never exceeded, barrier rule) | invariants hold on 1,000 random perturbations |
| interface contract | JSON schedule in / per-iteration timeline out, versioned schema, to be consumed by the Rust L5 worker (`simulator/src/worker`) through the existing PyO3 bridge | round-trip test |

## 1C-1 GPU captures (priority order, one rack-10 tray, frozen preset, nsys + host timeline)

| prio | capture | closes | size |
|---|---|---|---|
| 1 | synthetic request mix with **host-bound eager mixed iterations at long KV** (P 2–8k, kv0 32–128k, b ≤ 8) | c_kv (KV-dependent launch cost) and the eager p90 APE of 10–18 %; also the rank-2 host lateness under controlled load | ≈ 20 min |
| 2 | **decode b = 1–4 at kv < 32k** with both request mixes and a per-rank host timeline | the pass-dependent rank-lateness pattern (rank 0 vs ranks 1/3); b1-2 coverage | ≈ 10 min |
| 3 | **chunk chains of 10+ chunks** (kv0 to 160k) with the host timeline | queue depth Q (saturation on more than 8 chunks), indexer slope beyond 115k (out-of-range today) | ≈ 15 min |
| 4 | **MoE routing capture** (expert ids per layer per iteration, vLLM-side hook, code change to be reviewed first) | per-iteration routing (gap G3); replaces the statistical band | ≈ 10 min + review |
| 5 | **graph-path mixed sweep** (P ≤ 256, b 1–64) | two class medians → table | ≈ 10 min |
| 6 | **16k chunks at kv0 = 0 after idle GPU** (cold start) and after decode | f_idle (n = 6 today) and the 2–4 % under-prediction at kv0 = 0 | ≈ 5 min |

Analysis for every capture: run `costmodel.extract` (hash-verified), add the pass to a *new* frozen
partition as held-out only (never calibration), run all four Phase 1B modes, and compare against the
tables of this report. Acceptance for Phase 1C: held-out decode p90 APE ≤ 5 % in every batch bucket,
chunk p90 APE ≤ 3 % to 160k, eager host-bound p90 APE ≤ 10 %, full-workload span error ≤ 1 %, and
uncertainty coverage of 70–90 % at p10–p90 in every class with n ≥ 30.

## 1C-2 production integration (after 1C-1)

Wire the sequence engine behind an opt-in flag in the Rust L5 worker (sequence cadence shell), keep
the existing predictions bit-identical when the flag is off (golden test), and add the DSV41 TP4 GB300
arch as a new `profiling/env/arm64-gb300/dsv41-flash-tp4` kernel database only after the Phase 1C
captures confirm the coefficients.
