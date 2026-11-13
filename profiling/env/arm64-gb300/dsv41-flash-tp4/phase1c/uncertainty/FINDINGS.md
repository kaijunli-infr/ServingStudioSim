# Uncertainty audit — findings (Phase 1C-0)

Source tables: `RELIABILITY.md` / `reliability.json` (5 seeds × 32 draws × 3 passes, 467 s) and
`../out/logs/wait_decomposition.log` (`wait_decomposition.py`). Nothing in the model was changed; no band was widened;
no held-out row was used for any fit.

## 1. Reliability by class

- **Seed variation is negligible** for every class with n ≥ 50: the seed-to-seed standard deviation of the signed bias
  is ≤ 0.13 points and of the p10–p90 coverage ≤ 2.3 points (run2 decode b3-16: 0.013 points / 0.38 points). For
  classes with n ≤ 14 (eager, ring other, graph) coverage moves by whole iterations between seeds and is not
  interpretable; those classes are reported, not assessed.
- **Decode b3-16** (run2 n = 6,935 held-out): bias +0.5 %, median |err| 0.063 ms, p90 |err| 0.154 ms, coverage 58 %
  with a 0.16–0.18 ms band (2.8 % of the wall); 30 % of the measurements fall below the band and 12 % above. The
  salvaged pass behaves the same (bias +0.4 %, coverage 69 %).
- **Quadrant decode** (b3-16, b17-24, b≥25): bias +1.3 to +2.5 %, coverage 18–51 %, with 47–82 % of the measurements
  **below** the band: the band is displaced upward, not merely narrow. The median path (pred-det) sits on the other side,
  0.9–1.2 % below the measurement.
- **Decode b1-2**: wide bands (1.3–1.6 ms, 27–33 % of the wall) and over-coverage (85–96 %); the band inherits the
  bimodal rank-lateness pattern of the Phase 1A per-rank quantiles. Lower-confidence class (kept).
- **16k chunks**: coverage 89–100 % with an 8 ms (3.5 %) band; bias ≤ 0.3 %; well calibrated.
- **Salvaged decode b17-24** (n = 34 + 32): bias +3.3 / +4.9 %, coverage 0 %. The b17-24 table was built from quadrant
  calibration rows only; its transfer to another pass fails by 3–5 %. This is the clearest **distribution shift** in the
  data (same b, different pass, different request mix).

## 2. Interval displacement — decomposition

Switching the stochastic components on one at a time (`RELIABILITY.md`, section "Interval displacement"):

| component | effect on the mean (all decode classes, both passes) | effect on coverage |
|---|---|---|
| V0 → V1: independent per-(rank, segment) quantile draws composed through the barrier | **+3.1 to +3.7 points** (the whole displacement) | 3–21 % (band 0.05 ms, too narrow) |
| V1 → V2: common-mode dispersion factor | < 0.1 point | widens to 12–55 % |
| V2 → V3: MoE routing band | < 0.1 point | widens to 18–62 % |

So the displacement is **not** routing variability (the MoE band is symmetric around 1 and contributes 0.6–2.2 % of
relative band width, see `reliability.json["moe_band"]`), **not** the dispersion factor, and **not** parameter noise
(seed variation ≈ 0). It is a structural property of how per-rank variation is sampled.

## 3. Mechanism: how much of the per-segment variation is rank-independent?

`wait_decomposition.py` compares the **measured** mean collective wait per rank in the Phase 1A tables with the wait
implied by (a) the median path, (b) independent per-rank draws and (c) common-mode draws (same quantile on all ranks):

| b | K | measured wait / rank (ms) | median path | independent | common-mode | independent fraction |
|---|---|---|---|---|---|---|
| 4 | 163 | 0.169 0.106 0.091 0.107 | 0.092 0.031 0.009 0.023 | 0.206 0.142 0.126 0.142 | 0.097 0.032 0.017 0.033 | ≈ 0.68 |
| 8 | 163 | 0.225 0.188 0.182 0.183 | 0.102 0.039 0.038 0.042 | 0.258 0.224 0.214 0.216 | 0.106 0.073 0.062 0.065 | ≈ 0.80 |
| 12 | 163 | 0.250 0.211 0.209 0.209 | 0.129 0.092 0.075 0.074 | 0.312 0.273 0.273 0.270 | 0.117 0.078 0.077 0.075 | ≈ 0.68 |
| 24 | 86 | 0.168 0.077 0.066 0.068 | 0.128 0.027 0.007 0.010 | 0.294 0.194 0.177 0.180 | 0.129 0.029 0.012 0.015 | ≈ 0.30 |
| 48 | 86 | 0.140 0.115 0.076 0.077 | 0.079 0.050 0.013 0.009 | 0.366 0.341 0.304 0.304 | 0.074 0.049 0.012 0.011 | ≈ 0.25 |

(exact values in the log; the fraction is (measured − common) / (independent − common), averaged over ranks.)

Reading:

- The **median path has almost no collective wait** (ranks are at their medians, hence aligned), but the traces show
  0.1–0.25 ms of wait per rank per iteration. This missing wait is most of the median path's −1 to −3 % bias: at b = 8
  the median-path wall (5.021 ms) is 0.17 ms below the table's measured median wall (5.195 ms), and the missing wait is
  0.15 ms.
- **Independent draws over-produce wait**, slightly at b ≤ 16 (run2, salvaged: measured wait is 70–80 % of the way from
  common-mode to independent) and strongly at b ≥ 24 (quadrant: 25–30 %). That is exactly the pattern of the
  displacement: run2 pred-mc +0.5 %, quadrant +2.2 %.
- The **first-collective arrival spread** across ranks is tiny for b ≥ 3 in every pass (0.016–0.02 ms median,
  `reliability.json["table_draws"][b]["arrival_spread"]`), so host-arrival variation is **not** the source of the
  per-rank skew for b ≥ 3; the skew arises inside the forward (per-segment, likely the routed-expert load per rank).
  For b = 1–2 the arrival spread is 0.45–0.9 ms (rank-lateness pattern), which is a host-arrival effect and is why the
  b1-2 band is wide.
- Whether the rank-independent fraction depends on **b** (K = 163 collectives for b ≤ 16, 86 for b ≥ 17: a different
  kernel path) or on the **pass** (quadrant vs diverse) cannot be separated with these three passes: every b ≥ 17 row
  comes from the quadrant pass. This is the confound that experiment E1 (one pass, b sweep 1–48) removes.

## 4. Attribution summary

| source | evidence | contribution to the displacement / coverage gap |
|---|---|---|
| routing variability | MoE band 0.6–2.2 % of wall width, symmetric | ≈ 0 to the mean; part of the band width; its *per-rank* effect (expert load imbalance → rank skew) is the likely physical cause of the rank-independent fraction but is unobserved (gap G3) |
| host-arrival variation | first-collective spread 0.016–0.02 ms (b ≥ 3), 0.45–0.9 ms (b ≤ 2) | ≈ 0 for b ≥ 3; dominant for b1-2 |
| parameter uncertainty (sampling) | seed std ≤ 0.13 points; table n 664–2,425 for the main b; **interpolated b** (5, 10, 14, 15, 26, 28–47: n = 0–5) get bands 10–40× wider from the quantile interpolation (e.g. b = 14 width 0.90 ms vs 0.06 ms at b = 13) | small for the means; a visible artefact in `reliability.json["table_draws"]` that inflates the band of a few hundred quadrant iterations |
| distribution shift | salvaged b17-24 +3.3/+4.9 % vs quadrant-built table; b = 1 median 4.57 (run2) vs 4.81 ms (quadrant) | 3–5 % on cross-pass transfer of thin tables; ≈ 1–2 % between run2 and quadrant for the same b |
| model structure (rank sampling) | V0→V1 = whole displacement; measured wait between common-mode and independent | **the displacement itself**: median path −1 to −3 %, independent mean +0.5 to +2.5 % |

## 5. What follows (no change made here)

The natural revision is a per-segment common-mode / independent mixture for the decode draws (one fraction, or one per
collective kind), which would put the mean between the two current paths and widen the band through the correlated
part. Its value is not identifiable from the three passes without the b / pass confound; E1 measures it directly
(per-rank MoE time vs routed load) and E2 validates the revised model once, under the frozen protocol. Until then the
reported numbers stand as they are, with the median path and the stochastic mean bracketing the measurement by ≈ 3 %.
