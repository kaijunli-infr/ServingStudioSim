# Kernel rows per iteration in the preserved V2 and V3 captures (CPU-only, 2026-11-19)

Counts only, from the extract tables the original post-runs wrote (`phase1a-tables/e1_v2_forced_release`, `…/e1_v3_scale`:
`iterations.parquet`, `iter_rank.parquet`). No timing model, no rank-dependence parameter, no identification partition. Script:
`e1_rows_per_iteration.py`; generated tables: `E1-ROWS-PER-ITERATION.md`, `.json`. **M** = measured, **A** = assumption.

## 1. Measurements

Six waves were captured under the lifecycle controller: b=8 and b=48 at 32k context, once in V2 and twice in V3.

| cell | waves | iterations per wave | kernel rows per wave | rows / iteration, all | pure decode at full batch | prefill-carrying iterations |
|---|---|---|---|---|---|---|
| b=8, 32k | V2; V3 wave 0; V3 wave 1 | 1040 | 5 887 244 | 5660.8 | 1008 iterations: 5637 (759 ×) or 5645 (249 ×), mean 5639.0 | 17 iterations, 6229 – 7233, mean 6995.6 |
| b=48, 32k | V2; V3 wave 0; V3 wave 1 | 1120 | 6 363 588 | 5681.8 | 928 iterations: 5533 (435 ×) or 5541 (493 ×), mean 5537.3 | 97 iterations, 6229 – 7301, mean 7176.2 |

- **Sum check (M).** The waves add up exactly to the recovered traces: 12 250 832 rows (V2) and 24 501 664 rows (V3).
- **Wave-to-wave consistency (M).** The three waves of each cell are identical: same iteration count, same kernel rows, and the
  same sequence of (stage, decode requests, prefill chunks, rows, rows per rank) iteration by iteration (one SHA-256 over the
  sequence per wave; three equal values per cell). This holds across two separate launches (V2, V3) and within one session
  (V3 wave 0 and wave 1, 80 s apart).
- **Repeated cells are stable (M).** Kernel-row counts are deterministic for these two cells; no variation was observed at all.
- **Rank-to-rank (M).** In every iteration rank 0 launches exactly one kernel more than ranks 1 – 3, which are equal to each
  other: b=8 pure decode 1410 / 1409 / 1409 / 1409 (or 1412 / 1411 / 1411 / 1411); b=48 1384 / 1383 / 1383 / 1383 (or 1386 /
  1385 / 1385 / 1385). Per wave: b=8 1 472 591 on rank 0 against 1 471 551 on each other rank (difference 1040 = one per
  iteration); b=48 1 591 737 against 1 590 617 (difference 1120). No other rank difference exists in the counts.
- **Batch size (M).** A pure-decode iteration has 1.8 % fewer kernel rows at b=48 than at b=8 (5537 against 5639). Rows per
  iteration are therefore not a constant; they depend on the batch size, weakly, in the one comparison available.
- **Prefill (M).** An iteration that carries a prefill chunk has 6229 – 7301 rows, 10 – 30 % more than a decode iteration.
- **Iterations per wave (M).** 1040 = 1024 + 2 × 8 and 1120 = 1024 + 2 × 48: the output length plus two prefill chunks per
  request at 32k.

## 2. What this means for V4

V4 repeats these two cells four times. Its expected size is 4 × 12 250 832 = **49 003 328 rows in 8640 iterations**, from
repeated identical measurements of the same waves, not from a scaling assumption. The registered band stays ± 5 % as in V3.
The repeats add no information about other cells: a fourth identical wave is a replicate.

## 3. Is the earlier full-E1 estimate of about 216 M rows consistent with the cell-level data?

It is consistent, **under an assumption that the data cannot test**.

- Arithmetic (`full_e1_consistency` in the JSON): iterations per wave = 1024 + b × prefill chunks, with 1, 2 and 6 chunks at
  4k, 32k and 96k (this reproduces both measured cells; the chunk count at 4k and 96k is **A**). The 36 planned waves then
  have 38 214 iterations (12 438 / 12 588 / 13 188 for the three context lengths).
- Bracket: every decode iteration between the lowest and the highest measured mean (5537.3 and 5639.0 rows), every
  prefill-carrying iteration between the lowest and highest measured value (6229 and 7301 rows). Result: **212.5 – 217.7 M
  rows**. The earlier figure, 216 M, lies inside.
- The assumption (**A**): that rows per iteration at batch sizes 1, 2, 4, 16, 24 and 32, and at 4k and 96k context, lie inside
  the bracket of the two measured cells. Two of 24 cells are measured, both at 32k. The one measured dependence (batch size,
  1.8 %) is small, which makes a large deviation unlikely but does not exclude it; a dependence on context length has not been
  measured at all.
- So the 216 M estimate is neither confirmed nor contradicted. It is consistent with everything measured, and its uncertainty
  is no longer "± 20 %" by default: if the assumption holds, the bracket is ± 1.2 %; if it does not, the size is unknown.
