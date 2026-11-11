# Phase 2F: all-reduce by CUDA-graph replay on GB300, 2026-11-10

The 24 small-message `all_reduce`/nccl configurations (below 16 MiB, 12 at TP2
and 12 at TP4) were measured by CUDA-graph replay. **All 24 meet the stability
criterion**, every graph replay returned the correct sum on every rank, and
there is no discontinuity at the 16 MiB boundary. The graph figures are 13 to
47 µs at TP2 and 19 to 49 µs at TP4, against 42 to 62 µs stored.

The results are staged only. No existing measurement was replaced or deleted,
no database schema or simulator code was changed, and the tracked database is
untouched.

Follow-up: the integration that came after this phase, including two recorded
execution deviations (an extra 213 s GPU validation and a 345 MB disk peak),
is in `INTEGRATION_RESULTS.md`.

## Coverage and runs

tray12, rack 3; TP2 on GPUs 2 and 3, TP4 on GPUs 0 to 3, rank r on the r-th
GPU; clean worktree at `c5e247cf`.

| stage | window (UTC) | groups | candidate rows | boundary sizes | batches per row and method | SUM mismatches |
| --- | --- | --- | --- | --- | --- | --- |
| TP2 | 18:26:58 to 18:27:42 (44 s) | 5 of 5 | 12 of 12 | 8, 16, 128 MiB | 15 | 0 |
| TP4 | 18:28:31 to 18:29:28 (57 s) | 5 of 5 | 12 of 12 | 8, 16, 128 MiB | 15 | 0 |

GPU time 101 s of the 15 minutes allowed. Disk 82 MB, of which 74 MB is the
staging database copy. The environment guard passed before and after each
stage with an identical package fingerprint (torch 2.10.0+cu130, CUDA 13.0,
NCCL 2.28.9, `UV_NO_SYNC=1`). All 12 preflight checks show the expected UUIDs
and no compute process on any tray12 GPU. No timeout fired. Private DB
(`9474a224…`, 3128 GB300 rows) and tracked DB (`41c9cbe5…`) have unchanged
checksums.

## Harness and timing semantics

Existing graph-replay runners inspected: `flashinfer_trtllm.py`
(`_time_cuda_graph`), `flashinfer_mnnvl.py`, `vllm_lamport_triton.py`. Their
shared convention, reused unchanged in `allreduce_graph.py`:

- warm-up launches, capture, warm-up replays;
- `dist.barrier()`, then host wall time over the replays up to the end of
  `torch.cuda.synchronize()`, divided by the number of collectives;
- the row is the slowest rank's figure (`ReduceOp.MAX`);
- 10 back-to-back collectives per graph (`OPERATIONS_PER_GRAPH` in
  `flashinfer_mnnvl.py`, "amortizes host launch cost").

What is the same as the `all_reduce`/nccl runner: the call
(`dist.all_reduce(tensor)`, SUM), the bf16 buffer of `message_size_bytes`, the
launcher, one rank per GPU, NCCL and its algorithm. Only the timing boundary
differs. The runner itself is not modified; the harness is a separate file.

Added on top of the convention: three batches per group on one captured graph;
a correctness check of each graph before timing (refill with a deterministic
pattern, replay once, compare element by element, including the ten-collective
graph against sum × world_size⁹, exact in bf16); CUDA events around the same
replay loop as a second clock.

Method identifiers, stored with every observation and every staged row:

| identifier | meaning | role |
| --- | --- | --- |
| `nccl_torch_dist.cuda_graph_replay.ops10` | 10 collectives per graph, 100 replays per batch | production candidate |
| `nccl_torch_dist.cuda_graph_replay.ops1` | 1 collective per graph, 100 replays per batch | diagnostic |
| `nccl_torch_dist.eager_events` | the stored-row method | boundary sizes only |

Each rank file also carries the source commit, dirty flag, harness path,
torch and NCCL versions, GPU UUID and start time.

## Per-row results (production candidates, `…ops10`)

Times in µs. Group estimate = median of three batches; estimate = median of
the five group estimates. Stable = at least 4 of 5 within ±10 %. A spike batch
is one more than 25 % above the estimate. Nothing was discarded.

| TP | bytes | stored eager | graph replay | stored / graph | five group estimates | min | max | MAD | CV | within ±10 % | stable | spike batches | SUM mismatches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4 KiB | 57.4 | 13.7 | 4.2 | 13.7, 13.9, 13.7, 13.7, 13.9 | 13.7 | 13.9 | 0.06 | 0.9 % | 5/5 | pass | 0 of 15 | 0 |
| 2 | 8 KiB | 57.3 | 12.7 | 4.5 | 13.2, 12.7, 12.7, 12.7, 12.7 | 12.7 | 13.2 | 0.01 | 2.0 % | 5/5 | pass | 0 of 15 | 0 |
| 2 | 16 KiB | 53.0 | 14.3 | 3.7 | 14.9, 14.3, 14.3, 14.3, 14.3 | 14.3 | 14.9 | 0.01 | 2.1 % | 5/5 | pass | 0 of 15 | 0 |
| 2 | 32 KiB | 55.9 | 14.3 | 3.9 | 14.6, 14.3, 14.3, 14.3, 14.3 | 14.3 | 14.6 | 0.02 | 0.9 % | 5/5 | pass | 0 of 15 | 0 |
| 2 | 64 KiB | 52.9 | 14.4 | 3.7 | 14.4, 14.4, 14.4, 14.4, 14.4 | 14.4 | 14.4 | 0.00 | 0.1 % | 5/5 | pass | 0 of 15 | 0 |
| 2 | 128 KiB | 61.0 | 14.5 | 4.2 | 14.6, 14.4, 14.5, 14.5, 14.5 | 14.4 | 14.6 | 0.03 | 0.6 % | 5/5 | pass | 0 of 15 | 0 |
| 2 | 256 KiB | 59.5 | 14.5 | 4.1 | 14.5, 14.5, 14.5, 14.5, 14.5 | 14.5 | 14.5 | 0.00 | 0.1 % | 5/5 | pass | 0 of 15 | 0 |
| 2 | 512 KiB | 46.2 | 14.7 | 3.1 | 14.9, 14.7, 14.8, 14.7, 14.6 | 14.6 | 14.9 | 0.05 | 0.7 % | 5/5 | pass | 0 of 15 | 0 |
| 2 | 1 MiB | 49.3 | 17.4 | 2.8 | 17.3, 17.4, 17.4, 17.4, 17.3 | 17.3 | 17.4 | 0.04 | 0.3 % | 5/5 | pass | 0 of 15 | 0 |
| 2 | 2 MiB | 48.3 | 22.2 | 2.2 | 22.3, 22.3, 22.2, 22.2, 22.2 | 22.2 | 22.3 | 0.01 | 0.1 % | 5/5 | pass | 0 of 15 | 0 |
| 2 | 4 MiB | 61.8 | 41.4 | 1.5 | 41.4, 41.4, 41.3, 41.3, 41.4 | 41.3 | 41.4 | 0.04 | 0.1 % | 5/5 | pass | 0 of 15 | 0 |
| 2 | 8 MiB | 54.6 | 47.0 | 1.2 | 46.9, 46.8, 47.0, 47.1, 47.1 | 46.8 | 47.1 | 0.11 | 0.3 % | 5/5 | pass | 0 of 15 | 0 |
| 4 | 4 KiB | 51.8 | 19.1 | 2.7 | 19.3, 19.1, 18.8, 19.3, 18.6 | 18.6 | 19.3 | 0.23 | 1.7 % | 5/5 | pass | 0 of 15 | 0 |
| 4 | 8 KiB | 46.6 | 18.5 | 2.5 | 18.6, 18.5, 18.3, 18.7, 18.1 | 18.1 | 18.7 | 0.16 | 1.2 % | 5/5 | pass | 0 of 15 | 0 |
| 4 | 16 KiB | 52.0 | 19.7 | 2.6 | 20.0, 19.7, 19.6, 20.1, 19.2 | 19.2 | 20.1 | 0.26 | 1.8 % | 5/5 | pass | 0 of 15 | 0 |
| 4 | 32 KiB | 42.2 | 20.3 | 2.1 | 20.5, 20.3, 20.1, 20.5, 19.9 | 19.9 | 20.5 | 0.22 | 1.3 % | 5/5 | pass | 0 of 15 | 0 |
| 4 | 64 KiB | 49.2 | 19.6 | 2.5 | 19.8, 19.6, 19.4, 20.3, 19.1 | 19.1 | 20.3 | 0.28 | 2.3 % | 5/5 | pass | 0 of 15 | 0 |
| 4 | 128 KiB | 41.2 | 20.6 | 2.0 | 20.8, 20.6, 20.4, 20.9, 20.1 | 20.1 | 20.9 | 0.24 | 1.5 % | 5/5 | pass | 0 of 15 | 0 |
| 4 | 256 KiB | 44.3 | 20.8 | 2.1 | 20.8, 20.9, 20.7, 21.0, 20.3 | 20.3 | 21.0 | 0.13 | 1.2 % | 5/5 | pass | 0 of 15 | 0 |
| 4 | 512 KiB | 43.3 | 20.2 | 2.1 | 20.6, 20.2, 20.1, 20.3, 19.8 | 19.8 | 20.6 | 0.13 | 1.4 % | 5/5 | pass | 0 of 15 | 0 |
| 4 | 1 MiB | 46.1 | 23.3 | 2.0 | 23.3, 23.5, 23.0, 23.6, 22.8 | 22.8 | 23.6 | 0.31 | 1.5 % | 5/5 | pass | 0 of 15 | 0 |
| 4 | 2 MiB | 44.5 | 27.0 | 1.6 | 27.0, 27.0, 27.0, 27.1, 26.7 | 26.7 | 27.1 | 0.04 | 0.5 % | 5/5 | pass | 0 of 15 | 0 |
| 4 | 4 MiB | 43.9 | 40.1 | 1.1 | 40.1, 40.1, 40.1, 40.2, 39.2 | 39.2 | 40.2 | 0.04 | 1.1 % | 5/5 | pass | 0 of 15 | 0 |
| 4 | 8 MiB | 45.8 | 48.5 | 0.9 | 48.5, 48.8, 48.5, 48.6, 47.5 | 47.5 | 48.8 | 0.05 | 1.0 % | 5/5 | pass | 0 of 15 | 0 |

- Largest coefficient of variation 2.3 %; largest group-to-group range 5.9 %
  of the estimate; no spike batch in 360.
- The CUDA-event clock agrees with the host clock within 0.7 %, and the ranks'
  own times within a batch differ by at most 8 %.
- The one-collective graph gives the same figures (0.93 to 1.05 times the
  ten-collective graph) and is also stable 24 of 24, with 4 spike batches of
  360. One replay launch per collective is therefore not a bottleneck here.
- Shape: flat at 13 to 15 µs (TP2) and 19 to 21 µs (TP4) up to 512 KiB, then
  rising with size from 1 MiB. The eager rows hid this under the host floor.

## Boundary continuity

Eager and graph in the same rank groups; median and group range.

| TP | bytes | stored eager row | eager, this phase | graph ×10 | graph ×1 | graph ×10 / eager now |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 8 MiB | 54.6 | 43.8 (43.5 to 47.7) | 47.0 (46.8 to 47.1) | 45.7 | 1.07 |
| 2 | 16 MiB | 59.8 | 53.8 (53.6 to 53.9) | 56.2 (56.1 to 56.2) | 55.9 | 1.04 |
| 2 | 128 MiB | 282.6 | 275.6 (275.5 to 275.6) | 278.1 (278.0 to 278.2) | 277.8 | 1.01 |
| 4 | 8 MiB | 45.8 | 45.3 (45.0 to 51.2) | 48.5 (47.5 to 48.8) | 48.5 | 1.07 |
| 4 | 16 MiB | 68.3 | 68.4 (68.2 to 68.9) | 71.2 (70.6 to 71.4) | 71.1 | 1.04 |
| 4 | 128 MiB | 363.0 | 359.9 (357.0 to 361.1) | 362.8 (362.1 to 367.9) | 362.9 | 1.01 |

- From 8 MiB up the two methods agree within 1 to 7 %; graph replay is a
  constant 2.5 to 3 µs higher.
- The staged curve steps from graph at 8 MiB (47.0 / 48.5 µs) to the stored
  eager row at 16 MiB (59.8 / 68.3 µs). A graph measurement at 16 MiB would
  read 56.2 / 71.2 µs, so the mismatch at the boundary is −6 % at TP2 and
  +4 % at TP4. The curve stays monotonic; there is no discontinuity to fix.
- The stored TP2 rows at 8 and 16 MiB sit 11 to 25 % above this phase's eager
  repeats, the bias already seen in Phase 2D.

## Prediction comparison

Staging dataset: a copy of the private DB in which the 24 small-message rows
carry the graph estimates and every other row is unchanged. Total µs per
iteration; all-reduce share in brackets. TP1 does not depend on these rows.

| case | bytes | TP1 | TP2 eager rows | TP2 graph rows | TP4 eager rows | TP4 graph rows |
| --- | --- | --- | --- | --- | --- | --- |
| prefill, 256 tokens | 2 MiB | 4,564 | 6,062 (51 %) | 4,395 (32 %) | 5,059 (56 %) | 3,938 (44 %) |
| prefill, 2048 tokens | 16 MiB | 21,180 | 15,549 (25 %) | 15,549 (25 %) | 12,142 (36 %) | 12,142 (36 %) |
| decode, 1 request | 8 KiB | 3,375 | 6,210 (59 %) | 3,352 (24 %) | 4,987 (60 %) | 3,186 (37 %) |
| decode, 8 requests | 64 KiB | 3,478 | 5,920 (57 %) | 3,453 (27 %) | 5,118 (62 %) | 3,227 (39 %) |
| decode, 64 requests | 512 KiB | 4,169 | 5,665 (52 %) | 3,648 (26 %) | 4,951 (56 %) | 3,475 (37 %) |
| decode, 256 requests | 2 MiB | 7,340 | 7,537 (41 %) | 5,870 (24 %) | 5,758 (50 %) | 4,636 (37 %) |

Rankings (fastest first):

| case | with eager rows | with graph rows |
| --- | --- | --- |
| prefill, 256 tokens | TP1, TP4, TP2 | TP4, TP2, TP1 |
| prefill, 2048 tokens | TP4, TP2, TP1 | unchanged |
| decode, 1 request | TP1, TP4, TP2 | TP4, then TP2 and TP1 tied |
| decode, 8 requests | TP1, TP4, TP2 | TP4, then TP2 and TP1 tied |
| decode, 64 requests | TP1, TP4, TP2 | TP4, TP2, TP1 |
| decode, 256 requests | TP4, TP1, TP2 | TP4, TP2, TP1 |

Sensitivity:

- The group-to-group range of a graph row is at most ±3 % of the row, which is
  at most ±1.3 % of an iteration. Every ranking above holds across that range
  except TP2 against TP1 at 1 and 8 requests, where they differ by 0.7 % and
  are a tie.
- Using a graph figure at 16 MiB instead of the stored eager row would move
  the 2048-token prefill by −1.5 % (TP2) and +1.5 % (TP4). No ranking changes.
- Per iteration the all-reduce is still 24 to 44 % of TP2/TP4 time, so the
  conclusion "communication is a large share of small-model TP" survives; the
  conclusion "TP is slower than TP1 for decode" does not.

## Comparing GB300 with H200 and B200

The H200 and B200 small-message rows are eager-timed (floors of about 31 and
27 to 45 µs). If GB300 used graph rows and they did not:

- a GB300 TP2 or TP4 decode iteration would be credited about 1.5 to 2.8 ms
  (64 all-reduces × 23 to 44 µs) that comes from the method, not the hardware;
- any GB300-against-H200/B200 comparison of TP decode or small prefill would
  be biased towards GB300 by that amount, on iterations of 3 to 6 ms;
- TP1 comparisons and large-prefill TP comparisons are unaffected (rows from
  16 MiB up agree between methods within 7 %).

The database has no column for the measurement method, so the distinction
would live only in the run record's commit and in documentation unless the
graph rows get their own backend name.

## Recommended integration policy

1. **Integrate now:** the 1,854 single-GPU rows and the 14 eager all-reduce
   rows from 16 MiB up, as already accepted.
2. **Adopt the 24 graph-replay rows for GB300**, replacing the 24 eager rows
   in the private database, then integrate. They are stable, correct, and
   consistent with how every other leaf is costed. This needs your approval
   because it replaces private measurements.
3. **Make the method visible.** Preferred: a distinct backend name for
   graph-timed rows (for example `nccl_graph`), which needs a small
   registration and backend-list change but no schema or cost-model change,
   and keeps the eager rows side by side. Minimum: a documented note plus the
   run record. This also needs approval.
4. **Do not publish GB300-against-H200/B200 TP comparisons for decode or small
   prefill** until the other GPUs' small-message rows are re-measured the same
   way upstream. GB300-only TP predictions are sound with the graph rows.
5. Keep the eager rows and the Phase 2D, 2E and 2F raw samples as they are.

Not established: whether vLLM's own NCCL wrapper replays at the same cost as
`torch.distributed`'s call, and why the stored eager TP2 rows sit above later
eager repeats.

## Artifacts

Under `$GB300_ROOT/artifacts/phase2f/`:

- `tp2-run-20261110-182658/`, `tp4-run-20261110-182831/` (`tp2-latest`,
  `tp4-latest`): `raw/group<g>-rank<r>.json` with every batch of every method
  (host clock, event clock, slowest-rank figure), correctness counts and
  provenance; manifests, guards, isolation logs, GPU samples.
- `graph-stability.json` (committed as `phase2f/graph-stability.json`).
- `staging/graph-rows.json` (committed as `phase2f/graph-rows.json`): the 24
  candidate rows with method identifier, group and batch values, the eager
  value each would replace, and provenance.
- `staging/profile-graph-staging.db`, sha256 `f301632f…`: the isolated
  prediction dataset. Not a production database.
- `predict/`: presets, cases, outputs; summary committed as
  `phase2f/predict-summary.json`.
