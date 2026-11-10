# Phase 2D: small-message all-reduce stability on GB300, 2026-11-10

The 24 `all_reduce`/nccl configurations below 16 MiB (12 at 2 GPUs, 12 at 4)
were re-measured diagnostically: five fresh rank groups each, three batches of
100 collectives per group, all 360 batch observations kept. **12 of 24 meet the
stability criterion** (4 of 12 at TP2, 8 of 12 at TP4). The new estimates are
therefore not validated, and nothing in a profile database was changed.

What the data does establish:

- The small-message latency is one size-independent floor. All 24 diagnostic
  estimates lie between 43.1 and 50.4 µs, from 4 KiB to 8 MiB.
- The floor is paced by the host, not by the GPUs or the links. The host's
  per-call enqueue time is 75 to 97 % of the measured time, and the median gap
  between successive calls (32 to 47 µs) tracks each group's floor.
- Groups differ from each other. A group keeps its own level across all sizes
  (TP2 group medians 38.2, 38.7, 45.1, 46.7, 50.5 µs), which is why most rows
  fail a ±10 % rule over five groups.
- Spikes are host stalls: 39 of 360 batches exceed 1.25 times their group's
  median, and in 30 of them some rank's host paused for more than 0.5 ms
  between two calls (longest gap 14.8 ms). The other 9 are not explained.
- The stored TP2 rows are biased high (median change −18 %); the stored TP4
  rows scatter around the new estimates (median +2 %).

Why groups settle at different levels is not established.

## Coverage and runs

tray12, rack 3; TP2 on GPUs 2 and 3, TP4 on GPUs 0 to 3, rank r on the r-th
GPU; clean worktree at `a8c3c88f`. The 14 configurations at or above 16 MiB and
all single-GPU kernels were not touched.

| stage | window (UTC) | groups | configurations | batches | SUM mismatches |
| --- | --- | --- | --- | --- | --- |
| TP2 | 17:04:27 to 17:05:00 (33 s) | 5 of 5 | 12 of 12 | 180 | 0 on every rank |
| TP4 | 17:05:19 to 17:06:05 (46 s) | 5 of 5 | 12 of 12 | 180 | 0 on every rank |

GPU time used: 79 s of the 20 minutes allowed. Disk: 97 MB, of which 74 MB is
the isolated database copy for the sensitivity runs. Environment guard passed
before and after each stage with an identical package fingerprint (torch
2.10.0+cu130, CUDA 13.0, NCCL 2.28.9, `UV_NO_SYNC=1`). All 12 preflight checks
show the expected UUIDs and no compute process on any tray12 GPU. No timeout
fired, nothing was killed. sha256 of the private DB (`9474a224…`) and of the
tracked DB (`41c9cbe5…`) are the same before and after; 3128 GB300 rows.

Each fresh group ran all 12 sizes in ascending order, as the runner's chunk
does, so every configuration saw five independent groups.

## Harness inspection

What a stored row is (`profiling/runners/comm/nccl.py`, `_batch.py`):

- **Quantity.** Rank 0's `start.elapsed_time(end) / 100`, two CUDA events on the
  rank's current stream around 100 back-to-back `dist.all_reduce(tensor)`
  calls. One batch, one number; no per-call or per-rank data, no host clock.
  Other ranks compute the same figure and discard it.
- **What the events bracket.** `dist.all_reduce` enqueues the NCCL kernel and
  returns; with `TORCH_NCCL_BLOCKING_WAIT` unset (as here) its `wait()` only
  makes the current stream wait for the NCCL stream. The end event therefore
  fires when the last collective has finished on the GPU, and the interval
  covers the host's enqueue loop plus any GPU work still queued.
- **Host stalls.** They contaminate the result. When the host is the slower
  side, as below 16 MiB, the GPU idles between calls and any pause of the
  Python loop on any rank lands inside the interval. A 5 ms pause adds 50 µs
  to a 100-call mean.
- **Initialisation and warm-up.** `init_process_group` does the TCP rendezvous
  only. NCCL creates its communicator in the first collective, which is the
  first of 50 warm-up calls for the first size; `torch.cuda.synchronize()` and
  `dist.barrier()` follow before the start event. Teardown is after the last
  size. Neither is inside a timed interval.
- **Aggregation above the runner.** The profiler stores that one mean. The
  outlier policy for this kind is a placeholder, so no retry happens.

Harness added for this phase (no change to the runner, launcher, NCCL, the
reduction, tensor shapes or the simulator):

- `allreduce_diag.py`: one group through the same `TorchMpLauncher`; per size a
  SUM check on every rank, the runner's warm-up, then three batches timed with
  the runner's exact event sequence. Added: one `perf_counter_ns()` read after
  each call (under 0.5 % of a call) and host wall time per batch, on every rank.
- `run_allreduce_diag.sh`: five groups per stage behind the guard and preflight,
  per-step timeouts, database checksums before and after.
- `allreduce_diag_report.py`: the table below. Group estimate = median of the
  group's three batches (rank 0, CUDA events); diagnostic estimate = median of
  the five group estimates. No observation is discarded.

## Per-row results

Times in µs. Stable = at least 4 of 5 group estimates within ±10 % of their
median. A spike batch is one more than 25 % above the diagnostic estimate.

| TP | bytes | stored | new | change | five group estimates | min | max | MAD | CV | within ±10 % | stable | spike batches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4 KiB | 57.4 | 46.6 | -18.9 % | 46.9, 46.6, 45.0, 50.9, 40.2 | 40.2 | 50.9 | 1.6 | 8.4 % | 4/5 | pass | 1 of 15 (max 80) |
| 2 | 8 KiB | 57.3 | 45.0 | -21.5 % | 36.4, 45.3, 45.0, 59.6, 37.0 | 36.4 | 59.6 | 8.0 | 21.0 % | 2/5 | FAIL | 2 of 15 (max 68) |
| 2 | 16 KiB | 53.0 | 44.6 | -15.9 % | 38.9, 45.4, 44.6, 47.6, 35.3 | 35.3 | 47.6 | 3.0 | 12.0 % | 3/5 | FAIL | 0 of 15 (max 48) |
| 2 | 32 KiB | 55.9 | 44.3 | -20.7 % | 37.7, 45.0, 44.3, 47.3, 34.8 | 34.8 | 47.3 | 3.0 | 12.7 % | 3/5 | FAIL | 1 of 15 (max 100) |
| 2 | 64 KiB | 52.9 | 45.7 | -13.6 % | 42.5, 46.7, 45.7, 47.9, 33.8 | 33.8 | 47.9 | 2.2 | 13.1 % | 4/5 | pass | 1 of 15 (max 116) |
| 2 | 128 KiB | 61.0 | 44.2 | -27.5 % | 40.6, 46.0, 44.2, 48.2, 34.6 | 34.6 | 48.2 | 3.6 | 12.5 % | 4/5 | pass | 1 of 15 (max 127) |
| 2 | 256 KiB | 59.5 | 45.6 | -23.5 % | 38.6, 48.8, 45.6, 50.2, 40.1 | 38.6 | 50.2 | 4.6 | 11.6 % | 2/5 | FAIL | 2 of 15 (max 194) |
| 2 | 512 KiB | 46.2 | 46.8 | +1.1 % | 37.7, 47.7, 46.8, 49.5, 38.3 | 37.7 | 49.5 | 2.8 | 12.6 % | 3/5 | FAIL | 0 of 15 (max 50) |
| 2 | 1 MiB | 49.3 | 44.8 | -9.0 % | 37.5, 47.2, 44.8, 51.9, 40.1 | 37.5 | 51.9 | 4.7 | 12.9 % | 2/5 | FAIL | 0 of 15 (max 52) |
| 2 | 2 MiB | 48.3 | 45.7 | -5.3 % | 36.1, 46.8, 45.7, 52.4, 38.7 | 36.1 | 52.4 | 6.7 | 15.0 % | 2/5 | FAIL | 0 of 15 (max 53) |
| 2 | 4 MiB | 61.8 | 45.0 | -27.2 % | 39.3, 47.5, 45.0, 55.4, 39.6 | 39.3 | 55.4 | 5.4 | 14.6 % | 2/5 | FAIL | 2 of 15 (max 143) |
| 2 | 8 MiB | 54.6 | 45.2 | -17.1 % | 43.8, 47.2, 45.2, 54.7, 43.8 | 43.8 | 54.7 | 1.5 | 9.7 % | 4/5 | pass | 1 of 15 (max 147) |
| 4 | 4 KiB | 51.8 | 44.6 | -13.8 % | 46.2, 44.6, 41.7, 43.5, 46.4 | 41.7 | 46.4 | 1.6 | 4.4 % | 5/5 | pass | 1 of 15 (max 56) |
| 4 | 8 KiB | 46.6 | 43.9 | -5.8 % | 56.1, 43.9, 41.3, 41.6, 45.5 | 41.3 | 56.1 | 2.4 | 13.3 % | 4/5 | pass | 2 of 15 (max 61) |
| 4 | 16 KiB | 52.0 | 43.1 | -17.2 % | 53.2, 43.1, 39.5, 41.3, 45.0 | 39.5 | 53.2 | 2.0 | 12.0 % | 4/5 | pass | 3 of 15 (max 103) |
| 4 | 32 KiB | 42.2 | 44.7 | +6.1 % | 44.7, 42.0, 86.0, 41.8, 45.4 | 41.8 | 86.0 | 2.8 | 36.7 % | 4/5 | pass | 2 of 15 (max 91) |
| 4 | 64 KiB | 49.2 | 44.9 | -8.7 % | 44.9, 46.2, 41.8, 41.3, 45.2 | 41.3 | 46.2 | 1.3 | 5.0 % | 5/5 | pass | 3 of 15 (max 115) |
| 4 | 128 KiB | 41.2 | 44.2 | +7.4 % | 44.2, 54.3, 42.8, 42.9, 106.4 | 42.8 | 106.4 | 1.5 | 47.2 % | 3/5 | FAIL | 3 of 15 (max 189) |
| 4 | 256 KiB | 44.3 | 50.4 | +14.0 % | 88.8, 54.2, 42.1, 41.8, 50.4 | 41.8 | 88.8 | 8.3 | 35.0 % | 2/5 | FAIL | 3 of 15 (max 131) |
| 4 | 512 KiB | 43.3 | 47.0 | +8.6 % | 47.0, 50.8, 42.2, 44.6, 57.6 | 42.2 | 57.6 | 3.8 | 12.4 % | 3/5 | FAIL | 3 of 15 (max 184) |
| 4 | 1 MiB | 46.1 | 45.9 | -0.3 % | 46.8, 45.9, 41.5, 42.8, 49.7 | 41.5 | 49.7 | 3.1 | 7.2 % | 5/5 | pass | 2 of 15 (max 196) |
| 4 | 2 MiB | 44.5 | 45.9 | +3.1 % | 45.9, 52.4, 40.6, 42.8, 70.4 | 40.6 | 70.4 | 5.3 | 23.8 % | 2/5 | FAIL | 5 of 15 (max 171) |
| 4 | 4 MiB | 43.9 | 46.7 | +6.5 % | 46.7, 48.5, 42.5, 42.7, 50.0 | 42.5 | 50.0 | 3.2 | 7.3 % | 5/5 | pass | 0 of 15 (max 56) |
| 4 | 8 MiB | 45.8 | 46.4 | +1.3 % | 46.4, 50.0, 45.4, 45.3, 50.4 | 45.3 | 50.4 | 1.1 | 5.2 % | 5/5 | pass | 3 of 15 (max 121) |

Within one batch the ranks' own event intervals differ by up to 3.9 times,
since a stall on one rank's host stretches that rank's interval most.

## Prediction sensitivity

`timing-predict --no-gpu` on six cases, against the private DB and against an
isolated copy (`profile-diag.db`) in which only the 24 rows carry the
diagnostic estimates. Total µs per iteration; share = all-reduce / total.

| case | bytes per all-reduce | TP1 | TP2 stored | TP2 diagnostic | TP4 stored | TP4 diagnostic |
| --- | --- | --- | --- | --- | --- | --- |
| prefill, 256 tokens | 2 MiB | 4,564 | 6,062 (51 %) | 5,898 (50 %) | 5,059 (56 %) | 5,147 (57 %) |
| prefill, 2048 tokens | 16 MiB | 21,180 | 15,549 (25 %) | 15,549 (25 %) | 12,142 (36 %) | 12,142 (36 %) |
| decode, 1 request | 8 KiB | 3,375 | 6,210 (59 %) | 5,420 (53 %) | 4,987 (60 %) | 4,815 (58 %) |
| decode, 8 requests | 64 KiB | 3,478 | 5,920 (57 %) | 5,458 (54 %) | 5,118 (62 %) | 4,843 (59 %) |
| decode, 64 requests | 512 KiB | 4,169 | 5,665 (52 %) | 5,698 (53 %) | 4,951 (56 %) | 5,190 (58 %) |
| decode, 256 requests | 2 MiB | 7,340 | 7,537 (41 %) | 7,373 (40 %) | 5,758 (50 %) | 5,845 (50 %) |

- **Change from the new estimates:** at most −13 % (TP2, decode of 1 request)
  and +5 % (TP4, decode of 64). The 2048-token prefill does not move because
  its 16 MiB rows are outside this phase.
- **Ranking:** unchanged in every case. TP1 is fastest for small prefill and
  for decode up to 64 requests; TP4 is fastest for the large prefill and for
  decode of 256; TP4 is ahead of TP2 throughout.
- **Uncertainty:** an iteration has 64 all-reduces, so the group-to-group
  range of the floor (about 37 to 52 µs between the lowest and highest
  typical group) moves a TP2 or TP4 iteration by about ±0.5 ms. That is ±9 %
  of a decode iteration of 64 requests and ±8 % of the 256-token prefill.
- **Rankings that the uncertainty can flip:** TP1 against TP2 at decode of 256
  (7,340 against 6,757 to 7,801 over the group range) and TP2 against TP4
  where they are within 0.5 ms. TP1 staying fastest for small decode batches
  is outside the range (lowest TP4 value at 64 requests: 4,883).

## Recommendation

- **Ready:** the 1,854 single-GPU rows and the 14 all-reduce rows from 16 MiB
  up (repeat within about 2 %).
- **Not validated:** the 24 small-message rows, stored or new. The ±10 %
  criterion fails for half of them and cannot be met by more repeats of the
  same kind, because the spread is between process groups.
- **Suggested decision:** replace the 24 stored values with the Phase 2D
  estimates and integrate all 1,892 rows with a documented ±20 % uncertainty
  on the small-message floor. The estimates remove the TP2 bias and the stored
  spikes, and predictions barely depend on which of the two sets is used. This
  needs approval: it replaces private-database rows with values aggregated
  outside the runner.
- **Alternative:** integrate the stored rows as they are, with the same
  caveat. Simpler provenance, but TP2 small-batch decode is then about 0.5 to
  0.8 ms pessimistic.
- **Open question worth a decision:** the simulator charges this host-paced
  floor as kernel time for every all-reduce. Whether a serving engine pays the
  same per-call host cost is a modelling question outside this phase.

## Artifacts

Under `$GB300_ROOT/artifacts/phase2d/`:

- `tp2-run-20261110-170427/`, `tp4-run-*/` (`tp2-latest`, `tp4-latest`):
  `raw/group<g>-rank<r>.json` with every batch time, host timings and the 100
  per-call host gaps per batch; manifests, guards, isolation log, GPU samples.
- `stability.json` (also committed as `phase2d/stability.json`).
- `profile-diag.db`, sha256 `a63d6746…`: the isolated copy for sensitivity.
- `predict/`: presets, cases, outputs; summary committed as
  `phase2d/predict-summary.json`.
