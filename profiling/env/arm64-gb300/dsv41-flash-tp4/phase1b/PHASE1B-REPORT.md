# Phase 1B report — opt-in TP4 sequence simulator (CPU-only)

Branch `phase1b-seqsim`, one commit on top of Phase 1A `a7725255` (`phase1a-costmodel`). Everything in this
phase ran on the CPU of rack 10 tray14 against the Phase 1A compact tables. No GPU run, vLLM launch,
profiling capture, database write, preset change or production integration happened. Phase 0
(`test-ss` ca9f8ed0), Phase 1 planning (cc06d9c2) and Phase 1A (a7725255) artifacts are unchanged; the
Rust `simulator` crate and all other repository sources are untouched
(`git diff a7725255 HEAD -- . ':!profiling/env/arm64-gb300/dsv41-flash-tp4/phase1b'` is empty, asserted by a test).

Inputs (hash-verified at run time, recorded in `out/results.json`):

| input | sha256 (prefix) |
|---|---|
| Phase 1A coefficients `../phase1a/out/coefficients.json` | 78f937c500ff9f20 |
| frozen partition `../phase1a/partition/partition.json` | 288e193f46323f4d |
| fitted sequence parameters `out/params.json` | 8dc79573c1481c7f |
| engine sources (`seqsim/*.py`) | recorded per file in `results.json["code"]` |

## 1. What was built

`seqsim` is a standalone Python package (numpy + pyarrow, same venv as Phase 1A) that simulates a whole
pass iteration by iteration from structural features only (batch size, prefill tokens, KV offsets,
chunk flags, kind of the previous two iterations), using the unchanged Phase 1A `costmodel` components for
every per-iteration cost and new, separately fitted *sequence* parameters for the coupling between
iterations. It is opt-in: nothing imports it, and the Rust simulator does not know it exists. Integration
behind a flag in the Rust L5 worker is proposed for Phase 1C (section 9) and was not started.

### 1.1 Required architecture → implementation

| requirement | implementation (`seqsim/engine.py` unless noted) |
|---|---|
| ordered compute and communication events per rank | each iteration is a per-rank sequence of K+1 compute windows separated by K collectives (Phase 1A `DecodeModel` per-batch tables for decode; `PrefillComputeModel` groups × measured segment share profile for eager/ring/mixed); `IterRecord.t0/t1` hold every segment start/end per rank and draw |
| host arrival and GPU execution as separate timelines | per rank: `host_start`, `fwd_start = host_start + h_pre`, launch pacing, `host_end = last launch + p_post`; GPU: segment starts/ends with `kernel start >= max(GPU free, launch + delta_launch)`; the two are coupled only by launches and the launch queue |
| explicit collective dependencies and rank synchronisation | collective completion = `max_r arrival_r + intrinsic(kind, tokens)`; per-rank wait = completion − own arrival (reported separately from intrinsic in `rank_wait`); tests `test_collective_barrier_rank_skew_not_a_sum`, `test_barrier_completion_is_max_arrival_plus_intrinsic` |
| iteration completion = critical path, not summed rank times | `wall = max_r t1[r, -1] − min_r host_start`; a test asserts that the barrier result is below the rank sum and equals the max-arrival rule; `grep` test asserts no multiplier/scale term in the engine |
| overlap between independent streams where evidenced | compute/communication overlap is the Phase 1A `overlap_cc_ns` (union of busy intervals, measured); unobserved stream assignments are **not** modelled (section 7); `test_wait_is_separate_from_intrinsic_and_overlap_is_union` |
| graph decode vs eager prefill vs mixed | `kind_of(f)` → decode (CUDA graph: one launch block, not launch-paced), graph-kind mixed (class medians), eager mixed and ring/16k chunks (launch-paced per segment with the queue rule); `test_graph_decode_is_not_launch_paced_but_eager_is` |
| KV-dependent sparse-indexer costs only in the validated range | Phase 1A coefficients used as-is; `kv_status` = `kv-extrapolation` beyond kv_max 131,063 (decode) / KVW 98.2 (prefill); results are tabulated by status; `test_kv_range_boundaries_are_flagged_not_silently_extrapolated` |
| stochastic MoE routing uncertainty | stochastic mode applies the Phase 1A corpus MoE band to the MoE share of the decode path; no per-iteration routing is claimed; b=64 (outside the corpus) returns status `extrapolation` without a central value; `test_missing_routing_gives_band_not_point_and_no_per_iteration_claim` |

No global correction multiplier exists anywhere; the per-class residual biases below are reported, not removed.

### 1.2 Sequence rules and parameters (`seqsim/params.py`, fitted on calibration rows only)

| rule | parameter | value | status |
|---|---|---|---|
| host serial order per rank: `host_start_r(i) ≥ host_end_r(i−1) + a_host[r]` | `a_host` | 0.224 / 0.251 / 0.260 / 0.253 ms | measured (after mixed iterations) |
| scheduler one-ahead: `host_start_0(i) ≥ GPU_end(i−2) + eps_sched[kind_i, b_{i−2}]` | `eps_sched` table | decode b=2: 0.50 ms … b=48: 1.74 ms; b=1: 2.39 ms (anomaly, measured); ring 3.81, graph 2.03, eager 2.32 ms | measured, per b with MIN_N=5 (sparse b fall back to the nearest measured b) |
| `host_start_r(i) ≥ host_start_0(i)` | — | — | structural |
| first kernel ≥ host_start + f_idle (GPU idle) | `f_idle` | 0.543 ms | measured (n=4) |
| forward launches begin at host_start + h_pre | `h_pre[kind][r]` | decode 2.5–2.8, eager 5.6–6.0, ring 7.9–8.1 ms | measured |
| host end = last launch + p_post | `p_post[kind][r]` | decode 1.1–1.2, mixed 1.4–1.9 ms | measured |
| launch j completes ≥ GPU completion of launch j−Q | `Q` | 968 launches | measured lower bound (max implied depth on calibration ring rank-rows; CUDA nominal 1024) |
| launch → kernel start | `delta_launch` | 20 µs | inferred, second order |
| per-launch host cost in mixed iterations | Phase 1A `host_path()["fwd_ns"]/L` per rank | — | Phase 1A (c_kv = 0 kept, section 7) |
| segment compute share per rank (eager/ring) | `profile[kind]` | medians + quantiles | measured (ring: windows with host ≥ 20 ms ahead) |
| wall dispersion for bands | `disp_q[decode][b]`, `disp_q[ring]` | quantiles of wall/reference | measured, mean-1 factor; stochastic mode only |

Full parameter tables are in `out/RESULTS.md` (section "Sequence parameters").

### 1.3 Evaluation modes

| mode | what is measured, what is modelled | purpose |
|---|---|---|
| recon-gpu | measured segments, measured host times | engine bookkeeping check (should be ≈ exact) |
| recon-seq | measured segments and host durations; host *starts* from the sequence rules | validates the inter-iteration rules in isolation |
| pred-det | everything predicted, median path, 1 draw | required "fully predicted iteration latency" |
| pred-mc | everything predicted, 32 stochastic draws (per-(rank,segment) quantile draws, MoE band, dispersion) | bands and coverage; mean used for point error |

Excluded (JIT-affected) iterations of each pass are replayed from their measured timelines so that the sequence stays aligned; they are not scored. The first iteration of a pass is anchored at its measured host start.

## 2. CPU tests

`tests/test_seqsim.py`: 18 deterministic tests on synthetic data, 18 passed (`out/logs/pytest_seqsim.log`, 0.27 s).

| required test | test(s) |
|---|---|
| collective barriers and rank skew | `test_collective_barrier_rank_skew_not_a_sum`, `test_barrier_completion_is_max_arrival_plus_intrinsic` (incl. tail carry-over) |
| compute/communication overlap | `test_wait_is_separate_from_intrinsic_and_overlap_is_union` |
| host lead and synchronisation | `test_decode_steady_state_is_gpu_bound_with_scheduler_lead`, `test_host_bound_eager_mixed_is_paced_by_the_slowest_host_rank`, `test_lead_accumulates_along_a_chunk_chain_and_saturates_at_the_queue_depth`, `test_scheduler_constraint_binds_after_decode_and_serial_after_mixed` |
| CUDA Graph vs eager | `test_graph_decode_is_not_launch_paced_but_eager_is` |
| KV-range boundaries | `test_kv_range_boundaries_are_flagged_not_silently_extrapolated` |
| missing routing information | `test_missing_routing_gives_band_not_point_and_no_per_iteration_claim` |
| stochastic reproducibility | `test_stochastic_runs_are_reproducible_and_seed_sensitive`, `test_params_roundtrip_is_exact` |
| holdout leakage | `test_holdout_leakage_is_refused_by_the_frozen_partition`, `test_seq_params_fit_uses_only_calibration_rows` |
| preservation of existing simulator predictions | `test_phase1a_and_simulator_sources_untouched` (git diff vs a7725255 outside `phase1b/` is empty), `test_phase1a_coefficients_hash_recorded_and_unchanged` |
| other | `test_no_global_multiplier_in_engine_source`, `test_queue_log_interpolates_launch_completion` |

## 3. Model-versus-trace results

Metrics per class: n, signed bias (mean error, ms and % of the measured mean), median error, median
absolute error, p90 absolute percentage error, worst absolute error, measured median. Positive = model
slower than trace. Classes follow the frozen partition; roles: calibration (used for all fits) and
heldout (never used for any fit). All tables are in `out/RESULTS.md`; per-iteration rows are in
`out/iterations_<pass>_<mode>.parquet` (15 MB; not committed because `*.parquet` is git-ignored repository-wide,
same as the Phase 1A tables; regenerable with the command in `README.md`, copies kept on tray14).

### 3.1 Measured-segment critical-path reconstruction (run2)

recon-gpu, held-out rows (bookkeeping check):

| class | n | bias ms | median |err| ms | p90 APE % | worst ms |
|---|---|---|---|---|---|
| decode b1-2 | 471 | +0.020 | 0.020 | 0.55 | 0.03 |
| decode b3-16 | 6935 | +0.034 | 0.034 | 0.67 | 0.05 |
| prefill 16k chunk | 33 | −0.149 | 0.029 | 0.04 | 3.44 |
| mixed-eager (host-bound) | 3 | −0.307 | 0.002 | 0.37 | 0.92 |
| mixed-ring (other) | 4 | +0.076 | 0.063 | 0.07 | 0.17 |

The 0.02–0.03 ms decode offset is the host-side tail of the last segment (the pass-level "wall" includes the
sampler launch that the segment tables do not). Span error +0.36 %, Σ walls +0.43 %.

recon-seq (host starts from the sequence rules, everything else measured), held-out: decode unchanged;
16k chunks +0.38 ms (p90 APE 0.05 %, worst 8.0 ms), eager +2.8 ms (p90 APE 2.0 %), ring +1.3 ms. Span +0.69 %,
Σ walls +0.58 %. The sequence rules alone therefore explain the run2 timeline to < 1 %.

### 3.2 Fully predicted iteration latency, run2 (17,637 scored of 17,686 iterations)

pred-det (median path):

| role | class | n | bias ms | bias % | median |err| ms | p90 APE % | worst ms | meas. median ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | 1886 | +0.012 | +0.27 | 0.279 | 15.9 | 31.0 | 4.43 |
| calibration | decode b3-16 | 8226 | −0.152 | −2.70 | 0.136 | 4.2 | 33.2 | 5.24 |
| calibration | prefill 16k chunk | 55 | +0.572 | +0.25 | 0.686 | 1.5 | 8.6 | 227.2 |
| calibration | mixed-eager (host-bound) | 14 | +1.63 | +0.85 | 7.55 | 11.9 | 22.9 | 198.5 |
| calibration | mixed-ring (other) | 10 | +4.58 | +2.18 | 3.72 | 7.2 | 32.8 | 221.4 |
| heldout | decode b1-2 | 471 | +0.022 | +0.48 | 0.271 | 15.2 | 19.8 | 4.41 |
| heldout | decode b3-16 | 6935 | −0.167 | −2.83 | 0.153 | 4.7 | 14.7 | 6.39 |
| heldout | prefill 16k chunk | 33 | −0.019 | −0.01 | 0.458 | 0.56 | 10.9 | 232.2 |
| heldout | mixed-eager (host-bound) | 3 | −3.46 | −1.69 | 5.29 | 5.8 | 14.2 | 202.6 |
| heldout | mixed-ring (other) | 4 | −0.81 | −0.36 | 4.39 | 2.7 | 6.1 | 227.1 |

pred-mc (mean of 32 draws):

| role | class | n | bias ms | bias % | median |err| ms | p90 APE % | worst ms |
|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | 1886 | +0.081 | +1.76 | 0.286 | 18.0 | 31.1 |
| calibration | decode b3-16 | 8226 | +0.039 | +0.70 | 0.063 | 2.4 | 33.0 |
| calibration | prefill 16k chunk | 55 | +0.629 | +0.28 | 1.180 | 1.4 | 8.9 |
| calibration | mixed-eager (host-bound) | 14 | +1.44 | +0.75 | 7.46 | 12.1 | 22.5 |
| calibration | mixed-ring (other) | 10 | +4.62 | +2.20 | 4.02 | 7.1 | 32.5 |
| heldout | decode b1-2 | 471 | +0.094 | +2.05 | 0.270 | 17.5 | 19.1 |
| heldout | decode b3-16 | 6935 | +0.031 | +0.52 | 0.063 | 2.7 | 14.4 |
| heldout | prefill 16k chunk | 33 | +0.047 | +0.02 | 0.649 | 0.75 | 6.6 |
| heldout | mixed-eager (host-bound) | 3 | −3.60 | −1.76 | 5.29 | 5.9 | 14.4 |
| heldout | mixed-ring (other) | 4 | −0.63 | −0.28 | 4.18 | 2.6 | 6.0 |

The decode b3-16 worst cases (33 ms calibration, 14.7 ms held-out) are single iterations whose measured wall
contains a stall that no component predicts; they are in the tables, not filtered.

### 3.3 Decode batch buckets 17–24 and ≥ 25 (quadrant pass; run2 has no b > 16)

| mode | role | class | n | bias ms | bias % | median |err| ms | p90 APE % | worst ms | meas. median ms |
|---|---|---|---|---|---|---|---|---|---|
| pred-det | heldout | decode b1-2 | 53 | −0.054 | −1.10 | 0.375 | 16.2 | 3.0 | 4.83 |
| pred-det | heldout | decode b3-16 | 78 | −0.063 | −1.18 | 0.068 | 2.3 | 0.22 | 5.04 |
| pred-det | heldout | decode b17-24 | 49 | −0.082 | −1.06 | 0.074 | 2.5 | 0.31 | 7.74 |
| pred-det | heldout | decode b>=25 | 1071 | −0.084 | −0.89 | 0.101 | 2.4 | 7.3 | 9.75 |
| pred-mc | heldout | decode b1-2 | 53 | +0.070 | +1.44 | 0.235 | 13.2 | 2.7 | 4.83 |
| pred-mc | heldout | decode b3-16 | 78 | +0.133 | +2.49 | 0.124 | 3.6 | 0.64 | 5.04 |
| pred-mc | heldout | decode b17-24 | 49 | +0.104 | +1.34 | 0.119 | 2.6 | 0.27 | 7.74 |
| pred-mc | heldout | decode b>=25 | 1071 | +0.205 | +2.19 | 0.231 | 4.5 | 7.1 | 9.75 |

Calibration rows of the quadrant pass behave the same way (pred-det −0.9 to −1.3 %, pred-mc +1.4 to +2.2 %,
b1-2 +4.2 %). The median path under-predicts by ≈ 1 % and the stochastic mean over-predicts by ≈ 2 %
on this pass: the per-(rank, segment) quantile draws composed through the max-over-ranks barrier shift the
mean upward relative to the measured wall distribution. This is a property of the uncertainty model, not of
the median path, and was left as is (no tuning against the quadrant pass; its calibration rows were used for
the Phase 1A tables only).

### 3.4 Long-context prefill (16k chunks by KV offset, run2, pred-det)

| role | kv0 | status | n | bias ms | median |err| ms | p90 APE % | worst ms | meas. median ms |
|---|---|---|---|---|---|---|---|---|
| calibration | 0k | in-range (first chunk after decode) | 20 | +1.50 | 3.13 | 2.2 | 8.6 | 225.7 |
| calibration | 16k | inferred | 14 | +0.03 | 0.44 | 0.50 | 4.0 | 222.9 |
| calibration | 32k | inferred | 8 | −0.31 | 0.41 | 0.24 | 0.69 | 228.5 |
| calibration | 64k | inferred | 6 | +0.18 | 0.22 | 0.28 | 0.94 | 237.7 |
| calibration | 96k | inferred | 2 | +1.97 | 1.97 | 0.95 | 2.4 | 245.8 |
| heldout | 0k | in-range | 3 | −3.75 | 0.23 | 3.8 | 10.9 | 229.7 |
| heldout | 16k | inferred | 4 | +0.44 | 0.70 | 0.46 | 1.0 | 222.7 |
| heldout | 48k | inferred | 11 | +0.47 | 0.72 | 0.49 | 1.3 | 232.2 |
| heldout | 64k | inferred | 4 | +0.31 | 0.27 | 0.25 | 0.69 | 237.6 |
| heldout | 80k | inferred | 3 | −0.43 | 0.46 | 0.26 | 0.67 | 243.5 |
| heldout | 96k | inferred + extrapolation | 2 | +2.5 / +2.6 | 2.6 | 1.0 | 2.6 | 245.2 |
| heldout | 112k | extrapolation | 1 | −0.36 | 0.36 | 0.14 | 0.36 | 253.1 |

Chain chunks (kv0 ≥ 16k) are within ≈ 0.5 % along the whole chain to 112k; the first chunk after a decode
burst (kv0 = 0, status in-range) is the weakest chunk class (p90 APE 2–4 %, both signs). The salvaged pass
shows the same pattern (chain chunks p90 APE 0.2–0.9 %, kv0 = 0 held-out −7.0 ms on n=2).

### 3.5 Host-bound eager mixed iterations

run2: calibration n=14 (+1.6 ms bias, median |err| 7.5 ms, p90 APE 11.9 %, worst 22.9 ms), held-out n=3
(−3.5 ms, p90 APE 5.8 %). quadrant: calibration n=3 (+20.7 ms, p90 APE 36.9 %, worst 63.9 ms), held-out n=1
(+2.5 ms). salvaged: n=1+1 (−8.5 / −11.5 ms). This class stays **unresolved** (section 7): the errors are
dominated by host pacing variation that the trace does not identify (c_kv, rank-2 lateness, JIT residue), and
the sample is too small for a band. It is listed first in the Phase 1C capture plan.

### 3.6 Full-workload latency and throughput

Span = last GPU end − first host start of the scored sequence (excluded iterations replayed). Σ walls = sum of
scored iteration walls. "abs error" is |Σ predicted − Σ measured|; "Σ|err|" sums per-iteration absolute errors.

| pass | mode | span meas. s | span pred. s | span err | Σ walls all signed | Σ walls abs err s | Σ|err| s | Σ walls held-out signed | held-out abs err s |
|---|---|---|---|---|---|---|---|---|---|
| run2 | recon-gpu | 134.465 | 134.95 | +0.36 % | +0.43 % | 0.53 | — | — | — |
| run2 | recon-seq | 134.465 | 135.39 | +0.69 % | +0.58 % | 0.72 | — | — | — |
| run2 | pred-det | 134.465 | 132.386 | −1.55 % | −1.84 % | 2.288 | 3.833 | −2.22 % | 1.164 |
| run2 | pred-mc | 134.465 | 135.477 | +0.75 % | +0.66 % | 0.821 | 2.685 | +0.47 % | 0.246 |
| quadrant | pred-det | 49.839 | 49.212 | −1.26 % | −0.67 % | 0.296 | 0.821 | −0.82 % | 0.094 |
| quadrant | pred-mc | 49.839 | 50.448 | +1.22 % | +2.13 % | 0.946 | 1.202 | +2.13 % | 0.245 |
| salvaged | pred-det | 62.176 | 61.691 | −0.78 % | −1.77 % | 0.800 | 0.981 | — | — |
| salvaged | pred-mc | 62.176 | 62.630 | +0.73 % | +0.31 % | 0.142 | 0.645 | — | — |

(recon rows: span and Σ values from `out/RESULTS.md`; Σ|err| not meaningful there.)

Throughput over the scored sequence (tokens / span):

| pass | measured decode tok/s | pred-det | pred-mc | measured prefill tok/s | pred-det | pred-mc |
|---|---|---|---|---|---|---|
| run2 | 964.0 | 979.1 | 956.8 | 14,398 | 14,625 | 14,291 |
| quadrant | 3,555.6 | 3,601.0 | 3,512.7 | 3,575.0 | 3,620.6 | 3,531.8 |
| salvaged | 768.6 | 774.6 | 763.0 | 20,081 | 20,239 | 19,935 |

The held-out full-workload numbers were looked at once, after the last engine change; no parameter was
changed in response to them. The two engine fixes made during development (queue-log aliasing; host pacing
with the Phase 1A `fwd_ns` instead of c_host × all launches) were motivated by calibration-row chain behaviour
and by a unit test; `out/RESULTS.md` is from the final code and the pre-fix outputs are not part of the commit.

### 3.7 Host lead and host skew (run2, pred-mc)

Lead = GPU end of iteration i−1 minus the earliest host start of iteration i (positive: host runs ahead).

| transition (prev → this) | role | n | meas. lead median ms | pred. lead median ms | lead median |err| ms | lead p90 |err| ms | meas. host skew ms | pred. skew ms |
|---|---|---|---|---|---|---|---|---|
| decode → decode | heldout | 7373 | 5.60 | 5.73 | 0.07 | 0.21 | 0.011 | 0.000 |
| decode → ring | heldout | 3 | 1.87 | 1.43 | 1.97 | 4.66 | 0.55 | 0.000 |
| mixed → decode | heldout | 33 | 18.2 | 16.2 | 2.6 | 17.7 | 18.6 | 14.2 |
| mixed → ring (chains) | calibration | 44 | 73.7 | 84.0 | 11.9 | 22.9 | 22.3 | 28.4 |
| mixed → ring (chains) | heldout | 34 | 96.3 | 99.2 | 18.9 | 23.7 | 31.6 | 42.7 |
| mixed → eager | heldout | 1 | 17.9 | 10.5 | 7.4 | 7.4 | 18.4 | 9.4 |

The decode-phase lead (≈ 5–6 ms, the next iteration's host work hidden behind the current GPU work) is
reproduced to 0.07 ms. The chain leads of 70–130 ms and their saturation are reproduced to ≈ 10 % with the
measured queue-depth bound; per-chunk errors of 12–19 ms remain because the per-rank launch pacing is taken
from Phase 1A medians (the trace shows ±20 ms rank-to-rank variation that the model does not identify).
Host skew after decode is predicted as exactly 0 (measured 0.01 ms); after mixed iterations the per-rank
serial rule reproduces the 15–45 ms skews within ≈ 5–10 ms.

## 4. Uncertainty coverage (pred-mc, p10–p90 and p05–p95 bands of 32 draws)

| pass | role | class | n | cover p10–p90 % | cover p05–p95 % | band p10–p90 median ms | band rel % | below band % | above band % |
|---|---|---|---|---|---|---|---|---|---|
| run2 | heldout | decode b1-2 | 471 | 84.9 | 93.6 | 1.26 | 27.5 | 8.9 | 6.2 |
| run2 | heldout | decode b3-16 | 6935 | 57.8 | 67.5 | 0.16 | 2.8 | 30.4 | 11.8 |
| run2 | heldout | prefill 16k chunk | 33 | 100 | 100 | 8.2 | 3.6 | 0 | 0 |
| run2 | heldout | mixed-eager (host-bound) | 3 | 0 | 0 | 0.14 | 0.07 | 33 | 67 |
| run2 | heldout | mixed-ring (other) | 4 | 50 | 50 | 6.1 | 2.6 | 25 | 25 |
| run2 | calibration | decode b1-2 | 1886 | 85.6 | 93.4 | 1.25 | 27.8 | 8.0 | 6.4 |
| run2 | calibration | decode b3-16 | 8226 | 62.0 | 72.5 | 0.17 | 3.0 | 34.8 | 3.2 |
| run2 | calibration | prefill 16k chunk | 55 | 89.1 | 92.7 | 7.3 | 3.2 | 9.1 | 1.8 |
| quadrant | heldout | decode b1-2 | 53 | 96.2 | 96.2 | 1.58 | 32.9 | 1.9 | 1.9 |
| quadrant | heldout | decode b3-16 | 78 | 17.9 | 30.8 | 0.19 | 3.6 | 82.1 | 0 |
| quadrant | heldout | decode b17-24 | 49 | 51.0 | 61.2 | 0.24 | 3.2 | 46.9 | 2.0 |
| quadrant | heldout | decode b>=25 | 1071 | 38.5 | 48.0 | 0.37 | 3.8 | 59.4 | 2.1 |

Reading: the 16k-chunk band (≈ 3.5 % wide) is well calibrated. The decode b3-16 band on run2 is too narrow
(58–62 % at nominal 80 %) and on the quadrant pass the band sits above the measurement (18–51 % coverage,
measured walls below the band in 47–82 % of cases), consistent with the +2 % stochastic-mean bias of
section 3.3. The b1-2 band is wide (27–33 % of the wall) because the Phase 1A per-batch quantiles include the
rank-lateness pattern; it over-covers (85–96 %). Eager mixed has no usable band (n ≤ 14, no dispersion
reference; width 0.1 %). None of these were tuned; a wider decode band would require a dispersion model that
the current trace does not identify separately from the per-segment quantiles.

## 5. Regression evidence

| check | result | log |
|---|---|---|
| `seqsim` unit tests | 18 passed | `out/logs/pytest_seqsim.log` |
| sensitivity of inferred parameters (run2 pred-det) | see section 7 | `out/logs/sensitivity_run2.log` |
| repository CPU pytest tier (`pytest -m "not gpu and not agent and not bench and not presets"`, gb300 venv, tray14) | 5,063 passed, 50 skipped, 26 deselected, **2 failed** | `out/logs/pytest_cpu_tier.log` (summary lines) |
| the 2 failures re-run | `test_launcher_corpus::test_emit_backends_hands_the_builder_a_path_not_a_reference`: `FileNotFoundError: 'cargo'` in the non-login ssh shell; **passes** with `~/.cargo/bin` on PATH. `test_gdn_spec_decode::test_recurrent_verify_matches_sequential_decode_and_writes_every_checkpoint`: `torch.equal` exactness assertion on the aarch64 CPU torch build; fails identically with the `phase1b/` directory removed from the tree (a7725255 unchanged), so it is pre-existing and unrelated | `out/logs/pytest_cpu_tier_failures.log`, `out/logs/pytest_cpu_tier_launcher_rerun.log` |
| Rust `cargo test --workspace --exclude analyzer` | not re-run: no Rust source changed (the diff outside `phase1b/` is empty, asserted by a test); Phase 1A's run on the same tree gave 1,606 passed | Phase 1A report |
| Phase 1A tests | unchanged sources; `tests/test_costmodel.py` not re-run in this phase (hash of coefficients asserted) | — |
| goldens `tests/golden/**` | untouched (empty diff) | — |

Existing simulator predictions are preserved by construction: `seqsim` is not imported by any repository module.

## 6. What the trace does and does not support (evidence boundaries)

- Overlap between **independent GPU streams**: the nsys kernel table has stream ids, but the Phase 0 extraction
  measures busy unions only; the model uses the measured compute/communication union per segment and does not
  assign work to streams. Any stream-level overlap not already in the union is not modelled.
- **Queue depth**: 968 is the largest implied depth on calibration ring rows (58 saturated rows, p50 806,
  p90 958); the hard cap is not observed directly. Lead saturation is reproduced, the exact cap is a lower bound.
- **Per-segment launch distribution** is assumed uniform in segment index (the API trace does not attribute
  launches to layers in the compact tables).
- **Decode b = 1** scheduler gap (2.39 ms vs 0.50 ms at b = 2) is reproduced from the table but not explained.

## 7. Unresolved cases (kept explicit, no precise values inferred)

| case | how it is carried | evidence status |
|---|---|---|
| rank-2 host lateness | Phase 1A per-rank `c_host` (rank 2 slower) is used as measured; no cause is modelled; the ±20 ms per-rank pacing variation along chains is the main lead error (section 3.7) | not identifiable from the trace (gap G4) |
| host-bound eager mixed latency | predicted from Phase 1A host path with c_kv = 0; p90 APE 6–37 % depending on pass, n = 1–14; no band | unresolved; Phase 1C capture 1 |
| batch 1–2 rank-arrival variation | Phase 1A per-(rank, segment) quantiles reproduce the width (band 27–33 %), the median path is unbiased on run2 (+0.3/+0.5 %) but the pattern differs between passes (quadrant calibration +4 % in pred-mc) | unresolved; Phase 1C capture 2 |
| per-iteration MoE routing | corpus band on the MoE share only; no per-iteration claim; b = 64 outside the corpus flagged | unresolved; Phase 1C capture 4 |
| unobserved stream assignments | union-based overlap only (section 6) | not observable in the compact tables |
| KV-dependent host-launch cost (c_kv) | 0 kept from Phase 1A; the KV16 launch-count growth is applied, its per-launch cost is not | not identifiable (Phase 1A gap); Phase 1C capture 1 |

Parameters marked *inferred* (delta_launch 20 µs, uniform launch distribution) are second order, checked with
`tools/sensitivity.py` on run2 pred-det (`out/logs/sensitivity_run2.log`): disabling the queue throttle moves
the chain-lead medians by +14 ms (calibration) / +48 ms (held-out) but the span by −0.02 s (−0.017 %) and no
class bias by more than 1.5 ms (eager calibration); delta_launch at 0 or 40 µs moves every class bias by
≤ 0.014 ms and the span by ≤ 0.01 s.

## 8. Honest assessment

What works: the sequence rules reconstruct the run2 timeline to < 1 % when per-iteration costs are measured
(recon-seq), and the fully predicted run2 workload is within 0.75 % (stochastic mean) / −1.55 % (median path)
in span and within 0.5 % / 2.2 % in held-out Σ walls; chain chunks are predicted to ≈ 0.5 %; decode b3-16
held-out p90 APE is 2.7 % (pred-mc) / 4.7 % (pred-det); b17-24 and b ≥ 25 held-out p90 APE 2.4–4.5 %.

What does not yet: host-bound eager mixed (n small, 6–37 % p90 APE), the first 16k chunk after decode
(2–4 % p90 APE, both signs), decode b1-2 (p90 APE 13–18 %), and the uncertainty bands for decode, which are
either too narrow (run2 b3-16) or shifted high (quadrant). The ≈ 3 % spread between median-path and
stochastic-mean decode predictions brackets the measurement on every pass; it is a modelling-choice
uncertainty that a routing capture (Phase 1C) would be expected to reduce.

## 9. Proposed Phase 1C

See `PHASE1C-PLAN.md`: CPU-only invariant and interface work first (no new data), then six prioritised GPU
captures on one rack-10 tray (host-bound eager at long KV; decode b 1–4 at kv < 32k; 10+ chunk chains;
MoE routing hook; graph-path mixed sweep; cold-start chunks), each added to a *new* frozen partition as
held-out only, with the acceptance criteria listed there. Production integration (opt-in flag in the Rust L5
worker, bit-identical predictions with the flag off) is proposed for after the captures, not before.

**Nothing beyond this review is authorised. No GPU experiment or production integration is started.**
