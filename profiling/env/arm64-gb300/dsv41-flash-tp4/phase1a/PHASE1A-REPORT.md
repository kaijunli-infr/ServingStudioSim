# Phase 1A report — CPU-only cost model for DSV41-flash on GB300 TP4

Date: 2026-11-13. Scope as approved: CPU-only implementation and offline calibration on the frozen
Phase 0 traces. No GPU execution, no vLLM launch, no new capture, no production database write, no
public preset change, no serving rerun was performed. This report requests review before Phase 1B.

## 1. Architecture and changed files

Everything is additive under `profiling/env/arm64-gb300/dsv41-flash-tp4/phase1a/` (new directory).
No file outside it was modified; the Rust simulator crates, the Python parsers, presets and
profiling databases are untouched. Module map (details in `README.md`):

| module (approval item) | file | role |
|---|---|---|
| 1 decode GPU execution | `costmodel/decode.py` | per-regime (b≤16 / 17–24 / ≥25), per-batch tables of per-rank ordered compute windows and per-call collective intrinsic; stochastic 4-rank composition |
| 2 mixed eager host/GPU | `costmodel/mixed.py` | host/GPU timelines from launch & sync events: CPU prep, CUDA API, launch gaps, GPU exec, sync, overlap; launch-count model; per-rank host cost; lead-aware wall |
| 3 MoE expert GEMM | `costmodel/moe.py` | corpus routing statistics → distinct-experts-per-layer E(b) → statistical kernel-time model with central estimate and uncertainty |
| 4 long-context indexer / sparse-meta | `costmodel/longctx.py` | per-group per-rank compute vs (T, Σt², KV-weighted tokens, b); dedicated 16k-chunk linear-in-KV fit |
| 5 TP4 collectives | `costmodel/collectives.py`, `costmodel/compose.py` | intrinsic cost vs payload per kind; explicit collective synchronisation; critical path over ranks |
| 6 per-rank stream / sync composition | `costmodel/compose.py`, `costmodel/intervals.py` | union / overlap accounting (compute–compute, compute–comm, idle), max-over-ranks arrival + intrinsic |
| 7 host scheduling / sync | `costmodel/host.py`, `costmodel/mixed.py` | inter-iteration gaps, decode host evidence, launch back-pressure, host lead between iterations |
| classification | `calibrate.py` → `coefficients.json["classification"]` | every parameter tagged measured / inferred / not identifiable / assumed |
| data path | `extract.py`, `data.py`, `partition.py`, `groups.py` | hash-verified extraction from frozen captures, frozen partition with leakage guard |
| evaluation | `evaluate.py`, `predict.py` | all metrics in §5–§7, `out/results.json`, `out/RESULTS.md` |
| tests | `tests/test_costmodel.py` | 15 CPU tests (§9) |

Design rules honoured throughout: no global latency multiplier (asserted by construction: wall is a
critical-path composition of measured/fitted components, never scaled); overlap is accounted
explicitly as a union, never as a sum; rank durations are never summed (Σ over ranks / wall ≈ 3.9–4.0
in the composition check); collective waiting is derived from per-rank arrival times, not from a
multiplier; every prediction carries a status (`measured`, `interpolated`, `in-range`, `inferred`,
`extrapolation`, `measured-class`) and decode carries a separate `kv_status`.

## 2. Exact revisions

| item | value |
|---|---|
| planning base | branch `phase1-planning` @ `cc06d9c2` (Phase 1 plan; recorded in `coefficients.json["git_rev"]`) |
| implementation branch | `phase1a-costmodel`, branched from `cc06d9c2`; commit hash: see §12 |
| preserved, verified untouched | `test-ss` = `ca9f8ed0`, `phase1-planning` = `cc06d9c2`, `kevin/dsv41-gb300-main` = `58f5d779` |
| frozen Phase 0 artifact | `…/artifacts/phase0-dsv41-flash-gb300-tp4-20261112/` (rack 10 CephFS); all inputs verified against its `SHA256SUMS` before extraction (`provenance.tables.*.frozen_inputs.verified_against_SHA256SUMS = true`) |
| captures used | `run/diverse_nsys_run2` (parsed.json 54cf7c2d…, kernels d7a6d5f1…, host_timeline d6d97178…), `run/quadrant_nsys` (6869a786…, dcde40f0…, 97cc09cb…), `run/diverse_nsys` = label `diverse_nsys_salvaged` (cross-check only), `run/diverse_corpus/profile/token_corpus` (routes.u16 fa87a979…, 135,144 tokens × 40 layers × top-6 of 384) |
| compact tables | `/mnt/shared/home/dc/workspace/serving-studio/phase1a-tables/<label>/`; sha256 in `coefficients.json["provenance"]["tables"]` (run2: iterations bd4c2284…, iter_rank 518da161…, collectives 750086b1…; quadrant: 4f2612ab…, b2e45661…, aeab5af5…) |
| partition | `partition/partition.json` sha256 `288e193f46323f4d4361913a08c47afa9c4d9f98cc4c4be0d5b88f9629304b7a`, built from the planning outputs `phase1/out/validation_split.json` (c2376e92…) and `routing_stats.json` (c55a3194…) |
| coefficients | `out/coefficients.json` sha256 `78f937c500ff9f205db425a1573789d182df22cf67e8b84de39b22a7ae01c1c9` (`out/COEFFICIENTS.sha256`); seed 0; created 2026-11-13T02:20:39Z |
| code hashes | every `costmodel/*.py` sha256 is in `coefficients.json["provenance"]["code"]` and in each table's `meta.json` |

## 3. Frozen partition

Roles were copied verbatim from the Phase 1 planning split (`validation_split.json`) and frozen
(`PARTITION.sha256`) **before** any fitting; `Partition.calibration_only()` raises `LeakageError`
if a held-out id reaches a fit, and `load()` refuses a tampered file (tested). The partition was
not revised after inspecting errors. One prediction-side rule was added before looking at held-out
errors in detail: a per-batch decode table entry is used only if it rests on ≥ 5 calibration
iterations (`MIN_TABLE_N`), otherwise the neighbouring calibrated b are interpolated and the
prediction is tagged `interpolated` / `extrapolation`.

| pass | role | iterations | notes |
|---|---|---|---|
| diverse_nsys_run2 | calibration | 10,191 (10,112 decode, 55 16k chunks, 10 ring, 14 eager) | 49 JIT iterations excluded (10.09 s) |
| diverse_nsys_run2 | held-out | 7,446 (7,406 decode, 33 chunks, 4 ring, 3 eager) | chunk hold-out = whole chains 3, 9, 12, 21 plus KV bands |
| quadrant_nsys | calibration | 3,487 (3,475 decode, 4 graph, 5 ring, 3 eager) | 10 JIT excluded |
| quadrant_nsys | held-out | 1,253 (1,251 decode, 1 ring, 1 eager) | |
| diverse_nsys_salvaged | cross-check only | 4,667 | never used for calibration (its own roles are ignored) |
| MoE corpus | calibration 49 requests / 110,691 tokens; held-out 10 requests / 23,578 tokens | | held-out = 5 least + 5 most imbalanced requests with ≥ 100 tokens; 39 short requests excluded |

## 4. Coefficients and provenance (central values; full tables in `out/coefficients.json`)

**Decode (module 1, measured).** 35 calibrated batch sizes (b = 1…48, kv_max ≤ 131,063); per b and
per rank: first-kernel offset, K+1 compute-window lengths (K = 163 for b≤16: mhc×154 + oneshot×4 +
allgather×5; K = 86 for b≥17: oneshot or twoshot×81 + allgather×5), empirical 10-level quantiles of
each window, per-call intrinsic (min over ranks), intra-rank idle and stream overlap. Composition is
Monte-Carlo over the per-(rank, segment) quantiles (256 draws, fixed seed) because the deterministic
median path under-predicts the expected max over 4 ranks by ≈ 3.4 %. Collective intrinsics: mhc
2.5–2.8 µs, oneshot 4.0–4.4 µs, twoshot 5.6–5.8 µs, all-gather 11–22 µs per call.

**Collectives (module 5, measured).** Intrinsic per call (µs) vs payload tokens P+b:
twoshot 3.06 + 0.0253·tok (174–7,216 tok, n = 20, rmse 0.86 µs); NCCL all-reduce 23.1 + 0.0243·tok
(8,271–16,384 tok, n = 70); NCCL all-gather 28.8 + 0.0319·tok (n = 90). Mixed kind by payload:
graph (P+b ≤ 256), eager two-shot (< 8,192 total tokens), NCCL ring (≥ 8,192).

**Prefill compute (module 4, measured).** Per group g and rank r:
y_g,r = a + c_T·T + c_Q·Σt²/16384 + c_K·KVW + c_b·b with KVW = Σ t_i·kv0_i / 16384 / 1000, fitted
on 87 calibration mixed iterations, ranges T 306–16,383, KVW 0–98.2, b 1–47. Slowest-rank values:
indexer 10.84 ms + 0.2429 ms per 1k KV at 16k tokens; fmha 2.97 ms/1k tok (KV slope −0.01, i.e. none);
sparse_meta KV slope +0.003 (none); moe 9.67 + 2.36/1k tok; other 5.57 + 4.23/1k tok.
Dedicated 16k-chunk fit (55 chunks, kv0 0–98,262): wall = 223.7 + 0.2001·kv0_k ms (rmse 3.64),
slowest-rank busy = 219.9 + 0.2462·kv0_k, indexer = 10.84 + 0.2429·kv0_k (rmse 0.53).

**MoE (module 3, inferred from measured E(b)).** Corpus E(b) = mean distinct experts per layer
for a padded batch of b tokens (computed on the corpus pass, which is a different pass from the
timing passes). t_moe(b) = 0.878 + 0.02107·E(b_pad) + 0.0340·b_pad ms, R² 0.984, rmse 0.166 ms,
with b_pad the CUDA-graph padded batch (b≤2→b, ≤4→4, else ⌈b/8⌉·8). The raw-b alternative
(R² 0.962) is stored for comparison. Uncertainty band = bootstrap over corpus requests.

**Mixed eager host path (module 2/7).** Launch count L = 1503.1 + 0.0061·P + 7.33·(chunks−1) +
82.8·KV16 (KV16 = Σ t_i·kv0_i / 16384², rmse 12.3 launches, n = 87, rank-independent, measured).
Per-rank host cost per launch c_r = [121.0, 119.1, 129.8, 125.5] µs (measured on 14 host-bound
eager calibration iterations; rank 2 is the slowest host in 12 of 14), unthrottled CUDA launch API
13.5 µs/launch (measured; it inflates to > 30 µs under back-pressure in GPU-bound iterations), GPU
tail after last launch 2.87 ms (measured), host time outside the forward window 7.0–7.5 ms per rank.
c_kv (host cost of KV-dependent launches) is **inferred** and clamped to 0: its de-confounded
estimate is negative (median −43 µs, p10/p90 −205/+39 µs per KV16 unit), i.e. not identifiable from
these traces. Graph-path mixed iterations: class medians only (2 (P,b) classes, `measured-class`).

**Host lead (module 7, measured per iteration in evaluation, not yet simulated).** The host of
iteration i starts before the GPU of iteration i−1 finishes: median lead 68 ms (max 130) for ring
after mixed, 46 ms (max 128) for eager after mixed, 2.6–5.2 ms after decode. wall = max(GPU path,
host path − lead). Inter-iteration GPU gap is ≈ 0 (median −5 µs decode, +3 µs ring): the GPU is
saturated, so Phase 1B needs a sequence model for the lead rather than a gap term.

## 5. Calibration and held-out results (slowest-rank iteration latency)

Metrics per cell: n, median signed error, p90 absolute percentage error (APE, defined for measured
≥ 0.05 ms), bias, worst absolute error. `alt_*` in `RESULTS.md` is the deterministic-median
composition, shown for reference only.

| pass | role | class | n | median err ms | p90 APE % | bias % | worst ms |
|---|---|---|---|---|---|---|---|
| run2 | held-out | decode b≤16 | 7,406 | +0.044 | 3.3 | +0.12 | 19.7 |
| run2 | calibration | decode b≤16 | 10,112 | +0.053 | 5.4 | −0.15 | 33.0 |
| run2 | held-out | 16k chunk (mixed-ring) | 33 of 37 | +1.24 | 2.1 | +0.14 | 10.1 |
| run2 | calibration | mixed-ring | 65 | +0.90 | 4.0 | −0.72 | 25.7 |
| run2 | held-out | mixed-eager (host-bound) | 3 | −17.0 | 9.8 | −7.8 | 21.7 |
| run2 | calibration | mixed-eager (host-bound) | 14 | −6.3 | 18.0 | −5.9 | 43.4 |
| quadrant | held-out | decode b≤16 | 131 | +0.073 | 8.6 | −1.60 | 5.3 |
| quadrant | held-out | decode b17–24 | 49 | +0.098 | 2.4 | +1.17 | 0.25 |
| quadrant | held-out | decode b≥25 | 1,071 | +0.208 | 4.1 | +1.98 | 7.1 |
| quadrant | calibration | decode b≥25 | 2,762 | +0.191 | 4.2 | +1.76 | 10.0 |
| quadrant | held-out | mixed-ring / mixed-eager | 1 / 1 | +1.4 / −3.2 | 0.6 / 1.6 | | |
| quadrant | calibration | mixed-graph (class medians) | 4 | −0.6 | 8.3 | −0.8 | 2.9 |
| salvaged | cross-check | decode b≤16 | 4,527 | +0.043 | 2.8 | +0.36 | 7.4 |
| salvaged | cross-check | decode b17–24 (kv ≥ 96k, extrapolation) | 66 | +0.370 | 6.1 | +3.80 | 4.7 |
| salvaged | cross-check | mixed-ring | 71 | +1.05 | 4.1 | −0.49 | 51.9 |
| salvaged | cross-check | mixed-eager | 2 | −17.7 | 8.9 | −8.4 | 19.3 |
| salvaged | cross-check | mixed-graph (extrapolation) | 1 | +20.0 | 124 | | 20.0 |

**By decode regime and KV band (held-out).** run2 b≤16: kv ≥ 96k p90 APE 2.8 % (n 4,953), kv
32–96k 2.5 % (1,879), kv < 32k 11.1 % (574, bias −2.3 %). Quadrant (all kv < 32k): b≤16 8.6 %,
b17–24 2.4 %, b≥25 4.1 %. The kv < 32k / b≤16 cell is the weakest decode cell: at b = 1–2 the
rank-lateness pattern differs between passes (quadrant: rank 0 late by ≈ 1.0 ms; run2: ranks 1/3),
so a single per-rank table cannot match both (§7).

**By prediction status (run2 held-out).** measured 5,174 it., p90 APE 3.0 %; interpolated 2,227 it.,
3.8 %; kv-extrapolation 5 it., 24.5 % (reported, not folded into the in-range numbers);
extrapolation (mixed) 4 it., 9.7 %; inferred (chunks) 31 it., 1.2 %.

**Kernel-group accuracy (slowest-rank group time, run2 held-out).** Decode b≤16: fmha p90 APE
1.5 %, sparse_meta 3.8 %, moe 4.4 %, other 5.2 %, indexer 11.1 % (median −0.017 ms on 0.39 ms),
comm 13.9 % (median +0.033 ms on 0.77 ms; the comm group includes waiting). 16k chunks: fmha
1.5 %, indexer 4.1 %, sparse_meta 1.2 %, moe 1.5 %, comm 1.8 %, other 1.7 %.

**Per-rank GPU busy and idle (run2 held-out).** Busy: decode b≤16 p90 APE 3.1–3.4 % on all four
ranks (median err 0.039–0.046 ms on 6.19 ms); 16k chunks 1.5–1.7 % (median +2.2 to +2.5 ms on
231 ms). Idle: decode median error ≈ 0 ms, p90 absolute 0.005–0.134 ms (p90 APE 9–26 % of a
0.15 ms quantity); chunks median −0.04 to 0 ms on 1.1–1.2 ms idle, p90 absolute 5.4–9.7 ms
(APE 83–89 %: the idle in GPU-bound chunks is dominated by rare host stalls that a median model
cannot place, see §7).

**Composition rule check (held-out, measured own segments → predicted completion).** Decode:
n 400, median +0.034 ms, p90 |err| 0.038 ms, worst 0.045 ms, Σ over ranks / wall 3.86 (run2);
quadrant median 0.000, p90 0.003 ms. Chunks: n 33, median −0.019 ms, p90 0.078 ms, worst 0.120 ms,
Σ/wall 3.97. This confirms "completion = max over ranks of arrival + intrinsic" as the TP4 rule.

**Mixed host critical path.** Host-bound eager calibration (n 14): forward-window median error
−0.2 ms, p90 APE 5.0 %, bound (host vs GPU) predicted correctly 100 %, launch count median error
+1; held-out eager (n 3): forward window −4.5 ms, p90 APE 6.7 %, bound 100 %. GPU-bound ring
(n 60 / 35): bound predicted correctly 100 %, wall p90 APE 3.3 % / 1.7 %. The forward window in
GPU-bound ring iterations is itself off by ≈ 10 % (back-pressure), but it is not on the critical
path there, so wall is unaffected — the host window was never equated with unhidden latency.

**Full workload.** run2: measured GPU span 134.46 s, Σ predicted over the 17,637 evaluated
iterations 124.16 s vs measured 124.49 s, error −0.26 % (calibration −0.48 %, held-out +0.03 %);
the 49 excluded JIT iterations (10.09 s) are outside the model's scope, so predictions cover
92.6 % of the span. Quadrant: +1.37 % (held-out +1.65 %; b≥25 bias). Salvaged cross-check: +0.05 %.

**Long-context (16k chunks), held-out by KV offset bucket.** Dedicated chunk fit wall p90 APE
0.5–2.8 % in every bucket from kv0 = 0 to 112k (the 112k bucket is extrapolation beyond the 98k
calibration range: −6.5 ms, 2.6 %); indexer p90 APE 1.8–4.7 %. Slopes refitted independently:
held-out pooled indexer 0.2420 ms/1k (chains 0.2365–0.2455), salvaged 0.2429, calibration 0.2429;
held-out pooled wall 0.2377 vs calibration 0.2001 (per-chain wall slopes scatter 0.14–0.31 because
wall absorbs host stalls; the indexer slope is the robust quantity). Linearity holds across
0–115k on every chain; quadratic term at max kv0 is negligible.

**MoE (decode, held-out requests).** Central error vs measured median kernel time: +20.9 % (b=1),
+14.9 % (b=2), −3.4 % (b=3), +0.3 % (b=4), +8.0 % (b=5), +0.7 % (b=6), −3.9 % (b=7), ≤ 2 % at
b = 4, 6, 16, 24, 30, 48, −8 % (b=9), +11 % (b=25). Held-out-routing E(b) differs from calibration
E(b) by < 3 % (6.0 vs 6.0 at b=1, 37.6 vs 38.6 at b=6), so the routing statistics transfer; the
central-estimate errors at small b are a model-form limitation (kernel time is bucket-driven), not
a routing-leakage artefact. The decode table (module 1) remains the production path for decode
MoE time; module 3 is the explanatory/uncertainty model.

## 6. Residuals by class and what explains them

| class | residual pattern | explanation / status |
|---|---|---|
| decode b≤16, kv ≥ 32k | +0.04 ms median, symmetric | stochastic composition removes the −3.4 % deterministic bias; remaining is rank jitter |
| decode b≤16, kv < 32k, b = 1–2 | p90 APE 11 %, bias −2.3 % | per-rank lateness pattern differs between passes (rank 0 vs ranks 1/3); not a KV effect |
| decode b≥25 (quadrant) | +2 % bias | stochastic composition over-predicts slightly where rank jitter is correlated across ranks; deterministic gives −1 % |
| decode kv > 131k | 5 held-out it., 24.5 % | tagged `kv-extrapolation`, excluded from in-range claims |
| 16k chunk at kv0 = 0 | −5 ms (−2 to −4 %) | host and GPU nearly balanced at kv0 = 0; partial host stalls lengthen the GPU timeline without flipping the bound; max() cannot represent partial overlap |
| 16k chunk kv0 ≥ 16k | ±1–4 ms, unbiased | |
| mixed-ring (non-chunk) | +1 ms median, p90 ≤ 4 % | |
| mixed-eager, host-bound | −6 to −17 ms median, p90 APE 10–18 % | host time per launch is noisy (rank 2 slowness unexplained, gap G4); lead-aware composition over-credits the lead when the previous iteration itself stalled |
| mixed-graph | 2 classes only; 1 salvaged extrapolation +124 % | class medians only; outside the two classes the model has no basis (`extrapolation`) |

## 7. Non-identifiable and out-of-range cases (from `classification`)

- Stream assignment of overlapping decode kernels: graph-executor stream ids are not program
  streams (gap G6); overlap is carried as a measured union term, not decomposed per stream.
- Per-iteration MoE routing: not captured in the timing passes (gap G3); only corpus statistics.
- Host cost of KV-dependent launches c_kv: inferred, estimate negative → 0; needs a host-bound
  long-KV eager capture.
- Cause of rank-2 host slowness (≈ 8 % slower per launch): not identifiable (gap G4).
- Decode KV independence: assumed within kv_max ≤ 131,063; residual slope on held-out
  +0.028 ms per 100k kv_sum and +0.087 ms per 100k kv_max (≈ 1.5 % at the maximum), i.e. a small
  real KV effect exists but is below the rank-jitter noise; predictions beyond 131k are tagged.
- Chunk-length × KV interaction for non-16k chunks: inferred structure (gap G1).
- Graph-path mixed iterations outside (P∈{128}, b∈{46,47}): `extrapolation`.
- Host lead: measured per iteration here; predicting it needs the Phase 1B sequence simulator.

## 8. Regression against the existing simulator

- No existing source file was modified; the package is standalone under `profiling/env/…/phase1a`.
- `cargo test --workspace --exclude analyzer` on tray14 (linked against the uv-managed
  libpython3.12 because rack 10 lacks python3.12-dev): 1,606 passed, 0 failed, 10 ignored
  (`out/logs/cargo_test.log`). The `analyzer` crate was excluded because `protoc` is not installed
  on rack 10; it is unaffected by this branch.
- Repository Python parser tests (`test_alignment_nsys_parse`, `test_alignment_overlap`,
  `test_parsed_io`): 73 passed (`out/logs/pytest_alignment.log`).
- Extractor vs frozen parser: per-device kernel counts in the compact tables equal the sum of the
  `parsed.json` range `kernel_count` values on all 19,000 (iteration, device) pairs of the quadrant
  pass (max difference 0). Busy/idle in the compact tables are unions over the whole iteration GPU
  window (not per host-phase range, so not compared); the composition check (§5) shows they are
  self-consistent to < 0.05 ms per decode iteration.

## 9. CPU tests (`tests/test_costmodel.py`, 15 tests, all pass)

overlap/union accounting (compute vs comm, compute–compute); windowed union; 4-rank critical path
is not a sum and equals max arrival + intrinsic; collective wait vs intrinsic separation; batch ==
scalar composition; stochastic composition ≥ deterministic only when jitter exists; decode table
interpolation, extrapolation, `MIN_TABLE_N` and kv status; deterministic reproduction and
`to_dict/from_dict` round trip; host-bound eager with and without lead; GPU-bound ring and mixed
kind thresholds; indexer KV slope, out-of-range and inferred-structure statuses; MoE padding and
range status; frozen partition hash and leakage guard (tampered role → `LeakageError`); duplicate
iteration id rejection; code-hash provenance.

## 10. Remaining GPU experiments (not authorised; priorities for Phase 1B/1C)

| prio | experiment | why | resources |
|---|---|---|---|
| 1 | host-bound eager mixed iterations with long KV (P 2–8k, kv0 32–128k, b ≤ 8), nsys + host timeline | identify c_kv and the launch-count KV term independently of back-pressure; shrink eager p90 APE from 10–18 % | 1 tray (4 GPUs), ≈ 20 min capture, same frozen preset |
| 2 | decode b = 1–4 at kv < 32k, both passes' request mixes, with per-rank host timeline | explain the pass-dependent rank-lateness pattern (rank 0 vs ranks 1/3) | 1 tray, ≈ 10 min |
| 3 | MoE routing capture (expert ids per layer per iteration) on a decode-heavy pass | turn module 3 from statistical to per-iteration; closes gap G3 | 1 tray, needs a vLLM-side hook (code change, review first) |
| 4 | graph-path mixed sweep (P ≤ 256, b 1–64) | replace 2 class medians by a table | 1 tray, ≈ 10 min |
| 5 | 16k chunks at kv0 = 0 with host timeline, repeated | quantify partial host stalls behind the −2 to −4 % at kv0 = 0 | 1 tray, ≈ 10 min |
| 6 | non-16k chunk lengths (4k, 8k, 12k) at kv0 0–96k | confirm the chunk-length × KV interaction (gap G1) | 1 tray, ≈ 15 min |
| 7 | decode kv_max 131k–256k | extend the KV-independence range | 1 tray, ≈ 15 min |

## 11. Recommendation on Phase 1B readiness

Ready to proceed to Phase 1B (CPU-only integration of the decode, chunk and ring-mixed components
into a sequence simulator with a lead model), with the following conditions:

1. Decode (b≤16, all KV bands ≥ 32k) and 16k-chunk prefill are production-grade on held-out data
   (p90 APE 3.3 % and 2.1 %, bias ≤ 0.2 %, full-workload −0.26 %).
2. Decode b ≥ 17 carries a +1–2 % bias and b = 1–2 at low KV an 11 % p90 APE; acceptable for
   Phase 1B but to be revisited with experiment 2.
3. Mixed-eager host-bound iterations (≈ 2.5 % of the run2 span) are the weakest component
   (p90 APE 10–18 %); Phase 1B should carry their uncertainty explicitly and not tune them without
   experiment 1.
4. The host lead must be simulated (previous-iteration GPU end vs this iteration's host start),
   since the GPU is saturated and the lead is what decides whether a mixed iteration is host- or
   GPU-bound in practice.
5. Nothing in this phase authorises GPU execution or production integration; both need a
   separate go.

## 12. Commit

Implementation branch `phase1a-costmodel` (base `cc06d9c2`), single commit on top of `cc06d9c2` (`git log cc06d9c2..phase1a-costmodel`),
pushed to `kaijunli-infr/ServingStudioSim`. Committed: `costmodel/`, `tests/`, `partition/`,
`out/{coefficients.json, COEFFICIENTS.sha256, results.json, RESULTS.md, logs/}`, `README.md`,
this report, and the three exploration scripts (`explore1.py`, `explore2.py`, `debug_eager.py`)
kept for the record. Not committed: the compact tables (164 MB, on rack 10 CephFS, hashes recorded)
and `explore2.out`.
