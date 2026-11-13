# Phase 1C GPU experiment matrix (six captures, independently approvable; none authorised, none run)

Status: proposal for review. Every experiment below needs its own go (matrix, duration, tray shown; `nvidia-smi` and the
Kubernetes scheduler checked; reservation pod applied) before anything runs. The validation protocol that governs how
the results may be used is frozen in `VALIDATION-PROTOCOL.md` (hash in `PROTOCOL.sha256`) and must not be edited after
the first capture starts.

## Common setup

| item | value |
|---|---|
| hardware | one rack-10 GB300 tray (4 GPUs, TP4), reserved through `k8s/dsv41-gb300/reserve-tray16.yaml` (or another free tray) for the whole experiment; host-level GPU use only on a tray with no GPU pods, and still pre-emptible by the scheduler (stop condition) |
| software | the Phase 0 recipe exactly: vLLM fork `servingstudio-alignment-v0.31.0` (0ab18711), uv Python 3.12, TP4, chunk 16384, CUDA-graph capture up to 256 tokens, warm JIT/autotune caches under `gb300/cache/`; weights `/mnt/shared/weights/DeepSeek-V4.1-Flash-1017` (HF 08c8657f) |
| capture | nsys with CUDA, NVTX and OS-runtime (launch API) tracing, delayed start 10 s, **no bounded stop** (Phase 0 lesson: an in-inference `cudaProfilerStop` froze the engine); per-rank host NVTX ranges as in Phase 0 (preprocess / forward / postprocess / sample / bookkeep / eplb) plus the vLLM scheduler step range; `parsed.json` from the Phase 0 parser; compact tables from `costmodel.extract` (hash-recorded) |
| attribution | kernel → segment via the Phase 0 collective-kind sequence (K collectives per iteration), kernel names → groups (fmha / indexer / sparse_meta / moe / comm / other) via `costmodel.groups`; host launches via the CUDA API trace (`cuLaunchKernel`, `cudaGraphLaunch`) per rank; per-iteration ids = `iteration_index` |
| storage | nsys `.nsys-rep` + exported sqlite + `parsed.json` + compact tables; Phase 0 averaged ≈ 5 GB per nsys pass; each budget below includes a 2× margin; raw captures to `artifacts/<capture>/` with `SHA256SUMS` + `MANIFEST.md`; tables to `phase1a-tables/<label>/`; mirror on rack 3 |
| stop conditions (all experiments) | tray not free or a GPU pod scheduled on the tray; scheduler pre-emption; JIT compile burst > 20 modules after warm-up (cache cold = wrong cache dir); engine stall > 60 s; nsys report write > 10 min; free disk on the capture volume < 100 GB; any OOM / NCCL error; a configuration drift (vLLM commit, flags) from the recipe |
| exclusions (pre-registered) | JIT-affected iterations by the Phase 0 rule; the first 50 iterations after engine start (warm-up); iterations with `_kind_mismatch` in the collective sequence |

Expected-uncertainty-reduction ranking uses the Phase 1B run2 figures: share of the workload Σ walls per class × the
current spread between what the model can and cannot pin down for that class (median-path vs stochastic-mean bias,
p90 APE, or extrapolation status). Workload shares (run2): decode b3-16 and b≥25 ≈ 79 % of Σ walls, 16k chunks ≈ 19 %,
eager mixed ≈ 2.6 %, decode b1-2 ≈ 7 % of iterations but ≈ 6 % of walls, graph-kind mixed < 0.5 %.

## E1 — decode routing and rank-arrival capture (identification)

| field | content |
|---|---|
| unresolved question | Why do the median path and the stochastic mean of decode walls bracket the measurement by ≈ 3 % (run2 b3-16: −2.8 % / +0.5 %; quadrant: −1 % / +2 %)? Is the per-rank segment variation common-mode (same kernels, routing-driven) or independent (host-arrival driven)? What is the per-iteration routed-expert distribution and how much wall variance does it explain? |
| workload and configuration | synthetic decode-only runs at fixed batch b ∈ {1, 2, 4, 8, 16, 24, 32, 48} (≥ 400 iterations each) at three KV levels (≈ 4k, 32k, 96k tokens per request), the diverse corpus prompts; same recipe; plus the **routing hook**: a reviewed vLLM-fork patch that logs per-layer routed expert ids (or expert load counts) per iteration to a side file with the iteration index (patch reviewed and merged on the fork branch before the capture; no behaviour change) |
| why existing traces cannot answer | the Phase 0 traces have no routing information (gap G3); per-rank arrival at the first collective is measured, but with mixed batch sizes and interleaved prefill the common-mode vs independent question is confounded by the preceding iteration's skew carry-over; b=1–2 at kv<32k shows pass-dependent rank-lateness patterns (Phase 1A) |
| expected measurements and attribution | per iteration and rank: segment windows (R, K+1), collective intrinsic and wait, first-collective arrival spread, host launch times (graph launch), scheduler gap; routing: expert ids / loads per layer; derived: correlation of per-rank segment deviations across ranks (common-mode fraction), wall variance explained by routed-expert load, per-b dispersion quantiles by KV |
| designation | **identification** (new parameters: common-mode fraction of decode dispersion, routing-conditional MoE time, b1-2 arrival pattern). Results are never used for validation; E2 validates. |
| GPU occupancy and reservation | 1 tray × 4 GPUs, ≈ 60 min wall (8 b × 3 KV × ≈ 1.5 min + model load + warm-up) + 20 min nsys export; reservation pod for 2 h |
| storage and profiling budget | 24 short nsys runs or 3 runs (one per KV level) with the batch sweep inside: ≈ 3 × 6 GB nsys + sqlite ≈ 25 GB raw, ≈ 1 GB tables, routing log ≈ 200 MB; budget 40 GB |
| stop conditions (specific) | routing hook adds > 2 % to iteration wall in a dry run without nsys (measured against Phase 0 b=8 median 5.2 ms); any divergence of the generated tokens from a no-hook run on the same prompts (hook must be observation-only) |
| decision enabled | choose between common-mode and independent per-rank sampling (or a fitted mixture) for the decode band; decide whether a routing-conditional MoE term replaces the corpus band; decide whether b1-2 needs its own arrival model or stays lower-confidence |
| priority | **1** — decode is ≈ 79 % of the workload total; the 3 % median/mean spread is the largest single uncertainty in the full-workload prediction (≈ 2 % of span) |

## E2 — decode validation capture (validation only)

| field | content |
|---|---|
| unresolved question | Does the decode model revised after E1 predict a never-seen decode workload within the pre-registered acceptance (p90 APE ≤ 5 % in every batch bucket, coverage 70–90 % at p10–p90)? |
| workload and configuration | the Phase 0 `diverse_100` and `quadrant` request mixes re-issued with a different request-order seed and a different prompt subset (≥ 3,000 decode iterations spanning b 1–48), in a **separate vLLM launch on a separate day** from E1, no routing hook (production recipe) |
| why existing traces cannot answer | the Phase 1A held-out rows were used once to report Phase 1B and are frozen; any parameter identified in E1 needs an independent capture for a one-shot validation (protocol §4) |
| expected measurements and attribution | as Phase 0: per-iteration walls, segments, host timeline; evaluation with the four Phase 1B modes and the reliability tables of `uncertainty/reliability.py` |
| designation | **validation** (one-shot, protocol §5); never used for any fit |
| GPU occupancy and reservation | 1 tray, ≈ 30 min capture + 15 min export; reservation 1.5 h |
| storage and profiling budget | ≈ 8 GB nsys + sqlite, 300 MB tables; budget 15 GB |
| stop conditions (specific) | none beyond common |
| decision enabled | accept / reject the revised decode model for Phase 1C-2 integration; if rejected, the next revision requires a new validation capture |
| priority | **2** (paired with E1; without it E1 cannot change the model) |

## E3 — host-bound eager mixed iterations at long KV (identification + paired validation half)

| field | content |
|---|---|
| unresolved question | What is the KV-dependent per-launch host cost c_kv (Phase 1A: 0, not identifiable), what causes the rank-2 host lateness, and what sets the host pacing variation (±20 ms rank-to-rank) that dominates the eager p90 APE of 6–37 %? |
| workload and configuration | prefill-heavy mix forcing eager mixed iterations: prompt lengths 2–8k with kv0 32–128k (long prefixes via multi-turn continuation), b ≤ 8 decode requests in the same batch, ≥ 150 eager mixed iterations; per-rank host NVTX at launch granularity (Phase 0 ranges + per-layer markers, read-only patch) |
| why existing traces cannot answer | run2 has 17 eager mixed iterations (14 calibration), quadrant 4, salvaged 2; KV16 launch counts vary but their per-launch cost is confounded with the base host cost (collinear in the Phase 1A fit) |
| expected measurements and attribution | per rank: launches per layer, host time per launch vs KV (c_kv), rank-2 host offset per iteration, GPU idle gaps, wait at each collective; attribution of host time to Python stages via the per-layer markers |
| designation | **identification**; the capture is split by request sequence into an identification half and a validation half taken in a separate vLLM launch (two runs, same day acceptable, different seeds) |
| GPU occupancy and reservation | 1 tray, 2 × 20 min + exports; reservation 1.5 h |
| storage and profiling budget | 2 × 6 GB + sqlite; budget 25 GB |
| stop conditions (specific) | fewer than 100 eager mixed iterations after 15 min (mix not producing the class → stop and redesign, do not extend) |
| decision enabled | whether c_kv and a rank-2 offset become measured parameters (with a band) or the class stays lower-confidence in production; whether eager needs its own dispersion reference |
| priority | **3** — small workload share (2.6 %) but the largest per-class error and an explicitly unresolved production case |

## E4 — long chunk chains to 160k (validation of extrapolation range + Q)

| field | content |
|---|---|
| unresolved question | Does the 16k-chunk prediction (≈ 0.5 % along chains to 112k) hold beyond the validated KVW range (98.2k prefill, 131k decode) up to 160k, and what is the hard launch-queue cap (Q = 968 lower bound; CUDA nominal 1024)? |
| workload and configuration | 20 prompts of 150–160k tokens (10 chunks each), b = 0 or 1 decode alongside, ≥ 200 chunk iterations; the per-rank API trace on |
| why existing traces cannot answer | run2 chains end at 131k (8 chunks), saturation of the lead is reached only on 58 rows; kv0 ≥ 112k is `extrapolation` status today |
| expected measurements and attribution | chunk walls by kv0, indexer group slope vs kv0 beyond 115k, lead per chunk, implied queue depth per rank-row at saturation, launch count vs kv16 |
| designation | **validation** for the chunk model and Q (pre-registered acceptance: chunk p90 APE ≤ 3 % to 160k; lead p90 abs error ≤ 20 ms); indexer slope beyond 115k may be *identified* only from the first half of the chains, with the second half reserved for validation (protocol §4) |
| GPU occupancy and reservation | 1 tray, ≈ 25 min + export; reservation 1 h |
| storage and profiling budget | ≈ 8 GB; budget 15 GB |
| stop conditions (specific) | max_model_len or KV-cache capacity reached (stop; do not change the recipe) |
| decision enabled | extend the supported KV range of the production model to 160k or keep `kv-extrapolation` beyond 131k; replace the Q lower bound by a measured cap |
| priority | **4** — chunks are 19 % of the workload but already at 0.5 %; the value is the range extension and the unbounded extrapolation status |

## E5 — graph-path mixed sweep (identification, small)

| field | content |
|---|---|
| unresolved question | The two graph-kind mixed classes (P + b ≤ 256) are class medians (n = 4 + 1); what is the wall as a function of (P, b) under CUDA-graph capture? |
| workload and configuration | short prompts P ∈ {16, 64, 128, 240} with b ∈ {1, 4, 16, 64} decode requests, ≥ 50 iterations per cell |
| why existing traces cannot answer | 5 iterations in total across passes; the class does not occur in the diverse workload often enough |
| expected measurements and attribution | wall, segments and host window per (P, b) cell; graph launch count |
| designation | **identification** (table); validation through the mixed iterations of E2 (which contain this class at its natural rate) |
| GPU occupancy and reservation | 1 tray, ≈ 15 min; reservation 45 min |
| storage and profiling budget | ≈ 4 GB; budget 8 GB |
| stop conditions (specific) | none beyond common |
| decision enabled | replace the class medians by a (P, b) table with status; or leave the class as low-confidence (workload share < 0.5 %) |
| priority | **5** |

## E6 — cold-start and first-chunk-after-decode (identification, small)

| field | content |
|---|---|
| unresolved question | The first 16k chunk after a decode burst (kv0 = 0, status in-range) is the weakest chunk class (p90 APE 2–4 %, both signs, −7 ms on the salvaged held-out pair); f_idle (0.54 ms) rests on n = 4. What sets the first-chunk GPU start and its wall? |
| workload and configuration | alternating pattern: 30 decode iterations then one 16k prompt, repeated 60 times, at two decode batch sizes (b = 4, 32); plus 20 cold starts (GPU idle ≥ 1 s before the chunk) |
| why existing traces cannot answer | run2 has 23 such chunks (20 calibration) with mixed preceding batch sizes; cold-start rows n = 4 |
| expected measurements and attribution | first-kernel offset from host start (f_idle) by preceding kind and b, first-chunk segment profile vs chain-chunk profile, host pre-phase |
| designation | **identification** of f_idle and a first-chunk profile; validation via the natural occurrences in E2 |
| GPU occupancy and reservation | 1 tray, ≈ 15 min; reservation 45 min |
| storage and profiling budget | ≈ 5 GB; budget 10 GB |
| stop conditions (specific) | none beyond common |
| decision enabled | a separate first-chunk profile (status measured) or keep the shared profile with the current error |
| priority | **6** |

## Summary

| prio | experiment | designation | GPU time (tray-hours incl. export) | budget | expected reduction |
|---|---|---|---|---|---|
| 1 | E1 decode routing + rank arrival | identification | 2.0 | 40 GB | up to ≈ 2 % of workload span (decode median/mean spread), b1-2 pattern, MoE band → conditional |
| 2 | E2 decode validation | validation | 1.5 | 15 GB | makes E1 usable; one-shot acceptance |
| 3 | E3 host-bound eager at long KV | identification + validation half | 1.5 | 25 GB | eager p90 APE 6–37 % → target ≤ 10 %; c_kv, rank-2 |
| 4 | E4 chunk chains to 160k | validation (+ indexer slope id. on first half) | 1.0 | 15 GB | KV range 131k → 160k; Q cap |
| 5 | E5 graph-path mixed sweep | identification | 0.75 | 8 GB | class medians → table (< 0.5 % of workload) |
| 6 | E6 cold start / first chunk | identification | 0.75 | 10 GB | first-chunk 2–4 % → ≤ 1.5 %; f_idle n 4 → ≥ 50 |

Total if all six are approved: ≈ 7.5 tray-hours (≈ 30 GPU-hours), ≈ 115 GB raw budget. E1 + E2 alone cover the
largest uncertainty; E3 is the only one that addresses an unresolved production case; E4–E6 are range and
small-class refinements.
