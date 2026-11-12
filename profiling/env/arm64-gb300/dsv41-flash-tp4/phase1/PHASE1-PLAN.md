# Phase 1 planning — cost-model design for DeepSeek-V4.1-Flash GB300 TP4 (CPU-only analysis of the frozen Phase 0 traces)

Scope of this document: architecture proposal, long-context and TP4 critical-path analyses, validation split, measurement gaps,
expected implementation changes, GPU estimate. **No GPU was used, no vLLM launched, no profiling database modified**; every number
comes from read-only analysis of the frozen traces (`phase1/*.py`, outputs in `phase1/out/`). Phase 1 implementation and any execution
require separate approval. Baseline: `test-ss` @ `ca9f8ed0` (see `PHASE0-BASELINE.md`).

Conventions. "busy" = union of kernel intervals on one rank; "wall" = slowest-rank span (last kernel end − first kernel start over the
4 ranks); "intrinsic" collective cost = per-call minimum duration over ranks; "wait" = per-rank collective time above intrinsic;
"skew" = per-call spread of start times across ranks. Clean = JIT/module-load iterations excluded (`PHASE0-BASELINE.md` §3).

## 1. Cost-model architecture

Design principle (matches `skills/top-add-new-arch`): L1 = measured kernel-group timings keyed on model geometry and dtypes;
L2/L3/L4 compose them with explicit overlap rules (`sum` / `max`); **no global `gpu_time_multiplier`** (the B200 preset's 1.0361 is
replaced by explicitly modelled idle, overlap and skew terms below, each measured). Iteration time is `max over ranks` of a per-rank
timeline, never a sum over ranks.

### 1.1 Component models
| component | model | inputs | status |
|---|---|---|---|
| **Decode step** (graph path) | `T_dec(b) = max_r[ compute_r(b) + comm_r(b) ] + idle(b)`; three kernel-path variants selected by batch b: ≤16 (cute-DSL `all_reduce_mhc`×154 + oneshot×4, split-K dense GEMMs), 17–24 (oneshot×81 + mega_mhc), ≥25 (twoshot×81 + mega_mhc); compute piecewise in b following the graph buckets (measured steps at b=3, 5, 9); **independent of KV length** (measured flat: b=11 → 6.29–6.34 ms busy for Σkv 100k–400k+). | b, kernel-path variant | compute per group **measured** (diverse: b 1–16; quadrant: b 1, 3, 6, 12, 24–27, 48); idle 0.16 ms **measured** (graph gaps); multi-stream overlap 1.4–2.3 ms **measured** (sum−union) — the model must compose with `union`, not `sum` |
| **Mixed prefill+decode** | two regimes by prefill tokens P: (a) P ≤ ~8k, eager: host-launch-bound, `T = max_r H_r(n_launch)` with GPU compute (20–28 ms) fully hidden; `n_launch` ≈ 1,580–1,590 per rank, host window 186–205 ms (rank 2 slowest). (b) P ≥ ~10k: GPU-bound like a prefill chunk; decode requests add 0.031 ms/request (residual slope, r = 0.05 — negligible). | P, number of chunks, b | (a) host window **measured** (NVTX `forward` span, API call counts and durations); per-launch Python cost **inferred** (window / launches ≈ 120–130 µs, of which 14–16 µs is CUDA API); (b) **measured** |
| **Long-context prefill chunk** (16k) | `T_chunk(L=16k, K) = 219.1 + 0.260·K_k ms` (rank-0 busy; slowest-rank 219.7 + 0.251·K_k); wall = busy + 1.2 ms idle; decomposed per group in §2. Only the indexer term depends on K. Validity: K₀ ∈ [0, 115k], L = 16,370–16,384; no extrapolation beyond `max_model_len` 131,072. | chunk length L, KV offset K | **measured** at L = 16k for 8 KV depths; L-dependence between 256 and 16k at deep KV **not identifiable** from Phase 0 (only a handful of multi-chunk iterations) → gap G1 |
| **Sparse indexer + metadata** | prefill: `I(L,K) = I₀(L) + s·K`, s = 0.24 ms per 1k KV at L = 16k (mqa_logits 15.5 ms, topKPerRowPrefill 7.4 ms, sparse_mqa_logits 2.6 ms of the 22.4 ms growth 0→115k); sparse metadata (index build, sorts, scans, FullCache insert) ≈ 4.5–4.6 ms flat; decode: 0.20–0.28 ms per step, moves with b not K. | L, K, b | s **measured** and stable across 13 chains (0.232–0.256); L-scaling **inferred** (one L); decode indexer **measured** (launch-bound, flat in K) |
| **MoE expert GEMMs** | prefill: 38.5–39.4 ms per 16k chunk, flat in K (TRT-LLM MXFP4 grouped GEMM, `bmm_MxE4m3` + `bmm_Bfloat16` + routing/finalize 10.5 ms); decode: 0.66 (b=1) → 1.2 (3–4) → 1.7 (6–8) → 2.7 ms (≥9) → b=48 part of 9.4 ms compute; driver = distinct experts touched per layer and max tokens per expert, from the corpus: 11 tokens → 48.6 experts/layer, 48 → 133, 256 → 262, 16k → 378 (`out/routing_stats.out`). Experts are **TP-sharded** (no EP) in this deployment. | b or L, routing distribution | batch-level kernel time **measured**; per-expert load per iteration **not captured** (fused kernel, routes come from a different pass) → modelled **statistically** from the corpus (gap G3); layer-level popularity skew (max/mean up to 57) **measured** in the corpus, its kernel-time effect **not identifiable** |
| **TP4 collectives** | per kind: intrinsic per call (min over ranks) + wait term from rank-arrival skew. Decode ≤16: `all_reduce_mhc` 154 calls, 0.38–0.43 ms/step intrinsic (≈2.8 µs/call) + oneshot ×4 (16 µs) + NCCL AllGather ×5 (53–67 µs); 17–24: oneshot ×81 = 0.36 ms; ≥25: twoshot ×81 = 0.45–0.47 ms. Prefill 16k: NCCL ring AllReduce ×81 = 36.9 ms intrinsic (0.456 ms/call), AllGather ×5 = 2.76 ms; wait 1.1–2.8 ms. Eager small mixed: twoshot ×81, intrinsic 1.2–3.2 ms/iteration, the remaining 155–170 ms per rank is wait for the laggard rank (§3). Compute/comm overlap ≈ 0 (compute + comm_union = busy within 0.1 ms in decode, 0.5 ms in prefill). | kind, call count, payload (inferred from hidden size 5120 × tokens), b/L | intrinsic **measured**; payload bytes **inferred** (not in trace); wait **measured** but **caused by** host skew (decode b ≤ 2) or host launch rate (eager) — must be modelled through the host component, not as collective cost |
| **Host scheduling + synchronization** | decode: host runs 5.8 ms ahead, one `cudaEventSynchronize` per step closes the loop; step wall = GPU wall for b ≥ 3; for b = 1–2 rank 0's `forward` starts 2.0–2.3 ms after the others (rank-0 sampling/bookkeeping) → 0.8–0.9 ms collective wait on ranks 1–3 and 0.15–0.8 ms idle. Eager mixed: host-launch-bound (above); rank 2's host window is 5–10 % longer than ranks 0/1/3 in every eager iteration (its GPU idles 167–179 ms while the others wait in twoshot). Prefill 16k: GPU-bound (host window 200–211 ms < GPU 228 ms); rank 0 spends 52 ms in one `cudaEventSynchronize`. | b, P, launch count | host phase durations, run-ahead, skew, API counts **measured**; root cause of rank 2's slower host (CPU placement/NUMA on Grace) **not identifiable** from the GPU/API trace → gap G4 |

### 1.2 Composition rules (L3/L4)
- Per rank: `busy_r = union(kernels)`; groups are summed within a rank only where the trace shows no overlap (prefill: all groups serial; decode: multi-stream overlap of 1.4–2.3 ms must be reproduced by a `max`-type composition of the overlapping streams, e.g. indexer/attention side-stream vs. main stream, to be identified per stream in Phase 1 implementation from `streamId` in the sqlite).
- Across ranks: `wall = max_r (start_r + busy_r + wait_r + idle_r)`; `wait_r` arises from `start skew` (host) and from compute imbalance (none measured: per-rank compute agrees within 1 % in all classes).
- Iteration → request metrics: TTFT/TPOT from the scheduler replay (chunk 16384, max_num_seqs 128, graph cap 256 ⇒ eager prefill), validated end-to-end on the workload passes.

## 2. Long-context analysis (16k chunk cost vs KV)
Source: `out/longctx_run2.out` (88 clean single-chunk iterations, 13 chains, recovery capture) and `out/longctx_salv.out` (58, original salvaged capture).

Per-group rank-0 time (ms) along the 131k chain (iterations 7579–7586, clean):
| K₀ | 0 | 16k | 33k | 49k | 66k | 82k | 98k | 115k |
|---|---|---|---|---|---|---|---|---|
| busy | 219.9 | 221.3 | 227.3 | 230.7 | 236.3 | 242.0 | 243.9 | 251.6 |
| FMHA | 41.7 | 40.5 | 40.6 | 40.4 | 40.4 | 40.4 | 40.4 | 40.4 |
| indexer | 6.3 | 10.7 | 14.5 | 18.3 | 22.2 | 27.1 | 29.1 | 35.4 |
| sparse metadata | 4.5 | 4.5 | 4.5 | 4.5 | 4.5 | 4.5 | 4.5 | 4.5 |
| MoE | 38.9 | 38.5 | 38.9 | 38.5 | 39.0 | 38.8 | 39.0 | 38.9 |
| comm (rank 0, incl. wait) | 35.6 | 35.4 | 35.3 | 34.4 | 35.3 | 35.5 | 35.1 | 35.3 |
| other (mHC 23, dense GEMM 23.5, quantize 8.3, engram 3–4.5, …) | 93 | 92 | 94 | 94 | 95 | 96 | 96 | 97 |
| GPU idle (span − busy) | 1.2 | 1.2 | 1.2 | 1.2 | 1.2 | 1.2 | 1.2 | 1.1 |
| multi-stream overlap (sum − union) | 5.3 | 5.3 | 5.2 | 5.3 | 5.3 | 5.4 | 5.4 | 5.4 |
(values for K₀ = 16k–98k are from the matching rows of chains 10/12/17; see the full table in `out/longctx_run2.out`.)

Fits (rank-0 busy vs K₀ in 1k tokens):
- pooled linear: **219.1 + 0.260·K_k ms**, rmse 1.80 ms, max residual 7.1 ms; slowest-rank busy 219.7 + 0.251·K_k (rmse 2.0).
- per chain: slope 0.176–0.301 (13 chains); the **indexer group slope is 0.232–0.256 in every chain**; the spread of the total slope comes from
  the comm (−0.10 … +0.02) and idle (−0.19 … +0.005) terms, i.e. from wait/skew noise, not from kernel cost.
- quadratic: 219.7 + 0.201·K_k + 0.00068·K_k² (rmse 1.68); the curvature term is 11.7 ms over 0–131k, within the chain-to-chain noise →
  linear is adequate **inside the measured range (0–115k offset, i.e. up to 131k total)**; nothing is claimed beyond it.
- piecewise: [0,40k) 0.201, [40k,80k) 0.312, [80k,140k) 0.231 ms/1k; the indexer term alone is 0.252 / 0.240 / 0.205 — the only hint of
  sub-linearity is in the deepest band (n = 10) and is not separable from noise with the available samples.
- cross-check on the original salvaged capture (independent run): pooled 0.256 ms/1k, indexer 0.236–0.268 per chain — the slope persists.
- confound: residual vs number of concurrent decode requests r = 0.05 (0.031 ms/request); vs decode KV sum r = −0.01.
- per-kernel growth 0 → 115k (rank 0, 22.4 ms total): `sm100_mqa_logits` +15.5, `topKPerRowPrefill` +7.4, `sm100_sparse_mqa_logits` +2.6,
  `_block_scores_kernel` +1.5, torch top-k/sort helpers +2.6, engram lookup +0.6; FMHA, MoE, dense GEMM, mHC, quantize, NCCL: flat (|Δ| < 0.3).

Interpretation for the model: the 16k prefill chunk is 219 ms of KV-independent work plus an indexer term linear in KV offset. FMHA is
flat because the sparse attention reads a fixed top-k (2048) of KV entries per query; the indexer scores every KV entry, hence the linear term.

## 3. TP4 critical-path analysis
Source: `out/critpath_run2.out`, `out/critpath_quad.out` (per-class medians over up to 400 sampled clean iterations; representative per-rank tables inside).

| class (clean) | n | GPU wall | host wall | Σ busy (4 ranks) | slowest-rank busy | compute | comm union | intrinsic | wait | skew Σ | idle | overlap | host launch skew | slowest rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode b=1 (diverse) | 775 | 4.55 | 4.58 | 16.95 | 4.35 | 3.18 | 1.27 | 0.45 | 0.83 | 1.46 | 0.15 (r0: 0.78) | 1.94 | 2.34 | r2 78 %, r3 20 % |
| decode b=2 | 1582 | 4.40 | 4.41 | 16.57 | 3.93 | 3.34 | 0.54 (r0/1/3: ~1.0) | 0.46 | 0.09 (r0/1/3: 0.55) | 0.93 | 0.43 (r2) | 2.04 | 2.02 | r2 71 % |
| decode b=4 | 1414 | 4.60 | 4.59 | 17.77 | 4.44 | 3.95 | 0.56 | 0.48 | 0.09 | 0.49 | 0.15 | 1.88 | 0.35 | r2 75 % |
| decode b=8 | 1675 | 5.19 | 5.19 | 20.13 | 5.03 | 4.44 | 0.66 | 0.49 | 0.18 | 0.72 | 0.15 | 2.03 | 0.32 | r2 61 % |
| decode b=11 | 2451 | 6.48 | 6.49 | 25.23 | 6.31 | 5.67 | 0.71 | 0.51 | 0.21 | 0.87 | 0.16 | 2.04 | 0.33 | r2 59 %, r3 39 % |
| decode b=16 | 26 | 6.32 | 6.31 | 24.56 | 6.14 | 5.51 | 0.71 | 0.51 | 0.21 | 0.85 | 0.17 | 2.04 | 0.38 | r2 58 % |
| decode b=24 (quadrant, oneshot path) | 238 | 7.73 | 7.73 | 30.24 | 7.56 | 7.11 | 0.51 | 0.44 | 0.07 | 0.21 | 0.16 | 1.40 | 0.24 | r2 55 % |
| decode b=25 (twoshot path) | 262 | 8.15 | 8.14 | 31.89 | 7.97 | 7.61 | 0.63 | 0.55 | 0.08 | 0.20 | 0.16 | 1.61 | 0.25 | r2 60 % |
| decode b=48 | 3032 | 9.94 | 9.92 | 39.06 | 9.76 | 9.40 | 0.65 | 0.58 | 0.08 | 0.20 | 0.16 | 1.55 | 0.39 | r2 57 % |
| mixed < 1k prefill tok (eager) | 9 | 200.3 | 196.2 | 567 | 19.7 (r2) | 19.6–19.8 | 155–172 (r0/1/3), 1.7 (r2) | 1.45 | 154–170 (r0/1/3), 0.1 (r2) | 174 | 10–28 (r0/1/3), 179 (r2) | 1.4 | 0.68 | r2 7/9 |
| mixed 1k–10k | 13 | 198.8 | 188.4 | 623 | 70.9 (r2) | 60.7–60.9 | 96–106 (r0/1/3), 10.3 (r2) | 10.3 | 83–90 (r0/1/3), 0.1 (r2) | 90 | 4–10 (r0/1/3), 94 (r2) | 1.7 | 0.70 | r2 12/13 |
| mixed ≥ 10k multi-chunk | 9 | 228.7 | 219.4 | 911 | 227.6 | 190.5 | 38.0 | 36.9 | 1.1 (r0/3: 2.4–2.5) | 4.9 | 1.1 | 5.5 | 7.4 | r2 8/9 |
| prefill single 16k chunk | 88 | 228.5 | 204.9 | 903 | 226.4 | 187.5 | 38.5 | 36.9 | 1.6 (r0/3: 2.5–2.8) | 5.2 | 1.2 | 5.3 | 19.9 | r2 86 % |
(ms; "compute/comm/wait/idle/overlap" shown for the rank with the largest compute, per-rank values in the output files.)

Findings:
1. **Critical path = slowest rank.** Σ busy over ranks is 3.9–4.0× the slowest-rank busy in every GPU-bound class; summing ranks would overstate
   step time ~4×. Per-rank compute agrees within 1 % (no compute imbalance); rank 2 finishes last in 55–86 % of iterations in all classes.
2. **Decode, b ≥ 3: GPU-bound and serialized.** GPU wall = host wall (±0.01 ms); compute + comm_union = busy within 0.05–0.1 ms (no
   compute/comm overlap); intrinsic collective cost 0.45–0.58 ms/step; wait 0.07–0.21 ms; graph launch gaps (idle) 0.16 ms; multi-stream
   overlap 1.4–2.3 ms (kernel sum exceeds union — a `sum` composition would overstate decode by 25–35 %).
3. **Decode, b = 1–2: host-skew-bound on rank 0.** Rank 0 starts `forward` 2.0–2.5 ms after ranks 1–3 (host launch skew), which the others absorb
   as 0.8–0.9 ms of oneshot/ar_mhc wait (first oneshot call: 660 µs on ranks 1–3 vs 4 µs on rank 0) and rank 0 absorbs as idle; this is a
   host effect and must be priced in the host component.
4. **Eager mixed iterations (P ≤ ~8k): host-launch-bound.** ~1,585 launches per rank; host window 186–205 ms; CUDA API time only 22–26 ms of it
   (14–16 µs per launch), the rest is framework/Python time between launches; GPU compute 20–61 ms per rank is hidden; ranks 0/1/3 wait in
   twoshot for rank 2 whose host window is 5–10 % longer (rank 2 GPU idle 167–179 ms). twoshot intrinsic is only 1.2–3.2 ms per iteration.
5. **Prefill chunks: GPU-bound, NCCL ring.** 81 ring all-reduces at 0.456 ms each = 36.9 ms (16 % of the chunk), no overlap with compute,
   wait 1–3 ms, idle 1.2 ms, overlap 5.3 ms; host window (200–211 ms) is shorter than the GPU (228 ms).

## 4. Validation design (defined before any fitting)
Split generator: `phase1/split.py` → `out/validation_split.json` (deterministic: iteration-id hash for 80/20, whole held-out classes listed below).
Nsys passes used: diverse_nsys_run2 (record), quadrant_nsys run 2 (record), diverse_nsys salvaged (cross-check only, never for calibration).

Held-out classes (never used for fitting):
- **Decode batch sizes**: diverse b ∈ {5, 10, 14} (662 / 1,366 / 204 iterations); quadrant b ∈ {26, 30, 36} (129 / 127 / 126). Calibration uses
  the remaining b (1–4, 6–9, 11–13, 15–16; 1, 3, 6, 12, 24, 25, 27, 48) with a further 20 % per-class hash hold-out.
- **KV context**: 50 % of decode iterations with max KV ≥ 120k held out (in addition to the hash), so long-KV decode is tested, not fitted.
- **Mixed prefill/decode**: 20 % hash hold-out in each of the three mixed classes (clean n = 9 / 13 / 9 in the recovery capture; 4 / 4 / 6 in quadrant).
- **16k chunks at different KV offsets**: whole chains 3, 9, 12 (the 131k request), 21 held out → offsets 0–115k are covered by held-out samples;
  inside calibration chains the K₀ ∈ [47k, 52k) band is held out as an interpolation test. Recovery capture: 88 clean chunks → 55 calibration
  (K₀ ≈ 0k: 22, 16k: 14, 33k: 9, 65k: 6, 82k: 2, 98k: 2) and 33 held out (0k: 4, 16k: 4, 33k: 4, 49k: 11, 65k: 4, 82k: 3, 98k: 2, 115k: 1).
- **MoE routing distributions**: the corpus requests ranked by expert-load imbalance (`out/routing_stats.json`, `imb_mean` 29–47, distinct experts
  per layer 187–336): the 5 most and 5 least imbalanced requests are held out for the statistical expert-count model; iteration-level routing cannot be
  held out (not captured per iteration, gap G3).
- **End-to-end**: the two workload passes (quadrant_workload, diverse_workload; request-level TTFT/TPOT, no nsys) are held out entirely for the
  scheduler+timing replay.

Error metrics:
- per kernel group and class: median and p90 of |pred − meas| / meas on the slowest-rank group time; absolute p90 in ms for groups < 1 ms.
- per complete GPU iteration cycle: error on the slowest-rank wall (not Σ ranks), reported per class, plus the decode duty-cycle decomposition
  (busy / idle / overlap) error.
- long-context: slope error (ms per 1k KV) and residual at each held-out K₀.
- end-to-end: TTFT and TPOT p50/p90 relative error per trace, and per-request TTFT error vs input length bucket.
- Proposed acceptance (to be confirmed by Kevin): decode step ≤ 5 % median per class; 16k chunk ≤ 5 % at every held-out offset; eager mixed
  ≤ 15 % (host model); per-group ≤ 10 % median; end-to-end TPOT p50 ≤ 10 %, TTFT p50 ≤ 15 %.

## 5. Measurement gaps (identified, not resolved here)
- **G1 chunk length × KV**: the KV slope is measured only at L ≈ 16k; prefill chunks of 256 < L < 16k at deep KV occur only in multi-chunk
  mixed iterations (confounded). Needed for traces with small tails or non-16k chunking.
- **G2 decode batches 17–24 and ≥ 25 at long KV**: quadrant covers b 24–48 at KV < 32k only; diverse covers ≤ 16 at long KV. Decode is flat in
  KV at b ≤ 16; the assumption carries to b ≥ 17 without measurement.
- **G3 per-iteration MoE routing**: expert routes exist only as a per-token corpus from a separate pass (accepted tokens, no prompt tokens);
  the grouped-GEMM cost vs distinct-expert count is a statistical model; per-expert load effects inside the fused kernel are not identifiable.
- **G4 rank-2 host slowness and eager per-op cost**: the GPU/API trace shows the symptom (5–10 % longer host window on rank 2, ~125 µs per
  launch framework time) but not the cause; CPU-side sampling (py-spy/perf) would be needed.
- **G5 collective payloads**: byte counts are not in the trace; inferred from model geometry (hidden 5120 × tokens × bf16).
- **G6 CUDA-graph stream structure**: the 1.4–2.3 ms multi-stream overlap in decode is measured in aggregate; the stream assignment of kernel
  groups must be extracted from `streamId` in the frozen sqlite during implementation (CPU-only).
- **G7 engram CPU-offload lookups**: `_engram_lookup_kernel` (UVA reads) 1.7–1.9 ms per layer-call in prefill, 2.8–3.4 ms total; its dependence
  on host memory placement is not identifiable from the trace.
- Uncaptured by design: first 10 s of the diverse trace (requests 0–8); their shapes recur later.

## 6. Expected implementation changes (for approval; nothing implemented)
- New L4 arch `deepseek_v41_vllm` variant for GB300 TP4 (TP-sharded experts, chunk 16384, FLASHINFER_MLA_SPARSE_DSV41 + MXFP4 indexer, cute-DSL
  mHC all-reduce / oneshot / twoshot by batch, NCCL ring for ≥ 10k-token chunks, engram CPU offload) — `simulator/src/arch/deepseek_v41_vllm.rs`
  extended with a GB300 path selection rather than a copy; kernel kinds stay model-name-free.
- L1 kernel rows for GB300 (`sm103`/`sm100` kernels: FlashInfer blockscaled dense GEMM, DeepGEMM mqa_logits/sparse_mqa_logits/tf32 hc_prenorm/mega_mhc,
  TRT-LLM MXFP4 grouped MoE, fmhaSm100f sparse MLA, TileLang mHC pre/post) measured from the frozen captures (largest first per `gate5b.out`);
  the indexer rows keyed on (L, K) with the linear K term.
- L3 worklets: decode path variants by batch (graph buckets), explicit multi-stream `max` composition (G6), serial prefill composition, collective
  kind by batch/tokens with intrinsic cost + host-skew wait term.
- Host model: run-ahead with event sync (decode), launch-bound eager prefill (per-launch cost × launch count, slowest rank), rank-0 start skew for b ≤ 2.
- Remove the global `gpu_time_multiplier` for this preset; add explicit idle/overlap terms with measured defaults.
- Analyzer/alignment: a validation report that evaluates the split in `out/validation_split.json` with the metrics of §4 against the frozen parsed traces.
- Scheduler replay settings pinned to the capture (chunk 16384, max_num_seqs 128, graph cap 256).

## 7. Estimated GPU requirements
- Phase 1 implementation and calibration: **0 GPU-hours** (CPU-only on the frozen traces; tray14 or any CPU host).
- Phase 1 execution (separate approval), only to close the gaps that matter for the intended traces: G1 chunk-length × KV sweep (one controlled trace,
  ≈ 1 h on tray16 incl. 25 min boot); G2 decode b 17–48 at long KV (≈ 1 h); G4 CPU-side sampling of the eager path (≈ 0.5 h, optional); end-to-end
  validation replays without nsys (2 × 0.5 h). Total ≈ **3–4 tray-hours on one 4-GPU GB300 tray (≈ 12–16 GPU-hours)**, each run behind a
  reservation pod and an explicit go; no changes to the model configuration or public presets.
