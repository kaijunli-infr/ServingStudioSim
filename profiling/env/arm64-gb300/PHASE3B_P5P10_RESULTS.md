# Phase 3B measurement-validation pilot P5–P10 — GB300 (stopped)

Date: 2026-11-11. Rack 3 tray12, GPU 1 only (`GPU-9574ced0-46be-d5b7-0098-01fa1e84dccb`).

**Code:** scripts and runners at `1242d2cb` on `gb300-support`, run from the
clean profiling worktree.
- Driver `pilot3b2/pilot_driver2.py`, sha256 `abdb2cf0…`.
- Guard `pilot3b2/run_pilot2.sh`, sha256 `88a89445…`.
- Both are committed byte-identical to what ran (`run/commits.txt`).

**Environment:** serving venv (torch 2.13.0+cu130, vLLM fork `e1423a4d`,
FlashInfer 0.7.0), driver 610.57.04, Nsight Systems 2026.3.2.

**Not done:** no database write, profiling fill, preset change or serving run.

## 1. Outcome

**The run stopped 47 s into the timing pass:**
- The artifact guard measured 311,090,739 B against the 300,000,000 B limit.
- Per the approval, it was **not retried**.
- GPU occupancy was **50.3 s of 600 s**. The Nsight identity pass never ran.

| part | status | detail |
| --- | --- | --- |
| P5 RMSNorm | **partial**: 15/15 timing cases ran, all oracles pass; no Nsight identity | Launch width from Inductor's autotune record matches serving for all four norm roles (§4.1). Both tune-8192 controls are **invalid**: they reused the 2,048 compile (§4.2). |
| P6 fused embedding + norm | **partial**: 6/6 cases plus the placement diagnostic, all oracles pass; no Nsight identity | §4.3 |
| P7 packed KV append | **incomplete**: 3/7 recorded (1, 8, 64 tokens), all whole-cache bit-exact | Results of later sizes were lost at the stop (§3.3). |
| P8 P16 attention | **not run** | stopped while building the first case |
| P9 SiLU/RoPE graph backends | **not run** | |
| P10 attention accounting | **not run** | The serving-trace side is extracted (§4.5). |

## 2. Stop: cause and evidence

**What tripped the guard.** The first P8 case called FlashInfer's
`trtllm_batch_decode_with_kv_cache`. FlashInfer JIT-built its trtllm-gen FMHA
module (`get_trtllm_gen_fmha_module().build_and_load()`, then ninja/nvcc)
inside the GPU window. nvcc's temporary files went to the run's local
`TMPDIR`, which this guard counts with the artifacts. It reached
266,327,464 B within about 10 s (`run/sizes.log`):

| UTC | artifacts incl. local temp (B) | private cache (B) |
| --- | --- | --- |
| 17:44:27 | 44,370,664 | 656,943,648 |
| 17:44:29 | 48,646,454 | 656,943,648 |
| 17:44:31 | 157,783,675 | 656,943,648 |
| 17:44:36 | 227,288,872 | 656,943,648 |
| 17:44:38 | **311,090,739** → stop | 656,943,648 |

**Why the module was rebuilt.**
- The pilot copies the warm FlashInfer cache into the run directory.
- Ninja's `build.ninja` holds absolute paths, so a copy at a new path is
  rebuilt.
- The Phase 3B pilot rebuilt it the same way: its `fmha_gen` objects are dated
  11:25–11:26, inside that run. Its temporary files went to the system `/tmp`,
  which that guard did not count.

So this is not a measurement fault in the cases; it is a cost that was there
before and is now counted. A traceback in `run/timing.err` shows the stop
interrupting the ninja build.

## 3. Budget, isolation and cleanup

### 3.1 Occupancy and storage

| item | value |
| --- | --- |
| GPU occupancy | 50.3 s (17:43:51–17:44:41), aggregate 50.3 s of 600 s |
| private CUDA cache | initial 482,765,540 B (38 files); **peak sampled 698,702,536 B**, under 700,000,000 B. The driver (`CUDA_CACHE_MAXSIZE` = 700,000,000) then evicted to a final 656,943,648 B (45 files): +174,178,108 B, 7 files, net. Deleted. |
| source cache | listing identical before and after |
| other artifacts | **peak 311,090,739 B, above the 300,000,000 B limit; stopped.** After cleanup: 44,834,288 B in the run directory; the 266,327,464 B of local temporary files were deleted. |
| Nsight reports | none (the identity pass did not run) |

### 3.2 Isolation

| check | evidence |
| --- | --- |
| before launch | GPU 1 at 0 %, 89 MiB, no compute process |
| GPU 1 during the run | every 2 s sample: only the pilot's own processes on GPU 1, none on another GPU |
| GPU 2 | SGLang server pids 3145113 and 3146111, 249,514 MiB, unchanged in all 21 samples; never acted on |
| GPU 3 | idle at 3 MiB; the other user's server seen in the Phase 3B pilot was gone before launch (recorded only) |
| after | GPU 1 back to 87 MiB, no process; no `pilot_driver2` or `nsys` process left |
| environment | serving-env fingerprint `43672e8a…` unchanged |
| databases | tracked `3ec95dcd…`, private `9474a224…`, unchanged |

### 3.3 Pilot-tooling defects found

1. **Lost results.** The driver rewrites its results file every 8 cases, to
   spare CephFS. The stop's SIGINT is not an `Exception`, so the final write
   did not run. P7's cases above 64 tokens ran but are not recorded.
2. **Invalid controls.** The two tune-8192 control cases reused the 2,048
   compile (§4.2), so they test nothing.

## 4. Results from the data collected

All times are per-kernel medians in µs. Δ is relative to the serving trace's
per-kernel median for the same role.

Trace classes:
- 8 tokens = decode batch 8 (183 iterations);
- 256 / 568 = single-request prefills (3 iterations each);
- 2,048 = the mixed 2,047 + 1 step (3 iterations, eager).

**What the identity evidence is.** It comes from Inductor's autotune records
and Triton's compiled-kernel metadata in the run's caches, not from Nsight, so
grid and register counts are not confirmed.

### 4.1 P5: does tuning at 2,048 reproduce serving's launch?

| norm role (stores) | source at tuning 2,048 | autotune choice | served (trace) |
| --- | --- | --- | --- |
| next-input (sum + normed) | 7 loads, 2 stores, `size_hints x=2048` | **512 threads**, R0_BLOCK 4096 | 512 threads, smem 64 |
| post-attention (normed) | 5 loads, 1 store | 256 threads, R0_BLOCK 1024 | 256 threads, smem 32 |
| final (normed) | 7 loads, 1 store | 512 threads, R0_BLOCK 4096 | 512 threads, smem 64 |
| embedding + norm (residual + normed) | 3 loads, 2 stores | single config, 256 threads | 256 threads, smem 32 |

**Yes, on this evidence.**
- Tuning at 2,048 selects the served launch width for all four roles,
  including the next-input norm. Tuned at 8,192 in the Phase 3B pilot, that
  norm launched 256 threads.
- Output-storage semantics: each case's oracle checked exactly its planes:
  next-input 2 (sum, normed), post 1, final 1.
- The sources' store counts match the served kernels' (2, 1, 1, 2).
- Confirming grid and registers needs the Nsight pass.

**Timing** (oracle pass in every case, before and after graph replay):

| case | trace | cold eager | warm eager | graph replay |
| --- | --- | --- | --- | --- |
| next-input T1 | – | 2.27 | 1.95 | 1.70 |
| next-input T8 | 2.08 | 2.20 (+6 %) | 1.97 (−5 %) | 1.70 (−18 %) |
| next-input T64 | – | 2.66 | 2.39 | 2.15 |
| next-input T256 | 2.72 | 3.38 (+24 %) | 2.98 (+10 %) | 2.81 (+3 %) |
| next-input T568 | 4.13 | 4.69 (+14 %) | 4.33 (+5 %) | 4.15 (0 %) |
| next-input T1024 | – | 7.03 | 6.34 | 6.12 |
| next-input T2048 | 11.84 | 12.79 (+8 %) | 10.81 (−9 %) | 10.52 (−11 %) |
| post T8 | 3.20 | 2.71 (−15 %) | 2.36 (−26 %) | 2.04 (−36 %) |
| post T256 | 3.58 | 3.38 (−6 %) | 2.98 (−17 %) | 2.81 (−22 %) |
| post T2048 | 9.31 | 9.07 (−3 %) | 7.16 (−23 %) | 6.97 (−25 %) |
| final T8 | 1.95 | 2.17 (+11 %) | 1.93 (−1 %) | 1.65 (−15 %) |
| final T256 | 2.94 | 3.22 (+10 %) | 2.83 (−4 %) | 2.61 (−11 %) |
| final T2048 | 10.98 | 11.98 (+9 %) | 9.46 (−14 %) | 9.29 (−15 %) |

**Variability:** per-launch CV is 1.7–7 % cold, 0.6–9 % warm and 0.3–1.3 %
graph.

**Next-input norm improvement:** at 8 tokens it went from +82 % (Phase 3B
pilot, tuned 8,192) to +6 % cold.

**Note on post T256:** its three medians equal next-input T256's to 0.01 µs.
The oracle confirms it is the post graph (one stored plane), so the equality
is a measured result, but kernel identity per case awaits Nsight.

**Preliminary policy reading** (Nsight still missing; no policy selected):
- **Cold eager** is the only method within ±15 % for all three roles at 8 and
  2,048 tokens. Its worst cases are next-input at 256 (+24 %) and final at 8
  (+11 %).
- **Graph replay** fits the next-input norm at 256–568 tokens (0 to +3 %) but
  under-reads the post norm by 22–36 %.
- **One policy per role is not supported yet.** Post fits only cold;
  next-input fits graph at mid sizes; final fits warm at small sizes.

### 4.2 P5 controls invalid: Inductor cache reuse defeats `tune_tokens`

**What happened:**
- `p5_next_tune8192_T8` and `_T2048` built in 0.12 s each (a cache hit).
- They timed within 1 % of the 2,048 cases (2.19 vs 2.20, 12.77 vs 12.79 µs
  cold).
- The run's Inductor cache holds no source with `size_hints x=8192`.

**Cause.** `compile_like_vllm` resets Dynamo, but Inductor's on-disk caches
returned the kernel compiled and autotuned at 2,048. The token dimension is
symbolic, so the cache key does not include the tuning size.

**Consequence for profiling.** In one process or cache directory, rows keyed
by different `tune_tokens` can silently share whichever compile ran first. The
Phase 3B-2 key makes such rows distinguishable but not correct.

**Required fix (a runner change, not made in this pilot):** each tuning size
needs its own Inductor cache (or the FX-graph and autotune caches off), with a
test that two tuning sizes produce different sources.

### 4.3 P6: fused embedding + first norm

| case | trace | cold eager | warm eager | graph replay |
| --- | --- | --- | --- | --- |
| T1 | – | 4.53 | 2.81 | 2.55 |
| T8 | 5.63 | 4.36 (−23 %) | 2.83 (−50 %) | 2.56 (−55 %) |
| T64 | – | 4.67 | 3.31 | 3.04 |
| T256 | 5.66 | 4.81 (−15 %) | 3.75 (−34 %) | 3.63 (−36 %) |
| T568 | 5.18 | 6.52 (+26 %) | 4.20 (−19 %) | 4.06 (−22 %) |
| T2048 | 9.95 | 9.92 (0 %) | 8.32 (−16 %) | 8.12 (−18 %) |

**Correctness:** every oracle passed (residual bit-exact with the gathered
rows, normalized rows within tolerance), including after replays.

**Placement diagnostic:**
- Graph A holds the fused kernel only.
- Graph B holds a 1,024-element in-place add first, then the fused kernel.
- Order A, B, A; kernel medians in µs.
- The diagnostic kernel is reported separately: 0.86–0.90 warm, 1.38–1.44 cold.

| tokens | A warm | B warm | A again warm | A cold | B cold | A again cold |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 2.56 | 2.85 | 2.56 | 4.45 | 4.45 | 4.45 |
| 256 | 3.49 | 3.71 | 3.49 | 4.83 | 4.82 | 4.83 |
| 2048 | 8.35 | 8.61 | 8.51 | 10.02 | 9.86 | 10.06 |

**Reading:**
- Under cold L2, first-node and later-node placement give the same kernel time
  (within 1.6 %).
- Warm, the later node is 3–11 % slower, not faster.
- So "first kernel of the replay" does not explain the trace's higher time at
  8 and 256 tokens (served 5.63 / 5.66 against at most 4.45 / 4.83 here).

**Unresolved:** the measured cold time is 15–23 % below the trace at 8–256
tokens and +26 % at 568 tokens, and no measured condition explains this. No
correction factor is fitted. Possible causes not tested here: the served token
ids (repeated, real tokens) against uniform random ids, and the serving step's
preceding input-preparation kernels.

### 4.4 P7: packed KV append (3 of 7 sizes)

| case | trace | cold eager | warm eager | graph replay |
| --- | --- | --- | --- | --- |
| T1 | – | 3.34 | 1.99 | 1.87 |
| T8 | 2.21 | 3.09 (+40 %) | 1.98 (−10 %) | 1.86 (−16 %) |
| T64 | – | 3.69 | 2.39 | 2.27 |

- Whole-cache bit-exact oracle: pass at every recorded size, before timing and
  after replays.
- The cold result at 8 tokens repeats the Phase 3B pilot's (+35 % there with
  the serving layout).
- **No decision is possible** without the larger sizes.

### 4.5 P10 (trace side only)

The fill + FMHA pairs from the TP1 trace (`prep/trace_reference2.json`),
median µs:

| class | call | fill | FMHA | union | span | gap | overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode 8 | decode | 0.83 | 8.90 | 9.60 | 9.60 | 0 | 0.16 |
| decode 7 | decode | 0.83 | 8.70 | 9.39 | 9.39 | 0 | 0.16 |
| decode 4 | decode | 0.77 | 7.46 | 8.06 | 8.06 | 0 | 0.16 |
| mixed 2,048 | decode b1 | 1.04 | 7.14 | 8.10 | 8.13 | 0 | 0.10 |
| mixed 2,048 | prefill (4 req) | 1.06 | 28.70 | 29.73 | 29.73 | 0 | 0.10 |
| mixed 1,933 | prefill (4 req) | 1.10 | 27.39 | 28.37 | 28.38 | 0 | 0.13 |
| prefill 256 | prefill | 1.09 | 9.34 | 10.43 | **43.65** | **33.22** | 0 |
| prefill 568 | prefill | 1.09 | 12.22 | 13.30 | 15.76 | 2.66 | 0 |

**Two facts the runner must be compared with:**
1. In serving, the fill overlaps the FMHA kernel by 0.10–0.16 µs (it starts
   before the fill ends). The served union is therefore about 0.1 µs below
   fill + FMHA.
2. Single-request prefills show a gap between fill and FMHA (33 µs at 256, a
   piecewise-graph boundary; 2.7 µs at 568). The union excludes it and the
   span includes it.

The measured side was not reached.

## 5. Decisions the data supports, and what remains open

| item | status |
| --- | --- |
| Norm tuning size 2,048 | supported: the served launch width for all four roles (autotune records). Nsight confirmation pending. |
| Norm timing mode | **open.** No single method fits all roles; per-role reading above is preliminary. |
| Embedding timing mode | **open.** Cold is within 0 to −23 % except +26 % at 568; the remaining difference is unresolved and not explained by graph placement. |
| KV timing mode | open (3 sizes only) |
| P16/P32, batch-1 splits | not measured |
| SiLU/RoPE graph backends | not measured; kept provisionally as approved |
| P10 | trace side only |
| New runner issue | `tune_tokens` is defeated by Inductor cache reuse (§4.2); needs a fix before any fill keyed by it |

**Revised matrix:** unchanged from Phase 3B-2 (2,037 rows, ≈ 60–115 min of GPU
time). The norm and embedding rows need the §4.2 fix first.

## 6. Proposed continuation (needs a decision; nothing run)

**Fixes, all CPU-side, with no runner change:**
1. **FlashInfer prebuild.** Build FlashInfer's `fmha_gen` module before the
   GPU window, on CPU, into the pilot's FlashInfer directory, as the CUPTI
   extension is built now.
   - It costs no GPU time, and its temporary files (about 270 MB peak) are
     measured and reported separately.
   - Alternative: leave the FlashInfer cache in place rather than copying it,
     which avoids the rebuild but writes into the shared cache.
2. **Results written after every case**, plus a write on SIGINT/SIGTERM.
3. **Controls:** the tune-8192 controls get their own Inductor and Triton cache
   directories, so they compile and tune at 8,192.

**Run:**
- P8, P9 and P10 as approved.
- P7 again at all sizes.
- The Nsight identity pass for P5–P8.
- P5 and P6 timings again only if you want them in the same run (about 40 s);
  the data above stands otherwise.
- **Expected:** 330–400 s of GPU occupancy. 549.7 s remain under the 600 s
  aggregate cap if the cap carries over; or a new cap.

**Artifact limit:** with the prebuild outside the GPU window, the run's new
artifacts are expected under 150 MB. Decide whether the prebuild's transient
temporary files count against the 300 MB limit; if they do, the prebuild needs
a limit of its own.

## 7. Commits

- `1242d2cb`: the scripts as they ran:
  - `pilot3b2/pilot_driver2.py`, `run_pilot2.sh`;
  - `test_guard2.sh` and `guard-test.log` (9/9 pass);
  - `trace_reference2.py`, `triton_sources.py`.
- The commit with this report adds:
  - `pilot3b2/pilot_analyze2.py` (CPU analysis);
  - `pilot3b2/run/`: the stopped run's logs, size samples, occupancy, cache
    record, checksums, fingerprints, GPU samples, `timing.json`,
    `analysis.json`;
  - `pilot3b2/prep/trace_reference2.json`.
- Run directory on the rack:
  `$GB300_ROOT/serving-validation/phase3b2/pilot/run-20261111-174348/`
  (44.8 MB; the private cache and local temporary files were deleted).
