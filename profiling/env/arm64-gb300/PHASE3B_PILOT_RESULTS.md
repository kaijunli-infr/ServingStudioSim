# Phase 3B GPU correctness pilot (P1–P4) — GB300, Llama 3.1 8B TP1 kernels

Date: 2026-11-11.

**Where:** rack 3 tray12, GPU 1 only (`GPU-9574ced0-46be-d5b7-0098-01fa1e84dccb`).

**Code and environment:**
- Code at `897bfb65` on `gb300-support` (profiling worktree, clean).
- Serving venv: torch 2.13.0+cu130, vLLM fork `e1423a4d`, FlashInfer 0.7.0, Triton 3.7.1, cuBLASLt 13.1.1 (130101).
- Driver 610.57.04, Nsight Systems 2026.3.2.

**What was written:** no profile database, simulator or preset change. Scripts and
small artifacts are in `serving-validation/pilot3b/`. Full run directory:
`$GB300_ROOT/serving-validation/phase3b0/pilot/run-20261111-112503/`.

## 1. Outcome

- **All 32 cases completed** in the approved rerun. Every correctness oracle
  passed: before timing, after the first graph replay, after the timed replays,
  and again in the Nsight pass.
- **Nothing failed or stopped.**
- **The first attempt** (11:14:31–11:15:01 UTC) was stopped by its own disk
  guard. It had started with an empty CUDA driver cache. See §8.

| part | cases | result |
| --- | --- | --- |
| P1 | at 8 and 256 tokens: the 1-addend norm (diagnostic), the exact embedding + first norm, post / next-input / final norms, SiLU-and-mul, RoPE, KV append (runner layout and serving layout) | 18 / 18 pass |
| P2 | decode batch 8, batch 4, and batch 1 under three `max_seq_len` settings, plus batch 8 with serving's graph `max_seq_len` | 6 / 6 pass |
| P3 | prefill, one request, 256 new tokens | 1 / 1 pass |
| P4 | timing of all of the above, plus the five `torch_linear` GEMMs at m = 8 | 32 / 32 timed, 5 / 5 GEMM oracles pass |

## 2. How it was measured

**Construction.** Each case is built with the reviewed runner code: its
validators, block builders (`_build_block`, `compile_like_vllm`), workload
builders (`build_decode`, `build_prefill`, the KV-append helpers) and checks.
The exceptions are the named diagnostics:
- `embed_norm_exact` (the served first piece);
- `kv_append_served` (packed cache and strided V);
- `*_served_graph_msl` / `*_served_eager_msl` (serving's `max_seq_len`);
- the KV-append oracle, which the `vllm_cuda` path lacks. The pilot checks the
  appended rows bitwise.

**Timing methods**, all CUPTI kernel time:

| method | how | warmup / repetitions |
| --- | --- | --- |
| cold eager | `Timer.cupti` exactly as the runner calls it: a 64 MiB L2 flush before every launch, 100 ms probe, 2 s recapture on drift > 2 %; plus 30 single cold launches for per-kernel records | warmup 0, or 5 for KV append; probe 10–1,000 launches |
| warm eager | `Timer.cupti(clear_l2=False)`; plus one burst of 50 back-to-back launches | 5 warmup launches |
| graph replay | the call captured once in a CUDA graph; `Timer.cupti(graph.replay, clear_l2=False)`; plus a burst of 50 replays | 3 warmup calls before capture, 5 replays before the burst |

**Per-call figures** come from the burst records:
- each kernel's median duration;
- for an operation of several kernels: the summed time, the busy union and the
  span from first start to last end.

**Variability**, as the coefficient of variation of the operation's union time:

| method | median | maximum |
| --- | --- | --- |
| cold eager | 2.7 % | 5.8 % |
| warm eager | 2.8 % | 9.5 % |
| graph replay | 0.8 % | 7.2 % |

**Trace reference.** The TP1 smoke capture provides 183 decode-8 iterations
(5,856 calls per layer role), 3 decode-4 iterations, 3 prefill-256 iterations,
and 3 mixed iterations with one decode request. For each kernel the reference
is its median duration and its exclusive time (the part no other kernel
overlaps). For each operation it is the union time.

**Kernel identity** comes from a separate Nsight pass. It ran 3 eager calls and
3 graph replays per case in NVTX ranges, and recorded name, grid, block,
registers, dynamic shared memory and cluster shape. Generated Inductor sources
were compared with the served `torch_compile_cache`.

## 3. Kernel identity and launch geometry against the serving trace

| operation | pilot kernel (eager = graph) | grid / block / regs / smem / cluster vs trace | source vs served |
| --- | --- | --- | --- |
| first norm, exact embedding block | `triton_red_fused__to_copy_add_embedding_…` | all equal (8 or 256 × 1, 256 threads, 26 regs) | **body identical** to `…embedding_rms_norm_0` |
| first norm, 1-addend standalone | `triton_red_fused__to_copy_add_mean_…` | grid, block equal; 25 regs (26 served) | **different operation**: no token-id gather, no residual store (similarity 0.79) |
| post-attention norm (2 addends) | same family | all equal (256 threads, 32 regs) | identical except eps (1e-6 runner, 1e-5 Llama); tuning hint 8,192 vs 2,048 tokens |
| next-input norm (3 addends, sum + normed) | same family | **block 256 vs served 512**; smem 32 vs 64 | identical except eps; tuning hint 8,192 vs 2,048 → **different launch config** |
| final norm (3 addends, normed) | same family | grid, block, smem equal; 30 regs (32) | same loads and stores; served writes in place, pilot to a new buffer |
| SiLU-and-mul | `triton_poi_fused_mul_silu_slice_0` | all equal (112 or 3,584 × 128) | **body identical** |
| RoPE | `triton_poi_fused_0` (one combo kernel) | all equal (40 or 1,280 × 128, 44 regs) | same combo kernel; Q and K halves in the opposite order; tuning hint differs |
| KV append, runner layout | `vllm::reshape_and_cache_flash_kernel<bf16,bf16,0>` | name and geometry all equal | strides differ from serving (below) |
| KV append, serving layout | same kernel | all equal | strides as served |
| decode attention, b8 / b4 | fill + `fmhaSm100f…MultiCtasKvCga…Q8Kv256…` | grid (2, 8, 8) / (4, 8, 4) as traced; block, regs, smem, cluster equal | **page specialization P32 vs served P16** |
| prefill attention, 256 | fill + `fmhaSm103a…Q256Kv128PersistentContext` | grid (1, 32, 1) and the rest equal | **P32 vs served P16** |
| GEMMs at m = 8 | nvjet kernels as traced (QKV `64x8_64x16_4x1`, O and down split-K + `splitKreduce`, up/gate `192x8_64x8_2x1`, LM head `128x8_64x12_4x1`) | all equal | — |

**Launch counts.** Each standalone operation launched exactly the kernels the
trace shows:
- one kernel for norms, SiLU, RoPE and KV append;
- fill + FMHA for attention;
- main kernel + `splitKreduce` for O and down.

The sequence was the same in eager and graph replay, and stable across
repetitions.

**KV-append strides** (T = 8):

| | key | value | each cache view `[blocks, page, heads, D]` |
| --- | --- | --- | --- |
| runner | contiguous (1024, 128, 1) | contiguous (1024, 128, 1) | (32768, 128, 2048, 1): separate K and V planes |
| serving | contiguous (1024, 128, 1) | **strided slice of QKV** (6144, 128, 1) | **(32768, 256, 4096, 1)**: packed (block, head, page, 2D), K and V interleaved |

## 4. Batch-1 decode split count (actual grids)

| `max_seq_len` | source | FMHA grid | cold FMHA µs |
| --- | --- | --- | --- |
| max(568, 512 × 152) = 77,824 | runner | **(19, 8, 1)** | 6.98 |
| 8,192 (`max_model_len`) | serving CUDA graph (captured with `max_model_len`) | **(16, 8, 1)** | 6.88 |
| 568 (actual length) | serving eager step | **(2, 8, 1)**, same as the trace's mixed iterations | 6.91 |

**Outcome:**
- The three split counts differ: 19 in the runner, 16 under serving's CUDA
  graph, 2 in an eager step.
- At a 568-token context the kernel time does not depend on the split count
  (spread 1.5 %).
- Batch 8 and batch 4 give the same grid with either `max_seq_len`.
- Batch 1 at long contexts was not tested.
- The runner was not changed.

## 5. Cold eager, warm eager and graph replay against the trace

Per-call medians in µs, Δ against the trace. Notes on reading the table:
- Single-kernel operations: the kernel time.
- GEMMs with split-K: the operation's union time.
- Attention: FMHA and fill are listed separately, as the trace's roles are.
- Trace references: decode-8 for T = 8 and the GEMMs; prefill-256 for T = 256;
  decode-4; the mixed iterations' decode kernel for batch 1.

| operation | trace | cold eager | warm eager | graph replay |
| --- | --- | --- | --- | --- |
| first norm, 1-addend standalone, T8 | 5.63 | 2.56 (−54 %) | 2.21 (−61 %) | 1.89 (−66 %) |
| first norm, exact embedding block, T8 | 5.63 | 4.38 (−22 %) | 2.96 (−47 %) | 2.59 (−54 %) |
| post norm, T8 | 3.20 | 2.72 (−15 %) | 2.35 (−26 %) | 2.02 (−37 %) |
| next-input norm, T8 | 2.08 | 3.79 (+82 %) | 2.88 (+38 %) | 2.62 (+26 %) |
| final norm, T8 | 1.95 | 2.14 (+10 %) | 1.92 (−2 %) | 1.66 (−15 %) |
| SiLU-and-mul, T8 | 1.38 | 2.14 (+56 %) | 1.79 (+30 %) | 1.47 (+7 %) |
| RoPE, T8 | 1.76 | 2.37 (+34 %) | 1.98 (+13 %) | 1.76 (0 %) |
| KV append, runner layout, T8 | 2.21 | 2.80 (+27 %) | 1.95 (−12 %) | 1.82 (−17 %) |
| KV append, serving layout, T8 | 2.21 | 2.98 (+35 %) | 1.97 (−11 %) | 1.86 (−16 %) |
| first norm, 1-addend standalone, T256 | 5.66 | 3.36 (−41 %) | 2.69 (−52 %) | 2.56 (−55 %) |
| first norm, exact embedding block, T256 | 5.66 | 4.80 (−15 %) | 3.90 (−31 %) | 3.47 (−39 %) |
| post norm, T256 | 3.58 | 3.52 (−2 %) | 3.04 (−15 %) | 2.82 (−21 %) |
| next-input norm, T256 | 2.72 | 4.51 (+66 %) | 3.50 (+29 %) | 3.33 (+22 %) |
| final norm, T256 | 2.94 | 3.30 (+12 %) | 2.85 (−3 %) | 2.62 (−11 %) |
| SiLU-and-mul, T256 | 4.35 | 5.36 (+23 %) | 4.70 (+8 %) | 4.48 (+3 %) |
| RoPE, T256 | 3.01 | 3.26 (+8 %) | 2.99 (0 %) | 2.85 (−5 %) |
| KV append, runner layout, T256 | 4.02 | 3.65 (−9 %) | 2.82 (−30 %) | 2.62 (−35 %) |
| KV append, serving layout, T256 | 4.02 | 3.81 (−5 %) | 2.67 (−34 %) | 2.62 (−35 %) |
| decode b8 FMHA | 8.90 | 8.80 (−1 %) | 7.01 (−21 %) | 7.10 (−20 %) |
| decode b8 fill | 0.83 | 1.42 (+71 %) | 1.41 (+69 %) | 0.99 (+19 %) |
| decode b8 operation (fill ∪ FMHA) | 9.73 (sum of roles) | 10.16 (+4 %) | 8.40 (−14 %) | 8.13 (−16 %) |
| decode b4 FMHA | 7.46 | 7.36 (−1 %) | 6.05 (−19 %) | 6.21 (−17 %) |
| decode b1 FMHA (runner, 19 splits) | 7.14 (eager, 2 splits) | 6.98 (−2 %) | 6.14 (−14 %) | 6.08 (−15 %) |
| prefill 256 FMHA | 9.34 | 9.82 (+5 %) | 8.90 (−5 %) | 9.02 (−3 %) |
| GEMM QKV (6144 × 4096) | 10.24 | 10.72 (+5 %) | 6.88 (−33 %) | 6.72 (−34 %) |
| GEMM O (4096 × 4096, split-K, op union) | 9.50 | 9.92 (+4 %) | 6.85 (−28 %) | 6.56 (−31 %) |
| GEMM up/gate (28672 × 4096) | 37.95 | 37.23 (−2 %) | 38.21 (+1 %) | 37.79 (0 %) |
| GEMM down (4096 × 14336, split-K, op union) | 22.05 | 22.29 (+1 %) | 22.05 (0 %) | 21.76 (−1 %) |
| GEMM LM head (128256 × 4096) | 152.42 | 150.94 (−1 %) | 151.86 (0 %) | 151.28 (−1 %) |

**Reading.** The trace is per-kernel time inside vLLM's CUDA graph, in an
iteration where each kernel's input was just written by the previous kernel and
the weights and KV pages rotate through L2 across 32 layers. No single
measurement method reproduces that for every family:

1. **Large-operand kernels** (every GEMM, decode and prefill FMHA): cold eager
   is within −2 % to +5 % of the trace.
   - Warm and graph timing under-read QKV and O by about 30 %, because a
     33–50 MB weight stays in L2 across back-to-back launches. Serving never
     sees that.
   - FMHA is under-read by about 20 % for the same reason: the KV pages stay
     hot.
2. **Small activation-to-activation kernels** (SiLU, RoPE): cold eager
   over-reads by 34–56 % at 8 tokens. Graph replay is within +7 % at both
   sizes, warm eager within +30 %.
3. **Norms:** mixed.
   - **Post norm:** cold eager matches.
   - **Final norm:** warm eager matches.
   - **Next-input norm:** over-reads by 26–82 % under every method. Its
     launch configuration differs from serving (§3), so the timing method is
     not the cause.
4. **First norm:**
   - The standalone 1-addend norm under-reads the served operation by 41–66 %
     under every method. It is not a substitute.
   - The exact embedding block reproduces the served kernel bit for bit in
     source and geometry, yet still reads 15–22 % below the trace when cold.
     Its gathered table rows are random and cold in both cases.
   - The likely remaining difference is that the served kernel is the first
     kernel of every graph replay. This was not measured separately.
5. **KV append:** neither method fits both sizes. Cold over-reads at 8 tokens
   (+27 % / +35 %) and is close at 256 (−9 % / −5 %); warm under-reads at 256.
   - The serving layout is 4–6 % slower than the runner layout cold, and
     equal warm.
6. **The attention fill kernel** (about 0.8 µs) is inflated by 70 % under any
   eager method, because the flush and launch dominate a one-CTA kernel. Graph
   replay is +19 %.

None of these differences is used as a correction factor. No profiling row was
changed.

## 6. Revised row counts and profiling requirements

Corrections to the Phase 3B-0 matrix:

1. **The first norm (68 rows) is invalid as specified.**
   - The arch's (1 addend, store sum, store normed) config is rejected by the
     runner (`store_sum needs at least two addends`); confirmed on CPU before
     the GPU run.
   - The accepted (1 addend, normed) row is a different operation from the
     served one (§3, §5).
   - **Requirement:** a fused embedding + RMSNorm kind is needed, with token
     ids, table, residual output and normed output. The pilot's
     `embed_norm_exact` block reproduces the served kernel and could be its
     reference.
   - This is a new kind and needs design approval. The arch's `first_norm`
     leaf must change with it, which is a simulator change.
2. **multi_add_rms_norm: 3 configs × 68 = 204 rows** (post, next-input,
   final), not 272.
   - The next-input config must first match serving's launch configuration.
     `compile_like_vllm` tunes at 8,192 tokens; serving tuned at its
     `max_num_batched_tokens` of 2,048, and on GB300 that selects a different
     block for the 3-addend norm.
   - **Requirement:** the runner's tuning size becomes 2,048 for this
     deployment, or a recorded parameter. That is a runner change.
3. **kv_cache_append HND (70 rows)** must use serving's layout.
   - Layout: the packed `(blocks, heads, page, 2·D)` cache with V a strided
     slice of the QKV output.
   - Its key must not collide with the existing NHD/HND rows. For example, a
     distinct layout value such as `HND_packed`, or a new backend; this needs
     a design decision.
   - Same kernel, 4–6 % slower when cold.
4. **SiLU-and-mul 68 and RoPE 68: unchanged.** Rows should be timed with graph
   replay (§7).
5. **Attention:** 548 decode + 1,011 prefill rows, unchanged in number.
   - The runner should run the served P16 specialization. Its `_PAGE_SIZE = 32`
     cites "the kernel time measured the same for 16, 32 and 64", which was not
     measured on GB300 bf16.
   - The batch-1 split count follows `max_seq_len`; it did not matter at 568
     tokens.
6. **GEMMs:** no new rows. The existing `torch_linear` rows are cold-eager
   rows, the method that matches here (§5). They were measured in the main
   environment; this pilot re-timed the five shapes in the serving environment
   with the same nvjet kernels.

**Revised totals.** Norms 204 + SiLU 68 + KV append 70 + RoPE 68 = 410
rows of existing kinds, plus 68 rows of the new embedding + norm kind.

The earlier "410" was 272 norm rows + 68 SiLU + 70 KV append, with RoPE's 68
counted separately. The total is equal only by coincidence: the invalid
first-norm rows drop out and RoPE moves in.
Attention: 1,559 rows. Every non-GEMM group needs a runner or kind change first
(§7).

## 7. Recommendation per kernel family

| family | recommendation | basis |
| --- | --- | --- |
| GEMMs, `torch_linear` | **A**: existing cold eager profiling is suitable | same kernels and geometry; cold within −2 to +5 %; warm and graph −30 % on 33–50 MB weights |
| TRT-LLM attention, decode and prefill | **C, then A**: runner change to page size 16, then cold eager | geometry equal; P32 vs served P16; cold FMHA within −1 to +5 %, warm and graph about −20 % |
| SiLU-and-mul (`vllm_inductor`) | **B**: graph-replay timing | identical source and geometry; cold +23 to +56 %, graph +3 to +7 % |
| RoPE (`gqa_rope_neox`) | **B**: graph-replay timing | same combo kernel and geometry (Q/K order swapped); graph 0 / −5 %, cold +8 to +34 % |
| multi_add_rms_norm (`vllm_inductor`) | **C**: tune at serving's token count (2,048), then re-pilot the timing method | the next-input config launches a different block; post matches cold, final matches warm |
| first norm | **C**: new fused embedding + RMSNorm kind | the standalone norm is a different operation (−41 to −66 %); the exact block reproduces the served kernel |
| KV append (`vllm_cuda`) | **C**: serving layout (packed cache, strided V), then re-pilot the timing method | same kernel and geometry; strides differ; no method fits both 8 and 256 tokens |

## 8. GPU occupancy, cache and disk

| item | value |
| --- | --- |
| first attempt (stopped) | about 30 s, 11:14:31–11:15:01 UTC |
| rerun: timing pass | 117.7 s (11:25:05–11:27:03) |
| rerun: Nsight identity pass | 62.4 s (11:27:03–11:28:05) |
| **aggregate** | **210.1 s of 600** (rerun 180.1 s of its 540 s cap) |
| peak GPU 1 memory | 3.2 GiB (nvidia-smi, 1 s samples); torch reserved peak 3.87 GiB (timing pass) |
| private CUDA driver cache | copy of `cache/nv-compute`: initial 482,765,540 B (38 files), final 627,368,052 B (40 files), **+144,602,512 B written**; deleted at the end |
| source cache | file listing (path, size, mtime) identical before and after |
| other new artifacts | 50 MB (traces 3 MB, Inductor and Triton output, logs); Nsight report 1 MB |

**Cache cap.** The guard measured the private cache in MiB (`du -sm`) with a
600 MiB limit, and `CUDA_CACHE_MAXSIZE` was 600 MiB. The final 627,368,052 B is
598.3 MiB, under that limit, but 627 MB in decimal units, so it exceeds 600 MB
read decimally. The guard therefore did not stop. This is reported, not
resolved: if the cap was meant in decimal megabytes, the rerun went over it by
27 MB. The cache was deleted.

**First attempt.** An empty `CUDA_CACHE_PATH` made the driver re-JIT torch and
vLLM kernels into the run folder (274 MB in about 25 s). That breached the
300 MB artifact cap, and the guard stopped the run. Three cases had passed.
That 274 MB was deleted before the rerun.

## 9. Isolation and cleanup evidence

**Before launch** (11:25:05):
- GPU 1: expected UUID, 0 %, 91 MiB, no compute process.
- Serving venv package fingerprint equal to the committed one.
- DB checksums: `3ec95dcd…` and `9474a224…`.
- CUPTI extension built with no device visible.

**During the run:**
- Every pilot process ran with `CUDA_VISIBLE_DEVICES=1` and confirmed the UUID
  in-process.
- The guard is a child subreaper, so passes and their descendants stay in its
  tree.
- Every 2 s it checked for a foreign process on GPU 1 and for a pilot process
  on another GPU; neither occurred.

**Other GPUs:**
- GPU 2 (SGLang): 250,457 MiB throughout, processes unchanged.
- GPU 3 hosts another user's SGLang server, started between the attempts. It
  was recorded only; its memory varied (245,668 → 247,622 MiB) with its own
  load.

**After the run:**
- GPU 1 at 89 MiB with no process.
- No pilot process left.
- Serving fingerprint unchanged; both databases unchanged.
- Only the private cache was deleted.

**Guard fixes before the rerun**, all tested on CPU (`pilot3b/test_guard.sh`,
5 of 5 pass: normal, timeout, external stop, orphaned child, cache-budget
stop):
1. Occupancy is now accounted from launch until the pass and every descendant
   are gone, in every exit path. The first attempt's stop path had skipped it.
2. A `kill 0` hazard was removed: an unset helper PID expanded to `kill 0`,
   which signals the whole process group.
3. The guard became a child subreaper, so an orphaned descendant is still
   counted and stopped.

## 10. New implementation and measurement confounds

- **Inductor tuning size.**
  - Phase 3A's `compile_like_vllm` docstring states that tuning at 2,048 and at
    8,192 tokens selects the same configs (measured on B200).
  - On GB300 that is false for the 3-addend norm.
  - It held for every other Inductor kernel piloted here, which had the same
    geometry as served.
- **Attention page size.** P32 in the runner against P16 in serving.
- **The trace as a reference.**
  - Only 3 iterations at prefill-256 and decode-4.
  - At 8 tokens, the GEMM roles' trace times include serving's PDL/graph
    environment.
  - The batch-1 reference comes from eager mixed iterations, not a graph.
- **Eps constant.** The runner's eps is 1e-6, the model's 1e-5. It changes no
  traffic.
- **Fill kernel.** At sub-µs, all eager methods inflate it.
- **Nsight pass.** It ran compiled kernels from the timing pass's caches. Its
  launches are the ones that ran, not new compilations.

## 11. Commits

- **This phase:** the report and `serving-validation/pilot3b/`. The scripts are
  `pilot_driver.py`, `run_pilot.sh`, `test_guard.sh`, `pilot_analyze.py`,
  `trace_reference.py` and `triton_sources.py`. Also the preflight result, the
  guard-test log, the first attempt's step log and the rerun's small artifacts
  (`timing.json`, `identity.json`, `analysis.json`, logs and fingerprints).
- **Committed as run:** `pilot_driver.py` and `run_pilot.sh` are committed
  byte-identical to the files that ran, matching the sha256 values in
  `pilot3b/run/commits.txt`. They are not reformatted: the repo's ruff config
  reports style findings in them (pyflakes and syntax checks are clean).
- **Not changed:** runners, kinds, simulator and presets.

Every item in §6 and §7 needs a separate decision. So do any further GPU
execution and any database integration.
