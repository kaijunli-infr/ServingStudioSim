# Phase 3B measurement-validation pilot — continuation (Gates 0–2, P5 controls, P7–P10)

Date: 2026-11-11. Rack 3 tray12, GPU 1 only (`GPU-9574ced0-46be-d5b7-0098-01fa1e84dccb`).
It continues the stopped run (`PHASE3B_P5P10_RESULTS.md`), whose P5/P6 data stands.

**Scope:** no database write, profiling fill, simulator or preset change, or
serving run.

## 1. Outcome

**The continuation ran to completion with no stop and no retry.** Every case
passed its correctness checks.

**GPU occupancy:**
- this run: **125.6 s**, against its 500 s hard stop;
- **aggregate: 175.9 s of 600 s**, with the stopped run's 50.3 s.

| item | result |
| --- | --- |
| Gate 0: CPU FlashInfer prebuild | **pass** (third attempt; the first two were refused by the gate itself, §2) |
| Gate 1: result durability | **pass**: 5/5 interruption tests; guard self-test 12/12 |
| Gate 2: tuning isolation | **pass**: the 2,048 and 8,192 controls compiled independently and chose different next-input launch widths. **New: the final norm's autotune choice is not reproducible** (§4). |
| P5 tuning controls | 8/8 pass |
| P5/P6 identity (Nsight) | 6/6 pass. Next-input, post and embedding norms identical to serving in grid, block, registers and shared memory; final norm 512 threads but 30 vs 32 registers. |
| P7 packed KV append | 7/7 sizes; whole-cache bit-exact; Nsight identical to serving at 8/256/2,048 tokens |
| P8 P16 attention | 30/30 page-size cases and 4/4 split cases; P16 cubins are the served ones; grids match serving at every traced shape |
| P9 SiLU/RoPE graph backends | 14/14, each with 3 independent calls; single kernel per replay; oracles pass after replay |
| P10 attention accounting | done at all 10 traced shapes (§8) |
| FlashInfer | no rebuild: the build directory was unchanged around the first P8 case (both passes) and before vs after the run, apart from a new 0-byte lock file |

## 2. Gate 0: CPU-only FlashInfer prebuild

**What was built:** FlashInfer 0.7.0's trtllm-gen FMHA JIT module (`fmha_gen`).
This is the module both the decode and prefill calls load.

**Where:** in place, at its runtime path:
`$GB300_ROOT/serving-validation/phase3b2/cont/flashinfer/.cache/flashinfer/0.7.0/103a/cached_ops/fmha_gen/`.
- `FLASHINFER_WORKSPACE_BASE` points at it, and `FLASHINFER_CUDA_ARCH_LIST=10.3`
  (sm_103a, the arch the serving cache built for).
- It was never copied or moved.

**How the build was run:**
- Script: `pilot3b2/prebuild_flashinfer.sh` with `fi_prebuild.py`.
- Every step ran under `strace -f`, covering ninja, nvcc, cicc, ptxas and the
  rest, with `CUDA_VISIBLE_DEVICES=""`.

**Attempts:**

| attempt | result |
| --- | --- |
| `prebuild-20261111-180507` | **refused by the gate.** The build itself succeeded, but the Python helper opened `/dev/nvidiactl`, `/dev/nvidia-uvm` and **all four** `/dev/nvidia0–3`. Cause: importing tvm_ffi (FlashInfer's loader) calls `torch.cuda.is_available()`, which initializes the CUDA driver even with no visible device. |
| `prebuild-20261111-180858` | **refused.** With `TVM_FFI_DISABLE_TORCH_C_DLPACK=1`, the driver was initialized during the import of `cuda.tile` (via `flashinfer.cutile`) instead. |
| `prebuild-20261111-181348` | **accepted** (below) |

**Fix for the accepted attempt.** Unprivileged user namespaces are disabled on
tray12, so device nodes cannot be masked by a mount. Instead, every step ran
with `nogpu_shim.so` preloaded:
- the shim is built from `pilot3b2/nogpu_shim.c`;
- any `open`/`openat` of a `/dev/nvidia*` path fails with EACCES before
  reaching the kernel, in the helper and all its children, and is logged;
- `strace` then shows whether any device-node syscall happened at all.

**Evidence from the accepted attempt** (`pilot3b2/cont/gate0/prebuild-20261111-181348/`):

| check | result |
| --- | --- |
| GPU device access | **0** `/dev/nvidia*` syscalls in the build, check and load steps. The shim refused 21 attempts, all `/dev/nvidiactl` (the driver trying to start). `torch.cuda.is_initialized()` stayed False. |
| build | `fmha_gen.so` 15,645,248 B, 5 ninja steps |
| independence from driver state | `build.ninja` sha256 `b484f12f…` equals the first attempt's, which had the driver reachable |
| no rebuild on use | a second `spec.build()` (the runtime path minus the load) was a ninja no-op: every file's size and mtime unchanged |
| load | `spec.load()` succeeded with no device-node access, so it was CPU-verified, not deferred |
| temporary files | peak sampled **128,839,479 B**, against the 350,000,000 B allowance. Deleted (0 B left). |
| retained output | **33,605,987 B** (counted with the GPU run's artifacts) |
| shared caches | `$G/cache/flashinfer` and `$G/cache/nv-compute` listings identical before and after |
| capacity | 24.8 TB free on the shared filesystem, 731 GB in `/tmp` |

**Cleanup:** the two rejected attempts' build outputs (33.6 MB each) were
deleted after this report; their logs are kept. `strace` logs stay on the rack;
their hashes are in `gate0-strace-hashes.txt`.

**GPU run:**
- The guard used the prebuilt path in place and stops on any CUDA compiler
  process (nvcc, cicc, cudafe++, fatbinary, nvlink, or a non-Triton ptxas).
- None ran.
- The driver's snapshots around the first P8 case show no changed file.
- The workspace listing before vs after differs only by a new 0-byte
  `cached_ops/tmp/fmha_gen.lock`, from FlashInfer's file lock on load.

## 3. Gate 1: result durability

**Changes in `pilot_driver3.py`:**
- Each completed case is written to `cases/<seq>-<case>.json` by atomic write:
  temporary file, flush, `fsync`, rename, directory `fsync`.
- A case counts as complete only after that rename, which follows its
  correctness checks and measurements.
- `journal.jsonl` records start, done, not-run, failed and stopped events, each
  line `fsync`'d.
- SIGINT/SIGTERM are blocked during each write and delivered right after.
- On a stop signal the driver records the reason, the last completed case and
  the case in progress, writes the summary from the persisted cases, and exits
  128 + signal. The guard accounts the occupancy and cleans up its own
  processes only.

**Test (`test_persistence.sh`).** It runs the same loop with stand-in cases,
in the serving venv, and checks the signal's send time against the journal:

| scenario | proven signal position | result |
| --- | --- | --- |
| no signal | – | 5/5 persisted, done |
| SIGINT during a case | inside st_02 | st_00–01 durable; stop: last st_01, in progress st_02; exit 130 |
| SIGINT between cases | after st_00's done, before st_01's start | st_00 durable; nothing in progress |
| SIGINT during persistence (write window widened to 2 s) | inside st_00's write | st_00 durable and last completed; nothing in progress |
| SIGTERM during a case | inside st_02 | as SIGINT; exit 143 |

In every scenario:
- persisted cases form a prefix of the plan, and every case file is valid;
- no `.tmp` file is left;
- the journal's done events equal the case files.

**An earlier version of the test failed one scenario.** It pinned exact case
names, and the slower venv start moved the SIGTERM into an earlier case. The
driver's record was consistent; the test was then made timing-proof.

**Guard self-test (`test_guard3.sh`):** 12/12 pass. It covers:
- occupancy accounting with the prior 50.3 s;
- timeout, interrupt and orphaned-child handling;
- the source cache already at its cap;
- cache, artifact and temporary-file growth during a pass;
- a CUDA compiler process stopping the run, and Triton's own ptxas not stopping it.

## 4. Gate 2: P5 tuning controls in isolated caches

**Setup:**
- One process per tuning size.
- Each had its own, initially empty, `TORCHINDUCTOR_CACHE_DIR` and
  `TRITON_CACHE_DIR` under the run directory.

**Outcome by role** (Inductor sources, autotune records, cold-eager times at 8 tokens):

| | tune 2,048 | tune 8,192 |
| --- | --- | --- |
| Inductor sources compiled | 3, all `size_hints x=2048` | 3, all `size_hints x=8192` |
| shared source bodies across the two caches | none | none |
| first-case compile (build_s) | 8.49 s (compiled) | 7.32 s (compiled) |
| next-input (7 loads, 2 stores) | **512 threads**, R0_BLOCK 4096; cold T8 2.20 µs | **256 threads**, R0_BLOCK 1024; cold T8 **3.79 µs** |
| post (5 loads, 1 store) | 256 threads | 256 threads |
| final (7 loads, 1 store) | **256 threads** | 256 threads |

**What this shows:**
- **Which configuration was compiled:** each control's sources carry its own
  tuning size, so each tuned at its size.
- **Different implementations?** Partly.
  - Both sizes generate the same two Triton candidate kernels per norm (256- and
    512-thread variants). Their Triton hashes are identical across the two
    caches because the kernel bodies are identical; they were compiled
    separately in each empty cache.
  - The autotuner then selects differently: next-input gets 512 at 2,048
    (= served) and 256 at 8,192. That is the difference the Phase 3B pilot saw.
- **No reuse:** each cache started empty and its first case compiled. This
  confirms the stopped run's 0.12 s "controls" were cache hits.

**New finding: the final norm's autotune choice is not reproducible.** The
same final-norm graph at tuning 2,048 selected:
- **512 threads** in the stopped run (cold 2.17 µs at T8);
- **512 threads** in this run's identity pass (Nsight: block 512, 30 registers);
- **256 threads** in this run's control process (cold **3.07 µs**, +41 %).

Serving launched 512 (trace 1.95 µs). The two candidates are near-tied in
Inductor's autotune benchmark, so a profiling fill could record either.

## 5. P7: packed KV append (all sizes)

**Correctness:** the whole-cache bit-exact oracle passed at every size, before
timing and after replays.

**Identity:** `reshape_and_cache_flash_kernel`, grid (T, 1, 1), 512 threads,
32 registers. Identical to serving at 8, 256 and 2,048 tokens.

| tokens | trace | cold eager | warm eager | graph replay | CV (cold / warm / graph) |
| --- | --- | --- | --- | --- | --- |
| 1 | – | 3.31 | 1.97 | 1.85 | 4.7 / 4.9 / 0.8 % |
| 8 | 2.21 | 3.28 (+48 %) | 1.97 (−11 %) | 1.85 (−16 %) | 5.2 / 5.0 / 0.9 % |
| 64 | – | 3.69 | 2.39 | 2.27 | |
| 256 | 4.02 | 4.03 (0 %) | 2.76 (−31 %) | 2.63 (−35 %) | 5.9 / 3.5 / 0.6 % |
| 568 | 3.90 | 4.61 (+18 %) | 3.46 (−11 %) | 3.34 (−14 %) | |
| 1024 | – | 6.74 | 4.90 | 4.79 | |
| 2048 | 8.77 | 10.66 (+22 %) | 7.51 (−14 %) | 7.40 (−16 %) | 1.1 / 0.7 / 0.4 % |

**No single method is adequate across the range:**
- cold eager is 0 to +48 %;
- warm eager is −11 to −31 %;
- graph replay is −14 to −35 %.

**The trace reference is itself uncertain:** 568 tokens (3.90 µs) reads below
256 tokens (4.02 µs), from 3 iterations each.

## 6. P8: P16 attention (page-size and split effects kept apart)

**Identity:**
- P16 cases launch exactly the served cubins:
  - decode: `fmhaSm100fKernel_…PagedKvCausalP16MultiCtasKvCga…Q8Kv256…`;
  - prefill: `fmhaSm103aKernel_…PagedKvCausalP16VarSeqQ256Kv128PersistentContext`.
- P32 cases launch the same names with `P32`.
- Grids, blocks, registers and shared memory equal the trace at every traced
  shape:
  - decode (4,8,4) and (2,8,8);
  - prefill (1,32,1), (3,32,1) and (3,32,4);
  - eager batch-1 decode (2,8,1).

**Correctness:** on the same logical workload, P16 and P32 outputs are bitwise
equal at all 15 shapes.

**Page-size effect** (cold eager, runner split policy, same logical workload):

| shape | FMHA P16 vs P32 | union P16 vs P32 |
| --- | --- | --- |
| decode b1 568 / b4 630 / b4 569 / b7 630 / b8 600 | +2.3 / −2.6 / 0.0 / −1.5 / 0.0 % | +2.5 / −3.8 / −1.3 / −1.5 / −0.6 % |
| decode b1 4096 / 8192 / 32768 | +0.2 / 0.0 / −0.2 % | −0.8 / +0.2 / −0.8 % |
| decode b32 4096 / b128 4096 | +2.1 / +2.0 % | +2.3 / +2.0 % |
| prefill 256 / 568 / 2048 | +0.7 / +0.1 / +0.9 % | +0.6 / +0.8 / +1.0 % |
| prefill mixed 2,048 (4 requests) / mixed 1,933 | **+5.3 / +3.8 %** | +5.5 / +3.2 % |

- Under graph replay, P16 decode reads up to 7 % above P32 (b4 630:
  6.66 vs 6.24 µs).
- **Page size changes cost by up to about 5 %** and selects a different cubin,
  so P16 rows are needed for the served configuration. Existing P32 rows are
  not a substitute.

**Split effect** (P16, batch 1, cold FMHA µs):

| context | runner policy, `max_seq_len` 77,824 (19 splits) | serving FULL graph, 8,192 (16 splits) | serving eager step, own context |
| --- | --- | --- | --- |
| 568 | 7.23 | – | 7.17 (2 splits, −1 %) |
| 4,096 | 10.85 | 10.72 (−1.2 %) | **8.13** (8 splits, **−25 %**) |
| 8,192 | 13.63 | **12.90 (−5.4 %)** | = graph policy |
| 32,768 | 27.06 | – (`max_model_len` would be ≥ 32,768: 19 splits) | – |

Graph-replay times show the same pattern: at 4,096, 9.04 runner vs 6.72 eager
policy.

**Split selection materially changes cost:** −25 % at 4,096 tokens under an
eager step's policy, −5 % at 8,192 under the FULL-graph policy.

**Proposal (not implemented; the key is unchanged during the pilot):**
- Give `paged_attn_batch_decode` an optional key field for the split bound,
  e.g. `max_seq_len` (absent = today's saturating rule).
- The opt-in arch would set it per step type: `max_model_len` for FULL-graph
  decode steps, the step's longest context for mixed (eager-attention) steps.
- That arch side is a simulator cost-model change and needs approval.

## 7. P9: SiLU and RoPE graph backends

Each size was timed by the method comparison, then by three independent calls
of the `vllm_inductor_graph` runner's two steps (recompile + `time_graph_replay`).

| case | trace | cold | warm | graph | 3 calls (µs) | between-call / within-call spread |
| --- | --- | --- | --- | --- | --- | --- |
| SiLU 8 | 1.38 | +67 % | +29 % | **+6 %** | 1.47 / 1.44 / 1.44 | 2.2 / 2.2 % |
| SiLU 256 | 4.35 | +24 % | +8 % | **+3 %** | 4.48 ×3 | 0 / 0 % |
| SiLU 568 | 8.90 | +7 % | −7 % | −9 % | 8.11 / 8.10 / 8.10 | 0.2 / 0.4 % |
| SiLU 1024 | – | 15.14 | 13.98 | 13.77 | 13.57 / 13.76 / 13.76 | 1.4 / 0.5 % |
| SiLU 2048 | 29.60 | −4 % | 0 % | **0 %** | 29.41 / 29.34 / 29.36 | 0.2 / 0.1 % |
| SiLU 4096 / 8192 | – | 55.5 / 109.1 | 56.4 / 110.0 | 56.4 / 110.0 | stable | ≤ 0.1 % |
| RoPE 8 | 1.76 | +30 % | +13 % | **−1 %** | 1.73 ×3 | 0 / 0 % |
| RoPE 256 | 3.01 | +9 % | 0 % | −5 % | 2.85 / 2.85 / 2.82 | 1.1 / 1.1 % |
| RoPE 568 | 5.12 | **+1 %** | −7 % | −11 % | 4.54 / 4.58 / 4.54 | 0.7 / 0 % |
| RoPE 1024 | – | 7.61 | 7.11 | 6.92 | 6.91 ×3 | 0 / 0.5 % |
| RoPE 2048 | 14.08 | **−5 %** | −13 % | −14 % | 12.13 ×3 | 0 / 0.3 % |
| RoPE 4096 / 8192 | – | 25.3 / 50.8 | 24.0 / 52.0 | 23.9 / 51.8 | stable | ≤ 0.5 % |

**Stability:**
- Every call replayed exactly one kernel per replay (`triton_poi_fused_mul_silu_slice_0`,
  `triton_poi_fused_0`); identity equals serving in grid, block and registers.
- Oracles passed after the first replay and after the timed replays.
- Between-call spread ≤ 2.2 %, within-call batch spread ≤ 2.2 %.

**Recommendations, each independent:**
- **SiLU: retain `vllm_inductor_graph`.**
  - Within +6 to −9 % at every traced size and 0 % at 2,048.
  - Cold eager reads +24 to +67 % at the decode sizes.
- **RoPE: retain `vllm_inductor_graph`, with a recorded caveat.**
  - Graph is the best method at decode sizes (−1 % at 8, −5 % at 256), where
    RoPE runs every step.
  - It under-reads the prefill sizes by 11–14 %, where cold eager fits
    (+1 / −5 %). In the trace those prefills ran piecewise or eager, not from
    one replayed graph.
  - Neither method fits all sizes, and the key cannot carry a per-size method.
    The prefill under-read is unresolved.

## 8. P10: attention accounting (like for like)

**Columns** (median µs; all 10 traced shapes, P16):
- **trace:** fill / FMHA / union / span / gap / overlap of the served fill +
  FMHA pairs (`prep/trace_reference2.json`, 96 to 5,856 pairs per shape);
- **cold eager:** the runner's method; fill / FMHA / union, Δ against the trace;
- **graph:** fill / FMHA / union / span / gap / overlap;
- **hybrid:** graph-replay fill + cold FMHA, a candidate shown for comparison.

| shape | trace fill / FMHA / union / span / gap / overlap | cold fill / FMHA / union | graph fill / FMHA / union / span / gap / overlap | hybrid |
| --- | --- | --- | --- | --- |
| decode b1 568 (runner split) | 1.04 / 7.14 / 8.10 / 8.13 / 0 / 0.10 | 1.31 / 7.23 (+1.3 %) / 8.54 (+5.5 %) | 0.90 / 6.27 / 7.18 (−11.3 %) / 7.23 / 0.03 / 0 | 8.13 (+0.4 %) |
| decode b1 568 (eager split, as traced) | 1.04 / 7.14 / 8.10 / 8.13 / 0 / 0.10 | 1.33 / 7.17 (+0.4 %) / 8.46 (+4.5 %) | 0.96 / 6.34 / 7.26 (−10.3 %) / 7.42 / 0.16 / 0 | 8.13 (+0.4 %) |
| decode b4 630 | 0.77 / 7.46 / 8.06 / 8.06 / 0 / 0.16 | 1.18 / 7.23 (−3.0 %) / 8.53 (+5.8 %) | 0.90 / 6.66 / 7.55 (−6.3 %) / 7.71 / 0.16 / 0 | 8.13 (+0.8 %) |
| decode b4 569 (mixed 1,933) | 1.12 / 7.71 / 8.67 / 8.67 / 0 / 0.16 | 1.34 / 7.10 (−7.9 %) / 8.46 (−2.4 %) | 0.93 / 6.53 / 7.46 (−14.0 %) / 7.49 / 0.03 / 0 | 8.03 (−7.4 %) |
| decode b7 630 | 0.83 / 8.70 / 9.39 / 9.39 / 0 / 0.16 | 1.54 / 8.35 (−4.0 %) / 9.71 (+3.4 %) | 0.96 / 7.26 / 8.22 (−12.4 %) / 8.38 / 0.16 / 0 | 9.31 (−0.9 %) |
| decode b8 600 | 0.83 / 8.90 / 9.60 / 9.60 / 0 / 0.16 | 1.41 / 8.74 (−1.8 %) / 10.13 (+5.5 %) | 0.99 / 7.36 / 8.35 (−13.0 %) / 8.38 / 0.03 / 0 | 9.73 (+1.3 %) |
| prefill 256 | 1.09 / 9.34 / 10.43 / **43.65 / 33.22** / 0 | 1.33 / 9.70 (+3.8 %) / 11.10 (+6.4 %) | 0.90 / 9.06 / 9.95 (−4.6 %) / 10.14 / 0.19 / 0 | 10.59 (+1.5 %) |
| prefill 568 | 1.09 / 12.22 / 13.30 / 15.76 / 2.66 / 0 | 1.31 / 12.80 (+4.7 %) / 14.22 (+7.0 %) | 0.93 / 12.13 / 13.06 (−1.8 %) / 13.22 / 0.16 / 0 | 13.73 (+3.2 %) |
| prefill mixed 2,048 (4 req) | 1.06 / 28.70 / 29.73 / 29.73 / 0 / 0.10 | 1.31 / 29.12 (+1.4 %) / 30.54 (+2.7 %) | 0.93 / 27.14 / 27.94 (−6.0 %) / 27.94 / 0 / 0.13 | 30.05 (+1.1 %) |
| prefill mixed 1,933 (4 req, one with a 343-token prefix) | 1.10 / 27.39 / 28.37 / 28.38 / 0 / 0.13 | 1.33 / 31.31 (**+14.3 %**) / 32.59 (**+14.9 %**) | 0.93 / 29.34 / 30.14 (+6.3 %) / 30.14 / 0 / 0.13 | 32.24 (+13.6 %) |

**Cold-eager span is not shown:** it is 210–310 µs because the host launches
the fill and the FMHA kernel separately. It is a host-side quantity, not a
GPU one.

**Reading:**
1. **The same boundary everywhere: union.**
   - The simulator's attention leaf is one serial `Sum` term, and union is what
     a serially charged leaf should hold.
   - The trace's span adds only GPU idle: the 33 µs piecewise-graph boundary at
     prefill 256, 2.7 µs at prefill 568, about 0 elsewhere.
   - The arch deliberately leaves that idle to serving-execution terms.
2. **Cold-eager union is within −2.4 to +7.0 % of the served union at 9 of 10
   shapes.**
   - The fill is visibly inflated: cold 1.18–1.54 µs vs served 0.77–1.12 µs
     (+0.2 to +0.7 µs per call).
   - The served fill overlaps the FMHA kernel by 0.10–0.16 µs; the runner's
     overlap is 0.
   - Together these explain the excess at decode shapes, where cold FMHA alone
     is −8 to +1 % against the trace.
   - **The fill inflation is not hidden:** cold union is +3.4 to +5.8 % at the
     decode shapes, against FMHA alone at −1.8 to −4.0 %.
3. **Graph replay under-reads** the decode union by 6–14 %, as the Phase 3B
   pilot found for FMHA alone.
4. **The hybrid** (graph fill + cold FMHA) is within −0.9 to +3.2 % at 8 of 10
   shapes. It is not implemented; it is a candidate for a separate decision.
5. **Unresolved: the mixed 1,933-token prefill reads +14 to +15 %** under cold
   eager (+6 % graph). The request set matches the trace (3 × 568 new tokens
   plus 225 new on a 343-token prefix). The served kernel may benefit from
   that prefix's pages having been written in the previous step and still being
   in L2; this was not tested.

**Conclusion:**
- Cold-eager union is suitable for the serial leaf, with a known bias of about
  +3 to +7 %, mostly fill inflation, and one unresolved shape.
- A hybrid measurement would reduce the bias to about ±3 %. That would be a
  runner and method change, proposed separately here.

## 9. Timing-method recommendations

**Status labels:**
- **confirmed:** identity checked by Nsight, the method within about ±10 % of
  the trace at all traced sizes, and the result reproduced;
- **provisional:** fits, but with a recorded caveat;
- **unresolved:** no method fits.

| family | recommendation | status | evidence |
| --- | --- | --- | --- |
| GEMMs | cold eager (existing) | confirmed earlier (Phase 3B pilot) | – |
| next-input norm (512 threads at tune 2,048) | **warm eager** | confirmed | −5 / +10 / +5 / −9 % at 8 / 256 / 568 / 2,048. Cold +6 / +24 / +14 / +8 %, graph −18 / +3 / 0 / −11 %. Reproduced in the control (2.20 / 1.97 / 1.70 µs at T8 in both runs). |
| post-attention norm | **cold eager** | confirmed | −15 / −6 / −3 %; warm −17 to −26 %, graph −22 to −36 %. Reproduced (2.71 µs at T8 in both). |
| final norm | **unresolved** | autotune choice not reproducible (§4) | With 512 threads: cold +9 to +11 %, warm −1 / −4 / −14 %. With 256: cold +57 % at T8. The config must be pinned first. |
| fused embedding + norm | **unresolved** | identity confirmed (Nsight, all fields) | Cold −23 / −15 / +26 / 0 % at 8 / 256 / 568 / 2,048; first- vs later-node placement explains nothing (stopped run). No correction factor fitted. |
| KV append, packed | **unresolved** (no method adequate) | identity confirmed | Cold 0 to +48 %, warm −11 to −31 %, graph −14 to −35 %; the trace reference is non-monotonic |
| decode / prefill attention, P16 | **cold-eager union (existing)** | provisional | −2.4 to +7.0 % at 9/10 shapes; mixed 1,933 +14.9 % unresolved. Split policy needs a key (§6). Hybrid proposal in §8. |
| SiLU | **graph replay** (`vllm_inductor_graph`, retain) | confirmed | +6 / +3 / −9 / 0 %; spread ≤ 2.2 % |
| RoPE | **graph replay** (retain) | provisional | −1 / −5 % at decode sizes; −11 / −14 % at prefill sizes (unresolved) |

**Matrix:** still 2,037 candidate rows (≈ 60–115 min of GPU time, Phase 3B-2
§5). Three families cannot be filled until their gates close:
- the norm and embedding rows need the `tune_tokens` cache fix and final-norm
  config pinning (§10);
- KV append has no adequate method;
- decode attention needs the split decision.

## 10. Production `tune_tokens` cache-correctness fix (plan; mandatory before a norm or embedding fill)

**Problems:**
1. **Cache reuse.** Inductor's on-disk caches key the compiled graph without
   the tuning size, because the token dimension is symbolic. A compile at one
   `tune_tokens` is silently reused for another (the stopped run).
2. **Unstable autotune choice.** Near-tied candidates can be chosen differently
   from run to run (the final norm, §4).

**Plan** (runner change, CPU-testable, then a GPU check under its own approval):
1. In `compile_like_vllm`, give each `tune_tokens` value its own Inductor and
   Triton cache directory: `<base>/tune<N>/`. Equivalently, include the tuning
   size in Inductor's FX-graph cache key via `torch._inductor.config`, if the
   pinned torch supports it.
2. Record, per profiled row, the selected launch config:
   - the autotune `best_config` (XBLOCK, R0_BLOCK, num_warps);
   - the generated kernel's source hash.
   These go into the row's run metadata.
3. **Pin rather than re-tune.** For the opt-in GB300 vLLM variant, select the
   config the serving deployment's own Inductor cache chose; it is available in
   the TP1 smoke run's `torch_compile_cache` (the served kernels' launch
   widths: 512 for next-input and final, 256 for post and embedding). Fail the
   row if the compiled kernel's launch width differs.
4. **Tests:**
   - two tuning sizes in one process produce different sources and
     autotune records;
   - a re-run selects the pinned config;
   - a mismatched config fails the row instead of recording it.

## 11. Budget, storage, isolation and cleanup

**GPU occupancy:**

| pass | seconds |
| --- | --- |
| stopped run (2026-11-11 17:43:51–17:44:41) | 50.3 |
| timing (P7–P9) | 67.2 |
| control 2,048 | 21.0 |
| control 8,192 | 14.0 |
| identity (Nsight) | 23.4 |
| **this run** | **125.6** (hard stop 500) |
| **aggregate** | **175.9 of 600** |

**Storage** (decimal bytes, sampled every 2 s, 53 samples):

| item | value |
| --- | --- |
| private CUDA cache | initial 482,765,540 (38 files); **peak 679,395,352** (cap 700,000,000; driver limit 680,000,000); final 664,484,280 (39 files), +181,718,740 net; deleted |
| source CUDA cache | listing identical before and after |
| GPU-run artifacts, including the 33,605,987 B prebuilt module | **peak 84,998,676** (cap 300,000,000); 49,796,211 at the end |
| Nsight report | 1,286,682 (cap 200,000,000); sqlite export 3,325,952 |
| local temporary files | 0 at the end; deleted |
| CPU prebuild (accepted) | temporary peak 128,839,479 (allowance 350,000,000); retained 33,605,987 |

**Peak memory:** GPU 1 6,331 MiB (nvidia-smi); torch reserved 5.18 GiB
(timing) and 6.62 GiB (identity).

**Isolation:**
- **before:** GPU 1 at 0 %, 87 MiB, no compute process.
- **during:** no foreign process on GPU 1 and no pilot process on another GPU
  in any 2 s sample. GPU 2's SGLang server (pids 3145113 and 3146111,
  249,514 MiB) was unchanged in all 53 samples and never acted on. GPU 3 was
  idle (3 MiB).
- **after:** GPU 1 at 81 MiB with no process; no pilot, nsys or compiler
  process left.

**Integrity checks:**
- serving-env fingerprint `43672e8a…` unchanged;
- tracked `profile.db` `3ec95dcd…` and private `9474a224…` unchanged;
- prebuilt FlashInfer workspace unchanged except the lock file.

## 12. Scripts, hashes, commits

**Gate 0 scripts** (`pilot3b2/`):
- `prebuild_flashinfer.sh`: script sha256 `ec67e568…` as run;
- `fi_prebuild.py`: `91b49729…`;
- `nogpu_shim.c`: `fcee1d41…`.

**Gate 1 / continuation scripts** (worktree `df637ab7`, clean):
- `pilot_driver3.py`: `2ac35650…`;
- `run_pilot3.sh`: `1e77f602…`;
- `test_guard3.sh`, `test_persistence.sh`.

**Analysis:** `pilot_analyze3.py` (CPU).

**Commits on `gb300-support`:**

| commit | content |
| --- | --- |
| `616718d5` | Gate 0 scripts |
| `19dbb799` | tvm_ffi switch (second attempt) |
| `bb119b59` | GPU-node shim (accepted attempt) |
| `df637ab7` | continuation driver, guard and self-tests, as run |
| (this report) | report, analysis script, small artifacts in `pilot3b2/cont/` (Gate 0 evidence, self-test logs, run logs, per-pass summaries and journals, `analysis3.json`) |

**On the rack:**
- run directory `$GB300_ROOT/serving-validation/phase3b2/cont/run-20261111-182631/`;
- prebuilt workspace `…/phase3b2/cont/flashinfer/` (kept: 33.6 MB, the
  pilot's private FlashInfer cache).

**Next decisions** (each needs approval):
- the `tune_tokens` cache fix and final-norm pinning;
- a decode-attention split-bound key, and its arch use;
- the attention hybrid measurement;
- the KV and embedding methods;
- any further GPU run, fill, database integration or prediction validation.
