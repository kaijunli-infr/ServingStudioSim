# Phase 3B-2 — runner, profiling-schema and architecture corrections (CPU only)

Date: 2026-11-11. Branch `gb300-support`.

**Scope:** CPU-only code and tests, following the accepted GPU correctness
pilot (`PHASE3B_PILOT_RESULTS.md`).

**Not done:**
- no GPU execution;
- no write to either profiling database;
- no public preset change;
- no TP1/TP2/TP4 validation;
- no prediction-error claim.

**Where it ran:** tests and checks on rack 3 tray12 with
`CUDA_VISIBLE_DEVICES=` (CPU only). Small artifacts are in
`serving-validation/phase3b2/`. Full outputs:
`$GB300_ROOT/serving-validation/phase3b2/`.

## 1. Outcome

| item | result |
| --- | --- |
| Pilot deviation | Recorded as an experiment deviation in `PHASE3B_PILOT_RESULTS.md` §12; §1–§11 unchanged. |
| A. Fused embedding + first RMSNorm | New kind `embedding_rms_norm`, backend `vllm_inductor`. The arch's first norm now uses it; the 1-addend, sum-stored config is gone. |
| B. Remaining RMSNorms | `multi_add_rms_norm` gains optional keys `tune_tokens` and `rms_eps`. The arch sets 2,048 and 1e-5. Existing rows keep their keys. |
| C. KV append | New layout `HND_PACKED` and optional key `value_row_stride`, `vllm_cuda` only, with a bit-exact oracle over the whole cache. |
| D. TRT-LLM attention | Optional key `page_size` (absent = 32) on both kinds. The arch sets 16. Qwen3.5 and B200 configs unchanged. |
| E. Graph-replay timing | New `profiling.profilers.graph_replay` and backends `vllm_inductor_graph` for `silu_and_mul` and `gqa_rope_neox`. Eager backends unchanged. |
| F. Revised matrix | 2,037 candidate rows, confirmed by a dry run against the tracked DB (§5). GPU pilot proposed in §6. |
| G. Gates | All pass (§4). Rust 1,604 pass. pytest 5,072 pass, plus 8 known environmental failures. Presets 9 pass. 186,297 stored keys unchanged. Predictions identical. |

## 2. Recorded deviation from the pilot

The private CUDA cache ended at 627,368,052 B. That is 598.3 MiB, under the
guard's MiB limit, but **627.4 MB, 27.4 MB over the approved 600 MB**.

- The pilot report now records this as a deviation in its §12. §1–§11 are
  unchanged.
- The pilot is not described as budget-compliant anywhere.
- Correction for later runs: measure bytes against decimal limits (§6.4).

## 3. What changed

### 3.0 Schema primitive: `optional_arg(absent=...)`

`profiling/db/doc.py` and `profiling/db/args.py`.

**Before:** an optional key field could only mean "equal to another field"
(`same_as`), as `kv_cache_append.head_dim_v` uses.

**Now:** `optional_arg(absent=<constant>)` declares a field whose absence means
the value every existing row was measured with.
- Unset, or spelled out equal to that constant, the field stays out of
  `args_hash`, so existing rows keep their keys.
- Any other value gets its own key.
- `canonical_args` handles both forms. A field must declare exactly one of
  the two.
- Documented in `profiling/README.md`.

### A. `embedding_rms_norm` (new kind)

**What it measures:** the served first-piece kernel,
`triton_red_fused__to_copy_embedding_rms_norm`. It:
1. reads an int32 token id;
2. gathers that row of the embedding table;
3. stores the row as the residual;
4. stores its RMSNorm.

**Files:**
- kind `profiling/kernels/embedding_rms_norm.py`;
- runner `profiling/runners/norm/embedding_rms_norm_vllm_inductor.py`;
- reference `profiling/runners/norm/embedding_rms_norm_reference.py`;
- Rust `simulator/src/timing/kernels/embedding_rms_norm.rs`.

**Block:** vLLM's own code, `F.embedding(ids.long(), table)` (the
unquantized `VocabParallelEmbedding` path) followed by the native `rms_norm`
IR op. vLLM's `LlamaDecoderLayer.forward` uses the embedding as layer 0's
residual; the CPU check confirmed this in the fork's source.

**Compile path:** compiled by `compile_like_vllm`, as the other Inductor
kinds, at `tune_tokens`.

**Key:** `num_tokens, hidden_dim, vocab_size, dtype, rms_eps, tune_tokens`.
All are required, since the kind is new.
- Rust holds the epsilon as `rms_eps_bits: u64`, as `qnorm_rope_kv_insert`
  does.
- Its grid is the same 68-point token axis as the norms.

**Oracle:** an independent reference (`index_select` plus an fp32 norm,
sharing no code with the block).
- The residual must equal the gathered rows bit for bit.
- The normed rows are checked at 2e-2.
- A test shows the check rejects the standalone-norm substitute (contiguous
  rows, no gather).

**Arch change (`llama3_dense_vllm`):** `first_norm` is now
`Op<EmbeddingRmsNormKernel>`, with Llama's `vocab` (128,256), the
checkpoint's `rms_norm_eps` (new `ModelSpec::rms_norm_eps()`) and the arch's
`max_num_batched_tokens`. The previous `multi_add_rms_norm` (1 addend, store
sum) leaf, which the runner rejected, is removed. A test asserts that no
1-addend `multi_add_rms_norm` leaf remains.

### B. Remaining RMSNorms (`multi_add_rms_norm`)

**New optional keys:**
- `tune_tokens` (absent = 8,192, the existing `TUNE_TOKENS`);
- `rms_eps` (absent = 1e-6, the runner's former constant).

**Runner and compile path:**
- The runner passes `tune_tokens` to `compile_like_vllm(..., tune_tokens=)`.
  Its first call runs at `max(tune_tokens, num_tokens)`.
- It builds the block with `rms_eps`.
- The block is now assembled by `block_from_ops`, so its output-plane wiring
  can be tested with any implementation of vLLM's op contracts.

**Docs:** `compile_like_vllm`'s docstring no longer says the tuning size is
not a deployment knob. It records the GB300 3-addend case, and notes that
"tuning at 2,048 gives block 512" is not yet measured.

**Arch:** a new selector param, `max_num_batched_tokens` (default 2,048, a
cache key), sets `tune_tokens`. The checkpoint sets `rms_eps` (1e-5 for
Llama 3.1). Configs are post (2 addends, normed), next-input (3, sum +
normed) and final (3, normed).

**Preserved:**
- Qwen3.5's arch sets both fields to `None`, so its keys are unchanged.
- All 340 existing `multi_add_rms_norm` rows keep their keys (§4).

**Tests:**
- Each served config stores exactly its planes and matches the reference.
- The epsilon is the block's argument.
- Legacy values keep their keys; 2,048 and 1e-5 each get their own key.
- Bad values are rejected.
- In the serving venv, `_build_block` runs over vLLM's real native ops
  (`phase3b2_vllm_cpu_check.py`): post gives 1 plane, next-input 2, final 1,
  all within one bf16 ulp.

### C. KV append, serving layout

**New layout `HND_PACKED`:** vLLM's FlashInfer cache, one
`[blocks, heads, page, 2·head_dim]` buffer whose last-dimension halves are
the K and V views.

**New optional key `value_row_stride`** (absent or 0 = contiguous values):
the values are the trailing slice of a wider row. For Llama, that row is the
QKV output, (32 + 2·8)·128 = 6,144 elements.

**Runner:** only `vllm_cuda` accepts either. The new path builds the cache
and inputs as served (`serving_kv_tensors`).
- The traced strides are asserted in a test: cache views
  `(32768, 256, 4096, 1)`, key `(1024, 128, 1)`, value `(6144, 128, 1)`.
- The non-packed layouts mirror `create_kv_caches_with_random_flash`'s
  allocation.

**Oracle (`check_append_exact`):** after one launch, every element of the
cache storage is compared with a copy written through the same strides.
Mapped slots must hold K or V, and nothing else may change. Tests show it
catches an off-slot write and a V-into-K offset bug.

**Rejections (no fallback):**
- `torch` and `vllm_triton_diffkv` reject `value_row_stride` and
  `HND_PACKED`.
- `vllm_cuda` rejects a stride that does not exceed one value row, and an
  FP8 cache on the serving path.

**Keys:** NHD, HND and HND_PACKED, each with or without a stride, give six
distinct keys. A stride of 0 keys like unset. The existing NHD/HND path of
`vllm_cuda` is byte-for-byte the same code path as before.

**Arch:** `cache_layout = "HND_PACKED"`, `value_row_stride = 6144`.

### D. TRT-LLM attention, page size

**New optional key `page_size`** (absent = 32) on `paged_attn_batch_decode`
and `paged_attn_batch_prefill`.
- The runner lays out its pages and page table by it.
- It accepts only 16, 32 and 64 (the trtllm-gen cubins) and raises
  `ProfilerNotImplemented` otherwise.
- The kinds' old caveat ("16, 32 and 64 measured within 1 %, so not part of
  the key") is replaced: that measurement was FP8 and was not repeated for
  bf16 on GB300.

**Arch:** `llama3_dense_vllm` sets `page_size = 16`, the served
`P16` cubin.

**Preserved:** the Qwen3.5 worklet and every other caller pass `None`, so
their keys and payloads are unchanged. All 701 decode and 1,080 prefill rows
keep their keys.

**Representation of the remaining dimensions:**

| dimension | how it is keyed | status |
| --- | --- | --- |
| page size | `page_size` | explicit, 16 for the variant |
| dtype | `q_dtype`, `kv_dtype`, `o_dtype` | explicit; a mixed pair is rejected |
| grid selection | by FlashInfer from head geometry, dtype and page size, all keyed | determined by the key |
| split (decode) | `max_seq_len = max(max_kv_len, 512 · SMs)` (`decode_max_seq_len`) | a documented function of the key and the GPU, tested |

**Open split difference:** the split rule still differs from serving's FULL
graph at batch 1 (19 splits against 16; `max_model_len` 8,192). It did not
matter at a 568-token context. Pilot item P8 (§6) measures it at long
contexts before deciding whether a key is needed.

### E. Graph-replay timing (opt-in)

**Module `profiling/profilers/graph_replay.py`, `time_graph_replay(fn, check)`.**
All counts and conditions are fixed constants:
1. 3 eager calls on a side stream, then one CUPTI capture of one eager call.
   This records the call's kernel sequence.
2. Capture into a CUDA graph with its own memory pool. Inputs are the
   caller's, allocated and seeded before capture.
3. One replay, then `check(outputs)`.
4. 20 warm-up replays, then 3 CUPTI captures of 100 back-to-back replays.
   There is no L2 flush.
   - Each replay must launch exactly the eager call's kernels, in order.
   - A missing launch, an extra kernel (e.g. a memcpy) or another order raises
     `GraphReplayError` instead of being timed.
   - Per replay: the sum of its kernels' CUPTI durations (device kernel time,
     not the gaps between kernels).
   - Result: the median of the batch medians.
5. `check(outputs)` again after the timed replays.

**Provenance:** the method id is
`cupti.cuda_graph_replay.warm_l2.eager3.warm20.batches3x100`.

**Backends:** `vllm_inductor_graph` for `silu_and_mul` and `gqa_rope_neox`.
- They have the same args schema and table as `vllm_inductor`, so their rows
  carry a different `backend` in the key and in provenance. Eager and graph
  rows can never share a key.
- Both runners were refactored so that the eager and graph functions share
  one compiled case (`_compiled_case`). The eager path's sequence (compile,
  first call, check, `Timer.cupti`) is unchanged.

**Arch:** `llama3_dense_vllm` now selects `vllm_inductor_graph` for its
activation and RoPE leaves. That is the pilot's recommendation B. It is a
one-line constant, `INDUCTOR_GRAPH_BACKENDS`, and is pending confirmation by
pilot item P9.

### Arch summary (`llama3_dense_vllm`, GB300)

| leaf | kind / backend | key fields new in 3B-2 |
| --- | --- | --- |
| first_norm | embedding_rms_norm / vllm_inductor | vocab 128,256; rms_eps 1e-5; tune_tokens 2,048 |
| post, next-input, final norm | multi_add_rms_norm / vllm_inductor | tune_tokens 2,048; rms_eps 1e-5 |
| rope | gqa_rope_neox / **vllm_inductor_graph** | backend |
| activation | silu_and_mul / **vllm_inductor_graph** | backend |
| kv_cache_append | kv_cache_append / vllm_cuda | HND_PACKED; value_row_stride 6,144 |
| prefill / decode attention | paged_attn_batch_* / flashinfer_trtllm_gen | page_size 16 |
| 5 GEMMs | single_gemm / torch_linear | none |

The Phase 3A mapping data (`serving-validation/vllm_llama_mapping.py`) was
updated to the same leaves. Its 25 tests pass, including a new one for the
norms' tuning size and eps.

## 4. Preservation and regression gates (G)

| gate | how | result |
| --- | --- | --- |
| Existing profiling keys and rows unchanged | `phase3b2_key_check.py`, read-only: rebuilds every stored row's args through the new schemas and compares the recomputed `args_hash` with the stored one. | Tracked DB: **186,297 rows in 100 tables, 0 mismatches, 0 errors**. Private GB300 DB: 186,259 rows, 0 failures. |
| Databases not modified | sha256 before and after | tracked `3ec95dcd…` (working file = HEAD blob); private `9474a224…` |
| Existing arch predictions unchanged | `regress.sh` at `df14792a` (clean) and on the final tree; 11 timing-predict presets (Llama B200/H200/GB300, Qwen3 235B iter/attn/ffn, AFD attn/ffn, PD, GLM-5.2 NVFP4 B200 EP4, DeepSeek V4); `pred_diff.py` | 10 runnable predictions: **identical** except run identity (PIDs, timestamps, prediction IDs, analyzer elapsed). `qwen36_local` fails identically before and after (needs a GPU). |
| Existing kernel configs unchanged | `simulator cost-trees --kernel-configs` on Qwen3.5 (B200, GB300), Llama 3 dense (GB300, H200) and Llama 3 dense TP (GB300) | the JSON is equal |
| New arch charges each operation once | Rust `every_operation_is_charged_exactly_once_per_layer_or_iteration`, plus the new first-norm test | first norm 1 (one `embedding_rms_norm` leaf); 10 layer leaves ×32; next-input ×31; final and LM head ×1; no embedding leaf; no 1-addend norm |
| New configs generate | `cost-trees` and `timing-predict --dry-run` on the new arch, GB300 and B200 | rc 0; GB300: **2,037 / 2,377 rows missing** (the GEMMs' 340 present); B200 builds (2,303 missing; not to be profiled) |
| No silent fallback | runner and Rust tests | every new leaf names one backend. Unsupported page size, stride, layout or dtype pairs raise (`ValueError` or `ProfilerNotImplemented`). Graph replay raises on any kernel-sequence mismatch. |
| No profiling-method collisions | registry plus key tests | eager and graph are separate backends (registry enforces unique kind:backend). P16 ≠ P32, 2,048 ≠ 8,192, 1e-5 ≠ 1e-6, and HND_PACKED ≠ HND ≠ NHD all give distinct keys. Legacy values spelled out key as before. |
| Unit tests | Rust `cargo test -p simulator --lib`; pytest CPU tier `-n 16`; presets tier | Rust **1,604 pass**, 0 fail. pytest **5,072 pass**, 8 fail: the same 8 environmental failures as untouched upstream on this host (CPU-only torch). Presets **9 pass**. |
| vLLM-native CPU check | `phase3b2_vllm_cpu_check.py` in the serving venv (torch 2.13, vLLM `e1423a4d`), no GPU | embedding residual bit-exact; normed within 0.031 (one bf16 ulp); norm planes 1/2/1 as served; Llama source check pass |

**Not run:** `rustfmt`. It is not installed on tray12, and the laptop has no
Rust toolchain. The Rust changes follow the surrounding style but are not
machine-formatted. Python is ruff-formatted and ruff-clean.

## 5. Revised profiling matrix (F)

The rows are the dry run's missing specs for `llama3_dense_vllm` on GB300
(`serving-validation/phase3b2/dry-run-GB300.json`), against the tracked DB.

| family | rows | kind / backend | key (beyond geometry and dtype) | timing mode | correctness oracle | est. GPU |
| --- | --- | --- | --- | --- | --- | --- |
| RMSNorm post / next-input / final | 3 × 68 = **204** | multi_add_rms_norm / vllm_inductor | tune_tokens 2,048; rms_eps 1e-5; output set | cold eager CUPTI (existing); **timing mode to be confirmed by P5** | fp32 reference per stored plane (sum 1e-2, normed 2e-2) | ≈ 4–6 min |
| Fused embedding + RMSNorm | **68** | embedding_rms_norm / vllm_inductor | vocab 128,256; rms_eps; tune_tokens | cold eager CUPTI; **to be confirmed by P6** | residual bit-exact; normed 2e-2 against the index_select reference | ≈ 2–3 min |
| SiLU-and-mul | **68** | silu_and_mul / vllm_inductor_graph | backend | graph replay, warm L2 | fp32 reference 2e-2, after the first replay and after the timed replays | ≈ 2 min |
| RoPE | **68** | gqa_rope_neox / vllm_inductor_graph | backend | graph replay, warm L2 | fp32 reference with vLLM's cos/sin cache 2e-2, before and after | ≈ 2 min |
| KV append, packed | **70** | kv_cache_append / vllm_cuda | HND_PACKED; value_row_stride 6,144 | cold eager CUPTI (existing `vllm_cuda` method); **to be confirmed by P7** | whole-cache bit-exact | ≈ 1 min |
| Decode attention | **548** | paged_attn_batch_decode / flashinfer_trtllm_gen | page_size 16 | cold eager CUPTI, union of fill + FMHA (existing) | fp32 reference on sampled rows (abs 0.05, rel-L2 0.06) | ≈ 15–30 min |
| Prefill attention | **1,011** | paged_attn_batch_prefill / flashinfer_trtllm_gen | page_size 16 | as decode | as decode | ≈ 35–70 min |
| **total** | **2,037** | | | | | **≈ 60–115 min** |

**GEMMs:** 340 rows (5 × 68) are present, cold-eager `torch_linear` rows
(pilot recommendation A); no new GEMM rows.

**Basis for the time estimates:**
- Phase 3B-0's per-row costs for the same runners: about 1.3 s per Inductor
  row (compile plus capture), 1.5–4 s per attention row (KV allocation and
  initialization dominate).
- Graph replay adds 323 replays per row, under 0.1 s for these kernels.

**Disk:** under 100 MB for Inductor caches and logs. A new DB holds about 0.5
MB of rows. Private-DB writes would need their own approval.

**Timing modes still open:** the families marked "to be confirmed" have no
mode yet that fits the trace at every size (pilot §5). The pilot in §6
decides them before any fill. If a different mode is chosen, it becomes its
own backend, as graph replay did; no row is retimed under an existing key.

## 6. Proposed follow-up GPU pilot (needs approval)

**Purpose:** decide the remaining timing modes and confirm the corrected
configurations on GB300 before any fill. No DB writes: runner functions are
called directly, and metrics go to JSON.

### 6.1 Cases

| part | cases | measures | decides |
| --- | --- | --- | --- |
| P5 next-input norm (and post/final) at `tune_tokens` 2,048 | T = 1, 8, 64, 256, 1024, 2048; the 3-addend sum-stored config, plus post and final at T = 8, 256 | Nsight geometry (expect block 512, smem 64, as served); cold, warm and graph timing; vs trace at T8 and T256 | whether tuning at 2,048 reproduces the served kernel; norm timing mode |
| P6 fused embedding + norm, through the new runner | T = 1, 8, 64, 256, 2048 | identity vs served `…embedding_rms_norm_0`; cold, warm and graph; vs trace 5.63 / 5.66 µs. Diagnostic: graph replay with a preceding 64 MiB L2-displacing kernel inside the same graph, to test the "first kernel of the replay" explanation | embedding timing mode |
| P7 KV append, packed layout, through the new runner | T = 1, 8, 64, 256, 1024, 2048 (2048 = the traced prefill chunk) | whole-cache oracle; identity; cold, warm and graph; vs trace T8, T256 and the 2048-chunk iterations | KV timing mode |
| P8 P16 attention | decode: b1, b4, b8, b32, b128 at contexts 568 / 4,096 / 32,768 (subset, ≤ 12 shapes); prefill: 1 × 256, 1 × 1024, 1 × 2048 new tokens | P16 kernel name and geometry; P16 vs P32 cold time at the same shapes. b1 at 8,192 and 32,768 with the runner's bound (19 splits) vs `max_model_len` 8,192 (16 splits) | P16 rows; whether the split needs a key |
| P9 graph stability, SiLU and RoPE | T = 8, 256, 1024, 2048, 4096, 8192; the new `vllm_inductor_graph` functions, 3 independent calls each | spread across calls and across the 3 internal batches; graph vs cold vs warm; vs trace at 8, 256, 2048 | keep `vllm_inductor_graph` in the arch |
| P10 attention serial-model check | traced shapes: decode b8 (600 / 4,789), b4, b1 (568), prefill 256 and 2048 | (a) runner cold union of fill + FMHA; (b) graph-replay kernel sum; (c) graph span including the fill→FMHA gap. Compared with the trace's per-layer fill + FMHA exclusive time and span, ×32 per iteration | whether a cold-eager union per leaf is the right quantity for a serial `Sum` (see below) |

**The P10 question.** The simulator charges attention as one leaf in a serial
`Sum`, priced by the runner's CUPTI interval union of the counter fill and
the FMHA kernel. In the trace both run back to back inside the graph:
- the fill is about 0.8 µs and is inflated about 70 % by any eager
  measurement;
- the gap between fill and FMHA is GPU idle, which the arch deliberately does
  not charge.

P10 measures, at every traced shape, whether union(fill, FMHA), cold, equals
the trace's per-layer exclusive time within the ±5 % seen for FMHA alone. If
it does not, it measures how much of the difference is the fill's inflation
and how much is the gap. The outcome is either "keep cold union" or a
proposal (fill from graph replay, FMHA cold) for a separate decision.

**Trace reference for 2,048 tokens:** a CPU-only addition to
`trace_reference.py` extracts a prefill-2048 class from the existing TP1
smoke trace. No new serving run is needed.

### 6.2 Cost and limits

| item | proposal |
| --- | --- |
| where | rack 3 tray12, GPU 1 only (`GPU-9574ced0-46be-d5b7-0098-01fa1e84dccb`); GPU 2 and its SGLang server untouched; GPU 3 (another user's server) only recorded |
| GPU occupancy | cap **600 s**, counted start to stop across every pass, interrupted ones included; expected 5–8 min (timing ≈ 4–5 min, Nsight ≈ 1.5 min) |
| passes | 1 timing pass, 1 Nsight identity pass; stop on timeout, no retry |
| private CUDA cache | a private copy of `cache/nv-compute` (482.8 MB). The previous run grew it by 144.6 MB to 627.4 MB; if this run JITs the same kernels, a 600 MB decimal cap would likely stop it. **Proposed cap: 700 MB decimal**, measured as bytes (`du -sb`), with `CUDA_CACHE_MAXSIZE` = 700,000,000. Alternative: keep 600 MB and let the driver evict. Delete only the private copy. Report initial, final and new bytes. |
| other artifacts | ≤ 300 MB (decimal, bytes); each Nsight report ≤ 200 MB |
| guard | `pilot3b/run_pilot.sh` with byte-based decimal limits (the only guard change), subreaper and ownership tracking, finally-path occupancy accounting; CPU self-test first |
| stop on | any oracle failure, unexpected device use, a foreign process on GPU 1, a resource-limit breach |
| not included | DB writes, fills, simulator or preset changes, TP2/TP4, prediction-error claims |

### 6.3 Deliverables

1. Per family: identity against the trace (P5–P8).
2. Cold, warm and graph timings against the trace.
3. A timing-mode decision for norms, the embedding norm and KV append.
4. P16 against P32.
5. Whether the split matters at long contexts.
6. Graph stability for SiLU and RoPE.
7. The P10 answer.
8. Occupancy, cache and disk.
9. Isolation evidence.
10. A final matrix with its timing modes, for a separate fill approval.

### 6.4 Guard change for this pilot (CPU-tested before use)

`du -sm` against MiB becomes `du -sb` against decimal bytes for every limit.
`CUDA_CACHE_MAXSIZE` is set from the decimal cap.

## 7. Decisions requested

1. **Approve the §6 pilot**, including the private-cache cap: 700 MB decimal,
   or 600 MB with driver eviction.
2. **Confirm or revert the arch's graph backends for SiLU and RoPE.** They
   are selected now (pilot recommendation B); P9 tests them at larger sizes.
3. After the pilot: a fill of the 2,037 rows into the private DB, under its
   own approval, with the timing modes the pilot decides.

## 8. Commits

- `66595c92`: the code, tests, Phase 3A mapping update and the two CPU check
  scripts (`serving-validation/phase3b2_key_check.py`,
  `serving-validation/phase3b2_vllm_cpu_check.py`).
- The commit after it: this report, the pilot report's §12 deviation, and the
  small artifacts in `serving-validation/phase3b2/` (key checks, prediction
  diff, dry-run reports, vLLM CPU check, test summary).
