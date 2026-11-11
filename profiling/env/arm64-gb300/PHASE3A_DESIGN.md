# Phase 3A: vLLM-specific Llama 3.1 8B BF16 TP1 variant, design (CPU-only), 2026-11-11

**Can the variant be built without changing the simulator's cost model? Yes.**

- It keeps the `llama3_dense` topology, its GEMM shapes and its serial sum of
  leaves.
- It only chooses other kernel kinds and backends for six operations and adds
  one RoPE leaf.
- No cost-model semantics change.

**Blocked items.** Two pieces need code that is not a cost-model change but
is outside this approval:

1. **bf16 attention.** The existing TRT-LLM attention kinds support only FP8
   query and KV, so they need a bf16 extension of their runner. Their grids
   also assume 148 SMs; GB300 has 152.
2. **RoPE.** No kind fits a plain GQA layer, so a new kernel kind is needed.

Nothing in the simulator, the profiling registry, any database or any preset
was changed. This phase added design data, tests and analysis only.

**Inputs:**

- the TP1 smoke trace and the trace audit (`TP1_SMOKE_RESULTS.md`,
  `TP1_TRACE_AUDIT.md`);
- the Inductor code vLLM compiled for that run, read from its compile cache;
- the kernel-kind sources;
- the profiling database (read only).

**Corrections to the audit:**

- `paged_attn_batch_*` is used by one existing arch (`qwen35_vllm_nvfp4_gdn_dense`),
  not several.
- The matching norm kind is `multi_add_rms_norm` / `vllm_inductor`, not
  `residual_rms_norm`, which prices vLLM's CUDA `fused_add_rms_norm` kernel.
- The GEMM gap is mostly explained (section 3).

## 1. Operation mapping

Proposed leaves per iteration of a TP1 step. Data in
`serving-validation/vllm_llama_mapping.py`.

| leaf | per iteration | kind / backend | configuration | replaces in `llama3_dense` | traced operation matches? |
| --- | --- | --- | --- | --- | --- |
| `first_norm` | 1 | `multi_add_rms_norm` / `vllm_inductor` | hidden 4096, addends 1, store sum, store normed, bf16 | `embedding` + layer 0 `input_norm` | approximately: the trace gathers the row from the embedding table, the kind loads it contiguously (same bytes, different access) |
| `qkv_proj` | 32 | `single_gemm` / `torch_linear` | n 6,144, k 4,096, bf16 | `qkv_proj` | yes |
| `rope` | 32 | **new** `gqa_rope_neox` / `vllm_inductor` | 32 Q heads, 8 KV heads, head 128, rotary 128, max position 131,072, bf16 | (nothing) | no existing kind; see below |
| `kv_cache_append` | 32 | `kv_cache_append` / `vllm_cuda` | 8 KV heads, head 128, block 16, bf16 in and cache, **HND**, tensor scale | `kv_cache_append` | yes, same kernel |
| `attention_decode` | 32 | `paged_attn_batch_decode` / `flashinfer_trtllm_gen` | 32/8 heads, head 128, q/kv/o bf16 | `decode` | same call and launch unit, **dtype not supported** |
| `attention_prefill` | 32 (zero in pure decode) | `paged_attn_batch_prefill` / `flashinfer_trtllm_gen` | as above | `prefill` | same, **dtype not supported** |
| `o_proj` | 32 | `single_gemm` / `torch_linear` | n 4,096, k 4,096 | `o_proj` | yes |
| `post_norm` | 32 | `multi_add_rms_norm` / `vllm_inductor` | addends 2, sum not stored, normed stored | `post_norm` | yes |
| `up_gate_proj` | 32 | `single_gemm` / `torch_linear` | n 28,672, k 4,096 | `up_gate_proj` | yes |
| `activation` | 32 | `silu_and_mul` / `vllm_inductor` | hidden 14,336, bf16 | `activation` | yes |
| `down_proj` | 32 | `single_gemm` / `torch_linear` | n 4,096, k 14,336 | `down_proj` | yes |
| `next_input_norm` | 31 | `multi_add_rms_norm` / `vllm_inductor` | addends 3, sum stored, normed stored | `input_norm` of layers 1 to 31 | yes |
| `final_norm` | 1 | `multi_add_rms_norm` / `vllm_inductor` | addends 3, sum not stored, normed stored | `final_norm` | yes |
| `lm_head` | 1 | `single_gemm` / `torch_linear` | n 128,256, k 4,096 | `lm_head` | yes |

### Per mapping: schema, fields, registration, equivalence

**1. Attention: `paged_attn_batch_decode` / `_prefill`, backend `flashinfer_trtllm_gen`.**

- **Registration.** Registered (`profiling/kernels/paged_attn_batch_decode.py`,
  `paged_attn_batch_prefill.py`). The runner is
  `profiling/runners/attention/paged_attn_batch_trtllm_gen.py`, run in the
  vLLM environment.
- **Decode args.** `num_qo_heads`, `num_kv_heads`, `head_dim`, `q_dtype`,
  `kv_dtype`, `o_dtype`, `batch_size`, `max_kv_len`, `total_kv_len`. Rust
  input `{batch_size, max_kv_len, total_kv_len}`, `Cache3DLinear`.
- **Prefill args.** The same geometry plus `requests` (cached prefix, new
  tokens).
- **The same operation.** The runner calls
  `flashinfer.decode.trtllm_batch_decode_with_kv_cache(..., kv_layout="HND", backend="trtllm-gen")`,
  vLLM's `FlashInferImpl` path. Its launch unit is the counter-zeroing fill
  plus the trtllm-gen kernel, timed as a CUPTI union. That is the pair the
  trace shows (`FillFunctor<unsigned char>` then `fmhaSm100fKernel_…ForGen`).
- **Not usable as is:**
  - `BackendSupport(compute={FP8_E4M3}, kv={FP8_E4M3})`, and the runner
    raises unless q/kv/o are FP8/FP8/bf16. The trace is bf16 throughout:
    `QkvBfloat16OBfloat16` in the kernel name, and
    `reshape_and_cache_flash<bf16, bf16>`. Reusing the rows would price a
    different kernel.
  - The batch grid's split knots use a hard-coded `B200_SMS = 148`
    (`simulator/src/timing/kernels/fa4_varlen_projection.rs:46`).
- **Needed (Phase 3B, approval required):**
  - bf16 query and bf16 KV in that runner: cache tensor type, packed K|V
    layout, I/O byte count, support set;
  - its reference check at bf16 tolerances;
  - the SM count per GPU in the grid, leaving B200 grids unchanged.

**2. Activation: `silu_and_mul`, backend `vllm_inductor`.**

- Args `num_tokens`, `hidden_dim`, `dtype`.
- The runner compiles vLLM's `SiluAndMul.forward_native` over a merged
  `[tokens, 2 × hidden]` input with vLLM's Inductor settings.
- The compiled kernel is the one in the trace: `triton_poi_fused_mul_silu_slice_1`
  reads the merged gate/up buffer and writes one output.
- Equivalent; already used by `qwen35_vllm_nvfp4_gdn_dense`.

**3. Norms: `multi_add_rms_norm`, backend `vllm_inductor`.**

- **Args.** `num_tokens`, `hidden_dim`, `num_addends`, `store_sum`,
  `store_normed`, `store_fp8`, `dtype`.
- **Runner.** It compiles vLLM's native `fused_add_rms_norm` / `rms_norm` the
  way vLLM does.
- **Each norm's addends and stores, read from the compiled code:**

  | trace kernel | inputs | stores | leaf |
  | --- | --- | --- | --- |
  | `triton_red_fused_fused_add_rms_norm_0` | attention output, residual, weight | normed only | `post_norm` (2, sum not stored) |
  | `triton_red_fused_fused_add_rms_norm_2`, layers 1 to 31 | MLP output, attention output, old residual, weight | new residual and normed | `next_input_norm` (3, sum stored) |
  | `triton_red_fused_fused_add_rms_norm_2`, last | three addends, weight, in place | normed only | `final_norm` (3, sum not stored) |
  | `triton_red_fused__to_copy_embedding_rms_norm_0` | token ids, embedding table, weight | the gathered row and normed | `first_norm` (1, sum stored), approximate |

- **The final-norm row is inferred.** It is identified by its signature: a
  graph piece with this one kernel, in place, three addends.
- **Why not `residual_rms_norm`.** It prices vLLM's CUDA `fused_add_rms_norm`
  kernel, a different implementation and a two-addend in-place contract. The
  Inductor graph recomputes the attention add inside the next input norm
  instead.

**4. KV append: `kv_cache_append`, backend `vllm_cuda`, `cache_layout = "HND"`.**

- The args have `cache_layout`, and the runner accepts NHD and HND
  (`profiling/runners/attention/kv_cache_append.py:27`).
- No arch passes HND today. The variant passes it because TRT-LLM attention
  reads an HND cache.
- Block size 16: the attention kernel name carries `P16`, and vLLM's default
  block size is 16.
- Equivalent; it needs GB300 rows at HND, since the existing ones are NHD.

**5. RoPE: no existing kind (verified against every rotary kind).**

- **Candidates ruled out.**
  - `qnorm_rope_kv_insert` needs a query norm and an MLA cache.
  - `qk_rms_norm_rope_gate` adds QK norms and a gate.
  - `q_pad_kv_rope_mxfp8_insert`, `vllm_mla_rope`, `mla_rope_quantize_fp8`
    and `inv_rope_fp8_quant` are MLA/DSv4-specific.
  - `sconv_qkvr_prep` and `q_kv_rms_norm` have no RoPE.
- **What the traced kernel does** (`triton_poi_fused_<n>`):
  - reads the merged QKV row (stride 6,144), the position ids and the cos/sin
    cache (131,072 rows);
  - rotates the full 128-dim head in NeoX halves;
  - writes contiguous K (8 × 128) and Q (32 × 128).
- **Proposed new kind.** Name `gqa_rope_neox`, args `num_tokens`,
  `num_qo_heads`, `num_kv_heads`, `head_dim`, `rotary_dim`, `max_position`,
  `dtype`. Backend `vllm_inductor`, the existing name for runners that compile
  vLLM's native op with vLLM's Inductor settings.
- **Runner.** vLLM's `get_rope(...)` with Llama 3 scaling (base 500,000,
  factor 8). The runner of `vllm_mla_rope` is the template, and the shared
  helper `profiling/runners/vllm_inductor.py::compile_like_vllm` already
  exists.
- **Fallback.** Other GQA archs (`mimo_v2`, `qwen36`) price RoPE with an
  `elementwise` placeholder. That would be an approximation, not the traced
  kernel.

## 2. Fusion accounting

Checked by `serving-validation/test_vllm_llama_mapping.py`: **23 tests, all
passed**, run with the main checkout's environment on tray12, CPU only. The
traced counts come from the 183 clean decode iterations
(`tp1-audit/tp1_audit_summary.json`).

| question | how it is guaranteed | test |
| --- | --- | --- |
| Is every traced kernel charged exactly once? | each role maps to one leaf; leaves × kernels per call equal the traced count per iteration, for all 14 architecture roles (417 forward kernels + LM head) | `test_every_traced_kernel_is_charged_exactly_once` |
| Is every original leaf replaced exactly as often as it was charged? | `input_norm` 32 = 1 (in `first_norm`) + 31 (`next_input_norm`); every other leaf one for one; `embedding` folded once | `test_every_original_leaf_is_replaced_exactly_as_often_as_it_was_charged` |
| Are residual add and RMSNorm double-counted? | 65 norm leaves = 65 traced norm kernels (1 + 32 + 31 + 1); no leaf prices a residual add on its own; `llama3_dense` had no residual-add leaf to remove; a sum is stored only where the trace stores it | `test_residual_add_and_rms_norm_are_not_double_counted` |
| Embedding plus initial RMSNorm? | one leaf replaces both `embedding` and layer 0's `input_norm`; the per-layer input norm then covers 31 layers | `test_embedding_and_first_norm_are_one_leaf` |
| Is the activation charged once? | one leaf, 32 per iteration = 32 traced kernels | `test_activation_is_charged_once` |
| Attention fill? | inside the `paged_attn_batch_decode` launch unit (its doc: "Both launches are the L1 unit"; CUPTI union over both); no separate fill leaf | `test_attention_fill_is_inside_the_attention_leaf` |
| Split-K reduce charged twice? | the O and down GEMMs are two kernels per call in the trace; `single_gemm` rows time all kernels of the call as one CUPTI union (`profiling/runners/gemm/torch.py`, `interval_union=True`); no reduce leaf exists | `test_split_k_reduce_is_inside_its_gemm_leaf` |
| GEMM shapes unchanged? | the five shapes equal `llama3_dense`'s | `test_gemm_shapes_are_the_simulators` |
| Do the reused kinds exist with the needed fields? | registry lookup of kind, backend and args fields for every reused leaf | `test_existing_kinds_and_backends_are_registered_with_the_needed_fields` (12 cases) |
| Blockers recorded | `gqa_rope_neox` not registered; `flashinfer_trtllm_gen` rejects bf16 and accepts FP8 on GB300 | `test_the_proposed_rope_kind_does_not_exist_yet`, `test_paged_attention_backend_rejects_bf16_today` |

**Runtime kernels.** The 16 kernels per iteration (input preparation,
sampling, bookkeeping) are not charged by any leaf, by design (section 4).

## 3. GEMM discrepancy analysis (CPU-only)

**Shapes and layout, from the trace and the compiled graph.**

- vLLM's graph calls `extern_kernels.mm(x, reinterpret_tensor(W, (k, n), (1, k)))`:
  a row-major `[m, k]` activation times the transposed `[n, k]` weight. That
  is the cuBLAS call `F.linear` makes, and the kernel names end in `_TNT`.
- The launch grids confirm N and the split-K factors:

  | role | kernel | grid | tile | N | split-K |
  | --- | --- | --- | --- | --- | --- |
  | QKV | `nvjet_sm103_tst_64x8_64x16_4x1` | 96 | 64 | 6,144 | 1 |
  | O | `nvjet_sm103_tst_64x8_64x16_4x1_splitK` | 128 | 64 | 4,096 | 2, plus reduce |
  | gate/up | `nvjet_sm103_tst_192x8_64x8_2x1` | 150 | 192 | 28,672 | 1 |
  | down | `nvjet_sm103_tst_64x8_64x16_2x1_splitK` | 448 | 64 | 4,096 | 7, plus reduce |
  | LM head | `nvjet_sm103_tst_128x8_64x12_4x1` | 1,004 | 128 | 128,256 | 1 |

- M = 8, the tile's second dimension. bf16 in and out; the split-K reduce is
  instantiated with `float` accumulation.

**Profiler side.**

- `torch`: `torch.mm(a[m,k], b[k,n])` with a contiguous `[k, n]` weight. A
  different operand layout from vLLM's.
- `torch_linear`: `F.linear(x[m,k], W[n,k])`, vLLM's layout.
- Both are eager launches with the L2 flushed, timed as a CUPTI union of all
  the call's kernels.
- Kernel names are not stored in rows or run artifacts. GB300 rows exist for
  both backends.
- The simulator picks the faster backend per cell, which is `torch` for all
  five shapes.

**Trace side.** Per-call times inside a CUDA graph. "Exclusive" removes each
kernel's overlap with its predecessor. The O and down overlaps are the
split-K GEMM and its reduce running concurrently, which the profiler's union
counts once. µs per call:

| role | trace, raw | trace, exclusive | `torch` row (used now) | `torch_linear` row (same layout) | `torch_linear` / exclusive |
| --- | --- | --- | --- | --- | --- |
| QKV | 10.28 | 10.28 | 10.79 | 10.87 | 1.057 |
| O | 10.71 | 9.20 | 9.93 | 10.09 | 1.097 |
| gate/up | 38.00 | 38.00 | 35.47 | 38.08 | 1.002 |
| down | 23.03 | 21.94 | 21.55 | 22.44 | 1.023 |
| LM head | 152.42 | 152.42 | 148.76 | 150.66 | 0.988 |
| iteration (32 layers + head) | 2,776.9 | 2,693.8 | 2,636.4 (0.979) | 2,758.0 (1.024) | |

**Explained without new GPU work:**

1. **Overlap.** The audit's "GEMMs 7 % low" compared the simulator with summed
   kernel durations. 82.8 µs of the 87.5 µs overlap per iteration is inside the
   O and down split-K pairs, which the profiler's union counts once. Against
   exclusive time the gap with the current leaves is 2.1 %, not 5 to 7 %.
2. **Operand layout.** The current leaves use `torch` rows, a different
   cuBLAS operand layout from vLLM's. For gate/up the same-layout row matches
   the trace to 0.2 %, where the `torch` row is 6.7 % lower.

**Not explainable without GPU measurement:**

- the residual per-shape differences of the same-layout rows: QKV +5.7 %,
  O +9.7 %, down +2.3 %, LM head −1.2 %; the trace is faster for QKV and O;
- candidate causes:
  - the cuBLAS version, 13.1.0.3 for the rows against 13.1.1.3 in serving;
  - kernel selection, since the rows record no kernel names;
  - CUDA-graph replay with programmatic dependent launch against isolated
    eager launches;
  - L2 state, flushed for the rows, while in serving the previous kernel's
    output is hot;
  - clocks.

None of these can be measured from existing artifacts. No correction factor
was applied.

**Recommendation for GEMMs:**

- Restrict the variant's GEMM backends to `torch_linear` (existing rows,
  vLLM's layout) instead of best-of-two.
- Compare against exclusive kernel time.
- A Phase 3B check could re-time the five shapes in the serving environment,
  as eager calls and inside a CUDA graph, and record kernel names. That is
  one GPU, about two minutes, and needs approval.

## 4. Runtime and idle accounting (no multiplier)

Per iteration, medians over the 183 decode iterations (µs).

| quantity | µs | belongs to | proposed treatment |
| --- | --- | --- | --- |
| RoPE kernels | 56.4 | architecture | the new `rope` leaf |
| runtime kernels: input preparation (7), sampling except the LM head (8), bookkeeping (1) | 41.8 | serving execution (engine runtime) | not in the architecture; reported as a separate measured term "engine runtime kernels", keyed by engine version and batch shape |
| idle inside the forward pass | 48.5 | serving execution (graph-replay dependencies) | reported separately; not charged by the architecture |
| idle between the iteration's phases | about 36 | serving execution (host-driven phase boundaries) | reported separately |
| idle between iterations | 0.3 (p90 15) | serving execution (scheduler) | reported separately |
| kernel overlap, intra-operation (split-K GEMM and reduce) | 82.8 | architecture leaf | already inside the GEMM rows' CUPTI union; compare against exclusive time |
| kernel overlap, between operations (attention start) | 4.8 | serving execution | reported separately |

**Proposed comparison contract for Phase 3C:**

- **Architecture term.** The variant's leaf sum is compared with the measured
  exclusive kernel time of the architecture roles. Runtime kernels are
  excluded.
- **Serving terms.** Runtime kernels and the three idle terms are measured
  from the same trace and reported beside the prediction, never folded in.
- **Iteration latency.** The GPU cycle equals architecture time, plus serving
  terms, minus inter-operation overlap. It is reported as a decomposition,
  not predicted.

The existing per-worker `gpu_time_multiplier` stays at 1.0. No new scheduling
or overlap model is proposed.

## 5. Missing-row profiling matrix (not run)

Single GPU, tray12 GPU 1, same guards as the smoke test. Phase 3B,
approval required.

| kind / backend | configurations | rows | correctness oracle | timing |
| --- | --- | --- | --- | --- |
| `multi_add_rms_norm` / `vllm_inductor` | hidden 4,096, bf16 × {(1, store sum, normed), (2, normed), (3, store sum, normed), (3, normed)} | 4 × 68 = 272 | `multi_add_rms_norm_reference`, fp32 | CUPTI, all launches of the compiled call, L2 flushed |
| `silu_and_mul` / `vllm_inductor` | hidden 14,336, bf16 | 68 | `silu_and_mul_reference`, rtol = atol = 2e-2 | as above |
| `kv_cache_append` / `vllm_cuda` | 8 KV heads, head 128, block 16, bf16 / bf16, **HND**, tensor scale | 70 | the runner's own check; the earlier shim probe matched the reference bitwise | CUPTI on `reshape_and_cache_flash_kernel` |
| `paged_attn_batch_decode` / `flashinfer_trtllm_gen` | 32/8 heads, head 128, bf16/bf16/bf16 | grid from the Rust sweep (several hundred cells) | fp32 varlen reference, abs 0.05, rel-L2 0.06 | CUPTI union of fill + kernel |
| `paged_attn_batch_prefill` / `flashinfer_trtllm_gen` | same | grid from the Rust sweep | same | same |
| `gqa_rope_neox` / `vllm_inductor` (new) | 32/8 heads, head 128, rotary 128, Llama 3 scaling, bf16 | 68 | fp32 RoPE reference | CUPTI, compiled call |

The attention rows are **blocked** until the bf16 runner extension. RoPE is
blocked until the new kind exists. Both are code changes to approve first.

The first three kinds can run as they stand. In total 410 rows.

**Environment.** These runners execute in the "vLLM environment". The GB300
KV shim is not enough for Inductor runners. The serving environment built in
Stage A has the full fork, and all four reusable runners import there on CPU
(checked: `multi_add_rms_norm_vllm_inductor`, `silu_and_mul_vllm_inductor`,
`kv_cache_append`, `paged_attn_batch_trtllm_gen`, plus `vllm.ir`,
`CompilationConfig` and `get_rope`). Pointing the profiler's vLLM environment
at it needs a `.venv` link inside the fork checkout, which is preparation.
The CUPTI extension will rebuild for its torch 2.13.

| estimate | value |
| --- | --- |
| GPU time | about 10 min for the 410 rows (Inductor compiles once per process; past fills ran 70 KV rows in 10 s) |
| disk | under 50 MB (rows, Inductor cache) |
| isolation and stop conditions | as in the TP2/TP4 fills: UUID, idle GPU, no compute process; stop on failed checks, outliers above 5 %, a timeout or a foreign process |
| database | rows go to a private copy, never the tracked database |

## 6. Recommendation

**Implementable without a cost-model change.** The variant is a new opt-in
arch tag with different leaf choices plus a RoPE leaf, in a serial `Sum` like
`llama3_dense`. The file set:

- `simulator/src/arch/<tag>.rs`;
- registration in `arch/mod.rs`, `arch/config.rs` (variant and `model()` /
  `max_model_len()` arms) and `arch/build.rs`;
- an arm in `deployment/unified.rs`.

`simulator cost-trees --kernel-configs` then enumerates its rows on CPU.

**Order of approvals I would ask for:**

1. **Phase 3B-1 (GPU, about 10 min):** the 410 rows of the three reusable
   kinds, into a private database copy.
2. **Phase 3B-2 (code):**
   - a bf16 support extension of the TRT-LLM attention runner;
   - an SM-count-aware grid, leaving B200 unchanged;
   - the `gqa_rope_neox` kind and runner;
   - then their rows.
3. **Phase 3B-3 (simulator):** the opt-in arch tag with GEMMs pinned to
   `torch_linear`.
4. **Phase 3C:** the TP1 comparison under the section 4 contract.

## Commits and worktrees

- This phase's commit is on `gb300-support` only; fork main is unchanged.
- The main checkout and the profiling worktree are clean.
- The serving worktree `ServingStudioSim-serving` carries its two intentional,
  uncommitted submodule links.
