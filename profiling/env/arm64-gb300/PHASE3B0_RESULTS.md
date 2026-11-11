# Phase 3B-0 — CPU-only implementation and validation (GB300, Llama 3.1 8B TP1, vLLM)

Date: 2026-11-11. Branch `gb300-support` of `kaijunli-infr/ServingStudioSim`.
Commits: `5e239ac0` (implementation) and `14e797be` (fp8 rejection, rotary
embedding built under a vLLM config, corrected CPU check). Base: `4710a07a`.
No GPU was used. Neither the tracked `profiling/profile.db` nor the private
DB changed. Fork main and upstream are untouched.

## 1. What changed

| approved item | change |
| --- | --- |
| 1. bf16 TRT-LLM attention | `paged_attn_batch_trtllm_gen.py` accepts (bf16, bf16, bf16) beside (fp8_e4m3, fp8_e4m3, bf16); mixed pairs raise `ProfilerNotImplemented`. Query and cache element type follow the pair; call, HND layout, unit scales, checks and launch set are unchanged. Registry `BackendSupport` adds BF16 to the compute and kv sets of both kinds. |
| 2. GPU-aware launch SMs | `launch_sms(gpu)` (in `fa4_varlen_projection.rs`): 152 for `NVIDIA GB300`, `B200_SMS` (148) for every other GPU. Used by the decode batch axis (`wave = SMs / ctas_per_request`) and the prefill tile axis (`wave * SMs / heads`). The Python runner already reads `multi_processor_count` from the device. |
| 3. RoPE kind | `gqa_rope_neox` / `vllm_inductor` (Python kind, runner, independent reference; Rust `KernelSpec`, 68-cell token grid, `Cache1DLinear`, slot input). The block is `qkv.split(...)` then `get_rope(...).forward_native(positions, q, k)`, compiled with `compile_like_vllm`. |
| 4.–6. arch | New opt-in tag `llama3_dense_vllm` (`simulator/src/arch/llama3_dense_vllm.rs`, registered in `arch/mod.rs`, `arch/config.rs`, `arch/build.rs`, `deployment/unified.rs`). GEMMs `torch_linear` only; norms, RoPE and SiLU `vllm_inductor`; KV append `vllm_cuda`, HND, block 16; attention `flashinfer_trtllm_gen` bf16. bf16 only: `fp8: true` and non-bf16 configs are rejected. No existing arch, backend list or cost-model rule changed. |
| mapping | `serving-validation/vllm_llama_mapping.py`: attention and RoPE are no longer `blocked`/`new_kind`; tests check every leaf's kind/backend is registered with its config fields. |

### Cost tree of `llama3_dense_vllm` (one GPU, serial `Sum`, as `llama3_dense`)

```
Sum[ first_norm,
     Labeled "layer" Scale{N}( Sum[ qkv_proj, rope, kv_cache_append, prefill_attention,
                                    decode_attention, o_proj, post_norm, up_gate_proj,
                                    activation, down_proj ] ),
     Scale{N-1}( next_input_norm ),          # omitted when N = 1
     final_norm, lm_head ]
```

| leaf | kind / backend | key fields (8B) | input per iteration |
| --- | --- | --- | --- |
| first_norm | multi_add_rms_norm / vllm_inductor | 1 addend, sum + normed | batch tokens |
| qkv_proj | single_gemm / torch_linear | n 6144, k 4096 | m = batch tokens |
| rope | gqa_rope_neox / vllm_inductor | 32/8 heads, 128, rotary 128, max_position 131072 | batch tokens |
| kv_cache_append | kv_cache_append / vllm_cuda | 8 × 128, block 16, bf16/bf16, HND, tensor | prefill new tokens + decodes |
| prefill_attention | paged_attn_batch_prefill / flashinfer_trtllm_gen | 32/8/128, bf16 ×3 | chunk pairs; zero if none |
| decode_attention | paged_attn_batch_decode / flashinfer_trtllm_gen | same | batch, max and total KV; zero if none |
| o_proj | single_gemm / torch_linear | n 4096, k 4096 | batch tokens |
| post_norm | multi_add_rms_norm / vllm_inductor | 2 addends, normed | batch tokens |
| up_gate_proj | single_gemm / torch_linear | n 28672, k 4096 | batch tokens |
| activation | silu_and_mul / vllm_inductor | 14336 | batch tokens |
| down_proj | single_gemm / torch_linear | n 4096, k 14336 | batch tokens |
| next_input_norm (×N−1) | multi_add_rms_norm / vllm_inductor | 3 addends, sum + normed | batch tokens |
| final_norm | multi_add_rms_norm / vllm_inductor | 3 addends, normed | batch tokens |
| lm_head | single_gemm / torch_linear | n 128256, k 4096 | m = requests |

**Embedding approximation (documented in the arch file).** The first norm is
priced as a one-addend RMSNorm that stores its input and its normed output.
vLLM fuses the embedding gather into that kernel (it reads the row from the
table by token id); the kind reads the row contiguously. The bytes per token
are the same and the access pattern differs. There is no separate embedding
leaf.

**Not modeled (as in `llama3_dense`).** CUDA-graph padding of a decode batch
(leaves run at the real token and request counts), engine runtime kernels and
GPU idle time. These are serving-execution terms, reported beside the
prediction.

## 2. CPU validation results

| requirement | evidence | result |
| --- | --- | --- |
| Existing FP8 attention configs unchanged | `test_fp8_geometry_is_unchanged_and_bf16_doubles_the_element`: FP8 element 1 B, KV bytes/token and io-bytes formula unchanged. Cost-tree export: identities of the GB300 Qwen3.5 FP8 attention configs unchanged (see the next row for their grids). | PASS |
| B200 attention launch configs unchanged | `launch_sms_keeps_148_for_every_gpu_but_gb300`; B200 Qwen3.5 block of `cost-trees --kernel-configs` **byte-identical** to baseline; B200 export of the new arch keeps the 148-SM knots (decode 18/19, prefill 13/14). | PASS |
| GB300 152-SM grid vs kernel launch constraints | `gb300_llama_batch_axis_uses_its_152_sms` (8 CTAs per request, wave 19, splits 2 at batch 8 and 4 at batch 4, the traced grid (splits, 8, batch)); `tile_axis_counts_gb300_waves_of_152_sms`. The only diff in the existing exports is the GB300 Qwen3.5 FP8 attention axes: decode batch knots 18/37 → 20/39, prefill tile knots 18…49 → 20…51. GB300 has no rows for these kinds, so no stored data is reinterpreted. | PASS (intended change) |
| bf16/FP8 keys cannot collide | `q_dtype`, `kv_dtype` and `o_dtype` are fields of both Args schemas and of the Rust enumerate payload (`test_bf16_and_fp8_rows_never_share_a_key`). | PASS |
| RoPE matches an independent reference | `tests/test_gqa_rope_neox.py`: the reference equals a complex-number rotation (NeoX halves, full and partial rotary dim, pass-through tail); it equals an element-wise port of the traced Inductor kernel's indexing (row stride (Hq+2Hkv)·D, K at column Hq·D, cos/sin row = position); dtype bf16 in and out, contiguous [T, Hq·D] / [T, Hkv·D]; the Llama 3 frequency bands (unscaled, ÷8, smoothed). | PASS |
| vLLM's own RoPE on CPU (serving venv, fork e1423a4d, torch 2.13) | `rope_vllm_cpu_check.py`: `Llama3RotaryEmbedding`, rotary 128, cache [131072, 128]. Float32 `forward_native` = reference, max abs diff **0.0**. Cache vs float64 Llama 3 formula: 8.3e-7 for positions < 16, and ≤ 0.67 × the float32 angle bound everywhere. bf16 cache = float32 cache rounded. bf16 output passes the runner's oracle (rtol = atol = 2e-2; max abs 0.031 = one bf16 ulp at 4). | PASS |
| New arch charges each operation exactly once | `every_operation_is_charged_exactly_once_per_layer_or_iteration` (unit time in one slot, aggregate): first/final norm and lm_head 1, ten layer leaves 32, next-input norm 31; 64 RMSNorms per iteration (32 input + 32 post) + final, as in the HF model; one SiLU, two attention leaves (prefill or decode is zero), no embedding leaf. One-layer model: no next-input norm. | PASS |
| GEMMs pinned to `torch_linear` | arch unit test plus `llama3_dense_vllm_builds_bf16_and_rejects_fp8` through `build_iter_model`. | PASS |
| Existing arch / preset predictions unchanged | `phase3b0/regress.sh` at `5e239ac0` against the `4710a07a` baseline. 11 presets, return codes identical (qwen36_local fails in both on a missing `vllm_fused_moe` row). 1,000 of 1,083 output files are byte-identical after path normalization. The other 83 differ only in pids, timestamps, elapsed times and the random `prediction_id`; every stdout is identical once durations are stripped (`post-5e239ac0/pred_diff.json`). Cost-tree exports: B200 Qwen3.5, GB300 and H200 `llama3_dense`, and GB300 `llama3_dense_tp` blocks byte-identical. | PASS |
| Database contents unchanged | tracked `3ec95dcd4226…8a3e`; private `profile-gb300.db` `9474a224feb0…e1af`; both equal before and after. | PASS |
| New configs generate without a GPU | `simulator cost-trees --kernel-configs` on the new arch for GB300 and B200 (`phase3b0/new-arch/`), rc 0, 14 configs each (counts in §3). | PASS |

**Found and fixed by the CPU checks, before any GPU run:**

1. The RoPE runner built `get_rope` outside a vLLM config. In this fork,
   `RotaryEmbedding` is a `CustomOp` and raises on `__init__` without one. It
   is now built inside `set_current_vllm_config(VllmConfig())`, as
   `vllm_mla_rope` does.
2. Dense archs ignore `fp8`, so `llama3_dense_vllm` with `fp8: true` had been
   building bf16 leaves silently. It now refuses.

**Test gates (rack 3 tray12, CPU, `CUDA_VISIBLE_DEVICES=` empty, at `14e797be`):**

- Rust `cargo test -p simulator --lib`: 1,595 passed, 0 failed (baseline 1,583; 12 new).
- pytest CPU tier: 5,025 passed and 8 failed. The 8 are the known environmental
  set (CPU-only torch on aarch64: 4 × `test_l1_single_gemm` energy, `kda` row
  provenance, 2 × `dsa_sparse_mla_attention` "Torch not compiled with CUDA", `gdn_spec_decode`).
  The baseline had 5,001 passed and 9 failed; one more `dsa` parametrization of
  the same CUDA assertion fails there depending on order.
- pytest presets tier: 9 passed.
- Mapping, RoPE and attention tests: 56 passed.
- rustfmt clean; ruff clean on every touched file. The repo-wide ruff count was
  already non-zero before this phase and is unchanged.

## 3. Profiling matrix review (nothing run; needs approval)

All rows are GB300, bf16 and TP1. Configurations come from the CPU export of
`llama3_dense_vllm` (`phase3b0/new-arch/cost_trees.json`).

### 3.1 Counts

| group | kind / backend | configs | grid | rows to profile | exists already |
| --- | --- | --- | --- | --- | --- |
| A1 norms | multi_add_rms_norm / vllm_inductor | 4: (1, sum+normed), (2, normed), (3, sum+normed), (3, normed); hidden 4096 | 68 token knots (1 … 65536) | 272 | none |
| A2 activation | silu_and_mul / vllm_inductor | 1: hidden 14336 | 68 | 68 | none |
| A3 KV write | kv_cache_append / vllm_cuda | 1: 8 KV heads × 128, block 16, bf16/bf16, **HND**, tensor | 70 (68 + 264, 265) | 70 | NHD only (8/4/2 heads, 70 each) |
| **A total** | | | | **410** | |
| B1 RoPE | gqa_rope_neox / vllm_inductor | 1: 32/8 × 128, rotary 128, max_position 131072 | 68 | 68 | none |
| B2 decode attention | paged_attn_batch_decode / flashinfer_trtllm_gen | 1: 32/8/128, bf16 ×3 | batch 20 knots (incl. 19, 20 at 152 SMs) × max KV 8 × fill 4 = 640 cells, 11 masked (> 32 GiB KV) | **548 unique** (629 feasible; a batch of 1 cannot fill) | none |
| B3 prefill attention | paged_attn_batch_prefill / flashinfer_trtllm_gen | same | requests 10 × tiles 22 (incl. 14, 15) × prefix 6 = 1,320 cells, 3 masked | **1,011 unique** (1,317 feasible; past `tiles` requests the batch saturates) | none |
| C GEMMs | single_gemm / torch_linear | 5 shapes | 68 | 0 | 68 rows each in the tracked DB |

### 3.2 Per group: shapes, oracle, timing, cost, and match to the trace

All runners time **eager launches** with `Timer.cupti`.

- **Probe.** A 100 ms capture: 10 estimate launches, then 10 … 1,000 launches.
  The probe is kept if drift is within 2 %; otherwise a 2,000 ms capture
  follows (up to 10,000 launches).
- **Statistic.** The mean of the per-launch **kernel-only** device duration
  from the CUPTI activity records. Launch overhead and gaps are not counted.
- **L2.** A 64 MiB read-only reduction displaces L2 before every launch, and
  its records are excluded.
- **Energy.** `Energy.perf` follows the timing, using its time.
- **Stability.** Probe drift ≤ 2 %. In previous fills the stop rule was on
  outliers above 5 % and failed checks.
- **Not CUDA graph.** None of these runners replays a CUDA graph, and that
  they import in the serving venv says nothing about graph semantics.

| group | shapes / dtypes / layout | oracle and tolerance | timing specifics | GPU time (estimate) / disk | trace correspondence and known gaps |
| --- | --- | --- | --- | --- | --- |
| A1 | addends [T, 4096] bf16, weight bf16; outputs as flagged | `multi_add_rms_norm_reference` in fp32: sum rtol/atol 1e-2, normed 2e-2 | Inductor block from `vllm.ir` native `rms_norm` / `fused_add_rms_norm`, compiled by `compile_like_vllm` (vLLM's Inductor options, dynamic token dim); `kernel_name=None` sums every kernel of the call | about 6 min (compile once per config, sub-second capture per row) / < 30 MB Inductor cache | Trace: `triton_*_fused_add_rms_norm` inside the piecewise (prefill) or FULL (decode) graph. Gaps: (a) **L2 cold** here, but in serving the input is the previous GEMM's output, hot in L2 (8 × 4096 × 2 B = 64 KB), so small-batch rows likely read high; (b) the standalone compile must produce the **same single kernel** as the served graph, which is unverified (pilot P1); (c) the first norm's embedding gather (§1). |
| A2 | [T, 2·14336] → [T, 14336] bf16 | `silu_and_mul_reference`, rtol = atol = 2e-2 | as A1 | about 1.5 min / small | Trace `triton_poi_fused_mul_silu_slice`. Gaps (a) and (b) as A1. |
| A3 | key/value [T, 8, 128] bf16 → cache [blocks, 8, 16, 128] **HND** view, slot mapping | `torch.equal`: bitwise K and V at the mapped slots | CUPTI filtered on `reshape_and_cache_flash`, warmup 5 | about 0.5 min (the earlier 70-row NHD fill took about 10 s) / small | Trace `reshape_and_cache_flash_kernel`, one launch per layer. Gaps: layout comes only from strides, so the pilot confirms the HND stride path gives the traced kernel and template; L2 cold. |
| B1 | positions int64 [T]; qkv [T, 6144] bf16 → q [T, 4096], k [T, 1024] contiguous; cos/sin cache [131072, 128] (built by vLLM `get_rope`, Llama 3 scaling) | `gqa_rope_neox_reference` (fp32) with vLLM's cache: rtol = atol = 2e-2 (validated on CPU above; the cache itself was checked against the independent formula) | CUPTI, `kernel_name=None`, L2 flushed | about 1.5 min / small | Trace `triton_poi_fused_<n>`: one kernel reads Q and K in place and writes K and Q. Gaps: the standalone block may compile into **one or two** kernels (pilot P1 checks the count); L2 cold. |
| B2 | q [B, 32, 128] bf16; HND paged cache bf16, page 16; canonical batch (one request at max, the rest equal; total = key); `max_seq_len = max(max_kv_len, 512 · SMs)`; workspace 128 MiB | fp32 varlen reference on sampled rows: abs 0.05, rel-L2 0.06 | CUPTI `interval_union=True` over counter fill + `fmhaSm100f` kernel; first launch (cubin load) outside timing | 548 rows at about 1.5–3 s each (allocation, random init up to 34 GB KV, capture, energy): **15–30 min**; to be refined from the pilot's per-row wall time / < 10 MB | Trace: fill + `fmhaSm100f…P16…` per layer, grid (splits, 8, batch), inside the FULL decode graph. Gaps: (a) **split count at batch 1**: serving had `max_model_len` 8192, so ≤ ceil(8192 / 512) = 16 CTAs per request, while the runner saturates at 19 (152 / 8). Batch ≥ 2 is unaffected (the SM bound is lower), as is the traced batch 8. (b) L2 cold vs a warm KV page set in serving. (c) In-graph launch has no CPU-side cost, and neither does a CUPTI kernel duration. |
| B3 | q [Σ new, 32, 128] bf16; same cache; cumulative q and page counts as vLLM builds them | as B2 | as B2 | 1,011 rows at 2–4 s (up to 21 GB KV): **35–70 min** / < 10 MB | Trace: prefill kernel inside a piecewise graph (≤ 1,024 tokens) or eager (2,048 chunks). Gap: L2 cold. |
| C | — | — | — | 0 | Existing `torch_linear` rows (cuBLAS 13.1.0.3 at profiling time, 13.1.1.3 in serving). The GEMM residuals (QKV +5.7 %, O +9.7 %) stay unexplained until a re-time (optional pilot P4). |

**Totals (estimates).** A: about 10 min, as Phase 3A estimated. B1: about
1.5 min. B2 + B3: about 50–100 min, the dominant cost; the pilot gives
per-row numbers to decide whether to thin the grid. Disk stays under 100 MB.
Rows go only to a private DB copy.

### 3.3 Graph semantics

The trace's per-kernel times come from **CUDA-graph replay**: FULL graph for
decode, piecewise for prefill ≤ 1,024 tokens. Replay runs with programmatic
dependent launch, and the previous kernel's output is hot in L2. Every runner
above measures **eager, L2-flushed** launches of the kernel alone.

CUPTI kernel durations exclude CPU launch cost, so the comparison basis is the
trace's **exclusive kernel time**. That leaves three effects untested by any
existing artifact:

1. warm versus cold L2;
2. PDL overlap at kernel boundaries;
3. whether a standalone `compile_like_vllm` reproduces the served graph's
   kernel partition and Triton configuration.

The pilot measures all three on a few shapes. No multiplier follows from it
without a separate decision.

## 4. Proposed GPU correctness pilot (needs a go)

**Where.**
- Rack 3, tray12, **GPU 1 only** (`GPU-9574ced0-46be-d5b7-0098-01fa1e84dccb`).
- At about 10:45 UTC on 2026-11-11 it was idle: 93 MiB, no compute process.
- GPU 2 holds another user's SGLang server (≈ 250 GB). It is not touched and
  not reset, and its memory is not used.
- Before launch, the guard re-checks GPU 1: no foreign process, < 1 GiB used.

**Environment.**
- The serving venv (torch 2.13, vLLM fork e1423a4d, FlashInfer 0.7.0), with
  `PYTHONPATH` = the profiling worktree at `14e797be`.
- A driver script calls the runner functions directly. There is no
  profiling-CLI subprocess and no `.venv` link in the fork.
- The CUPTI extension JIT-builds for torch 2.13 on CPU before the GPU window.

**Budget.**
- Occupancy cap 600 s; expected about 4 min.
- Disk ≤ 300 MB: one bounded nsys trace plus the Inductor cache.
- **No DB writes.** Results go to JSON under
  `$GB300_ROOT/serving-validation/phase3b0/pilot/`.

**Cases.** Token counts 8 and 256: decode batch 8 and the 256-token prefill of the smoke test.

| id | what | pass criterion |
| --- | --- | --- |
| P1 | A1 ×4 configs, A2, A3 (HND), B1 at T ∈ {8, 256}: run each runner end to end, with its oracle | every oracle passes; launch list per call = **one** kernel whose name/type matches the trace's (`triton_*fused_add_rms_norm*` / `mul_silu_slice` / `reshape_and_cache_flash` / one `triton_poi_fused` RoPE kernel); report kernel names and the Inductor source of each against the smoke run's `torch_compile_cache` |
| P2 | B2 decode at (batch 8, KV lens of the smoke decode steps) and (batch 4, same), plus batch 1 at max KV 8192 | oracle passes; launch list = counter fill + one `fmhaSm100f` kernel whose **name equals the traced kernel name** (bf16, P16); grid (2, 8, 8) and (4, 8, 4) from a ≤ 20 s nsys capture of these calls only; batch 1 grid recorded to quantify the 19-vs-16 split gap |
| P3 | B3 prefill: one request, 256 new tokens, prefix 0 | oracle passes; kernel name equals the traced prefill kernel |
| P4 (diagnostic, not stored) | for each P1–P3 call: (a) runner timing, eager L2 flushed; (b) eager without flush; (c) the same call captured in a CUDA graph and replayed, without flush; plus the five `torch_linear` GEMMs at m = 8 in (a) and (c), with kernel names | report all three numbers beside the trace's exclusive time for the same kernel. No multiplier and no DB write; the purpose is to say whether the eager rows are a valid basis before the full run |

**Stop conditions.**
- a foreign process on GPU 1;
- GPU 1 memory > 1 GiB before launch;
- any oracle failure (stop and report, no retry);
- 600 s elapsed;
- any change to GPU 2's process list or memory;
- nsys trace > 200 MB.

**Not included.** The 410-row fill, the full attention or RoPE fills, DB
writes, TP2/TP4, preset or simulator changes, prediction-error claims.

## 5. Commits and worktrees

- `gb300-support`: `4710a07a` → `5e239ac0` → `14e797be` (this report follows
  as a docs commit). It is identical on the laptop, the rack 3 main checkout and
  origin.
- The rack main checkout is clean. The profiling worktree `ServingStudioSim-gb300`
  was not refreshed in this phase; it is still detached at the Phase 3A head
  and will be moved to the pilot commit only when the pilot is approved. The serving worktree keeps its two
  intentional, uncommitted submodule links.
- Fork main and upstream are unchanged.

## 6. Artifacts (`$GB300_ROOT/serving-validation/phase3b0/`)

- `baseline/` (at `4710a07a`) and `post-5e239ac0/`: `regress.sh` snapshots;
  `post-5e239ac0/pred_diff.json` is the field-level comparison (`pred_diff.py`).
- `new-arch/`: `blocks.json`, `cost_trees.json` (GB300 and B200 exports of
  `llama3_dense_vllm`) and `fp8.err` (from the pre-fix binary; the fp8 refusal
  is covered by the Rust dispatch test).
- `cpu-checks/rope_vllm_cpu_check.json`: the vLLM CPU RoPE check.
