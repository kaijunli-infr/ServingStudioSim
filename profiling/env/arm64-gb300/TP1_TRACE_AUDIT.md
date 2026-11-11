# TP1 trace-accounting audit, 2026-11-11

CPU-only analysis of the TP1 smoke-test trace (`TP1_SMOKE_RESULTS.md`)
against the simulator's `llama3_dense` arch for the same batches. No GPU ran,
and no measurement, simulator file or preset changed.

**Recommendation: B. A small vLLM-specific operation mapping is needed before
a controlled TP1 validation.** The execution graph has the simulator's shape:

- each decoder layer issues the simulator's nine operations, in the
  simulator's order;
- 80 % of GPU kernel time is cuBLAS GEMMs with the simulator's shapes;
- the iteration is GPU-bound and its kernels run back to back.

What differs is local: the attention implementation, the fusion of residual
add into RMSNorm, a RoPE kernel, the activation implementation and the KV
cache layout. Together they are 18 % of kernel time. Four of them already
have matching kernel kinds in the repository.

Nothing here is a validated accuracy figure. The simulator's total for these
batches lands within 0.2 % of the measured cycle (3,478.0 against 3,474.1 µs) only because its errors
cancel: GEMMs about 7 % low, attention 49 % high.

Tools: `serving-validation/tp1_trace_audit.py` (self-test included), with
`launcher timing-predict --no-gpu` on the tracked database for the
simulator side. The summary is in `serving-validation/tp1-audit/`.

## Iterations audited

- 183 iterations in which exactly 8 requests decode one token each: the
  engine records show `decode_requests == 8` and no prefill tokens.
- Cached lengths run from 568 to 630.
- Every iteration ran as a full CUDA-graph replay (smoke test, section
  "Iteration records").
- The simulator got the same 183 batches, one case each, with the measured
  per-request KV lengths (`decode_kv_lens`).

## Stage A: accounting per iteration

GPU clock, µs; median, p10 to p90, coefficient of variation over 183
iterations.

| quantity | median | p10 to p90 | CV |
| --- | --- | --- | --- |
| forward: kernel sum | 3,282.1 | 3,276.7 to 3,289.6 | 0.25 % |
| forward: busy time (union of kernel intervals) | 3,194.4 | 3,189.6 to 3,201.2 | 0.24 % |
| forward: first to last kernel | 3,243.0 | 3,237.9 to 3,250.0 | 0.24 % |
| whole iteration (all phases): kernel sum | 3,476.8 | 3,471.1 to 3,484.3 | 0.24 % |
| whole iteration: busy time | 3,388.8 | 3,383.5 to 3,396.3 | 0.23 % |
| whole iteration: first to last kernel | 3,471.9 | 3,465.5 to 3,480.9 | 3.2 % |
| GPU cycle: first kernel to next iteration's first kernel | 3,474.1 | 3,466.6 to 3,489.8 | 3.2 % |
| host: `forward` NVTX range (graph launch) | 127.8 | 123.8 to 138.6 | 19 % |
| host: all of the iteration's NVTX ranges | 1,815.5 | 1,749.3 to 1,954.4 | 4.7 % |

**Reconciliation, medians:**

| item | µs | share of the cycle |
| --- | --- | --- |
| busy time | 3,388.8 | 97.5 % |
| idle inside the forward pass | 48.5 | 1.4 % |
| idle between phases of the iteration | 36.2 | 1.0 % |
| idle between iterations | 0.3 | 0.0 % |
| unattributed kernels in the cycle | 0 | 0 % |

- **Busy plus idle closes the cycle:** 3,388.8 + 48.5 + 36.2 + 0.3 = 3,473.8
  µs against the measured 3,474.1. The idle between phases is the whole
  iteration's idle, 84.7, minus the forward pass's, 48.5.
- **Kernel sum exceeds busy time.** Sum minus busy time is 87.5 µs (forward)
  and 88 µs (iteration): kernels overlap, about 2.7 % of busy time. The
  overlap is between consecutive kernels in the graph, consistent with
  programmatic dependent launch. A serial kernel sum therefore overstates GPU
  time by about 2.6 %.
- **Idle time is small.** Idle inside the forward pass is 48.5 µs, 1.4 % of
  its span. Between the iteration's phases (preprocess, forward, sample,
  bookkeep) there is about 36 µs; between iterations, 0.26 µs median and
  15 µs at p90.
- **Every kernel is attributed.** No kernel in any cycle is unattributed to
  an iteration range.
- **The run is GPU-bound.** The host spends 1.8 ms per iteration against
  3.47 ms of GPU time and launches the forward pass as one graph in 128 µs.
  Host scheduling is hidden behind GPU execution and is not part of the
  measured GPU time.
- **One outlier.** The first decode iteration of the first batch (iteration
  6) has 1.16 ms of idle inside its span, at the change from the batch's
  prefill to its decode. In 36 of the 183 iterations the gap to the next
  iteration exceeds 5 µs, never 17 µs.
- **Answer to the Stage A question:** summed kernel time is not iteration
  latency, but close to it here. The GPU cycle equals summed kernel time
  minus 2.6 % overlap plus 2.5 % idle; the cycle is 0.1 % below the sum.

## Stage B: operation mapping

Per layer the forward pass runs 13 kernels, in this order, identical in all
183 iterations. The per-iteration column is the median kernel time summed
over 32 layers.

| order | kernel | role | simulator leaf | class | µs per iteration |
| --- | --- | --- | --- | --- | --- |
| 1 | `triton_red_fused_fused_add_rms_norm_2` (layer 0: `…_to_copy_embedding_rms_norm_0`) | residual add + input RMSNorm (layer 0: embedding + RMSNorm) | `input_norm` (rms_norm, flashinfer); `embedding` | fused with another operation | 64.8 + 5.6 |
| 2 | `nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT` | QKV projection, m 8, n 6,144, k 4,096 | `qkv_proj` | directly comparable | 328.8 |
| 3 | `triton_poi_fused_3` | RoPE on Q and K | none | unmodeled | 56.4 |
| 4 | `vllm::reshape_and_cache_flash_kernel<bf16, bf16, kAuto>` | KV cache write | `kv_cache_append` (vllm_cuda) | comparable after layout adjustment | 71.3 |
| 5 | `at::native::vectorized_elementwise_kernel<FillFunctor<unsigned char>>` | TRT-LLM multi-CTA counter zeroing | part of attention | comparable after implementation adjustment | 26.4 |
| 6 | `fmhaSm100fKernel_…MultiCtasKvCgaVarSeqQ8Kv256…ForGen` | decode attention, TRT-LLM | `decode` (flashinfer_attn_decode, fa2) | comparable after implementation adjustment | 285.6 |
| 7, 8 | `nvjet_sm103_tst_64x8_64x16_4x1_v_bz_splitK_TNT` + `cublasLt::splitKreduce_kernel` | O projection, n 4,096, k 4,096 | `o_proj` | directly comparable | 342.7 |
| 9 | `triton_red_fused_fused_add_rms_norm_0` | residual add + post-attention RMSNorm | `post_norm` | fused with another operation | 102.6 |
| 10 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT` | gate/up projection, n 28,672, k 4,096 | `up_gate_proj` | directly comparable | 1,216.0 |
| 11 | `triton_poi_fused_mul_silu_slice_1` | SiLU and multiply | `activation` (elementwise, triton) | comparable after implementation adjustment | 42.9 |
| 12, 13 | `nvjet_sm103_tst_64x8_64x16_2x1_v_bz_splitK_TNT` + `splitKreduce` | down projection, n 4,096, k 14,336 | `down_proj` | directly comparable | 737.0 |
| after layer 31 | `triton_red_fused_fused_add_rms_norm_2` | residual add + final RMSNorm | `final_norm` | fused with another operation | 2.0 |
| sample phase | `nvjet_sm103_tst_128x8_64x12_4x1_v_bz_TNT` | LM head, m 8, n 128,256, k 4,096 | `lm_head` | directly comparable | 152.4 |
| other phases | 16 kernels: input preparation, slot mapping, block tables, Gumbel sampling, argmax, bookkeeping | runtime | none | runtime overhead | 41.8 |

Counts per iteration: 417 forward kernels (13 × 32 + the final norm) and 434
over all phases (417 + 7 in preprocess + 9 in sample + 1 in bookkeep).

**GEMMs.**

- **Shapes match.** Shapes follow from the model and vLLM's merged QKV and
  gate/up projections; the simulator's leaves use the same: QKV 6,144 × 4,096,
  O 4,096 × 4,096, gate/up 28,672 × 4,096, down 4,096 × 14,336, LM head
  128,256 × 4,096, all bf16, m = 8.
- **Same library family.** Both sides use cuBLAS `nvjet` kernels: the
  simulator's GB300 rows were measured with torch's matmul and `F.linear`.
- **Small differences remain:**
  - cuBLAS 13.1.0.3 in the profiling environment against 13.1.1.3 in the
    serving environment;
  - the serving engine's GEMMs run inside a CUDA graph and overlap their
    neighbours by about 2.6 %.
- **Split-K reduce.** The O and down projections are split-K with a separate
  reduce kernel. I counted the reduce with its GEMM. Whether the simulator's
  rows include a split-K reduce depends on cuBLAS choosing the same split for
  the same shape, which is not established.

**Attention.**

- The engine runs TRT-LLM's `fmhaSm100f` generation kernel (`MultiCtasKvCga`,
  8 query heads per KV group, KV tile 256), preceded by a fill that zeroes
  its multi-CTA counters. Total 312 µs per iteration.
- The simulator charges FlashInfer FA2 decode at 465 µs.
- The repository's `paged_attn_batch_decode` kind with backend
  `flashinfer_trtllm_gen` measures exactly this pair of launches. Its source
  is vLLM's `FlashInferImpl.forward` with HND layout. GB300 has no rows for
  it.

**Norms and activation.**

- **Fusion boundaries.** torch.compile fuses each residual add with the
  following RMSNorm (`fused_add_rms_norm`), and layer 0's embedding gather
  with its RMSNorm. SiLU and the multiply of the gate/up halves form one
  Inductor kernel (`mul_silu_slice`).
- **Simulator side.** It charges FlashInfer `rms_norm` without the residual
  add, a placeholder `elementwise` for the embedding, and a Triton
  `elementwise` for the activation.
- **Existing kinds.**
  - Residual add + RMSNorm: `residual_rms_norm` (backends `torch`,
    `vllm_cuda`), no Inductor backend.
  - Activation: `silu_and_mul` with backend `vllm_inductor`.
  - GB300 has rows for neither.

**KV append.** Same kernel as the simulator's `vllm_cuda` rows. The
simulator's arch passes `kv_cache_layout = "NHD"`, while TRT-LLM attention
uses the HND layout (per the `paged_attn_batch_decode` documentation). The
simulator also prices it 41 % above the trace (below). Layout and token
count are the first suspects; this audit does not prove which.

**RoPE.** One Inductor kernel per layer applies RoPE to Q and K. The
`llama3_dense` arch has no RoPE leaf, and no kernel kind for GQA RoPE exists
in the repository.

## Stage C: coverage and attribution

Shares of the iteration's GPU kernel time (sum, 3,476.8 µs median; constant
to 0.1 % across iterations):

| class | µs | share |
| --- | --- | --- |
| directly comparable: the five GEMMs | 2,776.9 | 79.9 % |
| comparable after layout adjustment: KV append | 71.3 | 2.0 % |
| comparable after implementation adjustment: attention + fill, activation | 354.9 | 10.2 % |
| fused with another operation: four norm roles incl. embedding | 175.0 | 5.0 % |
| unmodeled: RoPE | 56.4 | 1.6 % |
| runtime overhead: preprocess, sampling, bookkeeping kernels | 41.8 | 1.2 % |

**Do the missing operations matter?**

| missing from the simulator | µs per iteration | share |
| --- | --- | --- |
| RoPE | 56.4 | 1.6 % |
| runtime kernels | 41.8 | 1.2 % |
| GPU idle inside the iteration | 84.7 | 2.4 % |
| total | 182.9 | about 5 % |

Kernel overlap pulls the other way: the serial sum overstates GPU time by
87.5 µs (2.5 %). Each item alone is below 2.5 %. They are not negligible
against a ±5 % validation target, so they have to be modeled or accounted
for explicitly.

**Ordering.** The simulator's per-layer order (input norm, QKV, KV append,
decode attention, O, post norm, gate/up, activation, down) is the observed
order. The trace additionally has RoPE between QKV and the KV write, the fill
before attention, and split-K reduces after O and down. The simulator sums
leaves serially; the trace runs them back to back with about 2.6 % overlap.
The structure matches.

**Exploratory per-leaf comparison.** Median per iteration, µs. Not a
validation result; implementations differ for the rows below the line.

| simulator leaf | trace roles | measured | simulated | simulated / measured |
| --- | --- | --- | --- | --- |
| `qkv_proj` | QKV GEMM | 328.8 | 345.1 | 1.05 |
| `o_proj` | O GEMM + reduce | 342.7 | 317.7 | 0.93 |
| `up_gate_proj` | gate/up GEMM | 1,216.0 | 1,135.1 | 0.93 |
| `down_proj` | down GEMM + reduce | 737.0 | 689.6 | 0.94 |
| `lm_head` | LM head GEMM | 152.4 | 148.8 | 0.98 |
| `kv_cache_append` | KV write | 71.3 | 100.5 | 1.41 |
| `decode` (FA2) | TRT-LLM attention + fill | 312.1 | 464.6 | 1.49 |
| `activation` (Triton elementwise) | Inductor SiLU-and-mul | 42.8 | 73.7 | 1.72 |
| norms + embedding | Inductor fused add + RMSNorm (4 roles) | 175.0 | 202.8 | 1.16 |
| total | iteration | 3,474.1 (GPU cycle) | 3,478.0 | 1.00 |

The total agrees by accident:

- the simulated GEMMs are 7 % below the trace;
- the leaves with a different implementation are 41 to 72 % above it;
- RoPE, the runtime kernels and GPU idle (5.3 %) are missing from the
  simulation.

The GEMM shortfall is consistent across O, gate/up and down. This audit does
not settle whether it comes from the cuBLAS version, kernel selection, clocks
during profiling, or the overlap inside CUDA graphs.

## Stage D: recommendation

**B: a small vLLM-specific operation mapping, then a controlled TP1
validation.**

**Why not A:**

- 18 % of kernel time runs implementations the arch does not charge
  (attention, activation, fused norms, RoPE, KV layout);
- their simulated values differ from the trace by 16 to 72 %;
- any total-latency agreement would therefore be coincidental, as it is
  above.

**Why not C:**

- the execution graph is the simulator's: same operations, same order, one
  kernel per operation except split-K and the attention fill;
- no cross-operation fusion other than residual add into RMSNorm and
  embedding into the first RMSNorm;
- no overlap of whole operations;
- 80 % of time is directly comparable GEMMs.

A new arch is not needed; a variant of the leaf choices is.

**The mapping**, each item a later decision that needs approval:

1. **Attention:** use `paged_attn_batch_decode` / `paged_attn_batch_prefill`
   with backend `flashinfer_trtllm_gen`, which other vLLM archs already use.
   It needs GB300 rows (profiling).
2. **Activation:** use `silu_and_mul` with backend `vllm_inductor`. It needs
   GB300 rows.
3. **Norms:** charge residual add + RMSNorm as one leaf. Options: the
   existing `residual_rms_norm` (`vllm_cuda`, a near match), or a new
   Inductor backend for the exact kernel. Either needs GB300 rows; the second
   also needs a backend.
4. **KV append:** pass the HND layout the engine uses, and check the token
   count the leaf is costed at.
5. **RoPE:** either a new GQA RoPE kernel kind, or an explicit per-iteration
   adjustment of about 56 µs, documented as such.
6. **Runtime kernels and idle:** about 4 % together. Either a measured
   `gpu_time_multiplier` (the cycle over the busy time is 1.025 here, and the
   alignment analyzer computes it) or an explicit runtime leaf.
7. **GEMMs:** resolve the 7 % before claiming GEMM accuracy. One CPU-only
   step and one GPU step, both to approve:
   - compare kernel names and split-K choices between the profiler's rows and
     the trace;
   - a profiling check of the five shapes inside a CUDA graph.

Items 1 to 3 would be a vLLM-specific leaf selection for Llama TP1, a
simulator change. They come with profiling of three kinds on GB300. Nothing
of this has started.
