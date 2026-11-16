# Per-rank execution, collective wait and kernel identities: `e1_b8_recovery`

profile `/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15/e1_b8_recovery/profile`, tables `/mnt/shared/home/dc/workspace/serving-studio/phase1a-tables/e1_b8_recovery`. Pure-decode iterations only (decode_requests == b, no prefill, not JIT-excluded).
Sections A-D are observed on this pass alone. They say what the four ranks did; they do not by themselves
measure the routing hook's effect, which needs a hook-off pass under identical conditions.

## cell b=8 (1008 pure-decode iterations)

### A. per-rank GPU execution (ms per iteration, p50 / p90)

| rank | span | busy | compute_union | comm_union | idle | api_launch_sum | api_sync_sum | gpu_tail_after_host | kernels | launches |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 5.2789 / 5.3531 | 5.1305 / 5.2022 | 4.4999 / 4.558 | 0.6944 / 0.7611 | 0.1489 / 0.1605 | 0.5806 | 0.0 | 5.7881 | 1410.0 | 47.0 |
| 1 | 5.2791 / 5.3526 | 5.1286 / 5.2046 | 4.5538 / 4.6094 | 0.6394 / 0.6952 | 0.149 / 0.1626 | 0.5653 | 0.0 | 5.9216 | 1409.0 | 46.0 |
| 2 | 5.2778 / 5.3531 | 5.1255 / 5.1992 | 4.5616 / 4.6275 | 0.6287 / 0.6757 | 0.1512 / 0.1672 | 0.6816 | 0.0 | 5.4381 | 1409.0 | 46.0 |
| 3 | 5.2785 / 5.3532 | 5.1268 / 5.2 | 4.5496 / 4.6136 | 0.6448 / 0.6982 | 0.1509 / 0.1646 | 0.643 | 0.0 | 5.5876 | 1409.0 | 46.0 |

### B. rank alignment

iterations with all 4 ranks: 1008; GPU windows overlap on all ranks: 1008
first-kernel start spread ms p50/p90/max: 0.0041 / 0.009 / 0.0172
last-kernel end spread ms p50/p90/max: 0.0036 / 0.0064 / 0.0115
earliest-starting rank counts: {0: 174, 1: 777, 2: 17, 3: 40}; latest-ending rank counts: {0: 67, 1: 4, 2: 575, 3: 362}

### C. collective wait (per iteration, ms; wait = rank comm time - fastest rank's comm time at the same collective)

| rank | wait_sum p50/p90 | comm_sum p50/p90 | wait share of comm (p50) |
|---|---|---|---|
| 0 | 0.2196 / 0.2799 | 0.7118 / 0.7786 | 0.3088 |
| 1 | 0.1637 / 0.2166 | 0.6563 / 0.7126 | 0.2492 |
| 2 | 0.1535 / 0.1959 | 0.646 / 0.6931 | 0.2379 |
| 3 | 0.1695 / 0.2192 | 0.6618 / 0.7156 | 0.2559 |

collectives per iteration p50 163.0; collective end spread across ranks ms p50/p90/max 0.00096 / 0.00795 / 0.0249; least-waiting (straggler) rank 2, most-waiting rank 0
per kind (Phase 1A kind code): kind 0: n=620928 wait p50 0.0001 p90 0.0033 comm p50 0.0034; kind 1: n=16128 wait p50 0.002 p90 0.0096 comm p50 0.0063; kind 4: n=20160 wait p50 0.0009 p90 0.0028 comm p50 0.0122

### D. kernel identities (top 25 by GPU time; 95 distinct names; kernels/iteration per rank {0: 1410.49, 1: 1409.49, 2: 1409.49, 3: 1409.49})

| # | kernel | MoE/GEMM | count/iter (4 ranks) | mean us | share | streams |
|---|---|---|---|---|---|---|
| 1 | `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x8x256_s6_et128x8_m128x8x32_c1x1` | x | 148.0 | 27.15 | 0.1413 | 7 |
| 2 | `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100Bloc` | x | 520.0 | 5.08 | 0.0929 | 9 |
| 3 | `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x8x256_s6_et128x8_m128x8x32_c1x1x1_` | x | 160.0 | 16.31 | 0.0918 | 7 |
| 4 | `void deep_gemm::sm100_tf32_hc_prenorm_gemm_impl<(unsigned int)24, (unsigned int)` | x | 316.0 | 6.05 | 0.0673 | 7 |
| 5 | `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100_splitkSm` | x | 320.0 | 5.57 | 0.0627 | 9 |
| 6 | `kernel_cutlass_kernel_vllmmodelsdeepseek_v41nvidiaopscute_dslall_reduce_mhc_Lamp` | x | 308.0 | 5.66 | 0.0613 | 7 |
| 7 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` |  | 332.0 | 5.23 | 0.0611 | 7 |
| 8 | `fmhaSm100fKernel_QkvE4m3OBfloat16H512HVPerCta128PagedKvDenseDynamicTokenSparseP1` | x | 152.0 | 10.36 | 0.0554 | 7 |
| 9 | `_dsv4_topk_kernel` |  | 160.0 | 5.02 | 0.0283 | 7 |
| 10 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSw` | x | 320.0 | 2.16 | 0.0243 | 9 |
| 11 | `kernel_cutlass_device_kernel_tensorptrbf16gmemalign16o816512div85121_tensorptri6` | x | 160.0 | 4.29 | 0.0241 | 7 |
| 12 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_ll_bf16_splitkLLBf` | x | 160.0 | 3.98 | 0.0224 | 7 |
| 13 | `void moe::dev::routing::routingCustom::routingIndicesDynBlockKernel<moe::dev::ro` | x | 160.0 | 3.22 | 0.0181 | 7 |
| 14 | `void vllm::deepseek_v4_fused_ops::fusedDeepseekV4FullCacheKernel<c10::BFloat16, ` |  | 160.0 | 3.15 | 0.0177 | 7 |
| 15 | `_build_flashinfer_mixed_sparse_indices_kernel` |  | 160.0 | 2.86 | 0.0161 | 7 |
| 16 | `_q_kv_norm_quant_kernel` |  | 160.0 | 2.61 | 0.0147 | 7 |
| 17 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kerne` |  | 160.0 | 2.36 | 0.0133 | 7 |
| 18 | `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x8x512_s3_et128x8_m128x8x32_c1x1` | x | 12.0 | 28.13 | 0.0119 | 3 |
| 19 | `kernel_cutlass_kernel_vllmmodelsdeepseek_v41nvidiaopscute_dslall_reduce_mhc_Quad` | x | 148.0 | 2.25 | 0.0117 | 6 |
| 20 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &vllm::silu_kernel<` |  | 160.0 | 1.97 | 0.0111 | 3 |
| 21 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSw` | x | 160.0 | 1.76 | 0.0099 | 3 |
| 22 | `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cu` |  | 4.0 | 68.67 | 0.0097 | 1 |
| 23 | `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(` | x | 32.0 | 8.3 | 0.0093 | 3 |
| 24 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |  | 20.0 | 13.04 | 0.0092 | 5 |
| 25 | `kernel_cutlass_kernel_vllmmodelsdeepseek_v41nvidiaopscute_dslall_reduce_mhc_Shar` | x | 160.0 | 1.46 | 0.0082 | 7 |

### E. cross-session comparison with `diverse_nsys_run2` (1675 pure-decode iterations at b=8) — HYPOTHESIS MATERIAL ONLY

`diverse_nsys_run2` was a different session with a different workload backend and the hook OFF; `e1_b8_recovery` is hook ON. Any difference below
mixes session-to-session drift (FlashInfer re-tuning; Stage 1.5 showed ~5 % between sessions at b=48) with any hook effect and cannot separate them.

kernel names only in `e1_b8_recovery` (4): `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x8x256_s6_et128x8_m128x8x`, `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x8x512_s3_et128x8_m128`, `void at::native::elementwise_kernel<(int)128, (int)2, void at::native:`, `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_k`
kernel names only in `diverse_nsys_run2` (3): `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x8x256_s5_et128x8_m128x8x`, `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x8x512_s3_et128x8_m128x8x`, `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x8x512u2_s3_et128x8_m1`

per-rank busy ms p50 this / ref: r0: 5.1305 / 5.0376; r1: 5.1286 / 5.0395; r2: 5.1255 / 5.0361; r3: 5.1268 / 5.0349
per-rank span ms p50 this / ref: r0: 5.2789 / 5.1858; r1: 5.2791 / 5.1862; r2: 5.2778 / 5.1862; r3: 5.2785 / 5.1857

