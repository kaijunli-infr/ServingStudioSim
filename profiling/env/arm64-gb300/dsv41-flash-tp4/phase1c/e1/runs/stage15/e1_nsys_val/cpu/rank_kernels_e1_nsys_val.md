# Per-rank execution, collective wait and kernel identities: `e1_nsys_val`

profile `/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15/e1_nsys_val/profile`, tables `/mnt/shared/home/dc/workspace/serving-studio/phase1a-tables/e1_nsys_val`. Pure-decode iterations only (decode_requests == b, no prefill, not JIT-excluded).
Sections A-D are observed on this pass alone. They say what the four ranks did; they do not by themselves
measure the routing hook's effect, which needs a hook-off pass under identical conditions.

## cell b=8 (2 pure-decode iterations)

### A. per-rank GPU execution (ms per iteration, p50 / p90)

| rank | span | busy | compute_union | comm_union | idle | api_launch_sum | api_sync_sum | gpu_tail_after_host | kernels | launches |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 5.192 / 5.1921 | 5.0427 / 5.0454 | 4.4437 / 4.4501 | 0.6693 / 0.674 | 0.1493 / 0.1519 | 0.8054 | 0.0 | 5.3482 | 1410.0 | 47.0 |
| 1 | 5.1931 / 5.1946 | 5.049 / 5.0506 | 4.5021 / 4.5122 | 0.6171 / 0.6261 | 0.1441 / 0.1441 | 0.7159 | 0.0 | 5.8711 | 1409.0 | 46.0 |
| 2 | 5.1942 / 5.1975 | 5.0481 / 5.0515 | 4.5209 / 4.5337 | 0.5959 / 0.6052 | 0.146 / 0.1461 | 0.8233 | 0.0 | 5.4541 | 1409.0 | 46.0 |
| 3 | 5.1922 / 5.1952 | 5.051 / 5.0523 | 4.4805 / 4.4841 | 0.6409 / 0.6463 | 0.1412 / 0.1456 | 0.8598 | 0.0 | 5.2629 | 1409.0 | 46.0 |

### B. rank alignment

iterations with all 4 ranks: 2; GPU windows overlap on all ranks: 2
first-kernel start spread ms p50/p90/max: 0.0027 / 0.0031 / 0.0032
last-kernel end spread ms p50/p90/max: 0.0047 / 0.0067 / 0.0072
earliest-starting rank counts: {1: 2}; latest-ending rank counts: {2: 2}

### C. collective wait (per iteration, ms; wait = rank comm time - fastest rank's comm time at the same collective)

| rank | wait_sum p50/p90 | comm_sum p50/p90 | wait share of comm (p50) |
|---|---|---|---|
| 0 | 0.1942 / 0.1985 | 0.6874 / 0.6919 | 0.2824 |
| 1 | 0.1418 / 0.151 | 0.635 / 0.644 | 0.223 |
| 2 | 0.1202 / 0.1295 | 0.6134 / 0.6225 | 0.1956 |
| 3 | 0.165 / 0.1704 | 0.6582 / 0.6634 | 0.2506 |

collectives per iteration p50 163.0; collective end spread across ranks ms p50/p90/max 0.0009 / 0.00738 / 0.012; least-waiting (straggler) rank 2, most-waiting rank 0
per kind (Phase 1A kind code): kind 0: n=1232 wait p50 0.0001 p90 0.0025 comm p50 0.0031; kind 1: n=32 wait p50 0.0014 p90 0.0094 comm p50 0.0056; kind 4: n=40 wait p50 0.0011 p90 0.0031 comm p50 0.0121

### D. kernel identities (top 25 by GPU time; 94 distinct names; kernels/iteration per rank {0: 1410.0, 1: 1409.0, 2: 1409.0, 3: 1409.0})

| # | kernel | MoE/GEMM | count/iter (4 ranks) | mean us | share | streams |
|---|---|---|---|---|---|---|
| 1 | `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x8x512u2_s3_et128x8_m128x8x32_c1` | x | 148.0 | 26.04 | 0.1373 | 7 |
| 2 | `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100Bloc` | x | 520.0 | 5.09 | 0.0944 | 9 |
| 3 | `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x8x256_s5_et128x8_m128x8x32_c1x1x1_` | x | 148.0 | 15.74 | 0.083 | 7 |
| 4 | `void deep_gemm::sm100_tf32_hc_prenorm_gemm_impl<(unsigned int)24, (unsigned int)` | x | 316.0 | 6.02 | 0.0678 | 7 |
| 5 | `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100_splitkSm` | x | 320.0 | 5.56 | 0.0634 | 9 |
| 6 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` |  | 332.0 | 5.22 | 0.0618 | 7 |
| 7 | `kernel_cutlass_kernel_vllmmodelsdeepseek_v41nvidiaopscute_dslall_reduce_mhc_Lamp` | x | 308.0 | 5.38 | 0.0591 | 7 |
| 8 | `fmhaSm100fKernel_QkvE4m3OBfloat16H512HVPerCta128PagedKvDenseDynamicTokenSparseP1` | x | 152.0 | 10.37 | 0.0562 | 7 |
| 9 | `_dsv4_topk_kernel` |  | 160.0 | 5.3 | 0.0302 | 7 |
| 10 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSw` | x | 320.0 | 2.17 | 0.0248 | 9 |
| 11 | `kernel_cutlass_device_kernel_tensorptrbf16gmemalign16o816512div85121_tensorptri6` | x | 160.0 | 4.26 | 0.0243 | 7 |
| 12 | `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_ll_bf16_splitkLLBf` | x | 160.0 | 3.99 | 0.0228 | 7 |
| 13 | `void moe::dev::routing::routingCustom::routingIndicesDynBlockKernel<moe::dev::ro` | x | 160.0 | 3.23 | 0.0184 | 7 |
| 14 | `void vllm::deepseek_v4_fused_ops::fusedDeepseekV4FullCacheKernel<c10::BFloat16, ` |  | 160.0 | 3.2 | 0.0182 | 7 |
| 15 | `_build_flashinfer_mixed_sparse_indices_kernel` |  | 160.0 | 2.85 | 0.0162 | 7 |
| 16 | `_q_kv_norm_quant_kernel` |  | 160.0 | 2.6 | 0.0148 | 7 |
| 17 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kerne` |  | 160.0 | 2.33 | 0.0133 | 7 |
| 18 | `kernel_cutlass_kernel_vllmmodelsdeepseek_v41nvidiaopscute_dslall_reduce_mhc_Quad` | x | 148.0 | 2.26 | 0.0119 | 6 |
| 19 | `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x8x512_s3_et128x8_m128x8x32_c1x1` | x | 12.0 | 27.33 | 0.0117 | 3 |
| 20 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &vllm::silu_kernel<` |  | 160.0 | 1.98 | 0.0113 | 3 |
| 21 | `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cu` |  | 4.0 | 68.64 | 0.0098 | 1 |
| 22 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSw` | x | 160.0 | 1.72 | 0.0098 | 3 |
| 23 | `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(` | x | 32.0 | 8.25 | 0.0094 | 3 |
| 24 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |  | 20.0 | 13.01 | 0.0093 | 5 |
| 25 | `kernel_cutlass_kernel_vllmmodelsdeepseek_v41nvidiaopscute_dslall_reduce_mhc_Shar` | x | 160.0 | 1.44 | 0.0082 | 7 |

## cell b=48 (928 pure-decode iterations)

### A. per-rank GPU execution (ms per iteration, p50 / p90)

| rank | span | busy | compute_union | comm_union | idle | api_launch_sum | api_sync_sum | gpu_tail_after_host | kernels | launches |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 10.2962 / 10.421 | 10.1315 / 10.2543 | 9.6942 / 9.8025 | 0.7758 / 0.8774 | 0.1612 / 0.1916 | 0.5816 | 0.0 | 14.0906 | 1386.0 | 48.0 |
| 1 | 10.2963 / 10.4215 | 10.1335 / 10.2568 | 9.7603 / 9.8642 | 0.6767 / 0.7434 | 0.1606 / 0.1752 | 0.5522 | 0.0 | 14.2916 | 1385.0 | 47.0 |
| 2 | 10.2954 / 10.4193 | 10.1301 / 10.2514 | 9.7656 / 9.8739 | 0.666 / 0.7295 | 0.1629 / 0.1908 | 0.6068 | 0.0 | 14.1098 | 1385.0 | 47.0 |
| 3 | 10.2942 / 10.4204 | 10.1268 / 10.2543 | 9.7622 / 9.8759 | 0.6639 / 0.7351 | 0.1618 / 0.1857 | 0.6412 | 0.0 | 13.8224 | 1385.0 | 47.0 |

### B. rank alignment

iterations with all 4 ranks: 928; GPU windows overlap on all ranks: 928
first-kernel start spread ms p50/p90/max: 0.0074 / 0.0132 / 1.5694
last-kernel end spread ms p50/p90/max: 0.006 / 0.0086 / 0.0163
earliest-starting rank counts: {0: 172, 1: 742, 2: 11, 3: 3}; latest-ending rank counts: {0: 19, 1: 3, 2: 466, 3: 440}

### C. collective wait (per iteration, ms; wait = rank comm time - fastest rank's comm time at the same collective)

| rank | wait_sum p50/p90 | comm_sum p50/p90 | wait share of comm (p50) |
|---|---|---|---|
| 0 | 0.1928 / 0.2891 | 0.7758 / 0.8774 | 0.2486 |
| 1 | 0.0935 / 0.158 | 0.6767 / 0.7434 | 0.1387 |
| 2 | 0.0829 / 0.1449 | 0.666 / 0.7295 | 0.1243 |
| 3 | 0.0809 / 0.1488 | 0.6639 / 0.7351 | 0.1223 |

collectives per iteration p50 86.0; collective end spread across ranks ms p50/p90/max 0.00014 / 0.0003 / 0.0053; least-waiting (straggler) rank 3, most-waiting rank 0
per kind (Phase 1A kind code): kind 2: n=300672 wait p50 0.0008 p90 0.0034 comm p50 0.0067; kind 4: n=18560 wait p50 0.0004 p90 0.0022 comm p50 0.017

### D. kernel identities (top 25 by GPU time; 97 distinct names; kernels/iteration per rank {0: 1385.06, 1: 1384.06, 2: 1384.06, 3: 1384.06})

| # | kernel | MoE/GEMM | count/iter (4 ranks) | mean us | share | streams |
|---|---|---|---|---|---|---|
| 1 | `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x16x256u2_s6_et128x16_m256x16x32` | x | 160.0 | 77.78 | 0.2639 | 5 |
| 2 | `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x16x256_s6_et128x16_m256x16x32_c2x1` | x | 160.0 | 41.59 | 0.1411 | 5 |
| 3 | `void deep_gemm::sm100_mega_mhc_impl<(unsigned int)5120, (unsigned int)40, (unsig` | x | 308.0 | 14.23 | 0.093 | 6 |
| 4 | `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100Bloc` | x | 480.0 | 7.93 | 0.0807 | 7 |
| 5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H512PagedKvDenseDynamicTokenSparseP1MultiCtasKv` | x | 152.0 | 18.82 | 0.0607 | 5 |
| 6 | `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100Bloc` | x | 360.0 | 7.27 | 0.0555 | 7 |
| 7 | `void flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceKernel<(unsigned char)4` | x | 324.0 | 7.33 | 0.0504 | 7 |
| 8 | `kernel_cutlass_device_kernel_tensorptrbf16gmemalign16o4816512div85121_tensorptri` | x | 160.0 | 6.18 | 0.021 | 6 |
| 9 | `nvjet_sm103_tss_64x16_64x16_2x4_2cta_h_bz_splitK_TNT` | x | 164.0 | 5.51 | 0.0192 | 5 |
| 10 | `_dsv4_topk_kernel` |  | 160.0 | 4.29 | 0.0146 | 5 |
| 11 | `void moe::dev::finalize::finalizeKernel<moe::dev::finalize::KernelParams<cutlass` | x | 160.0 | 4.26 | 0.0145 | 5 |
| 12 | `void moe::dev::routing::routingCustom::routingIndicesClusterKernel<moe::dev::rou` | x | 160.0 | 4.04 | 0.0137 | 5 |
| 13 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, f` | x | 176.0 | 3.5 | 0.013 | 5 |
| 14 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSw` | x | 320.0 | 1.76 | 0.012 | 7 |
| 15 | `void vllm::deepseek_v4_fused_ops::fusedDeepseekV4FullCacheKernel<c10::BFloat16, ` |  | 160.0 | 3.4 | 0.0115 | 6 |
| 16 | `_build_flashinfer_mixed_sparse_indices_kernel` |  | 160.0 | 2.98 | 0.0101 | 6 |
| 17 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |  | 20.0 | 23.36 | 0.0099 | 5 |
| 18 | `_q_kv_norm_quant_kernel` |  | 160.0 | 2.31 | 0.0078 | 6 |
| 19 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSw` | x | 160.0 | 2.3 | 0.0078 | 3 |
| 20 | `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &vllm::silu_kernel<` |  | 160.0 | 2.08 | 0.0071 | 3 |
| 21 | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_a` |  | 160.0 | 1.79 | 0.0061 | 7 |
| 22 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kerne` |  | 160.0 | 1.71 | 0.0058 | 5 |
| 23 | `void deep_gemm::sm100_paged_sparse_mqa_logits<(unsigned int)128, (unsigned int)8` | x | 16.0 | 15.63 | 0.0053 | 4 |
| 24 | `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeLi` | x | 160.0 | 1.44 | 0.0049 | 5 |
| 25 | `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, s` |  | 160.0 | 1.4 | 0.0047 | 6 |

### E. cross-session comparison with `quadrant_nsys` (3032 pure-decode iterations at b=48) — HYPOTHESIS MATERIAL ONLY

`quadrant_nsys` was a different session with a different workload backend and the hook OFF; `e1_nsys_val` is hook ON. Any difference below
mixes session-to-session drift (FlashInfer re-tuning; Stage 1.5 showed ~5 % between sessions at b=48) with any hook effect and cannot separate them.

kernel names only in `e1_nsys_val` (2): `void at::native::elementwise_kernel<(int)128, (int)2, void at::native:`, `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_k`
kernel names only in `quadrant_nsys` (0): 

per-rank busy ms p50 this / ref: r0: 10.1315 / 9.7608; r1: 10.1335 / 9.7626; r2: 10.1301 / 9.7574; r3: 10.1268 / 9.7574
per-rank span ms p50 this / ref: r0: 10.2962 / 9.9236; r1: 10.2963 / 9.9238; r2: 10.2954 / 9.9224; r3: 10.2942 / 9.9228

common kernels whose count/iteration or time/iteration differs (top 20 by time difference):

| kernel | count/iter e1_nsys_val | count/iter quadrant_nsys | ms/iter e1_nsys_val | ms/iter quadrant_nsys |
|---|---|---|---|---|
| `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPer` | 480.0 | 320.0 | 3.8051 | 2.5769 |
| `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPer` | 360.0 | 520.0 | 2.6168 | 3.2885 |
| `void vllm::cooperative::cooperative_topk_cs2<(unsigned int)512>(vllm::cooperative::Coopera` | 16.0 | 16.0 | 0.2001 | 0.0681 |
| `void deep_gemm::sm100_paged_mqa_logits<(unsigned int)1, (unsigned int)32, (unsigned int)12` | 12.0 | 12.0 | 0.1926 | 0.0775 |
| `void deep_gemm::sm100_paged_sparse_mqa_logits<(unsigned int)128, (unsigned int)8, (unsigne` | 16.0 | 16.0 | 0.25 | 0.136 |
| `void deep_gemm::sm100_paged_mqa_logits<(unsigned int)1, (unsigned int)32, (unsigned int)12` | 4.0 | 4.0 | 0.1004 | 0.0301 |
| `void topk_select_bf16_normal::topk_kernel<topk_select_bf16_normal::TopkSelectKernelBF16Nor` | 16.0 | 16.0 | 0.1034 | 0.0766 |

