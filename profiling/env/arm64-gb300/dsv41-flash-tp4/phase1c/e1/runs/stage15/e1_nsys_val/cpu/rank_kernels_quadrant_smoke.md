# Per-rank execution, collective wait and kernel identities: `quadrant_nsys`

profile `/mnt/shared/home/dc/workspace/serving-studio/artifacts/phase0-dsv41-flash-gb300-tp4-20261112/run/quadrant_nsys/profile`, tables `/mnt/shared/home/dc/workspace/serving-studio/phase1a-tables/quadrant_nsys`. Pure-decode iterations only (decode_requests == b, no prefill, not JIT-excluded).
Sections A-D are observed on this pass alone. They say what the four ranks did; they do not by themselves
measure the routing hook's effect, which needs a hook-off pass under identical conditions.

## cell b=8: no pure-decode iterations

## cell b=48 (3032 pure-decode iterations)

### A. per-rank GPU execution (ms per iteration, p50 / p90)

| rank | span | busy | compute_union | comm_union | idle | api_launch_sum | api_sync_sum | gpu_tail_after_host | kernels | launches |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 9.9236 / 10.0947 | 9.7608 / 9.9295 | 9.3473 / 9.5085 | 0.7188 / 0.8061 | 0.1612 / 0.179 | 0.558 | 0.0 | 14.1486 | 1343.0 | 45.0 |
| 1 | 9.9238 / 10.0938 | 9.7626 / 9.9317 | 9.3737 / 9.5319 | 0.6939 / 0.7668 | 0.1604 / 0.1679 | 0.5736 | 0.0 | 14.158 | 1343.0 | 45.0 |
| 2 | 9.9224 / 10.0938 | 9.7574 / 9.9245 | 9.3879 / 9.5563 | 0.6561 / 0.7196 | 0.1635 / 0.1798 | 0.6349 | 0.0 | 13.6877 | 1343.0 | 45.0 |
| 3 | 9.9228 / 10.0936 | 9.7574 / 9.926 | 9.3939 / 9.5567 | 0.6568 / 0.721 | 0.164 / 0.1726 | 0.6237 | 0.0 | 13.8111 | 1343.0 | 45.0 |

### B. rank alignment

iterations with all 4 ranks: 3032; GPU windows overlap on all ranks: 3032
first-kernel start spread ms p50/p90/max: 0.0099 / 0.0154 / 1.9744
last-kernel end spread ms p50/p90/max: 0.0087 / 0.0124 / 0.0203
earliest-starting rank counts: {0: 2237, 1: 788, 2: 4, 3: 3}; latest-ending rank counts: {0: 1, 1: 1, 2: 1617, 3: 1413}

### C. collective wait (per iteration, ms; wait = rank comm time - fastest rank's comm time at the same collective)

| rank | wait_sum p50/p90 | comm_sum p50/p90 | wait share of comm (p50) |
|---|---|---|---|
| 0 | 0.1393 / 0.2227 | 0.7188 / 0.8061 | 0.1938 |
| 1 | 0.1144 / 0.1835 | 0.6939 / 0.7668 | 0.1648 |
| 2 | 0.0762 / 0.137 | 0.6561 / 0.7196 | 0.116 |
| 3 | 0.0768 / 0.1366 | 0.6568 / 0.721 | 0.1169 |

collectives per iteration p50 86.0; collective end spread across ranks ms p50/p90/max 0.00014 / 0.0003 / 0.0083; least-waiting (straggler) rank 2, most-waiting rank 0
per kind (Phase 1A kind code): kind 2: n=982368 wait p50 0.0007 p90 0.0029 comm p50 0.0065; kind 4: n=60640 wait p50 0.0005 p90 0.0023 comm p50 0.017

### D. kernel identities (top 10 by GPU time; 95 distinct names; kernels/iteration per rank {0: 1343.87, 1: 1343.87, 2: 1343.87, 3: 1343.87})

| # | kernel | MoE/GEMM | count/iter (4 ranks) | mean us | share | streams |
|---|---|---|---|---|---|---|
| 1 | `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x16x256u2_s6_et128x16_m256x16x32` | x | 160.0 | 76.46 | 0.2702 | 5 |
| 2 | `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x16x256_s6_et128x16_m256x16x32_c2x1` | x | 160.0 | 41.14 | 0.1454 | 5 |
| 3 | `void deep_gemm::sm100_mega_mhc_impl<(unsigned int)5120, (unsigned int)40, (unsig` | x | 308.0 | 14.17 | 0.0964 | 6 |
| 4 | `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100Bloc` | x | 520.0 | 6.32 | 0.0726 | 7 |
| 5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H512PagedKvDenseDynamicTokenSparseP1MultiCtasKv` | x | 152.0 | 17.99 | 0.0604 | 5 |
| 6 | `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100Bloc` | x | 320.0 | 8.05 | 0.0569 | 7 |
| 7 | `void flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceKernel<(unsigned char)4` | x | 324.0 | 7.18 | 0.0514 | 7 |
| 8 | `kernel_cutlass_device_kernel_tensorptrbf16gmemalign16o4816512div85121_tensorptri` | x | 160.0 | 6.18 | 0.0219 | 6 |
| 9 | `nvjet_sm103_tss_64x16_64x16_2x4_2cta_h_bz_splitK_TNT` | x | 164.0 | 5.45 | 0.0198 | 5 |
| 10 | `_dsv4_topk_kernel` |  | 160.0 | 4.39 | 0.0155 | 5 |

