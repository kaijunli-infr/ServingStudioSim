# Autotune tactic choices across launches and tactic-dependent GPU time (C3)

## 1. Pairwise agreement of the per-launch autotune caches

| launch A | launch B | common keys | differing | fraction | only A / only B | differing by op (diff/common) |
|---|---|---|---|---|---|---|
| stage1_off | stage1_on | 160 | 87 | 0.544 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 8/27; mxfp8_gemm 78/132 |
| stage1_off | off_v2 | 160 | 83 | 0.519 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 10/27; mxfp8_gemm 72/132 |
| stage1_off | on_v2 | 160 | 89 | 0.556 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 11/27; mxfp8_gemm 77/132 |
| stage1_off | off2_v2 | 160 | 86 | 0.537 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 13/27; mxfp8_gemm 72/132 |
| stage1_off | nsys_val | 160 | 99 | 0.619 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 11/27; mxfp8_gemm 87/132 |
| stage1_off | b8_recovery | 160 | 104 | 0.65 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 12/27; mxfp8_gemm 91/132 |
| stage1_on | off_v2 | 160 | 86 | 0.537 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 10/27; mxfp8_gemm 75/132 |
| stage1_on | on_v2 | 160 | 107 | 0.669 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 12/27; mxfp8_gemm 94/132 |
| stage1_on | off2_v2 | 160 | 94 | 0.588 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 13/27; mxfp8_gemm 80/132 |
| stage1_on | nsys_val | 160 | 102 | 0.637 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 12/27; mxfp8_gemm 89/132 |
| stage1_on | b8_recovery | 160 | 99 | 0.619 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 13/27; mxfp8_gemm 85/132 |
| off_v2 | on_v2 | 160 | 88 | 0.55 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 11/27; mxfp8_gemm 76/132 |
| off_v2 | off2_v2 | 160 | 93 | 0.581 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 14/27; mxfp8_gemm 78/132 |
| off_v2 | nsys_val | 160 | 98 | 0.613 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 12/27; mxfp8_gemm 85/132 |
| off_v2 | b8_recovery | 160 | 96 | 0.6 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 14/27; mxfp8_gemm 81/132 |
| on_v2 | off2_v2 | 160 | 94 | 0.588 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 10/27; mxfp8_gemm 83/132 |
| on_v2 | nsys_val | 160 | 103 | 0.644 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 13/27; mxfp8_gemm 89/132 |
| on_v2 | b8_recovery | 160 | 94 | 0.588 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 10/27; mxfp8_gemm 83/132 |
| off2_v2 | nsys_val | 160 | 103 | 0.644 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 15/27; mxfp8_gemm 87/132 |
| off2_v2 | b8_recovery | 160 | 100 | 0.625 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 11/27; mxfp8_gemm 88/132 |
| nsys_val | b8_recovery | 160 | 82 | 0.512 | 0 / 0 | _generation 1/1; trtllm_fp4_block_scale_moe 12/27; mxfp8_gemm 69/132 |

## 2. GPU time per pure-decode iteration by kernel family (4 ranks, ms)

| pass | b | n | total | moe_bmm | dense_gemm | quantize | moe_routing | attention | collective | other | tactic-dependent (bmm+gemm+quant) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e1_nsys_val | 48 | 928 | 47.1574 | 19.0992 | 12.5608 | 1.1783 | 2.5438 | 3.4496 | 3.0283 | 5.2973 | 32.8383 |
| quadrant_nsys | 48 | 3032 | 45.2722 | 18.8148 | 11.6809 | 1.1776 | 2.4053 | 3.3177 | 2.9886 | 4.8873 | 31.6733 |
| e1_b8_recovery | 8 | 1008 | 28.4282 | 6.9653 | 8.0606 | 1.2079 | 1.9406 | 2.131 | 4.5036 | 3.6192 | 16.2337 |
| diverse_nsys_run2 | 8 | 1675 | 28.3142 | 6.9586 | 8.1561 | 1.2722 | 1.9834 | 2.1313 | 4.5937 | 3.2189 | 16.387 |

### top kernels `e1_nsys_val` b=48 (count/iter over 4 ranks, ms/iter)

| kernel | family | count/iter | ms/iter |
|---|---|---|---|
| `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x16x256u2_s6_et128x16_m256x16x32_c2x1x1_rM_TN_transO` | moe_bmm | 160.0 | 12.4442 |
| `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x16x256_s6_et128x16_m256x16x32_c2x1x1_rM_TN_transOut_sc` | moe_bmm | 160.0 | 6.655 |
| `void deep_gemm::sm100_mega_mhc_impl<(unsigned int)5120, (unsigned int)40, (unsigned int)152, (bool)1` | dense_gemm | 308.0 | 4.3842 |
| `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDen` | dense_gemm | 480.0 | 3.8051 |
| `fmhaSm100fKernel_QkvE4m3OBfloat16H512PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ16Kv128Static` | attention | 152.0 | 2.8608 |
| `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDen` | dense_gemm | 360.0 | 2.6168 |
| `void flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceKernel<(unsigned char)4, __nv_bfloat16, (bo` | collective | 324.0 | 2.3744 |
| `kernel_cutlass_device_kernel_tensorptrbf16gmemalign16o4816512div85121_tensorptri64gmemo481_tensorptr` | dense_gemm | 160.0 | 0.9896 |
| `nvjet_sm103_tss_64x16_64x16_2x4_2cta_h_bz_splitK_TNT` | other | 164.0 | 0.9043 |
| `_dsv4_topk_kernel` | moe_routing | 160.0 | 0.6863 |
| `void moe::dev::finalize::finalizeKernel<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, float,` | moe_routing | 160.0 | 0.682 |
| `void moe::dev::routing::routingCustom::routingIndicesClusterKernel<moe::dev::routing::routingCustom:` | moe_routing | 160.0 | 0.647 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float` | other | 176.0 | 0.6154 |
| `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_` | quantize | 320.0 | 0.5644 |
| `void vllm::deepseek_v4_fused_ops::fusedDeepseekV4FullCacheKernel<c10::BFloat16, (bool)1, (bool)1, (b` | other | 160.0 | 0.5439 |
| `_build_flashinfer_mixed_sparse_indices_kernel` | attention | 160.0 | 0.4775 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | collective | 20.0 | 0.4671 |
| `_q_kv_norm_quant_kernel` | other | 160.0 | 0.3688 |
| `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_` | quantize | 160.0 | 0.3683 |
| `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &vllm::silu_kernel<c10::BFloat16>, &vll` | other | 160.0 | 0.3335 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10::BFloat16>, s` | other | 160.0 | 0.2865 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::na` | other | 160.0 | 0.2729 |
| `void deep_gemm::sm100_paged_sparse_mqa_logits<(unsigned int)128, (unsigned int)8, (unsigned int)2, (` | dense_gemm | 16.0 | 0.25 |
| `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeLinearKernel_object_at` | quantize | 160.0 | 0.2304 |
| `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char *, (u` | other | 160.0 | 0.2234 |
| `nvjet_sm103_tst_256x48_64x5_4x1_v_bz_TNT` | other | 4.0 | 0.2216 |
| `void vllm::cooperative::cooperative_topk_cs2<(unsigned int)512>(vllm::cooperative::CooperativeTopKPa` | moe_routing | 16.0 | 0.2001 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::FillFunctor<unsigned char>, std::` | other | 160.0 | 0.1954 |
| `void deep_gemm::sm100_paged_mqa_logits<(unsigned int)1, (unsigned int)32, (unsigned int)128, (unsign` | dense_gemm | 12.0 | 0.1926 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::na` | other | 24.0 | 0.1579 |

### top kernels `quadrant_nsys` b=48 (count/iter over 4 ranks, ms/iter)

| kernel | family | count/iter | ms/iter |
|---|---|---|---|
| `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x16x256u2_s6_et128x16_m256x16x32_c2x1x1_rM_TN_transO` | moe_bmm | 160.0 | 12.233 |
| `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x16x256_s6_et128x16_m256x16x32_c2x1x1_rM_TN_transOut_sc` | moe_bmm | 160.0 | 6.5818 |
| `void deep_gemm::sm100_mega_mhc_impl<(unsigned int)5120, (unsigned int)40, (unsigned int)152, (bool)1` | dense_gemm | 308.0 | 4.3658 |
| `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDen` | dense_gemm | 520.0 | 3.2885 |
| `fmhaSm100fKernel_QkvE4m3OBfloat16H512PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ16Kv128Static` | attention | 152.0 | 2.735 |
| `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDen` | dense_gemm | 320.0 | 2.5769 |
| `void flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceKernel<(unsigned char)4, __nv_bfloat16, (bo` | collective | 324.0 | 2.3253 |
| `kernel_cutlass_device_kernel_tensorptrbf16gmemalign16o4816512div85121_tensorptri64gmemo481_tensorptr` | dense_gemm | 160.0 | 0.9896 |
| `nvjet_sm103_tss_64x16_64x16_2x4_2cta_h_bz_splitK_TNT` | other | 164.0 | 0.8944 |
| `_dsv4_topk_kernel` | moe_routing | 160.0 | 0.7025 |
| `void moe::dev::finalize::finalizeKernel<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, float,` | moe_routing | 160.0 | 0.6859 |
| `void moe::dev::routing::routingCustom::routingIndicesClusterKernel<moe::dev::routing::routingCustom:` | moe_routing | 160.0 | 0.6502 |
| `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_` | quantize | 320.0 | 0.5707 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float` | other | 176.0 | 0.5414 |
| `void vllm::deepseek_v4_fused_ops::fusedDeepseekV4FullCacheKernel<c10::BFloat16, (bool)1, (bool)1, (b` | other | 160.0 | 0.534 |
| `_build_flashinfer_mixed_sparse_indices_kernel` | attention | 160.0 | 0.4717 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | collective | 20.0 | 0.4711 |
| `_q_kv_norm_quant_kernel` | other | 160.0 | 0.371 |
| `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_` | quantize | 160.0 | 0.3628 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10::BFloat16>, s` | other | 160.0 | 0.2913 |
| `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &vllm::silu_kernel<c10::BFloat16>, &vll` | other | 160.0 | 0.2829 |
| `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeLinearKernel_object_at` | quantize | 160.0 | 0.229 |
| `nvjet_sm103_tst_256x48_64x5_4x1_v_bz_TNT` | other | 4.0 | 0.2224 |
| `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char *, (u` | other | 160.0 | 0.2108 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::FillFunctor<unsigned char>, std::` | other | 160.0 | 0.1961 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::na` | other | 24.0 | 0.1635 |
| `void deep_gemm::sm100_paged_sparse_mqa_logits<(unsigned int)128, (unsigned int)8, (unsigned int)2, (` | dense_gemm | 16.0 | 0.136 |
| `_engram_lookup_kernel` | other | 8.0 | 0.1201 |
| `nvjet_sm103_tst_64x16_64x16_1x4_h_bz_splitK_TNT` | other | 32.0 | 0.1151 |
| `mhc_post_tilelang_kernel` | collective | 12.0 | 0.1122 |

### top kernels `e1_b8_recovery` b=8 (count/iter over 4 ranks, ms/iter)

| kernel | family | count/iter | ms/iter |
|---|---|---|---|
| `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x8x256_s6_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_sc` | moe_bmm | 148.0 | 4.0177 |
| `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDen` | dense_gemm | 520.0 | 2.6406 |
| `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x8x256_s6_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_schPd` | moe_bmm | 160.0 | 2.6101 |
| `void deep_gemm::sm100_tf32_hc_prenorm_gemm_impl<(unsigned int)24, (unsigned int)20480, (unsigned int` | dense_gemm | 316.0 | 1.912 |
| `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100_splitkSm100BlockScaledSplitK` | dense_gemm | 320.0 | 1.7831 |
| `kernel_cutlass_kernel_vllmmodelsdeepseek_v41nvidiaopscute_dslall_reduce_mhc_LamportMHCDeviceKernel_o` | collective | 308.0 | 1.744 |
| `mhc_pre_big_fuse_with_norm_tilelang_kernel` | collective | 332.0 | 1.7369 |
| `fmhaSm100fKernel_QkvE4m3OBfloat16H512HVPerCta128PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ16` | attention | 152.0 | 1.5744 |
| `_dsv4_topk_kernel` | moe_routing | 160.0 | 0.8033 |
| `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_` | quantize | 320.0 | 0.692 |
| `kernel_cutlass_device_kernel_tensorptrbf16gmemalign16o816512div85121_tensorptri64gmemo81_tensorptrf3` | dense_gemm | 160.0 | 0.6861 |
| `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_ll_bf16_splitkLLBf16SplitK_object_at__` | dense_gemm | 160.0 | 0.6375 |
| `void moe::dev::routing::routingCustom::routingIndicesDynBlockKernel<moe::dev::routing::routingCustom` | moe_routing | 160.0 | 0.5159 |
| `void vllm::deepseek_v4_fused_ops::fusedDeepseekV4FullCacheKernel<c10::BFloat16, (bool)1, (bool)1, (b` | other | 160.0 | 0.5046 |
| `_build_flashinfer_mixed_sparse_indices_kernel` | attention | 160.0 | 0.4568 |
| `_q_kv_norm_quant_kernel` | other | 160.0 | 0.4176 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::na` | other | 160.0 | 0.3769 |
| `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x8x512_s3_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_sc` | moe_bmm | 12.0 | 0.3375 |
| `kernel_cutlass_kernel_vllmmodelsdeepseek_v41nvidiaopscute_dslall_reduce_mhc_QuadFinalizePublishDevic` | collective | 148.0 | 0.3337 |
| `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &vllm::silu_kernel<c10::BFloat16>, &vll` | other | 160.0 | 0.3149 |
| `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_` | quantize | 160.0 | 0.2813 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorIn` | moe_routing | 4.0 | 0.2747 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | other | 32.0 | 0.2657 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | collective | 20.0 | 0.2609 |
| `kernel_cutlass_kernel_vllmmodelsdeepseek_v41nvidiaopscute_dslall_reduce_mhc_SharedOnlyPublishDeviceK` | collective | 160.0 | 0.2332 |
| `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeLinearKernel_object_at` | quantize | 160.0 | 0.2197 |
| `nvjet_sm103_tst_128x8_64x12_1x1_h_bz_splitK_TNT` | other | 4.0 | 0.2152 |
| `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char *, (u` | other | 160.0 | 0.2118 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::FillFunctor<unsigned char>, std::` | other | 160.0 | 0.1752 |
| `void vllm::cooperative::cooperative_topk_cs8<(unsigned int)512>(vllm::cooperative::CooperativeTopKPa` | moe_routing | 16.0 | 0.1531 |

### top kernels `diverse_nsys_run2` b=8 (count/iter over 4 ranks, ms/iter)

| kernel | family | count/iter | ms/iter |
|---|---|---|---|
| `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x8x256_s6_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_sc` | moe_bmm | 148.0 | 3.9132 |
| `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDen` | dense_gemm | 520.0 | 2.8304 |
| `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x8x512_s3_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_schPd` | moe_bmm | 148.0 | 2.5101 |
| `void deep_gemm::sm100_tf32_hc_prenorm_gemm_impl<(unsigned int)24, (unsigned int)20480, (unsigned int` | dense_gemm | 316.0 | 1.9156 |
| `kernel_cutlass_kernel_vllmmodelsdeepseek_v41nvidiaopscute_dslall_reduce_mhc_LamportMHCDeviceKernel_o` | collective | 308.0 | 1.8206 |
| `mhc_pre_big_fuse_with_norm_tilelang_kernel` | collective | 332.0 | 1.7453 |
| `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100_splitkSm100BlockScaledSplitK` | dense_gemm | 320.0 | 1.684 |
| `fmhaSm100fKernel_QkvE4m3OBfloat16H512HVPerCta128PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ16` | attention | 152.0 | 1.5769 |
| `_dsv4_topk_kernel` | moe_routing | 160.0 | 0.7659 |
| `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_` | quantize | 320.0 | 0.6923 |
| `kernel_cutlass_device_kernel_tensorptrbf16gmemalign16o816512div85121_tensorptri64gmemo81_tensorptrf3` | dense_gemm | 160.0 | 0.6846 |
| `kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_ll_bf16_splitkLLBf16SplitK_object_at__` | dense_gemm | 160.0 | 0.6396 |
| `void moe::dev::routing::routingCustom::routingIndicesDynBlockKernel<moe::dev::routing::routingCustom` | moe_routing | 160.0 | 0.5522 |
| `void vllm::deepseek_v4_fused_ops::fusedDeepseekV4FullCacheKernel<c10::BFloat16, (bool)1, (bool)1, (b` | other | 160.0 | 0.5269 |
| `_build_flashinfer_mixed_sparse_indices_kernel` | attention | 160.0 | 0.4553 |
| `_q_kv_norm_quant_kernel` | other | 160.0 | 0.4177 |
| `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_` | quantize | 160.0 | 0.3405 |
| `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x8x512u2_s3_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_` | moe_bmm | 12.0 | 0.3303 |
| `kernel_cutlass_kernel_vllmmodelsdeepseek_v41nvidiaopscute_dslall_reduce_mhc_QuadFinalizePublishDevic` | collective | 148.0 | 0.316 |
| `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorIn` | moe_routing | 4.0 | 0.3154 |
| `void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &vllm::silu_kernel<c10::BFloat16>, &vll` | other | 160.0 | 0.2764 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)` | other | 32.0 | 0.265 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | collective | 20.0 | 0.2624 |
| `kernel_cutlass_kernel_vllmmodelsdeepseek_v41nvidiaopscute_dslall_reduce_mhc_SharedOnlyPublishDeviceK` | collective | 160.0 | 0.2393 |
| `kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeLinearKernel_object_at` | quantize | 160.0 | 0.2245 |
| `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char *, (u` | other | 160.0 | 0.2163 |
| `nvjet_sm103_tst_128x8_64x12_1x1_h_bz_splitK_TNT` | other | 4.0 | 0.2159 |
| `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x8x256_s5_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_schPd` | moe_bmm | 12.0 | 0.2051 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::FillFunctor<unsigned char>, std::` | other | 160.0 | 0.1744 |
| `void vllm::cooperative::cooperative_topk_cs8<(unsigned int)512>(vllm::cooperative::CooperativeTopKPa` | moe_routing | 16.0 | 0.1679 |
