# FlashInfer autotune cache diff: off2 vs on

off2: `/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15/e1_dry_off2_v2/autotune_configs.c6867df7e514.json`
on: `/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15/e1_dry_on_v2/autotune_configs.6adeb850201b.json`

metadata identical: True; keys 159 / 159; common 159; only off2 0; only on 0

| op | runner | common keys | differing tactics | example key (truncated) | tactic A | tactic B |
|---|---|---|---|---|---|---|
| flashinfer::trtllm_fp4_block_scale_moe | MoERunner | 27 | 10 | `('flashinfer::trtllm_fp4_block_scale_moe', 'MoERunner', ((1, 0), (0,),` | ['MoERunner', [8, 1347]] | ['MoERunner', [8, 629]] |
| mxfp8_gemm | CuteDSLMxfp8GemmRunner | 132 | 83 | `('mxfp8_gemm', 'CuteDSLMxfp8GemmRunner', ((1, 1280), (1280, 8192), (-1` | ['CuteDSLMxfp8GemmRunner', [[128, 16], [4, 1], True, False, 1]] | ['CuteDSLMxfp8GemmRunner', [[128, 16], [1, 1], True, False, 1]] |

**93 of 159 common tactic entries differ (58 %).**

metadata A:
```
{
 "flashinfer_version": "0.7.0.post1",
 "cuda_version": "13.0",
 "cublas_version": "13.1.1",
 "cudnn_version": "92000",
 "cudnn_frontend_version": "1.31.0",
 "cutlass_dsl_version": "nvidia-cutlass-dsl-libs-base==4.7.1;nvidia-cutlass-dsl-libs-core==4.7.1;nvidia-cutlass-dsl-libs-cu12==4.7.1;nvidia-cutlass-dsl-libs-cu13==4.7.1;nvidia-cutlass-dsl==4.7.1",
 "gpu": "NVIDIA GB300"
}
```
