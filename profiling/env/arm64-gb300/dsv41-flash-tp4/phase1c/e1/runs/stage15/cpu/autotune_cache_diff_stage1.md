# FlashInfer autotune cache diff: hook-off vs hook-on

hook-off: `/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15/cpu/autotune_cache_stage1/hook_off_dir_c6867df7_saved_18-49-44Z.json`
hook-on: `/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15/cpu/autotune_cache_stage1/hook_on_dir_6adeb850_saved_19-16-44Z.json`

metadata identical: True; keys 159 / 159; common 159; only hook-off 0; only hook-on 0

| op | runner | common keys | differing tactics | example key (truncated) | tactic A | tactic B |
|---|---|---|---|---|---|---|
| flashinfer::trtllm_fp4_block_scale_moe | MoERunner | 27 | 8 | `('flashinfer::trtllm_fp4_block_scale_moe', 'MoERunner', ((1, 0), (0,),` | ['MoERunner', [8, 627]] | ['MoERunner', [8, 1507]] |
| mxfp8_gemm | CuteDSLMxfp8GemmRunner | 132 | 78 | `('mxfp8_gemm', 'CuteDSLMxfp8GemmRunner', ((1, 1280), (1280, 8192), (-1` | ['CuteDSLMxfp8GemmRunner', [[256, 64], [2, 2], True, False, 1]] | ['CuteDSLMxfp8GemmRunner', [[128, 8], [2, 1], True, False, 1]] |

**86 of 159 common tactic entries differ (54 %).**

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
