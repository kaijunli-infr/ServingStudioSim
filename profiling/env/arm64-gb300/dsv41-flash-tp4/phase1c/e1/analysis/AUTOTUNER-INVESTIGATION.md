# FlashInfer autotuner at server start: investigation (Phase 1.5 item C, CPU-only)

Question from Stage 1 (§5e of E1-STAGE1-REPORT.md): every launch spends ~20 min in "flashinfer AutoTuner" although the
preflight's "cache warm: flashinfer-autotune" check passes. Sources: the two Stage 1 server logs, the two autotune cache
files they wrote, the vLLM fork at 0ab18711 (`alignment/profiler/vllm`) and flashinfer 0.7.0.post1 in the fork venv. No GPU
was used; nothing was modified.

## 1. Mechanism (code)

1. vLLM `vllm/model_executor/warmup/kernel_warmup.py::flashinfer_autotune` (line 404). The cache file is
   `$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR/<sha256 of vllm_config.compute_hash(include_version=False)>/autotune_configs.json`
   (`resolve_flashinfer_autotune_file`). The leader rank reads the file, broadcasts it to all ranks (every rank logs
   "Loaded 159 configs"), then **always** runs `fi_utils.autotune(tune_mode=True, ...)` (line 460) over the warm-up shapes and
   saves the result back ("Saved 159 configs"). `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS` (`envs.py:223`) can exclude ops from tuning.
2. flashinfer `flashinfer/autotuner/autotuner.py::AutoTuner.search_cache` (line 1898) decides whether a loaded entry is
   used. It compares the op's requested profiling policy with the default (lines 1937-1938) and, in tuning mode, refuses the
   cached entry when `cached_policy != requested_policy` (line 1958) or `requested_policy != default_policy` (line 1972).
   The policy is "cold" when the op's `TuningConfig` sets `use_cold_l2_cache=True` and "hot" otherwise (line 3300).
3. The two ops that dominate the warm-up set both request cold-L2 profiling: `trtllm_fp4_block_scale_moe`
   (`flashinfer/fused_moe/core.py`, eight `use_cold_l2_cache=True` sites, e.g. 2062, 2336, 2576) and the CuteDSL `mxfp8_gemm`
   runner (`flashinfer/gemm/gemm_base.py` 1467, 7858, 7889, 7950). Their requested policy ("cold") differs from the default
   ("hot"), so in tuning mode the loaded file is ignored for them and every launch profiles all tactics again.

Consequence: the file cache is loaded and saved but never *used* for these ops at start-up; the preflight check ("directory
non-empty") tests the wrong thing.

## 2. Evidence (Stage 1 logs and cache files)

| | hook off (`e1_dry_off`) | hook on (`e1_dry_on`) |
|---|---|---|
| API server banner | 18:27:50 | 18:55:05 |
| model loading done (weights 56-57 s) | 18:29:34 | 18:56:48 |
| first CUDA-graph capture (30 s) | 18:30:15 | 18:57:27 |
| "Using FlashInfer autotune cache file" | `…/c6867df7e514…/autotune_configs.json` | `…/6adeb850201b…/autotune_configs.json` |
| "Loaded 159 configs" | 4 ranks, 18:30:36 | 4 ranks, 18:57:46 |
| "Autotuning process starts" -> "ends" | 18:30:36 -> 18:49:44 (**19 min 08 s**) | 18:57:46 -> 19:16:44 (**18 min 58 s**) |
| second graph capture (18 s), startup complete | 18:50:03, ~18:50:20 | 19:17:03, ~19:17:20 |
| "Saved 159 configs" | yes | yes |
| "No tuned config covers mxfp8_gemm …; falling back" warnings | 48 lines | 48 lines |

* Two cache directories: the hook flag (`aux_output_config.enable_return_routed_experts`) is part of `compute_hash`, so the
  arms never share a cache file. Even a working file cache would not make the arms use identical tactics unless one file were
  copied into the other arm's directory.
* `analysis/autotune_cache_diff.py` on the two saved files (`runs/stage15/autotune_cache_diff_stage1.md`): metadata identical
  (flashinfer 0.7.0.post1, CUDA 13.0, cuBLAS 13.1.1, cuDNN 92000, GB300), 159 keys each, all common; **86 of 159 winning tactics
  differ** (mxfp8_gemm 78 of 132, MoE 8 of 27). Example: mxfp8_gemm (1,1280)x(1280,8192) picked `[[256,64],[2,2]]` in one launch
  and `[[128,8],[2,1]]` in the other. Since the two files come from one hook-off and one hook-on launch, this diff mixes
  run-to-run tuning noise with any genuine hook effect; the off/off control separates them (E1-OFFOFF-PROTOCOL.md).
* The 48 "No tuned config covers mxfp8_gemm" warnings per run are emitted after tuning, for prefill shapes (e.g. M=256 with
  K=1280, N=4096) not in the warm-up set; the kernel falls back to a default tactic. These shapes only occur in ramp/prefill
  iterations, which the pure-decode statistics exclude.

## 3. What this means for E1

1. **Occupancy.** ~19 min of every launch is autotuning (22.5 min startup of a 25.4 min dry pass). Every Stage 1.5 / Stage 2
   budget line carries it (E1-STAGE15-BUDGET.md). It cannot be removed without changing the recipe.
2. **Comparability between arms.** Decode tactics (and therefore kernel timings) can differ between any two launches, hook or
   not. The size of that effect on the pure-decode median is unknown from Stage 1 and is exactly what the off/off control
   measures. Until it is measured, the b=48 on/off ratio of 1.0195 cannot be attributed to the hook alone (nor to tuning noise).
3. **Comparability with Phase 0.** Phase 0 passes went through the same re-tuning, so Phase 0 numbers already contain this
   launch-to-launch variation.

## 4. Options (none authorized; for the review)

| option | what | effect on comparability with Phase 0 | needs |
|---|---|---|---|
| O1 (recommended for Stage 1.5) | keep the recipe; quantify tuning + other launch noise with the off/off control; archive `autotune_configs.json` of every launch and diff arms as a standard artifact | none (same recipe) | nothing new; `collect_e1_stage15.sh` copies the cache files |
| O2 | pin tactics: copy one arm's `autotune_configs.json` into the other arm's hash directory **and** make the cache honoured in tuning mode (flashinfer `search_cache` policy rule, or vLLM calling `autotune(tune_mode=False)` when a file exists) | changes the recipe (flashinfer or vLLM code change); Phase 0 was not run that way | a code change outside this repo's launcher; new fork commit; re-validation |
| O3 | `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=…` or `enable_flashinfer_autotune=false` | different kernels from Phase 0 (default tactics) | config change only, but breaks the Phase 0 recipe |
| O4 | toggle the hook inside one server process | not possible: the flag is server configuration (`AuxOutputConfig`) | - |

Recommendation: O1 now. If the off/off control shows the null ratio's interval reaching the 2 % gate, O2 becomes the way to
make the hook comparison resolvable, and it would have to be authorized as a recipe change with its own Phase 0 re-check.

## 5. Preflight change

`preflight_e1_stage15.sh` reports the autotune cache state as INFO (directory, number of hash subdirectories, newest file)
instead of a PASS that suggests a warm start; the budget assumes a full re-tune per launch.
