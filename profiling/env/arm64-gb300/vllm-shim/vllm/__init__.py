"""`vllm` shim for the GB300 profiling environment.

This is NOT vLLM. It is the smallest importable `vllm` package that lets the
ServingStudio Sim profiler's `kv_cache_append`/`vllm_cuda` runner
(`profiling/runners/attention/kv_cache_append.py`) run the exact CUDA kernel
it measures elsewhere, on an arm64 host where the alignment fork's venv does
not exist:

- `vllm._custom_ops.reshape_and_cache_flash` -> `torch.ops._C_cache_ops.reshape_and_cache_flash`,
  provided by a CUDA extension built from the fork's unmodified
  `csrc/libtorch_stable/cache_kernels.cu` (see `build_shim.py`, `manifest.json`).
- `vllm.utils.torch_utils.create_kv_caches_with_random_flash`, the fork's cache
  allocator, copied verbatim with its helpers.

Source of every copied line: https://github.com/serendipity-zk/vllm.git servingstudio-alignment @ e1423a4d7496dc6535862ae11bab8cb4d87b155f
(the submodule pin of `alignment/profiler/vllm`). Nothing else of vLLM is here;
any other `vllm.*` import fails, which is intended.
"""

__version__ = "0.0.0+servingstudio-gb300-shim"
