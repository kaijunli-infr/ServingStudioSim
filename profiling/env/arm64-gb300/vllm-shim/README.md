# `vllm` shim: the fork's `reshape_and_cache_flash` on arm64 without vLLM

The Llama arch's `kv_cache_append` kernel has one backend, `vllm_cuda`, whose
runner (`profiling/runners/attention/kv_cache_append.py`) imports
`vllm._custom_ops.reshape_and_cache_flash` and
`vllm.utils.torch_utils.create_kv_caches_with_random_flash` from the
alignment fork's venv (`profiling/exec/env.py`, `vllm_env`). That venv does not
exist on the GB300 trays (x86 build), and no official wheel can stand in for it
(see "Why not the wheel"). This directory builds the smallest faithful
substitute: the fork's own kernel source, compiled unchanged for sm_103, behind
a `vllm` package that contains exactly the two entry points the runner uses.

| file | role |
| --- | --- |
| `shim_bindings.cpp` | registers `_C_cache_ops.reshape_and_cache_flash` (schema and impl line verbatim from the fork's `csrc/libtorch_stable/torch_bindings.cpp`) |
| `build_shim.py` | checks the fork checkout is at the pinned commit with the pinned file hashes, compiles `csrc/libtorch_stable/cache_kernels.cu` + `shim_bindings.cpp` with `torch.utils.cpp_extension` and the fork's nvcc flags, writes `manifest.json` |
| `vllm/` | the shim package: `_custom_ops.py` (loads the library, `reshape_and_cache_flash` verbatim), `utils/torch_utils.py` (the allocator and its helpers verbatim); every block cites its source lines and the source file's sha256 |
| `install_shim.sh` | creates `$GB300_ROOT/vllm-shim` (`.venv -> $GB300_ROOT/venv`, the package, the build) |
| `probe_shim.py` | the two-spec GPU probe: byte-level correctness vs the Torch reference, kernel identity, CUPTI capture, explicit fp8 refusal |

Source pin: `alignment/profiler/vllm` = https://github.com/serendipity-zk/vllm.git
branch `servingstudio-alignment` at `e1423a4d7496dc6535862ae11bab8cb4d87b155f`
(upstream base `04730e82`). The files used here are byte-identical between the
fork commit and its upstream base (`git diff 04730e82 e1423a4d -- csrc/libtorch_stable/cache_kernels.cu
csrc/libtorch_stable/torch_bindings.cpp vllm/utils/torch_utils.py vllm/_custom_ops.py` is empty),
so the kernel measured is upstream vLLM's `reshape_and_cache_flash_kernel` as
of 2026-09-24, the same source the H200 and B200 rows were measured from.

## What is identical to the fork, and what is not

Identical: the kernel (`vllm::reshape_and_cache_flash_kernel`, template
`<scalar_t, cache_t, Fp8KVCacheDataType>`; for bf16/bf16 the `kAuto`
instantiation, NHD + tensor-scale fast path with 16-byte vectorized copies), the
host function (stride and grid derivation: one block per token, `block_size =
key_cache.size(1)`, strides from the tensors), the op schema, the Python
wrapper, the cache allocator (`(num_blocks, 2, block_size, num_heads,
head_size)` allocation permuted per layout, `uniform_(-head_size**-0.5, +)`
after `set_random_seed(42)`), and the compile flags (`-DENABLE_FP8`, C++20,
torch's common nvcc flags without the `__CUDA_NO_HALF_*` defines).

Different, and why it does not change the measurement: the library is built
with `torch.utils.cpp_extension` instead of the fork's CMake (same nvcc, same
source, same flags; only the output name differs); it is built for `sm_103`
only (the fork's wheel builds a list of arches, each kernel runs its own
arch's SASS either way); the Python package holds two functions instead of
vLLM (the runner imports only those two; nothing else of vLLM runs inside the
timed region, which is the single op call). `set_random_seed` calls
`torch.cuda.manual_seed_all` directly, which is what the fork's CUDA platform
does. The fp8 cache path (`convert_fp8`) is not built: asking for it raises
`NotImplementedError` instead of falling back to another conversion.

## Why not the wheel

Official aarch64 wheels exist on PyPI (vllm 0.24 to 0.33, 2026), but each pins
a torch the environment does not have (0.24 to 0.27 torch 2.11.0, 0.28 to
0.30 torch 2.13.0, 0.31 to 0.33 torch 2.14.0) with about a hundred
dependencies, against this env's torch 2.10.0+cu130. Installing one would
replace torch under the FlashInfer, Triton and DeepGEMM builds the first 1,166
rows were measured with, and its `_C_cache_ops` library would be a different
build (different upstream commit, different compiler flags, torch 2.11+ stable
ABI) of a kernel whose source has kept moving upstream since 04730e82. The
fork's pinned source compiled against the env's own torch is the smaller and
the demonstrably identical change.

## Build and verify (tray12)

```bash
source profiling/env/arm64-gb300/gb300.env            # sets VIBESIM_VLLM_FORK_ROOT=$GB300_ROOT/vllm-shim
CUDA_VISIBLE_DEVICES= profiling/env/arm64-gb300/vllm-shim/install_shim.sh   # CPU: minutes of nvcc
# GPU probe (one approved GPU), from the .venv-less worktree:
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$GB300_ROOT/vllm-shim:$PWD \
  $GB300_ROOT/vllm-shim/.venv/bin/python profiling/env/arm64-gb300/vllm-shim/probe_shim.py \
  --out $GB300_ROOT/artifacts/vllm-shim/probe.json --expect-uuid GPU-... --num-tokens 64 8192
```

Results of the probe and of the 70-row fill: `../TP1_RESULTS.md`.

## Provenance of the measured rows

`BUILD_MANIFEST.tray12-20261110.json` is the verbatim `manifest.json` of the
build whose library measured the 70 GB300 `kv_cache_append` / `vllm_cuda` rows
(tray12, 2026-11-10): the fork commit and the sha256 of each fork file used,
the sha256 of the shim sources (`build_shim.py` and `shim_bindings.cpp` as of
commit `3b17315d`), the compiler and linker flags, torch and nvcc versions,
and the library's sha256 (`ca75929f…`, 2,266,536 bytes). The run directory's
`shim-manifest.json` is the same file. `build_seconds` is 0.4 because that
invocation reused the objects of the first successful compile (16 s) a few
minutes earlier with identical inputs. The library itself is not committed;
`install_shim.sh` rebuilds it and `build_shim.py` refuses a fork checkout
whose commit or file hashes differ from the pinned ones.
