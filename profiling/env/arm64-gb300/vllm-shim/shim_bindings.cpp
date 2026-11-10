// Registers the one vLLM cache op the ServingStudio Sim `kv_cache_append` /
// `vllm_cuda` runner calls, against the fork's UNMODIFIED
// csrc/libtorch_stable/cache_kernels.cu, which is compiled into the same shared
// library by build_shim.py. Everything else that the fork's
// csrc/libtorch_stable/torch_bindings.cpp registers is left out on purpose: the
// shim exists to measure `reshape_and_cache_flash_kernel`, nothing more.
//
// Fork: https://github.com/serendipity-zk/vllm.git, branch
// servingstudio-alignment, commit e1423a4d7496dc6535862ae11bab8cb4d87b155f (the
// pin of alignment/profiler/vllm). The schema string and the impl line below
// are copied verbatim from that commit's csrc/libtorch_stable/torch_bindings.cpp
// (lines 961-966 and 1140); build_shim.py checks that they still occur there.
#include <torch/csrc/stable/library.h>

// The fork's own declaration of the host entry point. The include root is the
// fork's csrc/, as in its CMake build, so every "torch_utils.h" resolves as the
// fork resolves it (csrc/torch_utils.h for the attention headers, the
// libtorch_stable one for the files in that directory).
#include "libtorch_stable/ops.h"

STABLE_TORCH_LIBRARY_FRAGMENT(_C_cache_ops, ops) {
  ops.def(
      "reshape_and_cache_flash(Tensor key, Tensor value,"
      "                        Tensor! key_cache,"
      "                        Tensor! value_cache,"
      "                        Tensor slot_mapping,"
      "                        str kv_cache_dtype,"
      "                        Tensor k_scale, Tensor v_scale) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C_cache_ops, CUDA, ops) {
  ops.impl("reshape_and_cache_flash", TORCH_BOX(&reshape_and_cache_flash));
}
