"""Shim of the fork's `vllm/utils/torch_utils.py`: the KV-cache allocator the
`kv_cache_append`/`vllm_cuda` runner calls, and the helpers it uses.

Every block below is copied verbatim from
  https://github.com/serendipity-zk/vllm.git servingstudio-alignment @ e1423a4d7496dc6535862ae11bab8cb4d87b155f
  vllm/utils/torch_utils.py (sha256 31e8a695f7efa52584ce20cc33edfb643b70c28ce83d273feb34326c61ab631c)
with its line range noted above it. The one substitution is `current_platform`:
a stand-in whose `manual_seed_all` is what the fork's CUDA platform does
(`vllm/platforms/cuda.py` @ e1423a4d: `torch.cuda.manual_seed_all(seed)`).
"""

from __future__ import annotations

import random

import numpy as np
import torch


class _CudaPlatformStandIn:
    """`vllm.platforms.current_platform`, as the fork's CUDA platform implements
    the one method used here."""

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        torch.cuda.manual_seed_all(seed)


current_platform = _CudaPlatformStandIn


# --- torch_utils.py lines 33-55 ---
STR_DTYPE_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "half": torch.half,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
    "int8": torch.int8,
    "int4_per_token_head": torch.uint8,
    "int8_per_token_head": torch.int8,
    "fp8_per_token_head": torch.uint8,
    "fp8_inc": torch.float8_e4m3fn,
    "fp8_ds_mla": torch.uint8,
    "nvfp4_ds_mla": torch.uint8,
    "turboquant_k8v4": torch.uint8,
    "turboquant_4bit_nc": torch.uint8,
    "turboquant_k3v4_nc": torch.uint8,
    "turboquant_3bit_nc": torch.uint8,
    "nvfp4": torch.uint8,
    "nvfp4_4over6": torch.uint8,
}


# --- torch_utils.py lines 404-422 ---
def _generate_random_fp8(
    tensor: torch.Tensor,
    low: float,
    high: float,
) -> None:
    # NOTE(zhaoyang): Due to NaN and Inf representation for fp8 data type,
    # it may occur Inf or NaN if we directly use torch.randint
    # to generate random data for fp8 data.
    # For example, s.11111.00 in fp8e5m2 format represents Inf.
    #     | E4M3        | E5M2
    # -----|-------------|-------------------
    # Inf | N/A         | s.11111.00
    # NaN | s.1111.111  | s.11111.{01,10,11}
    from vllm import _custom_ops as ops

    tensor_tmp = torch.empty_like(tensor, dtype=torch.float16)
    tensor_tmp.uniform_(low, high)
    ops.convert_fp8(tensor, tensor_tmp)
    del tensor_tmp


# --- torch_utils.py lines 425-445 ---
def get_kv_cache_torch_dtype(
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str) and model_dtype in STR_DTYPE_TO_TORCH_DTYPE:
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in STR_DTYPE_TO_TORCH_DTYPE:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype


# --- torch_utils.py lines 533-540 ---
def set_random_seed(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        from vllm.utils.torch_utils import current_platform  # shim: stand-in below

        current_platform.manual_seed_all(seed)


# --- torch_utils.py lines 543-545 ---
def nvfp4_kv_cache_full_dim(head_size: int) -> int:
    """Packed last dim for NVFP4 KV cache: fp4 data + fp8 block scales."""
    return head_size // 2 + head_size // 16


# --- torch_utils.py lines 599-652 ---
def create_kv_caches_with_random_flash(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
    seed: int | None = None,
    device: str | None = "cuda",
    cache_layout: str | None = "NHD",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    set_random_seed(seed)

    dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
    generic_kv_cache_shape = (num_blocks, 2, block_size, num_heads, head_size)
    assert cache_layout in ("NHD", "HND")
    stride_order = (0, 1, 2, 3, 4) if cache_layout == "NHD" else (0, 1, 3, 2, 4)

    kv_cache_allocation_shape = tuple(generic_kv_cache_shape[i] for i in stride_order)
    scale = head_size**-0.5

    key_caches: list[torch.Tensor] = []
    value_caches: list[torch.Tensor] = []

    for _ in range(num_layers):
        if isinstance(cache_dtype, str) and cache_dtype.startswith("nvfp4"):
            # Full page dim: fp4 data + fp8 block scales per head.
            # Per page layout: [K_data | K_scale | V_data | V_scale]
            # Returns [:, 0] and [:, 1] like all other dtypes.
            full_dim = nvfp4_kv_cache_full_dim(head_size)
            nvfp4_shape = (num_blocks, 2, block_size, num_heads, full_dim)
            nvfp4_phys = tuple(nvfp4_shape[i] for i in stride_order)
            inv = [stride_order.index(i) for i in range(len(stride_order))]
            key_value_cache = torch.randint(
                0,
                256,
                nvfp4_phys,
                dtype=dtype,
                device=device,
            ).permute(*inv)
        else:
            key_value_cache = torch.empty(
                size=kv_cache_allocation_shape, dtype=dtype, device=device
            ).permute(*stride_order)
            if cache_dtype in ["auto", "half", "bfloat16", "float"]:
                key_value_cache.uniform_(-scale, scale)
            elif cache_dtype == "fp8":
                _generate_random_fp8(key_value_cache, -scale, scale)
            else:
                raise ValueError(f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_value_cache[:, 0])
        value_caches.append(key_value_cache[:, 1])
    return key_caches, value_caches
