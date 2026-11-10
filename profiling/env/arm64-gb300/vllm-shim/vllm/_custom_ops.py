"""Shim of the fork's `vllm/_custom_ops.py`, limited to the one op the
`kv_cache_append`/`vllm_cuda` runner calls.

`reshape_and_cache_flash` is copied verbatim from
  https://github.com/serendipity-zk/vllm.git servingstudio-alignment @ e1423a4d7496dc6535862ae11bab8cb4d87b155f
  vllm/_custom_ops.py lines 2544-2563 (sha256 5788e575730631725ba05f0756d39958174a5381c790020ba6af07ca5309c078).
The op it dispatches to, `torch.ops._C_cache_ops.reshape_and_cache_flash`, is
registered by the shared library this module loads: the fork's unmodified
`csrc/libtorch_stable/cache_kernels.cu` plus `shim_bindings.cpp`, built by
`build_shim.py`, which records the provenance in `manifest.json` next to this
package. `convert_fp8` (fp8 KV caches only) is not provided and says so
instead of substituting a conversion.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

_SHIM_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_PATH = _SHIM_ROOT / "manifest.json"
if not _MANIFEST_PATH.exists():
    raise ImportError(f"vllm shim: {_MANIFEST_PATH} is missing; run build_shim.py first")
MANIFEST = json.loads(_MANIFEST_PATH.read_text())
_LIBRARY = _SHIM_ROOT / MANIFEST["library"]["path"]
if not _LIBRARY.exists():
    raise ImportError(f"vllm shim: extension {_LIBRARY} is missing; run build_shim.py")
torch.ops.load_library(str(_LIBRARY))
if not hasattr(torch.ops._C_cache_ops, "reshape_and_cache_flash"):
    raise ImportError(
        "vllm shim: the extension did not register _C_cache_ops.reshape_and_cache_flash"
    )


# --- _custom_ops.py lines 2544-2563 ---
def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def convert_fp8(*args, **kwargs):
    raise NotImplementedError(
        "the GB300 vllm shim builds only reshape_and_cache_flash; fp8 KV caches "
        "need the fork's _C_cache_ops.convert_fp8, which is not built here"
    )
