"""Kernel-name → kernel-group and collective-kind classification.

Rules are copied verbatim from the accepted Phase 1 planning analysis
(`phase1/longctx_chains.py::group`, `phase1/critpath.py::is_comm/comm_kind`, branch
phase1-planning @ cc06d9c2) so the groups in this package mean the same thing as the
numbers quoted in PHASE1-PLAN.md.  Pure functions, no I/O.
"""
from __future__ import annotations

import numpy as np

GROUPS = ("fmha", "indexer", "sparse_meta", "moe", "comm", "other")
GROUP_INDEX = {g: i for i, g in enumerate(GROUPS)}

COMM_KINDS = ("mhc", "oneshot", "twoshot", "nccl_allreduce", "nccl_allgather", "nccl_other")
COMM_KIND_INDEX = {k: i for i, k in enumerate(COMM_KINDS)}


def is_comm(name: str) -> bool:
    return any(t in name for t in ("twoshotAllreduce", "oneshotAllreduceFusion", "ncclDevKernel", "all_reduce_mhc"))


def comm_kind(name: str) -> str | None:
    if "all_reduce_mhc" in name:
        return "mhc"
    if "oneshotAllreduceFusion" in name:
        return "oneshot"
    if "twoshotAllreduce" in name:
        return "twoshot"
    if "ncclDevKernel_AllReduce" in name:
        return "nccl_allreduce"
    if "ncclDevKernel_AllGather" in name:
        return "nccl_allgather"
    if "ncclDevKernel" in name:
        return "nccl_other"
    return None


def group(name: str) -> str:
    n = name
    if is_comm(n):
        return "comm"
    if n.startswith("fmhaSm100f") or "flashmla" in n or "core_attn" in n:
        return "fmha"
    if ("mqa_logits" in n or "cooperative_topk" in n or "topKPerRow" in n or "_dsv4_topk" in n
            or "fp8_fp4_gemm" in n or "indexer" in n.lower()):
        return "indexer"
    if ("sparse_indices" in n or "swa_indices" in n or "lookback" in n or "ring_slot" in n or "radixSort" in n
            or "cub::" in n or "DeviceScan" in n or "DeviceRadixSort" in n or "FullCacheKernel" in n
            or "_q_kv_norm_quant" in n):
        return "sparse_meta"
    if n.startswith("bmm_MxE4m3") or n.startswith("bmm_Bfloat16") or "routing" in n or "finalize" in n or "act_and_mul" in n:
        return "moe"
    return "other"


def build_lookup(kernel_names: dict[int, str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (group_of_name_id, comm_kind_of_name_id) int8 arrays indexed by kernel name id.

    comm kind is -1 for non-communication kernels.
    """
    size = max(kernel_names) + 1
    g = np.zeros(size, dtype=np.int8)
    c = np.full(size, -1, dtype=np.int8)
    for k, v in kernel_names.items():
        g[k] = GROUP_INDEX[group(v)]
        kind = comm_kind(v)
        if kind is not None:
            c[k] = COMM_KIND_INDEX[kind]
    return g, c
