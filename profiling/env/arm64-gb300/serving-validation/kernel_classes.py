"""Classify the kernels of a serving trace by communication path (CPU-only).

    python kernel_classes.py --self-test
    python kernel_classes.py --sqlite run.sqlite --out classes.json

The question each serving run must answer before any error figure is read: did
the tensor-parallel all-reduce run as NCCL, as FlashInfer's standalone
all-reduce, or fused with residual add and RMSNorm? Kernel names decide it.
The rules come from source (FlashInfer 0.7.0 headers under
`flashinfer/data/include/flashinfer/comm/`, vLLM `csrc`, NCCL):

| class | kernel name |
| --- | --- |
| nccl_all_reduce | `ncclDevKernel_AllReduce_*` |
| nccl_other | other `ncclDevKernel_*` (logits all-gather and gather) |
| fi_fused_ar_rmsnorm | `allreduce_fusion_kernel_{oneshot_lamport,twoshot_sync}` with template pattern 1 (`kARResidualRMSNorm`); `oneshotAllreduceFusionKernel` with `RMSNormFusion = true`; `rmsNormLamport` (the norm half of the two-shot multicast path) |
| fi_standalone_ar | the same FlashInfer kernels with pattern 0 (`kAllReduce`) or `RMSNormFusion = false`; `twoshotAllreduceKernel` |
| vllm_custom_ar | `cross_device_reduce_1stage`, `cross_device_reduce_2stage` (vLLM `csrc/custom_all_reduce.cuh`) |
| torch_symm_mem_ar | names containing `multimem_all_reduce` or `two_shot_all_reduce` |
| attention_trtllm | `fmha`, `trtllm` or `Fmha` in the name |
| attention_flashinfer | `BatchPrefill`, `BatchDecode` or `flashinfer::` without the above |
| kv_append | `reshape_and_cache` |
| gemm_cublas | `nvjet_*`, `cublasLt::*` (including `splitKreduce_kernel`) |
| inductor_triton | `triton_*`: kernels torch.compile generated (fused add + RMSNorm, SiLU-and-mul, embedding) |
| vllm_runtime_triton | `_<name>_kernel`: the model runner's own Triton kernels (slot mapping, sampling, bookkeeping) |
| torch_aten | `at::native::*`, `at_cuda_detail::*`: PyTorch's own elementwise, index, fill and scan kernels |

`twoshotAllreduceKernel` is followed by `rmsNormLamport` when fused, so a
run's fused count is read from the latter. With `--sqlite`, kernels are read
from Nsight Systems' export (`CUPTI_ACTIVITY_KIND_KERNEL`, names from
`StringIds`); a kernel launched from a CUDA graph has a non-null
`graphNodeId`, which gives the graph-replay share per class. The sqlite path
ran first on the TP1 smoke trace of 2026-11-11. The communication samples in
`_SAMPLES` are still written from kernel templates, since a TP1 run has no
collective; a TP2 run confirms or corrects them.
"""

# Kernel names and the rule table are quoted whole, so lines may be long.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

_FI_PATTERN = re.compile(
    r"allreduce_fusion_kernel_(?:oneshot_lamport|twoshot_sync)<\(?(?:\w+::)*"
    r"(?:AllReduceFusionPattern\)?)?\)?\s*(\d+|k\w+)"
)
_FI_MNNVL_ONESHOT = re.compile(r"oneshotAllreduceFusionKernel<[^,>]+,\s*[^,>]+,\s*(true|false|1|0)")


def classify(name: str) -> str:
    if name.startswith("ncclDevKernel_AllReduce") or "ncclDevKernel_AllReduce" in name:
        return "nccl_all_reduce"
    if "ncclDevKernel_" in name or name.startswith("ncclKernel"):
        return "nccl_other"
    if "allreduce_fusion_kernel_" in name:
        m = _FI_PATTERN.search(name)
        token = m.group(1) if m else ""
        if token in ("0", "kAllReduce"):
            return "fi_standalone_ar"
        if token in ("1", "kARResidualRMSNorm"):
            return "fi_fused_ar_rmsnorm"
        return "fi_allreduce_unknown_pattern"
    if "oneshotAllreduceFusionKernel" in name:
        m = _FI_MNNVL_ONESHOT.search(name)
        if m is None:
            return "fi_allreduce_unknown_pattern"
        return "fi_fused_ar_rmsnorm" if m.group(1) in ("true", "1") else "fi_standalone_ar"
    if "rmsNormLamport" in name:
        return "fi_fused_ar_rmsnorm"
    if "twoshotAllreduceKernel" in name:
        return "fi_standalone_ar"
    if "cross_device_reduce_scatter" in name:
        return "vllm_custom_reduce_scatter"
    if "cross_device_reduce_1stage" in name or "cross_device_reduce_2stage" in name:
        return "vllm_custom_ar"
    if "multimem_all_reduce" in name or "two_shot_all_reduce" in name:
        return "torch_symm_mem_ar"
    if "reshape_and_cache" in name:
        return "kv_append"
    if re.search(r"fmha|Fmha|trtllm", name):
        return "attention_trtllm"
    if re.search(r"BatchPrefill|BatchDecode|flashinfer::", name):
        return "attention_flashinfer"
    if name.startswith("nvjet_") or "cublasLt::" in name or "splitKreduce" in name:
        return "gemm_cublas"
    if name.startswith("triton_"):
        return "inductor_triton"
    if re.fullmatch(r"_\w+_kernel", name):
        return "vllm_runtime_triton"
    if "at::native::" in name or "at_cuda_detail::" in name:
        return "torch_aten"
    return "other"


_SAMPLES = {
    "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)": "nccl_all_reduce",
    "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)": "nccl_other",
    "void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport"
    "<(flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, __nv_bfloat16, 2, true, true>"
    "(flashinfer::trtllm_allreduce_fusion::AllReduceFusionParams<__nv_bfloat16>)": "fi_fused_ar_rmsnorm",
    "void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_twoshot_sync"
    "<(flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)0, __nv_bfloat16, 4, true>"
    "(flashinfer::trtllm_allreduce_fusion::AllReduceFusionParams<__nv_bfloat16>, "
    "std::array<int, 4>)": "fi_standalone_ar",
    "void flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel<(unsigned char)2, "
    "__nv_bfloat16, true, (flashinfer::QuantType)0>(flashinfer::AllReduceKernelParams<__nv_bfloat16>)": "fi_fused_ar_rmsnorm",
    "void flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel<(unsigned char)4, "
    "__nv_bfloat16, false, (flashinfer::QuantType)0>(flashinfer::AllReduceKernelParams<__nv_bfloat16>)": "fi_standalone_ar",
    "void flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceKernel<(unsigned char)4, __nv_bfloat16, "
    "true, float4>(flashinfer::AllReduceKernelParams<__nv_bfloat16>)": "fi_standalone_ar",
    "void flashinfer::trtllm_mnnvl_allreduce::rmsNormLamport<__nv_bfloat16, (flashinfer::QuantType)0, "
    "false>(flashinfer::AllReduceKernelParams<__nv_bfloat16>)": "fi_fused_ar_rmsnorm",
    "void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, "
    "vllm::Signal*, __nv_bfloat16*, int, int)": "vllm_custom_ar",
    "void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, "
    "(vllm::Fp8KVCacheDataType)0>(...)": "kv_append",
    # Names below this line are copied from the TP1 smoke trace (2026-11-11).
    "fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP16MultiCtasKvCgaVarSeqQ8Kv256StaticGroupedSwapsAbForGen": "attention_trtllm",
    "fmhaSm103aKernel_QkvBfloat16OBfloat16H128PagedKvCausalP16VarSeqQ256Kv128PersistentContext": "attention_trtllm",
    "void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0>(const T1 *, const T1 *, T2 *, T2 *, const long *, long, long, long, long, long, int, int, int, const float *, const float *, int)": "kv_append",
    "nvjet_sm103_tst_192x8_64x8_2x1_v_bz_TNT": "gemm_cublas",
    "void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, float, __nv_bfloat16, __nv_bfloat16>(...)": "gemm_cublas",
    "triton_red_fused_fused_add_rms_norm_0": "inductor_triton",
    "triton_poi_fused_mul_silu_slice_1": "inductor_triton",
    "_compute_slot_mappings_kernel": "vllm_runtime_triton",
    "void at::native::vectorized_elementwise_kernel<(int)8, at::native::FillFunctor<unsigned char>, std::array<char *, (unsigned long)1>>(int, T2, T3)": "torch_aten",
    "void flashinfer::BatchPrefillWithPagedKVCacheKernel<...>(...)": "attention_flashinfer",
    "void vllm::cross_device_reduce_scatter<__nv_bfloat16, 4>(...)": "vllm_custom_reduce_scatter",
    "some_unknown_kernel": "other",
}


def self_test() -> int:
    bad = {
        name: (classify(name), want) for name, want in _SAMPLES.items() if classify(name) != want
    }
    for name, (got, want) in bad.items():
        print(f"FAIL {got} != {want}: {name[:100]}")
    print(f"self-test: {len(_SAMPLES) - len(bad)}/{len(_SAMPLES)} passed")
    return 1 if bad else 0


def from_sqlite(path: Path) -> dict:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cols = {r[1] for r in con.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    name_col = "demangledName" if "demangledName" in cols else "shortName"
    graph = "k.graphNodeId" if "graphNodeId" in cols else "NULL"
    rows = con.execute(
        f"select s.value, k.end - k.start, {graph} from CUPTI_ACTIVITY_KIND_KERNEL k "
        f"join StringIds s on s.id = k.{name_col}"
    )
    count, time_ns, graphed = Counter(), Counter(), Counter()
    examples: dict[str, set] = defaultdict(set)
    for name, duration, node in rows:
        cls = classify(name)
        count[cls] += 1
        time_ns[cls] += duration
        graphed[cls] += node is not None
        if len(examples[cls]) < 3:
            examples[cls].add(name[:200])
    return {
        "name_column": name_col,
        "classes": {
            cls: {
                "kernels": count[cls],
                "time_ms": time_ns[cls] / 1e6,
                "graph_replayed_share": graphed[cls] / count[cls],
                "examples": sorted(examples[cls]),
            }
            for cls in sorted(count)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--sqlite", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    report = from_sqlite(args.sqlite)
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
