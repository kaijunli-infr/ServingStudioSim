#!/usr/bin/env python
"""Two-spec GPU probe of the GB300 vllm shim (Phase 2C, step B5). Inserts nothing.

For each `--num-tokens` value, at the Llama 3.1 8B TP1 KV-cache geometry (8 KV
heads, head_dim 128, block_size 16, NHD, bf16 in / bf16 cache, tensor scales):

1. byte-level correctness: the shim's `reshape_and_cache_flash` and the Torch
   reference write (`cache[block, offset] = rows`, the runner's `torch` backend)
   start from identical random caches and must leave bit-identical caches, with
   exactly `num_tokens` rows changed in each of K and V;
2. kernel identity: torch.profiler lists the CUDA kernels one call launches
   (expected: exactly one, `vllm::reshape_and_cache_flash_kernel<...>`);
3. CUPTI capture: the repository's CUPTI profiler, with the runner's
   `kernel_name_contains="reshape_and_cache_flash_kernel"`, must match that
   kernel on every launch; its time is cross-checked against in-stream CUDA
   events around 200 launches;
4. the fp8-cache path must fail explicitly (the shim provides no `convert_fp8`).

    python probe_shim.py --out report.json --expect-uuid GPU-... --num-tokens 64 8192

Run on the shim's interpreter with the Sim checkout and the shim root on
PYTHONPATH, `CUDA_VISIBLE_DEVICES` set to the one approved GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE, LAYOUT = 8, 128, 16, "NHD"
KERNEL = "reshape_and_cache_flash_kernel"


def num_blocks(num_tokens: int) -> int:  # the runner's `_num_blocks`
    return max(256, math.ceil(num_tokens / BLOCK_SIZE))


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def identity(expect_uuid: str | None) -> dict:
    props = torch.cuda.get_device_properties(0)
    uuid = str(props.uuid)
    uuid = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
    out = {
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "uuid": uuid,
        "device_count": torch.cuda.device_count(),
        "sm_count": props.multi_processor_count,
    }
    if expect_uuid:
        out["expected_uuid"] = expect_uuid
        out["uuid_match"] = uuid == expect_uuid
        if not out["uuid_match"]:
            raise SystemExit(f"GPU uuid {uuid} is not the approved {expect_uuid}")
    if out["device_count"] != 1:
        raise SystemExit(f"{out['device_count']} devices visible; set CUDA_VISIBLE_DEVICES to one")
    return out


def probe(num_tokens: int, ops, create_kv_caches_with_random_flash, cupti, Timer) -> dict:
    torch.manual_seed(1234 + num_tokens)
    nb = num_blocks(num_tokens)
    key = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    value = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    slot_mapping = torch.randperm(nb * BLOCK_SIZE, device="cuda", dtype=torch.int64)[:num_tokens]
    k_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    v_scale = torch.ones_like(k_scale)
    key_caches, value_caches = create_kv_caches_with_random_flash(
        nb,
        BLOCK_SIZE,
        1,
        NUM_KV_HEADS,
        HEAD_DIM,
        "auto",
        torch.bfloat16,
        seed=42,
        device="cuda",
        cache_layout=LAYOUT,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]
    before_k, before_v = key_cache.clone(), value_cache.clone()
    ref_k, ref_v = key_cache.clone(), value_cache.clone()
    result = {
        "num_tokens": num_tokens,
        "num_blocks": nb,
        "cache_shape": list(key_cache.shape),
        "cache_stride": list(key_cache.stride()),
        "key_stride": list(key.stride()),
    }

    def kernel():
        ops.reshape_and_cache_flash(
            key, value, key_cache, value_cache, slot_mapping, "auto", k_scale, v_scale
        )

    # 1. correctness, byte level, against the Torch reference write
    kernel()
    block_idx = torch.div(slot_mapping, BLOCK_SIZE, rounding_mode="floor")
    block_off = slot_mapping % BLOCK_SIZE
    ref_k[block_idx, block_off] = key
    ref_v[block_idx, block_off] = value
    torch.cuda.synchronize()
    k_equal = torch.equal(bits(key_cache), bits(ref_k))
    v_equal = torch.equal(bits(value_cache), bits(ref_v))
    k_rows_changed = int((bits(key_cache) != bits(before_k)).any(dim=-1).sum().item())
    v_rows_changed = int((bits(value_cache) != bits(before_v)).any(dim=-1).sum().item())
    written_k = torch.equal(bits(key_cache[block_idx, block_off]), bits(key))
    written_v = torch.equal(bits(value_cache[block_idx, block_off]), bits(value))
    result["correctness"] = {
        "k_cache_bitwise_equal_to_reference": k_equal,
        "v_cache_bitwise_equal_to_reference": v_equal,
        "k_rows_changed": k_rows_changed,
        "v_rows_changed": v_rows_changed,
        "written_rows_equal_inputs": written_k and written_v,
        # rows are (block, offset, head) vectors of head_dim: one per token per KV head.
        "expected_rows_changed": num_tokens * NUM_KV_HEADS,
        "pass": k_equal
        and v_equal
        and written_k
        and written_v
        and k_rows_changed == num_tokens * NUM_KV_HEADS
        and v_rows_changed == num_tokens * NUM_KV_HEADS,
    }

    # 2. kernel identity through torch.profiler (Kineto), in a child process:
    #    Kineto's CUPTI subscription would leave the repository's CUPTI extension
    #    (step 3, same process) without records, so each runs in its own process.
    child = subprocess.run(
        [sys.executable, "-P", __file__, "--kineto-child", str(num_tokens)],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if child.returncode != 0:
        raise RuntimeError(f"kineto child failed: {child.stderr[-2000:]}")
    cuda_kernels = json.loads(child.stdout.strip().splitlines()[-1])
    result["kineto_cuda_kernels"] = cuda_kernels
    result["kernel_identity_pass"] = len(cuda_kernels) == 1 and KERNEL in cuda_kernels[0]

    # 3. CUPTI capture with the runner's filter, then the timer the runner calls
    summary = cupti.profile_kernel(kernel, num_warmup=5, num_iter=50, kernel_name_contains=KERNEL)
    result["cupti"] = {
        "matched_kernel_names": sorted(set(summary.matched_kernel_names)),
        "matched_per_run": sorted(set(summary.matched_kernel_count_per_run)),
        "num_iter": summary.num_iter,
        "mean_us": summary.mean_ms * 1e3,
        "median_us": summary.median_ms * 1e3,
        "min_us": summary.min_ms * 1e3,
        "max_us": summary.max_ms * 1e3,
        "pass": all(KERNEL in n for n in summary.matched_kernel_names)
        and set(summary.matched_kernel_count_per_run) == {1},
    }
    result["timer_cupti_us"] = Timer.cupti(kernel, warmup=5, kernel_name=KERNEL) * 1e3
    for _ in range(20):
        kernel()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(200):
        kernel()
    end.record()
    torch.cuda.synchronize()
    result["events_warm_us_per_launch"] = start.elapsed_time(end) * 1e3 / 200
    result["pass"] = (
        result["correctness"]["pass"] and result["kernel_identity_pass"] and result["cupti"]["pass"]
    )
    return result


def kineto_child(num_tokens: int) -> None:
    """List the CUDA kernels one `reshape_and_cache_flash` call launches (JSON on
    the last stdout line). Same geometry and allocator as `probe`."""
    from torch.profiler import ProfilerActivity, profile
    from vllm import _custom_ops as ops
    from vllm.utils.torch_utils import create_kv_caches_with_random_flash

    nb = num_blocks(num_tokens)
    key = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    value = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    slot_mapping = torch.randperm(nb * BLOCK_SIZE, device="cuda", dtype=torch.int64)[:num_tokens]
    k_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    v_scale = torch.ones_like(k_scale)
    key_caches, value_caches = create_kv_caches_with_random_flash(
        nb,
        BLOCK_SIZE,
        1,
        NUM_KV_HEADS,
        HEAD_DIM,
        "auto",
        torch.bfloat16,
        seed=42,
        device="cuda",
        cache_layout=LAYOUT,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    def kernel():
        ops.reshape_and_cache_flash(
            key, value, key_cache, value_cache, slot_mapping, "auto", k_scale, v_scale
        )

    kernel()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        kernel()
        torch.cuda.synchronize()
    names = sorted(
        {
            e.name
            for e in prof.events()
            if str(e.device_type).endswith("CUDA")
            and "Memcpy" not in e.name
            and "Memset" not in e.name
        }
    )
    print(json.dumps(names))


def fp8_refused(create_kv_caches_with_random_flash) -> dict:
    try:
        create_kv_caches_with_random_flash(
            256,
            BLOCK_SIZE,
            1,
            NUM_KV_HEADS,
            HEAD_DIM,
            "fp8",
            torch.bfloat16,
            seed=42,
            device="cuda",
            cache_layout=LAYOUT,
        )
    except NotImplementedError as exc:
        return {"explicit_failure": True, "error": str(exc)}
    return {"explicit_failure": False, "error": "fp8 cache allocation unexpectedly succeeded"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, help="report JSON (required unless --kineto-child)")
    ap.add_argument("--expect-uuid")
    ap.add_argument("--num-tokens", type=int, nargs="+", default=[64, 8192])
    ap.add_argument("--kineto-child", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.kineto_child is not None:
        kineto_child(args.kineto_child)
        return
    if args.out is None:
        ap.error("--out is required")
    t0 = time.time()
    report: dict = {
        "host": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report["specs"] = []
    report["pass"] = False
    try:
        report["identity"] = identity(args.expect_uuid)
        import vllm
        from vllm import _custom_ops as ops
        from vllm.utils.torch_utils import create_kv_caches_with_random_flash

        from profiling.profilers import cupti_kernel_profiler as cupti
        from profiling.profilers.timer import Timer

        report["shim"] = {
            "vllm_file": vllm.__file__,
            "library": ops.MANIFEST["library"],
            "fork_commit": ops.MANIFEST["fork"]["commit"],
        }
        for n in args.num_tokens:
            report["specs"].append(probe(n, ops, create_kv_caches_with_random_flash, cupti, Timer))
        report["fp8_path"] = fp8_refused(create_kv_caches_with_random_flash)
        report["pass"] = (
            all(s["pass"] for s in report["specs"]) and report["fp8_path"]["explicit_failure"]
        )
    except Exception as exc:  # noqa: BLE001 - the report must record the failure
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["seconds"] = round(time.time() - t0, 1)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("identity", "pass", "seconds")}))
    for s in report["specs"]:
        print(
            f"num_tokens={s['num_tokens']}: correctness={s['correctness']['pass']} "
            f"kernels={s['kineto_cuda_kernels']} cupti={s['cupti']['pass']} "
            f"mean={s['cupti']['mean_us']:.2f}us timer={s['timer_cupti_us']:.2f}us "
            f"events_warm={s['events_warm_us_per_launch']:.2f}us"
        )
    print("fp8 path:", report["fp8_path"])
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
