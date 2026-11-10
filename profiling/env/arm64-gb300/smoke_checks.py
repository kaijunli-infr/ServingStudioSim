"""GB300 smoke checks on the one visible GPU: device identity, correctness of
the four smoke kernels against fp32 references, first-call (JIT) time, and
steady-state latency distributions.

Mirrors the profiler's runners (shapes, layouts, FlashInfer wrappers), but
times with in-stream CUDA events over back-to-back launches (warm L2, no cache
flush), so the numbers are a sanity band next to the profiler's CUPTI
measurements, not a replacement.

    smoke_checks.py --expect-uuid GPU-... --only identity   # abort gate
    smoke_checks.py --expect-uuid GPU-... --out checks.json  # everything
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128  # Llama 3.1 8B
GEMM_N, GEMM_K = 6144, 4096  # Llama 3.1 8B fused QKV projection
HIDDEN = 4096
EPS = 1e-6


def dir_bytes(path: str | None) -> int:
    if not path or not os.path.isdir(path):
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def timed_first_call(fn):
    """Wall time of the first call (JIT compile + first launch), synchronized."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def latency_us(fn, warmup: int = 10, iters: int = 200) -> dict:
    """Per-launch GPU time from events recorded in-stream between back-to-back
    launches, with no host sync inside the loop: the GPU stays busy, so each
    interval is kernel time plus any inter-kernel gap, not launch latency."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(iters + 1)]
    events[0].record()
    for i in range(iters):
        fn()
        events[i + 1].record()
    torch.cuda.synchronize()
    samples = sorted(events[i].elapsed_time(events[i + 1]) * 1000.0 for i in range(iters))
    q = lambda p: samples[min(len(samples) - 1, int(round(p * (len(samples) - 1))))]
    return {
        "method": "in-stream cuda events, back-to-back launches",
        "iters": iters,
        "min": samples[0],
        "p50": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
        "max": samples[-1],
        "mean": statistics.fmean(samples),
        "stdev": statistics.pstdev(samples),
    }


def compare(out: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> dict:
    out32 = out.float()
    diff = (out32 - ref).abs()
    tol = atol + rtol * ref.abs()
    mismatches = int((diff > tol).sum().item())
    return {
        "atol": atol,
        "rtol": rtol,
        "max_abs_err": diff.max().item(),
        "mean_abs_err": diff.mean().item(),
        "ref_max_abs": ref.abs().max().item(),
        "mismatches": mismatches,
        "elements": ref.numel(),
        "pass": mismatches == 0 and math.isfinite(diff.max().item()),
    }


def identity(expect_uuid: str | None) -> dict:
    count = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    uuid = f"GPU-{props.uuid}"
    free, total = torch.cuda.mem_get_info(0)
    x = torch.ones(1024, 1024, dtype=torch.bfloat16, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    exact = bool((y == 1024).all().item()) and y.float().sum().item() == float(1024**3)
    out = {
        "device_count": count,
        "name": props.name,
        "capability": [props.major, props.minor],
        "uuid": uuid,
        "expected_uuid": expect_uuid,
        "uuid_match": (expect_uuid is None) or (uuid == expect_uuid),
        "total_memory_gib": total / 2**30,
        "free_memory_gib": free / 2**30,
        "sm_count": props.multi_processor_count,
        "ones_matmul_exact": exact,
        "pass": count == 1
        and props.name == "NVIDIA GB300"
        and (props.major, props.minor) == (10, 3)
        and ((expect_uuid is None) or uuid == expect_uuid)
        and exact,
    }
    return out


def gemm() -> list[dict]:
    results = []
    for m in (64, 2048, 4096):
        g = torch.Generator(device="cuda").manual_seed(1000 + m)
        a = torch.randn(m, GEMM_K, dtype=torch.bfloat16, device="cuda", generator=g)
        w = torch.randn(GEMM_N, GEMM_K, dtype=torch.bfloat16, device="cuda", generator=g)
        out, first_s = timed_first_call(lambda: F.linear(a, w))
        ref = a.float() @ w.float().T
        # bf16 output rounding of values ~N(0, sqrt(K)=64): half-ulp near 0.25 at
        # |v|=64, 1.0 at |v|=256; fp32 accumulation inside is exact enough.
        results.append(
            {
                "m": m, "n": GEMM_N, "k": GEMM_K, "dtype": "bf16", "backend": "torch_linear",
                "first_call_s": first_s,
                "correctness": compare(out, ref, atol=1.0, rtol=2e-2),
                "latency_us": latency_us(lambda: F.linear(a, w)),
                "tflops_at_p50": None,
            }
        )
        r = results[-1]
        r["tflops_at_p50"] = 2 * m * GEMM_N * GEMM_K / (r["latency_us"]["p50"] * 1e-6) / 1e12
    return results


def rmsnorm(cache_dir: str | None) -> list[dict]:
    import flashinfer.norm

    results = []
    for m in (64, 2048):
        g = torch.Generator(device="cuda").manual_seed(2000 + m)
        x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=g)
        w = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda", generator=g)
        before = dir_bytes(cache_dir)
        out, first_s = timed_first_call(lambda: flashinfer.norm.rmsnorm(x, w, eps=EPS))
        x32 = x.float()
        ref = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS) * w.float()
        results.append(
            {
                "m": m, "hidden": HIDDEN, "dtype": "bf16", "backend": "flashinfer",
                "first_call_s": first_s,
                "cache_growth_bytes": dir_bytes(cache_dir) - before,
                "correctness": compare(out, ref, atol=5e-2, rtol=2e-2),
                "latency_us": latency_us(lambda: flashinfer.norm.rmsnorm(x, w, eps=EPS)),
            }
        )
        r = results[-1]
        moved = 2 * m * HIDDEN * 2 + HIDDEN * 2
        r["gbps_at_p50"] = moved / (r["latency_us"]["p50"] * 1e-6) / 1e9
    return results


def prefill(cache_dir: str | None) -> list[dict]:
    import flashinfer
    from profiling.runners.attention import _common

    results = []
    for prefix_len in (0, 2048):
        append_len = 2048
        torch.manual_seed(3000 + prefix_len)
        inp = _common.build_ragged_inputs(
            q_len=append_len, kv_len=prefix_len + append_len,
            num_qo_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM,
            q_dtype="bf16", kv_dtype="bf16", o_dtype="bf16",
        )
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            _common.make_workspace(), kv_layout="NHD", backend="fa2"
        )
        plan = dict(
            qo_indptr=inp.qo_indptr, kv_indptr=inp.kv_indptr,
            num_qo_heads=HEADS, num_kv_heads=KV_HEADS,
            head_dim_qk=HEAD_DIM, head_dim_vo=HEAD_DIM, causal=True,
            q_data_type=torch.bfloat16,
        )
        before = dir_bytes(cache_dir)

        def first():
            wrapper.plan(**plan)
            return wrapper.run(inp.q, inp.k, inp.v)

        out, first_s = timed_first_call(first)
        # fp32 reference: GQA causal attention of the append block over prefix + itself.
        q = inp.q.float().permute(1, 0, 2)  # [H, q_len, D]
        k = inp.k.float().permute(1, 0, 2).repeat_interleave(HEADS // KV_HEADS, dim=0)
        v = inp.v.float().permute(1, 0, 2).repeat_interleave(HEADS // KV_HEADS, dim=0)
        scores = q @ k.transpose(1, 2) / math.sqrt(HEAD_DIM)  # [H, q_len, kv_len]
        qi = torch.arange(append_len, device="cuda")[:, None]
        kj = torch.arange(prefix_len + append_len, device="cuda")[None, :]
        scores = scores.masked_fill(kj > prefix_len + qi, float("-inf"))
        ref = (torch.softmax(scores, dim=-1) @ v).permute(1, 0, 2)  # [q_len, H, D]
        del scores
        results.append(
            {
                "num_qo_heads": HEADS, "num_kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
                "prefix_len": prefix_len, "append_len": append_len, "dtype": "bf16", "backend": "fa2",
                "first_call_s_plan_and_run": first_s,
                "cache_growth_bytes": dir_bytes(cache_dir) - before,
                "correctness": compare(out, ref, atol=2e-2, rtol=2e-2),
                "latency_us": latency_us(lambda: wrapper.run(inp.q, inp.k, inp.v)),
            }
        )
        r = results[-1]
        flops = _common.attention_flops(
            q_len=append_len, kv_len=prefix_len + append_len,
            num_qo_heads=HEADS, head_dim=HEAD_DIM, causal=True,
        )
        r["tflops_at_p50"] = flops / (r["latency_us"]["p50"] * 1e-6) / 1e12
    return results


def decode(cache_dir: str | None) -> list[dict]:
    import flashinfer
    from profiling.runners.attention import _common

    page = 16
    results = []
    for batch_size, total_tokens in ((1, 4096), (64, 38400)):
        seq_len = max(1, total_tokens // batch_size)
        torch.manual_seed(4000 + batch_size)
        inp = _common.build_paged_decode_inputs(
            batch_size=batch_size, seq_len=seq_len,
            num_qo_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM,
            q_dtype="bf16", kv_dtype="bf16", o_dtype="bf16", page_size=page,
        )
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            _common.make_workspace(), kv_layout="NHD", backend="fa2", use_tensor_cores=True
        )
        plan = dict(
            indptr=inp.kv_indptr, indices=inp.kv_indices, last_page_len=inp.kv_last_page_len,
            num_qo_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM, page_size=page,
            q_data_type=torch.bfloat16,
        )
        cache = (inp.k_cache, inp.v_cache)
        before = dir_bytes(cache_dir)

        def first():
            wrapper.plan(**plan)
            return wrapper.run(inp.q, cache)

        out, first_s = timed_first_call(first)
        # fp32 reference per request: one query over its seq_len cached tokens.
        refs = []
        indptr = inp.kv_indptr.tolist()
        for b in range(batch_size):
            pages = inp.kv_indices[indptr[b] : indptr[b + 1]]
            k = inp.k_cache[pages].reshape(-1, KV_HEADS, HEAD_DIM)[:seq_len].float()
            v = inp.v_cache[pages].reshape(-1, KV_HEADS, HEAD_DIM)[:seq_len].float()
            k = k.permute(1, 0, 2).repeat_interleave(HEADS // KV_HEADS, dim=0)  # [H, S, D]
            v = v.permute(1, 0, 2).repeat_interleave(HEADS // KV_HEADS, dim=0)
            qb = inp.q[b].float()[:, None, :]  # [H, 1, D]
            scores = qb @ k.transpose(1, 2) / math.sqrt(HEAD_DIM)
            refs.append((torch.softmax(scores, dim=-1) @ v)[:, 0, :])  # [H, D]
        ref = torch.stack(refs)  # [B, H, D]
        results.append(
            {
                "num_qo_heads": HEADS, "num_kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
                "batch_size": batch_size, "total_tokens": total_tokens, "seq_len": seq_len,
                "dtype": "bf16", "backend": "fa2", "page_size": page,
                "first_call_s_plan_and_run": first_s,
                "cache_growth_bytes": dir_bytes(cache_dir) - before,
                "correctness": compare(out, ref, atol=2e-2, rtol=2e-2),
                "latency_us": latency_us(lambda: wrapper.run(inp.q, cache)),
            }
        )
        r = results[-1]
        r["gbps_at_p50"] = inp.bytes_accessed / (r["latency_us"]["p50"] * 1e-6) / 1e9
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--expect-uuid")
    parser.add_argument("--only", choices=["identity", "all"], default="all")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cache_dir = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    report: dict = {"identity": identity(args.expect_uuid)}
    print("identity:", json.dumps(report["identity"]))
    if not report["identity"]["pass"]:
        print("ABORT: device identity check failed", file=sys.stderr)
        if args.out:
            args.out.write_text(json.dumps(report, indent=2) + "\n")
        return 3
    if args.only == "all":
        t0 = time.perf_counter()
        for name, fn in (("gemm", gemm), ("rmsnorm", lambda: rmsnorm(cache_dir)),
                         ("prefill_fa2", lambda: prefill(cache_dir)), ("decode_fa2", lambda: decode(cache_dir))):
            t = time.perf_counter()
            try:
                report[name] = fn()
            except Exception as exc:  # noqa: BLE001 - a failing kernel is the finding
                report[name] = {"error": f"{type(exc).__name__}: {exc}"}
            report.setdefault("wall_s", {})[name] = time.perf_counter() - t
            for r in report[name] if isinstance(report[name], list) else []:
                c = r["correctness"]
                print(
                    f"{name}: {json.dumps({k: v for k, v in r.items() if k not in ('correctness', 'latency_us')})}"
                    f" | pass={c['pass']} max_abs_err={c['max_abs_err']:.4g} mismatches={c['mismatches']}"
                    f" | p50={r['latency_us']['p50']:.1f}us p90={r['latency_us']['p90']:.1f}us p99={r['latency_us']['p99']:.1f}us"
                )
            if "error" in report[name]:
                print(f"{name}: ERROR {report[name]['error']}")
        report["wall_s"]["total"] = time.perf_counter() - t0
        report["max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
