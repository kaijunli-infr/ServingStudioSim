"""Pre-fill probe for the GB300 TP1 campaign, one process, one visible GPU.

1. Device identity gate (same checks as smoke_checks.py).
2. Triton on SM103: which ptxas Triton picks, what target the compiled kernel
   carries, and whether the profiler's own elementwise kernels (fan-in sum and
   zero fill, uint8) produce the right bytes at two shapes.
3. An independent timing cross-check for the CUPTI extension's Kernel10
   record reading: torch.profiler (Kineto, its own CUPTI client) kernel
   durations for the smoke GEMM shapes, to compare with the profiler's
   Timer.cupti numbers measured in other processes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_checks import identity  # noqa: E402


def triton_probe() -> dict:
    import triton
    from triton.backends.nvidia import compiler as nv

    cap = torch.cuda.get_device_capability(0)
    arch = cap[0] * 10 + cap[1]
    tool = nv.get_ptxas(arch)
    ptxas_path = str(tool.path)
    version = subprocess.check_output([ptxas_path, "--version"], text=True).strip().splitlines()[-2:]
    out = {
        "triton_version": triton.__version__,
        "capability": list(cap),
        "ptxas_selected_by_triton": ptxas_path,
        "ptxas_version": version,
        "sm_arch_from_capability": nv.sm_arch_from_capability(arch),
    }

    from profiling.runners.elementwise import triton as rt

    # The profiler's kernels are @autotune'd; drive them exactly as the runner does.
    results = []
    for input_bytes, output_bytes in ((8192 * 2048, 8192 * 2048), (28672 * 2 * 1024, 14336 * 2 * 1024), (0, 4096 * 64)):
        fan_in = 0 if input_bytes == 0 else rt._rounded_fan_in(input_bytes, output_bytes)
        g = torch.Generator(device="cuda").manual_seed(input_bytes + output_bytes)
        inp = torch.randint(0, 256, (max(1, fan_in) * output_bytes,), dtype=torch.uint8, device="cuda", generator=g)
        out_t = torch.full((output_bytes,), 7, dtype=torch.uint8, device="cuda")
        use_int64 = rt._requires_int64_offsets(input_bytes, output_bytes)

        def grid(meta):
            return (triton.cdiv(output_bytes, meta["BLOCK_SIZE"]),)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if fan_in == 0:
            rt._elementwise_zero_kernel[grid](out_t, OUTPUT_SIZE=output_bytes, USE_INT64=use_int64)
            ref = torch.zeros_like(out_t)
        else:
            rt._elementwise_fan_kernel[grid](inp, out_t, OUTPUT_SIZE=output_bytes, FAN_IN=fan_in, USE_INT64=use_int64)
            # int32 accumulate, then truncation to uint8 == sum mod 256
            ref = inp.view(fan_in, output_bytes).to(torch.int32).sum(0).to(torch.uint8)
        torch.cuda.synchronize()
        first_s = time.perf_counter() - t0
        mism = int((out_t != ref).sum().item())
        results.append(
            {"input_size_bytes": input_bytes, "output_size_bytes": output_bytes, "fan_in": fan_in,
             "first_launch_s_incl_autotune": first_s, "mismatches": mism, "elements": output_bytes, "pass": mism == 0}
        )
    out["kernels"] = results

    # What the compiled binary actually targets: read it off the autotuner's best kernel.
    try:
        kern = rt._elementwise_fan_kernel
        cache = getattr(kern, "cache", None) or getattr(getattr(kern, "fn", None), "cache", None)
        targets = set()
        if cache:
            for dev_cache in cache.values():
                for compiled in dev_cache.values():
                    ptx = compiled.asm.get("ptx", "")
                    line = next((l for l in ptx.splitlines() if l.startswith(".target")), "")
                    targets.add(line.strip())
                    if "cubin" in compiled.asm:
                        targets.add(f"cubin:{len(compiled.asm['cubin'])}B")
        out["compiled_targets"] = sorted(targets)
    except Exception as exc:  # noqa: BLE001
        out["compiled_targets_error"] = f"{type(exc).__name__}: {exc}"
    out["pass"] = all(r["pass"] for r in results)
    return out


def kineto_gemm() -> list[dict]:
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile

    out = []
    for m in (64, 2048, 4096):
        a = torch.randn(m, 4096, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(6144, 4096, dtype=torch.bfloat16, device="cuda")
        flush = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
        for _ in range(5):
            F.linear(a, w)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(50):
                flush.zero_()  # cold-ish L2, as Timer.cupti flushes before each launch
                F.linear(a, w)
            torch.cuda.synchronize()
        durations = []
        names = {}
        for evt in prof.events():
            if evt.device_type.name != "CUDA":
                continue
            low = evt.name.lower()
            if any(t in low for t in ("elementwise", "fill", "memset", "memcpy", "vectorized")):
                continue  # the L2-flush zero_() kernel
            durations.append(evt.time_range.elapsed_us())
            names[evt.name] = names.get(evt.name, 0) + 1
        if not durations:
            names = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})[:8]
            out.append({"m": m, "error": "no GEMM kernel captured"})
            continue
        durations.sort()
        out.append(
            {"m": m, "n": 6144, "k": 4096, "kernels_captured": len(durations),
             "kineto_us": {"min": durations[0], "p50": durations[len(durations) // 2], "max": durations[-1],
                           "mean": statistics.fmean(durations)},
             "flops_formula": "2*m*n*k", "flops": 2 * m * 6144 * 4096,
             "tflops_at_p50": 2 * m * 6144 * 4096 / (durations[len(durations) // 2] * 1e-6) / 1e12}
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-uuid")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"identity": identity(args.expect_uuid)}
    print("identity:", json.dumps(report["identity"]))
    if not report["identity"]["pass"]:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        return 3
    for name, fn in (("triton", triton_probe), ("kineto_gemm", kineto_gemm)):
        try:
            report[name] = fn()
        except Exception as exc:  # noqa: BLE001
            report[name] = {"error": f"{type(exc).__name__}: {exc}", "pass": False}
        print(f"{name}: {json.dumps(report[name], default=str)[:1500]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    ok = report["triton"].get("pass", False)
    print("TRITON PROBE", "PASS" if ok else "FAIL")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
