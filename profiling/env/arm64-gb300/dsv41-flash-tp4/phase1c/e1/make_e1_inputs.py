#!/usr/bin/env python3
"""Write the E1 request traces and launcher profile configs (CPU-only; nothing is launched).

    python3 make_e1_inputs.py [--run-root DIR] [--devices 0,1,2,3]

Design (see E1-AUTHORIZATION-PACKET.md section 1): decode-only cells at fixed batch b in B at three
per-request context lengths KV in KVS, trace-timed arrivals so that one vLLM launch and one nsys session
cover every cell; cell k's requests arrive together after cell k-1's estimated duration x MARGIN + GAP_S.
Every request has output_len = OUT_LEN tokens (ignore_eos is always sent by req-frontend), so a cell
decodes at exactly b requests once its prefills are done and all b requests finish within a few
iterations of each other.

The server block, environment and nsys block are copied from the Phase 0 generator
(dsv41-gb300/make_configs.py, sha256 92b3f807...) so the recipe is identical; the only differences are
the workload trace, the request backend (openai, which is the only backend that returns routed experts)
and the opt-in `workload.capture_routed_experts` key that needs the launcher patch in this directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

SS = Path("/mnt/shared/home/dc/workspace/serving-studio")
GB300 = SS / "gb300"
MODEL = Path("/mnt/shared/weights/DeepSeek-V4.1-Flash-1017")   # HF rev 08c8657f
REPO = SS / "ServingStudioSim"
E1 = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1"
CORPUS = SS / "corpus" / "enwik9"
FORK_ENV = GB300 / "fork-env-v0.31.0"

B = [1, 2, 4, 8, 16, 24, 32, 48]
KVS = [4096, 32768, 98304]            # per-request prompt tokens ("4k / 32k / 96k")
OUT_LEN = 1024                        # generated tokens per request (ignore_eos)
WAVES = {1: 2, 2: 2, 4: 2, 8: 2, 16: 1, 24: 1, 32: 1, 48: 1}
CHUNK = 16384
CHUNK_S = 0.235                       # 16k prefill chunk wall (Phase 0: 220 ms + 0.29 ms/1k kv)
WALL_MS = {1: 4.65, 2: 4.41, 4: 4.60, 8: 5.20, 16: 6.30, 24: 7.73, 32: 8.54, 48: 9.93}  # Phase 1A medians
MARGIN = 1.5
GAP_S = 5.0
MIN_PURE_DECODE = 400                 # matrix requirement per cell

DRY_CELLS = [(8, 32768), (48, 32768)]


def prefill_iters(b: int, kv: int) -> int:
    """Chunk iterations a wave needs: prompts <= 16k pack several per iteration, longer ones chunk."""
    if kv <= CHUNK:
        return math.ceil(b / (CHUNK // kv))
    return b * math.ceil(kv / CHUNK)


def wave_duration_s(b: int, kv: int) -> float:
    return prefill_iters(b, kv) * CHUNK_S + OUT_LEN * WALL_MS[b] / 1e3


def pure_decode_iters(b: int, kv: int) -> int:
    """Lower bound: the first request decodes during the other prefills, so the pure phase is shorter."""
    return OUT_LEN - prefill_iters(b, kv)


def schedule(cells, waves) -> tuple[list[dict], list[dict]]:
    rows, plan, t = [], [], 0.0
    for (b, kv) in cells:
        for w in range(waves.get(b, 1)):
            dur = wave_duration_s(b, kv)
            for r in range(b):
                rows.append({"id": f"e1_b{b}_kv{kv // 1024}k_w{w}_r{r:02d}", "input_len": kv,
                             "output_len": OUT_LEN, "arrival_time": f"{t * 1000.0:.1f}"})
            plan.append({"b": b, "kv": kv, "wave": w, "arrival_s": round(t, 1), "prefill_iters": prefill_iters(b, kv),
                         "est_duration_s": round(dur, 1), "pure_decode_iters_min": pure_decode_iters(b, kv)})
            t += dur * MARGIN + GAP_S
    return rows, plan


def write_trace(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "input_len", "output_len", "arrival_time"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


# --- Phase 0 recipe, verbatim from dsv41-gb300/make_configs.py -------------------------------------
RECIPE_ARGS = [
    "--tokenizer-mode", "deepseek_v41",
    "--attention-config", json.dumps({"backend": "FLASHINFER_MLA_SPARSE_DSV41",
                                      "indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": True},
                                     separators=(",", ":")),
    "--kv-cache-dtype", "fp8",
    "--max-num-seqs", "128",
    "--engram-config", json.dumps({"cpu_offload": True}, separators=(",", ":")),
    "--max-model-len", "131072",
    "--language-model-only",
]
ENV = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
    "VLLM_CACHE_ROOT": str(GB300 / "cache/vllm"),
    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": str(GB300 / "cache/flashinfer-autotune"),
    "FLASHINFER_WORKSPACE_BASE": str(GB300 / "cache/flashinfer"),
    "TRITON_CACHE_DIR": str(GB300 / "cache/triton"),
    "TORCHINDUCTOR_CACHE_DIR": str(GB300 / "cache/inductor"),
    "DG_JIT_CACHE_DIR": str(GB300 / "cache/deepgemm"),
    "CUDA_CACHE_PATH": str(GB300 / "cache/nv-compute"),
    "HF_HOME": str(GB300 / "cache/huggingface"),
    "HF_HUB_OFFLINE": "1",
    "VLLM_LOGGING_LEVEL": "INFO",
}


def server():
    return {
        "model_path": str(MODEL), "tp_size": 4, "dp_size": 1, "chunk_size": CHUNK,
        "gpu_memory_utilization": 0.9, "enforce_eager": False, "enable_server_load_tracking": True,
        "api_server_count": 1, "served_model_name": "dsv41-flash-1017", "startup_timeout": 7200.0,
        "extra_args": list(RECIPE_ARGS),
    }


def workload(trace: Path, backend: str, capture_routes: bool):
    w = {
        "frontend": {"type": "independent", "path": str(trace)},
        "backend": {"type": backend},
        "text_file": str(CORPUS), "tokenizer": str(MODEL), "warmup": True,
        "max_model_len": 131072, "arrival_mode": "trace-timed",
    }
    if capture_routes:
        w["capture_routed_experts"] = True      # needs launcher-routes-optin.patch
    return w


def nsys(delay=10.0):
    return {"executable": "/usr/local/bin/nsys", "capture_mode": "cuda_profiler_api",
            "cuda_graph_trace": "node", "cuda_event_trace": False, "capture_start_delay_seconds": delay}


PASSES = {
    # dry-run pair (no nsys): hook off / hook on, same trace, same backend -> overhead and token identity
    "e1_dry_off":   dict(kind="workload_metrics", trace="e1_dry.csv", backend="openai", routes=False),
    "e1_dry_on":    dict(kind="workload_metrics", trace="e1_dry.csv", backend="openai", routes=True),
    # the capture (Option A: with routes; needs the launcher patch)
    "e1_main":      dict(kind="nsys", trace="e1_main.csv", backend="openai", routes=True, nsys=nsys()),
    # fallback (Option B: timing only, no routes, valid with today's launcher; Phase 0 backend)
    "e1_main_norouts": dict(kind="nsys", trace="e1_main.csv", backend="vllm_tokens", routes=False, nsys=nsys()),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=str(REPO / "logs" / "20261114_dsv41_flash_gb300_tp4_e1"))
    ap.add_argument("--devices", default="0,1,2,3")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent), help="where traces/ configs/ are written")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)

    cells = [(b, kv) for kv in KVS for b in B]
    main_rows, main_plan = schedule(cells, WAVES)
    dry_rows, dry_plan = schedule(DRY_CELLS, {b: 1 for b in B})
    write_trace(out / "traces" / "e1_main.csv", main_rows)
    write_trace(out / "traces" / "e1_dry.csv", dry_rows)
    for p in main_plan:
        assert p["pure_decode_iters_min"] >= MIN_PURE_DECODE, p

    hashes = {}
    for name, ps in PASSES.items():
        trace_in_repo = E1 / "traces" / ps["trace"]
        run_dir = Path(a.run_root) / name
        cfg = {
            "schema_version": 1,
            "name": f"dsv41_flash_1017_gb300_tp4_{name}",
            "log_dir": str(run_dir / "profile"),
            "gpu": "NVIDIA GB300",
            "cuda_visible_devices": a.devices,
            "profile_kind": ps["kind"],
            "engine": "vllm",
            "fork_python": str(FORK_ENV / "venv" / "bin" / "python"),
            "python_runtime": {"packages": [{"name": "nvtx", "version": "0.2.15", "index_url": "https://pypi.org/simple"}],
                               "environment": ENV, "lock_timeout_seconds": 60, "install_timeout_seconds": 600},
            "server": server(),
            "idle": {"enabled": True, "timeout": 600.0, "poll_interval": 0.5},
            "workload": workload(trace_in_repo, ps["backend"], ps["routes"]),
        }
        if ps["kind"] == "nsys":
            cfg["nsys"] = ps["nsys"]
        d = out / "configs" / name
        d.mkdir(exist_ok=True)
        (d / "profile.json").write_text(json.dumps(cfg, indent=1) + "\n")
        hashes[f"configs/{name}/profile.json"] = sha256(d / "profile.json")
    for t in ("e1_main.csv", "e1_dry.csv"):
        hashes[f"traces/{t}"] = sha256(out / "traces" / t)

    total_main = main_plan[-1]["arrival_s"] + main_plan[-1]["est_duration_s"] * MARGIN + GAP_S
    total_dry = dry_plan[-1]["arrival_s"] + dry_plan[-1]["est_duration_s"] * MARGIN + GAP_S
    summary = {
        "design": {"B": B, "KV": KVS, "OUT_LEN": OUT_LEN, "WAVES": WAVES, "CHUNK_S": CHUNK_S, "WALL_MS": WALL_MS,
                   "MARGIN": MARGIN, "GAP_S": GAP_S, "MIN_PURE_DECODE": MIN_PURE_DECODE},
        "main": {"requests": len(main_rows), "waves": len(main_plan), "trace_span_s": round(total_main, 1),
                 "prefill_iters": sum(p["prefill_iters"] for p in main_plan),
                 "decode_iters": OUT_LEN * len(main_plan),
                 "pure_decode_iters_min": sum(p["pure_decode_iters_min"] for p in main_plan),
                 "generated_tokens": len(main_rows) * OUT_LEN,
                 "max_kv_tokens_resident": max(p["b"] * (p["kv"] + OUT_LEN) for p in main_plan),
                 "plan": main_plan},
        "dry": {"requests": len(dry_rows), "trace_span_s": round(total_dry, 1), "plan": dry_plan},
        "sha256": hashes,
    }
    (out / "E1-INPUTS.json").write_text(json.dumps(summary, indent=1) + "\n")
    with (out / "INPUTS.sha256").open("w") as fh:
        for k, v in sorted(hashes.items()):
            fh.write(f"{v}  {k}\n")
    print(json.dumps({k: v for k, v in summary["main"].items() if k != "plan"}, indent=1))
    print("dry", {k: v for k, v in summary["dry"].items() if k != "plan"})
    print(f"{'b':>3} {'kv':>6} w {'arrival_s':>9} {'prefill':>7} {'dur_s':>6} {'pure>=':>6}")
    for p in main_plan:
        print(f"{p['b']:>3} {p['kv']:>6} {p['wave']} {p['arrival_s']:>9} {p['prefill_iters']:>7} {p['est_duration_s']:>6} {p['pure_decode_iters_min']:>6}")


if __name__ == "__main__":
    main()
