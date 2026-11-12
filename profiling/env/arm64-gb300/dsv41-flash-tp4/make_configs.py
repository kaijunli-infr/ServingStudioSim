#!/usr/bin/env python3
"""Write the five Phase-0 profile configs for DeepSeek-V4.1-Flash-1017 on one GB300 tray
(vLLM recipe TP strategy at TP4, no EP, no spec decode; see gb300-dsv41-plan.md section 2-3).

    python3 make_configs.py --fork-env <venv root> [--run-root DIR] [--devices 0,1,2,3]
"""
import argparse, json
from pathlib import Path

SS = Path("/mnt/shared/home/dc/workspace/serving-studio")
GB300 = SS / "gb300"
MODEL = Path("/mnt/shared/weights/DeepSeek-V4.1-Flash-1017")   # HF rev 08c8657f, 48 shards, 476 GB
TRACES = SS / "ServingStudioSim" / "trace"
CORPUS = SS / "corpus" / "enwik9"

p = argparse.ArgumentParser()
p.add_argument("--fork-env", required=True, help="venv root of the instrumented vLLM fork")
p.add_argument("--run-root", default=str(SS / "ServingStudioSim" / "logs" / "20261112_dsv41_flash_gb300_tp4"))
p.add_argument("--devices", default="0,1,2,3")
a = p.parse_args()
RUN = Path(a.run_root); RUN.mkdir(parents=True, exist_ok=True)
fork_python = str(Path(a.fork_env) / "venv" / "bin" / "python")

# The recipe command (TP4), minus the tool-call/reasoning parsers and
# --mm-encoder-tp-mode (text only), plus --max-model-len 131072 --language-model-only.
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
    "VLLM_ENGINE_READY_TIMEOUT_S": "3600",          # recipe
    "VLLM_CACHE_ROOT": str(GB300 / "cache/vllm"),   # persistent inductor/compile cache
    # vLLM 0.31 persists FlashInfer autotune tactics only when this is set (envs.py); the first
    # boot tunes for 75-90 min, later boots reuse it.
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
for v in ENV.values():
    if v.startswith(str(GB300)): Path(v).mkdir(parents=True, exist_ok=True)

def server():
    return {
        "model_path": str(MODEL),
        "tp_size": 4, "dp_size": 1,
        # vLLM 0.31 default max_num_batched_tokens on a >=160 GiB GPU (api-server usage
        # context); the profiler always passes --max-num-batched-tokens, so state it.
        "chunk_size": 16384,
        "gpu_memory_utilization": 0.9,              # vLLM default
        "enforce_eager": False,
        "enable_server_load_tracking": True,        # /load endpoint: warmup + idle checks
        "api_server_count": 1,                      # vLLM/recipe default (the profiler would default to 4)
        "served_model_name": "dsv41-flash-1017",
        "startup_timeout": 7200.0,                  # first boot: 75-90 min FlashInfer autotune
        "extra_args": list(RECIPE_ARGS),
    }

def workload(trace, backend, saturated_c=None):
    w = {
        "frontend": {"type": "independent", "path": str(TRACES / trace)},
        "backend": {"type": backend},
        "text_file": str(CORPUS),
        "tokenizer": str(MODEL),
        "warmup": True,
        "max_model_len": 131072,
    }
    if saturated_c is None:
        w["arrival_mode"] = "trace-timed"
    else:
        w.update(arrival_mode="saturated", max_concurrency=saturated_c)
    return w

def nsys(delay=None, duration=None):
    n = {"executable": "/usr/local/bin/nsys", "capture_mode": "cuda_profiler_api",
         "cuda_graph_trace": "node", "cuda_event_trace": False}
    if delay is not None: n["capture_start_delay_seconds"] = delay
    if duration is not None: n["capture_duration_seconds"] = duration
    return n

PASSES = {
    # A: trace/quadrant_c48_96.csv, saturated at 48 in flight, run to completion
    "quadrant_nsys":     dict(kind="nsys", trace="quadrant_c48_96.csv", backend="vllm_tokens", c=48, nsys=nsys()),
    "quadrant_workload": dict(kind="workload_metrics", trace="quadrant_c48_96.csv", backend="vllm_tokens", c=48),
    # B: trace/diverse_100.csv, Poisson arrivals as recorded (1 req/s, last at 114.9 s)
    "diverse_nsys":      dict(kind="nsys", trace="diverse_100.csv", backend="vllm_tokens", c=None, nsys=nsys(delay=10.0, duration=60.0)),
    "diverse_workload":  dict(kind="workload_metrics", trace="diverse_100.csv", backend="vllm_tokens", c=None),
    "diverse_corpus":    dict(kind="token_corpus", trace="diverse_100.csv", backend="openai", c=None),
}
for name, ps in PASSES.items():
    d = RUN / name; d.mkdir(exist_ok=True)
    cfg = {
        "schema_version": 1,
        "name": f"dsv41_flash_1017_gb300_tp4_{name}",
        "log_dir": str(d / "profile"),
        "gpu": "NVIDIA GB300",
        "cuda_visible_devices": a.devices,
        "profile_kind": ps["kind"],
        "engine": "vllm",
        "fork_python": fork_python,
        "python_runtime": {"packages": [{"name": "nvtx", "version": "0.2.15", "index_url": "https://pypi.org/simple"}],
                           "environment": ENV, "lock_timeout_seconds": 60, "install_timeout_seconds": 600},
        "server": server(),
        "idle": {"enabled": True, "timeout": 600.0, "poll_interval": 0.5},
        "workload": workload(ps["trace"], ps["backend"], ps["c"]),
    }
    if ps["kind"] == "nsys":
        cfg["nsys"] = ps["nsys"]
    (d / "profile.json").write_text(json.dumps(cfg, indent=1) + "\n")
    print("wrote", d / "profile.json")
