"""Render the TP1 smoke and TP2 collective-identification profile configs (CPU-only).

    python make_configs.py --root $GB300_ROOT/serving-validation --out-dir DIR

Writes, per run, a request trace and an alignment `profile.yaml` for
`python -m launcher alignment profile`, run from the serving worktree
(`ServingStudioSim-serving`, whose `alignment/profiler/vllm` and
`alignment/load_generator/req-frontend` link to the pinned checkouts).

Runs:

- `tp1-smoke`: GPU 1, engine defaults. 3 single prefills of 256 tokens, then
  3 batches of 8 decodes (568 prompt, 64 output tokens).
- `tp2-default`: GPUs 2 and 3, engine defaults, the same trace plus 3 batches
  of 64 decodes.
- `tp2-nccl`: as `tp2-default` with the four switches that make every
  tensor-parallel all-reduce an unfused PyNCCL call
  (SERVING_VALIDATION_STAGE_A.md, section 2).

Batches start 1 s apart (`arrival_mode: trace-timed`) so they do not overlap.
Nothing here starts a GPU process.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MODEL_REPO = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
NSYS = "/usr/local/bin/nsys"
NCCL_FORCING_ARGS = [
    "--disable-custom-all-reduce",
    "--compilation-config",
    '{"pass_config": {"fuse_allreduce_rms": false}}',
]
NCCL_FORCING_ENV = {"VLLM_ALLREDUCE_USE_FLASHINFER": "0", "VLLM_ALLREDUCE_USE_SYMM_MEM": "0"}

# (case, requests per batch, prompt tokens, output tokens, batches)
SMOKE = [("prefill256", 1, 256, 1, 3), ("decode8", 8, 568, 64, 3)]
IDENTIFY = SMOKE + [("decode64", 64, 568, 64, 3)]
RUNS = {
    # name: (cuda_visible_devices, tp, cases, nccl_forced)
    "tp1-smoke": ("1", 1, SMOKE, False),
    "tp2-default": ("2,3", 2, IDENTIFY, False),
    "tp2-nccl": ("2,3", 2, IDENTIFY, True),
}


def snapshot_dir(root: Path) -> Path:
    name = "models--" + MODEL_REPO.replace("/", "--")
    return root / "hf" / "hub" / name / "snapshots" / MODEL_REVISION


def write_trace(path: Path, cases: list) -> int:
    rows, t_ms = [], 0.0
    for case, requests, prompt, output, batches in cases:
        for batch in range(batches):
            rows += [(f"{case}-b{batch}-{i}", prompt, output, t_ms) for i in range(requests)]
            t_ms += 1000.0
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "input_len", "output_len", "arrival_time"])
        writer.writerows(rows)
    return max(requests for _, requests, *_ in cases)


def profile(root: Path, out: Path, name: str) -> dict:
    devices, tp, cases, nccl = RUNS[name]
    trace = out / f"{name}.csv"
    concurrency = write_trace(trace, cases)
    model = str(snapshot_dir(root))
    environment = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "HF_HOME": str(root / "hf"),
        "HF_HUB_OFFLINE": "1",
        "VLLM_CACHE_ROOT": str(root / "cache" / "vllm"),
        "FLASHINFER_WORKSPACE_BASE": str(root / "cache" / "flashinfer"),
        "TRITON_CACHE_DIR": str(root / "cache" / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(root / "cache" / "inductor"),
        "CUDA_CACHE_PATH": str(root / "cache" / "nv-compute"),
        "VLLM_LOGGING_LEVEL": "INFO",
    }
    extra = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--no-enable-prefix-caching",
        "--seed",
        "0",
    ]
    if tp > 1:
        environment |= {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,ENV"}
    if nccl:
        extra += NCCL_FORCING_ARGS
        environment |= NCCL_FORCING_ENV
    return {
        "schema_version": 1,
        "name": f"llama31_8b_gb300_{name.replace('-', '_')}",
        "log_dir": str(root / "runs" / name),
        "gpu": "NVIDIA GB300",
        "cuda_visible_devices": devices,
        "engine": "vllm",
        "fork_python": str(root / "venv" / "bin" / "python"),
        "python_runtime": {
            # Already installed by build_env.sh; listed because the launcher only
            # accepts an environment block beside at least one pinned package.
            "packages": [
                {"name": "nvtx", "version": "0.2.15", "index_url": "https://pypi.org/simple"}
            ],
            "environment": environment,
            "lock_timeout_seconds": 30,
            "install_timeout_seconds": 30,
        },
        "server": {
            "model_path": model,
            "tp_size": tp,
            "chunk_size": 2048,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": False,
            "enable_server_load_tracking": True,
            "api_server_count": 1,
            "served_model_name": "llama31-8b-instruct",
            "startup_timeout": 600.0,
            "extra_args": extra,
        },
        "nsys": {"executable": NSYS, "capture_mode": "cuda_profiler_api"},
        "workload": {
            "frontend": {"type": "independent", "path": str(trace)},
            "backend": {"type": "vllm_tokens"},
            "text_file": str(root / "corpus" / "enwik9"),
            "tokenizer": model,
            "warmup": True,
            "arrival_mode": "trace-timed",
            "max_concurrency": concurrency,
            "max_model_len": 8192,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in RUNS:
        config = profile(args.root, args.out_dir, name)
        (args.out_dir / f"{name}.yaml").write_text(json.dumps(config, indent=1) + "\n")
        print(name, config["cuda_visible_devices"], config["server"]["extra_args"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
