"""Provenance manifest for GB300 profiling runs (CPU-only to collect).

Records what a measurement depended on: the GPUs as nvidia-smi reports them
(identity, firmware, clocks, power limits, ECC/MIG/persistence state, NVLink
and topology), the software stack (interpreter, torch/triton/flashinfer/
deep_gemm versions, CUDA toolkit), the source revisions (this repository, its
submodules, DeepGEMM), the host, the isolation environment, and the run
configuration the caller passes. It never initializes CUDA: torch is imported
only for its version strings, and GPU facts come from nvidia-smi.

    uv run --no-sync python profiling/env/arm64-gb300/collect_manifest.py \
        --out "$GB300_ROOT/artifacts/manifest.json" --run-config preset=... [k=v ...]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

GPU_QUERY_FIELDS = (
    "index,name,uuid,serial,pci.bus_id,vbios_version,driver_version,compute_cap,"
    "memory.total,power.limit,power.default_limit,power.max_limit,"
    "clocks.max.sm,clocks.max.memory,clocks.applications.graphics,clocks.applications.memory,"
    "persistence_mode,ecc.mode.current,mig.mode.current,compute_mode,"
    "inforom.img,fabric.state,fabric.status"
)


def run(cmd: list[str], timeout: int = 60) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def nvidia_smi() -> dict:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    csv = run(["nvidia-smi", f"--query-gpu={GPU_QUERY_FIELDS}", "--format=csv,noheader"])
    gpus = []
    if csv:
        names = GPU_QUERY_FIELDS.split(",")
        for line in csv.splitlines():
            values = [v.strip() for v in line.split(", ")]
            if len(values) == len(names):
                gpus.append(dict(zip(names, values, strict=True)))
    return {
        "cuda_visible_devices": visible,
        "gpus": gpus,
        "header": (run(["nvidia-smi"]) or "").splitlines()[2:4],
        "topology": run(["nvidia-smi", "topo", "-m"]),
        "nvlink_status": run(["nvidia-smi", "nvlink", "-s"]),
        "clocks_detail": run(["nvidia-smi", "-q", "-d", "CLOCK,POWER,PERFORMANCE"]),
    }


def git(args: list[str], cwd: Path = REPO_ROOT) -> str | None:
    return run(["git", "-C", str(cwd), *args])


def software() -> dict:
    out: dict = {
        "python": sys.version,
        "executable": sys.executable,
        "nvcc": run([str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "nvcc"), "--version"]),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "packages": {},
    }
    for name in ("torch", "triton", "flashinfer", "deep_gemm", "nvshmem", "numpy", "transformers", "pynvml"):
        try:
            module = __import__(name)
            out["packages"][name] = getattr(module, "__version__", "present")
        except Exception as exc:  # noqa: BLE001 - a missing package is a finding, not a crash
            out["packages"][name] = f"import failed: {exc.__class__.__name__}: {exc}"
    try:
        import torch

        out["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "git_version": torch.version.git_version,
            "cuda_initialized": torch.cuda.is_initialized(),
        }
    except Exception as exc:  # noqa: BLE001
        out["torch"] = f"unavailable: {exc}"
    freeze = run([sys.executable, "-m", "pip", "freeze"], timeout=120)
    if not freeze and shutil.which("uv"):
        freeze = run(["uv", "pip", "freeze", "--python", sys.executable], timeout=120)
    out["freeze"] = freeze.splitlines() if freeze else None
    return out


def sources() -> dict:
    submodules = {}
    status = git(["submodule", "status", "--recursive"]) or ""
    for line in status.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            submodules[parts[1]] = parts[0]
    return {
        "repo": str(REPO_ROOT),
        "commit": git(["rev-parse", "HEAD"]),
        "branch": git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(git(["status", "--porcelain", "--untracked-files=no"])),
        "remote_origin": git(["remote", "get-url", "origin"]),
        "submodules": submodules,
        "profile_db": os.environ.get("VIBESIM_PROFILE_DB"),
    }


def host() -> dict:
    return {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "os": run(["bash", "-lc", ". /etc/os-release && echo $PRETTY_NAME"]),
        "cpu": run(["bash", "-lc", "lscpu | grep -m1 'Model name'"]),
        "driver": run(["bash", "-lc", "cat /proc/driver/nvidia/version | head -1"]),
    }


def isolation() -> dict:
    keys = (
        "GB300_ROOT", "UV_PROJECT_ENVIRONMENT", "UV_CACHE_DIR", "TRITON_CACHE_DIR",
        "FLASHINFER_WORKSPACE_BASE", "TORCH_EXTENSIONS_DIR", "DG_JIT_CACHE_DIR",
        "CUDA_CACHE_PATH", "HF_HOME", "VIBESIM_PROFILE_CACHE_DIR", "VIBESIM_PROFILE_RUNTIME",
        "VIBESIM_PROFILE_DB", "VIBESIM_PROFILE_GPUS", "TORCH_CUDA_ARCH_LIST", "CUDA_VISIBLE_DEVICES",
    )
    return {k: os.environ.get(k) for k in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-config", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    manifest = {
        "schema": "gb300-profiling-manifest/1",
        "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_config": dict(kv.split("=", 1) for kv in args.run_config if "=" in kv),
        "host": host(),
        "nvidia_smi": nvidia_smi(),
        "software": software(),
        "sources": sources(),
        "isolation": isolation(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
