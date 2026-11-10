#!/usr/bin/env python
"""Build the GB300 `vllm` shim extension from the pinned fork sources.

    python build_shim.py --fork $GB300_ROOT/src/vllm-fork --out $GB300_ROOT/vllm-shim

Compiles the fork's unmodified `csrc/libtorch_stable/cache_kernels.cu` together
with `shim_bindings.cpp` (next to this file) into one shared library with
`torch.utils.cpp_extension`, for the arch in `TORCH_CUDA_ARCH_LIST` (default
10.3), with the nvcc flags the fork's CMake uses for its libtorch-stable
extension (torch's common flags, `-DENABLE_FP8`, the `__CUDA_NO_HALF_*`
defines removed, C++20). Refuses to build when the fork checkout is not at
the pinned commit or a pinned file's hash differs. Writes `manifest.json`
(commit, file hashes, toolchain, flags, library hash) into `--out`, which
`vllm/_custom_ops.py` reads to find and trust the library. No GPU is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

PINNED_COMMIT = "e1423a4d7496dc6535862ae11bab8cb4d87b155f"
PINNED_REMOTE = "https://github.com/serendipity-zk/vllm.git"
PINNED_FILES = {
    "csrc/libtorch_stable/cache_kernels.cu": (
        "5ea4d65b32a10949bddc01fa788a78a5ab6532ee4fa62946f217806a8e754c12"
    ),
    "csrc/libtorch_stable/torch_bindings.cpp": (
        "67f3806cca8a5ca3b9f12e7e418e902a587a3cfb18a39a2fe2a7f0129284c6f7"
    ),
    "vllm/utils/torch_utils.py": (
        "31e8a695f7efa52584ce20cc33edfb643b70c28ce83d273feb34326c61ab631c"
    ),
    "vllm/_custom_ops.py": "5788e575730631725ba05f0756d39958174a5381c790020ba6af07ca5309c078",
}
EXT_NAME = "vllm_shim_cache_ops"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_fork(fork: Path) -> dict:
    head = subprocess.run(
        ["git", "-C", str(fork), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != PINNED_COMMIT:
        sys.exit(f"fork checkout {fork} is at {head}, not the pinned {PINNED_COMMIT}")
    hashes = {}
    for rel, expected in PINNED_FILES.items():
        actual = sha256(fork / rel)
        if actual != expected:
            sys.exit(f"{rel}: sha256 {actual} differs from the pinned {expected}")
        hashes[rel] = actual
    return {"remote": PINNED_REMOTE, "commit": head, "files": hashes}


def check_bindings_verbatim(fork: Path, shim_bindings: Path) -> None:
    """Every schema line and the impl line of shim_bindings.cpp must occur in the
    fork's torch_bindings.cpp: the shim registers what the fork registers."""
    fork_text = (fork / "csrc/libtorch_stable/torch_bindings.cpp").read_text()
    shim_text = shim_bindings.read_text()
    schema = re.search(r"ops\.def\((.*?)\);", shim_text, re.S)
    impl = re.search(r"ops\.impl\(.*\);", shim_text)
    if not schema or not impl:
        sys.exit("shim_bindings.cpp: no ops.def(...) / ops.impl(...) found")
    literals = re.findall(r'"([^"\n]*)"', schema.group(1))
    for literal in literals + [impl.group(0)]:
        if literal not in fork_text:
            sys.exit(f"shim_bindings.cpp text not found verbatim in the fork's torch_bindings.cpp: {literal!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", required=True, type=Path, help="fork checkout at the pinned commit")
    ap.add_argument("--out", required=True, type=Path, help="shim root (gets build/ and manifest.json)")
    ap.add_argument("--arch", default=os.environ.get("TORCH_CUDA_ARCH_LIST", "10.3"))
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    fork = args.fork.resolve()
    out = args.out.resolve()
    shim_bindings = here / "shim_bindings.cpp"

    provenance = check_fork(fork)
    check_bindings_verbatim(fork, shim_bindings)

    os.environ["TORCH_CUDA_ARCH_LIST"] = args.arch
    import torch
    from torch.utils import cpp_extension

    if not cpp_extension.CUDA_HOME:
        sys.exit("CUDA_HOME is not set (source gb300.env)")
    build_dir = out / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    sources = [str(fork / "csrc/libtorch_stable/cache_kernels.cu"), str(shim_bindings)]
    # Only csrc/, as the fork's CMake: quoted includes then resolve exactly as in
    # its build (cache_kernels.cu finds libtorch_stable/torch_utils.h beside
    # itself, csrc/attention/dtype_fp8.cuh finds csrc/torch_utils.h).
    include_paths = [str(fork / "csrc")]
    # The fork's CMake compiles its libtorch-stable target with USE_CUDA (the
    # stable shim headers guard their CUDA entry points on it) and a
    # TORCH_TARGET_VERSION: the oldest torch whose stable ABI the library may
    # assume. The fork pins 2.11 (its torch); this build runs on exactly the
    # torch it compiles against, 2.10, so it targets 2.10's ABI.
    abi_defs = ["-DUSE_CUDA", "-DTORCH_TARGET_VERSION=0x020A000000000000ULL"]
    cflags = ["-O2", "-std=c++20", *abi_defs]
    # The fork's CMake (cmake/utils.cmake get_torch_gpu_compiler_flags): torch's
    # COMMON_NVCC_FLAGS, plus -DENABLE_FP8, minus the __CUDA_NO_HALF_* defines on
    # CUDA >= 12. cpp_extension adds COMMON_NVCC_FLAGS itself; -U undoes those four.
    cuda_cflags = [
        "-O3",
        "-std=c++20",
        "-DENABLE_FP8",
        *abi_defs,
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    # cache_kernels.cu uses the CUDA driver API (cuda.h) besides the runtime, so
    # the library needs libcuda at load time: link it through the toolkit's stub
    # (the fork's CMake links CUDA::cuda_driver); the real driver resolves at run time.
    ldflags = [f"-L{Path(cpp_extension.CUDA_HOME) / 'lib64' / 'stubs'}", "-lcuda"]
    started = time.time()
    cpp_extension.load(
        name=EXT_NAME,
        sources=sources,
        extra_include_paths=include_paths,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=ldflags,
        build_directory=str(build_dir),
        is_python_module=False,
        with_cuda=True,
        verbose=True,
    )
    library = build_dir / f"{EXT_NAME}.so"
    if not library.exists():
        sys.exit(f"build produced no {library}")
    # Loaded by cpp_extension.load (torch.ops.load_library): the op must exist.
    if not hasattr(torch.ops._C_cache_ops, "reshape_and_cache_flash"):
        sys.exit("the built library did not register _C_cache_ops.reshape_and_cache_flash")
    nvcc = subprocess.run(
        [str(Path(cpp_extension.CUDA_HOME) / "bin" / "nvcc"), "--version"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[-2:]
    manifest = {
        "schema": 1,
        "purpose": "GB300 vllm shim: _C_cache_ops.reshape_and_cache_flash from the pinned fork kernel",
        "fork": provenance,
        "shim": {
            "shim_bindings.cpp": sha256(shim_bindings),
            "build_shim.py": sha256(Path(__file__).resolve()),
            "vllm_package": {
                str(p.relative_to(out)): sha256(p) for p in sorted((out / "vllm").rglob("*.py"))
            } if (out / "vllm").exists() else {},
        },
        "toolchain": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_home": cpp_extension.CUDA_HOME,
            "nvcc": nvcc,
            "arch_list": args.arch,
            "cflags": cflags,
            "cuda_cflags": cuda_cflags,
            "ldflags": ldflags,
            "sources": sources,
            "include_paths": include_paths,
        },
        "library": {"path": str(library.relative_to(out)), "sha256": sha256(library), "bytes": library.stat().st_size},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.time() - started, 1),
        "host": platform.node(),
        "machine": platform.machine(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"built {library} ({manifest['library']['bytes']} bytes) in {manifest['build_seconds']} s")
    print(f"manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
