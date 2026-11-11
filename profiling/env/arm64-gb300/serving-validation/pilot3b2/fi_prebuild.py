#!/usr/bin/env python
"""CPU-only helper for prebuild_flashinfer.sh: FlashInfer's trtllm-gen FMHA JIT module.

    fi_prebuild.py build|check|load --out JSON

The module is the one `paged_attn_batch_trtllm_gen` loads for decode and
prefill (`flashinfer.decode/prefill.get_trtllm_gen_fmha_module`:
`gen_trtllm_gen_fmha_module().build_and_load()`). The caller sets
FLASHINFER_WORKSPACE_BASE (the module's final runtime location) and
FLASHINFER_CUDA_ARCH_LIST=10.3, so the spec, its build.ninja and its directory
are computed without querying a device, exactly as the GPU run computes them
with the same variables.

  build  `spec.build()`: write build.ninja and run ninja (nvcc) in place.
  check  snapshot the build directory, `spec.build()` again, snapshot again:
         the runtime path minus the load. Pass = ninja did no work (every file's
         size and mtime unchanged).
  load   `spec.load()` (tvm_ffi.load_module of the .so), as build_and_load
         does after build. The caller runs it under strace to see whether
         loading touches a GPU device node.
Each mode records torch.cuda.is_initialized() at the end (must stay False).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def snapshot(build_dir: Path) -> dict:
    out = {}
    for p in sorted(build_dir.rglob("*")):
        if p.is_file():
            st = p.stat()
            out[str(p.relative_to(build_dir))] = [st.st_size, st.st_mtime_ns]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "check", "load"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    for var in ("FLASHINFER_WORKSPACE_BASE", "FLASHINFER_CUDA_ARCH_LIST"):
        if not os.environ.get(var):
            sys.exit(f"{var} must be set")
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        sys.exit("CUDA_VISIBLE_DEVICES must be set to the empty string")
    import torch
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.attention.modules import gen_trtllm_gen_fmha_module

    spec = gen_trtllm_gen_fmha_module()
    rep = {
        "mode": a.mode,
        "workspace_dir": str(jit_env.FLASHINFER_WORKSPACE_DIR),
        "build_dir": str(spec.build_dir),
        "library": str(spec.jit_library_path),
        "is_aot": spec.is_aot,
        "cubin_dir": str(jit_env.FLASHINFER_CUBIN_DIR),
    }
    if a.mode == "build":
        spec.build(verbose=True)
    elif a.mode == "check":
        before = snapshot(spec.build_dir)
        spec.build(verbose=True)
        after = snapshot(spec.build_dir)
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        rep.update(files=len(after), changed=changed, no_work=not changed)
    else:
        mod = spec.load()
        rep["loaded"] = type(mod).__name__
    lib = spec.jit_library_path
    if lib.exists():
        rep["library_bytes"] = lib.stat().st_size
        rep["library_sha256"] = hashlib.sha256(lib.read_bytes()).hexdigest()
    ninja = spec.ninja_path
    if ninja.exists():
        rep["build_ninja_sha256"] = hashlib.sha256(ninja.read_bytes()).hexdigest()
    rep["torch_cuda_initialized"] = torch.cuda.is_initialized()
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep))
    ok = not rep["torch_cuda_initialized"] and (a.mode != "check" or rep["no_work"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
