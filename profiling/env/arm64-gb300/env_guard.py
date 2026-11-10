"""Fail-closed check of the GB300 profiling environment (CPU-only).

    uv run --no-sync python profiling/env/arm64-gb300/env_guard.py --out guard.json
    ... env_guard.py --out guard-post.json --same-as guard.json

Exits non-zero unless this interpreter is the GB300 venv with the expected
torch build (CUDA 13.0, NCCL backend compiled in and linked to the expected
NCCL) and `UV_NO_SYNC=1` is set. It repairs nothing: a mismatch means stop and
run bootstrap.sh by hand (see README, "Environment incident, 2026-11-10").
`--same-as` additionally requires the installed-package fingerprint to equal an
earlier guard's, which shows that nothing synced the venv in between. CUDA is
never initialised.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path

EXPECT = {"torch": "2.10.0+cu130", "cuda": "13.0", "nccl": [2, 28, 9]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--same-as")
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions())
    nccl = list(torch.cuda.nccl.version()) if dist.is_nccl_available() else None
    report = {
        "python": sys.executable,
        "prefix": sys.prefix,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl_available": dist.is_nccl_available(),
        "nccl": nccl,
        "nccl_dist": next((p for p in packages if p.startswith("nvidia-nccl")), None),
        "uv_no_sync": os.environ.get("UV_NO_SYNC"),
        "uv_project_environment": os.environ.get("UV_PROJECT_ENVIRONMENT"),
        "package_count": len(packages),
        "packages_sha256": hashlib.sha256("\n".join(packages).encode()).hexdigest(),
    }
    problems = []
    if report["torch"] != EXPECT["torch"]:
        problems.append(f"torch is {report['torch']}, expected {EXPECT['torch']}")
    if report["cuda"] != EXPECT["cuda"]:
        problems.append(f"torch CUDA is {report['cuda']}, expected {EXPECT['cuda']}")
    if not report["nccl_available"]:
        problems.append("torch.distributed.is_nccl_available() is False")
    elif nccl != EXPECT["nccl"]:
        problems.append(f"NCCL is {nccl}, expected {EXPECT['nccl']}")
    if report["uv_no_sync"] != "1":
        problems.append("UV_NO_SYNC is not 1")
    venv = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if not venv or Path(sys.prefix).resolve() != Path(venv).resolve():
        problems.append(f"interpreter prefix {sys.prefix} is not UV_PROJECT_ENVIRONMENT {venv}")
    if args.same_as:
        before = json.loads(Path(args.same_as).read_text())
        if before["packages_sha256"] != report["packages_sha256"]:
            problems.append(f"installed packages changed since {args.same_as}")
    report["problems"] = problems
    report["pass"] = not problems
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
