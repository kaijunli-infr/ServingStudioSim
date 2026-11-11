#!/usr/bin/env python
"""Query the public API locally over a profile DB and report what it says about
GB300 (Phase 2C validation step 4). CPU only; nothing is deployed.

    public_api_check.py --db $VIBESIM_PROFILE_DB --out report.json

Builds the service's objects the way `python -m public_api serve` does
(`Sources` over the DB, `DeploymentIndex.build` over the public presets, the
`KernelLibrary`) and drives the FastAPI app with a test client:
`/kernels` (the GPU list with spec-sheet peaks and per-GPU row counts),
`/kernels/<kind>/rows?gpu=NVIDIA GB300` for the Llama TP1 kinds and `all_reduce` (row counts
and the backends present), and `/kernels/kv_cache_append/configs`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PREFIX = "/api/public/v1"
GPU = "NVIDIA GB300"
KINDS = (
    "single_gemm",
    "rms_norm",
    "elementwise",
    "flashinfer_attn_prefill",
    "flashinfer_attn_decode",
    "kv_cache_append",
    "all_reduce",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--build-type", default="release")
    args = ap.parse_args()
    os.environ["VIBESIM_PROFILE_DB"] = str(args.db.resolve())
    from fastapi.testclient import TestClient

    from profiling.gpu_policy import disable_gpus
    from public_api.app import create_app
    from public_api.deployments import DeploymentIndex
    from public_api.kernels import KernelLibrary, git_commit, resolve_gpu_spec
    from public_api.sources import Sources

    disable_gpus()
    t0 = time.time()
    sources = Sources(db_path=args.db, build_type=args.build_type)
    index = DeploymentIndex.build(
        sources.cost_trees, sources.list_params(), sim_commit=git_commit()
    )
    with tempfile.TemporaryDirectory() as runs:
        client = TestClient(create_app(KernelLibrary(sources, index), Path(runs)))
        report: dict = {"db": str(args.db.resolve()), "index_seconds": round(time.time() - t0, 1)}
        report["health"] = client.get(f"{PREFIX}/health").json()
        catalog = client.get(f"{PREFIX}/kernels").json()
        gpus = {g["name"]: g for g in catalog["gpus"]}
        report["gpus"] = {
            name: {"rows": g["rows"], "peaks": g["peaks"]} for name, g in gpus.items()
        }
        spec = resolve_gpu_spec(GPU)
        report["gb300_spec_resolution"] = (
            None
            if spec is None
            else {
                k: getattr(spec, k)
                for k in dir(spec)
                if not k.startswith("_") and not callable(getattr(spec, k))
            }
        )
        report["gb300_in_catalog"] = GPU in gpus
        report["kinds"] = {}
        for kind in KINDS:
            doc = client.get(f"{PREFIX}/kernels/{kind}/rows", params={"gpu": GPU}).json()
            cols = doc["columns"]
            backend_col = cols.index("backend") if "backend" in cols else None
            backends = (
                sorted({row[backend_col] for row in doc["rows"]}) if backend_col is not None else []
            )
            entry = {"rows": len(doc["rows"]), "backends": backends}
            # coverage per GPU/backend/precision from the catalog entry of this kind
            for k in catalog["kernels"]:
                if k["kind"] == kind:
                    entry["coverage"] = k.get("coverage")
            report["kinds"][kind] = entry
        kv = client.get(
            f"{PREFIX}/kernels/kv_cache_append/rows", params={"gpu": GPU, "backend": "vllm_cuda"}
        ).json()
        tcol = kv["columns"].index("time_ms") if "time_ms" in kv["columns"] else None
        ncol = kv["columns"].index("num_tokens") if "num_tokens" in kv["columns"] else None
        if tcol is not None and ncol is not None and kv["rows"]:
            by_tokens = sorted((row[ncol], row[tcol]) for row in kv["rows"])
            report["kv_cache_append_gb300"] = {
                "rows": len(kv["rows"]),
                "num_tokens_min_max": [by_tokens[0][0], by_tokens[-1][0]],
                "time_us_at_min_max": [by_tokens[0][1] * 1e3, by_tokens[-1][1] * 1e3],
                "provenance": kv.get("provenance"),
            }
        configs = client.get(f"{PREFIX}/kernels/kv_cache_append/configs").json()
        report["kv_cache_append_public_configs_gb300"] = [
            c for c in configs.get("configs", []) if c.get("gpu") == GPU
        ]
        report["seconds"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("health", "gb300_in_catalog", "index_seconds", "seconds")}
        )
    )
    print("gpus:", {n: g["rows"] for n, g in report["gpus"].items()})
    print("GB300 peaks:", json.dumps(report["gpus"].get(GPU, {}).get("peaks")))
    for kind, e in report["kinds"].items():
        print(f"  {kind:26s} gb300 rows={e['rows']} backends={e['backends']}")
    print(
        "kv_cache_append GB300:", json.dumps(report.get("kv_cache_append_gb300"), default=str)[:400]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
