"""Turn the simulator's exported kernel configs (``simulator cost-trees
--kernel-configs``) into one profiler spec file per kernel kind.

    tp1_specs.py cost_trees.json OUT_DIR

Writes ``OUT_DIR/<kind>.json`` as ``{"specs": [...]}`` with the feasible grid
cells, de-duplicated across configs of the same kind, and ``OUT_DIR/summary.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Backends this campaign fills per kind: the Llama arch's host-runnable ones.
BACKENDS = {
    "single_gemm": ["torch", "torch_linear"],
    "rms_norm": ["flashinfer"],
    "elementwise": ["triton"],
    "flashinfer_attn_prefill": ["fa2"],
    "flashinfer_attn_decode": ["fa2"],
    # Phase 2C: the vLLM KV-cache append rows, through the vllm shim.
    "kv_cache_append": ["vllm_cuda"],
}


def main(cost_trees: Path, out_dir: Path) -> int:
    doc = json.loads(cost_trees.read_text())
    configs = doc[0]["kernel_configs"]["configs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    by_kind: dict[str, list[dict]] = {}
    seen: dict[str, set[str]] = {}
    summary: dict[str, dict] = {}
    for cfg in configs:
        kind = cfg["kind"]
        grid = cfg["grid"]
        cells = grid["cells"]
        infeasible = grid.get("infeasible") or []
        bad = {json.dumps(c, sort_keys=True) for c in infeasible if isinstance(c, dict)}
        bad_idx = {c for c in infeasible if isinstance(c, int)}
        kept = 0
        for i, cell in enumerate(cells):
            key = json.dumps(cell, sort_keys=True)
            if key in bad or i in bad_idx:
                continue
            if key in seen.setdefault(kind, set()):
                continue
            seen[kind].add(key)
            by_kind.setdefault(kind, []).append(cell)
            kept += 1
        s = summary.setdefault(kind, {"configs": 0, "cells": 0, "infeasible": 0, "specs": 0})
        s["configs"] += 1
        s["cells"] += len(cells)
        s["infeasible"] += len(infeasible)
    for kind, specs in by_kind.items():
        summary[kind]["specs"] = len(specs)
        summary[kind]["backends"] = BACKENDS.get(kind, [])
        summary[kind]["rows"] = len(specs) * len(BACKENDS.get(kind, []))
        (out_dir / f"{kind}.json").write_text(json.dumps({"specs": specs}, indent=1) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for kind, s in summary.items():
        print(f"{kind:26s} configs={s['configs']} cells={s['cells']} infeasible={s['infeasible']} specs={s['specs']} backends={s.get('backends')} rows={s.get('rows')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
