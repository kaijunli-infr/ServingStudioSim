#!/usr/bin/env python
"""Spec files for a follow-up fill: the cells of one or more `simulator
cost-trees --kernel-configs` exports that a baseline export does not have.

    tp_new_specs.py --baseline tp1/cost_trees.json --new tp2.json tp4.json --out DIR

Writes ``DIR/<kind>.json`` as ``{"specs": [...]}`` (the union of the new
exports' feasible cells minus the baseline's, de-duplicated, sorted) and
``DIR/summary.json`` with the per-export counts. Standard library only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cells_by_kind(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for build in json.loads(path.read_text()):
        for cfg in build["kernel_configs"]["configs"]:
            grid = cfg["grid"]
            infeasible = grid.get("infeasible") or []
            bad = {json.dumps(c, sort_keys=True) for c in infeasible if isinstance(c, dict)}
            bad_idx = {c for c in infeasible if isinstance(c, int)}
            for i, cell in enumerate(grid["cells"]):
                key = json.dumps(cell, sort_keys=True)
                if key in bad or i in bad_idx:
                    continue
                out.setdefault(cfg["kind"], set()).add(key)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--new", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    base = cells_by_kind(a.baseline)
    new = {p.name: cells_by_kind(p) for p in a.new}
    a.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for kind in sorted({k for cells in new.values() for k in cells}):
        have = base.get(kind, set())
        union: set[str] = set()
        per_export = {}
        for name, cells in new.items():
            fresh = cells.get(kind, set()) - have
            per_export[name] = {"cells": len(cells.get(kind, set())), "new": len(fresh)}
            union |= fresh
        specs = [json.loads(c) for c in sorted(union)]
        summary[kind] = {"baseline": len(have), "exports": per_export, "new_specs": len(specs)}
        if specs:
            (a.out / f"{kind}.json").write_text(json.dumps({"specs": specs}, indent=1) + "\n")
        print(f"{kind:26s} baseline={len(have):4d} new_specs={len(specs):4d} {per_export}")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
