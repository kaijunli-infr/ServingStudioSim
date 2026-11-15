#!/usr/bin/env python3
"""Compare FlashInfer autotune cache files (vLLM `autotune_configs.json`) saved by different runs (CPU, read-only).

    python analysis/autotune_cache_diff.py <file_or_dir A> <file_or_dir B> [--label-a X --label-b Y] [--out report.md]

Each file maps a cache key "('<op>', '<runner>', <input shapes>, <extras>)" to [runner, tactic]. The report counts,
per (op, runner), the keys present and the keys whose winning tactic differs between the two files: two launches of
the same recipe that re-tune pick different kernels for those shapes.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

KEY = re.compile(r"^\('([^']+)', '([^']+)'")


def load(path: str) -> dict:
    if os.path.isdir(path):
        path = os.path.join(path, "autotune_configs.json")
    return json.load(open(path))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--out")
    x = ap.parse_args(argv)
    A, B = load(x.a), load(x.b)
    meta_a, meta_b = A.get("_metadata"), B.get("_metadata")
    keys_a = {k for k in A if not k.startswith("_")}
    keys_b = {k for k in B if not k.startswith("_")}
    ops, diff, examples = collections.Counter(), collections.Counter(), {}
    for k in sorted(keys_a & keys_b):
        m = KEY.match(k)
        op = (m.group(1), m.group(2)) if m else ("?", "?")
        ops[op] += 1
        if A[k] != B[k]:
            diff[op] += 1
            examples.setdefault(op, (k, A[k], B[k]))
    lines = [f"# FlashInfer autotune cache diff: {x.label_a} vs {x.label_b}", "",
             f"{x.label_a}: `{x.a}`", f"{x.label_b}: `{x.b}`", "",
             f"metadata identical: {meta_a == meta_b}; keys {len(keys_a)} / {len(keys_b)}; common {len(keys_a & keys_b)}; only {x.label_a} {len(keys_a - keys_b)}; only {x.label_b} {len(keys_b - keys_a)}", "",
             "| op | runner | common keys | differing tactics | example key (truncated) | tactic A | tactic B |", "|---|---|---|---|---|---|---|"]
    for op in sorted(ops):
        ex = examples.get(op)
        lines.append(f"| {op[0]} | {op[1]} | {ops[op]} | {diff[op]} | `{ex[0][:70] if ex else ''}` | {ex[1] if ex else ''} | {ex[2] if ex else ''} |")
    total = sum(ops.values()); nd = sum(diff.values())
    lines += ["", f"**{nd} of {total} common tactic entries differ ({100.0 * nd / max(total, 1):.0f} %).**", ""]
    if meta_a:
        lines += ["metadata A:", "```", json.dumps(meta_a, indent=1), "```"]
    text = "\n".join(lines) + "\n"
    print(text)
    if x.out:
        open(x.out, "w").write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
