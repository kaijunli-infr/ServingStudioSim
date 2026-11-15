#!/usr/bin/env python3
"""Output-token differences between runs of the same trace (Stage 1.5 analysis item "full output-token differences").

    python analysis/e1_output_diff.py --run NAME=<profile dir> [--run NAME=<profile dir> ...] --out report.md

For every pair of runs, per request (matched by request id): output_len_actual, output_len_text_tokens, finish_reason,
server completion_tokens, chunk_count and output_preview are compared. The load generator keeps only the first 100
characters of each completion (`output_preview`) and the token counts; the full token sequence is not logged by the
pre-registered configuration, so "identical preview" is the finest available token comparison and is labelled as such.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
from collections import Counter


def load(profile_dir: str) -> dict:
    path = os.path.join(profile_dir, "load_generator", "replay.jsonl")
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        o = r["outcome"]
        out[r["source"]["data"]["id"]] = {"len": o.get("output_len_actual"), "text_tokens": o.get("output_len_text_tokens"),
                                          "finish": o.get("finish_reason"), "completion_tokens": (o.get("server_usage") or {}).get("completion_tokens"),
                                          "chunks": o.get("chunk_count"), "preview": o.get("output_preview"), "status": o.get("status"),
                                          "dur_ms": o.get("total_duration_ms")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=profile_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {}
    for spec in a.run:
        name, path = spec.split("=", 1)
        runs[name] = load(path)
    lines = ["# Output differences between runs (same trace, greedy, temperature 0)", "",
             "Finest available comparison: `output_preview` (first 100 characters) plus token counts; full token sequences are not logged "
             "by the pre-registered load-generator configuration.", "",
             "| run | requests | all SUCCESS | output_len values | finish_reason | chunk_count values | median duration ms |", "|---|---|---|---|---|---|---|"]
    rep = {"runs": {}, "pairs": {}}
    for n, R in runs.items():
        lens = Counter(v["len"] for v in R.values()); fin = Counter(v["finish"] for v in R.values()); ch = Counter(v["chunks"] for v in R.values())
        durs = sorted(v["dur_ms"] for v in R.values() if v["dur_ms"] is not None)
        ok = all(v["status"] == "SUCCESS" for v in R.values())
        rep["runs"][n] = {"requests": len(R), "all_success": ok, "len": dict(lens), "finish": dict(fin), "chunks": dict(ch)}
        lines.append(f"| {n} | {len(R)} | {ok} | {dict(lens)} | {dict(fin)} | {dict(ch)} | {durs[len(durs) // 2]:.1f} |")
    lines += ["", "| pair | paired | preview identical | preview differs | len differs | completion_tokens differs | chunk_count differs | differing requests (first 6) |", "|---|---|---|---|---|---|---|---|"]
    for x, y in itertools.combinations(runs, 2):
        A, B = runs[x], runs[y]
        common = sorted(set(A) & set(B))
        same = [r for r in common if A[r]["preview"] == B[r]["preview"]]
        diff = [r for r in common if A[r]["preview"] != B[r]["preview"]]
        dlen = [r for r in common if A[r]["len"] != B[r]["len"]]
        dct = [r for r in common if A[r]["completion_tokens"] != B[r]["completion_tokens"]]
        dch = [r for r in common if A[r]["chunks"] != B[r]["chunks"]]
        rep["pairs"][f"{x}/{y}"] = {"paired": len(common), "preview_identical": len(same), "preview_differs": diff, "len_differs": dlen, "completion_tokens_differs": dct, "chunk_differs": dch}
        lines.append(f"| {x} vs {y} | {len(common)} | {len(same)} | {len(diff)} | {len(dlen)} | {len(dct)} | {len(dch)} | {', '.join(diff[:6])} |")
    # per-cell breakdown of preview identity
    lines += ["", "## Preview identity per cell", "", "| pair | cell | identical / paired |", "|---|---|---|"]
    for x, y in itertools.combinations(runs, 2):
        A, B = runs[x], runs[y]
        cells = Counter(); same = Counter()
        for r in set(A) & set(B):
            cell = r.rsplit("_r", 1)[0]
            cells[cell] += 1; same[cell] += A[r]["preview"] == B[r]["preview"]
        for cell in sorted(cells):
            lines.append(f"| {x} vs {y} | {cell} | {same[cell]} / {cells[cell]} |")
    text = "\n".join(lines) + "\n"
    open(a.out, "w").write(text)
    json.dump(rep, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(text)


if __name__ == "__main__":
    main()
