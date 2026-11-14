#!/usr/bin/env python3
"""E1 Stage 1: per-cell pure-decode outlier scan and count of tokens emitted in non-pure iterations (CPU, read-only).

    python analysis/e1_stage1_outliers.py <run dir with e1_dry_off/ and e1_dry_on/>
"""
import glob
import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np

TAG = re.compile(r"e1_b(\d+)_kv(\d+)k_w(\d+)_r(\d+)")
R = sys.argv[1]
for side in ("off", "on"):
    M = glob.glob(f"{R}/e1_dry_{side}/profile/vllm/*_metrics.jsonl")[0]
    recs = [json.loads(l) for l in open(M)]
    cells = defaultdict(list)
    mixed_tok, pure_tok = Counter(), Counter()
    for d in recs:
        prog = d.get("decode_request_progress") or []
        tags = [TAG.search(p["request_id"]) for p in prog]
        tags = [t.groups() for t in tags if t]
        b = int(tags[0][0]) if tags else 0
        pure = bool(tags) and d["prefill_tokens"] == 0 and d["decode_requests"] == b
        if pure:
            cells[b].append((d["iteration_index"], d["observed_elapsed_ms"]))
        for p in prog:
            if p["emitted_tokens"] >= 1 and 1 <= p["output_tokens_before"] <= 1023:
                (pure_tok if pure else mixed_tok)[p["request_id"]] += p["emitted_tokens"]
    print(f"== {side}: iterations {recs[0]['iteration_index']}..{recs[-1]['iteration_index']} n={len(recs)}")
    for b, v in sorted(cells.items()):
        a = np.array([x[1] for x in v])
        idx = [x[0] for x in v]
        med = np.median(a)
        out = [(idx[i], round(float(a[i]), 2)) for i in range(len(a)) if a[i] > 1.5 * med]
        print(f"  b={b}: pure n={len(a)} iters {idx[0]}..{idx[-1]} median={med:.3f} max={a.max():.2f} >1.5x-median={out[:6]} (total {len(out)})")
    reqs = sorted(set(pure_tok) | set(mixed_tok))
    mt = [mixed_tok[r] for r in reqs]
    print(f"  tokens (positions 1..1023) emitted in non-pure iterations per request: min {min(mt)} median {int(np.median(mt))} max {max(mt)}; requests with 0 such tokens: {sum(1 for x in mt if x == 0)} of {len(reqs)}")
    big = [(d["iteration_index"], round(d["observed_elapsed_ms"], 1), d["prefill_tokens"], d["decode_requests"]) for d in recs if d["observed_elapsed_ms"] > 250]
    print("  iterations >250 ms:", big[:8], "total", len(big))
