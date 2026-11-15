#!/usr/bin/env python3
"""Workload identity of the b=8 requests between two runs (CPU, read-only).

    python checks/e1_workload_identity.py --a <run dir or profile dir> --b <run dir or profile dir> [--prefix e1_b8_] [--out report.json]

Pre-run the identity is by construction (same first 8 trace rows, same request ordinals, same token pool); this check is the
post-run confirmation on the load generator's per-request records (`load_generator/replay.jsonl`: id, input_len, prompt_len,
output_len_target, arrival_time_ms, server prompt/completion token counts, finish_reason) and, when both runs captured
routes, on the per-request routed-experts arrays (`routed_experts/independent_<id>.npy`: shape must agree; element agreement
is reported as information, since expert routing of identical prompts can differ between launches by numerics).
Exit code 1 if a request is missing on one side or a compared field differs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

FIELDS_SRC = ("id", "input_len", "prompt_len", "output_len_target", "arrival_time_ms")
FIELDS_OUT = ("output_len_actual", "output_len_text_tokens", "echoed_prompt_tokens", "finish_reason")
FIELDS_USAGE = ("prompt_tokens", "completion_tokens", "cached_prompt_tokens")


def profile_dir(p: str) -> str:
    return p if os.path.isdir(os.path.join(p, "load_generator")) else os.path.join(p, "profile")


def replay_records(p: str, prefix: str) -> dict:
    out = {}
    for path in glob.glob(os.path.join(profile_dir(p), "load_generator", "replay*.jsonl")):
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            src = (r.get("source") or {}).get("data") or {}
            if str(src.get("id", "")).startswith(prefix):
                out[src["id"]] = r
    return out


def summary(r: dict) -> dict:
    src = r["source"]["data"]; o = r.get("outcome") or {}; u = o.get("server_usage") or {}
    s = {k: src.get(k) for k in FIELDS_SRC}
    s.update({k: o.get(k) for k in FIELDS_OUT}); s.update({f"usage.{k}": u.get(k) for k in FIELDS_USAGE})
    return s


def routes_compare(pa: str, pb: str, rid: str):
    try:
        import numpy as np
    except Exception:
        return None
    fa, fb = (os.path.join(profile_dir(p), "routed_experts", f"independent_{rid}.npy") for p in (pa, pb))
    if not (os.path.exists(fa) and os.path.exists(fb)):
        return None
    a, b = np.load(fa), np.load(fb)
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "agree": None}
    return {"shape": list(a.shape), "agree": float((a == b).mean())}


def run(a: str, b: str, prefix: str):
    ra, rb = replay_records(a, prefix), replay_records(b, prefix)
    res, data = [], {"requests": {}}
    ids = sorted(set(ra) | set(rb))
    res.append(("PASS" if ids and set(ra) == set(rb) else "FAIL", f"requests with prefix {prefix!r}: a={len(ra)} b={len(rb)} common={len(set(ra) & set(rb))}"))
    ndiff = 0
    for rid in ids:
        if rid not in ra or rid not in rb:
            continue
        sa, sb = summary(ra[rid]), summary(rb[rid])
        diff = {k: (sa[k], sb[k]) for k in sa if sa[k] != sb[k]}
        ndiff += bool(diff)
        data["requests"][rid] = {"a": sa, "diff": diff, "routes": routes_compare(a, b, rid)}
    res.append(("PASS" if ndiff == 0 else "FAIL", f"requests whose replay record fields differ ({', '.join(FIELDS_SRC + FIELDS_OUT)}, usage.*): {ndiff}"))
    rc = [v["routes"] for v in data["requests"].values() if v["routes"]]
    if rc:
        shapes_ok = sum(1 for r in rc if r.get("agree") is not None)
        res.append(("PASS" if shapes_ok == len(rc) else "FAIL", f"routed-experts arrays with equal shape: {shapes_ok} of {len(rc)}"))
        agree = [r["agree"] for r in rc if r.get("agree") is not None]
        if agree:
            res.append(("INFO", f"routed-experts element agreement (informational): min {min(agree):.4f} median {sorted(agree)[len(agree) // 2]:.4f}"))
    else:
        res.append(("INFO", "no routed-experts arrays on both sides (hook off or not yet run)"))
    data["checks"] = [{"status": s, "msg": m} for s, m in res]
    return res, data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--prefix", default="e1_b8_"); ap.add_argument("--out")
    x = ap.parse_args(argv)
    res, data = run(x.a, x.b, x.prefix)
    for s, m in res:
        print(f"{s:5s} {m}")
    fails = sum(1 for s, _ in res if s == "FAIL")
    if x.out:
        json.dump(data, open(x.out, "w"), indent=1, default=str)
    print("RESULT", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
