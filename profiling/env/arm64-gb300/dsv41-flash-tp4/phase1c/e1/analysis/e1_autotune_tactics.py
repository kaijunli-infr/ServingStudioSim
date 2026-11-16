#!/usr/bin/env python3
"""C3: can FlashInfer autotune tactic differences account for the Stage 1.5 b=48 timing shift? Existing data only (CPU).

    python e1_autotune_tactics.py --cache <label>=<autotune_configs.json> ... --kernels <label>=<profile dir>:<tables dir>:<b> ... --out <dir>

Part 1: pairwise tactic agreement between the per-launch autotune caches (same key = same op + shape signature; value = chosen
tactic). Part 2: per kernel-family GPU time per pure-decode iteration (4 ranks) from the nsys kernel tables of the passes that
have one, split into tactic-dependent families (FlashInfer autotuned: trtllm MoE bmm, dense block-scaled GEMM, quantize) and
the rest, so the time that tactic choice can move is bounded. Nothing is fitted.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re

import numpy as np
import pyarrow.parquet as pq

FAMILIES = [("moe_bmm", r"^bmm_"), ("dense_gemm", r"flashinfergemmkernelsdense|cutlass_device_kernel_tensorptr|cute_dsl_ll_bf16|cute_dsl.*gemm|deep_gemm"),
            ("quantize", r"quantiz"), ("moe_routing", r"routing|topk|_dsv4_topk|trtllm.*moe|moe::"), ("attention", r"fmha|mqa|indexer|sparse|paged"),
            ("collective", r"all_reduce|allreduce|AllGather|nccl|ncclDev|mhc"), ("other", r".")]


def family(name):
    for f, rx in FAMILIES:
        if re.search(rx, name, re.I):
            return f
    return "other"


def load_cache(path):
    d = json.load(open(path))
    meta = d.pop("_metadata", {})
    out = {}
    for k, v in d.items():
        try:
            key = ast.literal_eval(k)
            op = key[0]
        except Exception:
            op = k.split(",")[0].strip("(' ")
        out[k] = (op, json.dumps(v))
    return meta, out


def part1(caches):
    labels = list(caches)
    rows = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            ka, kb = caches[a], caches[b]
            common = set(ka) & set(kb)
            diff = [k for k in common if ka[k][1] != kb[k][1]]
            byop = {}
            for k in common:
                op = ka[k][0]; byop.setdefault(op, [0, 0]); byop[op][0] += 1; byop[op][1] += ka[k][1] != kb[k][1]
            rows.append({"a": a, "b": b, "common_keys": len(common), "only_a": len(set(ka) - set(kb)), "only_b": len(set(kb) - set(ka)), "differing": len(diff),
                         "differing_frac": round(len(diff) / max(len(common), 1), 3), "by_op": {op: f"{d}/{n}" for op, (n, d) in sorted(byop.items())}})
    return rows


def part2(label, profile, tables, b):
    it = pq.read_table(os.path.join(tables, "iterations.parquet")).to_pydict()
    its = {i for s, p, d, e, i in zip(it["stage"], it["prefill_tokens"], it["decode_requests"], it["excluded"], it["iteration"]) if s == "decode" and p == 0 and d == b and not e}
    pj = json.load(open(os.path.join(profile, "parsed.json")))
    names = pj.get("kernel_names") or pj.get("names")
    det = pj["iteration_details"]
    di2it = {n: d["iteration"] for n, d in enumerate(det)}
    kt = pq.read_table(os.path.join(profile, "parsed.kernels.parquet"), columns=["detail_index", "name_id", "start_ns", "end_ns"])
    di = kt["detail_index"].to_numpy(); nid = kt["name_id"].to_numpy(); dur = (kt["end_ns"].to_numpy().astype(np.int64) - kt["start_ns"].to_numpy().astype(np.int64))
    it_of = np.array([di2it.get(int(x), -1) for x in np.unique(di)]); lut = dict(zip(np.unique(di).tolist(), it_of.tolist()))
    sel = np.array([lut[int(x)] in its for x in di])
    nid = nid[sel]; dur = dur[sel]; n = len(its)
    if isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
    else:
        names = {i: v for i, v in enumerate(names)}
    tot = np.bincount(nid, weights=dur, minlength=max(names) + 1); cnt = np.bincount(nid, minlength=max(names) + 1)
    fam_t = {}; fam_c = {}; per = []
    for i in range(len(tot)):
        if cnt[i] == 0:
            continue
        f = family(names[i]); fam_t[f] = fam_t.get(f, 0) + tot[i]; fam_c[f] = fam_c.get(f, 0) + cnt[i]
        per.append((names[i], f, cnt[i] / n, tot[i] / n / 1e6))
    per.sort(key=lambda x: -x[3])
    busy = sum(fam_t.values()) / n / 1e6
    return {"label": label, "b": b, "n_iterations": n, "gpu_time_ms_per_iteration_4ranks": round(busy, 4),
            "families_ms_per_iteration": {f: round(fam_t[f] / n / 1e6, 4) for f in sorted(fam_t)}, "families_count_per_iteration": {f: round(fam_c[f] / n, 1) for f in sorted(fam_c)},
            "tactic_dependent_ms": round(sum(fam_t.get(f, 0) for f in ("moe_bmm", "dense_gemm", "quantize")) / n / 1e6, 4),
            "top": [{"kernel": k[:100], "family": f, "count_per_iter": round(c, 1), "ms_per_iter": round(t, 4)} for k, f, c, t in per[:40]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="append", default=[]); ap.add_argument("--kernels", action="append", default=[]); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    caches = {}; metas = {}
    for spec in a.cache:
        lab, path = spec.split("=", 1); metas[lab], caches[lab] = load_cache(path)
    p1 = part1(caches)
    p2 = []
    for spec in a.kernels:
        lab, rest = spec.split("=", 1); prof, tab, b = rest.rsplit(":", 2); p2.append(part2(lab, prof, tab, int(b)))
    json.dump({"caches": {k: {"keys": len(v), "meta": metas[k]} for k, v in caches.items()}, "pairwise": p1, "kernel_families": p2}, open(os.path.join(a.out, "autotune_tactics.json"), "w"), indent=1)
    L = ["# Autotune tactic choices across launches and tactic-dependent GPU time (C3)", "", "## 1. Pairwise agreement of the per-launch autotune caches", "",
         "| launch A | launch B | common keys | differing | fraction | only A / only B | differing by op (diff/common) |", "|---|---|---|---|---|---|---|"]
    for r in p1:
        L.append(f"| {r['a']} | {r['b']} | {r['common_keys']} | {r['differing']} | {r['differing_frac']} | {r['only_a']} / {r['only_b']} | " + "; ".join(f"{op.split('::')[-1][:40]} {v}" for op, v in r["by_op"].items()) + " |")
    L += ["", "## 2. GPU time per pure-decode iteration by kernel family (4 ranks, ms)", "", "| pass | b | n | total | moe_bmm | dense_gemm | quantize | moe_routing | attention | collective | other | tactic-dependent (bmm+gemm+quant) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in p2:
        f = r["families_ms_per_iteration"]; g = lambda k: f.get(k, 0.0)
        L.append(f"| {r['label']} | {r['b']} | {r['n_iterations']} | {r['gpu_time_ms_per_iteration_4ranks']} | {g('moe_bmm')} | {g('dense_gemm')} | {g('quantize')} | {g('moe_routing')} | {g('attention')} | {g('collective')} | {g('other')} | {r['tactic_dependent_ms']} |")
    for r in p2:
        L += ["", f"### top kernels `{r['label']}` b={r['b']} (count/iter over 4 ranks, ms/iter)", "", "| kernel | family | count/iter | ms/iter |", "|---|---|---|---|"]
        L += [f"| `{t['kernel']}` | {t['family']} | {t['count_per_iter']} | {t['ms_per_iter']} |" for t in r["top"][:30]]
    open(os.path.join(a.out, "AUTOTUNE-TACTICS.md"), "w").write("\n".join(L) + "\n")
    for r in p1:
        print(f"{r['a']} vs {r['b']}: {r['differing']}/{r['common_keys']} differ")
    for r in p2:
        print(f"{r['label']} b={r['b']}: total {r['gpu_time_ms_per_iteration_4ranks']} ms, tactic-dependent {r['tactic_dependent_ms']} ms")


if __name__ == "__main__":
    main()
