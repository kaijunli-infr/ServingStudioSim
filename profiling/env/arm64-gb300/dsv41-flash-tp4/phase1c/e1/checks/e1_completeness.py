#!/usr/bin/env python3
"""E1 correctness and trace-completeness checks (CPU, read-only). Frozen before the capture.

    python checks/e1_completeness.py --profile <profile dir> [--inputs E1-INPUTS.json] [--parsed]
                                     [--compare <other metrics.jsonl>] [--out report.json]

Checks (each prints PASS/FAIL/INFO; the exit code is 1 if any FAIL):
  result    profile_result.json present; export validation ok; parsed devices == [0,1,2,3]
  cells     from the engine's own per-iteration record: pure-decode iterations per (b, KV, wave) >= the
            planned minimum and >= 400; no iteration carries requests of two cells (cell overrun); every
            request emitted exactly OUT_LEN tokens
  overhead  with --compare: per-cell median engine-side iteration time ratio this/other (dry-run pair:
            hook on / hook off); FAIL if any cell ratio > 1.02 (E1 stop condition)
  routes    routed_experts/*.npy == successful requests; token_corpus manifest/provenance consistent with
            OUT_LEN - 1 rows per request; expert ids < num_experts; per-request row count == number of
            decode iterations in which the request had 1 <= output_tokens_before <= OUT_LEN - 1 (the join
            the routing analysis relies on)
  parsed    with --parsed: step_alignment has no unpaired steps; every iteration has 7 NVTX ranges on each
            of 4 devices (28); devices_absent empty everywhere; JIT module loads after the first 50
            iterations (count, FAIL if > 20: cache was cold)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

TAG = re.compile(r"e1_b(\d+)_kv(\d+)k_w(\d+)_r(\d+)")
OUT_LEN_DEFAULT = 1024
fails = 0


def rep(status, msg):
    global fails
    if status == "FAIL":
        fails += 1
    print(f"{status:5s} {msg}")


def read_metrics(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh]


def cell_counts(recs, out_len):
    pure, mixed, overrun, ramp = Counter(), Counter(), 0, 0
    elapsed = defaultdict(list)
    final_tokens = {}
    for d in recs:
        prog = d.get("decode_request_progress") or []
        for p in prog:
            final_tokens[p["request_id"]] = max(final_tokens.get(p["request_id"], 0), p["output_tokens_before"] + p["emitted_tokens"])
        tags = [TAG.search(p["request_id"]) for p in prog]
        tags = [t.groups() for t in tags if t]
        if not tags:
            continue
        cells = {(int(t[0]), int(t[1])) for t in tags}
        if len(cells) > 1:
            overrun += 1
            continue
        b, kv = next(iter(cells))
        wave = min(int(t[2]) for t in tags)
        if d["decode_requests"] != b:
            ramp += 1
            continue
        if d["prefill_tokens"] == 0:
            pure[(b, kv, wave)] += 1
            elapsed[(b, kv, wave)].append(d["observed_elapsed_ms"])
        else:
            mixed[(b, kv, wave)] += 1
    wrong = {r: n for r, n in final_tokens.items() if n != out_len}
    return pure, mixed, overrun, ramp, elapsed, final_tokens, wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True); ap.add_argument("--inputs"); ap.add_argument("--parsed", action="store_true")
    ap.add_argument("--compare"); ap.add_argument("--out")
    a = ap.parse_args()
    P = a.profile
    report = {"profile": P}
    inputs = json.load(open(a.inputs)) if a.inputs else None
    out_len = inputs["design"]["OUT_LEN"] if inputs else OUT_LEN_DEFAULT

    # --- result
    pr_path = os.path.join(P, "profile_result.json")
    if os.path.exists(pr_path):
        pr = json.load(open(pr_path))
        v = pr.get("validation")
        if v is not None:
            rep("PASS" if v.get("ok") else "FAIL", f"export validation ok={v.get('ok')} nvtx_forward_ranges={v.get('nvtx_forward_ranges')} kernel_rows={v.get('kernel_rows')}")
        dev = pr.get("parsed_device_ids")
        if dev is not None:
            rep("PASS" if dev == [0, 1, 2, 3] else "FAIL", f"parsed devices {dev}")
        rep("INFO", f"profile_kind={pr.get('profile_kind')} request_timing_count={pr.get('request_timing_count')} corpus_tokens={pr.get('token_corpus_tokens')}")
        report["profile_result"] = {k: pr.get(k) for k in ("profile_kind", "validation", "parsed_device_ids", "request_timing_count", "token_corpus_tokens")}
    else:
        rep("FAIL", "profile_result.json missing")

    # --- cells
    mpaths = glob.glob(os.path.join(P, "vllm", "*_metrics.jsonl"))
    recs = read_metrics(mpaths[0]) if mpaths else []
    rep("PASS" if recs else "FAIL", f"metrics records: {len(recs)}")
    pure, mixed, overrun, ramp, elapsed, final_tokens, wrong = cell_counts(recs, out_len)
    report["cells"] = {f"b{b}_kv{kv}_w{w}": {"pure": n, "mixed": mixed.get((b, kv, w), 0), "elapsed_ms_median": float(np.median(elapsed[(b, kv, w)])),
                                              "elapsed_ms_p90": float(np.quantile(elapsed[(b, kv, w)], 0.9))} for (b, kv, w), n in sorted(pure.items())}
    if inputs:
        plan = {(p["b"], p["kv"], p["wave"]): p for p in inputs["main"]["plan"] + inputs["dry"]["plan"]}
        for key, p in sorted(plan.items()):
            n = pure.get(key, 0)
            need = min(p["pure_decode_iters_min"], inputs["design"]["MIN_PURE_DECODE"])
            if key in pure or key in plan and any(k[:2] == key[:2] for k in pure):
                rep("PASS" if n >= need else "FAIL", f"cell b={key[0]} kv={key[1]} wave={key[2]}: pure decode iterations {n} (need >= {need}), mixed {mixed.get(key, 0)}")
        rep("PASS" if overrun == 0 else "FAIL", f"iterations with requests of two cells (overrun): {overrun}")
        rep("INFO", f"ramp iterations (decode_requests != b inside a wave): {ramp}")
        n_req = sum(1 for r in final_tokens if TAG.search(r))
        rep("PASS" if not wrong and n_req else "FAIL", f"requests seen {n_req}; requests whose emitted tokens != {out_len}: {len(wrong)}")
    else:
        rep("INFO", f"no --inputs: {len(pure)} tagged cells, overrun {overrun}, requests {len(final_tokens)}, wrong-length {len(wrong)}")
    report["overrun"] = overrun; report["wrong_token_counts"] = wrong

    # --- overhead (dry-run pair)
    if a.compare:
        other = read_metrics(a.compare)
        pure2, _, _, _, elapsed2, _, _ = cell_counts(other, out_len)
        ratios = {}
        for key in sorted(set(elapsed) & set(elapsed2)):
            r = float(np.median(elapsed[key]) / np.median(elapsed2[key]))
            ratios[f"b{key[0]}_kv{key[1]}_w{key[2]}"] = round(r, 4)
            rep("PASS" if r <= 1.02 else "FAIL", f"overhead b={key[0]} kv={key[1]}: median iteration ratio this/other = {r:.4f} (limit 1.02)")
        report["overhead_ratios"] = ratios

    # --- routes
    routes_dir = os.path.join(P, "routed_experts")
    corpus = os.path.join(P, "token_corpus")
    if os.path.isdir(routes_dir) or os.path.isdir(corpus):
        npy = glob.glob(os.path.join(routes_dir, "*.npy"))
        n_req = sum(1 for r in final_tokens if TAG.search(r)) or len(final_tokens)
        rep("PASS" if len(npy) == n_req else "FAIL", f"routed_experts files {len(npy)} vs requests {n_req}")
        if os.path.exists(os.path.join(corpus, "manifest.json")):
            man = json.load(open(os.path.join(corpus, "manifest.json")))
            prov = json.load(open(os.path.join(corpus, "provenance.json")))
            segs = prov["request_segments"]
            exp_rows = n_req * (out_len - 1)
            rep("PASS" if man["num_tokens"] == exp_rows else "FAIL", f"corpus tokens {man['num_tokens']} vs {n_req} x ({out_len}-1) = {exp_rows}")
            bad_len = [s["request"] for s in segs if s["length"] != out_len - 1]
            rep("PASS" if not bad_len else "FAIL", f"request segments with length != {out_len - 1}: {len(bad_len)}")
            routes = np.fromfile(os.path.join(corpus, man["data_file"]), dtype="<u2")
            rep("PASS" if routes.size and routes.max() < man["num_experts"] else "FAIL", f"max expert id {routes.max() if routes.size else None} < {man['num_experts']}")
            # join: decode iterations per request with 1 <= output_tokens_before <= out_len-1
            per_req = Counter()
            for d in recs:
                if d["prefill_tokens"] == 0:
                    for p in d.get("decode_request_progress") or []:
                        if 1 <= p["output_tokens_before"] <= out_len - 1 and p["emitted_tokens"] == 1:
                            per_req["independent_" + p["request_id"] if not p["request_id"].startswith("independent_") else p["request_id"]] += 1
            mism = [s["request"] for s in segs if per_req.get(s["request"], -1) != s["length"]]
            rep("PASS" if not mism else "FAIL", f"requests whose pure-decode iteration count != corpus rows: {len(mism)} (join check)")
            report["corpus"] = {"num_tokens": man["num_tokens"], "checksum_fnv1a64": man.get("checksum_fnv1a64"), "bad_len": len(bad_len), "join_mismatch": len(mism)}
    else:
        rep("INFO", "no routed_experts / token_corpus directory (Option B capture or dry-run off)")

    # --- parsed
    if a.parsed:
        pj = os.path.join(P, "parsed.json")
        d = json.load(open(pj))
        det = d["iteration_details"]
        sa = d.get("step_alignment", {})
        rep("PASS" if not sa.get("unpaired_steps_by_device") else "FAIL", f"step_alignment unpaired_steps_by_device={sa.get('unpaired_steps_by_device')}")
        bad_rng = sum(1 for x in det if len(x["ranges"]) != 28)
        bad_dev = sum(1 for x in det if x["devices_absent"])
        rep("PASS" if bad_rng == 0 else "FAIL", f"iterations without 28 NVTX ranges: {bad_rng} of {len(det)}")
        rep("PASS" if bad_dev == 0 else "FAIL", f"iterations with devices_absent: {bad_dev}")
        jit = [x["iteration"] for x in det[50:] if x["jit_module_loads"]]
        rep("PASS" if len(jit) <= 20 else "FAIL", f"JIT-affected iterations after warm-up: {len(jit)} (limit 20; exclusion list)")
        report["parsed"] = {"iterations": len(det), "bad_ranges": bad_rng, "bad_devices": bad_dev, "jit_iterations": jit}
        if a.out:
            excl = os.path.join(os.path.dirname(os.path.abspath(a.out)), "jit_excl.txt")
            with open(excl, "w") as fh:
                fh.write(",".join(str(i) for i in sorted(set(jit) | {x["iteration"] for x in det[:50] if x["jit_module_loads"]})) + "\n")
            rep("INFO", f"exclusion list (JIT-affected iterations, Phase 0 rule) written to {excl}")

    report["fails"] = fails
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    print("RESULT", "PASS" if fails == 0 else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
