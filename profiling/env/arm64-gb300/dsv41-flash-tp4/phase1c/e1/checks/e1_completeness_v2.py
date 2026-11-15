#!/usr/bin/env python3
"""E1 correctness and trace-completeness checks, version 2 (CPU, read-only).

Version history
  v1  checks/e1_completeness.py   frozen before Stage 1 (packet 5c86ae1c). Kept unchanged as the pre-registered checker.
  v2  this file                   Phase 1.5 correction (2026-11-14). Differences from v1, each with a regression test in
                                  tests/test_e1_completeness_v2.py:
      (a) trace tag `e1_b{b}_kv{N}k_w{w}_r{r}` is parsed as KV = N * 1024 tokens, so cells match the plan keys
          (v1 compared 32 with 32768 and silently skipped the per-cell minimum sub-check);
      (b) the per-cell pure-decode minimum is checked for every cell of the plan the run executed (--plan dry|main|auto,
          auto = the plan whose request count matches the requests seen) and fails on a planned cell that is absent;
      (c) token accounting separates pure-decode iterations from ramp (mixed prefill+decode) iterations; every emitted
          position is accounted for in one of the two; nothing requires the pure-decode count to equal the corpus rows
          (v1's join sub-check did, and failed on correct data because ramp iterations also emit tokens);
      (d) the join check is the pre-registered rule itself: for every request and every position j in 1..OUT_LEN-1 there is
          exactly one iteration with output_tokens_before == j and emitted_tokens == 1, the iterations are increasing in j,
          and the corpus segment row j-1 is that token's route;
      (e) routing records are validated structurally: .npy count, shape (OUT_LEN, L, K), dtype uint16, expert ids < E,
          no duplicate expert inside a (token, layer) top-k, corpus file size == num_tokens*L*K*2, segments unique,
          contiguous, non-overlapping, lengths OUT_LEN-1, segment names == metrics request ids, and npy[1:] == segment.

Usage
    python checks/e1_completeness_v2.py --profile <profile dir> [--inputs E1-INPUTS.json] [--parsed]
                                        [--compare <other profile dir or metrics.jsonl>] [--out report.json]
                                        [--out-len 1024] [--checksum] [--plan auto|dry|main]

Exit code 1 if any check FAILs. The overhead gate (--compare) is the pre-registered point-estimate rule
(per-cell median iteration time ratio <= 1.02). The block-bootstrap interval printed beside it is the prospective
Stage 1.5 operational criterion (E1-OVERHEAD-METHODOLOGY.md); it is reported, it does not change this checker's verdict.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

TAG = re.compile(r"e1_b(\d+)_kv(\d+)k_w(\d+)_r(\d+)")
OUT_LEN_DEFAULT = 1024
GATE = 1.02
VERSION = "2.0.0"


@dataclass
class Report:
    checks: list = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def add(self, status: str, msg: str) -> None:
        self.checks.append((status, msg))
        print(f"{status:12s} {msg}")

    @property
    def fails(self) -> int:
        return sum(1 for s, _ in self.checks if s == "FAIL")


def parse_tag(request_id: str):
    """(b, kv_tokens, wave, r) or None. KV in the tag is in units of 1024 tokens."""
    m = TAG.search(request_id)
    if not m:
        return None
    b, kv_k, w, r = (int(x) for x in m.groups())
    return b, kv_k * 1024, w, r


def read_metrics(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def metrics_path(profile_or_file: str) -> str:
    if os.path.isfile(profile_or_file):
        return profile_or_file
    paths = glob.glob(os.path.join(profile_or_file, "vllm", "*_metrics.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no vllm/*_metrics.jsonl under {profile_or_file}")
    return paths[0]


@dataclass
class IterationAccounting:
    pure: Counter = field(default_factory=Counter)            # (b, kv, wave) -> pure-decode iterations
    ramp: Counter = field(default_factory=Counter)            # (b, kv, wave) -> iterations with fewer/more decode reqs or prefill
    elapsed: dict = field(default_factory=lambda: defaultdict(list))  # (b, kv, wave) -> observed_elapsed_ms of pure iterations
    overrun: int = 0                                          # iterations carrying requests of two cells
    untagged: set = field(default_factory=set)                # request ids without an E1 tag
    final_tokens: dict = field(default_factory=dict)          # request -> output_tokens_before + emitted at last sight
    pos_iters: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))  # req -> pos -> [iteration]
    pos_pure: dict = field(default_factory=lambda: defaultdict(dict))  # req -> pos -> bool (emitted in a pure iteration)
    multi_emit: int = 0                                       # progress entries with emitted_tokens not in {0, 1}


def account(recs: list[dict]) -> IterationAccounting:
    acc = IterationAccounting()
    for d in recs:
        prog = d.get("decode_request_progress") or []
        tags = {}
        for p in prog:
            t = parse_tag(p["request_id"])
            if t is None:
                acc.untagged.add(p["request_id"])
            else:
                tags[p["request_id"]] = t
        for p in prog:
            acc.final_tokens[p["request_id"]] = max(acc.final_tokens.get(p["request_id"], 0), p["output_tokens_before"] + p["emitted_tokens"])
        cells = {(t[0], t[1]) for t in tags.values()}
        pure = False
        if len(cells) > 1:
            acc.overrun += 1
        elif len(cells) == 1:
            b, kv = next(iter(cells))
            wave = min(t[2] for t in tags.values())
            key = (b, kv, wave)
            if d["decode_requests"] == b and d["prefill_tokens"] == 0:
                acc.pure[key] += 1
                acc.elapsed[key].append(d["observed_elapsed_ms"])
                pure = True
            else:
                acc.ramp[key] += 1
        for p in prog:
            e = p["emitted_tokens"]
            if e not in (0, 1):
                acc.multi_emit += 1
            if e == 1:
                pos = p["output_tokens_before"]
                acc.pos_iters[p["request_id"]][pos].append(d["iteration_index"])
                acc.pos_pure[p["request_id"]][pos] = pure
    return acc


def check_result(rep: Report, P: str) -> None:
    pr_path = os.path.join(P, "profile_result.json")
    if not os.path.exists(pr_path):
        rep.add("FAIL", "profile_result.json missing")
        return
    pr = json.load(open(pr_path))
    v = pr.get("validation")
    if v is not None:
        rep.add("PASS" if v.get("ok") else "FAIL", f"export validation ok={v.get('ok')} nvtx_forward_ranges={v.get('nvtx_forward_ranges')} kernel_rows={v.get('kernel_rows')}")
    dev = pr.get("parsed_device_ids")
    if dev is not None:
        rep.add("PASS" if dev == [0, 1, 2, 3] else "FAIL", f"parsed devices {dev}")
    rep.add("INFO", f"profile_kind={pr.get('profile_kind')} request_timing_count={pr.get('request_timing_count')} corpus_tokens={pr.get('token_corpus_tokens')}")
    rep.data["profile_result"] = {k: pr.get(k) for k in ("profile_kind", "validation", "parsed_device_ids", "request_timing_count", "token_corpus_tokens")}


def select_plan(inputs: dict, acc: IterationAccounting, which: str) -> tuple[str, dict]:
    """(plan name, {(b, kv, wave): cell}) for 'dry' / 'main', or 'auto': the plan whose request count equals the requests
    seen; failing that the plan whose cells overlap the observed cells most (ties -> the smaller plan)."""
    plans = {k: {(p["b"], p["kv"], p["wave"]): p for p in inputs[k]["plan"]} for k in ("dry", "main") if k in inputs}
    if which != "auto":
        return which, plans[which]
    seen = len([r for r in acc.final_tokens if parse_tag(r)])
    exact = [k for k in plans if inputs[k].get("requests") == seen]
    if len(exact) == 1:
        return exact[0], plans[exact[0]]
    observed = set(acc.pure) | set(acc.ramp)
    best = sorted(plans, key=lambda k: (-len(observed & set(plans[k])), len(plans[k])))[0]
    return best, plans[best]


def check_cells(rep: Report, acc: IterationAccounting, inputs: dict | None, out_len: int, which_plan: str = "auto") -> None:
    rep.add("INFO", f"ramp iterations (prefill present or decode_requests != b inside a cell): {sum(acc.ramp.values())}")
    rep.add("PASS" if acc.overrun == 0 else "FAIL", f"iterations carrying requests of two cells (overrun): {acc.overrun}")
    rep.data["cells"] = {f"b{b}_kv{kv}_w{w}": {"pure": n, "ramp": acc.ramp.get((b, kv, w), 0),
                                              "elapsed_ms_median": float(np.median(acc.elapsed[(b, kv, w)])),
                                              "elapsed_ms_p90": float(np.quantile(acc.elapsed[(b, kv, w)], 0.9))}
                         for (b, kv, w), n in sorted(acc.pure.items())}
    if inputs is None:
        rep.add("INFO", f"no --inputs: {len(acc.pure)} tagged cells observed")
        return
    plan_name, plan = select_plan(inputs, acc, which_plan)
    floor = inputs["design"].get("MIN_PURE_DECODE", 400)
    observed = set(acc.pure) | set(acc.ramp)
    rep.add("INFO", f"plan '{plan_name}' ({len(plan)} cells, {inputs[plan_name].get('requests')} requests) selected by --plan {which_plan}; every planned cell is checked")
    rep.data["plan"] = plan_name
    if not plan:
        rep.add("FAIL", f"plan '{plan_name}' has no cells")
    for key in sorted(plan):
        p = plan[key]
        need = min(p["pure_decode_iters_min"], floor)
        n = acc.pure.get(key, 0)
        rep.add("PASS" if n >= need else "FAIL", f"cell b={key[0]} kv={key[1]} wave={key[2]}: pure-decode iterations {n} (need >= {need}), ramp {acc.ramp.get(key, 0)}")
    unplanned = sorted(k for k in observed if k not in plan)
    rep.add("PASS" if not unplanned else "FAIL", f"observed cells not in the plan: {unplanned}")
    rep.add("PASS" if not acc.untagged else "FAIL", f"request ids without an E1 tag: {len(acc.untagged)} {sorted(acc.untagged)[:3]}")


def check_tokens(rep: Report, acc: IterationAccounting, out_len: int) -> list[str]:
    reqs = sorted(r for r in acc.final_tokens if parse_tag(r))
    wrong = {r: n for r, n in acc.final_tokens.items() if r in reqs and n != out_len}
    rep.add("PASS" if reqs and not wrong else "FAIL", f"requests seen {len(reqs)}; requests whose emitted tokens != {out_len}: {len(wrong)} {list(wrong.items())[:3]}")
    rep.add("PASS" if acc.multi_emit == 0 else "FAIL", f"progress entries with emitted_tokens not in {{0,1}}: {acc.multi_emit}")
    # ramp vs pure accounting for positions 1..out_len-1
    ramp_tok, pure_tok = [], []
    for r in reqs:
        pp = acc.pos_pure.get(r, {})
        pure_tok.append(sum(1 for pos, ispure in pp.items() if 1 <= pos <= out_len - 1 and ispure))
        ramp_tok.append(sum(1 for pos, ispure in pp.items() if 1 <= pos <= out_len - 1 and not ispure))
    if reqs:
        tot_ok = all(a + b == out_len - 1 for a, b in zip(pure_tok, ramp_tok))
        rep.add("PASS" if tot_ok else "FAIL", f"per request, positions 1..{out_len - 1} emitted in pure + ramp iterations == {out_len - 1}: {'all' if tot_ok else 'NOT all'}; ramp-emitted per request min/median/max = {min(ramp_tok)}/{int(np.median(ramp_tok))}/{max(ramp_tok)}")
        rep.data["token_accounting"] = {"requests": len(reqs), "ramp_emitted_min": min(ramp_tok), "ramp_emitted_median": float(np.median(ramp_tok)), "ramp_emitted_max": max(ramp_tok)}
    return reqs


def check_join(rep: Report, acc: IterationAccounting, reqs: list[str], out_len: int) -> None:
    dup = miss = nonmono = 0
    examples = []
    for r in reqs:
        pi = acc.pos_iters.get(r, {})
        its = []
        for pos in range(1, out_len):
            L = pi.get(pos, [])
            if len(L) > 1:
                dup += 1
                if len(examples) < 3:
                    examples.append((r, pos, L))
            elif not L:
                miss += 1
                if len(examples) < 3:
                    examples.append((r, pos, "missing"))
            else:
                its.append(L[0])
        if any(b <= a for a, b in zip(its, its[1:])):
            nonmono += 1
    rep.add("PASS" if dup == 0 else "FAIL", f"(request, position) pairs emitted in more than one iteration (duplicate join): {dup}")
    rep.add("PASS" if miss == 0 else "FAIL", f"(request, position) pairs never emitted (missing join): {miss}")
    rep.add("PASS" if nonmono == 0 else "FAIL", f"requests whose position->iteration map is not strictly increasing: {nonmono}")
    if examples:
        rep.add("INFO", f"join examples: {examples}")
    rep.data["join"] = {"duplicate": dup, "missing": miss, "non_monotone": nonmono}


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def check_routes(rep: Report, P: str, acc: IterationAccounting, reqs: list[str], out_len: int, checksum: bool) -> None:
    routes_dir = os.path.join(P, "routed_experts")
    corpus = os.path.join(P, "token_corpus")
    if not (os.path.isdir(routes_dir) or os.path.isdir(corpus)):
        rep.add("INFO", "no routed_experts / token_corpus directory (hook-off arm or Option B capture)")
        return
    man_path = os.path.join(corpus, "manifest.json")
    if not os.path.exists(man_path):
        rep.add("FAIL", "token_corpus/manifest.json missing")
        return
    man = json.load(open(man_path))
    prov = json.load(open(os.path.join(corpus, "provenance.json")))
    L, K, E = man["num_layers"], man["top_k"], man["num_experts"]
    npy_paths = sorted(glob.glob(os.path.join(routes_dir, "*.npy")))
    npy_names = {os.path.basename(p)[:-4] for p in npy_paths}
    reqset = set(reqs)
    rep.add("PASS" if npy_names == reqset else "FAIL", f"routed_experts files {len(npy_paths)} vs requests {len(reqs)}; missing {sorted(reqset - npy_names)[:3]} extra {sorted(npy_names - reqset)[:3]}")
    bad_shape = bad_dtype = bad_ids = dup_topk = 0
    arrays = {}
    for p in npy_paths:
        a = np.load(p, mmap_mode="r")
        name = os.path.basename(p)[:-4]
        if a.dtype != np.uint16:
            bad_dtype += 1
        if a.shape != (out_len, L, K):
            bad_shape += 1
            continue
        arr = np.asarray(a)
        arrays[name] = arr
        if arr.size and int(arr.max()) >= E:
            bad_ids += 1
        srt = np.sort(arr, axis=2)
        dup_topk += int((srt[:, :, 1:] == srt[:, :, :-1]).any(axis=2).sum())
    rep.add("PASS" if bad_shape == 0 else "FAIL", f".npy files with shape != ({out_len}, {L}, {K}): {bad_shape}")
    rep.add("PASS" if bad_dtype == 0 else "FAIL", f".npy files with dtype != uint16: {bad_dtype}")
    rep.add("PASS" if bad_ids == 0 else "FAIL", f".npy files with an expert id >= {E}: {bad_ids}")
    rep.add("PASS" if dup_topk == 0 else "FAIL", f"(token, layer) rows with a duplicate expert in the top-{K}: {dup_topk}")
    # corpus
    data_path = os.path.join(corpus, man["data_file"])
    exp_rows = len(reqs) * (out_len - 1)
    rep.add("PASS" if man["num_tokens"] == exp_rows else "FAIL", f"corpus num_tokens {man['num_tokens']} vs {len(reqs)} x ({out_len}-1) = {exp_rows}")
    size = os.path.getsize(data_path) if os.path.exists(data_path) else -1
    rep.add("PASS" if size == man["num_tokens"] * L * K * 2 else "FAIL", f"corpus file size {size} vs num_tokens*L*K*2 = {man['num_tokens'] * L * K * 2}")
    if size != man["num_tokens"] * L * K * 2:
        return
    routes = np.fromfile(data_path, dtype="<u2").reshape(-1, L, K)
    rep.add("PASS" if routes.size == 0 or int(routes.max()) < E else "FAIL", f"corpus max expert id {int(routes.max()) if routes.size else None} < {E}")
    segs = prov["request_segments"]
    names = [s["request"] for s in segs]
    rep.add("PASS" if len(set(names)) == len(names) else "FAIL", f"segments {len(segs)}, unique request names {len(set(names))}")
    bad_len = [s["request"] for s in segs if s["length"] != out_len - 1]
    rep.add("PASS" if not bad_len else "FAIL", f"segments with length != {out_len - 1}: {len(bad_len)}")
    ordered = sorted(segs, key=lambda s: s["start"])
    contiguous = all(a["start"] + a["length"] == b["start"] for a, b in zip(ordered, ordered[1:])) and (not ordered or (ordered[0]["start"] == 0 and ordered[-1]["start"] + ordered[-1]["length"] == man["num_tokens"]))
    rep.add("PASS" if contiguous else "FAIL", "segments contiguous, non-overlapping and covering [0, num_tokens)")
    rep.add("PASS" if set(names) == reqset else "FAIL", f"segment names == metrics request ids: missing {sorted(reqset - set(names))[:3]} extra {sorted(set(names) - reqset)[:3]}")
    mism = 0
    for s in segs:
        a = arrays.get(s["request"])
        if a is None:
            mism += 1
            continue
        seg = routes[s["start"]:s["start"] + s["length"]]
        if seg.shape != a[1:].shape or not np.array_equal(a[1:], seg):
            mism += 1
    rep.add("PASS" if mism == 0 else "FAIL", f"requests whose corpus segment != npy[1:] (row 0 = last prompt token's forward): {mism}")
    # join rule applied to the corpus: segment row j-1 <-> the unique iteration with output_tokens_before == j
    joinable = sum(1 for s in segs if all(len(acc.pos_iters.get(s["request"], {}).get(j, [])) == 1 for j in range(1, out_len)))
    rep.add("PASS" if joinable == len(segs) else "FAIL", f"segments whose every row maps to exactly one iteration: {joinable} of {len(segs)}")
    if checksum:
        h = fnv1a64(routes.tobytes())
        rep.add("PASS" if h == man.get("checksum_fnv1a64") else "FAIL", f"corpus checksum_fnv1a64 {h} vs manifest {man.get('checksum_fnv1a64')}")
    rep.data["corpus"] = {"num_tokens": man["num_tokens"], "segments": len(segs), "bad_len": len(bad_len), "npy_mismatch": mism, "dup_topk": dup_topk}


def block_bootstrap_ratio(x_on, x_off, block: int, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x_on, x_off = np.asarray(x_on, float), np.asarray(x_off, float)

    def mbb(x):
        n = len(x)
        L = max(1, min(block, n))
        nb = int(np.ceil(n / L))
        starts = rng.integers(0, n - L + 1, nb)
        idx = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]
        return np.median(x[idx])

    r = np.array([mbb(x_on) / mbb(x_off) for _ in range(draws)])
    return float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def check_overhead(rep: Report, acc: IterationAccounting, other: IterationAccounting, block: int, draws: int) -> None:
    ratios = {}
    for key in sorted(set(acc.elapsed) & set(other.elapsed)):
        x_on, x_off = acc.elapsed[key], other.elapsed[key]
        r = float(np.median(x_on) / np.median(x_off))
        lo, hi = block_bootstrap_ratio(x_on, x_off, block, draws, seed=20261114)
        name = f"b{key[0]}_kv{key[1]}_w{key[2]}"
        ratios[name] = {"ratio_median": round(r, 5), "n_this": len(x_on), "n_other": len(x_off), "block_bootstrap_ci95": [round(lo, 5), round(hi, 5)], "block": block}
        rep.add("PASS" if r <= GATE else "FAIL", f"overhead {name}: median iteration ratio this/other = {r:.4f} (pre-registered point gate <= {GATE})")
        rep.add("PASS" if hi <= GATE else ("INCONCLUSIVE" if r <= GATE else "FAIL"), f"  prospective criterion {name}: moving-block bootstrap 95 % CI [{lo:.4f}, {hi:.4f}] (block {block}) upper <= {GATE} (reported, not gating)")
    rep.add("PASS" if ratios else "FAIL", f"overhead cells compared: {len(ratios)}")
    rep.data["overhead"] = ratios


def check_parsed(rep: Report, P: str, out_path: str | None) -> None:
    pj = os.path.join(P, "parsed.json")
    if not os.path.exists(pj):
        rep.add("FAIL", "parsed.json missing (--parsed)")
        return
    d = json.load(open(pj))
    det = d["iteration_details"]
    sa = d.get("step_alignment", {})
    rep.add("PASS" if not sa.get("unpaired_steps_by_device") else "FAIL", f"step_alignment unpaired_steps_by_device={sa.get('unpaired_steps_by_device')}")
    bad_rng = sum(1 for x in det if len(x["ranges"]) != 28)
    bad_dev = sum(1 for x in det if x["devices_absent"])
    rep.add("PASS" if bad_rng == 0 else "FAIL", f"iterations without 28 NVTX ranges: {bad_rng} of {len(det)}")
    rep.add("PASS" if bad_dev == 0 else "FAIL", f"iterations with devices_absent: {bad_dev}")
    jit = [x["iteration"] for x in det[50:] if x["jit_module_loads"]]
    rep.add("PASS" if len(jit) <= 20 else "FAIL", f"JIT-affected iterations after warm-up: {len(jit)} (limit 20; exclusion list)")
    rep.data["parsed"] = {"iterations": len(det), "bad_ranges": bad_rng, "bad_devices": bad_dev, "jit_iterations": jit}
    if out_path:
        excl = os.path.join(os.path.dirname(os.path.abspath(out_path)), "jit_excl.txt")
        with open(excl, "w") as fh:
            fh.write(",".join(str(i) for i in sorted(set(jit) | {x["iteration"] for x in det[:50] if x["jit_module_loads"]})) + "\n")
        rep.add("INFO", f"exclusion list written to {excl}")


def run(profile: str, inputs_path: str | None = None, parsed: bool = False, compare: str | None = None,
        out: str | None = None, out_len: int | None = None, checksum: bool = False, block: int = 25, draws: int = 2000,
        plan: str = "auto") -> Report:
    rep = Report()
    rep.data["checker_version"] = VERSION
    rep.data["profile"] = profile
    inputs = json.load(open(inputs_path)) if inputs_path else None
    out_len = out_len or (inputs["design"]["OUT_LEN"] if inputs else OUT_LEN_DEFAULT)
    check_result(rep, profile)
    recs = read_metrics(metrics_path(profile))
    rep.add("PASS" if recs else "FAIL", f"metrics records: {len(recs)}")
    acc = account(recs)
    check_cells(rep, acc, inputs, out_len, plan)
    reqs = check_tokens(rep, acc, out_len)
    check_join(rep, acc, reqs, out_len)
    if compare:
        other = account(read_metrics(metrics_path(compare)))
        check_overhead(rep, acc, other, block, draws)
    check_routes(rep, profile, acc, reqs, out_len, checksum)
    if parsed:
        check_parsed(rep, profile, out)
    rep.data["checks"] = [{"status": s, "msg": m} for s, m in rep.checks]
    rep.data["fails"] = rep.fails
    if out:
        json.dump(rep.data, open(out, "w"), indent=1, default=str)
    print("RESULT", "PASS" if rep.fails == 0 else f"FAIL ({rep.fails})", f"[checker v{VERSION}]")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--inputs")
    ap.add_argument("--parsed", action="store_true")
    ap.add_argument("--compare", help="other arm: profile dir or metrics.jsonl")
    ap.add_argument("--out")
    ap.add_argument("--out-len", type=int)
    ap.add_argument("--checksum", action="store_true", help="verify the corpus fnv1a64 (slow, pure python)")
    ap.add_argument("--block", type=int, default=25, help="moving-block bootstrap block length (iterations)")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--plan", choices=["auto", "dry", "main"], default="auto", help="which E1-INPUTS plan the run executed")
    a = ap.parse_args(argv)
    rep = run(a.profile, a.inputs, a.parsed, a.compare, a.out, a.out_len, a.checksum, a.block, a.draws, a.plan)
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
