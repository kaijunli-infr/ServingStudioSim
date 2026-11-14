#!/usr/bin/env python3
"""E1 Stage 1: paired comparison of the two-cell dry runs (routing hook off vs on). CPU, read-only.

    python analysis/e1_pair_compare.py --off <e1_dry_off profile dir> --on <e1_dry_on profile dir> \
        --inputs E1-INPUTS.json --out <report dir>

Pre-registered overhead gate (E1 packet, item 6): per cell, median engine-side observed_elapsed_ms over the
pure-decode iterations of the cell, hook-on / hook-off <= 1.02.  This script applies that gate and adds a
bootstrap 95 % interval on the ratio of medians; a cell whose point estimate passes but whose interval upper
bound exceeds 1.02 is reported INCONCLUSIVE (not PASS).  Thresholds are not tuned here.

Also reported, per cell and overall, with variability (median / mean / std / p10 / p90):
  * engine iteration time (observed_elapsed_ms) of pure-decode iterations, hook off vs on
  * request latency from replay.jsonl: first_token_ms and total_duration_ms per request, paired by request id
  * throughput: emitted output tokens / wall span of the cell's wave, from replay timestamps when present,
    otherwise from the engine record (sum emitted_tokens / sum observed_elapsed_ms)
  * token identity: output_len_actual, server completion_tokens, and output_preview text equality off vs on
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

TAG = re.compile(r"e1_b(\d+)_kv(\d+)k_w(\d+)_r(\d+)")
GATE = 1.02


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def stats(xs):
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "median": float(np.median(a)), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "p10": float(np.quantile(a, 0.1)), "p90": float(np.quantile(a, 0.9)), "min": float(a.min()), "max": float(a.max())}


def pure_decode_elapsed(recs, out_len):
    """(b, kv, wave) -> list of observed_elapsed_ms over pure-decode iterations with exactly b decode requests of that cell."""
    el = defaultdict(list)
    emitted = defaultdict(int)
    for d in recs:
        prog = d.get("decode_request_progress") or []
        tags = [TAG.search(p["request_id"]) for p in prog]
        tags = [t.groups() for t in tags if t]
        if not tags:
            continue
        cells = {(int(t[0]), int(t[1]) * 1024) for t in tags}
        if len(cells) != 1:
            continue
        b, kv = next(iter(cells))
        wave = min(int(t[2]) for t in tags)
        if d["decode_requests"] != b or d["prefill_tokens"] != 0:
            continue
        el[(b, kv, wave)].append(d["observed_elapsed_ms"])
        emitted[(b, kv, wave)] += sum(p.get("emitted_tokens", 0) for p in prog)
    return el, emitted


def boot_ratio_ci(x_on, x_off, draws, rng):
    x_on, x_off = np.asarray(x_on), np.asarray(x_off)
    r = np.empty(draws)
    for i in range(draws):
        r[i] = np.median(rng.choice(x_on, x_on.size)) / np.median(rng.choice(x_off, x_off.size))
    return float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def req_id(rec):
    for k in ("request_id", "id", "external_request_id", "name"):
        if k in rec:
            return str(rec[k])
    for k in ("request", "session", "outcome"):
        if isinstance(rec.get(k), dict):
            r = req_id(rec[k])
            if r:
                return r
    return None


def dig(rec, *keys):
    """First value found for any of keys, searching one level of nested dicts too."""
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    for v in rec.values():
        if isinstance(v, dict):
            for k in keys:
                if k in v and v[k] is not None:
                    return v[k]
    return None


def load_replay(profile_dir):
    paths = glob.glob(os.path.join(profile_dir, "**", "replay.jsonl"), recursive=True)
    if not paths:
        return {}, None
    out = {}
    for rec in read_jsonl(paths[0]):
        rid = req_id(rec)
        if not rid:
            continue
        out[rid] = rec
    return out, paths[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True); ap.add_argument("--on", required=True)
    ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=2000); ap.add_argument("--seed", type=int, default=20261114)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inputs = json.load(open(a.inputs))
    out_len = inputs["design"]["OUT_LEN"]
    rng = np.random.default_rng(a.seed)
    rep = {"off": a.off, "on": a.on, "gate": GATE, "out_len": out_len, "cells": {}, "requests": {}, "verdicts": {}}
    lines = ["# E1 Stage 1 dry-run pair: routing hook OFF vs ON", "", f"off: `{a.off}`", f"on:  `{a.on}`", ""]

    # ---- engine iteration times
    m_off = read_jsonl(glob.glob(os.path.join(a.off, "vllm", "*_metrics.jsonl"))[0])
    m_on = read_jsonl(glob.glob(os.path.join(a.on, "vllm", "*_metrics.jsonl"))[0])
    el_off, em_off = pure_decode_elapsed(m_off, out_len)
    el_on, em_on = pure_decode_elapsed(m_on, out_len)
    lines += ["## Engine iteration time, pure-decode iterations (observed_elapsed_ms)", "",
              "| cell | n off | n on | median off | median on | ratio on/off | boot 95 % CI | mean off | mean on | std off | std on | p10/p90 off | p10/p90 on | gate |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    verdict_overall = "PASS"
    plan = {(p["b"], p["kv"], p["wave"]): p for p in inputs["dry"]["plan"]}
    min_pure = inputs["design"].get("MIN_PURE_DECODE")
    count_verdict = "PASS"
    for key in sorted(set(el_off) | set(el_on) | set(plan)):
        so, sn = stats(el_off.get(key, [])), stats(el_on.get(key, []))
        name = f"b{key[0]}_kv{key[1]}_w{key[2]}"
        cell = {"off": so, "on": sn}
        if key in plan:
            need = plan[key]["pure_decode_iters_min"] if min_pure is None else min(plan[key]["pure_decode_iters_min"], min_pure)
            ok = so["n"] >= need and sn["n"] >= need
            cell["pure_decode_min"] = need; cell["pure_decode_count_ok"] = ok
            if not ok: count_verdict = "FAIL"
        if so["n"] and sn["n"]:
            ratio = sn["median"] / so["median"]
            lo, hi = boot_ratio_ci(el_on[key], el_off[key], a.draws, rng)
            if ratio > GATE:
                v = "FAIL"
            elif hi > GATE:
                v = "INCONCLUSIVE"
            else:
                v = "PASS"
            cell.update({"ratio_median": ratio, "ratio_ci95": [lo, hi], "verdict": v})
            lines.append(f"| {name} | {so['n']} | {sn['n']} | {so['median']:.3f} | {sn['median']:.3f} | {ratio:.4f} | [{lo:.4f}, {hi:.4f}] | {so['mean']:.3f} | {sn['mean']:.3f} | {so['std']:.3f} | {sn['std']:.3f} | {so['p10']:.3f}/{so['p90']:.3f} | {sn['p10']:.3f}/{sn['p90']:.3f} | {v} |")
        else:
            v = "FAIL"
            cell["verdict"] = v
            lines.append(f"| {name} | {so['n']} | {sn['n']} | - | - | - | - | - | - | - | - | - | - | FAIL (cell missing on one side) |")
        if v == "FAIL" or (v == "INCONCLUSIVE" and verdict_overall != "FAIL"):
            verdict_overall = v
        # throughput from the engine record
        if so["n"] and sn["n"]:
            cell["engine_tok_per_s"] = {"off": em_off[key] / (sum(el_off[key]) / 1000.0), "on": em_on[key] / (sum(el_on[key]) / 1000.0)}
        rep["cells"][name] = cell
    rep["verdicts"]["overhead_gate"] = verdict_overall
    rep["verdicts"]["pure_decode_counts_vs_plan"] = count_verdict
    lines += ["", "pure-decode iterations per cell vs planned minimum: " + ", ".join(f"{n}: off {c['off']['n']} / on {c['on']['n']} (need >= {c.get('pure_decode_min')})" for n, c in rep["cells"].items()) + f" -> {count_verdict}"]
    lines += ["", f"**Overhead gate (median ratio <= {GATE} per cell, bootstrap CI must also lie <= {GATE} for PASS): {verdict_overall}**", ""]

    # ---- request level
    r_off, p_off = load_replay(a.off)
    r_on, p_on = load_replay(a.on)
    lines += ["## Request-level comparison (replay.jsonl, paired by request id)", ""]
    if not r_off or not r_on:
        lines.append(f"replay.jsonl not found (off: {p_off}, on: {p_on})")
        rep["requests"]["status"] = "missing"
    else:
        common = sorted(set(r_off) & set(r_on))
        only_off, only_on = sorted(set(r_off) - set(r_on)), sorted(set(r_on) - set(r_off))
        rep["requests"].update({"n_off": len(r_off), "n_on": len(r_on), "n_common": len(common), "only_off": only_off, "only_on": only_on})
        lines.append(f"requests: off {len(r_off)}, on {len(r_on)}, paired {len(common)}, unpaired {len(only_off) + len(only_on)}")
        per_cell = defaultdict(lambda: {"ttft_off": [], "ttft_on": [], "dur_off": [], "dur_on": [], "len_off": [], "len_on": [],
                                        "preview_equal": 0, "preview_diff": [], "wrong_len": []})
        for rid in common:
            t = TAG.search(rid)
            if not t:
                continue
            name = f"b{t.group(1)}_kv{int(t.group(2)) * 1024}_w{t.group(3)}"
            c = per_cell[name]
            o, n = r_off[rid], r_on[rid]
            for side, rec in (("off", o), ("on", n)):
                ttft = dig(rec, "first_token_ms", "ttft_ms", "time_to_first_token_ms")
                dur = dig(rec, "total_duration_ms", "duration_ms", "latency_ms")
                ln = dig(rec, "output_len_actual", "completion_tokens", "output_tokens")
                if ttft is not None: c[f"ttft_{side}"].append(float(ttft))
                if dur is not None: c[f"dur_{side}"].append(float(dur))
                if ln is not None:
                    c[f"len_{side}"].append(int(ln))
                    if int(ln) != out_len: c["wrong_len"].append((rid, side, int(ln)))
            po, pn = dig(o, "output_preview"), dig(n, "output_preview")
            if po is not None and pn is not None:
                if po == pn: c["preview_equal"] += 1
                else: c["preview_diff"].append(rid)
        lines += ["", "| cell | n | TTFT med off/on (ms) | TTFT p90 off/on | total dur med off/on (ms) | dur p90 off/on | dur ratio on/off | out tokens off/on (all == OUT_LEN?) | output_preview identical |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name in sorted(per_cell):
            c = per_cell[name]
            so, sn, do, dn = stats(c["ttft_off"]), stats(c["ttft_on"]), stats(c["dur_off"]), stats(c["dur_on"])
            n = len(c["dur_off"])
            lens_ok = all(x == out_len for x in c["len_off"] + c["len_on"]) and c["len_off"] and c["len_on"]
            ratio = dn["median"] / do["median"] if do.get("n") and dn.get("n") else float("nan")
            lines.append(f"| {name} | {n} | {so.get('median', float('nan')):.1f}/{sn.get('median', float('nan')):.1f} | {so.get('p90', float('nan')):.1f}/{sn.get('p90', float('nan')):.1f} | {do.get('median', float('nan')):.1f}/{dn.get('median', float('nan')):.1f} | {do.get('p90', float('nan')):.1f}/{dn.get('p90', float('nan')):.1f} | {ratio:.4f} | {sorted(set(c['len_off']))}/{sorted(set(c['len_on']))} ({'yes' if lens_ok else 'NO'}) | {c['preview_equal']}/{c['preview_equal'] + len(c['preview_diff'])} |")
            rep["requests"][name] = {"ttft_off": so, "ttft_on": sn, "dur_off": do, "dur_on": dn, "dur_ratio_median": ratio, "len_off": sorted(set(c["len_off"])),
                                      "len_on": sorted(set(c["len_on"])), "preview_equal": c["preview_equal"], "preview_diff": c["preview_diff"], "wrong_len": c["wrong_len"]}
        all_lens_ok = all(not v["wrong_len"] for v in rep["requests"].values() if isinstance(v, dict) and "wrong_len" in v)
        all_prev = all(not v["preview_diff"] for v in rep["requests"].values() if isinstance(v, dict) and "preview_diff" in v)
        rep["verdicts"]["token_counts"] = "PASS" if all_lens_ok and not only_off and not only_on else "FAIL"
        rep["verdicts"]["output_preview_identity"] = "PASS" if all_prev else "DIFFERENT"
        lines += ["", f"token counts == OUT_LEN for every request on both sides and all requests paired: {rep['verdicts']['token_counts']}",
                  f"output_preview text identical off vs on for every paired request: {rep['verdicts']['output_preview_identity']} (previews are text prefixes, not token ids; identity is reported, not gated)", ""]

    # ---- throughput
    lines += ["## Throughput (engine record: emitted tokens / summed pure-decode iteration time)", "", "| cell | tok/s off | tok/s on | ratio on/off |", "|---|---|---|---|"]
    for name, c in rep["cells"].items():
        if "engine_tok_per_s" in c:
            t = c["engine_tok_per_s"]
            lines.append(f"| {name} | {t['off']:.1f} | {t['on']:.1f} | {t['on'] / t['off']:.4f} |")
    summ = {}
    for side, P in (("off", a.off), ("on", a.on)):
        sp = glob.glob(os.path.join(P, "**", "summary.json"), recursive=True)
        if sp:
            s = json.load(open(sp[0]))
            summ[side] = s.get("replay", s)
    if summ:
        rep["replay_summary"] = {k: {kk: vv for kk, vv in (v.get("common", v) or {}).items() if not isinstance(vv, (dict, list))} for k, v in summ.items()}
        lines += ["", "replay summary.json (common):", "```", json.dumps(rep["replay_summary"], indent=1), "```"]

    lines += ["", "## Verdicts", "", "```", json.dumps(rep["verdicts"], indent=1), "```"]
    with open(os.path.join(a.out, "E1-PAIR-COMPARE.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    json.dump(rep, open(os.path.join(a.out, "e1_pair_compare.json"), "w"), indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
