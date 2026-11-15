#!/usr/bin/env python3
"""E1 paired comparison, version 2 (Phase 1.5): serial-correlation-aware overhead statistics and an off/off control.

    python analysis/e1_pair_compare_v2.py --off <hook-off profile dir> --on <hook-on profile dir> --inputs <E1 inputs json>
                                          --out <report dir> [--control <second hook-off profile dir>] [--block N] [--draws 2000]

Keeps analysis/e1_pair_compare.py (used for the Stage 1 report) unchanged. Adds, per cell, on the pure-decode
iteration times (observed_elapsed_ms):

  * the pre-registered point gate: median(on)/median(off) <= 1.02                        -> PASS / FAIL (unchanged)
  * serial-correlation diagnostics: ACF at lags 1..5, 10, 20; integrated autocorrelation time tau (Geyer initial-positive
    truncation); the block length used below L = clip(ceil(5 * tau), 10, n // 20)
  * the prospective operational criterion (E1-OVERHEAD-METHODOLOGY.md): moving-block bootstrap 95 % interval of the
    ratio of medians (blocks of L consecutive iterations, resampled independently in each arm); PASS if the upper bound
    <= 1.02, INCONCLUSIVE if the point estimate passes but the upper bound does not, FAIL if the point estimate fails
  * with --control: the same statistics for control/off (two hook-off runs). Their ratio is the run-to-run null: it
    includes autotuner tactic re-selection, clock/thermal drift and scheduling noise. The report shows on/off next to
    control/off, a noise flag (NOISY if the control/off interval reaches outside [1/1.02, 1.02]) and the excess
    (on/off) / (control/off) with its interval from the same bootstrap draws. E1-OVERHEAD-METHODOLOGY.md says how these
    are read; the excess carries no gate.

Request-level pairing, throughput and output_preview identity are reported as in v1 (functions imported from it).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_v1_spec = importlib.util.spec_from_file_location("e1_pair_compare_v1", Path(__file__).with_name("e1_pair_compare.py"))
v1 = importlib.util.module_from_spec(_v1_spec)
sys.modules["e1_pair_compare_v1"] = v1
_v1_spec.loader.exec_module(v1)

GATE = 1.02


def acf(x: np.ndarray, maxlag: int) -> np.ndarray:
    x = np.asarray(x, float)
    x = x - x.mean()
    v = float((x * x).sum()) or 1.0
    return np.array([1.0] + [float((x[:-k] * x[k:]).sum()) / v for k in range(1, maxlag + 1)])


def tau_int(a: np.ndarray) -> float:
    """Integrated autocorrelation time with Geyer's initial-positive-sequence truncation."""
    t = 1.0
    for k in range(1, len(a)):
        if a[k] <= 0:
            break
        t += 2.0 * a[k]
    return t


def block_length(n: int, tau: float, fixed: int | None) -> int:
    if fixed:
        return max(1, min(fixed, n))
    return int(max(10, min(math.ceil(5 * tau), max(10, n // 20))))


def mbb_median(x: np.ndarray, L: int, rng) -> float:
    n = len(x)
    L = max(1, min(L, n))
    nb = int(math.ceil(n / L))
    starts = rng.integers(0, n - L + 1, nb)
    idx = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]
    return float(np.median(x[idx]))


def ratio_draws(x_num, x_den, L_num, L_den, draws, rng) -> np.ndarray:
    x_num, x_den = np.asarray(x_num, float), np.asarray(x_den, float)
    return np.array([mbb_median(x_num, L_num, rng) / mbb_median(x_den, L_den, rng) for _ in range(draws)])


def ci(r: np.ndarray):
    return float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975)), float(r.std())


def ratio_ci(x_num, x_den, L_num, L_den, draws, rng):
    return ci(ratio_draws(x_num, x_den, L_num, L_den, draws, rng))


def diagnostics(x) -> dict:
    x = np.asarray(x, float)
    a = acf(x, 20)
    tau = tau_int(a)
    h = len(x) // 2
    return {"n": int(len(x)), "median": float(np.median(x)), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "p10": float(np.quantile(x, 0.1)), "p90": float(np.quantile(x, 0.9)),
            "acf": {f"lag{k}": round(float(a[k]), 3) for k in (1, 2, 3, 4, 5, 10, 20)}, "tau_int": round(tau, 2),
            "median_first_half": float(np.median(x[:h])), "median_second_half": float(np.median(x[h:])),
            "per_100_medians": [round(float(np.median(x[i:i + 100])), 3) for i in range(0, len(x), 100)]}


def verdict(point: float, hi: float) -> str:
    if point > GATE:
        return "FAIL"
    return "PASS" if hi <= GATE else "INCONCLUSIVE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True); ap.add_argument("--on", required=True)
    ap.add_argument("--control", help="second hook-off run (off/off null)")
    ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=0, help="fixed block length; 0 = rule clip(ceil(5 tau), 10, n//20)")
    ap.add_argument("--draws", type=int, default=2000); ap.add_argument("--seed", type=int, default=20261114)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inputs = json.load(open(a.inputs))
    out_len = inputs["design"]["OUT_LEN"]
    rng = np.random.default_rng(a.seed)
    import glob
    arms = {"off": a.off, "on": a.on}
    if a.control:
        arms["control"] = a.control
    el = {}
    for k, P in arms.items():
        recs = v1.read_jsonl(glob.glob(os.path.join(P, "vllm", "*_metrics.jsonl"))[0])
        el[k], _ = v1.pure_decode_elapsed(recs, out_len)
    rep = {"arms": arms, "gate": GATE, "cells": {}, "verdicts": {}}
    lines = ["# E1 paired comparison v2 (serial-correlation-aware)", "", *[f"{k}: `{v}`" for k, v in arms.items()], "",
             "## Pure-decode iteration time: diagnostics and overhead statistics", "",
             "| cell | arm | n | median | mean | std | acf1 | acf2 | acf5 | acf10 | tau_int | block L | first/second half median |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    overall = "PASS"
    for key in sorted(set(el["off"]) | set(el["on"])):
        name = f"b{key[0]}_kv{key[1]}_w{key[2]}"
        cell = {}
        Ls = {}
        for arm in arms:
            x = el[arm].get(key, [])
            if not x:
                cell[arm] = {"n": 0}
                continue
            d = diagnostics(x)
            Ls[arm] = block_length(d["n"], d["tau_int"], a.block)
            d["block"] = Ls[arm]
            cell[arm] = d
            lines.append(f"| {name} | {arm} | {d['n']} | {d['median']:.3f} | {d['mean']:.3f} | {d['std']:.3f} | {d['acf']['lag1']} | {d['acf']['lag2']} | {d['acf']['lag5']} | {d['acf']['lag10']} | {d['tau_int']} | {Ls[arm]} | {d['median_first_half']:.3f}/{d['median_second_half']:.3f} |")
        if cell["off"]["n"] and cell["on"]["n"]:
            point = cell["on"]["median"] / cell["off"]["median"]
            r_on = ratio_draws(el["on"][key], el["off"][key], Ls["on"], Ls["off"], a.draws, rng)
            lo, hi, sd = ci(r_on)
            v = verdict(point, hi)
            cell["on_over_off"] = {"point": point, "ci95_block": [lo, hi], "boot_sd": sd, "point_gate": "PASS" if point <= GATE else "FAIL", "prospective": v}
            if v == "FAIL" or (v == "INCONCLUSIVE" and overall != "FAIL"):
                overall = v
            if "control" in arms and cell.get("control", {}).get("n"):
                pc = cell["control"]["median"] / cell["off"]["median"]
                r_c = ratio_draws(el["control"][key], el["off"][key], Ls["control"], Ls["off"], a.draws, rng)
                lo_c, hi_c, sd_c = ci(r_c)
                lo_x, hi_x, _ = ci(r_on / r_c)                    # excess = (on/off) / (control/off), joint draws
                cell["control_over_off"] = {"point": pc, "ci95_block": [lo_c, hi_c], "boot_sd": sd_c,
                                            "noise_flag": "NOISY" if hi_c > GATE or lo_c < 1 / GATE else "quiet"}
                cell["excess_on_vs_control"] = point / pc
                cell["excess_ci95_block"] = [lo_x, hi_x]
        else:
            cell["on_over_off"] = {"point_gate": "FAIL", "prospective": "FAIL", "note": "cell missing on one side"}
            overall = "FAIL"
        rep["cells"][name] = cell
    lines += ["", "## Overhead gates per cell", "",
              "| cell | on/off median ratio | pre-registered point gate (<= 1.02) | block-bootstrap 95 % CI | prospective criterion | control/off | control CI (noise flag) | excess on vs control [CI] |",
              "|---|---|---|---|---|---|---|---|"]
    for name, c in rep["cells"].items():
        o = c["on_over_off"]
        if "point" not in o:
            lines.append(f"| {name} | - | FAIL | - | FAIL | - | - | - |"); continue
        cc = c.get("control_over_off")
        lines.append(f"| {name} | {o['point']:.4f} | {o['point_gate']} | [{o['ci95_block'][0]:.4f}, {o['ci95_block'][1]:.4f}] | {o['prospective']} | "
                     + (f"{cc['point']:.4f} | [{cc['ci95_block'][0]:.4f}, {cc['ci95_block'][1]:.4f}] ({cc['noise_flag']}) | {c['excess_on_vs_control']:.4f} [{c['excess_ci95_block'][0]:.4f}, {c['excess_ci95_block'][1]:.4f}] |" if cc else "- | - | - |"))
    rep["verdicts"] = {"point_gate": "PASS" if all(c["on_over_off"].get("point_gate") == "PASS" for c in rep["cells"].values()) else "FAIL",
                       "prospective_block_bootstrap": overall}
    lines += ["", f"**Pre-registered point gate: {rep['verdicts']['point_gate']}. Prospective block-bootstrap criterion: {overall}.**", ""]

    # request-level and throughput, as in v1
    r_off, _ = v1.load_replay(a.off); r_on, _ = v1.load_replay(a.on)
    if r_off and r_on:
        common = sorted(set(r_off) & set(r_on))
        same = sum(1 for r in common if v1.dig(r_off[r], "output_preview") == v1.dig(r_on[r], "output_preview"))
        dur = [(float(v1.dig(r_off[r], "total_duration_ms")), float(v1.dig(r_on[r], "total_duration_ms"))) for r in common]
        chunks = (sorted({v1.dig(r_off[r], "chunk_count") for r in common}), sorted({v1.dig(r_on[r], "chunk_count") for r in common}))
        rep["requests"] = {"paired": len(common), "preview_identical": same, "chunk_counts_off_on": chunks,
                           "dur_ratio_median": float(np.median([b / a_ for a_, b in dur]))}
        lines += ["## Requests", "", f"paired {len(common)}; output_preview identical {same}/{len(common)}; chunk_count values off {chunks[0]} on {chunks[1]} "
                  f"(equal sets = same wire mode); median per-request duration ratio on/off {rep['requests']['dur_ratio_median']:.4f}", ""]
        if "control" in arms:
            r_c, _ = v1.load_replay(a.control)
            if r_c:
                cc = sorted(set(r_off) & set(r_c))
                same_c = sum(1 for r in cc if v1.dig(r_off[r], "output_preview") == v1.dig(r_c[r], "output_preview"))
                rep["requests"]["preview_identical_control_vs_off"] = same_c
                lines += [f"control vs off: output_preview identical {same_c}/{len(cc)} (run-to-run token determinism)", ""]
    lines += ["## Verdicts", "", "```", json.dumps(rep["verdicts"], indent=1), "```"]
    Path(a.out, "E1-PAIR-COMPARE-V2.md").write_text("\n".join(lines) + "\n")
    json.dump(rep, open(os.path.join(a.out, "e1_pair_compare_v2.json"), "w"), indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
