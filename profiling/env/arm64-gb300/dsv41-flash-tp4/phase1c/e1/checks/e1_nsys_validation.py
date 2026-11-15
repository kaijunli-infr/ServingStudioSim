#!/usr/bin/env python3
"""Short Nsight validation: CPU checks on one nsys pass (Phase 1.5 deliverable 6; the GPU run is not authorized yet).

    python checks/e1_nsys_validation.py --profile <profile dir> [--tables <Phase 1A compact tables dir>]
                                        [--inputs <E1 inputs json>] [--out report.json]
                                        [--spread-median-ms 0.02 --spread-p90-ms 0.05]

Validates, each PASS/FAIL (exit 1 on any FAIL):
  export        profile_result.validation.ok; nvtx_forward_ranges == 4 x parsed iterations; parsed devices [0,1,2,3];
                .nsys-rep and .sqlite present and non-empty
  NVTX          every iteration has 28 ranges (7 phases x 4 devices); devices_absent empty; step_alignment
                unpaired_steps_by_device == {}; iteration_index_by_device identical on all devices (NVTX ownership)
  alignment     per iteration, the four ranks' GPU kernel windows [first_start, last_end] (iter_rank table, one nsys
                timeline) overlap; GPU start spread and host-side forward NVTX start spread reported median / p90;
                every device's iteration start sequence strictly increasing (timestamp consistency)
  JIT           iterations with jit_module_loads after the first 50: <= 20 (Phase 0 exclusion rule)
  collectives   with --tables (costmodel.extract output): per cell (decode batch b), collective end-time spread across
                ranks median <= 0.02 ms and p90 <= 0.05 ms (packet 4.3); modal collective count K and n_other_K reported
  routes        checks/e1_completeness_v2.py join + routing-record checks on the same profile (hook on)

The parsed.json structure is the Phase 0 parser's (alignment/nsys/parse.py): iteration_details[i].ranges[] with
device_id, phase, start_ns, end_ns. Tables are Phase 1A PassData tables (iterations / iter_rank / collectives parquet);
loading them needs pyarrow (Phase 1A venv).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Rep:
    def __init__(self):
        self.checks, self.data = [], {}

    def add(self, s, m):
        self.checks.append((s, m)); print(f"{s:5s} {m}")

    @property
    def fails(self):
        return sum(1 for s, _ in self.checks if s == "FAIL")


def check_export(rep: Rep, P: str, n_iter: int | None):
    pr_path = os.path.join(P, "profile_result.json")
    if not os.path.exists(pr_path):
        rep.add("FAIL", "profile_result.json missing"); return
    pr = json.load(open(pr_path))
    v = pr.get("validation") or {}
    rep.add("PASS" if v.get("ok") else "FAIL", f"export validation ok={v.get('ok')} kernel_rows={v.get('kernel_rows')} kernel_span_ms={v.get('kernel_span_ms')}")
    if n_iter is not None and v.get("nvtx_forward_ranges") is not None:
        rep.add("PASS" if v["nvtx_forward_ranges"] == 4 * n_iter else "FAIL", f"nvtx_forward_ranges {v['nvtx_forward_ranges']} == 4 x {n_iter} parsed iterations")
    rep.add("PASS" if pr.get("parsed_device_ids") == [0, 1, 2, 3] else "FAIL", f"parsed devices {pr.get('parsed_device_ids')}")
    for ext in (".nsys-rep", ".sqlite"):
        fs = glob.glob(os.path.join(P, "nsys", f"*{ext}"))
        size = os.path.getsize(fs[0]) if fs else 0
        rep.add("PASS" if size > 0 else "FAIL", f"nsys {ext} present: {os.path.basename(fs[0]) if fs else None} {size / 1e9:.2f} GB")
        rep.data[f"size{ext}"] = size


def check_nvtx_alignment(rep: Rep, det: list, sa: dict):
    rep.add("PASS" if not sa.get("unpaired_steps_by_device") else "FAIL", f"step_alignment unpaired_steps_by_device={sa.get('unpaired_steps_by_device')} (rule {sa.get('rule')}, steps {sa.get('steps')})")
    bad_rng = sum(1 for x in det if len(x["ranges"]) != 28)
    bad_dev = sum(1 for x in det if x["devices_absent"])
    bad_idx = sum(1 for x in det if len(set(x.get("iteration_index_by_device", {}).values())) > 1)
    rep.add("PASS" if bad_rng == 0 else "FAIL", f"iterations without 28 NVTX ranges: {bad_rng} of {len(det)}")
    rep.add("PASS" if bad_dev == 0 else "FAIL", f"iterations with devices_absent: {bad_dev}")
    rep.add("PASS" if bad_idx == 0 else "FAIL", f"iterations whose NVTX iteration index differs between devices: {bad_idx}")
    phases = Counter(r["phase"] for x in det for r in x["ranges"])
    rep.add("INFO", f"NVTX phases per device: {dict(sorted(phases.items()))}")
    # forward-window overlap and start spread across devices; per-device monotone starts
    spreads, no_overlap = [], 0
    starts_by_dev = defaultdict(list)
    for x in det:
        fw = {r["device_id"]: (r["start_ns"], r["end_ns"]) for r in x["ranges"] if r["phase"] == "forward"}
        if len(fw) != 4:
            continue
        s = np.array([fw[d][0] for d in sorted(fw)]); e = np.array([fw[d][1] for d in sorted(fw)])
        spreads.append(s.max() - s.min())
        if s.max() >= e.min():
            no_overlap += 1
        for d in fw:
            starts_by_dev[d].append(fw[d][0])
    spreads = np.array(spreads, float) / 1e6
    rep.add("INFO", f"iterations whose four host-side forward NVTX windows do not overlap: {no_overlap} of {len(spreads)} (host forward is a sub-ms graph launch; GPU-window overlap is checked from the tables)")
    nonmono = sum(1 for d, s in starts_by_dev.items() if any(b <= a for a, b in zip(s, s[1:])))
    rep.add("PASS" if nonmono == 0 else "FAIL", f"devices whose iteration start sequence is not strictly increasing: {nonmono}")
    if spreads.size:
        rep.add("INFO", f"forward start spread across devices: median {np.median(spreads):.4f} ms, p90 {np.quantile(spreads, 0.9):.4f} ms, max {spreads.max():.4f} ms (host launch skew, reported)")
        rep.data["forward_start_spread_ms"] = {"median": float(np.median(spreads)), "p90": float(np.quantile(spreads, 0.9)), "max": float(spreads.max())}
    jit = [x["iteration"] for x in det[50:] if x["jit_module_loads"]]
    rep.add("PASS" if len(jit) <= 20 else "FAIL", f"JIT-affected iterations after warm-up: {len(jit)} (limit 20)")
    rep.data.update({"iterations": len(det), "bad_ranges": bad_rng, "bad_devices": bad_dev, "bad_index": bad_idx, "jit_iterations": jit})


def check_collectives(rep: Rep, tables: str, det: list, med_lim: float, p90_lim: float):
    costmodel_dir = (HERE.parent.parent.parent / "phase1a").resolve()   # profiling/env/arm64-gb300/dsv41-flash-tp4/phase1a
    sys.path.insert(0, str(costmodel_dir))
    try:
        from costmodel.data import PassData  # Phase 1A, frozen
    except Exception as exc:  # pragma: no cover
        rep.add("FAIL", f"cannot import Phase 1A costmodel.data from {costmodel_dir}: {exc}"); return
    pd = PassData.load(os.path.basename(tables.rstrip("/")), tables)
    rep.add("PASS" if len(pd.it["iteration"]) == len(det) else "FAIL", f"tables iterations {len(pd.it['iteration'])} == parsed iterations {len(det)}")
    # GPU-side per-rank iteration alignment on the single nsys timeline: for each iteration the four ranks' kernel windows
    # [first_start, last_end] must overlap (collectives synchronise them); the start spread is reported
    ir = pd.ir
    if len(ir["iteration"]) == 4 * len(pd.it["iteration"]) and (ir["iteration"].reshape(-1, 4) == ir["iteration"].reshape(-1, 4)[:, :1]).all():
        fs, le = ir["first_start_ns"].reshape(-1, 4).astype(np.int64), ir["last_end_ns"].reshape(-1, 4).astype(np.int64)
        gpu_no_overlap = int((fs.max(axis=1) >= le.min(axis=1)).sum())
        gspread = (fs.max(axis=1) - fs.min(axis=1)) / 1e6
        rep.add("PASS" if gpu_no_overlap == 0 else "FAIL", f"iterations whose four ranks' GPU kernel windows do not overlap: {gpu_no_overlap} of {len(fs)}")
        rep.add("INFO", f"GPU first-kernel start spread across ranks: median {np.median(gspread):.4f} ms, p90 {np.quantile(gspread, 0.9):.4f} ms, max {gspread.max():.4f} ms")
        rep.data["gpu_start_spread_ms"] = {"median": float(np.median(gspread)), "p90": float(np.quantile(gspread, 0.9)), "max": float(gspread.max())}
    else:
        rep.add("FAIL", "iter_rank table does not hold exactly 4 ranks per iteration")
    pure = (pd.it["prefill_tokens"] == 0) & (pd.it["excluded"] == 0)
    spread_by_b, K_by_b, n_unaligned = defaultdict(list), defaultdict(Counter), 0
    for it, b, ok in zip(pd.it["iteration"], pd.it["decode_requests"], pure):
        if not ok:
            continue
        c = pd.collectives(int(it))
        if c is None:
            n_unaligned += 1
            continue
        end = c["end_rel_ns"]                      # (R, K): end of collective k on rank r, relative to iteration t0
        spread_by_b[int(b)].extend(((end.max(axis=0) - end.min(axis=0)) / 1e6).tolist())
        K_by_b[int(b)][end.shape[1]] += 1
    rep.add("INFO", f"pure-decode iterations without rank-aligned collectives (unequal K across ranks): {n_unaligned} of {int(pure.sum())}")
    out = {}
    for b in sorted(spread_by_b):
        s = np.array(spread_by_b[b]); med, p90 = float(np.median(s)), float(np.quantile(s, 0.9))
        Kmode, nK = K_by_b[b].most_common(1)[0]
        other = sum(K_by_b[b].values()) - nK
        rep.add("PASS" if med <= med_lim and p90 <= p90_lim else "FAIL", f"collective end spread b={b}: median {med:.4f} ms (<= {med_lim}), p90 {p90:.4f} ms (<= {p90_lim}); n={len(s)}; modal K={Kmode} in {nK} iterations, n_other_K={other}")
        out[b] = {"median_ms": med, "p90_ms": p90, "n": int(len(s)), "K_mode": int(Kmode), "n_other_K": int(other)}
    if not out:
        rep.add("FAIL", "no pure-decode iteration with aligned collectives in the tables")
    rep.data["collective_end_spread"] = out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True); ap.add_argument("--tables"); ap.add_argument("--inputs"); ap.add_argument("--out")
    ap.add_argument("--spread-median-ms", type=float, default=0.02); ap.add_argument("--spread-p90-ms", type=float, default=0.05)
    ap.add_argument("--skip-routes", action="store_true")
    a = ap.parse_args(argv)
    rep = Rep()
    P = a.profile
    pj = os.path.join(P, "parsed.json")
    det, sa = [], {}
    if os.path.exists(pj):
        d = json.load(open(pj)); det, sa = d["iteration_details"], d.get("step_alignment", {})
    else:
        rep.add("FAIL", "parsed.json missing")
    check_export(rep, P, len(det) if det else None)
    if det:
        check_nvtx_alignment(rep, det, sa)
    if a.tables:
        if det:
            check_collectives(rep, a.tables, det, a.spread_median_ms, a.spread_p90_ms)
    else:
        rep.add("INFO", "no --tables: collective end spread not checked (run costmodel.extract first)")
    if not a.skip_routes:
        v2 = _load("e1_completeness_v2", HERE / "e1_completeness_v2.py")
        print("---- routes / join (checks/e1_completeness_v2.py)")
        sub = v2.run(P, a.inputs, parsed=False)
        rep.add("PASS" if sub.fails == 0 else "FAIL", f"completeness v2 on the same profile: {sub.fails} FAIL")
        rep.data["completeness_v2"] = sub.data.get("checks")
    rep.data["checks"] = [{"status": s, "msg": m} for s, m in rep.checks]
    rep.data["fails"] = rep.fails
    if a.out:
        json.dump(rep.data, open(a.out, "w"), indent=1, default=str)
    print("RESULT", "PASS" if rep.fails == 0 else f"FAIL ({rep.fails})")
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
