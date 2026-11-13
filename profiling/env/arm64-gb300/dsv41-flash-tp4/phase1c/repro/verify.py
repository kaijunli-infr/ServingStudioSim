"""Regenerate the Phase 1B outputs from the frozen inputs and verify them against repro/MANIFEST.json.

Steps: (1) input checksums (tables, coefficients, partition, sources); (2) full regeneration into a scratch
directory with the manifest's command, seed and draws; (3) canonical content checksums of every parquet output,
params.json sha256, and an exact comparison of every reported metric in results.json["passes"].

Usage (tray14, gb300 venv): python repro/verify.py --tables <tables> [--work /tmp/phase1b-verify]
Exit code 0 = everything matches.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (MODES, P1A, P1B, P1C, PASSES, TABLE_FILES, canonical_parquet_checksum, environment,  # noqa: E402
                    sha256_file, source_hashes)


def walk(a, b, path, diffs, tol):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in ("seconds",):
                continue
            if k not in a or k not in b:
                diffs.append((path + "/" + str(k), "missing"))
            else:
                walk(a[k], b[k], path + "/" + str(k), diffs, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path, f"len {len(a)} != {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", diffs, tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if (math.isnan(a) and math.isnan(b)):
            return
        if abs(a - b) > tol * max(1.0, abs(a)):
            diffs.append((path, f"{a} != {b}"))
    elif a != b:
        diffs.append((path, f"{a!r} != {b!r}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--work", default="/tmp/phase1b-verify")
    ap.add_argument("--manifest", default=os.path.join(P1C, "repro", "MANIFEST.json"))
    ap.add_argument("--tol", type=float, default=0.0, help="relative tolerance on metrics (0 = exact)")
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    ok = True
    report = {"environment": environment(), "manifest_environment": man["environment"], "checks": {}}

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        report["checks"][name] = {"ok": bool(cond), "detail": detail}
        print(f"[{'ok' if cond else 'FAIL'}] {name} {detail}", flush=True)

    # 1. inputs
    check("coefficients.sha256", sha256_file(os.path.join(P1A, "out", "coefficients.json")) == man["inputs"]["coefficients"]["sha256"])
    check("partition.sha256", sha256_file(os.path.join(P1A, "partition", "partition.json")) == man["inputs"]["partition"]["sha256"])
    src = source_hashes()
    bad = [k for k, v in man["sources"].items() if src.get(k) != v]
    check("sources unchanged", not bad, ",".join(bad))
    for lab in PASSES:
        for fn in TABLE_FILES:
            p = os.path.join(a.tables, lab, fn)
            exp = man["inputs"]["tables"]["passes"][lab]["files"][fn]["sha256"]
            check(f"table {lab}/{fn}", os.path.exists(p) and sha256_file(p) == exp)
    # 2. regenerate
    if os.path.exists(a.work):
        shutil.rmtree(a.work)
    os.makedirs(a.work)
    cmd = [sys.executable, "-m", "seqsim.run", "--tables", a.tables, "--partition", os.path.join(P1A, "partition"), "--coeffs", os.path.join(P1A, "out", "coefficients.json"),
           "--out", a.work, "--draws", str(man["draws"]), "--seed", str(man["seed"]), "--git-rev", man["produced_by_commit"]]
    t = time.time()
    r = subprocess.run(cmd, cwd=P1B, capture_output=True, text=True)
    check("regeneration ran", r.returncode == 0, f"{time.time() - t:.0f}s" if r.returncode == 0 else r.stderr[-2000:])
    if r.returncode != 0:
        json.dump(report, open(os.path.join(P1C, "repro", "VERIFY.json"), "w"), indent=1)
        sys.exit(1)
    # 3. outputs
    check("params.json sha256", sha256_file(os.path.join(a.work, "params.json")) == man["inputs"]["params"]["sha256"])
    for lab in PASSES:
        for mode in MODES:
            fn = f"iterations_{lab}_{mode}.parquet"
            c = canonical_parquet_checksum(os.path.join(a.work, fn))
            exp = man["outputs"][fn]
            check(f"canonical {fn}", c["canonical_sha256"] == exp["canonical_sha256"] and c["rows"] == exp["rows"] and c["columns"] == exp["columns"],
                  f"rows {c['rows']}")
    new = json.load(open(os.path.join(a.work, "results.json")))
    ref = json.load(open(os.path.join(P1B, "out", "results.json")))
    diffs = []
    walk(ref["passes"], new["passes"], "passes", diffs, a.tol)
    check("results.json metrics identical", not diffs, f"{len(diffs)} differences" + (f"; first: {diffs[0]}" if diffs else ""))
    report["metric_differences"] = diffs[:50]
    report["ok"] = ok
    report["work_dir"] = a.work
    json.dump(report, open(os.path.join(P1C, "repro", "VERIFY.json"), "w"), indent=1)
    print("VERIFY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
