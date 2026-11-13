"""CLI: calibrate the sequence parameters (calibration role only) and evaluate all modes.

  python -m seqsim.run --tables <dir> --partition <dir> --coeffs <phase1a coefficients.json> --out <dir> [--draws 32] [--seed 0]
                       [--passes diverse_nsys_run2,quadrant_nsys,diverse_nsys_salvaged] [--modes recon-gpu,recon-seq,pred-det,pred-mc]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

from costmodel import partition as part
from costmodel.data import PassData

from . import __version__
from .evaluate import evaluate_pass, write_markdown
from .params import SeqParams

ALL_PASSES = ("diverse_nsys_run2", "quadrant_nsys", "diverse_nsys_salvaged")
ALL_MODES = ("recon-gpu", "recon-seq", "pred-det", "pred-mc")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--coeffs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passes", default=",".join(ALL_PASSES))
    ap.add_argument("--modes", default=",".join(ALL_MODES))
    ap.add_argument("--params", default=None, help="reuse an existing params.json instead of fitting")
    ap.add_argument("--git-rev", default="unknown")
    ap.add_argument("--progress", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    partition = part.load(a.partition)
    coeffs = json.load(open(a.coeffs))
    labels = [x for x in a.passes.split(",") if x]
    passes = [PassData.load(lab, os.path.join(a.tables, lab), partition) for lab in labels]
    print(f"loaded {len(passes)} passes in {time.time() - t0:.0f}s", flush=True)
    if a.params:
        params = SeqParams.load(a.params)
    else:
        params = SeqParams.fit(passes, partition, coeffs)
        params.save(os.path.join(a.out, "params.json"))
    ppath = os.path.join(a.out, "params.json")
    if not os.path.exists(ppath):
        params.save(ppath)
    psha = sha256_file(ppath)
    open(os.path.join(a.out, "PARAMS.sha256"), "w").write(f"{psha}  params.json\n")
    print(f"params: Q={params.queue_depth:.0f} a_host={[round(x/1e6,3) for x in params.a_host_ns]} f_idle={params.f_idle_ns/1e6:.3f} ({time.time() - t0:.0f}s)", flush=True)
    results = dict(seqsim_version=__version__, git_rev=a.git_rev, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seed=a.seed, draws=a.draws,
                   params_sha256=psha, coefficients_sha256=sha256_file(a.coeffs), partition_sha256=partition.sha256 if hasattr(partition, "sha256") else open(os.path.join(a.partition, "PARTITION.sha256")).read().split()[0],
                   code={f: sha256_file(os.path.join(os.path.dirname(__file__), f)) for f in sorted(os.listdir(os.path.dirname(__file__))) if f.endswith(".py")},
                   passes={})
    modes = [m for m in a.modes.split(",") if m]
    for pd in passes:
        print(f"== {pd.label}", flush=True)
        results["passes"][pd.label] = evaluate_pass(pd, coeffs, params, modes, a.draws, a.seed, a.out, a.progress)
        for mode, R in results["passes"][pd.label]["modes"].items():
            w = R["workload"]
            print(f"   {mode}: {R['seconds']} s, span err {w['span_err_pct']:+.2f} %, Σ all {w['sums']['all']['signed_bias_pct']:+.2f} %", flush=True)
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
    write_markdown(results, params, os.path.join(a.out, "RESULTS.md"))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
