"""Sensitivity of the Phase 1B predictions to the two inferred parameters (queue throttle, delta_launch).

CPU-only; reads the Phase 1A tables and the fitted params.json, runs run2 pred-det under variants and prints
Σ walls by kind, chain-lead medians and per-class bias deltas. Usage (tray14, gb300 venv):
    python -m tools.sensitivity --tables <tables> --partition ../phase1a/partition --coeffs ../phase1a/out/coefficients.json --params out/params.json
"""
import argparse, copy, json, os, time
import numpy as np
import seqsim  # noqa: F401  (path setup)
from costmodel.data import PassData
from costmodel import partition as part
from seqsim.engine import Engine, measured_from_pass
from seqsim.evaluate import rows_from, table, lead_table, workload
from seqsim.params import SeqParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True); ap.add_argument("--partition", required=True)
    ap.add_argument("--coeffs", required=True); ap.add_argument("--params", required=True)
    ap.add_argument("--pass", dest="label", default="diverse_nsys_run2")
    a = ap.parse_args()
    partition = part.load(a.partition)
    coeffs = json.load(open(a.coeffs))
    pd = PassData.load(a.label, os.path.join(a.tables, a.label), partition)
    base = SeqParams.load(a.params)
    sched = sorted(pd.feats, key=lambda f: f.iteration)
    measured = {f.iteration: m for f in sched if (m := measured_from_pass(pd, f.iteration)) is not None}
    sched = [f for f in sched if f.iteration in measured]
    variants = {
        "baseline": dict(params=base, throttle=True),
        "no_queue_throttle": dict(params=base, throttle=False),
        "delta_launch_0us": dict(params=copy.deepcopy(base), throttle=True),
        "delta_launch_40us": dict(params=copy.deepcopy(base), throttle=True),
    }
    variants["delta_launch_0us"]["params"].delta_launch_ns = 0.0
    variants["delta_launch_40us"]["params"].delta_launch_ns = 40_000.0
    out = {}
    for name, v in variants.items():
        t = time.time()
        eng = Engine(coeffs, v["params"], draws=1, seed=0, stochastic=False, moe_band=False, throttle=v["throttle"])
        recs = eng.run(sched, measured=measured, recon=None)
        rows = rows_from(pd, recs)
        ev = [r for r in rows if not r["excluded"]]
        w = workload(rows, pd)
        cls = {tuple(x["key"]) if isinstance(x["key"], list) else x["key"]: x for x in table(ev, lambda r: (r["role"], r["cls"]))}
        lead = {tuple(x["key"]) if isinstance(x["key"], list) else x["key"]: x for x in lead_table(rows)}
        out[name] = dict(seconds=round(time.time() - t, 1), workload=w, cls=cls, lead=lead)
        print(f"[{name}] {out[name]['seconds']} s  span_pred_s={w.get('span_pred_s', w.get('span', {}))}", flush=True)
    b = out["baseline"]
    print("\n== Σ walls / span (variant minus baseline)")
    for name, o in out.items():
        print(name, json.dumps({k: o["workload"][k] for k in o["workload"] if not isinstance(o["workload"][k], (dict, list))}, default=str)[:400])
    print("\n== per-class predicted-mean delta vs baseline (ms)")
    for k in b["cls"]:
        base_bias = b["cls"][k]["bias_ms"]
        print(k, {n: round(o["cls"][k]["bias_ms"] - base_bias, 4) for n, o in out.items() if n != "baseline"})
    print("\n== lead median delta vs baseline (ms), by transition")
    for k in b["lead"]:
        bl = b["lead"][k]["pred_lead_median_ms"]
        print(k, "n=%d" % b["lead"][k]["n"], {n: round(o["lead"][k]["pred_lead_median_ms"] - bl, 3) for n, o in out.items() if n != "baseline"})


if __name__ == "__main__":
    main()
