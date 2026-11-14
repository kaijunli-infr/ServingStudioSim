#!/usr/bin/env python3
"""Write phase1c/partition/partition-1c.json for a Phase 1C capture (protocol section 3), after the capture.

    python3 make_partition_1c.py --tables <pass table dir> --label e1_main --designation identification
                                 [--excl jit_excl.txt] --out ../partition/partition-1c.json

Every non-excluded iteration of the capture receives the capture's pre-registered designation
(E1: `identification`; E2: `validation`); excluded iterations (JIT rule, kind mismatch) get `excluded`.
The file has the Phase 1A `passes.<label>.{role, cls, exclusions, counts}` layout so the frozen
`costmodel.partition.Partition` leakage guard (`calibration_only` refuses fits on rows whose role is not the
fitting role) applies unchanged: identification rows are the only rows a Phase 1C fit may read, and a
validation capture's rows can never enter a fit. The Phase 1A partition file is not touched: this file
is additive, and `tests/test_invariants.py` pins the Phase 1A partition hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import pyarrow.parquet as pq

ROLE_FOR = {"identification": "identification", "validation": "validation"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--designation", required=True, choices=sorted(ROLE_FOR)); ap.add_argument("--excl")
    ap.add_argument("--out", required=True); ap.add_argument("--protocol-sha", default="07ab7d7171ae3185d44bcc92c5a5b6ef0f03c3bb8d82eec27461d429e80327be")
    a = ap.parse_args()
    it = pq.read_table(os.path.join(a.tables, "iterations.parquet")).to_pydict()
    excl = set()
    if a.excl:
        excl = {int(x) for x in open(a.excl).read().replace("\n", ",").split(",") if x.strip()}
    role, cls = {}, {}
    for i, iteration in enumerate(it["iteration"]):
        iteration = int(iteration)
        if it["excluded"][i] or iteration in excl:
            role[str(iteration)] = "excluded"
        else:
            role[str(iteration)] = ROLE_FOR[a.designation]
        b, pt = int(it["decode_requests"][i]), int(it["prefill_tokens"][i])
        cls[str(iteration)] = f"decode nreq={b}" if pt == 0 and b else (f"prefill chunk {pt}" if not b else f"mixed P={pt} b={b}")
    part = json.load(open(a.out)) if os.path.exists(a.out) else {"schema_version": 1, "protocol_sha256": a.protocol_sha, "rule": "role = the capture's pre-registered designation (frozen matrix); never changed after the capture", "passes": {}}
    if a.label in part["passes"]:
        sys.exit(f"{a.label} already in {a.out}: a designation is assigned once and never rewritten")
    meta = json.load(open(os.path.join(a.tables, "meta.json")))
    part["passes"][a.label] = {"designation": a.designation, "role": role, "cls": cls, "exclusions": sorted(excl), "counts": dict(Counter(role.values())),
                               "tables_meta_inputs": {k: v.get("sha256") for k, v in meta.get("inputs", {}).items()}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(part, open(a.out, "w"), indent=1, sort_keys=True)
    print(a.label, dict(Counter(role.values())), "sha256", hashlib.sha256(open(a.out, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
