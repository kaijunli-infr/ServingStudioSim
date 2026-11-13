"""Shared helpers for the Phase 1B reproducibility manifest and verifier (CPU-only)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
P1C = os.path.abspath(os.path.join(HERE, ".."))
P1B = os.path.abspath(os.path.join(P1C, "..", "phase1b"))
P1A = os.path.abspath(os.path.join(P1C, "..", "phase1a"))
PASSES = ["diverse_nsys_run2", "quadrant_nsys", "diverse_nsys_salvaged"]
MODES = ["recon-gpu", "recon-seq", "pred-det", "pred-mc"]
TABLE_FILES = ["collectives.parquet", "iter_rank.parquet", "iterations.parquet", "meta.json"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_parquet_checksum(path: str) -> dict:
    """Content checksum independent of the parquet encoder: rows sorted by `iteration`, columns sorted by name,
    floats serialised with repr (round-trip exact), NaN as the string 'NaN'. Returns rows, columns and the digest."""
    t = pq.read_table(path)
    d = t.to_pydict()
    cols = sorted(d)
    order = np.argsort(np.asarray(d["iteration"]), kind="stable") if "iteration" in d else np.arange(t.num_rows)

    def canon(v):
        if isinstance(v, float):
            return "NaN" if v != v else repr(v)
        if isinstance(v, (np.floating,)):
            return canon(float(v))
        if isinstance(v, (np.integer,)):
            return int(v)
        return v

    h = hashlib.sha256()
    for i in order:
        h.update(json.dumps([canon(d[c][i]) for c in cols], sort_keys=True, separators=(",", ":")).encode())
    schema = {f.name: str(f.type) for f in t.schema}
    return dict(rows=t.num_rows, columns=schema, canonical_sha256=h.hexdigest())


def environment() -> dict:
    return dict(python=sys.version.split()[0], numpy=np.__version__, pyarrow=pa.__version__, platform=platform.platform(),
                machine=platform.machine())


def source_hashes() -> dict:
    out = {}
    for base, sub in ((P1B, "seqsim"), (P1A, "costmodel")):
        d = os.path.join(base, sub)
        for n in sorted(os.listdir(d)):
            if n.endswith(".py"):
                out[f"{sub}/{n}"] = sha256_file(os.path.join(d, n))
    return out
