"""Phase 1A calibration driver: fits every component on calibration iterations only and writes
`coefficients.json` with provenance and a measured / inferred / not-identifiable classification of
every parameter group.

Usage:
  python -m costmodel.calibrate --tables <dir with <label>/ subdirs> --partition <dir> --corpus <token_corpus dir>
                                --out <dir> [--git-rev <sha>] [--seed 0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from . import __version__, groups as G
from .collectives import CollectiveModel
from .data import PassData
from .decode import DecodeModel
from .host import backpressure_evidence, decode_host_evidence, gap_table
from .longctx import PrefillComputeModel
from .mixed import MixedModel
from .moe import Corpus, MoeModel
from .partition import load as load_partition

CAL_PASSES = ("diverse_nsys_run2", "quadrant_nsys")


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hashes() -> dict:
    d = os.path.dirname(__file__)
    return {f: sha256_file(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".py")}


def table_provenance(tables_dir: str, labels) -> dict:
    out = {}
    for lab in labels:
        d = os.path.join(tables_dir, lab)
        meta = json.load(open(os.path.join(d, "meta.json")))
        out[lab] = dict(frozen_inputs={k: {"sha256": v["sha256"], "bytes": v["bytes"], "verified_against_SHA256SUMS": v.get("verified")} for k, v in meta["inputs"].items()},
                        tables={f: sha256_file(os.path.join(d, f)) for f in ("iterations.parquet", "iter_rank.parquet", "collectives.parquet", "meta.json")},
                        extract_sha256=meta["extract_sha256"], n_iterations=meta["n_iterations"], n_excluded=meta["n_excluded"])
    return out


def classification(decode: DecodeModel, coll: CollectiveModel, prefill: PrefillComputeModel, mixed: MixedModel, moe: MoeModel) -> list[dict]:
    rows = [
        ("1 decode", "per-batch per-rank compute windows (K+1 per b), first-kernel offsets", "measured", f"medians over {decode.n_fit} calibration iterations, {len(decode.table)} batch sizes"),
        ("1 decode", "per-call collective intrinsic cost (mhc / oneshot / twoshot / all-gather) per b", "measured", "min over ranks per call, median over iterations"),
        ("1 decode", "multi-stream overlap, intra-window idle per rank", "measured", "Σ durations − union; window span − union"),
        ("1 decode", "values at non-calibrated batch sizes", "inferred", "linear interpolation between neighbouring calibrated b of the same regime"),
        ("1 decode", "KV independence", "assumed within observed range", f"kv_max <= {decode.kv_max_observed}; tested in evaluate (residual vs kv)"),
        ("1 decode", "stream assignment of overlapping kernels", "not identifiable", "graph-executor stream ids are not program streams (gap G6); overlap carried as a measured term"),
        ("2 mixed", "launch count L(P, b, KV)", "measured", f"linear fit, rmse {mixed.launch.get('rmse', float('nan')):.1f} launches"),
        ("2 mixed", "per-rank per-launch host cost c_r", "measured", f"{len(mixed.host_bound_ids)} host-bound eager calibration iterations"),
        ("2 mixed", "per-launch CUDA API cost c_api (unthrottled)", "measured", ""),
        ("2 mixed", "GPU tail after the last launch", "measured", ""),
        ("2 mixed", "cost of KV-dependent launches c_kv", mixed.c_kv_evidence.get("status", "n/a").split(" (")[0], mixed.c_kv_evidence.get("status", "")),
        ("2 mixed", "graph-path mixed iterations (P+b <= 256)", "measured class medians only", f"{len(mixed.graph_table)} (P,b) classes"),
        ("2 mixed", "cause of rank-2 host slowness", "not identifiable", "gap G4"),
        ("3 moe", "E(b) distinct experts per layer", "measured (corpus pass)", f"{moe.corpus_info.get('calibration_tokens')} calibration tokens"),
        ("3 moe", "α, β, γ of t_moe(b) = α + β·E(b) + γ·b", "inferred", f"fit through batch-level kernel medians, R² {moe.r2:.3f}"),
        ("3 moe", "per-iteration routing", "not identifiable", "not captured in timing passes (gap G3)"),
        ("4 prefill/longctx", "per-group per-rank compute vs (T, Σt², KV, b)", "measured", f"{prefill.n_fit} calibration mixed iterations"),
        ("4 prefill/longctx", "indexer / sparse-metadata KV slope", "measured", f"{prefill.chunk.get('n')} calibration 16k chunks, kv0 <= {prefill.chunk.get('kv0_max')}"),
        ("4 prefill/longctx", "chunk-length × KV interaction (t != 16k at kv0 > 0)", "inferred structure", "gap G1: only a handful of multi-chunk calibration rows"),
        ("4 prefill/longctx", "kv0 beyond the deepest calibration chunk", "extrapolation (flagged)", "no silent prediction"),
        ("5 collectives", "eager/ring intrinsic per call vs payload tokens", "measured", ", ".join(f"{k}: n={v['n']}" for k, v in coll.eager.items())),
        ("5 collectives", "payload bytes", "inferred", "tokens × hidden (not in trace)"),
        ("5 collectives", "waiting time", "derived", "composition (rank skew) or host model; never a collective parameter"),
        ("6 composition", "completion = max_r arrival + intrinsic", "verified", "reconstruction error reported in evaluate"),
        ("7 host", "inter-iteration GPU gap", "measured", "≈ 0 in all classes (GPU-saturated, host runs ahead)"),
        ("7 host", "rank-specific decode launch lateness (b <= 2)", "measured (effect)", "cause inferred: rank-0 sampling/bookkeeping"),
        ("7 host", "back-pressure inflation of launch API time", "measured", "evidence table"),
    ]
    return [dict(component=c, parameter=p, status=s, evidence=e) for c, p, s, e in rows]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--git-rev", default="unknown")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    part = load_partition(a.partition)
    passes = [PassData.load(lab, os.path.join(a.tables, lab), part) for lab in CAL_PASSES]
    print("passes loaded", {p.label: len(p.feats) for p in passes}, f"{time.time()-t0:.0f}s", flush=True)
    decode = DecodeModel().fit(passes, part)
    print(f"decode: {decode.n_fit} iterations, batch sizes {sorted(decode.table)}", flush=True)
    coll = CollectiveModel().fit(passes, part)
    prefill = PrefillComputeModel().fit(passes, part)
    mixed = MixedModel().fit(passes, part)
    corpus = Corpus(a.corpus)
    moe = MoeModel().fit(corpus, part.part["moe_calibration_requests"], part.part["moe_heldout_requests"], decode.table, seed=a.seed)
    print(f"collectives/prefill/mixed/moe fitted {time.time()-t0:.0f}s", flush=True)
    host = dict(gap_ms=gap_table(passes, part), decode=decode_host_evidence(passes, part), backpressure=backpressure_evidence(passes, part, mixed.c_api_ns))
    coeffs = dict(
        schema_version=1, costmodel_version=__version__, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), git_rev=a.git_rev, seed=a.seed,
        provenance=dict(tables=table_provenance(a.tables, CAL_PASSES), partition_sha256=part.part["_sha256"], code=code_hashes(),
                        corpus_manifest=corpus.manifest, calibration_passes=list(CAL_PASSES)),
        groups=list(G.GROUPS), comm_kinds=list(G.COMM_KINDS),
        decode=decode.to_dict(), collectives=coll.to_dict(), prefill=prefill.to_dict(), mixed=mixed.to_dict(), moe=moe.to_dict(), host=host,
        classification=classification(decode, coll, prefill, mixed, moe),
        decode_summary=decode.summary_rows(),
    )
    p = os.path.join(a.out, "coefficients.json")
    with open(p, "w") as f:
        json.dump(coeffs, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o)))
    with open(os.path.join(a.out, "COEFFICIENTS.sha256"), "w") as f:
        f.write(f"{sha256_file(p)}  coefficients.json\n")
    print(f"wrote {p} ({os.path.getsize(p)/1e6:.1f} MB) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
