"""Write repro/MANIFEST.json for the frozen Phase 1B run: inputs (with checksums), environment, exact command, seed,
output schema, row counts and canonical content checksums of the per-iteration parquet files.

Usage (tray14, gb300 venv):  python repro/make_manifest.py --tables <dir with the three pass table dirs> [--out-dir ../phase1b/out]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (MODES, P1A, P1B, P1C, PASSES, TABLE_FILES, canonical_parquet_checksum, environment,  # noqa: E402
                    sha256_file, source_hashes)


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=P1C, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out-dir", default=os.path.join(P1B, "out"))
    ap.add_argument("--mirror", default="rack3:/mnt/shared/home/kevin/dsv41-phase1-mirror (SHA256SUMS.mirror)")
    a = ap.parse_args()
    results = json.load(open(os.path.join(a.out_dir, "results.json")))
    man = dict(
        schema_version=1,
        description="Reproducibility manifest for the Phase 1B sequence-simulation outputs (branch phase1b-seqsim, commit 29082793).",
        produced_by_commit=results["git_rev"], created=results["created"], seed=results["seed"], draws=results["draws"],
        environment=environment(), git_head=git("rev-parse", "HEAD"),
        command=("cd profiling/env/arm64-gb300/dsv41-flash-tp4/phase1b && $PY -m seqsim.run --tables <TABLES> --partition ../phase1a/partition "
                 "--coeffs ../phase1a/out/coefficients.json --out out --draws 32 --seed 0 --git-rev $(git rev-parse HEAD)"),
        inputs={}, sources=source_hashes(), outputs={}, provenance={}, second_location=a.mirror,
    )
    # frozen inputs
    for name, path in (("coefficients", os.path.join(P1A, "out", "coefficients.json")), ("partition", os.path.join(P1A, "partition", "partition.json")),
                       ("params", os.path.join(a.out_dir, "params.json"))):
        man["inputs"][name] = dict(path=os.path.relpath(path, P1C), sha256=sha256_file(path), bytes=os.path.getsize(path))
    assert man["inputs"]["coefficients"]["sha256"] == results["coefficients_sha256"]
    assert man["inputs"]["partition"]["sha256"] == results["partition_sha256"]
    assert man["inputs"]["params"]["sha256"] == results["params_sha256"]
    # compact tables (not committed): per-file checksums and meta
    tables = {}
    for lab in PASSES:
        d = os.path.join(a.tables, lab)
        files = {}
        for fn in TABLE_FILES:
            p = os.path.join(d, fn)
            files[fn] = dict(sha256=sha256_file(p), bytes=os.path.getsize(p))
        meta = json.load(open(os.path.join(d, "meta.json")))
        tables[lab] = dict(files=files, meta=meta)
    man["inputs"]["tables"] = dict(root_on_tray14="/mnt/shared/home/dc/workspace/serving-studio/phase1a-tables", passes=tables,
                                   regenerate="python -m costmodel.extract (Phase 1A, see ../phase1a/README.md) from the Phase 0 captures; "
                                              "the extractor records the source database hashes in meta.json")
    # Phase 0 provenance committed in the repository
    prov = {}
    for p in sorted(glob.glob(os.path.join(P1C, "..", "phase0", "**", "SHA256SUMS*"), recursive=True)) + sorted(glob.glob(os.path.join(P1C, "..", "SHA256SUMS*"))):
        prov[os.path.relpath(p, P1C)] = sha256_file(p)
    man["provenance"] = dict(phase0_checksum_files=prov,
                             raw_captures="/mnt/shared/home/dc/workspace/serving-studio/artifacts/phase0-dsv41-flash-gb300-tp4-20261112/ (SHA256SUMS, MANIFEST.md)")
    # outputs
    for lab in PASSES:
        for mode in MODES:
            fn = f"iterations_{lab}_{mode}.parquet"
            man["outputs"][fn] = canonical_parquet_checksum(os.path.join(a.out_dir, fn))
    for fn in ("results.json", "RESULTS.md", "params.json", "PARAMS.sha256"):
        man["outputs"][fn] = dict(sha256=sha256_file(os.path.join(a.out_dir, fn)), bytes=os.path.getsize(os.path.join(a.out_dir, fn)))
    man["outputs"]["_expected_rows"] = {lab: results["passes"][lab]["modes"]["pred-det"]["n"] for lab in PASSES}
    man["outputs"]["_evaluated_rows"] = {lab: results["passes"][lab]["modes"]["pred-det"]["n_evaluated"] for lab in PASSES}
    out = os.path.join(P1C, "repro", "MANIFEST.json")
    json.dump(man, open(out, "w"), indent=1, sort_keys=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
