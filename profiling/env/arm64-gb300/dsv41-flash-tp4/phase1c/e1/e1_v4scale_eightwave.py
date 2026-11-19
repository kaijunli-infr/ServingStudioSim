#!/usr/bin/env python3
"""Prospective check that the scientific completeness and routing-record checker handles the eight-wave V4 trace (CPU-only).

    <python with numpy> e1_v4scale_eightwave.py --v2-profile <V2 run>/profile --out DIR [--keep-work]

No eight-wave capture exists yet. This builds one synthetically from the accepted V2 capture of the same 56 requests: the
iteration records, the per-request routing arrays and the routing corpus are taken as waves `w0`, and three further copies
of each are added as waves `w1`, `w2`, `w3` (request ids renamed, iteration indices and corpus offsets shifted, manifest
and checksum recomputed). The result has the structure the V4 trace must produce: 224 requests, eight waves of two cell
kinds. It is a structural fixture, not measured data, and its repeated waves are copies, not independent observations.
checks/e1_completeness_v2.py is then run on it, unchanged, through its command line with E1-V4-SCALE-INPUTS.json
(`--plan dry`, `--checksum`), on the positive fixture and on registered defects:

  eight_waves                expect exit 0, the eight cells each PASS
  v3_four_waves_only         two copies only (the V3 structure)           expect exit 1 (planned cells of waves 2, 3 absent)
  v2_two_waves_only          the V2 profile as it is                      expect exit 1
  wave3_b8_missing           wave 3 of the b=8 cell removed               expect exit 1
  wave2_b48_missing          wave 2 of the b=48 cell removed              expect exit 1
  routing_file_missing       one routing array of wave 3 removed          expect exit 1
  routing_array_altered      one expert id changed in one wave-2 array    expect exit 1 (array != corpus segment)
  corpus_checksum_wrong      manifest checksum off by one                 expect exit 1
  waves_overlap              wave-3 b=48 requests inside wave-3 b=8 iterations   expect exit 1 (two cells in one iteration)
  request_short_by_one_token one wave-3 request's last iteration removed  expect exit 1

Read-only on the V2 profile. Exit 0 only if every expectation holds.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

E1 = Path(__file__).resolve().parent
CHECKER = E1 / "checks" / "e1_completeness_v2.py"
INPUTS = E1 / "E1-V4-SCALE-INPUTS.json"
SHIFT = 10_000  # iteration-index offset between consecutive copies
COPIES = 4


def _checker():
    spec = importlib.util.spec_from_file_location("e1_completeness_v2", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # the checker defines dataclasses, which look their module up by name
    spec.loader.exec_module(mod)
    return mod


def wn(text: str, index: int) -> str:
    return text.replace("_w0_r", f"_w{index}_r")


def build(v2: Path, dst: Path, chk, *, copies: int = COPIES) -> None:
    """A profile directory with the files the checker reads: the V2 capture as wave index 0 plus `copies - 1` renamed copies."""
    (dst / "vllm").mkdir(parents=True)
    (dst / "routed_experts").mkdir()
    (dst / "token_corpus").mkdir()
    metrics = next((v2 / "vllm").glob("*_metrics.jsonl"))
    recs = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    out = list(recs)
    for index in range(1, copies):
        for r in recs:
            c = json.loads(wn(json.dumps(r), index))
            c["iteration_index"] += index * SHIFT
            out.append(c)
    (dst / "vllm" / "fixture_metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    for f in sorted((v2 / "routed_experts").glob("*.npy")):
        shutil.copyfile(f, dst / "routed_experts" / f.name)
        for index in range(1, copies):
            shutil.copyfile(f, dst / "routed_experts" / wn(f.name, index))
    man = json.loads((v2 / "token_corpus" / "manifest.json").read_text())
    prov = json.loads((v2 / "token_corpus" / "provenance.json").read_text())
    data = (v2 / "token_corpus" / man["data_file"]).read_bytes()
    if copies > 1:
        n, base = man["num_tokens"], prov["request_segments"]
        prov["request_segments"] = base + [{**s, "request": wn(s["request"], index), "start": s["start"] + index * n} for index in range(1, copies) for s in base]
        man["num_tokens"] = copies * n
        data = data * copies
        man["checksum_fnv1a64"] = chk.fnv1a64(data)
    (dst / "token_corpus" / man["data_file"]).write_bytes(data)
    (dst / "token_corpus" / "manifest.json").write_text(json.dumps(man))
    (dst / "token_corpus" / "provenance.json").write_text(json.dumps(prov))
    pr = json.loads((v2 / "profile_result.json").read_text())
    k = copies
    pr["validation"] = {**pr["validation"], "nvtx_forward_ranges": k * pr["validation"]["nvtx_forward_ranges"], "kernel_rows": k * pr["validation"]["kernel_rows"]}
    pr["request_timing_count"] = k * pr["request_timing_count"]
    pr["fixture_note"] = "synthetic eight-wave structure built from the accepted V2 capture; not measured data; the repeated waves are copies"
    (dst / "profile_result.json").write_text(json.dumps(pr))


def rewrite_metrics(profile: Path, fn) -> None:
    p = profile / "vllm" / "fixture_metrics.jsonl"
    recs = fn([json.loads(line) for line in p.read_text().splitlines() if line.strip()])
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))


def drop_requests(profile: Path, pred) -> None:
    """Remove the requests matching `pred` everywhere: iteration records, routing arrays, corpus segments."""

    def fn(recs):
        out = []
        for r in recs:
            prog = [p for p in r.get("decode_request_progress") or [] if not pred(p["request_id"])]
            if r.get("decode_request_progress") and not prog:
                continue
            out.append({**r, "decode_request_progress": prog})
        return out

    rewrite_metrics(profile, fn)
    for f in list((profile / "routed_experts").glob("*.npy")):
        if pred(f.stem):
            f.unlink()


def variants(chk) -> list:
    def wave3_b8_missing(p):
        drop_requests(p, lambda rid: "_b8_" in rid and "_w3_r" in rid)

    def wave2_b48_missing(p):
        drop_requests(p, lambda rid: "_b48_" in rid and "_w2_r" in rid)

    def routing_file_missing(p):
        next(f for f in sorted((p / "routed_experts").glob("*_w3_r*.npy"))).unlink()

    def routing_array_altered(p):
        f = sorted((p / "routed_experts").glob("*_w2_r*.npy"))[3]
        a = np.load(f)
        a[500, 7, 0] = (int(a[500, 7, 0]) + 1) % 384
        while len(set(a[500, 7].tolist())) < a.shape[2]:
            a[500, 7, 0] = (int(a[500, 7, 0]) + 1) % 384
        np.save(f, a)

    def corpus_checksum_wrong(p):
        m = json.loads((p / "token_corpus" / "manifest.json").read_text())
        m["checksum_fnv1a64"] += 1
        (p / "token_corpus" / "manifest.json").write_text(json.dumps(m))

    def waves_overlap(p):
        def fn(recs):
            b8 = [r for r in recs if any("_b8_" in x["request_id"] and "_w3_r" in x["request_id"] for x in r.get("decode_request_progress") or [])]
            b48 = [r for r in recs if any("_b48_" in x["request_id"] and "_w3_r" in x["request_id"] for x in r.get("decode_request_progress") or [])]
            for a, b in zip(b8[100:140], b48[100:140]):
                a["decode_request_progress"] = a["decode_request_progress"] + b["decode_request_progress"][:1]
            return recs

        rewrite_metrics(p, fn)

    def request_short_by_one_token(p):
        def fn(recs):
            target = next(x["request_id"] for r in recs for x in r.get("decode_request_progress") or [] if "_b48_" in x["request_id"] and "_w3_r" in x["request_id"])
            last = max(i for i, r in enumerate(recs) if any(x["request_id"] == target and x["emitted_tokens"] == 1 for x in r.get("decode_request_progress") or []))
            recs[last]["decode_request_progress"] = [x for x in recs[last]["decode_request_progress"] if x["request_id"] != target]
            return recs

        rewrite_metrics(p, fn)

    return [("eight_waves", COPIES, None, 0), ("v3_four_waves_only", 2, None, 1), ("v2_two_waves_only", 1, None, 1), ("wave3_b8_missing", COPIES, wave3_b8_missing, 1), ("wave2_b48_missing", COPIES, wave2_b48_missing, 1),
            ("routing_file_missing", COPIES, routing_file_missing, 1), ("routing_array_altered", COPIES, routing_array_altered, 1), ("corpus_checksum_wrong", COPIES, corpus_checksum_wrong, 1), ("waves_overlap", COPIES, waves_overlap, 1), ("request_short_by_one_token", COPIES, request_short_by_one_token, 1)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2-profile", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args(argv)
    chk = _checker()
    args.out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="v4scale-eightwave-"))
    rows = []
    try:
        for name, copies, mutate, want in variants(chk):
            prof = work / name
            build(args.v2_profile, prof, chk, copies=copies)
            if mutate:
                mutate(prof)
            p = subprocess.run([sys.executable, str(CHECKER), "--profile", str(prof), "--inputs", str(INPUTS), "--plan", "dry", "--checksum", "--out", str(args.out / f"{name}.json")], capture_output=True, text=True)
            (args.out / f"{name}.txt").write_text(p.stdout + (("\n--- stderr\n" + p.stderr) if p.stderr.strip() else ""))
            fails = [line[:170] for line in p.stdout.splitlines() if line.startswith("FAIL")]
            cells = [line.split("cell ")[1][:60] for line in p.stdout.splitlines() if line.startswith("PASS") and "cell b=" in line]
            rows.append({"fixture": name, "expected_exit": want, "exit": p.returncode, "met": p.returncode == want and (want == 1) == bool(fails) and (want == 1 or len(cells) == 2 * COPIES), "cells_passing": cells, "fail_lines": fails})
            shutil.rmtree(prof, ignore_errors=True)
    finally:
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)
    met = sum(r["met"] for r in rows)
    lines = [f"# eight-wave completeness and routing-record check through {CHECKER.name} (synthetic structure from the V2 capture; CPU-only) {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", f"# {len(rows)} fixtures; expectations met {met} of {len(rows)}"]
    for r in rows:
        lines.append(f"{'MET    ' if r['met'] else 'NOT MET'} {r['fixture']:28s} exit {r['exit']} (expected {r['expected_exit']}); cells passing {len(r['cells_passing'])}: {r['cells_passing']}; FAIL lines {len(r['fail_lines'])}: {r['fail_lines'][:3]}")
    (args.out / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    (args.out / "SUMMARY.json").write_text(json.dumps({"fixtures": len(rows), "met": met, "rows": rows}, indent=1) + "\n")
    print("\n".join(lines))
    return 0 if met == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
