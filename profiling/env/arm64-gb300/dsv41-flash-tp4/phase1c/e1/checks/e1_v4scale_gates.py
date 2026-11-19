#!/usr/bin/env python3
"""Prospective acceptance gates for V4 (intermediate lifecycle scale, four copies of the accepted trace, normal stop). Written
before any V4 launch.

    python3 -I checks/e1_v4scale_gates.py --run <run root>/<pass> --inputs E1-V4-SCALE-INPUTS.json --pin V4S-PIN.json
        --provenance BINARY-PROVENANCE.json [--binary <session_runner>] --importer <QdstrmImporter> --nsys <nsys>
        --durable-stream <copy on durable storage> [--scratch DIR] --identity identity_v4scale.json
        --completeness v2_<pass>.json [--postrun-storage storage_postrun.jsonl] [--postrun-memory mem_postrun.jsonl]
        [--reservation-deleted reservation_deleted.json] [--extra name=rc ...] [--out V4S-GATES.json]

G0 - G12 are the V3 gates: the code of checks/e1_v3scale_gates.py is imported and run unchanged with the V4 inputs (phase
clocks, release interval, deadlines, ceilings and coverage rules are documented there). Every limit they judge is read
from the V4 inputs: stop accepted 30 s, flush 300 s, import 300 s, stop to report 600 s (the controller's hard deadline,
unextended), session close 300 s, GPU release upper edge 90 s, GPU-held 2700 s, reservation 90 min, temporary storage
15 GB, run directory 18 GB, MemAvailable floor 64 GiB, importer 128 GiB, sampling gaps 30 s.

Three further gates, V4-specific:
  G13  workload and recipe identity: the identity report must come from the V4 checker under the V4 policy (version
       v4scale-1), with the trace rule exercised and every one of its checks holding, and verdict PASS. A PASS report
       produced by another checker or policy is a FAIL (this is the V3 defect made a gate).
  G14  four-copy trace in the run: the trace and config the driver copied into the run directory are byte for byte the
       registered V4 trace and config; the trace is re-derived here from the accepted trace (own code) as exactly four
       copies with wave indices 0-3 and offsets of 80 s; 224 requests.
  G15  request, iteration, routing and rank coverage, from the completeness checker's record: the eight planned cells
       and no other, each with at least the pure-decode floor; 224 request timings = token-accounting requests = routing
       corpus segments; four ranks; NVTX forward ranges = 4 per iteration with the iteration count in its band; no
       duplicate, missing or non-monotone join; no routing-array defect; no failed check.

Verdict: FAIL if any gate FAILs; INCONCLUSIVE if none fails but one could not be evaluated; PASS otherwise.
Exit 0 / 1 / 3. Stdlib only; read-only on the run directory (scratch under --scratch).
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = ("G13.workload_and_recipe_identity", "G14.four_copy_trace_in_run", "G15.request_iteration_routing_rank_coverage")


def _mod(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V3 = _mod("e1_v3scale_gates")
add, load, verdict_of = V3.add, V3.load, V3.verdict_of
release_interval, reading_windows, coverage = V3.release_interval, V3.reading_windows, V3.coverage  # unchanged V3 helpers


def _sha(data: bytes | None) -> str | None:
    return None if data is None else hashlib.sha256(data).hexdigest()


def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def gate_identity(res: list, identity: dict | None, inputs: dict) -> None:
    reg = inputs.get("identity") or {}
    if identity is None:
        add(res, NEW[0], "INCONCLUSIVE", "no configuration-identity report")
        return
    ti = identity.get("trace_identity") or {}
    checks = ti.get("checks") or []
    wrong = []
    if identity.get("checker") != reg.get("checker"):
        wrong.append(f"report written by {identity.get('checker')!r}, not by the registered {reg.get('checker')!r}")
    if identity.get("policy_version") != reg.get("policy_version"):
        wrong.append(f"policy version {identity.get('policy_version')!r}, registered {reg.get('policy_version')!r}")
    failed = [c.get("check") for c in checks if not c.get("ok")]
    verdict = identity.get("verdict")
    if not wrong and verdict == "INCONCLUSIVE":
        add(res, NEW[0], "INCONCLUSIVE", f"identity report INCONCLUSIVE (a launch record or the profile config could not be read): {[r.get('reason') for r in identity.get('results', []) if r.get('status') == 'INCONCLUSIVE'][:3]}")
        return
    if ti.get("exercised") is not True:
        wrong.append("the derived-trace rule was not exercised")
    if failed:
        wrong.append(f"trace-identity checks not holding: {failed}")
    if ti.get("exercised") is True and (ti.get("expected_checks") is None or len(checks) != ti.get("expected_checks")):
        wrong.append(f"{len(checks)} trace-identity checks in the report, a complete proof has {ti.get('expected_checks')}")
    if verdict != "PASS":
        wrong.append(f"verdict {verdict}: {[(r.get('scope'), r.get('field')) for r in identity.get('results', []) if r.get('status') == 'FAIL'][:6]}")
    add(res, NEW[0], "FAIL" if wrong else "PASS", f"identity report by {identity.get('checker')} under policy {identity.get('policy_version')}: verdict {verdict}, {len(identity.get('normalized', []))} normalized field(s), trace rule exercised={ti.get('exercised')}, {len(checks) - len(failed)} of {len(checks)} trace checks hold" + (f"; {wrong}" if wrong else ""))


def four_copies(source: bytes, copies: int, offset_ms: float) -> bytes | None:
    """The accepted trace `copies` times (own derivation; shares no code with the identity checker or the generator)."""
    try:
        head, *rows = source.decode("ascii").rstrip("\n").split("\n")
        out = [head]
        for i in range(copies):
            for r in rows:
                rid, inp, outlen, arr = r.split(",")
                pre, sep, post = rid.partition("_w0_r")
                if not sep or "_w0_r" in post:
                    return None
                out.append(f"{pre}_w{i}_r{post},{inp},{outlen},{round(float(arr) + i * offset_ms, 1)}")
        return ("\n".join(out) + "\n").encode("ascii")
    except (UnicodeDecodeError, ValueError):
        return None


def gate_trace(res: list, run: Path, inputs: dict, e1: Path) -> None:
    tr = inputs.get("trace") or {}
    trace, config, source = _read(run / "trace.csv"), _read(run / "profile.json"), _read(e1 / (inputs.get("derived_from") or {}).get("trace", "missing"))
    if trace is None or config is None or source is None:
        add(res, NEW[1], "INCONCLUSIVE", f"not readable: {[n for n, v in (('run trace.csv', trace), ('run profile.json', config), ('accepted trace', source)) if v is None]}")
        return
    wrong = []
    if _sha(source) != (inputs.get("derived_from") or {}).get("trace_sha256"):
        wrong.append(f"the accepted trace on disk is {_sha(source)[:16]}…, registered {str((inputs.get('derived_from') or {}).get('trace_sha256'))[:16]}…")
    if _sha(trace) != tr.get("sha256"):
        wrong.append(f"run trace.csv sha256 {_sha(trace)[:16]}… is not the registered V4 trace {str(tr.get('sha256'))[:16]}…")
    want = four_copies(source, tr.get("copies", 0), tr.get("copy_offset_ms", 0.0))
    if want is None or want != trace:
        wrong.append(f"run trace.csv is not the accepted trace exactly {tr.get('copies')} times (wave indices 0-{tr.get('copies', 1) - 1}, {tr.get('copy_offset_ms')} ms apart)")
    rows = trace.decode("ascii", errors="replace").split("\n")[1:-1]
    if len(rows) != tr.get("requests") or len({r.split(",")[0] for r in rows}) != len(rows):
        wrong.append(f"{len(rows)} rows, {len({r.split(',')[0] for r in rows})} distinct request ids (registered {tr.get('requests')})")
    if _sha(config) != inputs.get("config_sha256"):
        wrong.append(f"run profile.json sha256 {_sha(config)[:16]}… is not the registered V4 config {str(inputs.get('config_sha256'))[:16]}…")
    try:
        path = json.loads(config)["workload"]["frontend"]["path"]
    except (ValueError, KeyError, TypeError):
        path = None
    at_path = _read(Path(path)) if path else None
    if not path or not str(path).endswith("/" + str(tr.get("path"))) or _sha(at_path) != tr.get("sha256"):
        wrong.append(f"the config's trace path {path!r} is not the registered trace file, or the file there is not the registered trace ({str(_sha(at_path))[:16]}…)")
    add(res, NEW[1], "FAIL" if wrong else "PASS", f"run trace.csv {_sha(trace)[:16]}…: {len(rows)} requests = the accepted {len(source.splitlines()) - 1}-row trace x {tr.get('copies')} (re-derived here); run profile.json {_sha(config)[:16]}… = the registered config; trace path {path}" + (f"; {wrong}" if wrong else ""))


def gate_coverage(res: list, comp: dict | None, inputs: dict) -> None:
    if comp is None:
        add(res, NEW[2], "INCONCLUSIVE", "no completeness record (the completeness checker did not write one)")
        return
    exp, plan, floor = inputs.get("expected") or {}, (inputs.get("dry") or {}).get("plan") or [], (inputs.get("design") or {}).get("MIN_PURE_DECODE")
    want_cells = {f"b{p['b']}_kv{p['kv']}_w{p['wave']}" for p in plan}
    cells, pr = comp.get("cells") or {}, comp.get("profile_result") or {}
    val, ta, corpus, join = pr.get("validation") or {}, comp.get("token_accounting") or {}, comp.get("corpus") or {}, comp.get("join") or {}
    wrong = []
    if set(cells) != want_cells:
        wrong.append(f"cells {sorted(cells)} (planned {sorted(want_cells)})")
    low = {k: v.get("pure") for k, v in cells.items() if floor is None or v.get("pure") is None or v["pure"] < floor}
    if low:
        wrong.append(f"cells below the pure-decode floor {floor}: {low}")
    counts = {"request timings": pr.get("request_timing_count"), "token-accounting requests": ta.get("requests"), "routing corpus segments": corpus.get("segments")}
    if any(v != exp.get("requests") for v in counts.values()):
        wrong.append(f"request counts {counts} (registered {exp.get('requests')})")
    ranks = pr.get("parsed_device_ids")
    if not isinstance(ranks, list) or len(set(ranks)) != exp.get("ranks"):
        wrong.append(f"ranks with kernels {ranks} (registered {exp.get('ranks')})")
    nvtx, per, band = val.get("nvtx_forward_ranges"), exp.get("nvtx_per_iteration"), exp.get("iterations_band") or [None, None]
    if not isinstance(nvtx, int) or not per or nvtx % per or band[0] is None or not band[0] <= nvtx // per <= band[1]:
        wrong.append(f"NVTX forward ranges {nvtx} (registered {per} per iteration, iterations in {band})")
    bad = {k: v for k, v in {**{f"join.{k}": join.get(k) for k in ("duplicate", "missing", "non_monotone")}, **{f"corpus.{k}": corpus.get(k) for k in ("bad_len", "npy_mismatch", "dup_topk")}}.items() if v != 0}
    if bad:
        wrong.append(f"join / routing-array defects (or counters missing): {bad}")
    if comp.get("fails") != 0:
        wrong.append(f"the completeness checker recorded {comp.get('fails')} failed check(s)")
    add(res, NEW[2], "FAIL" if wrong else "PASS", f"cells {len(cells)} (pure-decode iterations {({k: v.get('pure') for k, v in sorted(cells.items())})}, floor {floor}); requests {counts}; ranks {ranks}; NVTX forward ranges {nvtx} = {None if not isinstance(nvtx, int) or not per else nvtx // per} iterations x {per}; join {join}; failed checks {comp.get('fails')}" + (f"; {wrong}" if wrong else ""))


def evaluate(run: Path, inputs: dict, *, identity=None, completeness=None, e1: Path | None = None, **kw) -> list:
    res = V3.evaluate(run, inputs, identity=identity, **kw)
    gate_identity(res, load(identity) if identity else None, inputs)
    gate_trace(res, run, inputs, e1 or HERE.parent)
    gate_coverage(res, load(completeness) if completeness else None, inputs)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--pin", type=Path)
    ap.add_argument("--provenance", type=Path)
    ap.add_argument("--binary", type=Path)
    ap.add_argument("--importer")
    ap.add_argument("--nsys")
    ap.add_argument("--scratch", type=Path)
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--durable-stream", type=Path)
    ap.add_argument("--identity", type=Path)
    ap.add_argument("--completeness", type=Path)
    ap.add_argument("--e1", type=Path, help="e1 directory holding the accepted trace (default: next to checks/)")
    ap.add_argument("--postrun-storage", type=Path)
    ap.add_argument("--postrun-memory", type=Path)
    ap.add_argument("--reservation-deleted", type=Path)
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=RC")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    extras = {}
    for e in args.extra:
        name, _, rc = e.partition("=")
        extras[name] = int(rc)
    now = None
    if args.binary is not None:
        now = V3._mod("e1_frontend_binary_gate").sha256_file(args.binary) if args.binary.is_file() else "missing"
    res = evaluate(args.run, json.loads(args.inputs.read_text()), identity=args.identity, completeness=args.completeness, e1=args.e1, pin=load(args.pin) if args.pin else None, provenance=load(args.provenance) if args.provenance else None,
                   binary_now=now, importer=args.importer, nsys=args.nsys, scratch=args.scratch, keep_scratch=args.keep_scratch, durable_stream=args.durable_stream, post_storage=args.postrun_storage, post_memory=args.postrun_memory,
                   reservation_deleted=args.reservation_deleted, extras=extras)
    verdict = verdict_of(res)
    counts = {s: sum(r["status"] == s for r in res) for s in ("PASS", "FAIL", "INCONCLUSIVE")}
    print(f"# V4 gates {args.run.name}: {len(res)} results, {counts['FAIL']} FAIL, {counts['INCONCLUSIVE']} INCONCLUSIVE → {verdict}")
    if args.out:
        args.out.write_text(json.dumps({"verdict": verdict, "counts": counts, "gates": res}, indent=1, default=str) + "\n")
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3}[verdict]


if __name__ == "__main__":
    sys.exit(main())
