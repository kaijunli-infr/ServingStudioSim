#!/usr/bin/env python3
"""Independent verification for the V3 configuration-identity remediation (CPU-only; reads preserved evidence only).

    python3 -I e1_v3scale_identity_independent.py --reference <accepted e1_nsys_val run dir> --run <V3 run dir (archive copy)>
        --original-report <cpu/identity_v2_e1_v3_scale.json of the original post-run> --out DIR

It shares no code with the identity checkers.
1. Trace against the approved generator and the frozen inputs: runs a copy of make_e1_inputs_v3scale.py in a scratch
   directory on a copy of the accepted trace and of the V2 config, and compares what it writes with the frozen trace, with
   the trace the run actually used (the run directory's trace.csv and the file at the path in its profile config), and
   with the hashes in INPUTS_v3scale.sha256, E1-V3-SCALE-INPUTS.json and FREEZE_v3scale.sha256.
2. Fields: flattens the two launch records and the two profile configs with its own code, counts the fields and lists
   every field that differs with both values. The differing fields must be exactly the 29 the original report normalized
   plus the trace path, and the field counts must be those the original report gives (182 and 74).
Exit 0 only if everything agrees.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

E1 = Path(__file__).resolve().parent


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def leaves(obj, path=()):
    """(path, value) for every scalar; an empty list is a leaf."""
    if isinstance(obj, dict):
        for k in obj:
            yield from leaves(obj[k], path + (str(k),))
    elif isinstance(obj, list):
        if not obj:
            yield path, []
        for i, v in enumerate(obj):
            yield from leaves(v, path + (f"[{i}]",))
    else:
        yield path, obj


def name(path) -> str:
    out = ""
    for part in path:
        out += part if part.startswith("[") else ("." + part if out else part)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--original-report", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    lines, ok_all = [], True

    def check(label: str, ok: bool, detail: str) -> None:
        nonlocal ok_all
        ok_all &= bool(ok)
        lines.append(f"{'PASS' if ok else 'FAIL'}  {label}: {detail}")

    # ---- 1. trace
    frozen = (E1 / "traces/e1_v3_scale.csv").read_bytes()
    source = (E1 / "traces/e1_dry.csv").read_bytes()
    with tempfile.TemporaryDirectory(prefix="v3scale-regen-") as tmp:
        t = Path(tmp)
        (t / "traces").mkdir()
        (t / "configs_fr/e1_v2_forced_release").mkdir(parents=True)
        shutil.copyfile(E1 / "make_e1_inputs_v3scale.py", t / "make_e1_inputs_v3scale.py")
        shutil.copyfile(E1 / "traces/e1_dry.csv", t / "traces/e1_dry.csv")
        shutil.copyfile(E1 / "configs_fr/e1_v2_forced_release/profile.json", t / "configs_fr/e1_v2_forced_release/profile.json")
        p = subprocess.run([sys.executable, "-I", str(t / "make_e1_inputs_v3scale.py")], capture_output=True, text=True, cwd=t)
        regen = (t / "traces/e1_v3_scale.csv").read_bytes() if (t / "traces/e1_v3_scale.csv").exists() else b""
        regen_cfg = json.loads((t / "configs_v3scale/e1_v3_scale/profile.json").read_text()) if p.returncode == 0 else {}
        regen_inputs = json.loads((t / "E1-V3-SCALE-INPUTS.json").read_text()) if p.returncode == 0 else {}
    check("generator.runs", p.returncode == 0, f"a scratch copy of make_e1_inputs_v3scale.py exit {p.returncode}")
    check("trace.generator_output_equals_frozen", regen == frozen, f"regenerated {sha(regen)[:16]}… ({regen.count(10) - 1} rows) vs frozen {sha(frozen)[:16]}… ({frozen.count(10) - 1} rows)")
    used = (a.run / "trace.csv").read_bytes()
    prof = json.loads((a.run / "profile.json").read_text())
    at_path = Path(prof["workload"]["frontend"]["path"])
    check("trace.run_copy_equals_frozen", used == frozen, f"the run directory's trace.csv {sha(used)[:16]}… = frozen: {used == frozen}")
    check("trace.file_at_configured_path_equals_frozen", at_path.is_file() and at_path.read_bytes() == frozen, f"{at_path}: equals frozen {at_path.is_file() and at_path.read_bytes() == frozen}")
    h = sha(frozen)
    manifest = (E1 / "INPUTS_v3scale.sha256").read_text()
    inputs = json.loads((E1 / "E1-V3-SCALE-INPUTS.json").read_text())
    freeze = (E1 / "FREEZE_v3scale.sha256").read_text()
    check("trace.hash_in_frozen_manifest", f"{h}  traces/e1_v3_scale.csv" in manifest and f"{sha((E1 / 'INPUTS_v3scale.sha256').read_bytes())}  INPUTS_v3scale.sha256" in freeze, f"{h} is the INPUTS_v3scale.sha256 entry, and that manifest is the one FREEZE_v3scale.sha256 pins")
    check("trace.hash_in_frozen_inputs", inputs["trace"]["sha256"] == h == regen_inputs.get("trace", {}).get("sha256") and inputs["trace"]["requests"] == 112 and f"{sha((E1 / 'E1-V3-SCALE-INPUTS.json').read_bytes())}  E1-V3-SCALE-INPUTS.json" in freeze, f"E1-V3-SCALE-INPUTS.json (pinned by the freeze manifest) declares {inputs['trace']['sha256'][:16]}…, {inputs['trace']['requests']} requests; the regenerated inputs declare the same hash")
    s_rows, f_rows = source.decode().split("\n")[1:-1], frozen.decode().split("\n")[1:-1]
    check("trace.is_the_accepted_rows_twice", len(s_rows) == 56 and len(f_rows) == 112 and f_rows[:56] == s_rows and [r.replace("_w1_r", "_w0_r").rsplit(",", 1)[0] for r in f_rows[56:]] == [r.rsplit(",", 1)[0] for r in s_rows]
          and [round(float(x.rsplit(",", 1)[1]) - float(y.rsplit(",", 1)[1]), 6) for x, y in zip(f_rows[56:], s_rows)] == [80000.0] * 56 and sha(source) == "c6d8e13b7229ffd40cfede127bd57ad345aa17213b7590d39f9b2ef697f60158",
          f"{len(s_rows)} accepted rows (sha256 {sha(source)[:16]}…), {len(f_rows)} V3 rows: first 56 identical in order, last 56 the same rows with wave index 1 and +80000.0 ms")
    cfg_frozen = json.loads((E1 / "configs_v3scale/e1_v3_scale/profile.json").read_text())
    strip = lambda d: {k: v for k, v in dict(leaves(d)).items() if name(k) != "workload.frontend.path"}  # noqa: E731
    check("config.generator_output_equals_frozen", bool(regen_cfg) and strip(regen_cfg) == strip(cfg_frozen) and strip(prof) == strip(cfg_frozen), "regenerated config = frozen config = the run's profile.json in every field except the absolute trace path of the scratch copy")
    # ---- 2. fields
    orig = json.loads(a.original_report.read_text())
    report = {}
    for scope, fa, fb in (("launch", sorted((a.reference / "profile/vllm").glob("*_launch.json"))[0], sorted((a.run / "profile/vllm").glob("*_launch.json"))[0]), ("profile", a.reference / "profile.json", a.run / "profile.json")):
        da, db = dict(leaves(json.loads(fa.read_text()))), dict(leaves(json.loads(fb.read_text())))
        keys = set(da) | set(db)
        diff = sorted(name(k) for k in keys if da.get(k, "<absent>") != db.get(k, "<absent>"))
        report[scope] = {"fields": len(keys), "equal": len(keys) - len(diff), "differing": [{"field": name(k), "reference": da.get(k, "<absent>"), "run": db.get(k, "<absent>")} for k in sorted(keys, key=name) if da.get(k, "<absent>") != db.get(k, "<absent>")]}
        info = next(r["reason"] for r in orig["results"] if r["status"] == "INFO" and r["scope"] == scope)
        check(f"{scope}.field_count", info.startswith(f"{len(keys)} fields compared"), f"{len(keys)} fields in the union of both records ({len(keys) - len(diff)} equal, {len(diff)} differing); the original report says: {info.split(';')[0]}")
        # the launch record's Nsight `-o` value is one argv element; the checker reports it under the key argv:-o
        want = sorted(r["field"] for r in orig["results"] if r.get("scope") == scope and r["status"] in ("NORMALIZED", "FAIL"))
        mine = sorted("argv:-o" if d.startswith("argv[") else d for d in diff)
        check(f"{scope}.differing_fields", mine == want, f"differing fields found here {mine if len(mine) < 6 else str(len(mine)) + ' fields'} = the fields the original report normalized or failed: {mine == want}")
    fails = [(r["scope"], r["field"]) for r in orig["results"] if r["status"] == "FAIL"]
    check("original.only_the_trace_failed", fails == [("profile", "workload.frontend.path")] and orig["verdict"] == "FAIL" and len(orig["normalized"]) == 29, f"original report: verdict {orig['verdict']}, {len(orig['normalized'])} normalized, failed fields {fails}")
    protected_diff = [d["field"] for sc in report.values() for d in sc["differing"] if any(d["field"].startswith(p) for p in ("server", "model", "python_runtime", "nsys_profiler", "fork_python", "workload.backend", "workload.arrival", "workload.capture", "workload.stream", "workload.warmup"))]
    check("no_model_routing_profiler_cuda_or_performance_field_differs", not protected_diff, f"differing fields under server / model / python_runtime / nsys_profiler / workload settings other than the trace path: {protected_diff}")
    lines.append(f"# independent verification: {'ALL AGREE' if ok_all else 'DISAGREEMENT'}")
    (a.out / "independent_identity_verification.txt").write_text("\n".join(lines) + "\n")
    (a.out / "independent_identity_verification.json").write_text(json.dumps({"ok": ok_all, "fields": report}, indent=1, default=str) + "\n")
    print("\n".join(lines))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
