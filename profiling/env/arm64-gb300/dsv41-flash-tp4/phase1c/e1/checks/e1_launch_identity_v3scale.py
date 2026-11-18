#!/usr/bin/env python3
"""Configuration identity for V3 (`e1_v3_scale`) under a V3-specific, fail-closed policy. Corrects the defect of the frozen
V3 post-run, which used the V1/V2 policy: that policy requires the trace to have the hash of the accepted dry-run trace,
which the approved V3 trace (those rows twice) cannot have.

    python3 -I checks/e1_launch_identity_v3scale.py --a <accepted e1_nsys_val run dir> --b <V3 run dir>
        --policy checks/e1_identity_policy_v3scale.json [--out identity_v3scale.json]

Everything except the trace is compared exactly as before: the field-by-field comparison, the rule kinds and the protected
patterns are those of checks/e1_launch_identity_v2.py, whose code is imported unchanged. The one new rule kind,
`derived_trace`, replaces `same_content_file` for `workload.frontend.path` and is satisfied only by proof, not by a path
or a hash alone:

  paths       the reference path is the accepted trace's documented location, the V3 path is the frozen V3 trace's
              documented location under a listed worktree root (the only path differences that are normalized)
  source      the reference trace has the pinned hash of the accepted 56-request trace
  hash        the V3 trace has the pinned V3 hash, and that hash is the entry of the frozen input manifest
              (INPUTS_v3scale.sha256, itself pinned by hash) and of E1-V3-SCALE-INPUTS.json (pinned by hash)
  derivation  re-derived here from the source bytes, independently of the generator: same header with the required
              fields; exactly 56 source rows and 112 rows; rows 1-56 byte-identical to the source in order; rows 57-112
              the same rows in the same order with the wave index renamed once and the arrival time shifted by the
              registered offset, every other field identical; no other row, no blank line; request ids unique; the four
              waves with their batch sizes, counts and arrival times; and the whole file equal to the re-derivation
  scope       the policy applies only to a run named dsv41_flash_1017_gb300_tp4_e1_v3_scale compared with e1_nsys_val,
              and the trace rule must actually have been exercised

Any check that cannot be made (a file unreadable, a pin missing) fails. Exit 0 PASS, 1 FAIL, 3 INCONCLUSIVE. Stdlib only.
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
E1 = HERE.parent
E1_REL = "/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/"


def _v2():
    """A private instance of the unchanged V2 comparison module (its rule dispatcher is extended on this instance only)."""
    spec = importlib.util.spec_from_file_location("e1_launch_identity_v2_for_v3scale", HERE / "e1_launch_identity_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_file(path: str) -> bytes | None:
    try:
        return Path(path).read_bytes()
    except OSError:
        return None


def rederive(source: bytes, spec: dict) -> bytes | None:
    """The V3 trace as it must be, built from the source bytes alone (not with the generator)."""
    try:
        lines = source.decode("ascii").split("\n")
    except UnicodeDecodeError:
        return None
    if not lines or lines[-1] != "" or len(lines) < 3:
        return None
    header, rows = lines[0], lines[1:-1]
    old, new = spec["wave_rename"]
    out = [header, *rows]
    for r in rows:
        f = r.split(",")
        if len(f) != 4 or f[0].count(old) != 1:
            return None
        try:
            arrival = round(float(f[3]) + spec["second_copy_offset_ms"], 1)
        except ValueError:
            return None
        out.append(",".join([f[0].replace(old, new), f[1], f[2], str(arrival)]))
    return ("\n".join(out) + "\n").encode("ascii")


def verify_trace(source: bytes | None, derived: bytes | None, spec: dict, pins: dict, manifest: bytes | None, inputs: bytes | None) -> list[dict]:
    """Every proof obligation of the derived_trace rule, each as {check, ok, detail}. Nothing is skipped: a check that cannot be made is not ok."""
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    have_s, have_d = source is not None, derived is not None
    add("source.readable", have_s, "accepted trace readable" if have_s else "accepted trace not readable")
    add("derived.readable", have_d, "V3 trace readable" if have_d else "V3 trace not readable")
    s_sha, d_sha = (sha256(source) if have_s else None), (sha256(derived) if have_d else None)
    add("source.sha256_is_pinned", s_sha == pins.get("trace_sha256"), f"accepted trace sha256 {s_sha} (pinned {pins.get('trace_sha256')})")
    add("derived.sha256_is_pinned", d_sha is not None and d_sha == pins.get("v3_trace_sha256"), f"V3 trace sha256 {d_sha} (pinned {pins.get('v3_trace_sha256')})")
    m_ok = manifest is not None and sha256(manifest) == pins.get("inputs_manifest_sha256")
    entry = None if manifest is None else next((ln.split()[0] for ln in manifest.decode(errors="replace").splitlines() if ln.split()[1:] == [spec["manifest_entry"]]), None)
    add("derived.sha256_in_frozen_manifest", m_ok and d_sha is not None and entry == d_sha, f"INPUTS_v3scale.sha256 {'is' if m_ok else 'is NOT'} the frozen manifest (pinned {str(pins.get('inputs_manifest_sha256'))[:16]}…); its entry for {spec['manifest_entry']}: {entry}")
    i_ok = inputs is not None and sha256(inputs) == pins.get("inputs_json_sha256")
    try:
        declared = json.loads(inputs)["trace"] if inputs is not None else {}
    except (ValueError, KeyError):
        declared = {}
    add("derived.sha256_in_frozen_inputs", i_ok and d_sha is not None and declared.get("sha256") == d_sha and declared.get("requests") == spec["rows"], f"E1-V3-SCALE-INPUTS.json {'is' if i_ok else 'is NOT'} the frozen file; it declares trace sha256 {declared.get('sha256')}, {declared.get('requests')} requests")
    s_lines = source.decode("ascii", errors="replace").split("\n") if have_s else []
    d_lines = derived.decode("ascii", errors="replace").split("\n") if have_d else []
    s_rows, d_rows = s_lines[1:-1], d_lines[1:-1]
    add("format.header", have_s and have_d and s_lines[0] == d_lines[0] == spec["header"], f"header source {s_lines[:1]} V3 {d_lines[:1]} (required {spec['header']!r})")
    add("format.no_blank_or_trailing_lines", have_s and have_d and s_lines[-1:] == [""] and d_lines[-1:] == [""] and all(s_rows) and all(d_rows) and b"\r" not in (derived or b"\r"), "one row per line, single final newline, no blank line, no carriage return")
    add("rows.source_count", len(s_rows) == spec["source_rows"], f"accepted trace rows {len(s_rows)} (required {spec['source_rows']})")
    add("rows.derived_count", len(d_rows) == spec["rows"] == spec["copies"] * spec["source_rows"], f"V3 trace rows {len(d_rows)} (required {spec['rows']} = {spec['copies']} x {spec['source_rows']})")
    n = len(s_rows)
    first = d_rows[:n]
    add("copy1.identical_in_order", bool(s_rows) and first == s_rows, f"rows 1-{n} byte-identical to the accepted trace in order: {first == s_rows}; first difference at row {next((i + 1 for i, (x, y) in enumerate(zip(first, s_rows)) if x != y), None)}")
    old, new = spec["wave_rename"]
    bad = []
    second = d_rows[n : 2 * n]
    for i, src in enumerate(s_rows):
        f = src.split(",")
        got = second[i].split(",") if i < len(second) else []
        try:
            want = [f[0].replace(old, new), f[1], f[2], round(float(f[3]) + spec["second_copy_offset_ms"], 1)]
            ok = len(f) == 4 and len(got) == 4 and f[0].count(old) == 1 and got[:3] == want[:3] and float(got[3]) == want[3] and got[3] == str(want[3])
        except (ValueError, IndexError):
            ok = False
        if not ok:
            bad.append(n + i + 1)
    add("copy2.same_rows_same_order_wave_renamed_arrival_shifted", bool(s_rows) and len(second) == n and not bad, f"rows {n + 1}-{2 * n}: id {old!r}→{new!r} once, input_len and output_len identical, arrival + {spec['second_copy_offset_ms']} ms; rows not matching: {bad[:8]}{' …' if len(bad) > 8 else ''}")
    add("rows.no_other_rows", len(d_rows) == 2 * n, f"rows beyond the two copies: {max(0, len(d_rows) - 2 * n)}; missing: {max(0, 2 * n - len(d_rows))}")
    ids = [r.split(",")[0] for r in d_rows]
    add("rows.ids_unique", bool(ids) and len(set(ids)) == len(ids), f"{len(ids)} request ids, {len(set(ids))} distinct")
    waves: dict = {}
    for r in d_rows:
        f = r.split(",")
        try:
            key = (f[0].rsplit("_r", 1)[0], float(f[3]))
        except (ValueError, IndexError):
            key = ("<unparseable>", None)
        waves[key] = waves.get(key, 0) + 1
    got_waves = sorted(([k[0], k[1], v] for k, v in waves.items()), key=lambda w: (w[1] is None, w[1]))
    add("waves.four_waves_two_cells", got_waves == spec["waves"], f"waves (id prefix, arrival ms, requests): {got_waves} (required {spec['waves']})")
    lens = sorted({tuple(r.split(",")[1:3]) for r in d_rows})
    add("rows.token_lengths", lens == [tuple(spec["input_output_len"])], f"(input_len, output_len) values: {lens} (required {spec['input_output_len']})")
    regen = rederive(source, spec) if have_s else None
    add("derivation.whole_file_equals_rederivation", regen is not None and have_d and regen == derived, f"V3 trace equals the re-derivation from the accepted trace byte for byte: {regen is not None and regen == derived} (re-derivation sha256 {sha256(regen) if regen else None})")
    return checks


def documented_paths(va, vb, roots: list[str], spec: dict) -> tuple[bool, str]:
    ok_a = isinstance(va, str) and va.endswith(E1_REL + spec["source_path"])
    ok_b = isinstance(vb, str) and any(vb == root + E1_REL + spec["derived_path"] for root in roots)
    return ok_a and ok_b, f"reference path is …{E1_REL}{spec['source_path']}: {ok_a}; V3 path is <listed worktree root>{E1_REL}{spec['derived_path']}: {ok_b}"


def run(a_dir: Path, b_dir: Path, policy: dict, *, reader=read_file, e1_dir: Path = E1) -> dict:
    v2 = _v2()
    base_apply = v2.apply_rule
    trace_report: dict = {"exercised": False}

    def apply_rule(rule: dict, va, vb, ctx: dict):
        if rule["kind"] != "derived_trace":
            return base_apply(rule, va, vb, ctx)
        spec = policy["trace_derivation"]
        paths_ok, paths_why = documented_paths(va, vb, policy["worktree_roots"], spec)
        checks = [{"check": "paths.documented_locations", "ok": paths_ok, "detail": paths_why}]
        checks += verify_trace(reader(str(va)), reader(str(vb)), spec, policy["pinned"], reader(str(e1_dir / "INPUTS_v3scale.sha256")), reader(str(e1_dir / "E1-V3-SCALE-INPUTS.json")))
        trace_report.update({"exercised": True, "checks": checks})
        failed = [c["check"] for c in checks if not c["ok"]]
        return not failed, (f"all {len(checks)} trace-identity checks hold: the V3 trace is the accepted {spec['source_rows']}-request trace exactly {spec['copies']} times ({spec['rows']} requests), hash = frozen manifest" if not failed else f"trace-identity checks failed: {failed}")

    v2.apply_rule = apply_rule  # on this private module instance only
    rules = [r for r in policy["profile_rules"] if r["field"] == "workload.frontend.path"]
    problems = []
    if policy.get("version") != "v3scale-1" or len(rules) != 1 or rules[0]["kind"] != "derived_trace":
        problems.append("the policy is not the V3 policy (version v3scale-1 with exactly one derived_trace rule for workload.frontend.path)")
    rep = v2.run(a_dir, b_dir, policy)
    results = rep.get("results", [])
    if rep.get("verdict") != "INCONCLUSIVE":
        names = [json.loads(Path(p).read_text()).get("name") for d in (a_dir, b_dir) for p in sorted((d / "profile" / "vllm").glob("*_launch.json"))[:1]]
        want = [policy["applies_to"]["reference_name"], policy["applies_to"]["run_name"]]
        if names != want:
            problems.append(f"this policy applies only to {want[1]} compared with {want[0]}; got {names}")
        if not trace_report["exercised"]:
            problems.append("the trace rule was not exercised: the run's trace path equals the reference's, so this is not the V3 workload")
    for p in problems:
        results.append({"scope": "policy", "status": "FAIL", "field": "applies_to", "reason": p})
    fails = [r for r in results if r["status"] == "FAIL"]
    inconclusive = [r for r in results if r["status"] == "INCONCLUSIVE"]
    rep.update({"policy_version": policy.get("version"), "checker": "e1_launch_identity_v3scale.py", "results": results, "trace_identity": trace_report,
                "verdict": "FAIL" if fails else ("INCONCLUSIVE" if inconclusive else "PASS"), "normalized": [r for r in results if r["status"] == "NORMALIZED"]})
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, type=Path)
    ap.add_argument("--b", required=True, type=Path)
    ap.add_argument("--policy", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    rep = run(args.a, args.b, _v2().load_policy(args.policy))
    for r in rep["results"]:
        if r["status"] == "NORMALIZED":
            print(f"NORMALIZED [{r['category']}] {r['scope']}.{r['field']}: a={r['a']!r} b={r['b']!r} — {r['reason']}")
        else:
            print(f"{r['status']:12s} {r.get('scope', '')}.{r.get('field', '')}: {r.get('reason', '')}" + (f" a={r['a']!r} b={r['b']!r}" if r["status"] == "FAIL" and "a" in r else ""))
    for c in rep["trace_identity"].get("checks", []):
        print(f"TRACE {'PASS' if c['ok'] else 'FAIL'}  {c['check']}: {c['detail']}")
    print(f"# configuration identity v3scale: {rep['verdict']} ({len(rep.get('normalized', []))} normalized field(s), every one listed above with its original values; trace rule exercised: {rep['trace_identity']['exercised']})")
    if args.out:
        args.out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    return {"PASS": 0, "FAIL": 1}.get(rep["verdict"], 3)


if __name__ == "__main__":
    sys.exit(main())
