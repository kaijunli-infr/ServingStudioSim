#!/usr/bin/env python3
"""Configuration identity under an explicit field-by-field normalization policy (V1 remediation, item D).

    python3 -I checks/e1_launch_identity_v2.py --a <reference run dir> --b <run dir> --policy checks/e1_identity_policy_v1.json
        [--out identity_v2.json]

Compares the launch records (`profile/vllm/*_launch.json`) and the profile configs (`profile.json`) of two run
directories field by field. Equal fields pass. A differing field passes only when the policy lists it and the rule's
condition holds; every normalized field is reported with both original values. A differing field that the policy
does not list, or that matches a `protected` pattern (model parameters, request schedule, routing, profiler recipe,
CUDA configuration, performance-relevant settings), is a FAIL. Nothing is normalized silently. Stdlib only; read-only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

ABSENT = "<absent>"
PASS_NAME = re.compile(r"^dsv41_flash_1017_gb300_tp4_[a-z0-9_]+$")


def flatten(d, prefix: str = "") -> dict:
    out: dict = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(d, list):
        if not d:
            out[prefix] = []
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


def flatten_launch(rec: dict) -> dict:
    """Flat view of a launch record; the Nsight `-o` value is keyed `argv:-o` (the only argv element with a rule)."""
    flat = flatten(rec)
    argv = rec.get("argv") or []
    for i, tok in enumerate(argv[:-1]):
        if tok == "-o":
            flat["argv:-o"] = flat.pop(f"argv[{i + 1}]")
    return flat


def sha256_file(path: str) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text())
    for scope, rules in (("launch", policy["launch_rules"]), ("profile", policy["profile_rules"])):
        seen = set()
        for rule in rules:
            if rule["field"] in seen:
                raise ValueError(f"policy lists {scope} field {rule['field']} twice")
            seen.add(rule["field"])
            for pat in policy["protected"][scope]:
                if re.search(pat, rule["field"]):
                    raise ValueError(f"policy rule on protected {scope} field {rule['field']} (pattern {pat})")
            if rule["category"] not in ("output_path", "worktree_source_path", "approved_lifecycle_control"):
                raise ValueError(f"unknown category {rule['category']} for {rule['field']}")
    return policy


def _strip_roots(value, roots: list[str]):
    if not isinstance(value, str):
        return value
    for root in roots:
        value = value.replace(root, "<WORKTREE>")
    return value


def apply_rule(rule: dict, va, vb, ctx: dict) -> tuple[bool, str]:
    kind = rule["kind"]
    if kind == "run_name":
        ok = isinstance(va, str) and isinstance(vb, str) and bool(PASS_NAME.match(va)) and bool(PASS_NAME.match(vb))
        return ok, "both are dsv41_flash_1017_gb300_tp4_<pass> run names"
    if kind == "output_stem":
        def norm(v, name):
            if not isinstance(v, str) or "/ServingStudioSim/logs/" not in v or not v.endswith(f"/profile/nsys/{name}"):
                return None
            return "<RUN_DIR>/profile/nsys/<NAME>"
        na, nb = norm(va, ctx["name_a"]), norm(vb, ctx["name_b"])
        return na is not None and na == nb, "<run dir>/profile/nsys/<name> under ServingStudioSim/logs/ on both sides"
    if kind == "output_dir":
        ok = all(isinstance(v, str) and "/ServingStudioSim/logs/" in v and v.endswith("/profile") for v in (va, vb))
        return ok, "<run dir>/profile under ServingStudioSim/logs/ on both sides"
    if kind == "worktree_path":
        same = _strip_roots(va, ctx["roots"]) == _strip_roots(vb, ctx["roots"]) and "<WORKTREE>" in str(_strip_roots(va, ctx["roots"]))
        src = ctx["source_identity"]
        return same and src["ok"], f"equal after replacing the worktree root; pinned source identity: {src['detail']}"
    if kind == "same_content_file":
        pin = ctx["pinned"][rule["sha256_pin"]]
        sa, sb = sha256_file(str(va)), sha256_file(str(vb))
        return sa == sb == pin, f"sha256 a={sa and sa[:16]} b={sb and sb[:16]} pinned={pin[:16]}"
    if kind == "approved_value":
        return va == rule["a"] and vb == rule["b"], f"approved values a={rule['a']!r} b={rule['b']!r}"
    return False, f"unknown rule kind {kind}"


def compare(scope: str, a: dict, b: dict, rules: list, protected: list, ctx: dict, results: list) -> None:
    by_field = {r["field"]: r for r in rules}
    used = set()
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key, ABSENT), b.get(key, ABSENT)
        if va == vb:
            continue
        entry = {"scope": scope, "field": key, "a": va, "b": vb}
        prot = [p for p in protected if re.search(p, key)]
        rule = by_field.get(key)
        if prot:
            entry.update(status="FAIL", reason=f"protected field differs (pattern {prot[0]}); never normalized")
        elif rule is None:
            entry.update(status="FAIL", reason="differs and the policy has no rule for this field")
        else:
            used.add(key)
            ok, why = apply_rule(rule, va, vb, ctx)
            entry.update(
                status="NORMALIZED" if ok else "FAIL",
                category=rule["category"],
                rule=rule["kind"],
                reason=why if ok else f"rule condition not met: {why}",
                doc=rule["doc"],
            )
        results.append(entry)
    results.append({"scope": scope, "status": "INFO", "field": "*", "reason": f"{len(set(a) | set(b))} fields compared; policy rules not triggered: {sorted(set(by_field) - used)}"})


def run(a_dir: Path, b_dir: Path, policy: dict) -> dict:
    results: list = []
    la = glob.glob(str(a_dir / "profile" / "vllm" / "*_launch.json"))
    lb = glob.glob(str(b_dir / "profile" / "vllm" / "*_launch.json"))
    if not la or not lb:
        return {"verdict": "INCONCLUSIVE", "results": [{"status": "INCONCLUSIVE", "reason": "launch record missing", "a": la, "b": lb}]}
    ra, rb = json.loads(Path(la[0]).read_text()), json.loads(Path(lb[0]).read_text())
    fa, fb = ra.get("fork_git") or {}, rb.get("fork_git") or {}
    pinned_head = policy["pinned"]["fork_head"]
    src_ok = fa.get("head") == fb.get("head") == pinned_head and not fa.get("status_short") and not fb.get("status_short")
    ctx = {
        "roots": policy["worktree_roots"],
        "pinned": policy["pinned"],
        "name_a": ra.get("name"),
        "name_b": rb.get("name"),
        "source_identity": {"ok": src_ok, "detail": f"fork head a={str(fa.get('head'))[:12]} b={str(fb.get('head'))[:12]} pinned={pinned_head[:12]}, status clean a={not fa.get('status_short')} b={not fb.get('status_short')}"},
    }
    results.append({"scope": "source", "status": "PASS" if src_ok else "FAIL", "field": "fork_git.head", "reason": ctx["source_identity"]["detail"]})
    compare("launch", flatten_launch(ra), flatten_launch(rb), policy["launch_rules"], policy["protected"]["launch"], ctx, results)
    pa, pb = a_dir / "profile.json", b_dir / "profile.json"
    if pa.is_file() and pb.is_file():
        compare("profile", flatten(json.loads(pa.read_text())), flatten(json.loads(pb.read_text())), policy["profile_rules"], policy["protected"]["profile"], ctx, results)
    else:
        results.append({"scope": "profile", "status": "INCONCLUSIVE", "field": "*", "reason": f"profile.json missing (a={pa.is_file()} b={pb.is_file()})"})
    fails = [r for r in results if r["status"] == "FAIL"]
    inconclusive = [r for r in results if r["status"] == "INCONCLUSIVE"]
    verdict = "FAIL" if fails else ("INCONCLUSIVE" if inconclusive else "PASS")
    return {"a": str(a_dir), "b": str(b_dir), "policy_version": policy["version"], "verdict": verdict, "normalized": [r for r in results if r["status"] == "NORMALIZED"], "results": results}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, type=Path)
    ap.add_argument("--b", required=True, type=Path)
    ap.add_argument("--policy", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    rep = run(args.a, args.b, load_policy(args.policy))
    for r in rep["results"]:
        if r["status"] == "NORMALIZED":
            print(f"NORMALIZED [{r['category']}] {r['scope']}.{r['field']}: a={r['a']!r} b={r['b']!r} — {r['reason']}")
        else:
            print(f"{r['status']:12s} {r.get('scope', '')}.{r.get('field', '')}: {r.get('reason', '')}" + (f" a={r['a']!r} b={r['b']!r}" if r["status"] == "FAIL" and "a" in r else ""))
    print(f"# configuration identity v2: {rep['verdict']} ({len(rep.get('normalized', []))} normalized field(s), every one listed above with its original values)")
    if args.out:
        args.out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    return {"PASS": 0, "FAIL": 1}.get(rep["verdict"], 3)


if __name__ == "__main__":
    sys.exit(main())
