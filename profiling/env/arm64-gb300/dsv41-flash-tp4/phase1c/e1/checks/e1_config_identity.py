#!/usr/bin/env python3
"""Configuration identity check for a paired comparison (Phase 1.5, deliverable 3).

    python checks/e1_config_identity.py --a <profile.json or run dir> --b <profile.json or run dir>
                                        [--allow workload.capture_routed_experts ...] [--ignore-env PATH ...] [--out report.json]

Pre-launch: compares two launcher profile configs. Every key must be identical except `name`, `log_dir` and the
keys listed with --allow (default: the hook flag `workload.capture_routed_experts`). The *effective wire mode*
(`stream and not capture_routed_experts`) must be identical on both sides: that is the Stage 1 confounder this check
exists to catch (hook-off streamed, hook-on did not).

Post-run: if the arguments are run directories holding `profile/vllm/*_launch.json`, the server argv, env and server
config are compared too. The only permitted argv difference is `--enable-return-routed-experts` on the hook-on side;
env differences fail unless the variable is listed with --ignore-env.

Exit code 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

DEFAULT_ALLOW = ("name", "log_dir", "workload.capture_routed_experts")
HOOK_ARGS = {"--enable-return-routed-experts"}


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def effective_stream(cfg: dict) -> bool:
    w = cfg.get("workload", {})
    return bool(w.get("stream", True)) and not bool(w.get("capture_routed_experts", False))


def load_profile(path: str) -> tuple[dict, str | None]:
    """(profile config, launch.json path or None) from a profile.json or a run directory."""
    if os.path.isfile(path):
        return json.load(open(path)), None
    cfg_path = os.path.join(path, "profile.json")
    launch = glob.glob(os.path.join(path, "profile", "vllm", "*_launch.json"))
    return json.load(open(cfg_path)), (launch[0] if launch else None)


def compare(a: dict, b: dict, allow: set[str], results: list) -> None:
    fa, fb = flatten(a), flatten(b)
    for key in sorted(set(fa) | set(fb)):
        if key in allow:
            continue
        if fa.get(key, "<absent>") != fb.get(key, "<absent>"):
            results.append(("FAIL", f"config key differs: {key}: {fa.get(key, '<absent>')!r} vs {fb.get(key, '<absent>')!r}"))
    results.append(("PASS" if not any(s == "FAIL" and m.startswith("config key differs") for s, m in results) else "FAIL",
                    f"profile configs identical apart from {sorted(allow)}"))
    ea, eb = effective_stream(a), effective_stream(b)
    results.append(("PASS" if ea == eb else "FAIL", f"effective wire mode (stream and not capture): a={'streaming' if ea else 'non-streaming'} b={'streaming' if eb else 'non-streaming'}"))


def compare_launch(la: str, lb: str, ignore_env: set[str], results: list) -> None:
    A, B = json.load(open(la)), json.load(open(lb))
    for key in ("argv", "server_argv"):
        xa, xb = A.get(key, []), B.get(key, [])
        only_a = [x for x in xa if x not in xb]
        only_b = [x for x in xb if x not in xa]
        ok = set(only_a) <= HOOK_ARGS and set(only_b) <= HOOK_ARGS
        results.append(("PASS" if ok else "FAIL", f"{key}: only in a {only_a}, only in b {only_b} (permitted: {sorted(HOOK_ARGS)})"))
    ea, eb = A.get("env", {}), B.get("env", {})
    diff = {k: (ea.get(k), eb.get(k)) for k in set(ea) | set(eb) if ea.get(k) != eb.get(k) and k not in ignore_env}
    results.append(("PASS" if not diff else "FAIL", f"server env differs in {sorted(diff)} (ignored: {sorted(ignore_env)})"))
    for key in ("server_config", "nsys_config", "python_runtime", "fork_python", "fork_git"):
        if A.get(key) != B.get(key):
            fa, fb = flatten(A.get(key, {}), key + "."), flatten(B.get(key, {}), key + ".")
            d = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
            results.append(("FAIL", f"launch {key} differs: {d[:6]}"))
    results.append(("PASS" if not any(m.startswith("launch ") for s, m in results if s == "FAIL") else "FAIL", "launch records identical apart from the hook flag"))


def run(a: str, b: str, allow=DEFAULT_ALLOW, ignore_env=(), out: str | None = None) -> list:
    results: list = []
    ca, la = load_profile(a)
    cb, lb = load_profile(b)
    compare(ca, cb, set(allow), results)
    if la and lb:
        compare_launch(la, lb, set(ignore_env), results)
    elif la or lb:
        results.append(("FAIL", "launch.json present on one side only"))
    for s, m in results:
        print(f"{s:5s} {m}")
    fails = sum(1 for s, _ in results if s == "FAIL")
    print("RESULT", "PASS" if fails == 0 else f"FAIL ({fails})")
    if out:
        json.dump({"a": a, "b": b, "checks": [{"status": s, "msg": m} for s, m in results], "fails": fails}, open(out, "w"), indent=1)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--allow", nargs="*", default=list(DEFAULT_ALLOW))
    ap.add_argument("--ignore-env", nargs="*", default=[])
    ap.add_argument("--out")
    x = ap.parse_args(argv)
    res = run(x.a, x.b, x.allow, x.ignore_env, x.out)
    return 1 if any(s == "FAIL" for s, _ in res) else 0


if __name__ == "__main__":
    sys.exit(main())
