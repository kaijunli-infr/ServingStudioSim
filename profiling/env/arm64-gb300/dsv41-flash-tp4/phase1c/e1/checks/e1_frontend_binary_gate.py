#!/usr/bin/env python3
"""Post-run gate S.binary_unchanged for V2: the load generator that ran was the pinned executable, and nothing compiled it.

    python3 -I checks/e1_frontend_binary_gate.py --driver-record <run>/binary_check.json
        --launcher-record <run>/profile/load_generator/frontend_binary.json --provenance BINARY-PROVENANCE.json
        [--binary <session_runner as it is now>] [--out binary_gate.json]

Two independent records are required and each must agree with the pin (BINARY-PROVENANCE.json `sha256`):
  driver    (run_e1_fr.sh, shell): sha256 before and after the launcher, file identity (inode, mtime, size) before and
            after, and the number of cargo "Compiling" lines in the run log
  launcher  (alignment.load_generator.runner): prebuilt mode, sha256 verified before the launch and again after it,
            cargo not consulted, build-tool guard installed with no blocked request
A missing or unreadable record is a FAIL: the absence of evidence is not accepted. Source-tree equality is never
consulted. Exit 0 only if every check holds. Stdlib only; read-only.
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import hashlib
import json
import sys
from pathlib import Path


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(driver: dict | None, launcher: dict | None, pin: str | None, binary_now: str | None = None) -> list[dict]:
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    add("pin.present", isinstance(pin, str) and len(pin) == 64, f"pinned sha256 {pin}")
    d = driver or {}
    add("driver.record_present", driver is not None, "driver record (binary_check.json) readable" if driver is not None else "driver record missing or unreadable")
    add("driver.sha256_before_is_pinned", d.get("sha256_before_launcher") == pin, f"before the launcher: {d.get('sha256_before_launcher')}")
    add("driver.sha256_after_is_pinned", d.get("sha256_after_launcher") == pin, f"after the launcher: {d.get('sha256_after_launcher')}")
    add("driver.unchanged", d.get("unchanged") is True, f"unchanged={d.get('unchanged')}")
    add("driver.file_identity_unchanged", bool(d.get("stat_before")) and d.get("stat_before") == d.get("stat_after"), f"inode/mtime/size before {d.get('stat_before')!r}, after {d.get('stat_after')!r}")
    add("driver.no_compiling_lines", d.get("cargo_compiling_lines_in_run_log") == 0, f"cargo 'Compiling' lines in the run log: {d.get('cargo_compiling_lines_in_run_log')}")
    f = launcher or {}
    guard = f.get("build_tool_guard") or {}
    add("launcher.record_present", launcher is not None, "launcher record (frontend_binary.json) readable" if launcher is not None else "launcher record missing or unreadable")
    add("launcher.prebuilt_mode", str(f.get("mode", "")).startswith("prebuilt"), f"mode {f.get('mode')!r}")
    add("launcher.sha256_before_is_pinned", f.get("sha256") == pin and f.get("pinned_sha256") == pin, f"verified before the launch: {f.get('sha256')} (record's pin {f.get('pinned_sha256')})")
    add("launcher.sha256_after_is_pinned", f.get("sha256_after_launch") == pin, f"verified after the launch: {f.get('sha256_after_launch')}")
    add("launcher.unchanged", f.get("unchanged") is True, f"unchanged={f.get('unchanged')}")
    add("launcher.cargo_not_consulted", f.get("cargo_consulted") is False, f"cargo_consulted={f.get('cargo_consulted')}")
    add("launcher.build_tool_guard_installed", guard.get("installed") is True and guard.get("active") is True, f"guard installed={guard.get('installed')} active={guard.get('active')}")
    add("launcher.no_build_tool_requested", guard.get("blocked") == [], f"blocked build-tool requests: {guard.get('blocked')}")
    if binary_now is not None:
        add("binary.now_is_pinned", binary_now == pin, f"executable at post-run time: {binary_now}")
    return checks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--driver-record", required=True, type=Path)
    ap.add_argument("--launcher-record", required=True, type=Path)
    ap.add_argument("--provenance", required=True, type=Path)
    ap.add_argument("--binary", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    pin = (load(args.provenance) or {}).get("sha256")
    now = None
    if args.binary is not None:
        now = sha256_file(args.binary) if args.binary.is_file() else "missing"
    checks = evaluate(load(args.driver_record), load(args.launcher_record), pin, now)
    for c in checks:
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['check']}: {c['detail']}")
    ok = all(c["ok"] for c in checks)
    print(f"# load-generator binary gate: {sum(not c['ok'] for c in checks)} FAIL of {len(checks)} → {'PASS' if ok else 'FAIL'}")
    if args.out:
        args.out.write_text(json.dumps({"verdict": "PASS" if ok else "FAIL", "pinned_sha256": pin, "checks": checks}, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
