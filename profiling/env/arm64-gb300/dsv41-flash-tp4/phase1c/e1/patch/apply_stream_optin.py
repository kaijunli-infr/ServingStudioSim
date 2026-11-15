#!/usr/bin/env python3
"""Apply the Phase 1.5 `workload.stream` opt-out to a ServingStudioSim tree (asserted string edits, idempotent).

    python3 patch/apply_stream_optin.py <repo tree>

Adds `LoadGeneratorConfig.stream: bool = True` (validated boolean, `effective_stream` property) and makes the
load-generator runner pass `--stream false` to session_runner when it is False. Capturing routes already implies
non-streaming on the Rust side; the Python property mirrors that so a configuration identity check can compare the
effective wire mode of two arms. Needs session_runner built with patch/req-frontend-stream-flag.patch whenever a
config sets stream=false (the launcher's dry-run does not exercise session_runner; preflight checks `--help`).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def edit(path: Path, old: str, new: str, marker: str) -> None:
    text = path.read_text()
    if marker in text:
        print(f"already applied: {path.name} ({marker!r})")
        return
    assert text.count(old) == 1, f"{path}: anchor not unique/found: {old[:60]!r}"
    path.write_text(text.replace(old, new))
    print(f"edited {path}")


def main(root: str) -> None:
    repo = Path(root).resolve()
    cfg = repo / "alignment" / "load_generator" / "config.py"
    run = repo / "alignment" / "load_generator" / "runner.py"

    edit(cfg,
         "    capture_routed_experts: bool = False\n\n    ARRIVAL_MODES",
         "    capture_routed_experts: bool = False\n"
         "    # Phase 1.5: wire mode of the replay. True streams (per-token events, TTFT/TPOT measured); False asks\n"
         "    # for one completed body per request, which is what a routed-experts capture does anyway. Set it to\n"
         "    # False on a hook-off control arm so both arms of a comparison use the same wire mode.\n"
         "    stream: bool = True\n\n    ARRIVAL_MODES",
         "stream: bool = True")
    edit(cfg,
         "        if not isinstance(self.capture_routed_experts, bool):\n            raise ValueError(\"capture_routed_experts must be a boolean\")\n",
         "        if not isinstance(self.capture_routed_experts, bool):\n            raise ValueError(\"capture_routed_experts must be a boolean\")\n"
         "        if not isinstance(self.stream, bool):\n            raise ValueError(\"stream must be a boolean\")\n",
         "stream must be a boolean")
    edit(cfg,
         "    @property\n    def context_limit_skip_enabled(self) -> bool:",
         "    @property\n    def effective_stream(self) -> bool:\n"
         "        \"\"\"Wire mode session_runner will use: capturing routes forces non-streaming.\"\"\"\n"
         "        return self.stream and not self.capture_routed_experts\n\n"
         "    @property\n    def context_limit_skip_enabled(self) -> bool:",
         "def effective_stream")
    edit(run,
         "    if routed_experts_dir is not None:\n        argv.extend([\"--routed-experts-dir\", str(routed_experts_dir)])\n",
         "    if routed_experts_dir is not None:\n        argv.extend([\"--routed-experts-dir\", str(routed_experts_dir)])\n"
         "    if not config.stream:\n        # `--stream false` (clap ArgAction::Set): one completed body per request, no routes written.\n"
         "        argv.extend([\"--stream\", \"false\"])\n",
         "\"--stream\", \"false\"")
    test_src = HERE / "test_alignment_stream_optin.py"
    test_dst = repo / "tests" / "test_alignment_stream_optin.py"
    shutil.copyfile(test_src, test_dst)
    print(f"copied {test_dst}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
