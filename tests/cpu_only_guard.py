"""pytest plugin: CPU-only guard (``-p cpu_only_guard``; ``tests/`` is on ``pythonpath``).

The tier auto-detection in ``conftest.py`` un-skips the ``gpu`` tier whenever a CUDA device is
visible, and several unmarked tests pick ``cuda`` when ``torch.cuda.is_available()``. A test run
that is meant to be CPU-only therefore has to *hide* the devices before torch is imported, and
has to prove afterwards that nothing opened one anyway. This plugin does both:

* at configure time it sets ``CUDA_VISIBLE_DEVICES=""`` unless the caller already did, so
  ``torch.cuda.is_available()`` is False in-process and in subprocesses that inherit the env;
* at session end it inspects ``/proc/self/fd`` for an open ``/dev/nvidia<N>`` device node and
  asks torch (only if it was imported) whether CUDA was initialised. Either one is a violation:
  the run exits with status 3 even when every test passed, and the summary names the evidence.

``tests/cpu_only.sh`` wraps the command. The guard watches the pytest process only (not xdist
workers): run it without ``-n``.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

DEVICE_NODE = re.compile(r"^/dev/nvidia\d+$")
EXIT_VIOLATION = 3


def open_device_nodes(fd_dir: Path = Path("/proc/self/fd")) -> list[str]:
    """Paths of open ``/dev/nvidia<N>`` device nodes (``/dev/nvidiactl`` and ``-uvm`` excluded:
    loading the driver library may open them without touching a device)."""
    found: list[str] = []
    try:
        entries = list(fd_dir.iterdir())
    except OSError:
        return found
    for fd in entries:
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if DEVICE_NODE.match(target):
            found.append(target)
    return sorted(found)


def torch_cuda_initialised() -> bool | None:
    """True/False if torch is imported, None if it is not (then it cannot have initialised CUDA)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        return bool(torch.cuda.is_initialized())
    except Exception:  # noqa: BLE001 - a broken torch.cuda is reported, not raised
        return True


def violations(nodes: list[str], cuda_initialised: bool | None) -> list[str]:
    out = [f"open device node {n}" for n in nodes]
    if cuda_initialised:
        out.append("torch.cuda.is_initialized() is True")
    return out


def pytest_configure(config) -> None:
    hidden_by_guard = os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    if hidden_by_guard:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config._cpu_only_guard = {"hidden_by_guard": hidden_by_guard, "violations": None}
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        config._cpu_only_guard["violations"] = ["torch.cuda initialised before the guard loaded"]


def pytest_sessionfinish(session, exitstatus) -> None:
    state = session.config._cpu_only_guard
    found = violations(open_device_nodes(), torch_cuda_initialised())
    if state["violations"]:
        found = state["violations"] + found
    state["violations"] = found
    if found:
        session.exitstatus = EXIT_VIOLATION


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    state = config._cpu_only_guard
    src = "set by the guard" if state["hidden_by_guard"] else "set by the caller"
    found = state["violations"] or []
    if found:
        terminalreporter.write_line(
            f"cpu_only_guard: VIOLATION — {'; '.join(found)} (exit status {EXIT_VIOLATION})",
            red=True,
        )
    else:
        terminalreporter.write_line(
            f"cpu_only_guard: CUDA hidden (CUDA_VISIBLE_DEVICES='' {src}); "
            "no /dev/nvidia<N> opened; torch.cuda not initialised"
        )
