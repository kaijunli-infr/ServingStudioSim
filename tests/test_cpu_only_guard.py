"""The CPU-only guard: device-node detection, verdict, and the end-to-end exit status."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import cpu_only_guard as guard

REPO = Path(__file__).resolve().parents[1]


def test_open_device_nodes_reports_only_numbered_device_nodes(tmp_path):
    fds = tmp_path / "fd"
    fds.mkdir()
    (fds / "3").symlink_to("/dev/nvidia0")
    (fds / "4").symlink_to("/dev/nvidiactl")
    (fds / "5").symlink_to("/dev/nvidia-uvm")
    (fds / "6").symlink_to("/dev/null")
    (fds / "7").symlink_to("/dev/nvidia3")
    assert guard.open_device_nodes(fds) == ["/dev/nvidia0", "/dev/nvidia3"]
    assert guard.open_device_nodes(tmp_path / "missing") == []


def test_violations_combine_device_nodes_and_torch_state():
    assert guard.violations([], None) == []
    assert guard.violations([], False) == []
    assert guard.violations(["/dev/nvidia1"], False) == ["open device node /dev/nvidia1"]
    assert guard.violations([], True) == ["torch.cuda.is_initialized() is True"]


def _run_pytest(
    tmp_path: Path, body: str, env_extra: dict[str, str]
) -> subprocess.CompletedProcess:
    test_file = tmp_path / "test_probe.py"
    test_file.write_text(textwrap.dedent(body))
    env = {**os.environ, **env_extra, "PYTHONPATH": str(REPO / "tests")}
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "cpu_only_guard",
            "-p",
            "no:cacheprovider",
            "--noconftest",
            "--rootdir",
            str(tmp_path),
            str(test_file),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


def test_guard_hides_cuda_and_passes_a_clean_session(tmp_path):
    out = _run_pytest(
        tmp_path,
        """
        import os
        def test_devices_hidden():
            assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        """,
        {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "cpu_only_guard: CUDA hidden (CUDA_VISIBLE_DEVICES='' set by the guard)" in out.stdout


def test_guard_turns_a_passing_session_into_exit_3_on_a_violation(tmp_path):
    # No device node can be opened on a CPU host (and never should be in a test), so the
    # violation path is driven through the detector: the probe module points it at a fake
    # /proc fd directory. `-p cpu_only_guard` and the probe import the same module object.
    fds = tmp_path / "fd"
    fds.mkdir()
    (fds / "9").symlink_to("/dev/nvidia0")
    out = _run_pytest(
        tmp_path,
        f"""
        from pathlib import Path
        import cpu_only_guard as guard
        _orig = guard.open_device_nodes
        guard.open_device_nodes = lambda fd_dir=None: _orig(Path({str(fds)!r}))
        def test_pass():
            assert True
        """,
        {},
    )
    assert out.returncode == guard.EXIT_VIOLATION, out.stdout + out.stderr
    assert "cpu_only_guard: VIOLATION" in out.stdout and "/dev/nvidia0" in out.stdout
