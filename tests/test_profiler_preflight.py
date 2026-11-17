"""CPU tests for the read-only identity preflight (alignment/profiler/preflight.py).

A fake Nsight install (CLI + QdstrmImporter) stands in for the real one; the vLLM
import probe is replaced; everything else (own-process identity, parsers against a
real server-log excerpt, recipe against a real launch record) runs for real.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

from alignment.profiler import preflight, vllm_server

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "e1_nsys_val"


def _fake_install(root: Path) -> Path:
    target = root / "nsight" / "target-linux-fake"
    host = root / "nsight" / "host-linux-fake"
    target.mkdir(parents=True)
    host.mkdir(parents=True)
    nsys = target / "nsys"
    nsys.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        '  --version) echo "NVIDIA Nsight Systems version 9.9.9-test";;\n'
        '  profile) echo "  --cuda-event-trace";;\n'
        "esac\n"
    )
    nsys.chmod(nsys.stat().st_mode | stat.S_IXUSR)
    imp = host / "QdstrmImporter"
    imp.write_text("#!/bin/sh\nexit 0\n")
    imp.chmod(imp.stat().st_mode | stat.S_IXUSR)
    return nsys


def _config(tmp_path: Path, nsys: Path):
    from launcher.alignment_config import load_profile_config

    cfg = load_profile_config(FIX / "profile.json", require_python_runtime=False)
    cfg.nsys.executable = str(nsys)
    cfg.fork_python = sys.executable
    cfg.log_dir = str(tmp_path / "run" / "profile")
    stream = tmp_path / "stream"
    stream.mkdir()
    cfg.nsys.tmp_dir = str(stream)
    cfg.nsys.finalize_soft_deadline_s = 300
    cfg.nsys.finalize_hard_deadline_s = 600
    cfg.nsys.gpu_held_ceiling_s = 2700
    cfg.nsys.min_free_bytes_stream = 1
    cfg.nsys.min_free_bytes_report = 1
    return cfg


def _probe_ok(python, env):
    assert env["CUDA_VISIBLE_DEVICES"] == ""  # the probe must not touch a GPU
    return {
        "version": "0.31.1.dev17+g0ab187113",
        "file": str(vllm_server.VLLM_SOURCE_ROOT / "vllm" / "__init__.py"),
        "python": python,
    }


def _by_name(checks):
    return {c.name: c for c in checks}


def _reference_launch_for(tmp_path: Path, python: str) -> Path:
    """The accepted launch record with the server interpreter swapped for this one."""
    launch = json.loads((FIX / "launch.json").read_text())
    orig = launch["server_argv"][0]
    launch["server_argv"][0] = python
    launch["argv"] = [python if a == orig else a for a in launch["argv"]]
    out = tmp_path / "launch.json"
    out.write_text(json.dumps(launch))
    return out


def test_preflight_passes_on_a_clean_pinned_environment(tmp_path):
    nsys = _fake_install(tmp_path)
    cfg = _config(tmp_path, nsys)
    checks, facts = preflight.run_preflight(
        cfg,
        reference_launch=_reference_launch_for(tmp_path, sys.executable),
        reference_server_log=FIX / "server_log_excerpt.log",
        probe=_probe_ok,
        nvidia_pids=lambda: [],
    )
    by = _by_name(checks)
    assert by["profiler.nsys"].status == "INFO"  # no pins yet
    assert facts["nsys"]["sha256"] == preflight.sha256_file(nsys)
    assert facts["importer"]["path"].endswith("host-linux-fake/QdstrmImporter")
    assert by["profiler.importer_matches_cli"].status == "PASS"
    assert by["server.vllm_import"].status == "PASS"
    assert by["server.model_path"].status == (
        "PASS" if Path(cfg.server.model_path).exists() else "FAIL"
    )
    assert by["recipe.server_argv"].status == "PASS"
    assert by["recipe.nsys_prefix"].status == "PASS"
    assert by["recipe.nsys_executable"].status == "FAIL"  # the fake is not /opt/nvidia/.../nsys
    assert by["recipe.env"].status == "FAIL"  # tmp_dir adds TMPDIR relative to the accepted launch
    for name in (
        "cleanup.identity",
        "cleanup.ancestry",
        "cleanup.refuses_self",
        "cleanup.detects_identity_change",
        "cleanup.gpu_query",
        "parsers.pids",
        "parsers.phases",
        "policy.valid",
        "policy.ceiling_set",
        "locations.lifecycle.json",
        "locations.lifecycle.control",
        "locations.log_dir",
        "locations.stream_dir",
        "locations.stream_leftovers",
        "locations.report_dir",
    ):
        assert by[name].status == "PASS", (name, by[name].detail)
    assert "EngineCore" in by["parsers.pids"].detail and "Worker" in by["parsers.pids"].detail
    # pins round trip
    pins = {k: facts[k] for k in ("nsys", "importer", "server")}
    checks2, _ = preflight.run_preflight(cfg, pins=pins, probe=_probe_ok, nvidia_pids=lambda: [])
    by2 = _by_name(checks2)
    assert by2["profiler.nsys"].status == "PASS"
    assert by2["profiler.importer"].status == "PASS"
    assert by2["server.identity"].status == "PASS"


def test_preflight_fails_on_pin_mismatch_leftovers_and_missing_ceiling(tmp_path):
    nsys = _fake_install(tmp_path)
    cfg = _config(tmp_path, nsys)
    cfg.nsys.tmp_dir = None  # no TMPDIR added
    cfg.nsys.gpu_held_ceiling_s = None
    _, facts = preflight.run_preflight(cfg, probe=_probe_ok, nvidia_pids=lambda: [])
    pins = {k: dict(facts[k]) for k in ("nsys", "importer", "server")}
    pins["nsys"]["sha256"] = "0" * 64
    pins["server"]["vllm_version"] = "0.0.0"
    nsys_dir = Path(cfg.log_dir) / "nsys"
    nsys_dir.mkdir(parents=True)
    (nsys_dir / "lifecycle.json").write_text("{}")
    checks, _ = preflight.run_preflight(cfg, pins=pins, probe=_probe_ok, nvidia_pids=lambda: None)
    by = _by_name(checks)
    assert by["profiler.nsys"].status == "FAIL" and "sha256" in by["profiler.nsys"].detail
    assert by["server.identity"].status == "FAIL" and "vllm_version" in by["server.identity"].detail
    assert by["policy.ceiling_set"].status == "FAIL"
    assert by["locations.lifecycle.json"].status == "FAIL"
    assert by["locations.log_dir"].status == "FAIL"
    assert by["cleanup.gpu_query"].status == "FAIL"
    assert by["recipe.reference"].status == "INFO"
    assert by["parsers.reference_log"].status == "INFO"


def test_preflight_cli_writes_pins_and_exits_nonzero_on_fail(tmp_path, monkeypatch):
    nsys = _fake_install(tmp_path)
    cfg_path = tmp_path / "profile.json"
    raw = json.loads((FIX / "profile.json").read_text())
    raw["nsys"]["executable"] = str(nsys)
    raw["nsys"]["gpu_held_ceiling_s"] = 2700
    raw["nsys"]["finalize_soft_deadline_s"] = 300
    raw["nsys"]["finalize_hard_deadline_s"] = 600
    raw["nsys"]["min_free_bytes_stream"] = 1
    raw["nsys"]["min_free_bytes_report"] = 1
    raw["fork_python"] = sys.executable
    raw["log_dir"] = str(tmp_path / "run" / "profile")
    cfg_path.write_text(json.dumps(raw))
    monkeypatch.setattr(preflight, "_python_probe", _probe_ok)
    monkeypatch.setattr(preflight.lifecycle, "_nvidia_smi_compute_pids", lambda: [])
    out = tmp_path / "preflight.json"
    pins = tmp_path / "pins.json"
    rc = preflight.main(
        [
            "--config",
            str(cfg_path),
            "--reference-server-log",
            str(FIX / "server_log_excerpt.log"),
            "--json",
            str(out),
            "--write-pins",
            str(pins),
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert {c["name"] for c in data["checks"]} >= {
        "profiler.nsys",
        "cleanup.refuses_self",
        "parsers.phases",
    }
    pinned = json.loads(pins.read_text())
    assert pinned["nsys"]["version"] == "NVIDIA Nsight Systems version 9.9.9-test"
    assert pinned["recipe"]["server_argv"][-1] == "--profiler-config.profiler=cuda"
    # second run against mutated pins fails
    pinned["importer"]["sha256"] = "f" * 64
    pins.write_text(json.dumps(pinned))
    rc = preflight.main(["--config", str(cfg_path), "--pins", str(pins)])
    assert rc == 1


def test_preflight_touches_no_gpu_and_creates_nothing(tmp_path):
    nsys = _fake_install(tmp_path)
    cfg = _config(tmp_path, nsys)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    preflight.run_preflight(cfg, probe=_probe_ok, nvidia_pids=lambda: [])
    after = sorted(str(p) for p in tmp_path.rglob("*"))
    assert before == after
    assert not Path(cfg.log_dir).exists()


def test_v1_config_expands_to_the_accepted_nsys_val_recipe():
    """Item 5 of the V1 GO: the lifecycle fields in configs_v1 change neither the server argv nor the nsys flags
    nor the environment of the accepted `e1_nsys_val` launch (only the output stem and the removed start delay)."""
    from launcher.alignment_config import load_profile_config

    e1 = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1"
    cfg = load_profile_config(
        e1 / "configs_v1" / "e1_v1_lifecycle" / "profile.json", require_python_runtime=False
    )
    launch = json.loads((FIX / "launch.json").read_text())
    recipe = preflight.expand_recipe(cfg)
    assert recipe["server_argv"] == launch["server_argv"]
    i = launch["argv"].index("-o")
    ref_prefix = (
        launch["argv"][:i]
        + launch["argv"][i + 2 : len(launch["argv"]) - len(launch["server_argv"])]
    )
    assert recipe["nsys_prefix_without_output"][1:] == ref_prefix[1:]
    exe = Path(cfg.nsys.executable)
    if exe.exists():  # on the rack: /usr/local/bin/nsys → the pinned 2025.6.3 install
        assert exe.resolve() == Path(ref_prefix[0]).resolve()
    assert cfg.nsys.tmp_dir is None  # no TMPDIR added: server env identical to the accepted launch
    assert cfg.nsys.capture_start_delay_seconds is None
    policy = cfg.nsys.finalize_policy()
    policy.validate()
    assert (policy.soft_deadline_s, policy.hard_deadline_s, policy.gpu_held_ceiling_s) == (
        300,
        600,
        2700,
    )
    assert policy.max_extension_s == 120 and policy.preserve_raw_stream is True
    for key in (
        "capture_mode",
        "cuda_graph_trace",
        "cuda_event_trace",
        "sample",
        "cpuctxsw",
        "cuda_flush_interval_ms",
    ):
        assert getattr(cfg.nsys, key) == launch["nsys_config"][key]
