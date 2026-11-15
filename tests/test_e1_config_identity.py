"""Tests for the Phase 1.5 configuration identity check (paired-arm comparisons)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

E1 = Path(__file__).resolve().parents[1] / "profiling" / "env" / "arm64-gb300" / "dsv41-flash-tp4" / "phase1c" / "e1"
spec = importlib.util.spec_from_file_location("e1_config_identity", E1 / "checks" / "e1_config_identity.py")
ci = importlib.util.module_from_spec(spec)
sys.modules["e1_config_identity"] = ci
spec.loader.exec_module(ci)


def _cfg(name, **workload):
    return {"schema_version": 1, "name": name, "log_dir": f"/logs/{name}/profile", "profile_kind": "workload_metrics",
            "server": {"model_path": "m", "tp_size": 4, "extra_args": ["--max-num-seqs", "128"]},
            "workload": {"frontend": {"type": "independent", "path": "/t.csv"}, "backend": {"type": "openai"}, "warmup": True, **workload}}


def _write(tmp_path, name, cfg):
    p = tmp_path / f"{name}.json"; p.write_text(json.dumps(cfg)); return str(p)


def fails(res):
    return [m for s, m in res if s == "FAIL"]


def test_stage1_configuration_is_flagged_as_mixed_wire_modes(tmp_path):
    off = _write(tmp_path, "off", _cfg("e1_dry_off"))
    on = _write(tmp_path, "on", _cfg("e1_dry_on", capture_routed_experts=True))
    res = ci.run(off, on)
    assert any("effective wire mode" in m and "a=streaming b=non-streaming" in m for m in fails(res))


def test_stream_false_control_matches_capture_arm(tmp_path):
    # Stage 1.5 configs: both arms declare stream=false explicitly; only the hook flag differs
    off = _write(tmp_path, "off", _cfg("e1_dry_off_v2", stream=False))
    on = _write(tmp_path, "on", _cfg("e1_dry_on_v2", stream=False, capture_routed_experts=True))
    res = ci.run(off, on)
    assert not fails(res)


def test_implicit_stream_on_one_side_is_a_config_difference(tmp_path):
    # same effective wire mode, but the key is declared on one side only: the identity check is strict about the file
    off = _write(tmp_path, "off", _cfg("off", stream=False))
    on = _write(tmp_path, "on", _cfg("on", capture_routed_experts=True))
    res = ci.run(off, on)
    assert any("workload.stream: False vs '<absent>'" in m for m in fails(res))
    assert any("effective wire mode" in m for s, m in res if s == "PASS")


def test_any_other_difference_fails(tmp_path):
    off = _write(tmp_path, "off", _cfg("a", stream=False))
    cfg_b = _cfg("b", stream=False); cfg_b["server"]["extra_args"] = ["--max-num-seqs", "64"]
    on = _write(tmp_path, "on", cfg_b)
    res = ci.run(off, on)
    assert any("server.extra_args" in m for m in fails(res))


def test_launch_records_permit_only_the_hook_flag(tmp_path):
    def run_dir(name, extra_argv, env):
        d = tmp_path / name; (d / "profile" / "vllm").mkdir(parents=True)
        (d / "profile.json").write_text(json.dumps(_cfg(name, stream=False)))
        launch = {"argv": ["python", "-m", "vllm", "serve", "m"] + extra_argv, "server_argv": ["serve", "m"] + extra_argv,
                  "env": env, "server_config": {"tp": 4}, "fork_python": "/py", "fork_git": "0ab18711"}
        (d / "profile" / "vllm" / f"{name}_launch.json").write_text(json.dumps(launch))
        return str(d)
    a = run_dir("off", [], {"PATH": "/a:/b", "X": "1"})
    b = run_dir("on", ["--enable-return-routed-experts"], {"PATH": "/b:/a", "X": "1"})
    res = ci.run(a, b)
    assert any("server env differs in ['PATH']" in m for m in fails(res))
    res = ci.run(a, b, ignore_env=["PATH"])
    assert not fails(res)
    c = run_dir("bad", ["--enable-eplb"], {"PATH": "/a:/b", "X": "1"})
    res = ci.run(a, c)
    assert any("argv: only in a []" in m and "--enable-eplb" in m for m in fails(res))
