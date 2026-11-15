"""CPU tests for the Phase 1.5 `workload.stream` opt-out (identical wire mode for hook-off and hook-on arms)."""

from __future__ import annotations

import json

import pytest

from alignment.load_generator import runner as lg_runner
from launcher.alignment_config import load_profile_config


def _profile(tmp_path, **workload):
    tmp_path.mkdir(parents=True, exist_ok=True)
    trace = tmp_path / "trace.csv"
    trace.write_text("id,input_len,output_len,arrival_time\n")
    (tmp_path / "corpus.txt").write_text("hello\n")
    cfg = {
        "schema_version": 1,
        "name": "stream_optin_test",
        "log_dir": str(tmp_path / "profile_run"),
        "gpu": "NVIDIA GB300",
        "profile_kind": "workload_metrics",
        "engine": "vllm",
        "server": {"model_path": "model"},
        "workload": {
            "frontend": {"type": "independent", "path": str(trace)},
            "text_file": str(tmp_path / "corpus.txt"),
            "tokenizer": "model/tokenizer.json",
            "backend": {"type": "openai"},
            **workload,
        },
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(cfg))
    return load_profile_config(path)


def _argv(cfg, tmp_path, routed_experts_dir=None):
    prepared = lg_runner.prepare_replay.__wrapped__(cfg.workload, tmp_path) if hasattr(lg_runner.prepare_replay, "__wrapped__") else None
    captured = {}

    def fake_run(argv, **kw):
        captured["argv"] = list(argv)

        class R:
            returncode = 0
        return R()

    orig = lg_runner.subprocess.run
    lg_runner.subprocess.run = fake_run
    try:
        prepared = lg_runner.PreparedReplay(trace_path=tmp_path / "trace.csv", text_file=tmp_path / "corpus.txt", tokenizer="tok",
                                            log_path=tmp_path / "lg" / "replay.jsonl", summary_path=tmp_path / "lg" / "summary.json")
        lg_runner.run_replay(cfg.workload, prepared, base_url="http://127.0.0.1:1", model="m", routed_experts_dir=routed_experts_dir)
    finally:
        lg_runner.subprocess.run = orig
    return captured["argv"]


def test_stream_defaults_to_true_and_adds_no_flag(tmp_path):
    cfg = _profile(tmp_path)
    assert cfg.workload.stream is True and cfg.workload.effective_stream is True
    argv = _argv(cfg, tmp_path)
    assert "--stream" not in argv


def test_stream_false_passes_the_flag(tmp_path):
    cfg = _profile(tmp_path, stream=False)
    assert cfg.workload.stream is False and cfg.workload.effective_stream is False
    argv = _argv(cfg, tmp_path)
    i = argv.index("--stream")
    assert argv[i + 1] == "false"
    assert "--routed-experts-dir" not in argv


def test_capture_implies_non_streaming_without_the_flag(tmp_path):
    cfg = _profile(tmp_path, capture_routed_experts=True)
    assert cfg.workload.stream is True and cfg.workload.effective_stream is False
    argv = _argv(cfg, tmp_path, routed_experts_dir=tmp_path / "routes")
    assert "--routed-experts-dir" in argv and "--stream" not in argv


def test_both_arms_have_the_same_effective_wire_mode(tmp_path):
    off = _profile(tmp_path / "off", stream=False)
    on = _profile(tmp_path / "on", capture_routed_experts=True)
    assert off.workload.effective_stream == on.workload.effective_stream is False


def test_stream_refuses_a_non_boolean(tmp_path):
    with pytest.raises(ValueError, match="boolean"):
        _profile(tmp_path, stream="no")
