"""CPU tests for the opt-in routed-experts capture beside a timing pass (Phase 1C E1)."""

from __future__ import annotations

import json

import pytest

from alignment import runner as alignment_runner
from alignment.profiler.config import ROUTING_PROFILE_KINDS
from launcher.alignment_config import load_profile_config


def _profile(tmp_path, **workload):
    trace = tmp_path / "trace.csv"
    trace.write_text("id,input_len,output_len,arrival_time\\n")
    (tmp_path / "corpus.txt").write_text("hello\\n")
    cfg = {
        "schema_version": 1,
        "name": "routes_optin_test",
        "log_dir": str(tmp_path / "profile_run"),
        "gpu": "NVIDIA GB300",
        "profile_kind": "nsys",
        "engine": "vllm",
        "server": {"model_path": "model"},
        "workload": {
            "frontend": {"type": "independent", "path": str(trace)},
            "text_file": str(tmp_path / "corpus.txt"),
            "tokenizer": "model/tokenizer.json",
            **workload,
        },
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(cfg))
    return load_profile_config(path)


def test_off_by_default_and_no_argv_change(tmp_path):
    cfg = _profile(tmp_path)
    assert cfg.workload.capture_routed_experts is False
    argv: list[str] = ["--tokens-only"]
    alignment_runner._append_backend_server_args(argv, cfg)
    assert argv == ["--tokens-only"]
    assert alignment_runner._prepared_routes_dir(cfg, tmp_path / "log") is None


def test_opt_in_adds_the_corpus_flag_once_and_never_eplb(tmp_path):
    cfg = _profile(tmp_path, capture_routed_experts=True, backend={"type": "openai"})
    assert cfg.profile_kind == "nsys" and cfg.profile_kind not in ROUTING_PROFILE_KINDS
    assert cfg.captures_expert_load is False
    argv: list[str] = []
    alignment_runner._append_backend_server_args(argv, cfg)
    alignment_runner._append_backend_server_args(argv, cfg)
    assert argv.count("--enable-return-routed-experts") == 1
    assert "--enable-eplb" not in argv and "--eplb-config" not in argv
    routes = alignment_runner._prepared_routes_dir(cfg, tmp_path / "log")
    assert routes == tmp_path / "log" / alignment_runner.ROUTED_EXPERTS_DIR and routes.is_dir()


def test_opt_in_refuses_a_backend_without_routes(tmp_path):
    with pytest.raises(ValueError, match="does not"):
        _profile(tmp_path, capture_routed_experts=True, backend={"type": "vllm_tokens"})


def test_opt_in_refuses_a_non_boolean(tmp_path):
    with pytest.raises(ValueError, match="boolean"):
        _profile(tmp_path, capture_routed_experts="yes")


def test_workload_metrics_pass_can_opt_in_too(tmp_path):
    cfg = _profile(tmp_path, capture_routed_experts=True, backend={"type": "openai"})
    cfg.profile_kind = "workload_metrics"
    argv: list[str] = []
    alignment_runner._append_backend_server_args(argv, cfg)
    assert argv == ["--enable-return-routed-experts"]
