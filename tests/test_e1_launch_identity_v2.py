"""CPU tests for the field-by-field configuration identity (checks/e1_launch_identity_v2.py + its policy)."""

from __future__ import annotations

# ruff: noqa: E501

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECKS = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/checks"
S = "/mnt/shared/home/dc/workspace/serving-studio"
HEAD = "0ab18711335780e062d65994a3243da0281d2081"


def _load():
    spec = importlib.util.spec_from_file_location(
        "e1_launch_identity_v2", CHECKS / "e1_launch_identity_v2.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _launch(name: str, wt: str, run: str, lifecycle: bool) -> dict:
    nsys = {
        "executable": "/usr/local/bin/nsys",
        "capture_mode": "cuda_profiler_api",
        "sample": "none",
        "cuda_graph_trace": "node",
        "capture_start_delay_seconds": None if lifecycle else 10.0,
    }
    if lifecycle:
        nsys.update(
            finalize_soft_deadline_s=300.0,
            finalize_hard_deadline_s=600.0,
            finalize_stall_timeout_s=300.0,
            finalize_on_hard_deadline="release_gpus",
            finalize_session_close_timeout_s=300.0,
            finalize_max_extension_s=120.0,
            gpu_held_ceiling_s=2700.0,
            preserve_raw_stream=True,
            raw_preserve_dir=None,
            min_free_bytes_stream=53687091200,
            min_free_bytes_report=21474836480,
            tmp_dir=None,
            qdstrm_importer=None,
        )
    return {
        "schema_version": 1,
        "name": name,
        "argv": [
            "/opt/nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "-o",
            f"{S}/ServingStudioSim/logs/{run}/profile/nsys/{name}",
            "--capture-range=cudaProfilerApi",
            "python",
            "-m",
            "vllm",
        ],
        "server_argv": ["python", "-m", "vllm", "--tensor-parallel-size", "4"],
        "server_config": {"model": "/weights/x", "tensor_parallel_size": 4},
        "env": {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "PYTHONPATH": f"{wt}/alignment/profiler/vllm",
            "VLLM_NVTX_SCOPES_FOR_PROFILING": "1",
        },
        "fork_git": {
            "path": f"{wt}/alignment/profiler/vllm",
            "head": HEAD,
            "branch": "",
            "status_short": "",
        },
        "nsys_config": nsys,
        "nsys_profiler": {"version": "2025.6.3"},
    }


def _profile(name: str, run: str, trace: Path, lifecycle: bool) -> dict:
    nsys = {"executable": "/usr/local/bin/nsys", "sample": "none"}
    if lifecycle:
        nsys.update(
            finalize_soft_deadline_s=300.0,
            finalize_hard_deadline_s=600.0,
            finalize_stall_timeout_s=300.0,
            finalize_on_hard_deadline="release_gpus",
            finalize_session_close_timeout_s=300.0,
            finalize_max_extension_s=120.0,
            gpu_held_ceiling_s=2700.0,
            preserve_raw_stream=True,
        )
    else:
        nsys["capture_start_delay_seconds"] = 10.0
    return {
        "name": name,
        "log_dir": f"{S}/ServingStudioSim/logs/{run}/profile",
        "server": {"tp": 4},
        "workload": {"frontend": {"path": str(trace), "stream": False}},
        "nsys": nsys,
    }


def _dirs(tmp_path: Path, mutate=None, mutate_profile=None) -> tuple[Path, Path, dict]:
    policy = json.loads((CHECKS / "e1_identity_policy_v1.json").read_text())
    trace_a, trace_b = tmp_path / "ta.csv", tmp_path / "tb.csv"
    for t in (trace_a, trace_b):
        t.write_text("request_id,input,output,arrival\n")
    policy["pinned"]["trace_sha256"] = hashlib.sha256(trace_a.read_bytes()).hexdigest()
    a, b = tmp_path / "a", tmp_path / "b"
    la = _launch(
        "dsv41_flash_1017_gb300_tp4_e1_nsys_val", f"{S}/wt-e1stage1", "stage15/e1_nsys_val", False
    )
    lb = _launch(
        "dsv41_flash_1017_gb300_tp4_e1_v1_lifecycle",
        f"{S}/wt-lifecycle",
        "v1/e1_v1_lifecycle",
        True,
    )
    pa = _profile(la["name"], "stage15/e1_nsys_val", trace_a, False)
    pb = _profile(lb["name"], "v1/e1_v1_lifecycle", trace_b, True)
    if mutate:
        mutate(lb)
    if mutate_profile:
        mutate_profile(pb)
    for d, launch, prof in ((a, la, pa), (b, lb, pb)):
        (d / "profile" / "vllm").mkdir(parents=True)
        (d / "profile" / "vllm" / "x_launch.json").write_text(json.dumps(launch))
        (d / "profile.json").write_text(json.dumps(prof))
    return a, b, policy


def test_documented_differences_are_normalized_and_each_one_is_reported(tmp_path):
    mod = _load()
    a, b, policy = _dirs(tmp_path)
    rep = mod.run(a, b, policy)
    assert rep["verdict"] == "PASS", [r for r in rep["results"] if r["status"] == "FAIL"]
    fields = {(n["scope"], n["field"]): n for n in rep["normalized"]}
    assert (
        ("launch", "argv:-o") in fields
        and ("launch", "env.PYTHONPATH") in fields
        and ("profile", "workload.frontend.path") in fields
    )
    assert all(
        "a" in n
        and "b" in n
        and n["category"] in ("output_path", "worktree_source_path", "approved_lifecycle_control")
        for n in rep["normalized"]
    )
    assert fields[("launch", "nsys_config.gpu_held_ceiling_s")]["b"] == 2700.0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda lb: lb["server_argv"].__setitem__(4, "8"),  # model/server parameter
        lambda lb: lb["env"].__setitem__("CUDA_VISIBLE_DEVICES", "0,1"),  # CUDA configuration
        lambda lb: lb["argv"].__setitem__(3, "--sample=cpu"),  # profiler recipe
        lambda lb: lb["nsys_config"].__setitem__("cuda_graph_trace", "graph"),  # profiler recipe
        lambda lb: lb["nsys_config"].__setitem__(
            "gpu_held_ceiling_s", 9999.0
        ),  # lifecycle control, but not the approved value
        lambda lb: lb["nsys_config"].__setitem__(
            "tmp_dir", "/scratch"
        ),  # would add TMPDIR to the server env
        lambda lb: lb["fork_git"].__setitem__("head", "f" * 40),  # source identity
        lambda lb: lb["env"].__setitem__(
            "PYTHONPATH", "/somewhere/else/vllm"
        ),  # not a worktree substitution
        lambda lb: lb["env"].__setitem__("NEW_FLAG", "1"),  # unlisted field
        lambda lb: lb["argv"].__setitem__(
            5, "/tmp/elsewhere/out"
        ),  # output outside the run directory layout
    ],
)
def test_anything_else_that_differs_fails(tmp_path, mutate):
    mod = _load()
    a, b, policy = _dirs(tmp_path, mutate=mutate)
    assert mod.run(a, b, policy)["verdict"] == "FAIL"


def test_request_schedule_and_workload_are_never_normalized(tmp_path):
    mod = _load()
    a, b, policy = _dirs(
        tmp_path, mutate_profile=lambda pb: pb["workload"]["frontend"].__setitem__("stream", True)
    )
    assert mod.run(a, b, policy)["verdict"] == "FAIL"
    (tmp_path / "x").mkdir()
    a, b, policy = _dirs(tmp_path / "x")
    Path(json.loads((b / "profile.json").read_text())["workload"]["frontend"]["path"]).write_text(
        "different schedule\n"
    )
    assert mod.run(a, b, policy)["verdict"] == "FAIL"  # same field, different trace bytes


def test_policy_cannot_whitelist_a_protected_field(tmp_path):
    mod = _load()
    policy = json.loads((CHECKS / "e1_identity_policy_v1.json").read_text())
    bad = copy.deepcopy(policy)
    bad["launch_rules"].append(
        {
            "field": "env.CUDA_VISIBLE_DEVICES",
            "category": "approved_lifecycle_control",
            "kind": "approved_value",
            "a": "x",
            "b": "y",
            "doc": "no",
        }
    )
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="protected"):
        mod.load_policy(p)
    mod.load_policy(CHECKS / "e1_identity_policy_v1.json")  # the committed policy loads
