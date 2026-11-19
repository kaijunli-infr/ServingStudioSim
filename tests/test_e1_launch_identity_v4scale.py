"""V4 configuration identity (prospective): the trace must be proved to be the accepted trace repeated exactly four times;
every other field is compared as under the V1/V2 policy; the policy is exercised on the actual generated V4 config; the V1/V2
and V3 checkers and policies are untouched and cannot be used for V4."""

from __future__ import annotations

# ruff: noqa: E501

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
E1 = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1"
CHECKS = E1 / "checks"
S = Path("/mnt/shared/home/dc/workspace/serving-studio")
AR15 = S / "artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115/run/e1_nsys_val"
V3RUN = S / "artifacts/phase1c-e1-v3scale-dsv41-flash-gb300-tp4-20261118/run/e1_v3_scale"
SRC = (E1 / "traces/e1_dry.csv").read_bytes()
V4T = (E1 / "traces/e1_v4_scale.csv").read_bytes()
MANIFEST = (E1 / "INPUTS_v4scale.sha256").read_bytes()
INPUTS = (E1 / "E1-V4-SCALE-INPUTS.json").read_bytes()
CONFIG = (E1 / "configs_v4scale/e1_v4_scale/profile.json").read_bytes()
LAUNCH = (REPO / "tests/fixtures/e1_v4_scale/launch.predicted.json").read_text()
needs_artifacts = pytest.mark.skipif(not (AR15 / "profile.json").is_file(), reason="accepted e1_nsys_val artifact not on this host")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, CHECKS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V4 = _load("e1_launch_identity_v4scale")
V3 = _load("e1_launch_identity_v3scale")
V2 = _load("e1_launch_identity_v2")
POLICY = json.loads((CHECKS / "e1_identity_policy_v4scale.json").read_text())
SPEC, PINS = POLICY["trace_derivation"], POLICY["pinned"]


def _failed(derived=V4T, source=SRC, manifest=MANIFEST, inputs=INPUTS, pins=PINS, spec=SPEC):
    return sorted(c["check"] for c in V4.verify_trace(source, derived, spec, pins, manifest, inputs) if not c["ok"])


def _rows(data=V4T):
    return data.decode().split("\n")[:-1]


def _join(rows):
    return ("\n".join(rows) + "\n").encode()


def _mut(fn):
    rows = _rows()
    fn(rows)
    return _join(rows)


# ---- the trace proof


def test_generated_v4_trace_is_the_accepted_trace_exactly_four_times():
    checks = V4.verify_trace(SRC, V4T, SPEC, PINS, MANIFEST, INPUTS)
    assert all(c["ok"] for c in checks) and len(checks) == 19 == V4.expected_checks(POLICY) - 1
    assert V4.rederive(SRC, SPEC) == V4T and hashlib.sha256(V4T).hexdigest() == PINS["v4_trace_sha256"] and hashlib.sha256(SRC).hexdigest() == PINS["trace_sha256"]
    rows = _rows()
    assert len(rows) == 225 and len(_rows(SRC)) == 57 and SPEC["copies"] == 4 and SPEC["rows"] == 224
    ids = [r.split(",")[0] for r in rows[1:]]
    assert len(set(ids)) == 224 and [sum(f"_w{i}_r" in x for x in ids) for i in range(4)] == [56, 56, 56, 56]
    assert [sum(x.startswith(f"e1_b{b}_") for x in ids) for b in (8, 48)] == [32, 192]  # both cells preserved: 4 x 8 and 4 x 48 requests
    arrivals = sorted({float(r.rsplit(",", 1)[1]) for r in rows[1:]})
    assert arrivals == [i * 80000.0 + off for i in range(4) for off in (0.0, 18627.2)]
    assert rows[1:57] == _rows(SRC)[1:]  # copy 0 is the accepted trace byte for byte


def test_generator_is_deterministic_and_matches_the_files_on_disk():
    p = subprocess.run([sys.executable, "-I", str(E1 / "make_e1_inputs_v4scale.py"), "--check"], capture_output=True, text=True)
    assert p.returncode == 0 and "5 of 5" in p.stdout, p.stdout + p.stderr


C1, CN = "copy0.rows_in_order", lambda i: f"copy{i}.rows_in_order"
W, CNT, EXTRA, UNIQ, LEN = "waves.eight_waves_two_cells", "rows.derived_count", "rows.no_other_rows", "rows.ids_unique", "rows.token_lengths"
CASES = {
    "row modified (output_len in copy 0)": (lambda r: r.__setitem__(5, r[5].replace(",1024,", ",1023,")), {C1, LEN}),
    "row modified (input_len in copy 2)": (lambda r: r.__setitem__(130, r[130].replace("32768", "32767")), {CN(2), LEN}),
    "arrival in copy 3 off by 0.1 ms": (lambda r: r.__setitem__(200, r[200].rsplit(",", 1)[0] + ",258627.3"), {CN(3), W}),
    "two rows of copy 0 swapped": (lambda r: r.__setitem__(slice(3, 5), [r[4], r[3]]), {C1}),
    "two rows of copy 3 swapped": (lambda r: r.__setitem__(slice(180, 182), [r[181], r[180]]), {CN(3)}),
    "copies 1 and 2 exchanged": (lambda r: r.__setitem__(slice(57, 169), r[113:169] + r[57:113]), {CN(1), CN(2)}),
    "one row omitted in copy 1": (lambda r: r.pop(70), {CNT, EXTRA, CN(1), CN(2), CN(3), W}),
    "last row omitted": (lambda r: r.pop(), {CNT, EXTRA, CN(3), W}),
    "one row duplicated": (lambda r: r.insert(20, r[20]), {CNT, EXTRA, UNIQ, C1, CN(1), CN(2), CN(3), W}),
    "an unexpected row appended": (lambda r: r.append("e1_b8_kv32k_w4_r00,32768,1024,320000.0"), {CNT, EXTRA, W}),
    "a fifth copy appended": (lambda r: r.extend(x.replace("_w3_r", "_w4_r") for x in list(r[169:])), {CNT, EXTRA, W}),
    "only two copies (the V3 trace)": (lambda r: r.__delitem__(slice(113, 225)), {CNT, EXTRA, CN(2), CN(3), W}),
    "only three copies": (lambda r: r.__delitem__(slice(169, 225)), {CNT, EXTRA, CN(3), W}),
    "copy 2 not renamed (ids of copy 1 twice)": (lambda r: r.__setitem__(slice(113, 169), [x.replace("_w2_r", "_w1_r") for x in r[113:169]]), {CN(2), UNIQ, W}),
    "copy 3 shifted by 220 s instead of 240 s": (lambda r: r.__setitem__(slice(169, 225), [x.rsplit(",", 1)[0] + "," + str(round(float(x.rsplit(",", 1)[1]) - 20000.0, 1)) for x in r[169:225]]), {CN(3), W}),
    "a b=48 row turned into another cell": (lambda r: r.__setitem__(100, r[100].replace("e1_b48_", "e1_b32_")), {CN(1), W}),
    "header field renamed": (lambda r: r.__setitem__(0, "id,input_len,output_len,arrival"), {"format.header"}),
    "blank line inside": (lambda r: r.insert(30, ""), {"format.no_blank_or_trailing_lines", CNT, EXTRA, C1, CN(1), CN(2), CN(3), W, LEN}),
}
ALWAYS = {"derived.sha256_is_pinned", "derived.sha256_in_input_manifest", "derived.sha256_in_registered_inputs", "derivation.whole_file_equals_rederivation"}


@pytest.mark.parametrize("name", list(CASES))
def test_modified_reordered_omitted_duplicated_and_unexpected_rows_fail(name):
    fn, expected = CASES[name]
    assert set(_failed(_mut(fn))) == expected | ALWAYS, name


def test_format_changes_and_unreadable_files_fail_closed():
    assert "format.no_blank_or_trailing_lines" in _failed(V4T + b"\n") and "format.no_blank_or_trailing_lines" in _failed(V4T.replace(b"\n", b"\r\n"))
    assert set(_failed(V4T[:-1])) >= ALWAYS
    assert {"derived.readable", "derived.sha256_is_pinned", "derivation.whole_file_equals_rederivation"} <= set(_failed(derived=None))
    assert {"source.readable", "source.sha256_is_pinned", "derivation.whole_file_equals_rederivation"} <= set(_failed(source=None))


def test_a_consistent_four_copy_trace_of_another_source_fails_on_the_pins():
    other = SRC.replace(b",1024,", b",512,")
    derived = V4.rederive(other, SPEC)
    assert set(_failed(derived, source=other)) == {"source.sha256_is_pinned", "derived.sha256_is_pinned", "derived.sha256_in_input_manifest", "derived.sha256_in_registered_inputs", LEN}


def test_hash_alone_or_manifest_alone_is_not_enough():
    tampered = _mut(lambda r: r.__setitem__(5, r[5].replace(",1024,", ",1023,")))
    sha = hashlib.sha256(tampered).hexdigest()
    man = MANIFEST.replace(PINS["v4_trace_sha256"].encode(), sha.encode())
    pins = {**PINS, "v4_trace_sha256": sha, "inputs_manifest_sha256": hashlib.sha256(man).hexdigest()}
    failed = _failed(tampered, pins=pins, manifest=man)
    assert "derived.sha256_is_pinned" not in failed and "derived.sha256_in_input_manifest" not in failed
    assert {C1, "derivation.whole_file_equals_rederivation"} <= set(failed)
    assert _failed(manifest=MANIFEST + b"# edited\n") == ["derived.sha256_in_input_manifest"]
    assert _failed(inputs=INPUTS.replace(b'"requests": 224', b'"requests": 225', 1)) == ["derived.sha256_in_registered_inputs"]
    assert _failed(manifest=None) == ["derived.sha256_in_input_manifest"] and _failed(inputs=None) == ["derived.sha256_in_registered_inputs"]


def test_only_the_documented_paths_are_normalized():
    roots = POLICY["worktree_roots"]
    a = "/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/traces/e1_dry.csv"
    b = roots[1] + "/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/traces/e1_v4_scale.csv"
    assert V4.documented_paths(a, b, roots, SPEC)[0] is True
    for bad_b in (b.replace("e1_v4_scale.csv", "e1_v3_scale.csv"), "/tmp/e1_v4_scale.csv", b.replace(roots[1], "/mnt/shared/home/dc/elsewhere"), b + ".bak", None):
        assert V4.documented_paths(a, bad_b, roots, SPEC)[0] is False
    assert V4.documented_paths("/tmp/e1_dry.csv", b, roots, SPEC)[0] is False


def test_policy_differs_from_the_v1_policy_only_in_the_trace_rule_and_pins():
    v1 = json.loads((CHECKS / "e1_identity_policy_v1.json").read_text())
    assert POLICY["launch_rules"] == v1["launch_rules"] and POLICY["protected"] == v1["protected"] and POLICY["worktree_roots"] == v1["worktree_roots"]
    strip = lambda rules: [r for r in rules if r["field"] != "workload.frontend.path"]  # noqa: E731
    assert strip(POLICY["profile_rules"]) == strip(v1["profile_rules"])
    assert {k: v for k, v in POLICY["pinned"].items() if k in v1["pinned"]} == v1["pinned"]
    rule = next(r for r in POLICY["profile_rules"] if r["field"] == "workload.frontend.path")
    assert rule["kind"] == "derived_trace" and "sha256_pin" not in rule and POLICY["version"] == "v4scale-1" == V4.VERSION
    assert hashlib.sha256(MANIFEST).hexdigest() == PINS["inputs_manifest_sha256"] and hashlib.sha256(INPUTS).hexdigest() == PINS["inputs_json_sha256"]
    assert f"{PINS['v4_trace_sha256']}  traces/e1_v4_scale.csv" in MANIFEST.decode() and json.loads(INPUTS)["trace"]["sha256"] == PINS["v4_trace_sha256"]


def test_v4_config_is_the_accepted_recipe_with_three_fields_changed():
    v4, v3, v2 = json.loads(CONFIG), json.loads((E1 / "configs_v3scale/e1_v3_scale/profile.json").read_text()), json.loads((E1 / "configs_fr/e1_v2_forced_release/profile.json").read_text())
    assert hashlib.sha256(CONFIG).hexdigest() == json.loads(INPUTS)["config_sha256"]
    for other in (v3, v2):
        a, b = copy.deepcopy(v4), copy.deepcopy(other)
        for d in (a, b):
            d.pop("name"), d.pop("log_dir"), d["workload"]["frontend"].pop("path")
        assert a == b  # model, server arguments, backend, routing hook, Nsight recipe and lifecycle bounds identical
    assert v4["workload"]["backend"]["type"] == "openai" and v4["workload"]["capture_routed_experts"] is True and v4["workload"]["stream"] is False
    assert v4["nsys"]["gpu_held_ceiling_s"] == 2700.0 and v4["nsys"]["finalize_hard_deadline_s"] == 600.0 and v4["nsys"]["finalize_soft_deadline_s"] == 300.0 and "server_term_grace_s" not in json.dumps(v4)


# ---- the whole comparison on the actual V4 config with the predicted launch record


def _dirs(tmp: Path, mutate_launch=None, mutate_profile=None, config: bytes = CONFIG, launch: str = LAUNCH):
    b = tmp / "b"
    (b / "profile" / "vllm").mkdir(parents=True)
    ld, pd = json.loads(launch), json.loads(config)
    if mutate_launch:
        mutate_launch(ld)
    if mutate_profile:
        mutate_profile(pd)
    (b / "profile" / "vllm" / "x_launch.json").write_text(json.dumps(ld) if mutate_launch else launch)
    (b / "profile.json").write_bytes(json.dumps(pd).encode() if mutate_profile else config)
    return AR15, b


def _reader(overrides):
    return lambda path: overrides[Path(path).name] if Path(path).name in overrides else V4.read_file(path)


@needs_artifacts
def test_actual_v4_config_passes_under_the_v4_policy(tmp_path):
    a, b = _dirs(tmp_path)
    rep = V4.run(a, b, copy.deepcopy(POLICY))
    ti = rep["trace_identity"]
    assert rep["verdict"] == "PASS" and rep["checker"] == "e1_launch_identity_v4scale.py" and rep["policy_version"] == "v4scale-1"
    assert ti["exercised"] is True and len(ti["checks"]) == ti["expected_checks"] == 20 and all(c["ok"] for c in ti["checks"])
    assert len(rep["normalized"]) == 30 and [r for r in rep["results"] if r["status"] == "FAIL"] == []
    info = {r["scope"]: r["reason"] for r in rep["results"] if r["status"] == "INFO"}
    assert info["launch"].startswith("182 fields compared") and info["profile"].startswith("74 fields compared")
    trace = next(r for r in rep["normalized"] if r["field"] == "workload.frontend.path")
    assert trace["rule"] == "derived_trace" and trace["a"].endswith("/traces/e1_dry.csv") and trace["b"].endswith("/traces/e1_v4_scale.csv")


@needs_artifacts
def test_the_v3_defect_is_detected_the_v1_policy_fails_on_the_v4_config(tmp_path):
    a, b = _dirs(tmp_path)
    rep = V2.run(a, b, V2.load_policy(CHECKS / "e1_identity_policy_v1.json"))
    fails = [r for r in rep["results"] if r["status"] == "FAIL"]
    assert rep["verdict"] == "FAIL" and [(r["scope"], r["field"]) for r in fails] == [("profile", "workload.frontend.path")] and len(rep["normalized"]) == 29
    rep3 = V3.run(a, b, V2.load_policy(CHECKS / "e1_identity_policy_v3scale.json"))  # the V3 policy does not apply to V4 either
    assert rep3["verdict"] == "FAIL"


@needs_artifacts
@pytest.mark.parametrize("name", ["row modified (output_len in copy 0)", "two rows of copy 3 swapped", "one row omitted in copy 1", "one row duplicated", "an unexpected row appended", "only two copies (the V3 trace)"])
def test_a_tampered_trace_file_fails_the_whole_comparison(tmp_path, name):
    a, b = _dirs(tmp_path)
    rep = V4.run(a, b, copy.deepcopy(POLICY), reader=_reader({"e1_v4_scale.csv": _mut(CASES[name][0])}))
    fails = [r for r in rep["results"] if r["status"] == "FAIL"]
    assert rep["verdict"] == "FAIL" and [(r["scope"], r["field"]) for r in fails] == [("profile", "workload.frontend.path")] and "trace-identity checks failed" in fails[0]["reason"]


UNAUTHORIZED = {
    "server argument changed": ("launch", lambda d: d["server_argv"].__setitem__(d["server_argv"].index("--max-num-seqs") + 1, "64")),
    "model path changed": ("profile", lambda d: d["server"].__setitem__("model_path", "/mnt/shared/weights/other")),
    "tensor-parallel size changed": ("profile", lambda d: d["server"].__setitem__("tp_size", 2)),
    "routing hook switched off": ("profile", lambda d: d["workload"].__setitem__("capture_routed_experts", False)),
    "load-generator backend changed": ("profile", lambda d: d["workload"]["backend"].__setitem__("type", "vllm_tokens")),
    "arrival mode changed": ("profile", lambda d: d["workload"].__setitem__("arrival_mode", "saturated")),
    "streaming switched on": ("profile", lambda d: d["workload"].__setitem__("stream", True)),
    "Nsight CUDA graph trace changed": ("profile", lambda d: d["nsys"].__setitem__("cuda_graph_trace", "graph")),
    "Nsight capture mode changed": ("profile", lambda d: d["nsys"].__setitem__("capture_mode", "immediate")),
    "CUDA device order changed in the server environment": ("launch", lambda d: d["env"].__setitem__("CUDA_DEVICE_ORDER", "FASTEST_FIRST")),
    "a new environment variable": ("launch", lambda d: d["env"].__setitem__("VLLM_SOMETHING", "1")),
    "Python package set changed": ("profile", lambda d: d["python_runtime"]["packages"].append({"name": "extra", "version": "1"})),
    "hard deadline not the approved value": ("profile", lambda d: d["nsys"].__setitem__("finalize_hard_deadline_s", 900.0)),
    "GPU-held ceiling not the approved value": ("profile", lambda d: d["nsys"].__setitem__("gpu_held_ceiling_s", 3600.0)),
    "a field the policy does not know": ("profile", lambda d: d.__setitem__("experimental_flag", True)),
    "delayed capture start reintroduced": ("profile", lambda d: d["nsys"].__setitem__("capture_start_delay_seconds", 5.0)),
}


@needs_artifacts
@pytest.mark.parametrize("name", list(UNAUTHORIZED))
def test_unauthorized_configuration_changes_fail(tmp_path, name):
    scope, fn = UNAUTHORIZED[name]
    a, b = _dirs(tmp_path, **({"mutate_launch": fn} if scope == "launch" else {"mutate_profile": fn}))
    rep = V4.run(a, b, copy.deepcopy(POLICY))
    fails = [r for r in rep["results"] if r["status"] == "FAIL"]
    assert rep["verdict"] == "FAIL" and fails and all(r["scope"] == scope for r in fails), name
    assert rep["trace_identity"]["exercised"] is True and all(c["ok"] for c in rep["trace_identity"]["checks"])  # the trace proof itself still holds


@needs_artifacts
def test_v4_policy_does_not_apply_to_other_runs_or_shapes(tmp_path):
    if (V3RUN / "profile.json").is_file():
        rep = V4.run(AR15, V3RUN, copy.deepcopy(POLICY))  # the V3 run: two copies, another name
        reasons = " | ".join(r["reason"] for r in rep["results"] if r["status"] == "FAIL")
        assert rep["verdict"] == "FAIL" and "applies only to dsv41_flash_1017_gb300_tp4_e1_v4_scale" in reasons and "trace-identity checks failed" in reasons
    a, b = _dirs(tmp_path / "name", mutate_launch=lambda d: d.__setitem__("name", "dsv41_flash_1017_gb300_tp4_e1_other"))
    assert V4.run(a, b, copy.deepcopy(POLICY))["verdict"] == "FAIL"
    for label, edit in (("kind", lambda p: next(r for r in p["profile_rules"] if r["field"] == "workload.frontend.path").__setitem__("kind", "run_name")), ("version", lambda p: p.__setitem__("version", "v3scale-1")), ("copies", lambda p: p["trace_derivation"].__setitem__("copies", 2))):
        weak = copy.deepcopy(POLICY)
        edit(weak)
        a, b = _dirs(tmp_path / label)
        rep = V4.run(a, b, weak)
        assert rep["verdict"] == "FAIL" and any("not the V4 policy" in r["reason"] for r in rep["results"] if r["status"] == "FAIL"), label
    a, b = _dirs(tmp_path / "old")  # the frozen V1/V2 checker does not know the rule kind: the V4 policy cannot loosen it
    rep = V2.run(a, b, V2.load_policy(CHECKS / "e1_identity_policy_v4scale.json"))
    assert rep["verdict"] == "FAIL" and "unknown rule kind derived_trace" in next(r["reason"] for r in rep["results"] if r["status"] == "FAIL")


@needs_artifacts
def test_missing_launch_record_is_inconclusive_not_pass(tmp_path):
    b = tmp_path / "b"
    b.mkdir()
    (b / "profile.json").write_bytes(CONFIG)
    rep = V4.run(AR15, b, copy.deepcopy(POLICY))
    assert rep["verdict"] == "INCONCLUSIVE" and rep["trace_identity"]["exercised"] is False


def test_earlier_identity_files_are_unchanged():
    scripts = (E1 / "SCRIPTS_fr.sha256").read_text()
    for name in ("checks/e1_launch_identity_v2.py", "checks/e1_identity_policy_v1.json"):
        assert f"{hashlib.sha256((E1 / name).read_bytes()).hexdigest()}  {name}" in scripts
    assert hashlib.sha256((CHECKS / "e1_identity_policy_v3scale.json").read_bytes()).hexdigest() and "v3scale-1" in (CHECKS / "e1_identity_policy_v3scale.json").read_text()
