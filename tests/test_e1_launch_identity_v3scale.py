"""V3 configuration identity: the trace must be proved to be the accepted trace repeated exactly twice; every other field is
compared as under the V1/V2 policy; the V1/V2 checker and policy are untouched and still protect their runs."""

from __future__ import annotations

# ruff: noqa: E501

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
E1 = REPO / "profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1"
CHECKS = E1 / "checks"
EVID = E1 / "runs/v3scale/run/evidence"
S = Path("/mnt/shared/home/dc/workspace/serving-studio")
AR15 = S / "artifacts/phase1c-e1-stage15-dsv41-flash-gb300-tp4-20261115/run/e1_nsys_val"
V2RUN = S / "artifacts/phase1c-e1-v2fr-dsv41-flash-gb300-tp4-20261118/run/e1_v2_forced_release"
V1RUN = S / "artifacts/phase1c-e1-v1-dsv41-flash-gb300-tp4-20261118/run/e1_v1_lifecycle"
SRC = (E1 / "traces/e1_dry.csv").read_bytes()
V3T = (E1 / "traces/e1_v3_scale.csv").read_bytes()
MANIFEST = (E1 / "INPUTS_v3scale.sha256").read_bytes()
INPUTS = (E1 / "E1-V3-SCALE-INPUTS.json").read_bytes()
needs_artifacts = pytest.mark.skipif(not (AR15 / "profile.json").is_file(), reason="accepted e1_nsys_val artifact not on this host")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, CHECKS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V3 = _load("e1_launch_identity_v3scale")
V2 = _load("e1_launch_identity_v2")
POLICY = json.loads((CHECKS / "e1_identity_policy_v3scale.json").read_text())
SPEC, PINS = POLICY["trace_derivation"], POLICY["pinned"]


def _failed(derived=V3T, source=SRC, manifest=MANIFEST, inputs=INPUTS, pins=PINS, spec=SPEC):
    return sorted(c["check"] for c in V3.verify_trace(source, derived, spec, pins, manifest, inputs) if not c["ok"])


def _rows(data=V3T):
    return data.decode().split("\n")[:-1]


def _join(rows):
    return ("\n".join(rows) + "\n").encode()


# ---- the trace proof


def test_frozen_v3_trace_is_the_accepted_trace_exactly_twice():
    checks = V3.verify_trace(SRC, V3T, SPEC, PINS, MANIFEST, INPUTS)
    assert all(c["ok"] for c in checks) and len(checks) == 17
    assert V3.rederive(SRC, SPEC) == V3T and hashlib.sha256(V3T).hexdigest() == PINS["v3_trace_sha256"] and hashlib.sha256(SRC).hexdigest() == PINS["trace_sha256"]
    assert len(_rows()) == 113 and len(_rows(SRC)) == 57


def _mut(fn):
    rows = _rows()
    fn(rows)
    return _join(rows)


CASES = {
    "row modified (output_len of a first-copy row)": (lambda r: r.__setitem__(5, r[5].replace(",1024,", ",1023,")), {"copy1.identical_in_order", "rows.token_lengths"}),
    "row modified (input_len of a second-copy row)": (lambda r: r.__setitem__(70, r[70].replace("32768", "32767")), {"copy2.same_rows_same_order_wave_renamed_arrival_shifted", "rows.token_lengths"}),
    "arrival of a second-copy row off by 0.1 ms": (lambda r: r.__setitem__(80, r[80].rsplit(",", 1)[0] + ",98627.3"), {"copy2.same_rows_same_order_wave_renamed_arrival_shifted", "waves.four_waves_two_cells"}),
    "two rows of the first copy swapped": (lambda r: r.__setitem__(slice(3, 5), [r[4], r[3]]), {"copy1.identical_in_order"}),
    "two rows of the second copy swapped": (lambda r: r.__setitem__(slice(60, 62), [r[61], r[60]]), {"copy2.same_rows_same_order_wave_renamed_arrival_shifted"}),
    "one row omitted": (lambda r: r.pop(40), {"rows.derived_count", "rows.no_other_rows", "copy1.identical_in_order", "copy2.same_rows_same_order_wave_renamed_arrival_shifted", "waves.four_waves_two_cells"}),
    "last row omitted": (lambda r: r.pop(), {"rows.derived_count", "rows.no_other_rows", "copy2.same_rows_same_order_wave_renamed_arrival_shifted", "waves.four_waves_two_cells"}),
    "one row duplicated": (lambda r: r.insert(20, r[20]), {"rows.derived_count", "rows.no_other_rows", "rows.ids_unique", "copy1.identical_in_order", "copy2.same_rows_same_order_wave_renamed_arrival_shifted", "waves.four_waves_two_cells"}),
    "an unexpected row appended": (lambda r: r.append("e1_b8_kv32k_w2_r00,32768,1024,160000.0"), {"rows.derived_count", "rows.no_other_rows", "waves.four_waves_two_cells"}),
    "a third copy appended": (lambda r: r.extend(x.replace("_w1_r", "_w2_r") for x in list(r[57:])), {"rows.derived_count", "rows.no_other_rows", "waves.four_waves_two_cells"}),
    "second copy not renamed (same ids twice)": (lambda r: r.__setitem__(slice(57, 113), [x.replace("_w1_r", "_w0_r") for x in r[57:113]]), {"copy2.same_rows_same_order_wave_renamed_arrival_shifted", "rows.ids_unique", "waves.four_waves_two_cells"}),
    "second pair shifted by 60 s instead of 80 s": (lambda r: r.__setitem__(slice(57, 113), [x.rsplit(",", 1)[0] + "," + str(round(float(x.rsplit(",", 1)[1]) - 20000.0, 1)) for x in r[57:113]]), {"copy2.same_rows_same_order_wave_renamed_arrival_shifted", "waves.four_waves_two_cells"}),
    "a b=48 row turned into another cell": (lambda r: r.__setitem__(100, r[100].replace("e1_b48_", "e1_b32_")), {"copy2.same_rows_same_order_wave_renamed_arrival_shifted", "waves.four_waves_two_cells"}),
    "header field renamed": (lambda r: r.__setitem__(0, "id,input_len,output_len,arrival"), {"format.header"}),
    "blank line inside": (lambda r: r.insert(30, ""), {"format.no_blank_or_trailing_lines", "rows.derived_count", "rows.no_other_rows", "copy1.identical_in_order", "copy2.same_rows_same_order_wave_renamed_arrival_shifted", "waves.four_waves_two_cells", "rows.token_lengths"}),
}
ALWAYS = {"derived.sha256_is_pinned", "derived.sha256_in_frozen_manifest", "derived.sha256_in_frozen_inputs", "derivation.whole_file_equals_rederivation"}


@pytest.mark.parametrize("name", list(CASES))
def test_modified_reordered_omitted_duplicated_and_unexpected_rows_fail(name):
    fn, expected = CASES[name]
    assert set(_failed(_mut(fn))) == expected | ALWAYS, name


def test_format_changes_fail():
    assert "format.no_blank_or_trailing_lines" in _failed(V3T + b"\n") and "format.no_blank_or_trailing_lines" in _failed(V3T.replace(b"\n", b"\r\n"))
    assert set(_failed(V3T[:-1])) >= ALWAYS  # no final newline


def test_a_consistent_copy_of_another_source_fails_on_the_pins():
    other = SRC.replace(b",1024,", b",512,")
    derived = V3.rederive(other, SPEC)
    assert set(_failed(derived, source=other)) == {"source.sha256_is_pinned", "derived.sha256_is_pinned", "derived.sha256_in_frozen_manifest", "derived.sha256_in_frozen_inputs", "rows.token_lengths"}


def test_hash_alone_or_manifest_alone_is_not_enough():
    tampered = _mut(lambda r: r.__setitem__(5, r[5].replace(",1024,", ",1023,")))
    sha = hashlib.sha256(tampered).hexdigest()
    # an attacker-controlled pin and manifest for the tampered file: the derivation still fails
    pins = {**PINS, "v3_trace_sha256": sha, "inputs_manifest_sha256": hashlib.sha256(MANIFEST.replace(PINS["v3_trace_sha256"].encode(), sha.encode())).hexdigest()}
    failed = _failed(tampered, pins=pins, manifest=MANIFEST.replace(PINS["v3_trace_sha256"].encode(), sha.encode()))
    assert "derived.sha256_is_pinned" not in failed and "derived.sha256_in_frozen_manifest" not in failed
    assert {"copy1.identical_in_order", "derivation.whole_file_equals_rederivation"} <= set(failed)
    # the real trace with a manifest that is not the frozen one, or frozen inputs that are not the frozen file
    assert _failed(manifest=MANIFEST + b"# edited\n") == ["derived.sha256_in_frozen_manifest"]
    assert _failed(inputs=INPUTS.replace(b"112", b"113", 1)) == ["derived.sha256_in_frozen_inputs"]
    assert _failed(manifest=None) == ["derived.sha256_in_frozen_manifest"] and _failed(inputs=None) == ["derived.sha256_in_frozen_inputs"]


def test_unreadable_files_fail_closed():
    assert {"derived.readable", "derived.sha256_is_pinned", "derivation.whole_file_equals_rederivation"} <= set(_failed(derived=None))
    assert {"source.readable", "source.sha256_is_pinned", "derivation.whole_file_equals_rederivation"} <= set(_failed(source=None))


def test_only_the_documented_paths_are_normalized():
    roots = POLICY["worktree_roots"]
    a = "/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/traces/e1_dry.csv"
    b = roots[1] + "/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/traces/e1_v3_scale.csv"
    assert V3.documented_paths(a, b, roots, SPEC)[0] is True
    for bad_b in (b.replace("e1_v3_scale.csv", "e1_v3_scale_copy.csv"), "/tmp/e1_v3_scale.csv", b.replace(roots[1], "/mnt/shared/home/dc/elsewhere"), b + ".bak", None):
        assert V3.documented_paths(a, bad_b, roots, SPEC)[0] is False
    assert V3.documented_paths("/tmp/e1_dry.csv", b, roots, SPEC)[0] is False


def test_policy_differs_from_the_v1_policy_only_in_the_trace_rule_and_pins():
    v1 = json.loads((CHECKS / "e1_identity_policy_v1.json").read_text())
    assert POLICY["launch_rules"] == v1["launch_rules"] and POLICY["protected"] == v1["protected"] and POLICY["worktree_roots"] == v1["worktree_roots"]
    strip = lambda rules: [r for r in rules if r["field"] != "workload.frontend.path"]  # noqa: E731
    assert strip(POLICY["profile_rules"]) == strip(v1["profile_rules"])
    assert {k: v for k, v in POLICY["pinned"].items() if k in v1["pinned"]} == v1["pinned"]
    rule = next(r for r in POLICY["profile_rules"] if r["field"] == "workload.frontend.path")
    assert rule["kind"] == "derived_trace" and "sha256_pin" not in rule and POLICY["version"] == "v3scale-1"
    assert hashlib.sha256(MANIFEST).hexdigest() == PINS["inputs_manifest_sha256"] and hashlib.sha256(INPUTS).hexdigest() == PINS["inputs_json_sha256"]
    freeze = (E1 / "FREEZE_v3scale.sha256").read_text()
    assert f"{PINS['inputs_manifest_sha256']}  INPUTS_v3scale.sha256" in freeze and f"{PINS['inputs_json_sha256']}  E1-V3-SCALE-INPUTS.json" in freeze


# ---- the whole comparison on the actual V3 records


def _dirs(tmp: Path, mutate_launch=None, mutate_profile=None, b_launch=None, b_profile=None):
    """Reference = the accepted e1_nsys_val artifact; run = the actual V3 launch record and profile config (committed evidence)."""
    b = tmp / "b"
    (b / "profile" / "vllm").mkdir(parents=True)
    launch = json.loads((b_launch or next(EVID.glob("*_launch.json"))).read_text())
    profile = json.loads((b_profile or EVID / "profile.json").read_text())
    if mutate_launch:
        mutate_launch(launch)
    if mutate_profile:
        mutate_profile(profile)
    (b / "profile" / "vllm" / "x_launch.json").write_text(json.dumps(launch))
    (b / "profile.json").write_text(json.dumps(profile))
    return AR15, b


def _reader(overrides=None):
    def read(path: str):
        if overrides and Path(path).name in overrides:
            return overrides[Path(path).name]
        return V3.read_file(path)

    return read


@needs_artifacts
def test_actual_v3_records_pass_under_the_v3_policy(tmp_path):
    a, b = _dirs(tmp_path)
    rep = V3.run(a, b, copy.deepcopy(POLICY))
    assert rep["verdict"] == "PASS" and rep["trace_identity"]["exercised"] is True and all(c["ok"] for c in rep["trace_identity"]["checks"]) and len(rep["trace_identity"]["checks"]) == 18
    assert len(rep["normalized"]) == 30 and [r for r in rep["results"] if r["status"] == "FAIL"] == []
    info = {r["scope"]: r["reason"] for r in rep["results"] if r["status"] == "INFO"}
    assert info["launch"].startswith("182 fields compared") and info["profile"].startswith("74 fields compared")
    trace = next(r for r in rep["normalized"] if r["field"] == "workload.frontend.path")
    assert trace["rule"] == "derived_trace" and trace["a"].endswith("/traces/e1_dry.csv") and trace["b"].endswith("/traces/e1_v3_scale.csv")


@needs_artifacts
def test_original_failure_is_reproducible_with_the_frozen_checker_and_policy(tmp_path):
    a, b = _dirs(tmp_path)
    rep = V2.run(a, b, V2.load_policy(CHECKS / "e1_identity_policy_v1.json"))
    fails = [r for r in rep["results"] if r["status"] == "FAIL"]
    assert rep["verdict"] == "FAIL" and [(r["scope"], r["field"]) for r in fails] == [("profile", "workload.frontend.path")] and len(rep["normalized"]) == 29
    assert "sha256 a=c6d8e13b7229ffd4 b=d9d2dc77b316b786 pinned=c6d8e13b7229ffd4" in fails[0]["reason"]


@needs_artifacts
@pytest.mark.parametrize("name", ["row modified (output_len of a first-copy row)", "two rows of the second copy swapped", "one row omitted", "one row duplicated", "an unexpected row appended"])
def test_a_tampered_trace_file_fails_the_whole_comparison(tmp_path, name):
    a, b = _dirs(tmp_path)
    rep = V3.run(a, b, copy.deepcopy(POLICY), reader=_reader({"e1_v3_scale.csv": _mut(CASES[name][0])}))
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
    "a field the policy does not know": ("profile", lambda d: d.__setitem__("experimental_flag", True)),
    "delayed capture start reintroduced": ("profile", lambda d: d["nsys"].__setitem__("capture_start_delay_seconds", 5.0)),
}


@needs_artifacts
@pytest.mark.parametrize("name", list(UNAUTHORIZED))
def test_unauthorized_configuration_changes_fail(tmp_path, name):
    scope, fn = UNAUTHORIZED[name]
    a, b = _dirs(tmp_path, **({"mutate_launch": fn} if scope == "launch" else {"mutate_profile": fn}))
    rep = V3.run(a, b, copy.deepcopy(POLICY))
    fails = [r for r in rep["results"] if r["status"] == "FAIL"]
    assert rep["verdict"] == "FAIL" and fails and all(r["scope"] == scope for r in fails), name
    assert rep["trace_identity"]["exercised"] is True and all(c["ok"] for c in rep["trace_identity"]["checks"])  # the trace proof itself still holds


@needs_artifacts
def test_v3_policy_does_not_apply_to_other_runs(tmp_path):
    if not (V2RUN / "profile.json").is_file():
        pytest.skip("V2 archive not on this host")
    rep = V3.run(AR15, V2RUN, copy.deepcopy(POLICY))  # the V2 run: its trace is the accepted one, its name is not V3's
    reasons = " | ".join(r["reason"] for r in rep["results"] if r["status"] == "FAIL")
    assert rep["verdict"] == "FAIL" and "applies only to dsv41_flash_1017_gb300_tp4_e1_v3_scale" in reasons
    assert "trace-identity checks failed" in reasons and "paths.documented_locations" in reasons and "derived.sha256_is_pinned" in reasons  # its trace is not the V3 trace


@needs_artifacts
def test_v3_trace_under_another_name_or_policy_shape_fails(tmp_path):
    a, b = _dirs(tmp_path, mutate_launch=lambda d: d.__setitem__("name", "dsv41_flash_1017_gb300_tp4_e1_other"))
    assert V3.run(a, b, copy.deepcopy(POLICY))["verdict"] == "FAIL"
    weak = copy.deepcopy(POLICY)
    next(r for r in weak["profile_rules"] if r["field"] == "workload.frontend.path")["kind"] = "run_name"  # a rule that would wave the path through
    a, b = _dirs(tmp_path / "2")
    rep = V3.run(a, b, weak)
    assert rep["verdict"] == "FAIL" and any("not the V3 policy" in r["reason"] for r in rep["results"] if r["status"] == "FAIL")


@needs_artifacts
def test_v1_and_v2_stay_protected_by_their_own_policy(tmp_path):
    v1_policy = V2.load_policy(CHECKS / "e1_identity_policy_v1.json")
    for run in (V2RUN, V1RUN):
        if not (run / "profile.json").is_file():
            pytest.skip(f"{run.name} archive not on this host")
        assert V2.run(AR15, run, v1_policy)["verdict"] == "PASS", run.name
    # the frozen checker does not know the new rule kind: the V3 policy cannot be used to loosen it
    a, b = _dirs(tmp_path)
    rep = V2.run(a, b, V2.load_policy(CHECKS / "e1_identity_policy_v3scale.json"))
    assert rep["verdict"] == "FAIL" and "unknown rule kind derived_trace" in next(r["reason"] for r in rep["results"] if r["status"] == "FAIL")
    # and a V2 run with a changed trace still fails under its own policy
    b2 = tmp_path / "v2changed"
    (b2 / "profile" / "vllm").mkdir(parents=True)
    (b2 / "profile" / "vllm" / "x_launch.json").write_text(next((V2RUN / "profile" / "vllm").glob("*_launch.json")).read_text())
    prof = json.loads((V2RUN / "profile.json").read_text())
    prof["workload"]["frontend"]["path"] = str(E1 / "traces/e1_v3_scale.csv")
    (b2 / "profile.json").write_text(json.dumps(prof))
    assert V2.run(AR15, b2, v1_policy)["verdict"] == "FAIL"


def test_frozen_v1_v2_identity_files_are_unchanged():
    scripts = (E1 / "SCRIPTS_fr.sha256").read_text()
    for name in ("checks/e1_launch_identity_v2.py", "checks/e1_identity_policy_v1.json"):
        assert f"{hashlib.sha256((E1 / name).read_bytes()).hexdigest()}  {name}" in scripts
