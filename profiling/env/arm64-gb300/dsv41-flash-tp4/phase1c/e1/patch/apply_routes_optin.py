#!/usr/bin/env python3
"""Apply the E1 launcher change: opt-in routed-experts capture alongside a timing pass.

Exact string replacements with asserts, so a drifted source fails loudly instead of half-applying.
Run from a ServingStudioSim checkout root:  python3 phase1c/e1/patch/apply_routes_optin.py
The resulting diff is recorded as launcher-routes-optin.patch (sha256 in SHA256SUMS of this directory).
"""
from pathlib import Path
import sys

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()


def edit(path: str, old: str, new: str, count: int = 1) -> None:
    p = ROOT / path
    s = p.read_text()
    assert s.count(old) == count, (path, s.count(old), old[:60])
    p.write_text(s.replace(old, new))


# 1. workload config: the opt-in key, validated against the backend
edit("alignment/load_generator/config.py",
     "    extra_args: list[str] = field(default_factory=list)\n\n    ARRIVAL_MODES",
     "    extra_args: list[str] = field(default_factory=list)\n"
     "    # Opt-in (Phase 1C E1): ask the vLLM fork to return every request's routed experts and\n"
     "    # persist them beside the timing capture (`<log_dir>/routed_experts`, packed into\n"
     "    # `<log_dir>/token_corpus`). Unlike a `token_corpus` pass this leaves the timing\n"
     "    # instrumentation (NVTX scopes, request timing records) on, and never asks for EPLB.\n"
     "    # Needs a backend that returns routes (`openai`).\n"
     "    capture_routed_experts: bool = False\n\n    ARRIVAL_MODES")
edit("alignment/load_generator/config.py",
     "        if not isinstance(self.warmup, bool):\n            raise ValueError(\"warmup must be a boolean\")\n",
     "        if not isinstance(self.warmup, bool):\n            raise ValueError(\"warmup must be a boolean\")\n"
     "        if not isinstance(self.capture_routed_experts, bool):\n"
     "            raise ValueError(\"capture_routed_experts must be a boolean\")\n"
     "        if self.capture_routed_experts and not self.backend.returns_routed_experts:\n"
     "            raise ValueError(\n"
     "                \"capture_routed_experts needs a workload backend that returns routed experts, \"\n"
     "                f\"and {self.backend.type!r} does not\"\n"
     "            )\n")

# 2. runner: routes directory, server flag (corpus flag only, never EPLB), corpus packing for timing passes
edit("alignment/runner.py",
     "    if cfg.profile_kind != \"token_corpus\":\n        return None\n    routes = log_dir / ROUTED_EXPERTS_DIR",
     "    if cfg.profile_kind != \"token_corpus\" and not cfg.workload.capture_routed_experts:\n        return None\n    routes = log_dir / ROUTED_EXPERTS_DIR")
edit("alignment/runner.py",
     "    if cfg.profile_kind not in ROUTING_PROFILE_KINDS:\n        return\n"
     "    options = list(_PROFILE_KIND_SERVER_ARGS.get(cfg.profile_kind, ()))\n"
     "    if cfg.captures_expert_load:\n        options += _EXPERT_LOAD_SERVER_ARGS\n",
     "    if cfg.profile_kind not in ROUTING_PROFILE_KINDS:\n"
     "        if not cfg.workload.capture_routed_experts:\n            return\n"
     "        # A timing pass that also records routes asks for the corpus flag alone, never EPLB.\n"
     "        options = list(_PROFILE_KIND_SERVER_ARGS[\"token_corpus\"])\n"
     "    else:\n"
     "        options = list(_PROFILE_KIND_SERVER_ARGS.get(cfg.profile_kind, ()))\n"
     "        if cfg.captures_expert_load:\n            options += _EXPERT_LOAD_SERVER_ARGS\n")
edit("alignment/runner.py",
     "    if is_workload_metrics:\n        result = {\n            **spec_artifacts,\n",
     "    corpus_artifacts = {}\n"
     "    if cfg.workload.capture_routed_experts:\n"
     "        # Same packing and coverage check as a token_corpus pass; the routes sit beside the timing\n"
     "        # artifacts so one capture carries both (Phase 1C E1).\n"
     "        _check_routes_cover_replay(prepared_replay.summary_path, log_dir / ROUTED_EXPERTS_DIR)\n"
     "        manifest = token_corpus.pack_token_corpus(\n"
     "            log_dir / ROUTED_EXPERTS_DIR,\n"
     "            log_dir / TOKEN_CORPUS_DIR,\n"
     "            num_experts=_num_routed_experts(cfg),\n"
     "            num_target_layers=_num_target_layers(cfg),\n"
     "        )\n"
     "        corpus_artifacts = {\n"
     "            \"token_corpus_manifest\": str(log_dir / TOKEN_CORPUS_DIR / \"manifest.json\"),\n"
     "            \"token_corpus_tokens\": manifest[\"num_tokens\"],\n"
     "            \"token_corpus_layers\": manifest[\"num_layers\"],\n"
     "        }\n\n"
     "    if is_workload_metrics:\n        result = {\n            **spec_artifacts,\n            **corpus_artifacts,\n")
edit("alignment/runner.py",
     "    result = {\n        **spec_artifacts,\n        \"profile_kind\": cfg.profile_kind,\n",
     "    result = {\n        **spec_artifacts,\n        **corpus_artifacts,\n        \"profile_kind\": cfg.profile_kind,\n")

# 3. launcher validation: the fork-only capability needs the vLLM engine
edit("launcher/alignment_config.py",
     "    if config.profile_kind == \"token_corpus\" and not config.workload.backend.returns_routed_experts:",
     "    if config.workload.capture_routed_experts and config.engine != \"vllm\":\n"
     "        raise ValueError(\n"
     "            \"invalid profile config: workload.capture_routed_experts requires engine vllm, \"\n"
     "            f\"got {config.engine!r}\"\n"
     "        )\n"
     "    if config.profile_kind == \"token_corpus\" and not config.workload.backend.returns_routed_experts:")

# 4. tests (the file next to this script, copied verbatim)
(ROOT / "tests" / "test_alignment_routes_optin.py").write_text((Path(__file__).resolve().parent / "test_alignment_routes_optin.py").read_text())
print("applied")
