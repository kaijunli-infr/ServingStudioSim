# DeepSeek-V4.1-Flash-1017 on GB300 (rack 10 tray16), vLLM recipe TP4 — alignment capture campaign

Reference copies of the scripts used for the 2026-11-12 capture (Phase 0 of `top-add-new-arch`, driven by
`operate-run-alignment`). They were run from `/mnt/shared/home/dc/workspace/serving-studio/{scripts,dsv41-gb300}` on rack 10;
the paths inside them (`$SS`, `$GB300_ROOT`) refer to that tree, see `env.sh`.

- `make_configs.py` — writes the five `profile.json` passes (quadrant_nsys, quadrant_workload, diverse_nsys,
  diverse_workload, diverse_corpus) with the recipe flags verbatim and the launcher plumbing.
- `build_fork_env_v0310.sh` — vLLM alignment fork rebased onto v0.31.0 (`servingstudio-alignment-v0.31.0`,
  0ab18711), precompiled cu130 aarch64 wheel + editable Python install from the `alignment/profiler/vllm` submodule.
- `bootstrap_sim_env.sh`, `build_req_frontend.sh` — launcher venv (uv-managed CPython 3.12, no python3-dev on the hosts)
  and the Rust load generator.
- `run_pass.sh <pass>` — preflight (GPUs idle) + `launcher alignment profile`.
- `progress.md` — snapshot of the log of record (live copy: `logs/20261112_dsv41_flash_gb300_tp4/progress.md` on rack 10).
