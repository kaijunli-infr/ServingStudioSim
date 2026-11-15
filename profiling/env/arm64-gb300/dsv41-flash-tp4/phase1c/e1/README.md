# phase1c/e1 — E1 authorization packet (decode routing and rank-arrival capture)

Everything needed to review, and later run, GPU experiment E1 of `../GPU-EXPERIMENT-MATRIX.md` under the frozen
`../VALIDATION-PROTOCOL.md`. **Nothing here has been executed on a GPU, and the launcher patch is not applied to any
branch.** Read `E1-AUTHORIZATION-PACKET.md` first; its section 12 lists the decisions requested with the go.

| path | what |
|---|---|
| `E1-AUTHORIZATION-PACKET.md` | the packet (ten review items, deviations, verbatim matrix / protocol excerpts) |
| `make_e1_inputs.py` → `traces/`, `configs/`, `E1-INPUTS.json`, `INPUTS.sha256` | deterministic request traces and launcher profile configs |
| `preflight_e1_host.sh` | read-only host preflight (run on the tray as dc) |
| `run_e1.sh <pass>` | the driver; refuses to start unless the preflight passes; one pass per invocation |
| `checks/e1_completeness.py` | correctness / completeness checks, dry-run overhead comparison |
| `analysis/e1_identify.py` | pre-registered estimators (no fitting) |
| `make_partition_1c.py`, `collect_e1.sh` | post-capture bookkeeping (partition file, artifact collection) |
| `patch/` | the 46-line launcher opt-in for recording routes beside a timing pass, its tests and applier |
| `out/` | CPU smoke outputs on frozen calibration rows, tray16 preflight log |
| `SHA256SUMS` | hashes of every file above (`sha256sum -c SHA256SUMS`) |

Stage 1 (dry-run pair, accepted at c7e8ed9b): `E1-STAGE1-REPORT.md`, `runs/e1_dry_*`, `runs/stage1/`, `analysis/e1_pair_compare.py`.

Stage 1.5 (CPU-only remediation, branch `phase1c-e1-stage15`; GPU passes await authorization):

| path | what |
|---|---|
| `E1-STAGE15-REMEDIATION.md` | the Stage 1.5 report (start here); `E1-STAGE15-GPU-AUTHORIZATION-REQUEST.md` the request |
| `checks/e1_completeness_v2.py`, `tests/test_e1_completeness_v2.py` | corrected, versioned checker; old vs new on Stage 1 in `runs/stage15/checker_v1_vs_v2.md` |
| `checks/e1_config_identity.py`, `tests/test_e1_config_identity.py` | configuration identity between arms (configs and launch records) |
| `patch/apply_stream_optin.py`, `patch/apply_stream_flag_rs.py`, `patch/req-frontend-stream-flag.patch` | explicit `workload.stream` / `session_runner --stream` (same wire mode for both arms) |
| `make_e1_inputs_v2.py` -> `configs_v2/`, `INPUTS_v2.sha256`, `E1-STAGE15-INPUTS.json` | Stage 1.5 passes: off_v2, on_v2, off2_v2 (control), nsys_val |
| `analysis/AUTOTUNER-INVESTIGATION.md`, `analysis/autotune_cache_diff.py` | why every launch re-tunes ~19 min; tactic drift between launches |
| `E1-OVERHEAD-METHODOLOGY.md`, `analysis/e1_pair_compare_v2.py` | Gate A preserved; prospective block-bootstrap Gate B; off/off reading rules |
| `E1-OFFOFF-PROTOCOL.md`, `E1-NSYS-VALIDATION-PROTOCOL.md`, `checks/e1_nsys_validation.py` | the prepared GPU passes and their checkers |
| `E1-STAGE15-BUDGET.md` | occupancy and storage |
| `preflight_e1_stage15.sh`, `run_e1_stage15.sh`, `collect_e1_stage15.sh` | Stage 1.5 drivers (Stage 1 scripts unchanged) |
| `BINARY-PROVENANCE.json`, `patch/request-factory-e1-stage15-stream-flag.bundle`, `checks/stream_wire_check.py` | pinned `session_runner` source (req-frontend branch `e1-stage15-stream-flag` = 5c20a259), build hash, mock-server wire-mode verification |
| `runs/stage15/engine-dead-error.md` | Stage 1 teardown `EngineDeadError`: explanation and abort conditions |
| `runs/stage15/cpu/` | every CPU output referenced above |
| `E1-STAGE15-REPORT.md` | **Stage 1.5 execution report (passes 1-3 run 2026-11-15 on tray16)**; `runs/stage15/e1_dry_*_v2/` derived files, `runs/stage15/cpu/final/` analysis; `postpass_e1_stage15.sh`, `analyze_e1_stage15.sh` the gate and analysis drivers |
| `E1-NSYS-VAL-REPORT.md`, `runs/stage15/e1_nsys_val/`, `postrun_e1_nsys_val.sh`, `analysis/e1_nsys_rank_kernels.py`, `analysis/e1_nsys_capture_window.py` | Stage 1.5 pass 4 (`e1_nsys_val`, 2026-11-15): pre-GPU verification, timeline, all nsys validation gates PASS on the b=48 cell, per-rank execution / collective wait / kernel identities, capture-window limitation (b=8 cell not captured), fork proposal |
| `E1-B8-RECOVERY-PACKET.md`, `make_e1_inputs_b8.py` → `traces/e1_b8_recovery.csv`, `configs_v2/e1_b8_recovery/`, `INPUTS_b8.sha256`, `E1-B8-RECOVERY-INPUTS.json`; `analysis/e1_capture_window.py`, `checks/e1_pids.py`, `checks/e1_process_census.sh`, `checks/e1_workload_identity.py`, `hash_scripts_b8.sh` → `SCRIPTS_b8.sha256`, `preflight_e1_b8.sh`, `run_e1_b8.sh`, `postrun_e1_b8.sh`, `closeout_e1.sh`, `collect_e1_b8.sh`, `runs/b8_recovery/`, repo `tests/test_e1_capture_window.py`, `tests/test_e1_pids_cleanup.py` | **b=8 recovery capture, CPU preparation only (2026-11-15; GPU execution needs a separate GO)**: census of the 1120 captured `e1_nsys_val` iterations, capture-window proof (no delayed start; simulation on the four Stage 1.5 timelines + regression tests), workload identity, full process census for preflight/closeout, one-pass driver, ceilings and frozen acceptance criteria, fork `kaijunli-infr/request-factory` (branch `e1-stage15-stream-flag` = 5c20a259) |

Regenerate / re-check on tray14 (CPU only):

```
export HOME=/mnt/shared/home/dc; S=$HOME/workspace/serving-studio; D=$S/ServingStudioSim/profiling/env/arm64-gb300/dsv41-flash-tp4
cd $D/phase1c/e1 && python3 make_e1_inputs.py --out . && sha256sum -c INPUTS.sha256
$S/gb300/venv/bin/python analysis/e1_identify.py --tables $S/phase1a-tables/quadrant_nsys --label quadrant_nsys --calibration-only $D/phase1a/partition/partition.json --out out/smoke_quadrant --boot 50
$S/gb300/venv/bin/python checks/e1_completeness.py --profile $S/artifacts/phase0-dsv41-flash-gb300-tp4-20261112/run/quadrant_nsys/profile --parsed --out out/smoke_quadrant_completeness.json
```
