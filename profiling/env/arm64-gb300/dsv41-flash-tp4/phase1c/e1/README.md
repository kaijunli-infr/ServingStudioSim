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

## CPU-only E1 readiness audit and authorization packet v2 (2026-11-16, branch `phase1c-e1-audit`; no GPU use)

| file | purpose |
|---|---|
| `audit/E1-READINESS-AUDIT.md` | Parts A–C: frozen-evidence manifests, rank-dependence analysis of the accepted b=8 / b=48 traces, discrepancy investigation (empty reference section, configuration-identity differences, autotune tactics) |
| `E1-AUTHORIZATION-PACKET-v2.md` | Part D: revised full-E1 request (`e1_main_v3`, one launch, no nsys delay, 4 h reservation, GPU-held ≈ 60 min, raw ≈ 43 GB / ceiling 80 GB), measurement strategy, partitions, preflight/monitor/abort/cleanup, hashes, acceptance criteria, risks |
| `audit/FROZEN-EVIDENCE.sha256`, `audit/FROZEN-EVIDENCE-EXTERNAL.sha256` | checksums of the accepted Stage 1.5 evidence (repo files; archives and Phase 1A tables, made read-only on CephFS) |
| `audit/rank_dependence/`, `audit/identify/`, `audit/tactics/`, `audit/schedule/`, `audit/rank_kernels_b8_vs_run2.md` | outputs of the audit analyses (markdown + JSON) |
| `audit/preflight/` | read-only v3 preflight logs from tray16 in the driver context (21 PASS) |
| `analysis/e1_rank_dependence.py` | per-rank arrival/wait/intrinsic reconstruction, tails, critical-path shares, cross-rank covariance, autocorrelation, coefficient-free W_ind/W_cm composition (numpy + pyarrow) |
| `analysis/e1_autotune_tactics.py` | pairwise autotune-tactic agreement between launches and GPU time per tactic-dependent kernel family |
| `analysis/e1_schedule_sim.py` | full-E1 wave/capture-window simulation from measured Stage 1.5 rates (iterations, rows, storage, durations) |
| `make_e1_inputs_v3.py` → `configs_v3/e1_main_v3/profile.json`, `INPUTS_v3.sha256`, `E1-MAIN-V3-INPUTS.json` | v3 inputs derived mechanically from the accepted `configs_v2/e1_nsys_val` recipe (name, log_dir, trace path, delay removed) |
| `preflight_e1_main_v3.sh`, `run_e1_main_v3.sh`, `monitor_e1.sh`, `postrun_e1_main_v3.sh`, `collect_e1_main_v3.sh`, `hash_scripts_v3.sh` → `SCRIPTS_v3.sha256` | full-E1 procedure scripts (NOT run; the driver refuses any pass other than `e1_main_v3`) |

## e1_main_v3 capture attempt (2026-11-16, GO on e4ce9108) — FAILED in the profiler-stop path; no trace; GO consumed

| file | purpose |
|---|---|
| `runs/e1_main_v3/E1-MAIN-V3-INCIDENT-REPORT.md` | prerequisites, timeline, hard limits (GPU release exceeded by ≈ 71–93 s through the launcher's own failure path), surviving/lost artifacts, all gates FAIL, root cause (launcher `/stop_profile` bound 600 s × 2 vs Nsight stop ≈ 22–25 min; `_shutdown` SIGINT aborts the import and deletes the stream), monitor defect, cleanup verification, proposals |
| `runs/e1_main_v3/` | evidence: driver-context preflight (21 PASS), negative control (STOP by design), driver preflight/run logs, launch records, closeout (CLEAN), post-run gates log, laptop monitor log, `cpu/` checker outputs |
| raw | `$S/artifacts/phase1c-e1-main-dsv41-flash-gb300-tp4-20261116/run/e1_main_v3/` + `SHA256SUMS` (468 files, 379 MB: 450 routing npy, server log with 39 412 host iteration-metric lines, load generator results) |

## V1 authorization packet: bounded validation of the profiler-lifecycle controller (2026-11-17, CPU-only; GO of 2026-11-17)

Branch `exp/profiler-lifecycle` only (56c28ce3 → ece6dacf code → 0c626b7e/b0be99b5 packet → 9d4abd3f review corrections → the pin commit de4e9ae4 and the results commit that follows it re-pinned results). Nothing was run on a GPU; V1 itself needs its own GO.

| file | purpose |
|---|---|
| `E1-V1-AUTHORIZATION-PACKET.md` | the request: isolation, exact pins (commit/tree ids, Nsight CLI + QdstrmImporter sha256, interpreter, fork, recipe, inputs, scripts), the one-launch workload and expected coverage (both waves, ≈ 2160 iterations, ≈ 12.25 M rows), ceilings (reservation 90 min, GPU-held 2700 s controller-enforced, temporary 5 GB, persistent 5/10 GB), recipe-invariance evidence, the read-only identity preflight and its result, proof requirements → gates, control-file tests, the STOP rule, the 90/100-minute reconciliation, procedures |
| `V1-PIN.json`, `V1-PIN-identities.json`, `V1-PIN-identities.preflight.json` | pinned code trees/gitlinks/binary and the identities observed on tray16 by `alignment.profiler.preflight --write-pins` |
| `make_e1_inputs_v1.py` → `configs_v1/e1_v1_lifecycle/profile.json`, `E1-V1-INPUTS.json`, `INPUTS_v1.sha256` | the accepted `e1_nsys_val` recipe with no delayed start and explicit lifecycle bounds (soft 300 s, hard 600 s, ceiling 2700 s, max extension 120 s, stream preserved; no `tmp_dir`) |
| `preflight_e1_v1.sh`, `run_e1_v1.sh` (NOT run), `v1_control_probe.sh`, `monitor_e1_v1.sh`, `postrun_e1_v1.sh`, `closeout_e1_v1.sh`, `pin_v1.sh`, `hash_scripts_v1.sh` → `SCRIPTS_v1.sha256` (30 files) | V1 procedure scripts; `checks/e1_lifecycle_gates.py` evaluates gates L1–L10b from `lifecycle.json`, the independent `nvidia-smi` census files (`checks/e1_gpu_release_census.py`, gate L6c) and the offline import + sqlite export of the preserved stream (L10b: in-session kernel/NVTX counts), plus the folded scientific checkers; any FAIL = STOP |
| `runs/v1/preflight/` | CPU-only driver-context preflight results from tray16: 2026-11-17 22:37Z on ece6dacf (32 PASS, 0 FAIL) and 2026-11-18 01:48:03–01:48:59Z on 9d4abd3f (36 PASS, 0 FAIL (identity preflight 26 checks, 0 FAIL; 50 tests passed CPU-only with the guard line; GPU census: no compute process, 0 MiB on every GPU; discovery CLEAN; registry plan clean)); logs, identity records, pytest logs (CPU-only guard), GPU census |
| `runs/v1/cpu/` | test logs and diffs; `INCIDENT-tray14-fulltier-gpu-allocation.md` (the whole-tier run of 2026-11-17 on tray14 let torch allocate 684 MiB on GPU 0: a test run, not a profiling experiment, unauthorized, remediated by the CPU-only guard); `pytest_fullsuite_cpu_only_tray14_20261118T0137Z.log` (5151 passed, 78 skipped, 6 failed; guard: no device node opened, 0 MiB on every GPU; the failures are the cargo-dependent launcher_corpus test, the gdn numerical test and four test_l1_single_gemm energy tests that fail the same way CPU-only on the untouched audit worktree 9a28358c) |
| `tests/cpu_only_guard.py`, `tests/cpu_only.sh`, `tests/test_cpu_only_guard.py` | CPU-only test execution: devices hidden before torch loads, session fails (exit 3) if a `/dev/nvidia<N>` node is opened or torch.cuda initialises |
| code (`alignment/profiler/lifecycle.py`, `config.py`, new `preflight.py`; tests + `tests/fixtures/e1_nsys_val/`) | GPU-held ceiling (`gpu_held_ceiling_s`, any phase, not extendable), control-file rules (reason required, `max_extension_s`, complete lines, every line recorded), raw-stream discovery by descriptor scan, read-only identity preflight module |

## V1 run 2026-11-18: FAIL (STOP)

One launch of `e1_v1_lifecycle` from 77f410a5 on tray16 (GO of 2026-11-18). The capture completed (2160 iterations, 12 250 832 kernel rows, both waves, GPU-held 1713 s, independent GPU census clean) but the pinned gates L1, L4 and S.config_identity failed and L10/L10b were not attempted, so V1 is not accepted. Report: `runs/v1/run/E1-V1-RUN-REPORT.md`; evidence `runs/v1/run/evidence/`; pre-launch audit note `runs/v1/preflight/AUDIT-NOTE-preflight-failure-20261118T0146Z.md`. No retry; V2 not started.

## V1 gate remediation and evidence replay (2026-11-18, CPU-only)

Two separate verdicts: original V1 **FAIL / STOP** under the pinned gates (record unchanged); retrospective evidence assessment under the corrected gates **PASS** (25 gates, 0 FAIL). Corrected validators are new files (`checks/e1_lifecycle_gates_v2.py`, `checks/e1_launch_identity_v2.py`, `checks/e1_identity_policy_v1.json`, `checks/e1_gpu_watch.py`, `replay_e1_v1.sh`, `SCRIPTS_v1_replay.sha256`); the pinned ones are untouched. Report `runs/v1/replay/V1-REMEDIATION-REPORT.md`, manifest `runs/v1/replay/EVIDENCE-MANIFEST.sha256`.

## Controller improvements and V2 readiness (2026-11-18, CPU-only)

Records kept separate: original V1 **FAIL / STOP**; retrospective V1 evidence assessment **PASS, qualified** (`runs/v1/replay/`, first failed replay attempt kept). Controller: launch host identity, stream observations, final stream record after session close (never a completeness claim), release context; preflight distinguishes a retained inactive recovery stream from an active session. V2 request: `E1-V2-FORCED-RELEASE-PACKET.md` (frozen gates `checks/e1_forced_release_gates.py`, scripts `*_e1_fr.sh`, `SCRIPTS_fr.sha256`, `V2-FR-PIN.json`). **Open blocker:** the read-only V2 preflight fails (45 PASS, 1 FAIL) because the V1 launcher rebuilt the load generator at launch; see packet §12 and `runs/v1/ADDENDUM-session-runner-rebuild.md`. Nothing GPU is authorized.

## session_runner provenance repair and V2 readiness correction (2026-11-18, CPU-only)

Option 2: the pinned load-generator executable (sha256 7f6eb81d…) is restored and verified byte-identical to the reference; the executable the V1 launcher rebuilt (95a1526c…) is preserved as incident evidence and not re-pinned (`runs/v2/binary/REPAIR-REPORT.md`). Authorized launches use the prebuilt contract (`launcher alignment profile … --frontend-provenance BINARY-PROVENANCE.json`): exact hash, pinned source and build-state record verified before any engine start, no cargo, hash verified again at the end. V2 gates strengthened without changing thresholds (packet §3, §5). Exact read-only V2 preflight on tray16: **53 PASS, 0 FAIL** (earlier failing attempts kept). V1 records unchanged. Nothing GPU is authorized.

## Final V2 gate rehearsal and packet freeze (2026-11-18, CPU-only)

No GPU was used and nothing was launched. The pinned load-generator executable is unchanged (sha256 7f6eb81d…). `rehearse_e1_fr.sh` (outputs in `runs/v2/rehearsal_final/`) ran the complete strengthened V2 post-run on the preserved V1 fixture: verdict **FAIL**, and that because V1 forced no release during its import (all six forced-release gates fail, the recovery and science gates pass on the real data; checked by `e1_fr_rehearsal.py expect-v1`). It then ran 50 synthetic fixtures through the command-line validators: 3 positive (every gate PASS) and 47 negative (11 INCONCLUSIVE, 36 FAIL, each at its registered gate), covering an import that completes before the SIGTERM, missing importer observations, late GPU release, binary mutation, destructive importer signals and incomplete recovery. A prebuilt launch now arms a process-wide build-tool guard, so no helper can start cargo, rustc or rustup; gate `S.binary_unchanged` is the tested checker `checks/e1_frontend_binary_gate.py`. Exact read-only V2 preflight on tray16 at 565206ee: 52 PASS, 0 FAIL. `E1-V2-FORCED-RELEASE-PACKET.md` is frozen (§13 rehearsal, §14 freeze record, `FREEZE_fr.sha256`). The V1 records are unchanged. V2 GPU execution is not authorized.

## V2 forced-release run 2026-11-18: PASS (frozen gates), outcome A

One launch of `e1_v2_forced_release` from dd4fbc6d on tray16 (conditional GO of 2026-11-18, consumed). The registered probe requested the release 1.2 s into the import; the API server got the identity-verified SIGTERM at 15:56:03Z; the independent sampler saw the GPUs free within 1.0 s; the importer was not signalled and wrote its report; the preserved stream (sha256 4739d658…) re-imports offline to the same 2160 iterations and 12 250 832 kernel rows. 20 gates, 0 FAIL, 0 INCONCLUSIVE. Report with three reviewer observations (probe confirmation not importer-specific, API server needed the SIGKILL after the grace, stale progress reading): `runs/v2/run/E1-V2-RUN-REPORT.md`; evidence `runs/v2/run/evidence/`. No retry; V3 and everything after it are not authorized.

## V2 closeout and V3 preparation 2026-11-18 (CPU-only)

V2's frozen verdict stays PASS (20/20) and is accepted with qualifications; packet, gate outputs and run report are unchanged. Added: `runs/v2/run/E1-V2-REVIEW-ADDENDUM.md` (three qualifications; GPU release stated as the sampled interval, observed released no later than 2.12 s after the request; peak post-run scratch unmeasured; stream hash, archive checksums, recovery evidence). `runs/v2/closeout/E1-V2-API-SERVER-SHUTDOWN-ANALYSIS.md`: the API server waited in uvicorn's connection wait on the still-open `/stop_profile` request until the SIGKILL; reproduced by a CPU fixture; nothing changed. `runs/v2/closeout/E1-IMPORT-PROBE-DESIGN.md` with `checks/e1_import_probe.py` and `tests/test_e1_import_probe.py` (23 tests): the importer is a child of the detached Nsight agent, not of the CLI; the new probe identifies it and counts only its own CPU/I-O. `E1-V3-SCALE-PACKET-DRAFT.md` with `make_e1_inputs_v3scale.py`, `traces/e1_v3_scale.csv`, `configs_v3scale/`, `E1-V3-SCALE-INPUTS.draft.json`: draft only (24.5 M rows, normal stop). No GPU run is authorized.

## V3 CPU-only implementation and proposed freeze packet 2026-11-18

No GPU run. Workload as approved: the accepted 56-request trace twice (112 requests, four waves, b=8 and b=48, expected 4320 iterations and 24.5 M kernel rows), V2 config with only name, log_dir and trace path changed; controller, vLLM fork and the 60 s server grace unchanged. `E1-V3-SCALE-PACKET.md` is the proposed freeze packet (`FREEZE_v3scale.sha256`, verified by the preflight); it supersedes the draft. Tools: `run_e1_v3scale.sh`, `preflight_e1_v3scale.sh`, `monitor_e1_v3scale.sh`, `closeout_e1_v3scale.sh`, `archive_e1_v3scale.sh`, `postrun_e1_v3scale.sh`, `checks/e1_v3scale_gates.py` (gates G0–G12, explicit phase clocks, GPU release judged on the upper edge of the sampled window, unmeasured peaks INCONCLUSIVE), `checks/e1_import_probe.py`, `checks/e1_mem_watch.py`, `checks/e1_storage_watch.py`; pins `V3S-PIN.json`, `V3S-PIN-identities.json`, `SCRIPTS_v3scale.sha256`, `INPUTS_v3scale.sha256`. Evidence: `runs/v3scale/` (three rehearsal attempts, all kept; regression logs; negative control and read-only preflight). V3 GPU execution is not authorized.

## V3 safety and execution-path closure 2026-11-18 (CPU-only)

No GPU run. `E1-V3-SCALE-PACKET.md` is the final freeze packet (`FREEZE_v3scale.sha256`). Added since 28dc7f01: confirmed deadline semantics (flush 300 s, import 300 s, stop → report 600 s from the original stop request; soft warning informational; hard deadline never extended; gate G4 checks it); `checks/e1_supervisor.py`, the registered consumer of the breach markers and sampler heartbeats with one bounded response (the registered release line through the unchanged controller during finalisation, the pinned registry cleanup otherwise); importer ceiling lowered to 128 GiB with a 448 GiB preflight requirement; importer ambiguity latches; `checks/e1_gpu_watch_v3.py`; rehearsal mode of the driver with `e1_v3scale_mock_launcher.py`, `rehearse_e2e_v3scale.sh`, `e1_v3scale_e2e_check.py`. Evidence: `runs/v3scale/e2e` (four scenarios met; five earlier attempts kept), `runs/v3scale/rehearsal` (99 of 99 fixtures), `runs/v3scale/preflight_final`. `alignment/` and `launcher/` are unchanged. V3 GPU execution is not authorized.
