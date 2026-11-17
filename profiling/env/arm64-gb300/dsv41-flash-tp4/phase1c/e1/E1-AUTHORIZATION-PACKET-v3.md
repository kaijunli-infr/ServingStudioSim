# E1 authorization packet v3 — profiler-lifecycle remediation (CPU-only; nothing here authorizes GPU use)

Prepared 2026-11-17 under the GO "CPU-only incident analysis, surviving-artifact audit and profiler-lifecycle remediation design".
It supersedes packet v2 §4, §6 and §8 where they concern the stop path, GPU release and limits; v2 §1–3, §5, §7, §9–10 (workload,
measurement strategy, partitions, pre-launch requirements, post-run gates, provenance, scientific risks) stand unchanged.
Companion documents: `runs/e1_main_v3/E1-MAIN-V3-FAILURE-ANALYSIS.md` (parts A–B), `runs/e1_main_v3/surviving/E1-MAIN-V3-SURVIVING-
ARTIFACTS.md` (part E), `runs/e1_main_v3/lifecycle/` (patch diff, test logs). No GPU was used for any of it; no coefficient was fitted;
the identification partition was not written; the frozen branches, the archive and the acceptance thresholds are unchanged.

## 1. Exact patched launcher commit and diff

| item | value |
|---|---|
| branch / worktree | `exp/profiler-lifecycle` at `$S/wt-lifecycle`, created from the incident commit `d366315c` |
| patched commit | **`56c28ce3`** ("Profiler lifecycle: idempotent unbounded stop, observable finalisation, no signals to an active importer") |
| diff | `git diff d366315c..56c28ce3`: 11 files, +2221 / −33; code part saved as `runs/e1_main_v3/lifecycle/patch_56c28ce3.diff`, tests as `tests_56c28ce3.diff`, `diffstat_56c28ce3.txt` |
| new modules | `alignment/profiler/lifecycle.py` (809 lines, `ProfilerStopController`, `FinalizePolicy`, `Phase`), `alignment/profiler/procident.py` (188 lines, identity-verified process handling) |
| changed | `alignment/runner.py` (+101/−~20: controller around the stop, `TMPDIR`, `import_stream` recovery), `alignment/profiler/vllm_server.py` (`timeout=None` = unbounded), `alignment/profiler/config.py` (`NsysConfig` finalize fields, `tmp_dir`, `qdstrm_importer`, `finalize_policy()`), `alignment/profiler/nsys_capture.py` (`qdstrm_importer_path`, `import_stream`), `launcher/alignment.py` (`profile --import-stream`); `tests/test_frontend_measurement_gate.py` (fakes adapted to the new stop contract: a Popen result with a pid, `set_cuda_profile(..., timeout=None)`, a contract-only stop controller; the assertion on the start/stop order around the measurement boundary is unchanged) |
| unchanged by construction | nsys prefix, server argv, server env (other than `TMPDIR` when `nsys.tmp_dir` is set — unset in every existing config), model execution, request scheduling, routing collection, workload inputs; golden test `test_launch_recipe_is_unchanged_against_the_frozen_e1_main_v3_launch` rebuilds argv + prefix from the frozen `profile.json` and compares to the frozen `…_launch.json` of the incident run |
| not touched | Rust production code, profiling database, presets, `e1_pids.py`, the frozen E1 scripts (`SCRIPTS_v3.sha256`) and configs |

What the patch changes in behaviour (the nine requirements of GO part C):

1. **Idempotent stop** — `request_stop()` sends exactly one `/stop_profile` per launch from a daemon thread; the timer path, the
   workload-done path and the `finally` path all call it; further calls increment `http.duplicate_requests_suppressed`.
2. **Three distinct observable events** — acceptance (`stop_accepted`: the API server's "Stopping profiler..." line), capture
   finalisation (`capture_ended` → `stream_written`: Nsight's "Capture range ended" / "Generating '<qdstrm>'"), import completion
   (`importing` with a percentage → `report_written`: non-empty `.nsys-rep` **and** the HTTP return or "Profiler stopped." line).
3. **No destructive signal to an active importer as timeout handling** — the only timeout action is `release_gpus`, aimed at the
   vLLM API server by its recorded identity; the nsys CLI / `QdstrmImporter` is signalled only after `report_written`, or when the
   capture never armed, or when no import activity has been seen for `stall_timeout_s` and the stop is not in progress; a shutdown
   that finds nsys still importing leaves it running and records a warning with the pids.
4. **Partial raw artifacts preserved** — as soon as nsys names the `.qdstrm`, a hard link is made beside it
   (`<stream dir>/<name>.raw-preserve/<name>.qdstrm`, same filesystem, zero copy); Nsight's own unlink no longer destroys the bytes;
   `profile --import-stream <qdstrm>` imports it offline with the matching `QdstrmImporter` and resumes finalisation (CPU only).
5. **GPU release and finalisation separately observable** — `lifecycle.json` carries `phase`, `import_progress_pct`, `report`,
   `gpu.released_at` (from server-process liveness and `nvidia-smi --query-compute-apps`), `nsys.alive`, `session_closed_at`.
6. **Bounded monitoring, escalation, intervention** — `soft_deadline_s` (escalation record + log line), `stall_timeout_s`
   (warning), `hard_deadline_s` (release action), control file `lifecycle.control` with `extend <s> <reason>`,
   `release-gpus <reason>`, `note <text>`; every command and its result is logged in the status file.
7. **No silent extension** — `FinalizePolicy.validate()` rejects `hard < soft`, non-positive values and unknown actions; an extension
   exists only as an operator command with a reason, recorded in `deadlines.hard_extension_s` and `control_commands`.
8. **Identity-verified cleanup** — `procident.signal_verified` refuses own pid / pid 1 / own ancestry, re-reads `/proc/<pid>/stat`
   immediately before `os.kill`, and returns `gone|changed|refused|signalled|eperm`; the server pids come from the server's own log
   lines and are accepted only if they are descendants of the nsys process the launcher spawned.
9. **Repeated stops, stalled imports, launcher exceptions, dropped sessions** — covered by 1, 6, the `finally` path (which now
   finishes an in-flight stop under the same deadlines before shutting down) and `ignore_sighup()` after the spawn (children run in
   their own session already; the launcher itself now survives a dropped tmux/ssh).

## 2. CPU test results (rack 10 tray14, Python 3.12.13 aarch64, pytest 9.2.0; no GPU)

| suite | result | log |
|---|---|---|
| `tests/test_profiler_lifecycle.py` (new, 16 tests) | **16 passed** in 3.9 s | `runs/e1_main_v3/lifecycle/pytest_lifecycle_56c28ce3.log` |
| launcher-adjacent: `test_alignment_launcher.py`, `test_alignment_v1.py`, `test_alignment_nsys_evidence.py`, `test_e1_pids_cleanup.py`, `test_frontend_measurement_gate.py` + the new file | **213 passed, 1 skipped** | `pytest_launcher_suites_56c28ce3.log` |
| whole `tests/` (cpu tier) | **5129 passed, 77 skipped, 2 failed, 4 errors** in 2 min; the 4 errors and 1 failure are `FileNotFoundError: 'cargo'` (Rust toolchain absent on the rack), the other failure is a pre-existing numerical assertion in `test_gdn_spec_decode.py`; **every one of them also fails on the unpatched baseline `d366315c`** (which additionally fails 4 environment-dependent tests in its own worktree) | `fullsuite_compare_56c28ce3_vs_d366315c.log` |
| ruff (`E,F,I,UP`, line 100) on all changed/new files | clean | this session |

Scenario coverage required by GO part D (all fixture-based: simulated clock, scripted server log, fake `/stop_profile`, fake process
table, recording signaller, injected disk usage):

| D scenario | test | what it proves |
|---|---|---|
| import > 20 min | `test_import_longer_than_twenty_minutes_completes_without_any_signal` | 25.5 min stop completes; soft escalation at 20 min recorded; **no signal sent**; preserved stream survives Nsight's unlink |
| repeated `/stop_profile` | `test_repeated_stop_calls_send_exactly_one_http_request` | 6 calls → 1 HTTP request, 5 suppressed |
| timeout during import | `test_hard_deadline_during_import_stops_the_server_not_the_importer` | hard deadline: SIGTERM to the API server pid only, nsys pid never signalled, `gpu.released_at` set, phase stays `importing`, stream preserved |
| launcher shutdown while importer active | `test_launcher_shutdown_while_importer_active_waits_then_leaves_importer` | `shutdown()` finishes the stop under its deadlines, never SIGINTs the group or nsys, leaves the importer with a warning |
| import failure with recoverable raw artifacts | `test_import_failure_with_recoverable_raw_stream_then_offline_import` | importer dies + stream unlinked → `stop_failed`, preserved bytes intact; `nsys_capture.import_stream` rebuilds the report offline |
| GPU-release deadline expiry | `test_gpu_release_deadline_expiry_is_recorded_and_not_extended_silently`, `test_operator_extension_is_explicit_and_logged` | release at exactly hard deadline (+ one poll + grace), `hard_extension_s == 0`; an `extend 600 <reason>` is honoured and logged |
| disk-space exhaustion | `test_disk_space_exhaustion_is_reported_before_the_stop`, `test_disk_full_during_import_fails_with_preserved_stream` | low-free warnings for stream and report dirs before the stop; ENOSPC mid-import → `stop_failed` with the stream preserved and the error line recorded |
| partial export and restart/recovery | `test_restart_recovery_reads_status_and_import_stream_resumes`, `test_cli_import_stream_implies_resume` | `lifecycle.json` readable by a second process (pids with identities, preserved path); `run_profile(import_stream=…)` never calls `require_gpu`, imports, then finalises |
| stalled import | `test_stalled_import_is_warned_but_never_signalled` | stall warning, no signal, completes when progress resumes |
| recipe unchanged | `test_launch_recipe_is_unchanged_against_the_frozen_e1_main_v3_launch` | argv/prefix/env identical to the incident launch record; `finalize_policy().validate()` passes on the frozen config |
| cleanup identity | `test_procident_refuses_own_ancestry_and_pid_reuse` (real `/proc`) | own pid and parent refused; changed starttime → `changed`; a real child is signalled once and then reported `gone` |
| unbounded timeout | `test_stop_profile_timeout_none_is_unbounded` | `timeout=None` → `urlopen(timeout=None)`; defaults untouched |

## 3. Profiler lifecycle state diagram

```
                      request_stop() [once]                      API server logs
  CAPTURING ───────────────────────────▶ STOP_REQUESTED ─────────────────────────▶ STOP_ACCEPTED
   (start_profile ok)                    POST in daemon thread   "Stopping profiler..."     │
                                                                                            │ nsys: "Capture range ended"
                                                                                            ▼
                                                                                      CAPTURE_ENDED  (flush; workers blocked
                                                                                            │          in cudaProfilerStop)
                                                                   nsys: "Generating '<x>.qdstrm'" │  → hard link  <dir>/<name>.raw-preserve/
                                                                                            ▼
                                                                                      STREAM_WRITTEN
                                                                                            │ "[1/1] [N%] …nsys-rep"
                                                                                            ▼
                                         stall_timeout_s w/o progress → WARNING       IMPORTING ──── progress % in lifecycle.json
                                                                                            │
                        non-empty .nsys-rep  AND (HTTP 200 | "Profiler stopped." | nsys exited) │
                                                                                            ▼
                                                                                      REPORT_WRITTEN  ──▶ shutdown(): SIGTERM server (verified pid)
                                                                                                              → wait nsys exit ≤ session_close_timeout_s
     any phase: nsys exits w/o report | HTTP error + no report 30 s  ──▶ STOP_FAILED           → SIGINT/SIGKILL nsys only now, if lingering
                                                              (preserved stream → `profile --import-stream`)

  timers (from stop_requested_at):  soft_deadline_s → ESCALATION (record, keep waiting)
                                    hard_deadline_s (+ operator extensions) → release_gpus(): SIGTERM API server, grace, SIGKILL recorded
                                        survivors; never nsys; then wait ≤ session_close_timeout_s for the importer; return current phase
  control file (any time):          extend <s> <reason> | release-gpus <reason> | note <text>   (all logged)
  GPU release observed separately:  gpu.released_at ← server processes gone ∧ no compute pid of this server
```

## 4. GPU-release strategy and proof requirements

Strategy: the GPUs are held by the vLLM worker processes, not by nsys. Releasing them means stopping the *server* by its own
identity (SIGTERM to the API server, whose process manager stops EngineCore and the workers; SIGKILL to recorded survivors after
`server_term_grace_s`). The importer is a separate process and is left alone. The release is either the normal shutdown after
`report_written`, or the hard-deadline action, or an operator `release-gpus` command. Nothing else in the launcher signals anything.

Phase-dependent consequence (this is the trade the operator makes at a hard deadline):

| phase at release | expected effect on the trace | evidence status |
|---|---|---|
| `stop_accepted` / `capture_ended` (flush running inside the workers) | the stream is being written by the Nsight agent *in the worker processes*; killing them truncates it → trace lost or partial | inferred from Nsight's architecture; **unproven** |
| `stream_written` / `importing` | the stream is complete on disk and preserved; the importer should finish on CPU, or the preserved stream is imported offline | **hypothesis (a)/(b) of the failure analysis; must be proven on a small capture before it is relied on** |
| `report_written` | none | established (normal path) |

Proof requirements before a full attempt (executed in the bounded validation of §10, each needs its own GO):

P1. On a small capture, `lifecycle.json` shows every phase in order with timestamps, and the preserved hard link appears at
    `stream_written` and is still importable after nsys unlinks the original.
P2. With a deliberately short `hard_deadline_s`, `release_gpus` during `importing` stops only the server (pids and identities in
    `actions`), the GPUs read 0 MiB within `server_term_grace_s` + 30 s, and the importer either completes (report valid: kernel row
    count equals an undisturbed capture of the same recipe ± 0.1 %) or the preserved stream imported offline yields that report.
P3. The 09:58:47-type SIGTERM: after a normal `report_written` shutdown, no process of the launch remains and no unexplained signal
    appears in the server log; after P2, the record shows which signals the launcher sent and whether Nsight sent any of its own.
P4. The capture recipe check (`test_launch_recipe_is_unchanged…`) passes against the launch record of each validation run.

## 5. Storage budgets

Temporary (tray root NVMe; `nsys.tmp_dir` to be set to a dedicated directory, proposed `/tmp/e1-nsys`, so the stream and its hard link
share one filesystem; tray16 root had 1165 GB free on 2026-11-16):

| item | size | basis |
|---|---|---|
| `.qdstrm` for 216 M rows | ≈ 5 GB (observed estimate, incident) — budget **10 GB** | incident report §4 |
| preserved hard link | 0 extra while the stream exists; ≤ 10 GB retained until the report is validated, then removed by the operator | design |
| importer RAM | 61 GB RSS at 49 % → budget **150 GB**; tray RAM 958 GB | process sample 09:56:03 |
| policy floors | `min_free_bytes_stream` 50 GiB (stream dir), `min_free_bytes_report` 20 GiB (report dir) — warnings before the stop | `FinalizePolicy` defaults |

Persistent (CephFS run directory; v2 ceiling 80 GB unchanged):

| item | size |
|---|---|
| `.nsys-rep` | ≈ 6.3 GB (29 B/row × 216 M; v2 said 6.2) |
| `.sqlite` export | ≈ 32 GB (v2 schedule simulation) |
| parsed.json + kernels parquet + host_timeline | ≈ 2.6 GB |
| routing npy + corpus + logs/metrics | ≈ 0.7 GB (routing 221 MB measured) |
| preserved `.qdstrm` copied to CephFS *only on a failed finalisation* | ≤ 10 GB |
| **total** | **≈ 42 GB nominal, ≤ 52 GB with a preserved stream; ceiling 80 GB** |

## 6. Worst-case finalisation and import estimates (calibration in the failure analysis)

| quantity | point estimate | worst case used for limits | basis |
|---|---|---|---|
| flush (stop accepted → stream written) | ≈ 14.5 min | 29 min | observed 09:37:36 → ≈ 09:52:00 |
| import (stream written → report) | ≈ 7.4 min | 15 min | 49 % in ≤ 217 s |
| "Profiler stopped." return + normal shutdown | ≈ 1 min | 2 min | accepted runs |
| **finalisation, workload done → GPUs free** | **≈ 23 min** | **46 min** | 2 × point estimate; the row count (216 M) is confirmed by the host-side iteration reconciliation (38 108 in-window iterations, 460 350 decode tokens, as planned) |
| sqlite export + parse (CPU, GPUs free) | ≈ 23 + 63 min | 3 h | v2 §4, unchanged |
| GPU-held total (startup 23–25 + capture 16.5 + finalisation) | ≈ 63 min | ≤ 90 min | |

## 7. Explicit reservation and abort limits (replace v2 §6 "GPUs still held 20 min after workload done")

| limit | value | who acts | action |
|---|---|---|---|
| startup without "Application startup complete" | 30 min | operator | `release-gpus` via control file is not available before the controller exists → `e1_pids.py cleanup --execute` on the registry (unchanged) |
| workload not finished after "measurement begins" | 25 min | operator | same |
| **finalisation soft deadline** (`finalize_soft_deadline_s`) | **30 min** after the stop request | controller | escalation record + log line; operator checks `import_progress_pct` |
| **finalisation hard deadline** (`finalize_hard_deadline_s`) | **50 min** after the stop request (≈ 2.2 × point estimate) | controller | `release_gpus` (server SIGTERM by pid); importer left running; stream preserved |
| operator extension | only `extend <s> <reason>` in `lifecycle.control`; each one logged; proposed cap: one extension ≤ 20 min, only while `import_progress_pct` is advancing | operator | — |
| importer still running after GPU release | `finalize_session_close_timeout_s` 10 min, then left untouched with a warning | operator | decide: wait (CPU only, GPUs already free) or `profile --import-stream` from the preserved stream later |
| stalled import | `finalize_stall_timeout_s` 15 min without progress | controller → operator | warning; the operator may `release-gpus` |
| GPU-held total | 100 min | operator | `release-gpus` command (never by pattern) |
| run directory | 80 GB | operator | unchanged |
| disk floors | root ≥ 100 GB, CephFS ≥ 1000 GB at preflight; 50 GiB / 20 GiB warnings at the stop | preflight / controller | unchanged / new |
| reservation | 4 h hard limit, nominal ≈ 2 h 45 min | operator | unchanged |

Config values for a future `e1_main_v4` profile (not created; requires a GO): `nsys.finalize_soft_deadline_s: 1800`,
`nsys.finalize_hard_deadline_s: 3000`, `nsys.finalize_stall_timeout_s: 900`, `nsys.finalize_on_hard_deadline: release_gpus`,
`nsys.preserve_raw_stream: true`, `nsys.tmp_dir: /tmp/e1-nsys`; everything else identical to `configs_v3/e1_main_v3/profile.json`.

## 8. One large capture or segmented captures — decision

**One large capture**, unchanged recipe and frozen 450-row trace, with the budget of §6–7. Reasons: (i) the frozen trace's inter-wave
gaps (≈ 12.5 s) cannot absorb a per-range stop (seconds to minutes of flush each), so segmentation would change the workload's
arrival schedule and the frozen plan; (ii) `--capture-range-end=repeat[:N]` is a recipe change whose overhead was never validated
(Stage 1/1.5 validated exactly the current prefix); (iii) the measured finalisation (≈ 23 min, worst 46 min) fits in the reservation
with the GPUs released by the controller if it does not. Segmented captures remain the fallback if the bounded validation refutes
hypothesis (a)/(b) (GPU release during import loses the trace) *and* a 50-minute GPU-held finalisation is unacceptable; they would
need their own overhead validation first.

## 9. Preflight, monitoring, cleanup and recovery procedures (additions to v2 §5–6)

Preflight (CPU, in the driver context, before any GPU GO): HEAD = the authorized commit on `exp/profiler-lifecycle` (or its merge);
`pytest tests/test_profiler_lifecycle.py` passes on the tray; `nsys.tmp_dir` exists on the root NVMe with ≥ 100 GB free;
`QdstrmImporter` at `<nsys install>/host-linux-armv8/` has the CLI's version; `lifecycle.json`/`lifecycle.control` absent in the run's
`nsys/` dir; all v2 §5 items unchanged.

Monitoring (laptop, every 60 s, read-only ssh): print `phase`, `import_progress_pct`, `deadlines.hard_at`, `gpu.released_at`,
`stream.preserved_size`, `warnings[-1]` from `lifecycle.json`; `nvidia-smi` memory; the reservation pod; alert on `soft_deadline`
and `hard_deadline` escalations and on `stop_failed`. The old "workload done" grep is replaced by `stop_requested_at`.

Cleanup: the controller's own shutdown (server by pid, nsys only after the report) is the normal path; the registry-based
`e1_pids.py cleanup` remains the operator's tool for everything outside the controller (driver shell, stray processes) and keeps
its lookalike-STOP rule; `closeout_e1.sh` unchanged, plus: `lifecycle.json` final phase recorded, preserved stream removed only after
the report's row count is validated.

Recovery: `stop_failed` or a dead launcher with a preserved stream → `python -m launcher alignment profile <profile.json>
--import-stream <preserved .qdstrm>` on any tray (CPU; implies `--resume`; uses the matching importer) → then the normal `--resume`
finalisation; nothing GPU-side is repeated.

## 10. Proposed bounded validation before another full E1 attempt (proposals; each needs its own GO; GPU-held figures are estimates)

| step | recipe | GPU-held | proves |
|---|---|---|---|
| V1 | `e1_nsys_val` recipe (6.4 M rows, 3 waves) with the patched launcher; one `note` and one `extend 60 test` written to the control file during the import | startup ≈ 23 min + capture ≈ 2 min + finalisation ≈ 1 min ≈ 27 min | P1, P4; phases and control logging on a real Nsight session; preserved link lifecycle |
| V2 | same recipe with `finalize_hard_deadline_s: 20` so the hard deadline fires during `importing` | ≈ 27 min | P2, P3: GPU release during import; importer completion or offline import from the preserved stream; row-count equality with V1 (± 0.1 %) |
| V3 | Phase 0 `quadrant_nsys`-size recipe (25.6 M rows) with the patched launcher | ≈ 30 min | third calibration point of the finalisation curve under the patched launcher; storage and RAM scaling |
| full E1 (`e1_main_v4`) | only after V1–V3 pass and are accepted; §7 limits | ≈ 63 min nominal, ≤ 90 min | — |

Acceptance of V1–V3 (proposed): all phases observed in order; no signal to nsys before `report_written`; `release_gpus` in V2 lists
only the server pids with identities; GPUs 0 MiB within 90 s of the release; report row counts V1 = V2 (offline or in-line); the
recipe test passes on each launch record; capture overhead of V1 vs the accepted `e1_nsys_val` within the Stage 1.5 band (host-side
TPOT ± 2 %), since the launcher change must not have touched the measurement.

## 11. Surviving-artifact analysis (part E) — what the 2026-11-16 run left behind

`runs/e1_main_v3/surviving/E1-MAIN-V3-SURVIVING-ARTIFACTS.md` + `surviving.json` (script `analysis/e1_main_v3_surviving.py`, CPU):
450/450 requests, 450 routing files (1024 × 40 × 6, ids < 384), 450 request timings, 38 108 host-side iteration records in the
measured window reconciling exactly to 450 × 1023 decode tokens and 20 275 200 prefill tokens; all 36 (b, KV, wave) cells complete and
contiguous with ≥ 740 pure-decode iterations each; arrival deviation p50 1.6 ms, max 3.1 ms; no plan deviation. Routing: pooled
per-layer entropy 7.43 bits (uniform 8.59), max/mean load ≈ 30, w0-vs-w1 JSD inside the permutation null for b ≥ 4; bootstrap CIs
are over requests (cells with 1–2 requests have none). Host-side timings are profiler-perturbed and routing-instrumented and are not
model-validation data. Autotune tactics differ 47–64 % between this and every earlier launch. Not reconstructable without Nsight:
per-rank kernel timings, f_ind by cell, collective alignment, critical-path/wait decomposition, tactic GPU time, the K = 163 proxy
and every pre-registered kernel-level contrast. Nothing was fitted; the identification partition was not written.

## 12. Not authorized and still open

No GPU execution, no full E1 retry, no identification-partition write, no model fitting, no E2–E6, no Rust integration, no profiling
database writes or preset changes were done or are requested by this packet. Open: hypotheses (a)–(c) of the failure analysis (where
a release action may be aimed without losing the trace; the 09:58:47 SIGTERM source) — resolvable only by V1–V2; whether a merge of
`exp/profiler-lifecycle` into the audit branch is wanted before any validation GO (it changes launcher code that `SCRIPTS_v3.sha256`
does not cover, so a new scripts/inputs manifest would be needed for a v4 run).
