# e1_main_v3 — the authorized full-E1 capture FAILED in the profiler-stop path (2026-11-16); GO consumed, no Nsight trace

**Outcome.** Kevin's GO (one `e1_main_v3` capture from `phase1c-e1-audit` e4ce9108) was executed once on rack 10 tray16. All
pre-launch prerequisites passed, the server started in 23 min, the profiler was armed before the first request, and the whole
workload (450 requests, 36 waves) completed in 896 s with 0 failures. The launcher then failed while stopping the profiler: its
`/stop_profile` call is bounded to 600 s and is retried once, Nsight's stop path for this trace took longer than those 20 min,
the launcher raised `RuntimeError: vLLM /stop_profile failed: timed out` and its shutdown routine SIGINT'd the nsys process group
while the report import was at ≈ 49 %. **The `.nsys-rep` was never written and the raw stream (`/tmp/nsys-report-4604.qdstrm`,
≈ 5 GB) was deleted by Nsight on abort.** No GPU-side trace exists; every post-run gate fails; the identification partition was
not written. Routing records (450 npy, 221 MB), the full server log with 39 412 per-iteration host metrics lines and 901 request
timing lines, the load-generator results and the launch records survive and are archived with checksums. The GPUs were released
by the launcher's own failure handling 21 min 11 s after workload completion, i.e. **the 20-minute GPU-release limit was exceeded
by about 71 s** (details in §4 and §6). Everything else stayed inside the limits. The GO is consumed; no retry was made.

## 1. Authorization and prerequisites (packet v2 §5; all before 08:53Z)

| # | requirement | evidence | result |
|---|---|---|---|
| 1 | HEAD e4ce9108, clean worktree | `wt-e1stage1` on `phase1c-e1-audit`, `git status` empty at 08:46Z | PASS |
| 2 | `SCRIPTS_v3.sha256` (25), `INPUTS_v3.sha256` (6) | `sha256sum -c` OK 08:46Z and again inside both preflights | PASS |
| 3 | frozen evidence hashes | `audit/FROZEN-EVIDENCE.sha256` 246 OK, `-EXTERNAL` 11 OK; raw archives re-verified (e1_nsys_val 86 files, e1_b8_recovery 40 files) 08:50Z | PASS |
| 4 | req-frontend 5c20a259 / tree 06a4a61e / binary 7f6eb81d… | gitlink, submodule HEAD, clean, `session_runner` sha256 7f6eb81d45e6b98f… 08:46Z | PASS |
| 5 | four GPUs idle, no foreign compute processes | nvidia-smi 4 × 0 MiB, 0 compute apps, no tmux server 08:46:41Z | PASS |
| 6 | exclusive reservation | pod `dsv41-gb300-reserve-tray16` Running 08:49:16Z, the only GPU pod on the node (checked 08:50Z and every 5 min during the run) | PASS |
| 7 | exact tmux/tee-context preflight | `preflight_tray16_driver_context_20261116T085207Z.log`: 21 PASS 0 FAIL, own pipeline + own tmux INFO, discovery CLEAN, registry plan clean | PASS |
| 8 | independent-lookalike negative control, verified cleanup | `negative_control_tray16_20261116T085125Z.log`: lookalike pid 433461 → FOUND + LOOKALIKE, PREFLIGHT FAIL rc 1 by design; scratch-registry cleanup CLEAN, 0 alive, no tmux | PASS |
| 9 | profiler armed synchronously, no delay, no bounded capture | preflight checks (no `capture_start_delay_seconds`, no `capture_duration_seconds`, nsys block == Phase 0 quadrant); driver guard re-checked trace sha 2e147402… / 450 rows / no delay before launch | PASS |
| 10 | registry and cleanup protections | registry absent before launch; driver registered pid 462764 (`driver:e1_main_v3`) at 08:53:37Z; pinned `e1_pids.py` 1c8aa34d | PASS |
| 11 | storage | CephFS 21 100 GB free, tray root 1165 GB free | PASS |

Reservation applied 08:49:16Z (hard limit 12:49Z). Launch 08:53:22Z via the neutral-path wrapper (`tmux e1main-v3`, driver
`run_e1_main_v3.sh`); the driver's own preflight passed again (`preflight.log`), launcher tree e4ce9108, config sha 8bd51ebe0f8e3363,
trace sha 2e147402b5a24f9e, `session_runner` 7f6eb81d.

## 2. Timeline (UTC, 2026-11-16)

| time | event | source |
|---|---|---|
| 08:53:37 | driver start, launcher started | run.log |
| ≈ 09:16:50 | server ready ("Application startup complete"; first seen by the 09:17:29 sample) — startup ≈ 23 min 20 s | monitor_laptop_v3.log, server log |
| 09:22:40 | `/start_profile` → "Profiler started"; "frontend ready; measurement begins" | server log 5267, run.log 588 |
| 09:22:40 → 09:37:36 | 450/450 requests completed, 0 failed, elapsed 896.4 s (simulated 895.6 s) | run.log, load_generator/summary.json, drive_summary |
| 09:37:36 | `/stop_profile` requested ("Stopping profiler...") — **workload completion** | server log 120601 |
| 09:37:36 → 09:47:36 | engine frozen inside the stop (EngineCore `shm_broadcast` "no available block" every 60 s); first `/stop_profile` times out after 600 s | server log |
| 09:47:36 | second `/stop_profile` (launcher `finally` emergency stop) | server log 120613 |
| ≈ 09:52 | Nsight: "Capture range ended", "Generating /tmp/nsys-report-4604.qdstrm", import starts (`QdstrmImporter`, 100 % CPU, 61 GB RSS); `.nsys-rep` created with 0 bytes 09:52:01 | server log 120621, process sample 09:56:03 |
| ≈ 09:55:38 | import at 49 % | server log 120626 |
| 09:57:36 | second `/stop_profile` times out → `RuntimeError`; launcher `_shutdown`: SIGINT to the nsys process group | run.log traceback |
| 09:57:36 → 09:58:06 | Nsight aborts: "rpc Shutdown … canceled because the timeout period is expired (30 sec)"; qdstrm and 0-byte rep removed; nsys exits; launcher prints "workload done", driver `rc=1` 09:58:06 | server log, run.log 2441-2442 |
| 09:58:47 | EngineCore/API server receive SIGTERM and shut down (orphaned for ≈ 40 s after the driver exit) | server log |
| 09:58:47 → 09:59:09 | GPUs released (4 × 0 MiB at the 09:59:09 sample) — **21 min 11 s – 21 min 33 s after workload completion** | monitor_laptop_v3.log |
| 10:02:31 | closeout: 4 × 0 MiB, 0 compute apps, discovery CLEAN, registry plan GONE, no tmux | closeout_tray16_20261116T100231Z.log |
| 10:03:40 | reservation pod deleted; no GPU pod on the node | kubectl |
| 10:04 – 10:07 | raw copy + SHA256SUMS (468 files, 379 MB), post-run gates recorded (all FAIL) | collect, postrun_20261116T100658Z.log |

## 3. What the capture produced and what is lost

| artifact | state |
|---|---|
| `.nsys-rep`, `.sqlite`, `parsed.json`, kernels parquet, host_timeline, `profile_result.json` | **absent** — the report was never written; the raw stream was deleted by Nsight on abort; nothing was found on the tray (`find / -xdev -name '*.nsys-rep' -o -name '*.qdstrm'`, /tmp) |
| `routed_experts/*.npy` | 450 files, 221 MB (one per measured request) |
| server log | 120 648 lines: 39 412 `VibeSimAlignmentIteration` lines (host-side iteration metrics, schema 4, up to iteration 39485), 901 `VibeSimAlignmentRequestTiming` lines (450 measured + 451 warm-up) |
| load generator | `summary.json`, `replay.jsonl` (450), warm-up replays; `drive_summary.json` with `replay_start/end_monotonic_ns`, `reached_idle: true` |
| launch records | `*_launch.json`, `profile.json`, `trace.csv`, `autotune_configs.6adeb850201b.json` |
| metrics jsonl, token corpus | not produced (`_finalize_profile` never ran) |

Raw copy: `$S/artifacts/phase1c-e1-main-dsv41-flash-gb300-tp4-20261116/run/e1_main_v3/` + `SHA256SUMS` (468 files, 379 MB).
Small copies in `runs/e1_main_v3/` (this directory). The host-side iteration metrics and routing records could support a CPU-only
host-timing/routing analysis of the 36 cells; they contain no per-rank GPU information and cannot replace the E1 trace.

## 4. Hard limits (GO §3)

| limit | measured | result |
|---|---|---|
| reservation ≤ 4 h | 08:49:16 → 10:03:40 = 1 h 14 min | OK |
| GPU-held ≤ 90 min | 08:53:37 → ≈ 09:58:50 ≈ 65 min | OK |
| raw artifacts ≤ 80 GB | 379 MB (the deleted stream was ≈ 5 GB on the tray root) | OK |
| startup ≤ 30 min | ≈ 23 min 20 s | OK |
| workload completion ≤ 25 min after measurement begins | 14 min 56 s | OK |
| **GPU release ≤ 20 min after workload completion** | **≈ 21 min 11 s – 21 min 33 s** (release by the launcher's failure path) | **EXCEEDED by ≈ 71–93 s** |
| other abort rules (foreign GPU process, engine error before shutdown, GPU pod other than the reservation, disk) | none occurred; EngineDeadError-family lines: none | OK |

## 5. Post-run gates (GO §4) — all FAIL for want of the trace (`postrun_20261116T100658Z.log`, rc 6)

| gate | result |
|---|---|
| 36 waves captured; contiguous iterations; first iteration present | FAIL — no `parsed.json`, no trace (census rc 1) |
| ≥ 400 clean pure-decode iterations per cell / per-wave minima | FAIL — not measurable |
| four-rank NVTX and collective alignment; end spread within thresholds | FAIL — `profile_result.json` / `parsed.json` missing |
| complete routing joins and corpus shapes | FAIL — `profile_result.json` missing (completeness v2 rc 1); the 450 npy exist but the token corpus and metrics jsonl were never packed |
| JIT exclusions within limits | FAIL — not measurable |
| shared-wave workload identity vs e1_b8_recovery / e1_nsys_val | FAIL (rc 1) — the checker needs the finalised request records |
| checksummed export | FAIL — nothing to export |
| resource and cleanup verification | PASS (closeout CLEAN, pod deleted) except the GPU-release limit above |
| configuration identity vs e1_nsys_val | configs identical apart from the four allowed keys; launch-record differences are the enumerated ones (nsys output path, `capture_start_delay_seconds`) |

Identification estimators, rank-dependence analysis, contrasts and kernel-tactic distributions: **not run** (no tables). The
identification partition (`partition-1c.json`) was **not written**. Phase 1A/1B evidence untouched.

## 6. Root cause and contributing factors

1. **Launcher bound on the profiler stop** (`alignment/profiler/vllm_server.py::set_cuda_profile`): `timeout = 600.0` for
   `/stop_profile`, called once in the normal path and once more in the `finally` emergency path (`alignment/runner.py` ≈ 634/645),
   i.e. at most 20 min for Nsight to finish inside the worker. Stage 1.5 traces (≤ 6 M kernel rows) stopped in seconds and the
   Phase 0 `diverse_nsys_run2` trace (97 M rows) within the bound; this trace (≈ 216 M rows expected) needed ≈ 14.5 min to flush
   the capture (09:37:36 → ≈ 09:52) plus an import that was at 49 % after ≈ 3.5 min, i.e. ≈ 22–25 min in total.
2. **Shutdown aborts Nsight.** `_shutdown` SIGINTs the process group and escalates to SIGKILL after 120 s; its docstring assumes
   SIGINT "lets nsys finalize", but Nsight's agent RPC timed out after 30 s and the stream and partial report were deleted. The
   failure path therefore destroys exactly the artifact the pass exists to produce.
3. **My estimate in packet v2 §4** ("nsys stop + report finalisation ≤ 20 min", scaled from Phase 0 run2 ≈ 10 min for 2.7 GB) was
   wrong by a factor ≈ 1.2 and, more importantly, the audit did not identify the launcher's 600 s × 2 stop bound as a stop-path
   risk (risk 8 covered only export/parse time). The 20-minute GPU-release limit adopted in the GO was derived from that estimate.
4. **Monitor defect** (pinned `monitor_e1.sh`): `$(grep -c … || echo 0)` yields two lines when nothing matches, so its
   startup/GPU-held/engine-error conditions misfire (one false "GPUs still held" alert at 08:55Z, and the startup-limit check could
   never fire). The limit logic was re-implemented in the neutral-path wrapper (`scratch/mon_v3.sh`) without touching the pinned
   script; the laptop log of both samplers is `monitor_laptop_v3.log`. The "workload done" marker it relies on is printed by the
   launcher only after the stop returns, so the monitor's "workload not finished" alerts from 09:49Z were proxy lag (the load
   generator had finished at 09:37:36); the real limit was not violated.

## 7. Operator conduct at the limit

The GPU-release limit fell at 09:57:36Z. From 09:52Z I had established that the workload had completed on time and that Nsight was
finalising with visible progress; at 09:56:03Z a process sample showed the importer at 100 % CPU; at 09:57:40Z an attempt to copy
the raw stream to CephFS as preserved evidence found it already deleted (the launcher had raised at 09:57:36Z). No operator
termination was executed: by the next sample (09:58:06Z) the driver had exited rc 1 and the launcher's own shutdown had already
released the GPUs at 09:58:47–09:59:09Z. Had the launcher not failed, an identity-verified cleanup at the limit would have killed
nsys during the import and destroyed the trace in the same way; I record that I had not yet executed it when the limit passed.

## 8. Cleanup verification (GO §7)

`closeout_tray16_20261116T100231Z.log`: 4 × 0 MiB, 0 compute apps; process discovery 11/11 PASS (no vllm, nsys, session_runner,
launcher, driver shells, watchers, monitors, cwd/cmdline references, experiment tmux sessions); registry plan: driver GONE,
nothing owned alive, 0 lookalikes; no tmux server. Reservation pod deleted 10:03:40Z; no GPU pod on the node. Note:
`closeout_e1.sh` hard-codes the Stage 1.5 run root, so its items were run through the wrapper `scratch/closeout_wrap_v3.sh` against
the v3 registry with the same pinned checks.

## 9. Files in this directory

`preflight_tray16_driver_context_20261116T085207Z.log`, `negative_control_tray16_20261116T085125Z.log`, `preflight.log` (driver's
own), `run.log` (launcher output incl. the traceback), `profile.json`, `trace.csv`, `*_launch.json`, `*_drive_summary.json`,
`load_generator_summary.json`, `autotune_configs.6adeb850201b.json`, `closeout_tray16_20261116T100231Z.log`,
`postrun_20261116T100658Z.log`, `monitor_laptop_v3.log`, `cpu/` (checker outputs), this report.

## 10. What a further attempt would require (proposals only; nothing is authorized)

* A launcher change so that the stop path cannot destroy the trace: `/stop_profile` waited on without a fixed bound (poll the
  worker/nsys state, or a bound sized to the trace, ≥ 60 min for ≈ 200 M rows) and a shutdown that never signals nsys while
  `QdstrmImporter` is running; alternatively stop the capture from the nsys side (`nsys stop`/session) with the server idle.
* A measured stop-path duration for a trace of this size, which cannot be obtained without a GPU run; the next request should
  therefore budget GPU release ≥ 45 min after workload completion or capture in smaller sessions (e.g. per KV length, ≈ 70 M rows).
* A new GO; this one is consumed.
