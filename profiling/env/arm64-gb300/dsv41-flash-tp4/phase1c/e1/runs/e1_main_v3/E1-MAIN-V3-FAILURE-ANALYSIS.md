# e1_main_v3 — failure analysis of the profiler-stop path (CPU-only; GO of 2026-11-17, parts A and B)

Incident commit `d366315c` (branch `phase1c-e1-audit`); launcher code at the time of the run = `e4ce9108` (the `alignment/`
tree is identical at `d366315c`). Archive `$S/artifacts/phase1c-e1-main-dsv41-flash-gb300-tp4-20261116` (`SHA256SUMS`, 468 files,
verified 468/468 OK on 2026-11-17 and made read-only). All times UTC, 2026-11-16, from the frozen server log
(`run/e1_main_v3/profile/vllm/dsv41_flash_1017_gb300_tp4_e1_main_v3_server.log`, 120 648 lines), the driver `run.log`, the laptop
monitor log and the closeout logs. Nothing in the archive or in the frozen acceptance thresholds was modified.

## A. Evidence preserved

| item | where | state |
|---|---|---|
| raw run copy | `$AR/run/e1_main_v3/` + `$AR/SHA256SUMS` | 468/468 OK (`cd $AR/run && sha256sum -c ../SHA256SUMS`), `chmod -R a-w`, 0 writable entries |
| launcher code that ran | `runs/e1_main_v3/incident/launcher_snapshot_e4ce9108/{runner.py,vllm_server.py,nsys_capture.py,config.py}` + `BLOBS.txt` (git blob ids c7bb7aba, a0f5cf5c, 22127f1a, a65393bb) | copied from `git show e4ce9108:…`; the stop bound (`timeout = 120.0 if active else 600.0`), the second stop in `finally` and `_shutdown` (killpg SIGINT → 120 s → SIGKILL) are in these files |
| stop-path log excerpts | `incident/server_log_stop_path_excerpt.txt` (lines 120595–120650), `incident/run_log_stop_path_excerpt.txt` (lines 2370–2450) | verbatim copies with line numbers |
| failure timeline | this document, §B; incident report §2 | |
| irrecoverability check | `incident/irrecoverability_check_tray16.log` (2026-11-17T11:44:38Z) | `nsys sessions list` empty; no `*.qdstrm` / `*.nsys-rep` in `/tmp`, on the tray root disk newer than 2026-11-16 08:00Z, in the CephFS run root or the artifacts tree; the run's `nsys/` directory is empty (mtime 09:57, the SIGINT minute). `QdstrmImporter` 2025.6.3 is present at `/opt/nvidia/nsight-systems/2025.6.3/host-linux-armv8/` for future offline imports. **The trace is irrecoverable: no raw stream and no report exist anywhere reachable.** |

## B. Timeline of the stop path

| time | event | owner (process) | source | status |
|---|---|---|---|---|
| 09:22:40 | `/start_profile` → "Profiler started"; measurement begins | launcher main thread → API server 477312 → EngineCore 477870 → Worker_TP0..3 478137–478140 | server log 5267, run.log 588 | established |
| 09:37:36 | **workload completion**: 450/450 requests, 0 failed, elapsed 896.4 s; last measured iteration 39485 | load generator (req-frontend `session_runner`, child of the launcher) | run.log 2377, server log 120595–120599 | established |
| 09:37:36 | **first `/stop_profile`** sent (`runner.py:634`, urllib timeout 600 s); API server logs "Stopping profiler..." | launcher main thread → API server route handler (`api_router.py:31`) | server log 120601, run.log traceback | established |
| 09:37:36 → | the request travels `AsyncLLM.stop_profile → core_client.profile_async(False) → call_utility_async("profile") → EngineCore.profile() → executor collective RPC → Worker.profile(is_start=False) → cudaProfilerStop` | API server → EngineCore → 4 workers | vLLM source (fork-env v0.31.0), not logged | established path, inferred timing |
| 09:38:37 … 09:58:38 | EngineCore prints "No available shared memory broadcast block found in 60 seconds" every 60 s (21 lines): the engine is blocked inside the collective RPC — the workers are inside `cudaProfilerStop` | EngineCore 477870 (blocked); workers (in Nsight agent code) | server log 120604–120618, 120626–120631 | established (blocked); the *cause* (workers in cudaProfilerStop) is a hypothesis supported by 6/6 earlier captures in which "Profiler stopped." appears only after the import bar reaches 100 % |
| 09:37:36 → ≈09:51:40 | **profiler flushing** (`--flush-on-cudaprofilerstop`, default true): ≈ 14 min for ≈ 216 M rows; no log line marks the flush | Nsight agent inside the workers + nsys CLI | absence of lines; "Capture range ended" appears after the 09:51:38 line | established duration bounds (09:47:36 < end ≤ 09:52:01) |
| 09:47:36 | first request times out after 600 s → `RuntimeError("vLLM /stop_profile failed: timed out")` raised at `runner.py:634`; control enters `finally` | launcher | run.log traceback 2436–2440 | established |
| 09:47:36 | **second `/stop_profile`** (the `finally` emergency stop, `runner.py:645`, 600 s); API server logs a second "Stopping profiler..." | launcher → API server | server log 120613 | established (accepted and logged) |
| 09:47:36 → | second request queued behind the first on the EngineCore utility path (the engine is blocked); it never reached a worker | API server / EngineCore input queue | no second "Profiler stopped." ever; EngineCore blocked throughout | hypothesis (vLLM does not log queue depth) |
| ≈09:51:40–09:52:01 | "Capture range started/ended in the application."; "Generating '/tmp/nsys-report-4604.qdstrm'"; a 0-byte `…e1_main_v3.nsys-rep` is created (09:52:01) | nsys CLI (prints into the same stdout as the server) | server log 120619–120621; incident report process/file sample | established |
| 09:52:01 → | **report import**: `[1/1] [N%] …nsys-rep` progress; 49 % reached before 09:55:38; `QdstrmImporter` at 100 % CPU, 61 GB RSS at 09:56:03 | `QdstrmImporter` (separate process; parent presumed the nsys CLI) | server log 120622–120626, process sample | established progress; parentage hypothesis |
| 09:57:36 | second request times out (600 s); the `RuntimeError` is swallowed; `_shutdown(proc)`: `os.killpg(os.getpgid(nsys_pid), SIGINT)` | launcher | snapshot `runner.py` 645–650, `_shutdown`; run.log end time | established (code + timing) |
| 09:57:36 → 09:57:40 | **raw stream deleted**: `/tmp/nsys-report-4604.qdstrm` absent by 09:57:40 (operator `cp` attempt); the 0-byte rep removed | nsys CLI, reacting to SIGINT | operator log (incident report §2), `nsys/` dir mtime 09:57 | established |
| 09:57:36 → 09:58:06 | nsys: "rpc Shutdown(.Agent.ShutdownRequest) … is canceled because the timeout period is expired (30 sec)"; nsys exits; the launcher's buffered "[profile] workload done" line is flushed; driver `rc=1` at 09:58:06 | nsys CLI; launcher | server log 120629–120630, run.log 2441–2442 | established |
| 09:57:36 | **no SIGINT reached the vLLM processes**: no API-server or EngineCore signal/shutdown line at that time (both log signal receipt, see 09:58:47) | — | server log 120626–120631 | established (absence); the reason — Nsight runs the target in its own process group, so `killpg(nsys pgid)` did not include it — is a hypothesis consistent with Nsight's `--kill` documentation ("to the target application's process group") |
| 09:58:37 | driver wrapper discovery: API server 477312 and EngineCore 477870 alive; 4 workers hold 4 × 261 242 MiB; registry plan **refused** (driver pid is the cleanup's own ancestor) and **nothing was signalled** | `e1_pids.py` (read-only here) | run.log 2444–2450 and following | established |
| 09:58:47 | **SIGTERM** received by EngineCore and the API server in the same second; API server stops the engine client, force-kills EngineCore (`utils.py:663`), shuts down HTTP | API server 477312 (process manager) | server log 120632–120648 | established receipt; **source is a hypothesis**: not the registry cleanup (its log says nothing was signalled), not the launcher (exited 09:58:06), not tmux (would be SIGHUP, and the server was in its own session); the remaining candidate is Nsight's session teardown (`--kill=sigterm` default) delivered 41 s after the CLI's exit |
| 09:58:47 → 09:59:09 | **GPU release**: workers gone, 4 × 0 MiB at the 09:59:09 sample — 21 min 11 s – 21 min 33 s after workload completion (GO limit 20 min) | kernel/driver after process exit | monitor_laptop_v3.log 563 | established |
| 10:02:31 | closeout CLEAN: 0 compute apps, discovery CLEAN, registry GONE, no tmux | `closeout_e1.sh` | closeout log | established |
| 10:03:40 | **reservation cleanup**: pod `dsv41-gb300-reserve-tray16` deleted; node allocation 0/4 | operator (`kubectl`) | incident report §2 | established |

### Did the two stop requests overlap or duplicate an in-progress finalisation?

Yes. The second request was accepted by the API server at 09:47:36 while the first was still executing in the workers (the engine stayed
blocked until at least 09:58:38). It was a duplicate of an in-progress finalisation, added nothing, and extended the launcher's own
bound by a second 600 s. Had the first stop completed in time, the second would have executed `cudaProfilerStop` on an already stopped
session (effect under `--capture-range-end=stop`: unknown; no evidence). The new controller sends exactly one request per launch and
counts, but never re-sends, further stop calls (`http.duplicate_requests_suppressed`).

### What finished the trace off

1. The launcher's bound (2 × 600 s) was shorter than the finalisation (≥ 20 min 30 s elapsed at the SIGINT, import at > 49 %).
2. The shutdown signalled the *profiler* (SIGINT to nsys's process group) as ordinary timeout handling. Nsight reacted by aborting the
   import and deleting its stream within 4 s, then timed out its agent RPC after 30 s and exited. The code comment assumed SIGINT "lets
   nsys finalize"; that holds for an idle session, not for a session mid-import.
3. The vLLM server was not in the signalled group and survived the launcher by 41 s with the GPUs held; it was ended by a SIGTERM whose
   origin is not recorded.

### Projection (hypothesis, for the budget in packet v3)

Import rate: 49 % in ≤ 217 s (09:52:01 → 09:55:38) → full import ≈ 7.4 min → import end ≈ 09:59:25; plus the "Profiler stopped." return
and the normal shutdown ≈ 1 min. Absent the SIGINT, the report would most likely have been written ≈ 09:59:30–10:00:30, i.e. a
finalisation of ≈ 22–23 min after workload completion, ≈ 2 min after the moment it was destroyed, and ≈ 3 min over the GO's 20-minute
GPU-release limit. That limit was therefore infeasible for this trace size with this capture recipe, independently of the launcher bug.

## Calibration of finalisation time (server logs of the accepted runs; "Profiler stopped." = import 100 %)

| run | kernel rows | stop → "Profiler stopped." | `.nsys-rep` | s per M rows |
|---|---|---|---|---|
| e1_b8_recovery | 5.89 M | 45 s | 170 MB | 7.6 |
| e1_nsys_val | 6.36 M | 63 s | 223 MB | 9.9 |
| quadrant_nsys (Phase 0) | 25.6 M | 163 s | 697 MB | 6.4 |
| quadrant_run1 (Phase 0) | 25.6 M | 160 s | — | 6.3 |
| diverse_nsys (Phase 0) | ≈ 26 M | 164 s | 768 MB | ≈ 6.3 |
| diverse_nsys_run2 (Phase 0) | 97.15 M | 567 s (flush ≥ 300 s, then import) | 2.83 GB (29 B/row) | 5.8 |
| **e1_main_v3** | ≈ 216 M (plan) | **≥ 1230 s, destroyed at import > 49 %**; projected ≈ 1320–1380 s | 0 B | ≥ 5.7 |

The stop time is close to linear in rows (≈ 6 s per M rows beyond 25 M rows) with the flush taking the larger share for big traces;
the 216 M-row projection (≈ 22–23 min) is consistent with the observed incomplete run. Worst case for the budget: 2 × = 46 min.

## Established vs hypothesis — summary

Established by logs/files: every timestamp in §B marked "established"; the two stop requests and their 600 s bounds; the engine being
blocked for ≥ 20 min; the import progress and importer resource use; the SIGINT target (nsys pgid, from code) and the vLLM processes
not receiving it; the stream deletion within 4 s; the SIGTERM at 09:58:47 and the GPU release by 09:59:09; the registry cleanup
signalling nothing; the irrecoverability of the trace.

Hypotheses (consistent with all evidence, not directly logged): (a) `cudaProfilerStop` blocks the worker until Nsight's import has
completed (the 6/6 pattern); (b) Nsight placed the target in a separate process group, which is why `killpg` missed vLLM; (c) the
09:58:47 SIGTERM came from Nsight's session teardown (`--kill=sigterm`); (d) the import would have completed ≈ 09:59:25 and the report
≈ 09:59:30–10:00:30; (e) the second stop request was queued and never executed in a worker; (f) `QdstrmImporter` is a child of the
nsys CLI. (a), (b) and (c) decide where a GPU-release action may be aimed and are therefore listed as proof requirements for the
bounded validation in packet v3 §9; the remediation does not depend on (d)–(f).
