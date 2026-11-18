# E1 V2 forced-release validation — run report (2026-11-18)

**Verdict under the frozen gates: PASS** (20 gates, 0 FAIL, 0 INCONCLUSIVE; `evidence/cpu/V2-GATES.txt`), outcome A: the
importer survived the forced release and wrote its report. One launch of `e1_v2_forced_release` on `pod4-gb300-10-tray16`
from `exp/profiler-lifecycle` dd4fbc6d under the conditional GO of 2026-11-18. The GO is consumed. No retry, no V3, nothing else
was run. Section 5 lists three observations a reviewer should weigh; they do not change a gate result and none is reinterpreted here.

## 1. Pre-launch checks (all PASS; `evidence/prelaunch/`)

| GO check | result | evidence |
|---|---|---|
| 1 HEAD dd4fbc6d, worktree clean | PASS | `prelaunch_checks_20261118T151954Z.log` |
| 2 code commit e32fbab7, pin commit 565206ee, 22 pinned trees/blobs = HEAD = code commit; gitlinks 5c20a259 / 0ab18711 | PASS | same |
| 3 `FREEZE_fr.sha256`, all 7 entries OK (manifest sha256 fcfbf5cd…) | PASS | same |
| 4 `session_runner` sha256 7f6eb81d45e6b98f3b6ebb56ab20b60ce8aaf4584aa3d4f25e3449b2d05a00c2 | PASS | same; again before/after the launch (§3) |
| 5 prebuilt path selected (`--frontend-provenance`, one launcher line), guard tests in the preflight | PASS | same + preflight |
| 6 exact read-only V2 preflight | 53 PASS, 0 FAIL (52 + the reservation record), 203 tests | `preflight_fr_tray16_driver_context_20261118T152707Z.log` |
| 7 four GPUs 0 MiB, no compute process, only daemonset pods on the node, reservation pod Running holding 4/4 GPUs | PASS | `reservation.json`, preflight census |
| 8 storage: `/tmp` 1164 GB free, CephFS 20 771 GB free | PASS | prelaunch log |
| 9 registry, cleanup, sampler, probe, signal recorder, stream preservation present and pinned; after the launch driver, sampler and probe registered and alive (`monitor.log`, 15:29:50Z) | PASS | prelaunch log, `pids_owned.tsv` |
| 10 packet ceilings = GO (90 min, 2700 s, 90 s, 5 / 5 / 10 GB) | PASS | prelaunch log |

Negative control (packet §10 step 2): with an independent lookalike present the preflight stopped (51 PASS, 2 FAIL: process
discovery and registry plan); the lookalike was removed through the scratch registry only (RESULT CLEAN).

## 2. Timeline (UTC)

| time | event | source |
|---|---|---|
| 15:22:34 | reservation pod Running on the node | kubectl |
| 15:28:21 | driver launched (tmux `e1v2-forced`); its own preflight PASS | `LAUNCHED.txt`, `preflight.log` |
| 15:29:20 | launcher start; driver, GPU sampler, probe registered | `run.log`, `pids_owned.tsv` |
| ≈15:52 / ≈15:54 | server startup complete / measurement begins | monitor |
| 15:55:00.05 | the one profiler-stop request, accepted | lifecycle |
| 15:56:01.90 | phase `importing` (flush took 62 s) | lifecycle |
| 15:56:02.28 | probe first sees `importing`; samples at 15:56:02.34 and 15:56:02.79, importer identity verified, phase `importing` in both | `probe.json` |
| 15:56:02.79 | probe writes the registered control line | `probe.json`, `lifecycle.control` |
| 15:56:03.073 | controller accepts it in phase `importing` (import 8 %, Nsight alive, no report yet) | lifecycle |
| 15:56:03.074 | SIGTERM to the identity-verified API server (pid 1556645), delivery recorded | lifecycle `gpu.api_server_sigterm`, `actions` |
| 15:56:01.91 → 15:56:04.09 | independent sampler: last "held" → first "released" (3 MiB, no compute process): at most 1.0 s after the request | `gpu_observations.jsonl` |
| 15:56:03.5 / 05.2 / 08.2 | probe: importer identity intact, alive | `probe.json` |
| 15:56:30.82 | report file last written (import finished 27 s after the SIGTERM) | file mtime |
| 15:57:03.11 | API server still alive after the 60 s grace: SIGKILL to the same identity-verified pid (frozen release action) | lifecycle `actions` |
| 15:57:04.12 | Nsight session closed, phase `report_written` | lifecycle |
| 16:02:12 | launcher rc=0 (in-session export and parse done) | `run.log` |
| 16:04:27 | closeout CLEAN | `closeout_tray16_20261118T160427Z.log` |
| 16:05:23–16:06:20 | reservation pod deleted; node has daemonset pods only, 0 GPUs allocated | `reservation_deleted.txt` |
| 16:06:59 | post-run (CPU-only) → PASS | `postrun_tray16_20261118T160659Z.log` |

## 3. Acceptance requirements of the GO

| # | requirement | result |
|---|---|---|
| 1 | pinned source and executable | launch from dd4fbc6d, alignment tree 7b4df784; executable 7f6eb81d… before and after the launcher, same inode/mtime/size, 0 compile lines; launcher record: prebuilt mode, guard armed, nothing blocked; binary gate 16/16 (S.binary_unchanged PASS) |
| 2 | one effective profiler-stop request | FR1 PASS: 1 request sent, 0 duplicates, 1 "Stopping profiler" line |
| 3 | verified active import immediately before SIGTERM | FR3 PASS (see §5.1) |
| 4 | signal delivered while import active | FR3/FR4 PASS: SIGTERM 15:56:03.074 in phase `importing`, report not yet written |
| 5 | importer survival, no destructive signal to Nsight | FR3b, FR4 PASS: only pid 1556645 (API server) was signalled; Nsight pid 1556601 identity intact in 2 controller and 3 probe checks; it finished the report |
| 6 | independent GPU release ≤ 90 s | FR5 PASS: ≤ 1.0 s (756 observations, none "held" after the first "released") |
| 7 | raw stream preserved, checksum, durability | FR7 PASS: 348 288 111 B, sha256 4739d6581d1b035bae1f1a0f15a3163bf8e708dc022527fb64228f1143db839c; tray original = durable copy = archive copy |
| 8 | report completion or offline recovery | both: in-session report (outcome A) and FR8 offline import of the durable stream rc 0 in 26 s |
| 9 | semantic validation | FR8b PASS: 2160 iterations, 8640 NVTX forward ranges, 12 250 832 kernel rows, four ranks (3 062 168 ×3, 3 064 328), capture boundaries present; recovered trace equals the in-session export |
| 10 | scientific completeness, identity, cleanup, storage | S.completeness_v2, S.extract, S.nsys_validation, S.config_identity, S.launcher_recovery, S.closeout, H, FR0, FR2, FR6, FR9 PASS |

Limits: reservation held ≈ 44 min (≤ 90); GPU-held 1664 s (≤ 2700); release ≤ 1.0 s (≤ 90); `/tmp`: new stream 0.33 GB + retained
V1 stream 0.33 GB, post-run scratch removed (its peak was not measured; packet estimate 2.4 GB); run directory 3.0 GB (≤ 5);
with the archive 6.3 GB (≤ 10). No escalation, no warning, no operator control line, no abort condition, no extension.

## 4. Closeout

GPUs 0 MiB ×4, no compute process; registry: 3 records, all gone, no lookalike; no tmux session; reservation pod deleted.
Archive `artifacts/phase1c-e1-v2fr-dsv41-flash-gb300-tp4-20261118` (3.3 GB, 141 files, `SHA256SUMS`, verified against the source,
read-only): run directory, preserved stream + `RETENTION.json`, operator logs. The tray originals of the V1 and V2 streams stay in
`/tmp`. The worktree's executable is still 7f6eb81d…; the V1 records and the frozen packet are unchanged.

## 5. Observations for the reviewer (not gate results)

1. **The probe's "import in progress" confirmation was not specific to the importer.** The frozen criterion is "CPU time of the
   recorded Nsight process tree, or reported progress, advanced while the phase stayed `importing`". It was met by +3 CPU ticks in
   the tree, and those came from the vLLM worker processes, which are in that tree; the Nsight process's own ticks (168) and the
   reported progress (8 %) did not move in the 0.45 s between the two samples, and no separate importer process appears in the
   sampled tree. That the import was really running at the SIGTERM is supported by other evidence: Nsight's own progress output
   stood at 8 % at the signal and went on to 100 %, and the report file was last written 27 s later (V1's whole import took 27 s).
   The gate FR3 passed as frozen; a stricter probe would need a criterion tied to the importer itself.
2. **The API server did not exit on SIGTERM.** The engine and the four workers exited at once (GPUs free within 1 s), but the API
   server process stayed until the controller's SIGKILL after the 60 s grace, as the frozen release action prescribes. The pending
   stop request therefore returned an error ("Remote end closed connection") at 15:57:03Z; FR1 accepts an error only at or after
   the release. The Nsight session closed 1 s after the server was gone, 61 s after the request (limit 300 s).
3. **The controller's progress reading stayed at 8 %** while it waited out the grace period; the server log shows the import
   reaching 100 % in that time. The lifecycle record's `import_progress_pct` after the release is therefore stale, not a stall.

## 6. Not done

No second attempt, no V3, no E1 retry, no identification partition, no fitting, no E2–E6, no production integration, no profiling
database write, no preset change. Stop for review.
