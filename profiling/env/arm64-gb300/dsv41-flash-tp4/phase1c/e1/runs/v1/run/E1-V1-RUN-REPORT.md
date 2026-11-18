# E1 V1 run report — `e1_v1_lifecycle`, 2026-11-18, tray16 — **V1 FAIL (STOP)**

One launch under the V1 GO from `exp/profiler-lifecycle` 77f410a5. The capture itself completed and every scientific checker of the
trace passed, but **three pinned gates failed (L1, L4, S.config_identity) and L10/L10b were therefore not attempted by the pinned
post-run**. Per the packet §9 and the GO, that is a STOP: V1 is **not accepted**, nothing was retried, V2 is not started. The GO is
consumed. All evidence is preserved; nothing recoverable was deleted (the preserved stream is still in `/tmp` on tray16 and in the
archive).

## Decision record

| item | value |
|---|---|
| verdict (pinned `postrun_e1_v1.sh`) | `V1 ACCEPTANCE: STOP` — 24 gates, 3 FAIL, L10 NOT RUN |
| classification | **FAIL** on the pinned gates. The three failed conditions are stated exactly below, with what the record shows about each. Whether any of them is a gate-specification defect rather than a lifecycle defect is for review; no gate was changed or re-run. |
| retries | none |
| next step | none without a new GO |

## Times

| event | UTC |
|---|---|
| reservation pod create requested / Running / deleted | 02:49:04 / 02:49:20 / 03:31:40 (42 min 36 s; limit 90 min) |
| negative control (independent lookalike → preflight STOP, 34 PASS / 2 FAIL, lookalike never signalled by the preflight; removed via its scratch registry) | 02:49:47–02:50:37 |
| fresh driver-context preflight (36 PASS, 0 FAIL) | 02:51:10–02:51:58 |
| driver launched (tmux `e1v1-lifecycle`); its own preflight 36 PASS; `start` line | 02:52:36; 02:53:23 |
| launcher `launch_started_at` (GPU-held clock starts) | 02:54:13.264 |
| stop requested / accepted | 03:21:12.594 / 03:21:12.595 |
| control probe: `note` recorded / `extend 60` recorded | 03:21:13.858 / 03:21:16.089 |
| capture ended, stream announced, import started | 03:22:16.35 |
| report written (stop → report **91 s**; predicted ≈ 90 s) | 03:22:43.346 |
| server SIGTERM (reason `shutdown`, identity-verified API server pid 1048248) | 03:22:45.426 |
| GPUs released (controller) — **GPU-held 1713 s** (limit 2700 s) | 03:22:46.701 |
| nsys session closed | 03:22:48.711 |
| launcher rc=0 (sqlite export, parse) | 03:27:45 |
| independent GPU census t0 / t20: RELEASED, no compute process, 0 MiB × 4 | 03:27:45 / 03:28:06 |
| driver's final process census: CLEAN | 03:28:16 |
| pinned closeout: CLEAN (GPUs 0 MiB, registry 2 GONE, 0 owned, 0 lookalikes) | 03:31:02–03:31:13 |
| pinned post-run (CPU, CUDA hidden): STOP | 03:31:55 → |
| supplementary recovery diagnostic (outside the gates) | 03:48:38–03:50:06 |

## Workload and trace (scientific checkers: all PASS)

- 56/56 requests, each 1024 tokens; both waves captured: pure-decode iterations b=8 **1008**, b=48 **928**; 0 overrun, 0 duplicate or
  missing joins (completeness v2 rc 0).
- NVTX forward ranges **8640** = 4 × **2160** iterations (band 2100–2250); kernel rows **12 250 832** (expected 12 250 832; band
  11.64–12.86 M); kernel span 50 516 ms. Four ranks: extract wrote 8640 rank rows for 2160 iterations, unaligned 0.
- `e1_nsys_validation.py` RESULT PASS (90 PASS lines); `costmodel.extract` rc 0; recipe check on the launch record rc 0 (server argv,
  nsys flags, nsys executable and env additions unchanged by the controller). No EngineDeadError-family line.
- Storage: run directory 2.7 GB (limit 5); `/tmp` 0.35 GB preserved stream (limit 5); archive 3.1 GB → 5.8 GB persistent in total (limit 10).

## Gate results (pinned `checks/e1_lifecycle_gates.py`, `V1-GATES.txt`)

| gate | result | evidence |
|---|---|---|
| L0 record | PASS | schema 1, phase `report_written` |
| **L1 idempotent stop** | **FAIL** | `requests_sent=1`, `http.status=ok`, **`duplicate_requests_suppressed` absent (0)**; the gate requires ≥ 1 |
| L2 / L2b phase order, stream phase | PASS | `stop_requested → stop_accepted → capture_ended → stream_written → importing → report_written`, timestamps monotone |
| L3 / L3b report complete, coverage | PASS | report 393 018 929 B, validation ok, rows and iterations in band |
| **L4 stream preserved** | **FAIL** | recorded `stream.size` 347 628 923 ≠ `preserved_size` 347 628 953 = on-disk 347 628 953 (30 bytes); sha256 not recorded by the gate |
| L5 / L5b no signal to the importer, server stopped by identity | PASS | one action: SIGTERM to the API server after the report |
| L6 / L6b GPU-held, finalisation | PASS | 1713 s ≤ 2700 s; 91 s ≤ 600 s |
| L6c independent GPU release | PASS | census t20 at 03:28:06Z after the controller's 03:22:46.7Z: no compute pid, 0/0/0/0 MiB |
| L7 control probe | PASS | exactly the two pre-registered lines: `noted`; `hard deadline extended by 60 s (operator); GPU-held ceiling unchanged` |
| L8 / L8b / L8c escalations, warnings, extension accounting | PASS | none; none; 60 s = 60 s |
| L9 session closed | PASS | nsys and server gone |
| **L10 / L10b recovery proof** | **NOT RUN** | the pinned gate does not attempt the import when L4 fails |
| R launch record | PASS | Nsight 2025.6.3.541, pinned executable |
| S.completeness_v2, S.extract, S.nsys_validation, S.recipe | PASS | exit codes 0 |
| **S.config_identity** | **FAIL** | 5 FAIL lines comparing this launch record with the accepted `e1_nsys_val` one (below) |

### The failed conditions, exactly

1. **L1.** The controller sent exactly one stop request and it returned `ok`. No second request was made or suppressed: the record
   has no `duplicate_requests_suppressed` entry. The gate was written on the assumption that the runner's `finally` path always
   re-requests the stop; on this successful path it did not. "Exactly one effective stop request" and "no duplicate finalisation"
   hold in the record (one POST, one `report_written`, one server SIGTERM), but the pinned condition `duplicates ≥ 1` is not met.
2. **L4.** The preserved hard link is on disk with 347 628 953 bytes, equal to the recorded `preserved_size`. The controller's
   `stream.size` (347 628 923) is 30 bytes smaller: it was sampled while Nsight was still appending to the stream (file mtime
   03:22:42.58, i.e. 26 s after the `stream_written` phase line and 0.8 s before the report). The pinned byte-completeness condition
   `stream.size == on-disk size` therefore fails, the gate records no sha256, and L10 is skipped. The record alone cannot show which
   of the two sizes is the complete stream.
3. **S.config_identity.** `e1_config_identity.py` on the two launch records reports: the `-o` output stem differs (each run's own
   nsys directory and name); server env `PYTHONPATH` differs (`wt-e1stage1/alignment/profiler/vllm` vs `wt-lifecycle/alignment/profiler/vllm`;
   both checkouts at fork head 0ab18711); `fork_git.path` differs the same way; and `nsys_config.capture_start_delay_seconds`,
   `nsys_config.finalize_*` differ — the approved no-delay and lifecycle keys, which the post-run allow-list names as `nsys.*`
   (the profile-config namespace) and not as `nsys_config.*` (the launch-record namespace). The profile-config comparison of the same
   checker passed, as did `server_argv`. The preflight had exercised this checker on configs only, never on a launch record from the
   new worktree, so these differences were not seen before the run. This is a preparation gap of the operator (Claude).

## Supplementary diagnostic (outside the pinned gates; does not change the verdict)

Because the pinned L10 was not attempted, the preserved stream was checked once by hand on tray16, CPU-only, without touching the run
directory (`cpu/supplementary_recovery.json`):

- preserved stream sha256 `8f5905d95c61c8e07cf9d557fad97749191146f97c0e1b1a11dc23edb097057f`, 347 628 953 B;
- offline `QdstrmImporter` rc 0 → report 393 018 979 B (in-session 393 018 929 B);
- `nsys export --type sqlite` rc 0; with the gate's own `validate_sqlite`: kernel rows **12 250 832**, NVTX forward ranges **8640**,
  kernel span 50 516.325865 ms — **identical to the in-session values**.

So the 347 628 953-byte preserved file imports to a report with the same counts as the original. The derived scratch report and
sqlite were removed after counting; the preserved stream was not modified.

## Independent GPU-release evidence

`gpu_release_census_t0.json` (03:27:45Z) and `_t20.json` (03:28:06Z), taken by the driver from `nvidia-smi` outside the controller:
no compute process on any GPU, memory.used 0 MiB on all four. The pinned closeout at 03:31:02Z and every later sample read 0 MiB.
The controller's own record: the four `VLLM::Worker` compute pids 1049180–1049183 gone at 03:22:46.7Z.

## Deviations and failures

1. V1 FAIL on three pinned gates; L10/L10b not attempted (above).
2. The first census sample was taken at the launcher's exit (03:27:45Z), about 5 minutes after the controller's release time,
   because the launcher finishes its CPU post-processing before it returns. The census therefore confirms release at 03:27:45Z, not
   at 03:22:46Z. Monitoring samples by the operator read 0 MiB / 0 compute apps from 03:23Z onward.
3. `gpu_release_census_postrun.json` was not written: `postrun_e1_v1.sh` takes it only if `lifecycle.json` carries a `host` field,
   which the controller does not write. L6c passed on the driver's two samples.
4. The audit note required before launch was placed as a new untracked file (`runs/v1/preflight/AUDIT-NOTE-preflight-failure-20261118T0146Z.md`,
   02:48:46Z) rather than appended to a tracked document, so that HEAD stayed exactly 77f410a5 with a clean tree at launch. It is
   committed with this report.
5. The supplementary recovery diagnostic and this report's analysis are additions outside the pinned procedure.
6. No hard limit was reached; no abort condition was printed by the monitor; nothing was signalled by the operator.

## Where things are

- Run directory (2.7 GB): `ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_v1/` on the shared filesystem.
- Archive with `SHA256SUMS`: `artifacts/phase1c-e1-v1-dsv41-flash-gb300-tp4-20261118/` (run directory, cpu results, preflight, pid
  registry, preserved `.qdstrm`).
- Preserved stream, still in place: `/tmp/dsv41_flash_1017_gb300_tp4_e1_v1_lifecycle.raw-preserve/` on tray16 (kept: V1 did not pass).
- Phase 1A tables: `phase1a-tables/e1_v1_lifecycle` (13 MB).
- Small evidence in this directory: `evidence/` (lifecycle record and control file, gates, census files, probe log, driver preflight
  log, run log, launch record, profile result, checker outputs, supplementary diagnostic, archive checksums), closeout and post-run logs.
