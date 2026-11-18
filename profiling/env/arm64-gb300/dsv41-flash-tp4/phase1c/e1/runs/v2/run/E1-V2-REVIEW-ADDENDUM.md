# E1 V2 forced-release validation — review addendum (2026-11-18)

Separate from, and changing nothing in, the frozen packet (`E1-V2-FORCED-RELEASE-PACKET.md`, `FREEZE_fr.sha256`), the original gate
outputs (`evidence/cpu/V2-GATES.json` sha256 bcd6b043…, `V2-GATES.txt` ca65b722…) and the run report (`E1-V2-RUN-REPORT.md`
cc7d11d1…). **Frozen verdict: PASS (20/20 gates). Review disposition: ACCEPTED WITH QUALIFICATIONS.** The one-launch GO is consumed.
This addendum records the qualifications and corrects two statements of the run report that claimed more than the evidence supports
(§2, §3). Supporting analysis: `../closeout/E1-V2-API-SERVER-SHUTDOWN-ANALYSIS.md`, `../closeout/E1-IMPORT-PROBE-DESIGN.md`.

## 1. Qualifications

### Q1. Importer-specific progress was not demonstrated by the pre-signal samples

- The frozen probe's criterion was "CPU time of the recorded Nsight process tree, or reported progress, advanced". Its two pre-signal
  samples (15:56:02.34, 15:56:02.79) show the Nsight CLI's own CPU time flat (168 ticks) and reported progress flat (8 %); the
  tree total rose by 3 ticks, all from the four vLLM worker processes.
- Established since (CPU-only, `../closeout/synth_a.out`): the importer is **not in that tree at all**. `QdstrmImporter --input-file
  <stream> --output-file <report>` is a child of the detached Nsight agent (`nsys --start-agent --session-name profile-<CLI pid>`,
  parent pid 1, own session). The V2 probe therefore could not have observed the importer, and "importer identity verified" in FR3,
  FR3b and FR4 means the identity of the **Nsight CLI process** (pid 1556601), not of the process that performed the import.
- The reported "8 %" is not import progress either. The server log around the signal reads 0, 8, 7, 6, 5, 5, 5, 0 %, then the
  shutdown lines, then 5, 6, 7 … 50, 100 %: the figure moved down before the signal and rose monotonically only after it.
- What does support an import running at the signal, indirectly: the controller's phase `importing` from 15:56:01.90 (Nsight had
  printed "Generating '…qdstrm'" and a progress line); the monotone rise to 100 % after the signal; the report's last write at
  15:56:30.82, 28.9 s after the phase began (V1's import: 27.0 s; offline import of the same stream today: 26.0 s).
- Not demonstrated: any direct pre-signal observation of the importer process (existence, identity, CPU or I/O activity).
  "Importer survival" is likewise inferred from the report having been written and validated, not from observing the importer.
- Consequence: FR3 passed as frozen and stays PASS; the acceptance requirement "two qualifying import-progress observations" is met
  only in the frozen probe's sense. An importer-specific probe is designed and tested in `../closeout/E1-IMPORT-PROBE-DESIGN.md`.

### Q2. The API server required the identity-verified SIGKILL after the grace period

- SIGTERM to pid 1556645 at 15:56:03.074; the server logged its shutdown sequence within the same second, force-killed the engine
  (GPUs free), then logged uvicorn's "Waiting for connections to close" and nothing more.
- 60.04 s later (15:57:03.114, `server_term_grace_s` 60) the controller sent SIGKILL to the same pid after re-checking its identity,
  as the frozen release action prescribes. The pending stop request returned "Remote end closed connection without response" 4 ms
  later; the Nsight session closed at 15:57:04.12.
- Cause (analysis document, evidence-backed): the one outstanding `/stop_profile` request never completes once the engine is gone,
  and the server's HTTP shutdown waits for it without a time limit.
- No destructive signal went to the Nsight CLI, the agent or the importer. GPU release did not depend on the API server exiting.

### Q3. The controller's progress reading stayed at 8 %

- `import_progress_pct` is 8 in both controller checks (15:56:03.07 and 15:57:03.11) and in the monitor sample of 15:56:54.
- During those 60 s the controller was inside the release action's grace wait and did not re-read the log; the log shows the bar
  reaching 100 % in that window. The value is stale, and (Q1) it was never a reliable progress measure.

## 2. Independent GPU-release interval (precise statement)

Source: `evidence/gpu_observations.jsonl`, 754 observations by the registered sampler (19 idle before the model load, 705 held,
30 released; none held after the first released one; none unavailable; gap between observation stamps 2.12–3.24 s, median 2.21 s).
Each observation's time stamp is taken **before** its two `nvidia-smi` queries; the sampler then sleeps 2.0 s. A reading was
therefore taken between its own stamp and the next stamp minus 2.0 s.

| observation | stamp (UTC) | reading taken within | relative to the release request (15:56:03.073) |
|---|---|---|---|
| last "held" (4 × 260 943 MiB, 4 compute processes) | 15:56:01.915 | 15:56:01.915 – 15:56:02.092 | −1.159 s … −0.981 s |
| first "released" (4 × 3 MiB, no compute process) | 15:56:04.092 | 15:56:04.092 – 15:56:05.193 | +1.019 s … +2.119 s |

- Supported claim: the GPUs were still held no earlier than 1.16 s before the request and were observed released **no later than
  2.12 s after it**. The sampling does not support a tighter bound, and by itself it does not show that the release followed the
  request (the interval straddles it); that ordering rests on the SIGTERM record and on the probe seeing the four workers running
  at 15:56:02.79 and as zombies at 15:56:03.54.
- Correction: the run report and the gate text say "at most 1.0 s after the request". That figure is the stamp of the first
  "released" observation (+1.019 s) and ignores the query time after the stamp. The gate (deadline 90 s) is unaffected.
- The controller's own `gpu.released_at` (15:57:04.12) is the end of its release action, not an observation of the GPUs.

## 3. Peak post-run scratch usage: unmeasured

The post-run's scratch directory (`/tmp/v2-recovery-e1_v2_forced_release`) was created and removed by the gate checker and was not
sampled. Its peak size is **unmeasured**. The packet's 2.4 GB was an estimate; the run report's "0.66 GB of retained streams" is the
state after the run, not a peak. What is known: the offline report it wrote was 391 797 436 B and the in-session export of the same
trace is 1 964 322 816 B. Compliance with the 5 GB temporary-storage limit during the post-run is therefore not demonstrated by
measurement (free space on `/tmp` was 1164 GB before and after).

## 4. Preserved evidence (unchanged)

| item | value |
|---|---|
| raw stream, complete SHA-256 | `4739d6581d1b035bae1f1a0f15a3163bf8e708dc022527fb64228f1143db839c`, 348 288 111 B |
| copies with that hash | tray16 `/tmp/dsv41_flash_1017_gb300_tp4_e1_v2_forced_release.raw-preserve/` (original, retained); run directory `preserved/`; archive `preserved/` and `run/…/preserved/` |
| controller's final stream record | taken 15:57:04Z after session close; equals the above (FR7) |
| archive | `artifacts/phase1c-e1-v2fr-dsv41-flash-gb300-tp4-20261118`, 141 files, read-only; `SHA256SUMS` sha256 `1a8959435c584bca5a4b318244e5221009fd3852d3903d141caaf1119f1b99b4`, committed as `evidence/ARCHIVE-SHA256SUMS` |
| recovery | FR8: offline import of the durable copy rc 0 in 26 s → 391 797 436 B; FR8b: 2160 iterations, 8640 NVTX forward ranges, 12 250 832 kernel rows, kernels per rank 3 062 168 ×3 and 3 064 328, boundaries 115 055 661 – 50 862 664 576 ns; ten equality checks against the in-session export all true |
| repeated today (CPU, tray14, archive copy, hash re-verified) | offline import rc 0 in 26.2 s → 391 797 430 B (`../closeout/import_final.out`); the report size varies by a few bytes between imports of the same stream |
| gate outputs, lifecycle record, probe record, sampler, registry, signal records | `evidence/`, unchanged |
