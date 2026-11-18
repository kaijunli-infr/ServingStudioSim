# E1 V3 intermediate-scale validation — run report (2026-11-18)

**Verdict under the frozen gates: FAIL** — one gate, G9, on its configuration-identity component; the other thirteen gate results
are PASS and none is INCONCLUSIVE (`evidence/cpu/V3S-GATES.txt`). One launch of `e1_v3_scale` on `pod4-gb300-10-tray16` from
`exp/profiler-lifecycle` e82c8df9 under the conditional GO of 2026-11-18. The GO is consumed; no retry, nothing else was run.
The failed gate is not reinterpreted here. §4 states exactly what failed and why, including that the cause is a defect in the gate
set I prepared; what to make of it is the reviewer's decision.

## 1. Pre-launch checks (all passed; `evidence/operator/`)

| GO check | result |
|---|---|
| 1 HEAD e82c8df9, worktree clean | PASS (`prelaunch_checks_20261118T212127Z.log`) |
| 2 code commit d3b9c264 → packet commit 2fa0f367 → HEAD; 27 pinned trees/blobs = HEAD = code commit; gitlinks = checkouts; `alignment/` and `launcher/` unchanged since e32fbab7; 66 scripts; inputs; Nsight CLI and importer hashes | PASS |
| 3 `FREEZE_v3scale.sha256`, all 7 entries (manifest sha256 0478fdcc…) | PASS |
| 4 `session_runner` sha256 7f6eb81d45e6b98f3b6ebb56ab20b60ce8aaf4584aa3d4f25e3449b2d05a00c2 | PASS; same bytes, inode, mtime and size after the launch |
| 5 prebuilt path (one launcher line with `--frontend-provenance`), guard | PASS; launcher record: prebuilt mode, guard armed, nothing blocked |
| 6 exact read-only preflight | 69 PASS, 0 FAIL (`preflight_v3scale_tray16_driver_context_20261118T212711Z.log`) |
| 7 negative control | preflight stopped on the lookalike (66 PASS, 3 FAIL); only the scratch-registered lookalike was signalled |
| 8 reservation pod Running with 4/4 GPUs, GPUs 0 MiB, no compute process, only daemonset pods | PASS |
| 9 MemAvailable 858–861 GiB (≥ 448); `/tmp` 1163 GB and CephFS 20 771 GB free | PASS |
| 10 tools pinned, present, responding; registry; recovery tools | PASS |
| 11 supervisor | standalone check before the launch PASS; in the run its first monitored record (21:29:54Z, about 50 s after the launcher start) showed a check 1 s old and all three sampler heartbeats under 2 s. The in-run heartbeat cannot be read before the driver starts it |
| 12 ceilings and deadlines of the frozen inputs = the GO | PASS |

## 2. Timeline (UTC)

| time | event | source |
|---|---|---|
| 21:23:51 | reservation pod Running | kubectl |
| 21:27:47 | driver launched; its own preflight PASS | `LAUNCHED`, `preflight.log` |
| 21:29:06.80 | launcher start; driver, four observers and the supervisor registered | lifecycle, registries |
| ≈ 21:53 / ≈ 21:54:30 | server startup complete / measurement begins | monitor |
| 21:56:49.05 | the one stop request, accepted 3 ms later | lifecycle |
| 21:58:41.04 | controller phase `importing` (flush 112.0 s) | lifecycle |
| 21:58:41.48 | importer first seen by the probe (pid 3030970; child of the agent `nsys --start-agent --session-name profile-3016231`, parent pid 1) | `import_probe.json` |
| 21:58:42.77 | `ACTIVE`: two consecutive intervals of the importer's own CPU / I-O | `import_probe.json` |
| 21:59:34.44 | stop request returned OK | lifecycle |
| 21:59:34.52 | importer gone (import 53.0 s) | `import_probe.json` |
| 21:59:35.27 | `report_written` (166.2 s after the stop request) | lifecycle |
| 21:59:37.37 | normal shutdown: SIGTERM to the identity-verified API server | lifecycle `actions` |
| 21:59:37.4 – 37.8 / 39.8 – 40.2 | sampler: last "held" reading / first "released" reading | `gpu_observations.jsonl` |
| 21:59:40.07 | session closed (2.7 s after the SIGTERM) | lifecycle |
| 22:09:53 | launcher rc 0 (in-session export and parse) | `run.log` |
| 22:15:51 | closeout CLEAN | `closeout_tray16_20261118T221551Z.log` |
| 22:24:06 | reservation pod deleted, no GPU pod on the node (held 60.2 min) | `reservation_deleted.json` |
| 22:25:07 | post-run → FAIL (G9) | `postrun_tray16_20261118T222507Z.log` |
| 22:30:26 | archive VERIFIED | `archive_tray16.log` |

No escalation, no soft warning, no control line, no forced release, no breach marker, no supervisor response (2463 checks, monitors
effective throughout, no signal sent), no operator intervention.

## 3. Gates

| gate | result | measured |
|---|---|---|
| G0 source and executable | PASS | launch commit e82c8df9, pinned trees and gitlinks, clean, real mode; 16 of 16 binary checks |
| G1 host identity | PASS | lifecycle = run log = reservation node = 863 GPU observations = probe, memory, storage, supervisor |
| G2 one stop request | PASS | one request, returned OK 2.9 s before the SIGTERM, one "Stopping profiler" and one "Profiler stopped." line |
| G3 phases in order | PASS | the six registered phases; no escalation, control line or release request |
| G4 phase deadlines | PASS | stop accepted 0.003 s (30); flush 112.0 s (300); import 53.0 s (300); stop → report 166.2 s (600); session close 2.7 s (300); controller hard deadline exactly 600 s after the stop request, extension 0 |
| G5 importer observed | PASS | exactly one importer, identified by stream argument, report argument and open stream; parent is the agent of the recorded CLI; activity from its own counters; completed; no signal outside the server processes |
| G6 GPU release interval | PASS | last "held" reading within [0.056, 0.399] s after the SIGTERM, first "released" within [2.399, 2.860] s: released no later than 2.86 s (deadline 90 s); both censuses released |
| G7 stream durable | PASS | 661 640 958 B, sha256 764d063d951d4c4683f1b4bea8336652846459de6b73ffb953a49d30744f8ba3: controller's final record = durable copy = tray original |
| G8 offline import | PASS | rc 0 in 56 s → 792 004 342 B |
| G8b recovered trace | PASS | 4320 iterations (4200–4500), 17 280 NVTX forward ranges, 24 501 664 kernel rows (23 276 580–25 726 747), per rank 6 124 336 ×3 and 6 128 656, capture boundaries present; all ten equality checks with the in-session export true |
| **G9 scientific completeness** | **FAIL** | four-wave completeness and routing records exit 0 (cells 1008 / 928 / 1008 / 928 pure-decode iterations, 112 routing arrays, corpus checksum); extract exit 0 (4320 iterations, 17 280 rank rows); Nsight validation exit 0; **configuration identity FAIL** (§4) |
| G10 bounds | PASS | GPU-held 1833 s by controller and by sampler upper edge (2700); session close 2.7 s; reservation 60.2 min (90) |
| G11 storage and memory | PASS | temporary peak 5.219 GB (10), in the post-run; run root peak 6.311 GB (8); MemAvailable minimum 547.2 GiB (64); importer 15.61 GiB in session and 15.61 GiB offline (128); all four sample sets cover their intervals; supervisor record complete, no response |
| G12 closeout | PASS | closeout exit 0; reservation deleted, no GPU pod left |

Archive (after the gates): run root 6.285 GB + archive 6.947 GB = 13.231 GB (ceiling 16).

## 4. The failed gate

- **What failed.** G9 requires the three checker exit codes to be 0 and the configuration-identity report to be PASS. The exit codes
  are 0. The identity report (`evidence/cpu/identity_v2_e1_v3_scale.txt`) is FAIL on one field:
  `profile.workload.frontend.path: rule condition not met: sha256 a=c6d8e13b… b=d9d2dc77… pinned=c6d8e13b…`.
- **Why.** The post-run uses the identity policy written for V1 and V2 (`checks/e1_identity_policy_v1.json`). Its rule for the trace
  path accepts a different path only if the file has the pinned hash of the accepted dry-run trace, c6d8e13b…. The V3 trace is the
  approved one, the accepted 56 rows twice (d9d2dc77…), so it can never have that hash. The other fields compared equal: 182 launch
  fields with no rule triggered and 74 profile fields, 29 normalized as listed in the report.
- **Whose defect.** Mine, in the preparation: I carried the V2 identity step into the frozen V3 post-run without checking it against
  the V3 config. No rehearsal exercised it on a V3-shaped run: the V2-fixture rehearsal used V2's own config (identity PASS), and
  the end-to-end rehearsal had no launch record, so G9 was already expected to fail there for other reasons. The gate as frozen
  could not have passed for the approved workload.
- **What is established independently about the workload identity** (evidence, not a substitute for the gate): the preflight
  verified before the launch that the trace is exactly the accepted rows twice with wave index 1 and +80 s, and that the config
  equals the V2 config except name, log_dir and trace path; `INPUTS_v3scale.sha256` and the freeze manifest pin both; G0 passed.
- **Not done.** The identity checker was not re-run with another policy, and no gate result was changed. V3 is FAIL under the frozen
  criteria.

## 5. Measurements (the calibration V3 was meant to give)

| quantity | V3, 24.50 M rows | prediction in the packet | V2, 12.25 M rows |
|---|---|---|---|
| iterations / NVTX ranges / kernel rows | 4320 / 17 280 / 24 501 664 | 4320 / 17 280 / 24 501 664 | 2160 / 8640 / 12 250 832 |
| flush (stop → importing) | 112.0 s | 60 – 140 s | 61.9 s |
| import (importer first seen → gone) | 53.0 s | 27 – 110 s | ≈ 28.9 s (not observed directly) |
| stop → report | 166.2 s | 140 – 190 s | 90.8 s |
| offline import | 56 s | — | 26 s |
| raw stream | 661 640 958 B (27.0 B/row) | ≈ 0.70 GB | 348 288 111 B |
| report / in-session export | 0.792 GB / 3.955 GB | ≈ 0.78 / 3.9 GB | 0.392 / 1.964 GB |
| run root peak | 6.311 GB | ≈ 6.3 GB | 3.16 GB |
| temporary peak (post-run, with retained V1/V2 streams) | 5.219 GB | ≈ 5.2 GB | 2.617 GB |
| importer memory, in session / offline | 15.61 / 15.61 GiB | unknown | not measured / 4.9 – 8.7 GiB |
| host MemAvailable, minimum during the run | 547.2 GiB (858 GiB idle) | never measured before | not measured |
| GPU-held | 1833 s | ≈ 1800 s | 1664 s |

At twice the rows the flush took 1.81 times as long and the import 1.83 times (against V2's inferred import); the importer's memory
roughly doubled against the highest V2-size sampling. Two sizes of one recipe do not establish a scaling law.

First observations inside a real session: the importer's arguments, its open stream and its agent parent (`--session-name
profile-<CLI pid>`, parent pid 1) are as in the CPU-only observation, so the importer-specific probe and the memory sampler worked;
the supervisor ran against the real server for the whole run without a cause; the server's host footprint is visible for the first
time (MemAvailable fell by about 310 GiB from idle at its lowest).

## 6. Closeout

GPUs 0 MiB ×4, no compute process; both registries: all six records gone, no lookalike; no Nsight CLI, agent or importer left; no
tmux session; reservation pod deleted. Archive `artifacts/phase1c-e1-v3scale-dsv41-flash-gb300-tp4-20261118` (207 files, read-only,
`SHA256SUMS` sha256 44122927ea6c29995a5af63846bdc03e40c40def73eb221f80acd43c05bcda97): run root copy equal to the source file by
file, the stream twice with the controller's hash. The tray originals of the V1, V2 and V3 streams stay in `/tmp` (1.3 GB). The
worktree's executable is still 7f6eb81d…; the frozen packet, manifests and gate code are unchanged. The memory and storage sample
files are thinned in `evidence/` (every sample with the importer kept); the complete files are in the archive.

## 7. Not done

No second launch, no V2 retry, no E1 retry, no identification partition, no fitting, no E2–E6, no production integration, no
profiling database write, no preset change. Stop for review.
