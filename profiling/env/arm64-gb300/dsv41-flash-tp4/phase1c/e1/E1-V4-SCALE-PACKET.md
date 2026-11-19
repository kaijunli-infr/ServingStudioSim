# E1 V4 — intermediate lifecycle-scale validation (≈ 49 M kernel rows): PROPOSED packet

**Status: PROPOSED. Prepared CPU-only under the review decision of 2026-11-19 on `598d4a2d`. Not approved, not frozen,
not launched. V4 GPU execution: NO GO.** The manifests named here (`SCRIPTS_v4scale.sha256`, `INPUTS_v4scale.sha256`,
`V4S-PIN.json`, `V4S-PIN-identities.json`, `FREEZE_v4scale.sha256`) exist so that a review can see exactly what would be
frozen; the freeze takes effect only if a review approves it, and any change after that needs a new one. Preparation evidence:
`runs/v4scale/prep/E1-V4-PREP-REPORT.md`.

Standing record, unchanged: original V3 verdict FAIL / STOP under its frozen gates; retrospective V3 evidence assessment PASS,
qualified, post hoc.

## 1. Purpose and non-purpose

- **Purpose.** One controlled capture at twice V3's size to test the profiler lifecycle and the operational safeguards against
  prospective limits: flush, import, total, GPU release, GPU-held time, reservation, importer and host memory, storage, stream
  preservation, offline recovery, identity-verified cleanup.
- **Non-purpose.** Scientific identification. The workload repeats two accepted cells. Its eight waves are lifecycle-scale
  replicates inside one session, not independent scientific observations. No new cell is added. No identification partition,
  no fit, and no use of this capture in E1's analysis.

## 2. Workload and recipe (`make_e1_inputs_v4scale.py`, deterministic; `--check` regenerates and compares)

| item | value |
|---|---|
| pass | `e1_v4_scale`, one launch, on `pod4-gb300-10-tray16` |
| source trace | `traces/e1_dry.csv`, 56 requests, sha256 c6d8e13b7229ffd40cfede127bd57ad345aa17213b7590d39f9b2ef697f60158 (accepted) |
| trace | `traces/e1_v4_scale.csv`, sha256 63d79100e5f482854fcc868bda859f206c611e194c18efec40749c8a31d76de1: the source rows four times. Copy i (0 – 3): wave index i in each request id (`_w0_r` → `_w{i}_r`), arrival + i × 80 000.0 ms, every other field identical, order kept |
| requests / waves | 224 requests, 224 distinct ids; 8 waves: b=8 at 0 / 80 / 160 / 240 s (8 requests each), b=48 18.6272 s after each (48 requests each). Both cells preserved: 32 b=8 and 192 b=48 requests |
| config | `configs_v4scale/e1_v4_scale/profile.json`: the V2 config (`configs_fr/e1_v2_forced_release/profile.json`) with `name`, `log_dir` and the trace path changed; identical to the V3 config in every other field (test) |
| backend, model, server arguments, routing hook, Nsight recipe | unchanged: openai backend, `capture_routed_experts` true, `stream` false, model and server arguments, Nsight capture flags equal to the Phase 0 block, no delayed start, no bounded stop |
| controller | unchanged code and policy: soft 300 s, hard 600 s, session close 300 s, GPU-held ceiling 2700 s, stream preserved, 60 s server grace (controller default, not set in the config) |
| executable | `session_runner` sha256 7f6eb81d45e6b98f3b6ebb56ab20b60ce8aaf4584aa3d4f25e3449b2d05a00c2, prebuilt contract, build-tool guard, as V2 / V3 |

## 3. Expected size (registered; `feasibility/E1-ROWS-PER-ITERATION-ANALYSIS.md`)

| quantity | expected | band / rule | basis |
|---|---|---|---|
| requests, request timings, routing arrays | 224 | exact (G15) | trace |
| iterations | 8640 (4 × 2160) | 8400 – 9000 | measured: 1040 per b=8 wave, 1120 per b=48 wave, identical in three waves each |
| NVTX forward ranges | 34 560 | 4 per iteration | measured |
| kernel rows | 49 003 328 (4 × 12 250 832) | ± 5 %: 46 553 161 – 51 453 494 | measured: 5 887 244 per b=8 wave and 6 363 588 per b=48 wave, identical in V2, V3 wave 0 and V3 wave 1 |
| pure-decode iterations per cell | 1008 (b=8), 928 (b=48) | floor 400 per cell (checker rule) | measured |
| ranks | 4 | exact | — |

## 4. Predictions (registered before the run; not gates) and limits

Predictions: `feasibility/E1-SCALING-MODEL.md`, column "intermediate ~49 M": range over three curves through the V2 and V3
points, stress case in brackets. Two controlled points are not a validated scaling model.

| quantity | prediction | limit | enforced / judged by |
|---|---|---|---|
| stop accepted | ≈ 0 s | 30 s | G4 |
| flush (stop → controller phase `importing`) | 203 – 224 s (249) | 300 s | G4 |
| import (importer first seen → gone) | 97 – 106 s (118) | 300 s | G4 |
| stop → report | 301 – 331 s (367) | 600 s, no extension | controller hard deadline; G3, G4 |
| session close after the server SIGTERM | ≈ 3 s | 300 s | controller; G4, G10 |
| GPU release, upper edge after the server SIGTERM | ≤ 3 s | 90 s | G6 |
| GPU-held from the launcher start | 2092 – 2275 s (2311) | 2700 s | supervisor before the stop request, controller after it; G10 |
| reservation | 72 – 81 min | 90 min | driver (launch ≤ 600 s after creation), supervisor guard at 4680 s, operator budget 720 s; G10 |
| importer peak memory | 28 – 31 GiB (35) | 128 GiB | memory sampler → supervisor; G11 |
| host MemAvailable | ≈ 530 GiB minimum | preflight ≥ 448 GiB (retained); floor 64 GiB | preflight; memory sampler → supervisor; G11 |
| temporary storage | 10.4 – 10.5 GB (11.4) | 15 GB | storage sampler → supervisor (run), post-run sampler; G11 |
| run directory | 12.6 GB (14.0) | 18 GB | storage sampler → supervisor; G11 |
| run directory + archive | 26.5 GB (29.5) | 38 GB | `checks/e1_archive_size.py` in the archive script |

Timeline, worst case, behaviour without a stop request and with a failed monitor, and the compatibility of these limits:
`E1-V4-TIMELINE.md`. Two findings there need a decision (§12).

## 5. Observers and supervisor (existing mechanisms; one addition)

| process | script | role |
|---|---|---|
| GPU sampler | `checks/e1_gpu_watch_v3.py`, 2 s | independent release evidence (G6) |
| import probe | `checks/e1_import_probe.py`, 0.5 s, observer mode | importer identity and activity (G4, G5) |
| memory sampler | `checks/e1_mem_watch.py`, 1 s | MemAvailable and importer memory; writes `MEMORY-BREACH` |
| storage sampler | `checks/e1_storage_watch.py`, 2 s | temporary and run-directory storage; writes `STORAGE-BREACH` |
| supervisor | `checks/e1_supervisor_v4.py`, 1 s, own registry | the unchanged V3 supervisor (`e1_supervisor.py`, imported) plus the **reservation guard**. Causes: breach markers, a lost monitor (60 s), the GPU-held ceiling before a stop request, the reservation guard. One response: the registered release line through the controller during finalisation, the pinned identity-verified registry cleanup otherwise. It signals nothing itself |

All are registered with identity (pid, start time, boot id, uid); cleanup acts only on registry records that still match.

## 6. Configuration identity (V4-specific, prospective)

- Policy `checks/e1_identity_policy_v4scale.json` (version `v4scale-1`), checker `checks/e1_launch_identity_v4scale.py`.
- Every field except the trace is compared as for V1 / V2: the comparison code of `e1_launch_identity_v2.py` is imported
  unchanged; launch rules, the other profile rules and the protected patterns are identical to the V1 policy (test).
- The trace rule `derived_trace` holds only if all 20 checks hold: documented paths; pinned source hash; pinned V4 hash equal
  to the entry in `INPUTS_v4scale.sha256` and in `E1-V4-SCALE-INPUTS.json` (both pinned by hash); re-derivation from the source
  bytes (header, 56 source rows, 224 rows, copy 0 identical, copies 1 – 3 with the wave index and offset, order, no other row,
  unique ids, eight waves, token lengths, whole-file equality).
- It applies only to `e1_v4_scale` against `e1_nsys_val`. The V1 / V2 and V3 policies fail on the V4 config; the V4 policy fails
  on V1 / V2 / V3 runs.
- **Exercised against the actual generated V4 configuration before any freeze** (`rehearse_e1_v4scale.sh`, steps 1 and 2; also
  in the preflight): the complete V4 post-run, including this step, on a run directory holding the V4 config byte for byte, the
  V4 trace and the predicted V4 launch record. A real V4 launch record exists only after a launch; the prediction is the V3
  record with the names substituted, a method that reproduces the actual V3 record from the V2 record in all 182 fields.

## 7. Gates (`checks/e1_v4scale_gates.py`; PASS / FAIL / INCONCLUSIVE each)

G0 – G12 are the V3 gate code, imported unchanged and run with the V4 inputs. G13 – G15 are new.

| gate | PASS requires | area |
|---|---|---|
| G0 source and executable | launch record = pinned trees and gitlinks, clean, real mode, descends from the pinned commit; executable = pinned bytes before and after, guard armed | source, binary |
| G1 host identity | lifecycle = run log = reservation node = every sampler | host |
| G2 one stop request | one request, returned OK before the server was signalled | lifecycle |
| G3 phases in order | stop_requested … report_written; no hard-deadline or ceiling escalation, no control line, no forced release | lifecycle |
| G4 phase deadlines | every phase of §4 at or below its deadline; hard deadline = 600 s from the stop request, unextended | deadlines |
| G5 importer observed | one importer identified by the probe, active by its own counters, completed; no signal to anything but the recorded server | importer |
| G6 GPU release | sampled release interval; upper edge ≤ 90 s; two clean censuses; never held again | GPU release |
| G7 stream durable | controller's final record = durable copy = tray original (SHA-256) | preservation |
| G8, G8b offline recovery | offline import of the durable copy; iterations, rows, NVTX in their bands; equal to the in-session export | recovery |
| G9 scientific checkers | eight-wave completeness and routing records, extract, Nsight validation exit 0; identity verdict PASS | completeness |
| G10 bounds | GPU-held ≤ 2700 s (controller and sampler), session close ≤ 300 s, reservation ≤ 90 min | ceilings |
| G11 storage and memory | measured peaks within 15 GB / 18 GB / 128 GiB / 64 GiB floor; all four sample sets cover their intervals (gaps ≤ 30 s); supervisor record complete, monitors effective, no response; no breach file | resources, sampling continuity |
| G12 closeout | closeout exit 0; reservation deleted, no GPU pod left | closeout |
| **G13 workload and recipe identity** | the identity report is the V4 checker's under policy `v4scale-1`, trace rule exercised, all 20 trace checks hold, verdict PASS. A PASS report from another checker or policy is a FAIL | workload, recipe |
| **G14 four-copy trace in the run** | the run directory's `trace.csv` and `profile.json` are the registered V4 trace and config byte for byte; the trace is re-derived in the gate (own code) as the accepted trace × 4; 224 distinct requests; the config's trace path holds the same bytes | trace derivation |
| **G15 coverage** | from the completeness record: exactly the eight planned cells, each ≥ 400 pure-decode iterations; 224 request timings = token-accounting requests = routing segments; four ranks; NVTX ranges = 4 × iterations with iterations in band; no join or routing-array defect; no failed check | requests, iterations, routing, ranks |

Verdict: PASS only if all sixteen results pass. Any FAIL → FAIL / STOP. No FAIL but an INCONCLUSIVE → INCONCLUSIVE / STOP.
No retry, no advance, on anything but PASS. A run with a supervisor response or an accepted control line is always a FAIL.

## 8. Reservation lifecycle

Default and only authorized sequence: reservation pod created and Running → operator pre-launch checks → driver (launch refused
later than 600 s after the pod's creation) → run → closeout CLEAN → reservation deleted immediately → post-run → archive. The
supervisor ends a run that has not ended 4680 s after the pod's creation. Early deletion after a verified GPU release is
designed and CPU-tested but **not enabled** (`E1-V4-RESERVATION-RELEASE-DESIGN.md`).

## 9. STOP rules (as V3)

Preflight FAIL → no launch. Any monitor `ABORT-CONDITION` → operator review; termination only through the verified registry
cleanup, never by pattern. Any gate not PASS → STOP. The preserved stream is never deleted by a script. Nothing recoverable is
removed on any path.

## 10. Scripts (derived from the frozen V3 scripts; `runs/v4scale/prep/derivation/`)

`run_`, `preflight_`, `monitor_`, `closeout_`, `postrun_`, `archive_e1_v4scale.sh`, `pin_v4scale.sh`, `hash_scripts_v4scale.sh`,
`rehearse_e1_v4scale.sh`, `rehearse_e2e_v4scale.sh`, `e1_v4scale_{rehearsal,eightwave,identity_rehearsal,mock_launcher,e2e_check}.py`.
Each was produced from its V3 counterpart by a fixed renaming plus listed edits; the directory holds one diff per file showing
only those edits. The V3 files are not modified (their manifests verify). Differences that are not renamings or new limits:

1. the driver refuses a launch later than 600 s after the reservation's creation;
2. the supervisor is the V4 wrapper with the reservation guard;
3. the memory and storage samplers and the supervisor run for up to 5400 s instead of 3400 s (the export after the release
   takes about 20 min at this size; with 3400 s the samplers would end first and the supervisor would abort a healthy run);
4. the post-run's identity step is the V4 checker and policy, and the gates receive the completeness record;
5. the preflight regenerates the inputs (`--check`), runs the identity rehearsal on the actual V4 config, and checks the
   reservation budget's arithmetic;
6. the archive script measures run root + archive with `checks/e1_archive_size.py` (38 GB).

## 11. Preparation evidence required before a freeze could be approved (`runs/v4scale/prep/`)

Guarded CPU-only tests; the identity rehearsal on the actual V4 config; the complete V4 post-run on the V4-shaped run directory
with registered expectations; the synthetic fixture matrix (117 fixtures: 3 positive, 6 boundary, 108 negative; every ceiling
and deadline at its boundary); the eight-wave completeness fixtures; the mocked end-to-end rehearsal (five scenarios, including
the reservation guard); the preflight on the capture tray. Every attempt, including failed ones, is kept.

## 12. Open decisions for review

1. Approval of the freeze as proposed, or changes.
2. Preflight host-memory minimum: retained at 448 GiB; with the server footprint measured in V3 (285 – 295 GiB) the consistent
   value would be 488 GiB (`E1-V4-TIMELINE.md` §6). Not changed.
3. The reservation corner: latest launch and a ceiling-long GPU hold together cannot finish inside 90 min under the existing
   lifecycle; the guard ends the export. Accept, tighten the launch limit, or enable early release.
4. The reservation guard itself (new enforcement) and its value, 4680 s.
5. Whether the early reservation release should be developed further (it needs gate changes; not part of this packet).
