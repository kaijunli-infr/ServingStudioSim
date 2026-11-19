# E1 V4 — intermediate lifecycle-scale validation (≈ 49 M kernel rows): PROPOSED packet

**Status: PROPOSED, revision 3. Prepared CPU-only under the review decisions of 2026-11-19 on `598d4a2d`, on `39561325`
(preflight memory, reservation budget; §13) and on `76d95289` (final safety closure: preflight memory 512 GiB, supervisor
latch, bounded closeout and deletion, closeout statements; §14). Not approved, not frozen, not launched. V4 GPU execution:
NO GO.** The manifests named here (`SCRIPTS_v4scale.sha256`, `INPUTS_v4scale.sha256`,
`V4S-PIN.json`, `V4S-PIN-identities.json`, `FREEZE_v4scale.sha256`) exist so that a review can see exactly what would be
frozen; the freeze takes effect only if a review approves it, and any change after that needs a new one. Preparation evidence:
`runs/v4scale/prep/E1-V4-PREP-REPORT.md` (revision 1), `runs/v4scale/prep2/E1-V4-CORRECTION-REPORT.md` (revision 2) and
`runs/v4scale/prep3/E1-V4-FREEZE-READINESS-REPORT.md` (revision 3).

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
| reservation | 70 – 80 min | 90 min | driver launch rule (4680 s − age ≥ run budget 4210 s, so age ≤ 470 s), supervisor guard at 4680 s (acted on in every state), registered watcher and deleter inside the 720 s that follow; G10, G16 |
| CPU-only processing under the reservation (release → stop file) | 1241 – 1247 s (1381) | 1400 s in the run budget; absolute end at reservation age 4680 s | launch rule; supervisor guard |
| importer peak memory | 28 – 31 GiB (35) | 128 GiB | memory sampler → supervisor; G11 |
| host MemAvailable | ≈ 530 GiB minimum | preflight ≥ 512 GiB = 128 + 64 + 320 (exact, in kB); runtime floor 64 GiB | preflight (`checks/e1_preflight_memory.py`); memory sampler → supervisor; G11 |
| temporary storage | 10.4 – 10.5 GB (11.4) | 15 GB | storage sampler → supervisor (run), post-run sampler; G11 |
| run directory | 12.6 GB (14.0) | 18 GB | storage sampler → supervisor; G11 |
| run directory + archive | 26.5 GB (29.5) | 38 GB | `checks/e1_archive_size.py` in the archive script |

Timeline, worst case, behaviour without a stop request and with a failed monitor, and the compatibility of these limits:
`E1-V4-TIMELINE.md` (revision 3: clock definitions, launch rule, closeout and deletion, remaining findings).

## 5. Observers, supervisor and closeout watcher

| process | script | role |
|---|---|---|
| GPU sampler | `checks/e1_gpu_watch_v3.py`, 2 s | independent release evidence (G6) |
| import probe | `checks/e1_import_probe.py`, 0.5 s, observer mode | importer identity and activity (G4, G5) |
| memory sampler | `checks/e1_mem_watch.py`, 1 s | MemAvailable and importer memory; writes `MEMORY-BREACH` |
| storage sampler | `checks/e1_storage_watch.py`, 2 s | temporary and run-directory storage; writes `STORAGE-BREACH` |
| supervisor | `checks/e1_supervisor_v4.py`, 1 s, own registry `pids/supervisor.tsv` | causes, release line, acknowledgement rule and registry cleanup of the unmodified V3 supervisor (`e1_supervisor.py`, imported) plus the **reservation guard**. Causes: breach markers, a lost monitor (60 s), the GPU-held ceiling before a stop request, the reservation guard. Responses: R1 the registered release line through the controller (server only; finalisation only), R2 the pinned identity-verified registry cleanup. **Latch:** R1 at most once; R2 at most once and final; never concurrent; a guard cause is always answered by R2, directly or as an escalation after an earlier acknowledged R1, whether that release completed or stalled. It signals nothing itself |
| closeout watcher | `checks/e1_reservation_closeout.py watch`, 5 s, own registry `pids/closeout.tsv`, armed by the driver through `arm_closeout_e1_v4scale.sh` | waits for the end of the run; permits the reservation's deletion on evidence (§8); observes the deletion with the tray's read-only credential; writes the verdict. It deletes nothing and signals nothing |

All are registered with identity (pid, start time, boot id, uid); cleanup acts only on registry records that still match.
Neither supervisor response and no closeout step can signal the importer or the detached Nsight agent: they are in no registry
and are not descendants of a registered process.

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

G0 – G12 are the V3 gate code, imported unchanged and run with the V4 inputs. G13 – G16 are new.

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
| G12 closeout | closeout exit 0 (resources clean, supervision valid, no abort response, reservation deletion not failed; §8); deleter's record present, no GPU pod left | closeout |
| **G13 workload and recipe identity** | the identity report is the V4 checker's under policy `v4scale-1`, trace rule exercised, all 20 trace checks hold, verdict PASS. A PASS report from another checker or policy is a FAIL | workload, recipe |
| **G14 four-copy trace in the run** | the run directory's `trace.csv` and `profile.json` are the registered V4 trace and config byte for byte; the trace is re-derived in the gate (own code) as the accepted trace × 4; 224 distinct requests; the config's trace path holds the same bytes | trace derivation |
| **G15 coverage** | from the completeness record: exactly the eight planned cells, each ≥ 400 pure-decode iterations; 224 request timings = token-accounting requests = routing segments; four ranks; NVTX ranges = 4 × iterations with iterations in band; no join or routing-array defect; no failed check | requests, iterations, routing, ranks |
| **G16 reservation deletion** | watcher's record and deleter's record both `DELETED_IN_TIME` for the UID in `reservation.json`; the watcher's first permit carries complete evidence and precedes the delete request; exactly one accepted request, with that UID as precondition; both observations within 5400 s of the pod's creation. A late deletion, a missing record or anything unverifiable is a **FAIL**, never INCONCLUSIVE | reservation |

Verdict: PASS only if all seventeen results pass. Any FAIL → FAIL / STOP. No FAIL but an INCONCLUSIVE → INCONCLUSIVE / STOP.
No retry, no advance, on anything but PASS. A run with a supervisor response or an accepted control line is always a FAIL.

## 8. Reservation lifecycle, closeout and deletion

Two credentials exist: the capture tray (dc) can read pods and cannot delete them; the operator's laptop can delete them. The
mechanism therefore has two registered processes, one on each side, and neither needs the other's credential.

| step | where | what |
|---|---|---|
| 1 record | laptop | `e1_reservation_closeout.py record`: the pod must be a reservation placeholder (this node, pause containers only, 4 GPUs, labels, no owner); a dry-run delete with the recorded UID as precondition must be accepted and one with another UID refused by the server; writes `reservation.json` (UID, creation time, proof) |
| 2 deleter | laptop, own registry | `… delete`, started before the driver: polls the watcher's record through a neutral-path helper (this is also its heartbeat) |
| 3 driver prechecks | tray | record complete with proof and live pod = record, else exit 5; arms the watcher and requires it evaluating with a fresh deleter heartbeat, else exit 8; launch rule, else exit 7. All before any observer or the launcher |
| 4 watcher | tray, registry `pids/closeout.tsv` | WAIT while a registered process of the run is alive (**no early release**); then PERMIT only if: D1 no live owned process and no refusal, D2 no unregistered process that looks like the experiment, D3 two consecutive pinned GPU censuses clean, D4 the live pod is exactly the recorded one. Otherwise REFUSE with the reason, re-evaluated every 5 s |
| 5 delete | laptop | after two consecutive permits from the same watcher for the recorded UID and node, and its own ownership check: **one** `DELETE` with `preconditions.uid`. A pod of the same name with another UID is not deleted (`REFUSED_FOREIGN`) |
| 6 verdict | both | each observes the pod gone with its own credential: `DELETED_IN_TIME` at or before 5400 s after the pod's creation; otherwise `DELETED_LATE`, `NOT_DELETED` (600 s after the ceiling), `GONE_WITHOUT_PERMIT`, `UNVERIFIED`, `REFUSED_FOREIGN`, `GONE_NOT_BY_DELETER`: all FAIL (G16) |
| 7 closeout | tray | `closeout_e1_v4scale.sh` after the watcher has ended: resource items, then four separate statements and one combined result (below) → post-run → archive |

The supervisor ends a run that has not ended 4680 s after the pod's creation; the watcher then permits within seconds and the
deletion fits the remaining 720 s without an operator (mechanical path ≤ 300 s; `E1-V4-TIMELINE.md` §3). After a refused
launch (exit 7 or 8) the watcher stays armed and permits once the driver is gone. A live importer or Nsight agent is recorded
and does not block the permit (it holds no GPU and is never signalled). Early deletion after a verified GPU release remains
designed and CPU-tested only, **not enabled** (`E1-V4-RESERVATION-RELEASE-DESIGN.md`).

Closeout statements (`checks/e1_closeout_verdict.py`), reported separately:

| statement | values |
|---|---|
| resource cleanup | CLEAN / NOT CLEAN (GPUs idle, no owned process, no refusal or lookalike, supervisor and watcher gone, no Nsight process, terminal phase) |
| supervision | VALID / INVALID / NOT APPLICABLE. A lost supervisor (incomplete record, or last check before the end of the run), a lost monitor or an incomplete sampler summary is INVALID |
| run response | none / ABORTED (supervisor response or breach marker) |
| reservation deletion | DELETED IN TIME / PENDING / FAIL |
| frozen scientific gates | reported from the post-run record if present; never evaluated by the closeout |

Combined result CLEAN (exit 0) only if resources are clean, supervision is valid or not applicable, there is no abort response,
and the reservation deletion has not failed. A run whose supervisor was lost cannot be reported CLEAN.

## 9. STOP rules (as V3)

Preflight FAIL → no launch. Any monitor `ABORT-CONDITION` → operator review; termination only through the verified registry
cleanup, never by pattern. Any gate not PASS → STOP. The preserved stream is never deleted by a script. Nothing recoverable is
removed on any path.

## 10. Scripts (derived from the frozen V3 scripts; `runs/v4scale/prep/derivation/`)

`run_`, `preflight_`, `monitor_`, `closeout_`, `postrun_`, `archive_e1_v4scale.sh`, `pin_v4scale.sh`, `hash_scripts_v4scale.sh`,
`rehearse_e1_v4scale.sh`, `rehearse_e2e_v4scale.sh`, `e1_v4scale_{rehearsal,eightwave,identity_rehearsal,mock_launcher,e2e_check}.py`.
Each was produced from its V3 counterpart by a fixed renaming plus listed edits; the directory holds one diff per file showing
only those edits. The V3 files are not modified (their manifests verify). Differences that are not renamings or new limits:

1. the driver applies the launch rule of `E1-V4-TIMELINE.md` §2 before it starts anything (exit 7 on refusal);
2. the supervisor is the V4 supervisor with the reservation guard and the response latch of §5;
3. the memory and storage samplers and the supervisor run for up to 5400 s instead of 3400 s (the export after the release
   takes about 20 min at this size; with 3400 s the samplers would end first and the supervisor would abort a healthy run);
4. the post-run's identity step is the V4 checker and policy, and the gates receive the completeness record;
5. the preflight regenerates the inputs (`--check`), runs the identity rehearsal on the actual V4 config, checks the
   reservation budget's arithmetic, and checks host memory exactly (kB) against 512 GiB with five boundary fixtures;
6. the archive script measures run root + archive with `checks/e1_archive_size.py` (38 GB);
7. closeout and monitor print the reservation's deletion deadline;
8. (revision 3) the driver verifies the reservation's ownership, arms the closeout watcher and requires the deleter's heartbeat
   before the launch rule; closeout reports the statements of §8; the monitor reports the watcher, the deleter and a stalled
   release; the gates include G16; the rehearsal drives the watcher and the deleter against a stand-in for kubectl
   (`e1_v4scale_mock_kubectl.py`) in every scenario. New files: `arm_closeout_e1_v4scale.sh`,
   `checks/e1_reservation_closeout.py`, `checks/e1_closeout_verdict.py`, `e1_v4scale_mock_kubectl.py`.

Revision 2 derivation: `runs/v4scale/prep2/derivation/` (one diff per file against the frozen V3 original, all edits of both
revisions). Revision 3: `runs/v4scale/prep3/derivation/` (one diff per changed file against revision 2 at `76d95289`).

## 11. Preparation evidence required before a freeze could be approved

Guarded CPU-only tests; the identity rehearsal on the actual V4 config; the complete V4 post-run on the V4-shaped run directory
with registered expectations; the synthetic fixture matrix (133 fixtures: 3 positive, 6 boundary, 124 negative; every ceiling
and deadline at its boundary; 15 for the reservation deletion); the eight-wave completeness fixtures; the mocked end-to-end
rehearsal (eighteen scenarios: the thirteen of revision 2, each now with the watcher and the deleter, and five for the latch
and the deletion: a breach followed by the guard, the deleter not armed, the reservation pod replaced, the deleter lost, a late
deletion); the preflight on the capture tray. Every attempt, including failed ones, is kept in its own read-only directory
(`runs/v4scale/prep/`, `runs/v4scale/prep2/`, `runs/v4scale/prep3/`).

## 12. Open decisions for review

1. Approval of the freeze as proposed in revision 3, or changes.
2. The remaining findings of `E1-V4-TIMELINE.md` §7: the deleting side runs on the operator's laptop; the live cluster's
   handling of the UID precondition is proven only by the dry run of step 1 at launch time; a reservation that disappears
   during the run is reported, not acted on; a stalled release is left to the controller until the guard; before a stop request
   a lost supervisor is covered by the monitor only.
3. Whether the early reservation release should be developed further (it needs gate changes; not part of this packet; not
   enabled).

## 13. Revision 2 (review decision of 2026-11-19 on `39561325`)

| item | revision 1 | revision 2 |
|---|---|---|
| preflight host MemAvailable | ≥ 448 GiB, whole GiB | ≥ 488 GiB = importer ceiling 128 + runtime floor 64 + 296 for the run (V3 measured 295.2 at the stop request); exact in kB; fixtures at 487, 488 − 1 kB, 488, 488 + 1 kB, 489 GiB |
| runtime floor, importer ceiling, breach responses | 64 GiB, 128 GiB, sampler → supervisor | unchanged |
| launch rule | age ≤ 600 s | 4680 − age ≥ 4210 = 20 + 2700 + 90 + 1400 (age ≤ 470 s); refusal before anything is started |
| CPU-only processing before the deletion | not bounded explicitly | 1400 s in the run budget (stress prediction 1360 + 40 s tail); absolute end at the guard |
| reservation limit, guard, closeout budget | 90 min, 4680 s, 720 s | unchanged |
| trace, config, expected size, controller, deadlines, GPU-held and storage ceilings, identity policy rules, gates | — | unchanged (trace and config byte-identical; the identity policy file changes only in the pinned hash of `E1-V4-SCALE-INPUTS.json`) |
| early reservation deletion | not enabled | not enabled |

## 14. Revision 3 (review decision of 2026-11-19 on `76d95289`: final CPU-only safety closure)

| item | revision 2 | revision 3 |
|---|---|---|
| preflight host MemAvailable | ≥ 488 GiB = 128 + 64 + 296 | ≥ 512 GiB = importer ceiling 128 + runtime floor 64 + 320 for the run (V3 measured 295.2: margin 24.8 GiB, more than the growth projected for the longer capture); exact in kB; fixtures at 511, 512 − 1 kB, 512, 512 + 1 kB, 513 GiB; the 487 – 489 GiB fixtures are kept and are all refused |
| runtime floor, importer ceiling, breach responses | 64 GiB, 128 GiB, sampler → supervisor | unchanged |
| supervisor latch | one response; after a release line the guard was only recorded | release line at most once, cleanup at most once and final, never concurrent; the guard is always answered by the cleanup (directly, or as an escalation after an earlier release line); a stalled release is reported |
| guard during finalisation (not reachable under the launch rule) | release line | registry cleanup (the release line does not end the run) |
| reservation deletion | by the operator after a CLEAN closeout, deadline printed | registered watcher (tray) and deleter (laptop), §8; exact ownership (UID precondition); evidence before the permit; no early release; verdicts; gate G16 |
| launch prechecks | reservation record, launch rule | reservation record with deletion path proof, live pod = record, watcher armed, deleter heartbeat, launch rule |
| closeout | one result; a lost supervisor could be CLEAN | resource cleanup, supervision, run response, reservation deletion and gates stated separately; lost supervisor → supervision INVALID → NOT CLEAN |
| gates | G0 – G15 (16 results) | G0 – G16 (17 results) |
| closeout and deletion budget | 720 s, operator-dependent (345 s mechanical) | 720 s, no operator action required (≤ 300 s mechanical) |
| trace, config, expected size, controller and its 60 s grace, deadlines, GPU-held and storage ceilings, reservation limit, guard, launch rule, identity policy rules | — | unchanged (trace and config byte-identical; the identity policy file changes only in the pinned hash of `E1-V4-SCALE-INPUTS.json`) |
| early reservation deletion | not enabled | not enabled |
| V1 – V3 records, frozen manifests and gate code | — | not modified |
