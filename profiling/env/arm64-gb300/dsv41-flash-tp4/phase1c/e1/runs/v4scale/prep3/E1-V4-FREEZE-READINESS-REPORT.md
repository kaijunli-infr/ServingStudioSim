# E1 V4 — final CPU-only safety closure: freeze-readiness report (2026-11-19)

Review decision on `76d95289`: CPU-only preparation accepted with remaining safety findings; V4 freeze NO GO; V4 GPU execution
NO GO; GO for a final CPU-only safety closure. This report covers that closure.

**Status.** Proposed packet revision 3 at `996074f9` (code `35ddd280`). No GPU was used. No pod was created and no request other
than read-only queries (`auth can-i`, `get`) was sent to the cluster. Nothing is frozen, nothing was launched. Frozen V1 – V3
records, manifests and gate code are unchanged. **Stopping for review.**

| check at the proposed packet | result |
|---|---|
| read-only preflight on tray16 | 86 PASS, 0 FAIL |
| guarded CPU-only tests, targeted / whole tier (tray14) | 1052 passed, 1 skipped / 5967 passed, 78 skipped, 6 failed (the same failing tests as the baseline; unrelated) |
| identity rehearsal on the actual V4 config | 13 of 13 expectations, including 7 negative controls |
| complete V4 post-run on the V4-shaped directory | FAIL on exactly G8b, G9, G15, G16, as registered (V3 capture data; the V3 reservation was deleted by hand, which G16 does not accept) |
| synthetic gate fixtures | 133 of 133 (3 positive, 6 boundary, 124 negative) |
| eight-wave completeness fixtures | 10 of 10 |
| mocked end-to-end scenarios | 18 of 18 scenarios, 398 of 398 checks, first run at this commit |

## 1. Memory

Preflight minimum 488 → **512 GiB** = importer ceiling 128 + runtime floor 64 + 320 for the run. The check stays exact in kB
(`checks/e1_preflight_memory.py`, unchanged code; the sum is asserted by the checker, the preflight and the tests). Runtime
floor 64 GiB, importer ceiling 128 GiB and the breach responses (sampler → supervisor) are unchanged.

| fixture (MemAvailable) | result |
|---|---|
| 511 GiB; 512 GiB − 1 kB | refused |
| 512 GiB; 512 GiB + 1 kB; 513 GiB | accepted |
| 487, 488 − 1 kB, 488, 488 + 1 kB, 489 GiB (the fixtures of the previous proposal, kept) | all refused |

Against V3: 512 − 295.2 (footprint at the stop request) − 128 = 88.8 GiB left, 24.8 above the floor; with the growth projected
for V4's longer capture (5.0 GiB, a projection from one run) 83.8 GiB left. `feasibility/E1-V4-MEMORY-BUDGET.md`.

## 2. Supervisor latch (`checks/e1_supervisor_v4.py`)

The supervisor now runs its own loop with the cause, acknowledgement and cleanup helpers of the unmodified V3 supervisor.
Release line (R1) at most once; registry cleanup (R2) at most once and final; never concurrent; a reservation-guard cause is
always answered by R2. Unit tests use the actual supervisor with the actual pinned cleanup on real stand-in processes; an
unregistered importer stand-in and an unregistered Nsight-agent stand-in must be alive and unsignalled afterwards.

| case | result |
|---|---|
| earlier release acknowledged and completing; run ends before the guard | one release line, no cleanup, nothing ended by the supervisor |
| earlier release acknowledged and completed; run still going at the guard | cleanup at the guard (escalation); one line, one cleanup. Also end-to-end (`breach_then_guard`): release acknowledged after 0.99 s, guard seen at 420.3 s (rehearsal guard 420 s), cleanup finished 0.44 s later |
| earlier release acknowledged but stalled | reported as stalled after 150 s; nothing ended before the guard; cleanup at the guard |
| missing acknowledgement | cleanup after the 10 s wait, never overlapping it; the guard passing meanwhile adds no second cleanup |
| guard reached after an earlier breach response | the guard is recorded as a later cause **and** enforced by the cleanup (previously recorded only) |
| simultaneous breach and reservation deadline (also during finalisation) | one response, the cleanup; no release line |
| importer still active during a deadline response | registered tree ended; importer and agent stand-ins alive, in no signal list; stream bytes unchanged |

A guard during finalisation now gives the cleanup, not the release line (the release line does not end the run). That
state is not reachable under the launch rule. Other causes after an acknowledged release are recorded and do not act.

## 3. Closeout and reservation deletion (`checks/e1_reservation_closeout.py`)

The capture tray's credential can read pods and cannot delete them (checked by the preflight); the deleting credential exists
only on the operator's laptop and is not copied. The mechanism therefore has two registered, bounded processes.

- **Watcher** (tray, registry `pids/closeout.tsv`, armed by the driver before the launch rule). Waits while any registered
  process of the run is alive: no early release. Then permits only with all of: no live owned process and no refusal; no
  unregistered process that looks like the experiment; two consecutive clean GPU censuses; the live pod exactly the recorded
  one. Observes the deletion with the tray's read-only credential and writes the verdict. It cannot delete and signals nothing.
- **Deleter** (laptop, own registry, started before the driver; its polling is the heartbeat the driver requires). Deletes
  only after two consecutive permits of the same watcher for the recorded UID and node and its own ownership check. One
  `DELETE` with `preconditions.uid`. A pod of the same name with another UID is not deleted.
- **Ownership.** `record` writes `reservation.json` from the live pod (name, namespace, UID, creation time, node, labels, pause
  containers only, GPU request, no owner) and proves by two dry runs that the server accepts a delete with the recorded UID
  and refuses one with another UID. The driver refuses to start without that record and without the live pod matching it.
- **Deadline.** Both sides use the pod's creation time + 5400 s. `DELETED_IN_TIME` only at or before it. Late, not deleted
  (600 s after the ceiling), gone without a permit, unverifiable, foreign: all failures. Gate **G16** fails on any of them and
  on a missing record; it has no INCONCLUSIVE outcome.
- **Budget.** Mechanical path from the end of the run to the observed deletion ≤ 300 s of the 720 s; measured in the
  rehearsal 18 – 25 s after a registry cleanup and 51 – 63 s after a normal end (the stand-in pod takes 3 s to go; the real one took 62 s in V3).

A live importer or Nsight agent is recorded and does not block the permit (no GPU, never signalled). Early release after a
verified GPU release stays disabled.

## 4. Supervisor health and closeout verdict (`checks/e1_closeout_verdict.py`)

The closeout prints its resource items, then five lines and one result:

| statement | lost supervisor (`supervisor_lost`) | abort (`breach_then_guard`) | late deletion (`deletion_late`) |
|---|---|---|---|
| resource cleanup | CLEAN | CLEAN | CLEAN |
| supervision | **INVALID** | VALID | VALID |
| run response | none | ABORTED | none |
| reservation deletion | DELETED IN TIME | DELETED IN TIME | **FAIL** |
| frozen scientific gates | reported from the post-run record only | | |
| combined result | **NOT CLEAN** | NOT CLEAN | NOT CLEAN |

In revision 2 the lost-supervisor run was reported CLEAN. V1 – V3 closeout scripts and records are not changed.

## 5. Rehearsal at the proposed packet

Attempt a3 at `996074f9`, first run at that commit. Every scenario runs the actual driver, supervisor, watcher, deleter, closeout
and (where a run completed) post-run around mock processes; the cluster is a stand-in for kubectl.

| scenario | checks | supervisor steps | watcher / deleter verdict | end of run → deletion observed | closeout: resources / supervision / run response / deletion | result |
|---|---|---|---|---|---|---|
| `success` | 26/26 | – | DELETED_IN_TIME / DELETED_IN_TIME | 57.5 s | CLEAN / VALID / none / DELETED IN TIME | CLEAN |
| `ambiguous` (two importer candidates) | 25/25 | – | in time / in time | 63.1 s | CLEAN / VALID / none / DELETED IN TIME | CLEAN |
| `memory_breach` (during the import) | 24/24 | R1 | in time / in time | 55.9 s | CLEAN / VALID / ABORTED / DELETED IN TIME | NOT CLEAN |
| `storage_breach` (during the capture) | 23/23 | R2 | in time / in time | 22.7 s | NOT CLEAN / VALID / ABORTED / DELETED IN TIME | NOT CLEAN |
| `reservation_guard` (during the capture) | 24/24 | R2 | in time / in time | 20.1 s | NOT CLEAN / VALID / ABORTED / DELETED IN TIME | NOT CLEAN |
| `launch_late_permitted` (check at age 437.9 s) | 24/24 | – | in time / in time | 50.7 s | CLEAN / VALID / none / DELETED IN TIME | CLEAN |
| `launch_refused` (check at age 529.7 s, exit 7) | 11/11 | – | in time / in time | – | CLEAN / NOT APPLICABLE / none / DELETED IN TIME | CLEAN |
| `ceiling_before_stop` | 24/24 | R2 | in time / in time | 24.8 s | NOT CLEAN / VALID / ABORTED / DELETED IN TIME | NOT CLEAN |
| `complete_near_guard` | 24/24 | – | in time / in time | 55.9 s | CLEAN / VALID / none / DELETED IN TIME | CLEAN |
| `export_near_guard` | 26/26 | R2 | in time / in time | 18.3 s | CLEAN / VALID / ABORTED / DELETED IN TIME | NOT CLEAN |
| `guard_and_storage` | 24/24 | R2 | in time / in time | 20.2 s | NOT CLEAN / VALID / ABORTED / DELETED IN TIME | NOT CLEAN |
| `sampler_lost` | 24/24 | R2 | in time / in time | 19.6 s | NOT CLEAN / INVALID / ABORTED / DELETED IN TIME | NOT CLEAN |
| `supervisor_lost` | 20/20 | – | in time / in time | – | CLEAN / **INVALID** / none / DELETED IN TIME | **NOT CLEAN** |
| `breach_then_guard` (new) | 25/25 | R1 + R2 | in time / in time | 20.9 s | CLEAN / VALID / ABORTED / DELETED IN TIME | NOT CLEAN |
| `deleter_not_armed` (new; exit 8) | 9/9 | – | NOT_DELETED / no record | – | CLEAN / NOT APPLICABLE / none / FAIL | NOT CLEAN |
| `reservation_replaced` (new) | 22/22 | – | GONE_WITHOUT_PERMIT / GONE_NOT_BY_DELETER | – | CLEAN / VALID / none / FAIL | NOT CLEAN |
| `deleter_lost` (new) | 21/21 | – | NOT_DELETED / no verdict | – | CLEAN / VALID / none / FAIL | NOT CLEAN |
| `deletion_late` (new) | 22/22 | – | DELETED_LATE / DELETED_LATE | – | CLEAN / VALID / none / FAIL | NOT CLEAN |

18 of 18 scenarios met, 398 of 398 checks. In every scenario with a deletion: exactly one delete request that was not a dry
run, by the deleting side, with the recorded UID as precondition; none with the tray's credential; the first permit after
the end of the run. In the four deletion failures nothing was deleted without a permit, the foreign pod was untouched, and
gate G16 failed. "NOT CLEAN" resources after an R2 during the capture are the non-terminal lifecycle phase (no report exists).

Negative controls beyond these: gate fixtures (15 for G16: watcher record missing, one side late by 0.5 s, not deleted, unverified, gone without a permit,
request before the permit, foreign UID refused, another UID, no precondition, two accepted requests, permit without the second
census, with a live registered process, without ownership verification, record without UID, watcher incomplete); unit tests
with a server that ignores the precondition (`record` refuses), a pod replaced between the ownership check and the request
(the precondition stops it), permits from another watcher, for another UID or node, a single permit, alternating permits,
an unreachable cluster, a busy GPU, a single clean census, an unregistered lookalike, a registry refusal.

## 6. Attempts (all kept, each in its own read-only directory; copies under `runs/v4scale/prep3/attempts/`)

| attempt | commit | what | result |
|---|---|---|---|
| `unit-tests-uncommitted-tray14/…-dev1` | uncommitted | targeted tests | 414 passed, **3 failed**: a `None` in G16's late-deletion detail (my defect, fixed), and a test of mine that forbade a phrase the closeout script printed |
| `…-dev2` | uncommitted | same | 416 passed, **1 failed**: the same test tripped over the preflight's own grep pattern (test corrected) |
| `…-dev3`, `…-dev4`, `…-dur1` | uncommitted | that test; all targeted tests; durations | 43 passed; 418 passed; 265 passed |
| `rehearsal-uncommitted-r3dev1` | uncommitted | gate rehearsal | met: 133 of 133 fixtures; post-run FAIL on exactly G8b, G9, G15, G16 |
| `devsmoke-a`, `devsmoke-b` | uncommitted | development smoke of 7 end-to-end scenarios with the driver's own preflight removed (it needs a pinned tree); not evidence | 7 met |
| `attempt-…-a1-fedd6476` | `fedd6476` | preflight, rehearsal, end-to-end | **preflight FAIL**: 84 PASS, 2 FAIL, both new credential checks: kubectl is in the shared home's `.local/bin`, not on the standalone preflight's PATH (fixed in `10272704`). Rehearsal met. **End-to-end stopped by me in its first scenario** on the assumption that the driver's preflight would fail the same way. It would not have: the driver sets PATH, and its preflight had passed. Stopped through the verified cleanup of the registries the rehearsal itself had written (`a1_stop_*.jsonl`, `a1_stopped_e2e_records/`); nothing by pattern |
| `regression-…-fedd6476` | `fedd6476` | targeted / whole tier | 1052 passed, 1 skipped / 5967 passed, 78 skipped, 6 failed (the baseline's failing tests) |
| `attempt-…-a2-314d0b36` | `314d0b36` | preflight, rehearsal, end-to-end | preflight 86 PASS, 0 FAIL; rehearsal met; **end-to-end 17 of 18**: `launch_late_permitted` not met. The driver now needs 161 – 167 s from its start to the launch rule; with the reservation backdated by 320 s the check came at 482.9 s and the rule refused, correctly. The scenario's backdating was stale, not the rule (fixed in `35ddd280`: 275 s and 365 s) |
| `regression-…-314d0b36` | `314d0b36` | targeted / whole tier | as above |
| `attempt-…-a3-996074f9` | `996074f9` | preflight, rehearsal, end-to-end | **preflight 86 PASS, 0 FAIL; rehearsal met; end-to-end 18 of 18, 398 of 398 checks** |
| `regression-…-996074f9` | `996074f9` | targeted / whole tier | 1052 passed, 1 skipped / 5967 passed, 78 skipped, 6 failed (the baseline's failing tests: energy-performance and emit-backends, unrelated) |
| `preflight-…-final-…` | evidence commit | read-only preflight | added by the commit that follows this report (86 PASS, 0 FAIL expected; the result is in that directory) |

The scratch originals are read-only; nothing was overwritten. The rehearsal script now refuses an output directory that
exists.

## 7. Unchanged

224 requests, eight waves, four copies; 8640 iterations and 49 003 328 kernel rows; trace and config byte-identical; controller
and its 60 s grace; deadlines 300 / 300 / 600 s; GPU-held 2700 s; storage 15 / 18 / 38 GB; reservation limit 90 min, guard
4680 s, launch rule (age ≤ 470 s); identity policy rules (the policy file changes only in the pinned hash of the inputs file);
gates G0 – G15. Early reservation release: not enabled.

## 8. Remaining findings for review

1. **The deleting side runs on the operator's laptop.** If it is offline at the end of the run, nothing deletes the
   reservation: the watcher records `NOT_DELETED`, G16 fails, the GPUs are released regardless. The driver refuses to start
   without the deleter's heartbeat and the monitor reports a stale one.
2. **The UID precondition was not exercised against the real cluster** (no pod was created under this authorization). It is
   proven per reservation by the dry runs of `record`; without that proof the driver does not start. `kubectl delete --raw`
   was confirmed against a local stand-in server to send the options body (without a Content-Type header; whether the
   cluster accepts that is exactly what the two dry runs prove).
3. **A reservation pod that disappears during the run** is detected within a minute and reported (`GONE_WITHOUT_PERMIT`,
   monitor `ABORT-CONDITION`); the run is not ended automatically.
4. **A stalled release** is reported after 150 s and left to the controller's deadlines until the guard.
5. **Before a stop request, a lost supervisor** is still covered by the operator's monitor only. Such a run can no longer be
   CLEAN, and the watcher does not permit a deletion while the run is active (the reservation then outlives its 90 minutes
   and fails G16, with the GPUs still reserved).
6. **Memory growth over the longer capture** is unmeasured; 512 GiB covers the projection with about 20 GiB to spare.
7. **Launch window.** The driver now needs 161 – 167 s from its start to the launch rule (35 rehearsal launches in attempts a1 – a3: 160.7 – 167.5 s; 78 s in V3):
   its preflight runs 795 tests, then come the ownership check and the arming. At V3's manual operator pace (236 s before
   the driver) the launch age would be about 400 s of the 470 s; with the pre-launch steps scripted, about 200 s. A late
   start is refused and the mechanism deletes the reservation. No ceiling and no part of the driver's preflight was changed.

## 9. Operator sequence that revision 3 implies (for review; not executed)

1. Laptop: create the reservation pod; `record` → `reservation.json`; copy it to the run directory on the tray.
2. Laptop: start the deleter, registered, with the two neutral-path helper commands through ssh.
3. Tray: start the driver (neutral wrapper, tmux). It runs the preflight, checks the record and the live pod, arms the watcher,
   requires the deleter's heartbeat, applies the launch rule, and only then starts observers, supervisor and launcher.
4. Monitor every 60 s. No operator action is needed for the deletion.
5. After the watcher's verdict: closeout, post-run, archive.
