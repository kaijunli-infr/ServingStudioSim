# V4 worst-case timeline, reservation budget and compatibility of the limits (CPU-only analysis; revision 3, 2026-11-19)

Revision 3 follows the review decision on `76d95289`: preflight host memory 488 → 512 GiB; a supervisor latch under which the
reservation guard is acted on in every state; a registered, bounded closeout and deletion of the reservation; separate closeout
statements. Revision 2 (on `39561325`) introduced the launch eligibility rule and the bound on CPU-only processing. The
90-minute reservation limit, the 4680 s guard, the launch rule, every deadline, every other ceiling, the controller and its
60 s server grace are unchanged. Earlier revisions are in the history at `39561325` and `76d95289`.

**M** measured (V2 / V3), **E** predicted (`feasibility/E1-SCALING-MODEL.md`; range, then stress case), **L** a limit with the
mechanism that enforces it.

## 1. Clock definitions

| clock | starts | ends | limit | read from | enforced by |
|---|---|---|---|---|---|
| reservation | pod creation (`reservation.json` `creation_timestamp`) | deletion observed, by the watcher (`reservation_closeout.json`) and by the deleter (`reservation_deleted.json`) | 5400 s | kubectl records of both credentials | launch rule + supervisor guard + watcher and deleter; gates G10, G16 |
| launch age | pod creation | the driver's eligibility check (before any observer or the launcher exists) | 470 s (derived, §2) | `run.log` | driver, exit 7 |
| check → launcher start | eligibility check | `lifecycle.json` `launch_started_at` | 20 s allowance (M: 1 s in V3) | records | not enforced; inside the budget |
| GPU-held | `launch_started_at` | GPUs released (`gpu.released_at`; sampler upper edge) | 2700 s | controller and GPU sampler | trigger at 2700 s: supervisor before a stop request (registry cleanup), controller after it (server stop). Gate G10 judges launch → **released** ≤ 2700 s |
| release | server SIGTERM (or cleanup SIGTERM) | GPUs observed free | 90 s | GPU sampler | controller: 60 s grace, then SIGKILL; cleanup: 30 s grace, then SIGKILL; gate G6 |
| CPU-only processing under the reservation | GPUs released | the driver's stop file (launcher ended, first census, 20 s) | 1400 s budget (E stress 1360 s export and parse + 40 s tail; M tail 21 s) | records | the reservation guard: age 4680 s is the absolute end |
| guard | pod creation | — | 4680 s | supervisor record | supervisor: registry cleanup, in every state of its latch (§5) |
| closeout and deletion | guard (latest end of the run) | deletion observed | 720 s | records | watcher (evidence, permit, observation) and deleter (one delete with the UID precondition); no operator action; gate G16 |

Release time and the GPU-held ceiling. Gate G10 measures GPU-held up to the release, so in a run that passes, the release is
inside the 2700 s. Enforcement is different: the ceiling is a trigger at 2700 s on the launch clock and the release follows it
by up to 90 s. The run budget therefore contains 2700 + 90 once, as the enforced bound on launch → GPUs free. It is not added
a second time anywhere: the 1400 s of CPU-only processing start at the release.

## 2. Launch eligibility rule

```
run budget  B = 20 (check → launcher start) + 2700 (GPU-held ceiling) + 90 (release after the ceiling) + 1400 (CPU-only) = 4210 s
eligible    iff  4680 - age >= 4210        (age <= 470 s; equality is permitted; a negative or unreadable age refuses)
```

Evaluated by the driver (`checks/e1_reservation_guard.py eligible --guard-s 4680 --run-budget-s 4210`) after its preflight and
before it starts any observer, the supervisor or the launcher. Refusal: exit 7, `LAUNCH REFUSED` in `run.log`, no observer and
no launcher started, no GPU touched; the closeout watcher, armed just before the rule, permits the reservation's deletion once
the driver is gone. The values are constants of the driver, asserted against
`E1-V4-SCALE-INPUTS.json` by the preflight; only rehearsal mode can override them.

| check | arithmetic | result |
|---|---|---|
| required worst-case path with an immediate launch | 0 + 4210 = 4210 ≤ 4680 | fits, 470 s to spare |
| launch window | 0 ≤ age ≤ 470 s | exists |
| measured launch age | V3: 316 s (operator 236 s + driver preflight 78 s + 1 s). V4 revision 3, 35 rehearsal launches (attempts a1 – a3): 160.7 – 167.5 s from the driver's start to the eligibility check (preflight about 150 s, its test suites having grown to 795 tests; then ownership check, arming and the armed check, about 10 s, bounded at 45 s) → about 400 s at V3's manual operator pace, plus the `record` step; about 200 s if the operator's steps before the driver are scripted | inside the window; **margin only about 60 s at V3's manual pace** (§7.7); a slower start is refused, not launched |
| latest launch, ceiling and release against the guard | 470 + 20 + 2700 + 90 = 3280 < 4680 | the guard can only fall into the CPU-only phase |
| guard + closeout budget against the reservation limit | 4680 + 720 = 5400 | equal by construction |
| predicted run at the latest launch | 470 + 20 + 2311 (E stress GPU-held) + 1360 + 40 = 4201 | ends 479 s before the guard |
| **all extremes together** (the corner of revision 1) | 470 + 20 + 2700 + 90 + 1400 = 4680 | **ends at the guard, not after it** |

The corner that revision 1 reported as not completable (latest launch + ceiling-long GPU hold + stress export) now fits: the
launch window was reduced from 600 s to 470 s instead of any ceiling being changed.

## 3. One timeline

| # | step | nominal | worst case allowed | enforced by |
|---|---|---|---|---|
| 1 | reservation acquisition → `record` → deleter start → driver preflight → ownership check, arming → eligibility check | ≈ 200 – 400 s (M V3 operator 236 s, manual; M V4 rehearsal driver start → check 161 – 167 s) | **age ≤ 470 s** | L: launch rule, exit 7 |
| 2 | launcher start; GPU acquisition, model startup, warm-up | 1487 – 1640 s (M: V2 1487, V3 ≈ 1523) | inside the GPU-held ceiling; operator abort condition at 30 min | monitor; supervisor at the ceiling |
| 3 | measurement: 8 waves, 224 requests | ≈ 313 s (E) | inside the ceiling | monitor; supervisor at the ceiling |
| 4 | stop request accepted | 0.003 s (M) | ≤ 30 s | L: G4 |
| 5 | flush / import / stop → report | 203 – 224 (249) / 97 – 106 (118) / 301 – 331 (367) s (E) | ≤ 300 / 300 / 600 s | L: controller hard deadline; G3, G4 |
| 6 | GPU release | ≤ 2.9 s (M) | ≤ 90 s | L: controller grace + SIGKILL; G6 |
| 6 | GPU-held, launcher start → released | 2092 – 2275 s (2311) (E) | 2700 s for a PASS; 2700 + 90 s enforced | L: supervisor / controller; G10 |
| 7 | export and parse after the release, tail to the stop file (CPU only) | 1220 – 1226 s (1360) + 21 s (E, M) | 1400 s in the budget; absolute end at age 4680 s | L: reservation guard |
| 8 | observers drain, driver end; watcher: evidence and permit | 12 s (M V3) + two evaluations of 5 s | inside the 720 s | watcher (no early release: only after the registered tree is gone) |
| 9 | deleter: two permits, ownership check, one delete; deletion observed by both | ≈ 15 s + 62 s after the request (M V3) | **age ≤ 5400 s** | deleter (UID precondition), watcher; G10, G16 |
| — | closeout script (after the watcher's verdict): resource items and the separate statements | ≈ 1 min | not under the reservation | G12 |
| — | post-run and archive | ≈ 8 + ≈ 10 min (E) | after the deletion; CPU only | not under the reservation |

Durations: GPU-held 35 – 39 min predicted (bound 2700 s, enforced 2790 s). Reservation occupancy about 70 – 80 min predicted
with V3's operator timings (bound 90 min). CPU-only processing under the reservation: 21 – 23 min predicted (budget 1400 s,
absolute end at the guard). CPU-only post-processing after the deletion: about 18 min.

Closeout and deletion budget (720 s), no operator action in it: supervisor cleanup response ≤ 45 s (only if the guard acted) +
observers drain and driver end 60 s (M 12 s) + watcher evidence and permit 30 s (two 5 s evaluations with two GPU censuses) +
deleter from the permit to the request 45 s (two fetches of 5 s through ssh, one ownership check) + deletion request →
observed 120 s (M 62 s in V3) = 300 s, leaving 420 s. In V3 the operator took 832 s from the stop file to the observed
deletion; that step no longer exists. The watcher records the deadline and both sides record a late deletion as a failure.

## 4. If the stop request never occurs

1. 30 min after the launch: monitor `ABORT-CONDITION startup exceeds 30 min`; 40 min: `no stop request` (operator actions).
2. 2700 s after the launch: supervisor cause `gpu_held_ceiling_before_stop` → pinned registry cleanup (SIGTERM, 30 s grace,
   SIGKILL). GPUs free within 90 s. No stream exists; the run is a FAIL; the watcher permits and the deleter deletes the
   reservation; closeout NOT CLEAN (run response ABORTED).
3. Latest GPU release on this path: 470 + 20 + 2700 + 90 = 3280 s after the pod's creation; 2120 s remain for the deletion.

## 5. Supervisor latch, and what happens if the supervisor, a monitor, the watcher or the deleter fails

Latch of `checks/e1_supervisor_v4.py`: the release line (R1) at most once; the registry cleanup (R2) at most once and final;
R2 never starts while R1 waits for its acknowledgement; a reservation-guard cause is always answered by R2.

| state when the guard passes | response |
|---|---|
| no earlier response | R2 (also during finalisation: the release line would release the GPUs only) |
| R1 acknowledged, release completed, run still going (export, or importer still importing) | R2 (escalation); importer and Nsight agent are not in the registry and are not signalled |
| R1 acknowledged, release stalled | reported as stalled after 150 s (the controller's 60 s server grace + the 90 s release bound; monitor `ABORT-CONDITION`), left to the controller's deadlines; at the guard R2 |
| R1 not acknowledged within 10 s | R2 already ran at that point (once); the guard adds nothing |
| R2 already ran | the supervisor has ended; the registered tree is gone |

| failure | what still acts | consequence |
|---|---|---|
| memory, storage or GPU sampler stops | supervisor: `monitor_lost` after 60 s → response | FAIL; closeout: supervision INVALID; stream preserved if the flush had completed |
| import probe stops | nothing aborts | G4 / G5 INCONCLUSIVE |
| supervisor stops | after the stop request the controller still enforces the 600 s deadline and the ceiling. Before it, and for the guard: nothing automatic ends the run. Monitor: `ABORT-CONDITION the supervisor has not checked for 30 s`, and at 78 min `the reservation is older than 78 min and the run has not ended` | G11 INCONCLUSIVE; closeout: supervision INVALID → NOT CLEAN → G12 FAIL. The watcher still permits when the run ends, and the reservation is deleted in time if it ends before the ceiling |
| run still going at the ceiling with a lost supervisor | the watcher does not permit (registered processes alive, GPUs possibly held): the reservation is NOT deleted | G16 FAIL (`NOT_DELETED`); the GPUs stay reserved, which is the safe side; operator: verified registry cleanup |
| launcher dies | driver ends; watcher: evidence, permit; closeout NOT CLEAN | FAIL; reservation deleted in time |
| watcher stops | nothing permits; the deleter deletes nothing | monitor `ABORT-CONDITION the closeout watcher has not evaluated for 60 s`; G16 FAIL unless the operator restores it in time |
| deleter stops, or the laptop is offline | the watcher permits and observes; nobody deletes | monitor `ABORT-CONDITION the deleter's heartbeat is … old`; watcher verdict `NOT_DELETED` 600 s after the ceiling; G16 FAIL; GPUs released regardless |
| the reservation pod disappears or is replaced during the run | the watcher reports `GONE_WITHOUT_PERMIT` and ends; the deleter deletes nothing (a pod of the same name with another UID is foreign) | monitor `ABORT-CONDITION the reservation pod is gone while the run is active`; G16 FAIL; the run is not ended automatically |
| the cluster cannot be read at the end | no permit, or no observation | `UNVERIFIED` or `NOT_DELETED`; G16 FAIL |

## 6. Compatibility of the limits

| relation | check | result |
|---|---|---|
| flush + import against stop → report | 300 + 300 = 600 | compatible |
| forced release against the server grace | 60 s + SIGKILL < 90 s | compatible |
| phases at their limits against the GPU-held ceiling | startup + 313 + 600 + 60 ≤ 2700 → startup ≤ 1727 s; measured 1487 – 1640 s | compatible (margin ≥ 87 s) |
| launch rule, ceiling, release, CPU budget against the guard | §2 | compatible, including all extremes together |
| guard + closeout against the reservation | 4680 + 720 = 5400; mechanical closeout and deletion ≤ 300 s | compatible, without operator action |
| watcher lifetime | it ends at the observed deletion, at the latest 600 s after the ceiling | bounded |
| sampler lifetimes | memory sampler, storage sampler and supervisor run 5400 s from their start (shortly before the launcher), so past the guard at every permitted launch age; GPU sampler 3300 s ≥ 2700 + 90 + 300 | compatible |
| storage | 18 GB + archive (1.105 ×) = 37.9 ≤ 38 GB; predictions 10.4 – 11.4 / 12.6 – 14.0 / 26.5 – 29.5 GB | compatible |
| importer ceiling + floor + run footprint against the preflight minimum | 128 + 64 + 295.2 (M, V3 at the stop request) = 487.2 ≤ 512 | compatible with the V3 measurement, margin 24.8 GiB; with the growth projected for the longer capture (about 5 GiB) still about 20 GiB; see §7.1 |

## 7. Remaining findings

1. **Memory.** 512 GiB leaves 24.8 GiB over the footprint measured in V3. The growth over V4's four additional waves is still
   unmeasured (projection from one run: about 5 GiB). During the run the protection is the 64 GiB floor, not the preflight.
   Details: `feasibility/E1-V4-MEMORY-BUDGET.md`.
2. **The deleting side runs on the operator's laptop**, because the tray's credential cannot delete pods and no deleting
   credential is placed in the shared account. If the laptop is offline at the end of the run, nothing deletes the reservation;
   the watcher records `NOT_DELETED` and the run fails G16. The driver refuses to start without a fresh deleter heartbeat, and
   the monitor reports a stale one.
3. **The UID precondition on the real cluster** was not exercised here (no pod was created, no request was sent to the
   cluster). It is proven per reservation by the dry run of `record` before the launch: accepted with the recorded UID, refused
   with another; without that proof the driver does not start.
4. **A reservation that disappears during the run** is detected within a minute and reported; the run is not ended
   automatically (that would be a new supervisor cause).
5. **A stalled release** after an acknowledged release line is reported after 150 s and left to the controller's deadlines
   until the guard.
6. **Before a stop request, a lost supervisor** is covered by the operator's monitor only (unchanged from V3). The closeout can
   no longer call such a run CLEAN, and the watcher never permits a deletion while the run is active.
7. **The launch window is 470 s and the driver now needs 161 – 167 s of it** (measured in 35 rehearsal launches; 78 s in V3,
   110 – 116 s in revision 2): its preflight runs 795 tests and the arming follows. At V3's manual operator pace (236 s
   before the driver) the launch age would be about 400 s, about 60 s inside the window; with the pre-launch steps scripted,
   about 200 s. A refusal is safe (the watcher permits, the deleter deletes) and costs one reservation cycle. Not changed
   here: no ceiling, no part of the driver's preflight.
