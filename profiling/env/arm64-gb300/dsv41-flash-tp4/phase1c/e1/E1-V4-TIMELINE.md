# V4 worst-case timeline, reservation budget and compatibility of the limits (CPU-only analysis; revision 2, 2026-11-19)

Revision 2 follows the review decision on `39561325`: preflight host memory 448 → 488 GiB; a launch eligibility rule derived
from the clock definitions; an explicit bound on CPU-only processing before the reservation is deleted. The 90-minute
reservation limit, the 4680 s supervisor guard, every deadline, every other ceiling, the controller and its 60 s server grace
are unchanged. Revision 1 is in the history at `39561325`.

**M** measured (V2 / V3), **E** predicted (`feasibility/E1-SCALING-MODEL.md`; range, then stress case), **L** a limit with the
mechanism that enforces it.

## 1. Clock definitions

| clock | starts | ends | limit | read from | enforced by |
|---|---|---|---|---|---|
| reservation | pod creation (`reservation.json` `creation_timestamp`) | deletion observed (`reservation_deleted.json`) | 5400 s | kubectl records | launch rule + supervisor guard + operator budget; gate G10 |
| launch age | pod creation | the driver's eligibility check (before any observer or the launcher exists) | 470 s (derived, §2) | `run.log` | driver, exit 7 |
| check → launcher start | eligibility check | `lifecycle.json` `launch_started_at` | 20 s allowance (M: 1 s in V3) | records | not enforced; inside the budget |
| GPU-held | `launch_started_at` | GPUs released (`gpu.released_at`; sampler upper edge) | 2700 s | controller and GPU sampler | trigger at 2700 s: supervisor before a stop request (registry cleanup), controller after it (server stop). Gate G10 judges launch → **released** ≤ 2700 s |
| release | server SIGTERM (or cleanup SIGTERM) | GPUs observed free | 90 s | GPU sampler | controller: 60 s grace, then SIGKILL; cleanup: 30 s grace, then SIGKILL; gate G6 |
| CPU-only processing under the reservation | GPUs released | the driver's stop file (launcher ended, first census, 20 s) | 1400 s budget (E stress 1360 s export and parse + 40 s tail; M tail 21 s) | records | the reservation guard: age 4680 s is the absolute end |
| guard | pod creation | — | 4680 s | supervisor record | supervisor: one response |
| closeout and deletion | guard (latest end of the run) | deletion observed | 720 s | records | operator, with the deadline printed by closeout and monitor; gate G10 |

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
before it starts any observer, the supervisor or the launcher. Refusal: exit 7, `LAUNCH REFUSED` in `run.log`, nothing started,
no GPU touched; the operator deletes the reservation. The values are constants of the driver, asserted against
`E1-V4-SCALE-INPUTS.json` by the preflight; only rehearsal mode can override them.

| check | arithmetic | result |
|---|---|---|
| required worst-case path with an immediate launch | 0 + 4210 = 4210 ≤ 4680 | fits, 470 s to spare |
| launch window | 0 ≤ age ≤ 470 s | exists |
| measured launch age | V3: 316 s (operator 236 s + driver preflight 78 s + 1 s). V4 driver preflight in rehearsal: 110 – 116 s → about 350 s at V3's operator speed | inside the window, margin about 120 s; a slower operator is refused, not launched |
| latest launch, ceiling and release against the guard | 470 + 20 + 2700 + 90 = 3280 < 4680 | the guard can only fall into the CPU-only phase |
| guard + closeout budget against the reservation limit | 4680 + 720 = 5400 | equal by construction |
| predicted run at the latest launch | 470 + 20 + 2311 (E stress GPU-held) + 1360 + 40 = 4201 | ends 479 s before the guard |
| **all extremes together** (the corner of revision 1) | 470 + 20 + 2700 + 90 + 1400 = 4680 | **ends at the guard, not after it** |

The corner that revision 1 reported as not completable (latest launch + ceiling-long GPU hold + stress export) now fits: the
launch window was reduced from 600 s to 470 s instead of any ceiling being changed.

## 3. One timeline

| # | step | nominal | worst case allowed | enforced by |
|---|---|---|---|---|
| 1 | reservation acquisition → operator checks → driver preflight → eligibility check | ≈ 350 s (M V3 operator 236 s; M V4 rehearsal preflight 110 – 116 s) | **age ≤ 470 s** | L: launch rule, exit 7 |
| 2 | launcher start; GPU acquisition, model startup, warm-up | 1487 – 1640 s (M: V2 1487, V3 ≈ 1523) | inside the GPU-held ceiling; operator abort condition at 30 min | monitor; supervisor at the ceiling |
| 3 | measurement: 8 waves, 224 requests | ≈ 313 s (E) | inside the ceiling | monitor; supervisor at the ceiling |
| 4 | stop request accepted | 0.003 s (M) | ≤ 30 s | L: G4 |
| 5 | flush / import / stop → report | 203 – 224 (249) / 97 – 106 (118) / 301 – 331 (367) s (E) | ≤ 300 / 300 / 600 s | L: controller hard deadline; G3, G4 |
| 6 | GPU release | ≤ 2.9 s (M) | ≤ 90 s | L: controller grace + SIGKILL; G6 |
| 6 | GPU-held, launcher start → released | 2092 – 2275 s (2311) (E) | 2700 s for a PASS; 2700 + 90 s enforced | L: supervisor / controller; G10 |
| 7 | export and parse after the release, tail to the stop file (CPU only) | 1220 – 1226 s (1360) + 21 s (E, M) | 1400 s in the budget; absolute end at age 4680 s | L: reservation guard |
| 8 | observers drain, driver end, closeout | 12 s + closeout (M V3) | inside the 720 s | operator; G12 |
| 9 | reservation deletion observed | 62 s after the request (M V3) | **age ≤ 5400 s** | operator with the printed deadline; G10 |
| — | post-run and archive | ≈ 8 + ≈ 10 min (E) | after the deletion; CPU only | not under the reservation |

Durations: GPU-held 35 – 39 min predicted (bound 2700 s, enforced 2790 s). Reservation occupancy about 70 – 80 min predicted
with V3's operator timings (bound 90 min). CPU-only processing under the reservation: 21 – 23 min predicted (budget 1400 s,
absolute end at the guard). CPU-only post-processing after the deletion: about 18 min.

Closeout and deletion budget (720 s): supervisor cleanup response ≤ 45 s (only if the guard acted) + observers drain and driver
end 60 s (M 12 s) + closeout script 120 s + deletion request → observed 120 s (M 62 s) = 345 s, leaving 375 s of operator
slack. In V3 the operator took 832 s from the stop file to the observed deletion, without a deadline (closeout 5.6 min after
the stop file, deletion requested 7.2 min after the closeout). That pace would not fit 720 s if the run ended at the guard. V4
prints the deletion deadline in the closeout and in every monitor pass. Nothing automatic deletes the reservation.

## 4. If the stop request never occurs

1. 30 min after the launch: monitor `ABORT-CONDITION startup exceeds 30 min`; 40 min: `no stop request` (operator actions).
2. 2700 s after the launch: supervisor cause `gpu_held_ceiling_before_stop` → pinned registry cleanup (SIGTERM, 30 s grace,
   SIGKILL). GPUs free within 90 s. No stream exists; the run is a FAIL; closeout NOT CLEAN; the operator deletes the reservation.
3. Latest GPU release on this path: 470 + 20 + 2700 + 90 = 3280 s after the pod's creation; 2120 s remain for the deletion.

## 5. If the supervisor or a monitor fails

| failure | what still acts | consequence |
|---|---|---|
| memory, storage or GPU sampler stops | supervisor: `monitor_lost` after 60 s → one response | FAIL; stream preserved if the flush had completed |
| import probe stops | nothing aborts | G4 / G5 INCONCLUSIVE |
| supervisor stops | after the stop request the controller still enforces the 600 s deadline and the ceiling. Before it, and for the reservation guard: nothing automatic. Monitor: `ABORT-CONDITION the supervisor has not checked for 30 s`, and at 78 min `the reservation is older than 78 min and the run has not ended` | G11 INCONCLUSIVE; operator-dependent |
| supervisor already used its one response through the controller (a breach during the import), and the run is still going at the guard | the guard is recorded as a later cause; **no second response** (frozen supervisor semantics) | operator-dependent for the reservation; the GPUs were released by the first response |
| launcher dies | driver ends; census and closeout NOT CLEAN | FAIL; operator: verified registry cleanup |
| closeout or deletion delayed past 5400 s | nothing automatic; closeout and monitor print the deadline and `ABORT-CONDITION` after it | G10 FAIL; the GPUs are idle but still allocated to the pod |
| laptop offline | supervisor and controller act on the tray; the reservation cannot be deleted | G10 FAIL possible; GPUs released regardless |

## 6. Compatibility of the limits

| relation | check | result |
|---|---|---|
| flush + import against stop → report | 300 + 300 = 600 | compatible |
| forced release against the server grace | 60 s + SIGKILL < 90 s | compatible |
| phases at their limits against the GPU-held ceiling | startup + 313 + 600 + 60 ≤ 2700 → startup ≤ 1727 s; measured 1487 – 1640 s | compatible (margin ≥ 87 s) |
| launch rule, ceiling, release, CPU budget against the guard | §2 | compatible, including all extremes together |
| guard + closeout against the reservation | 4680 + 720 = 5400 | compatible; operator-dependent |
| sampler lifetimes | memory sampler, storage sampler and supervisor run 5400 s from their start (shortly before the launcher), so past the guard at every permitted launch age; GPU sampler 3300 s ≥ 2700 + 90 + 300 | compatible |
| storage | 18 GB + archive (1.105 ×) = 37.9 ≤ 38 GB; predictions 10.4 – 11.4 / 12.6 – 14.0 / 26.5 – 29.5 GB | compatible |
| importer ceiling + floor + run footprint against the preflight minimum | 128 + 64 + 295.2 (M, V3 at the stop request) = 487.2 ≤ 488 | compatible with the V3 measurement, margin 0.8 GiB; see §7.1 |

## 7. Remaining findings

1. **The 488 GiB minimum is consistent with what V3 measured, by 0.8 GiB, and V4 runs twice as long.** In V3 the footprint grew
   9.3 GiB between warm-up and the stop request. Whether it keeps growing over V4's four additional waves was not measured; at
   the same mean rate it would be about 5 GiB more, and a host at exactly 488 GiB with an importer at exactly 128 GiB would
   then reach about 60 GiB, below the floor (the sampler and supervisor would abort; a safe outcome, but a FAIL). In practice
   the tray has about 860 GiB available when idle and the predicted minimum is about 530 GiB. Details:
   `feasibility/E1-V4-MEMORY-BUDGET.md`. The value is 488 as decided; no further change was made.
2. **One response.** After a response through the controller, the supervisor does not act on the reservation guard (§5).
3. **The launch window is 470 s** and the expected launch age is about 350 s. A refusal is safe and costs one reservation
   cycle; it is more likely than under revision 1.
4. **The closeout budget depends on the operator** (§3). V3's unhurried pace would exceed it in the worst case.
5. Before a stop request, a failed supervisor is covered only by the operator's monitor (unchanged from V3).
