# V4 worst-case timeline and compatibility of the retained limits (CPU-only analysis, 2026-11-19)

All times in seconds from the creation of the reservation pod (`reservation.json` `creation_timestamp`) unless stated.
**M** measured (V2 / V3), **E** predicted (`feasibility/E1-SCALING-MODEL.md`; range, then stress case), **L** a limit with the
mechanism that enforces it. Nothing here changes a limit, the controller or the 60 s server grace.

## 1. One timeline

| # | step | nominal | worst case allowed | enforced by |
|---|---|---|---|---|
| 1 | reservation acquisition: pod created → Running → operator pre-launch checks → driver and its preflight → launcher start | ≈ 320 s (M, V3: 236 s operator checks + 80 s driver preflight) | **launcher start ≤ 600 s** | L: the driver refuses to start later (`e1_reservation_guard.py age`, exit 7); the operator then deletes the reservation |
| 2 | GPU acquisition and model startup (launcher start → measurement begins, warm-up included) | 1487 – 1640 s (M: V2 1487, V3 ≈ 1523; 1640 kept as upper bound) | inside the GPU-held ceiling; operator abort condition at 30 min | monitor (operator); supervisor at the ceiling |
| 3 | measurement: 8 waves, 224 requests | ≈ 300 – 313 s (E: 3 × 80 s + 59 s; trace span 312.7 s) | inside the GPU-held ceiling; operator abort condition "no stop request 40 min after the launch" | monitor (operator); supervisor at the ceiling |
| 4 | profiler stop request | accepted in 0.003 s (M) | **≤ 30 s** | L: gate G4 |
| 5 | flush | 203 – 224 s (249) (E) | **≤ 300 s** | L: gate G4 |
| 5 | import | 97 – 106 s (118) (E) | **≤ 300 s** | L: gate G4 |
| 5 | stop → report | 301 – 331 s (367) (E) | **≤ 600 s** | L: controller hard deadline (releases the GPUs at 600 s, importer untouched); gates G3, G4 |
| 6 | GPU release after the server SIGTERM | ≤ 2.9 s (M, normal stop); ≤ 1.0 s (M, forced) | **≤ 90 s** | L: gate G6; the controller's SIGKILL after the unchanged 60 s grace |
| 6 | **GPU-held, from the launcher start** | 2092 – 2275 s (2311) (E) | **≤ 2700 s** (+ ≤ 90 s for the release) | L: supervisor before the stop request (registry cleanup), controller after it; gate G10 |
| 7 | launcher's export and parse after the release (CPU only) | 1220 – 1226 s (1360) (E) | until the reservation guard | L: supervisor reservation guard |
| 8 | driver tail (observers stop, census) and closeout | ≈ 360 s (M, V3, with operator latency) | part of the closeout budget | operator; gate G12 |
| 9 | reservation deletion | — | **≤ 5400 s** | L: supervisor guard at 4680 s (run ended or cleaned up), then 720 s operator budget; gate G10 |
| — | post-run (durable copy, identity, checkers, offline recovery, gates) and archive | ≈ 8 min + ≈ 10 min (E, twice V3's 3.7 and 5 min) | after the deletion, CPU only | not under the reservation |

## 2. The three durations asked for

| duration | nominal | bound |
|---|---|---|
| **GPU-held** (launcher start → GPUs released) | 2092 – 2311 s, 35 – 39 min (E) | 2700 s ceiling + 90 s release = 2790 s |
| **Reservation occupancy** (pod created → deleted) | 4490 – 4850 s, 75 – 81 min with V3's operator timings (E); 72 – 78 min if the deletion follows the closeout within 5 min | 5400 s |
| **CPU-only post-processing** | under the reservation: export and parse 1220 – 1360 s plus tail ≈ 120 s. After the deletion: post-run ≈ 8 min (offline import 112 – 124 s, offline export 260 – 288 s, checkers), archive ≈ 10 min (E) | under the reservation: ends at the guard (4680 s) at the latest. After it: no reservation, no GPU; post-run samplers run up to 7200 s |

Reservation budget: 600 (latest launch) + 4080 (run) + 720 (closeout and deletion) = 5400 s.

## 3. If the stop request never occurs

The launcher sends the stop request when the workload ends. If it never does (startup hangs, a request never returns):

1. 30 min after the launch: monitor line `ABORT-CONDITION startup exceeds 30 min`; 40 min: `no stop request`. These are operator
   actions (verified registry cleanup), not automatic.
2. 2700 s after the launch: the supervisor's cause `gpu_held_ceiling_before_stop`. The controller is not yet watching (it
   evaluates its ceiling only after a stop request), so the response is the pinned registry cleanup: SIGTERM to the registered
   tree, 30 s grace, SIGKILL. GPUs free about 45 s later (bound observed in the mock rehearsal). No stream exists (the capture
   was never stopped), so the trace is lost; the run is a FAIL; closeout is NOT CLEAN; the operator deletes the reservation.
3. 4680 s after the reservation's creation: the reservation guard, same response, if anything is still running.

Latest GPU release on this path: 600 + 2700 + 45 ≈ 3345 s after the reservation's creation.

## 4. If the supervisor or a monitor fails

| failure | what still acts | consequence |
|---|---|---|
| memory, storage or GPU sampler stops producing samples | supervisor: `monitor_lost` after 60 s → one response (release line through the controller during finalisation, registry cleanup otherwise) | FAIL; stream preserved if the flush had completed |
| import probe stops | nothing aborts (the probe is an observer without a heartbeat rule) | G4 / G5 INCONCLUSIVE; no PASS |
| supervisor stops | after the stop request: the controller still enforces the 600 s hard deadline and the ceiling. Before it: nothing automatic. The monitor prints `ABORT-CONDITION the supervisor has not checked for 30 s` (operator: registry cleanup) | G11 not PASS (record incomplete); operator-dependent before the stop request |
| launcher (controller) dies | the driver reaches its end; the census and the closeout show held GPUs or a non-terminal phase → NOT CLEAN; operator: verified registry cleanup | FAIL; operator-dependent |
| supervisor stops and no operator is watching, before the stop request | nothing | **unbounded GPU hold: residual risk**, as in V3. Not solved here; a tray-side watchdog would need its own design and review |
| laptop offline (no monitor, no kubectl) | supervisor and controller act on the tray as above; the reservation cannot be deleted until the laptop is back | the reservation may exceed 90 min → G10 FAIL; the GPUs are released regardless |

Rehearsed on mock processes: breach → release line through the controller; breach → registry cleanup; reservation guard →
registry cleanup (new). Unit-tested: lost monitor, ceiling before stop, guard boundary. Never yet triggered on a real server.

## 5. Are the retained limits mutually compatible?

| relation | check | result |
|---|---|---|
| flush + import against stop → report | 300 + 300 = 600 | compatible (a run at both phase limits is exactly at the total) |
| forced release against the server grace | 60 s grace + SIGKILL < 90 s | compatible |
| phases at their limits against the GPU-held ceiling | startup + 313 + 600 + 60 ≤ 2700 requires startup ≤ 1727 s; measured 1487 – 1640 s | compatible at every measured startup (margin ≥ 87 s). With the predicted finalisation (≤ 367 s) startup may be up to 1960 s |
| operator limits against the ceiling | 30 min startup + 10 min capture + 600 s = 3000 s > 2700 s | the ceiling binds first; the operator limits are not additional allowances |
| latest launch + ceiling + release against the reservation guard | 600 + 2700 + 90 = 3390 < 4680 | compatible: the guard can only fall into the CPU-only phase |
| guard + closeout budget against the reservation ceiling | 4680 + 720 = 5400 | compatible by construction |
| predicted run against the guard | launch 320 + GPU-held 2311 + export 1360 + tail 120 = 4111 < 4680 | compatible, margin 569 s in the stress case |
| one extreme against the guard | latest launch (600) with a predicted run: 4391; measured launch (320) with a ceiling-long GPU hold (2790): 4590 | compatible, margins 289 s and 90 s |
| **both extremes together** | 600 + 2790 + 1360 + 120 = 4870 > 4680 | **not completable**: the guard would end the export. The run would already be a FAIL (the ceiling acted). The limits stay enforced; what is lost is the in-session export, not the stream |
| sampler lifetimes against the run | memory, storage, supervisor 5400 s from the launch ≥ 4680 s from the creation; GPU sampler 3300 s ≥ 2700 + 90 + 300 | compatible. **The V3 values (3400 s) would not have been**: with a 20-minute export the samplers would have ended before the launcher and the supervisor would have aborted a healthy run. Changed in the V4 driver; listed in the packet |
| storage ceilings among themselves | run root at 18 GB plus its archive (1.105 ×) = 37.9 GB ≤ 38 GB | compatible |
| storage predictions against ceilings | temporary 10.4 – 11.4 ≤ 15; run root 12.6 – 14.0 ≤ 18; with archive 26.5 – 29.5 ≤ 38 | compatible |
| importer ceiling + floor + server against the preflight minimum | 128 + 64 + **285** (server footprint measured in V3) = 477 GiB > 448 GiB | **not compatible at the extremes**: see §6 |

## 6. Findings that need a decision (nothing was changed)

1. **Preflight host memory.** The retained minimum, 448 GiB, was built as 128 (importer ceiling) + 64 (floor) + 256 (server,
   then unmeasured). V3 measured the server's footprint at 285 GiB below idle (295 GiB at the stop request). A host that
   passes the preflight with exactly 448 GiB available and an importer at its 128 GiB ceiling would end at 35 GiB, below the
   64 GiB floor; the sampler and supervisor would abort. With the measured footprint the consistent minimum is 128 + 64 + 295
   = 487 GiB. In practice the tray has 858 GiB available when idle and the predicted minimum is about 530 GiB, so V4 is not
   affected; the inconsistency is between two limits at their extremes. The 448 GiB requirement is retained as instructed; a
   change to 488 GiB is proposed for separate review.
2. **Both reservation extremes together** (§5) cannot be completed inside 90 min under the existing reservation lifecycle. The
   options are: accept it (the run is a FAIL in that corner anyway), tighten the latest launch (for example 450 s), or enable the
   early reservation release after review (`E1-V4-RESERVATION-RELEASE-DESIGN.md`). The draft keeps 600 s and the existing
   lifecycle.
3. **The reservation guard is new enforcement.** In V3 the 90-minute ceiling was only measured afterwards. In V4 the supervisor
   ends a run that is still going 78 min after the reservation's creation. A healthy but slow export could be ended by it.
4. **Operator-dependent paths** remain where §4 says so.
