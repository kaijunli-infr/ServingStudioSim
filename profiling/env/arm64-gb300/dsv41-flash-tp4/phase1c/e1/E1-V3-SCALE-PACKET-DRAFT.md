# E1 V3 draft packet — intermediate trace size (≈ 25 M kernel rows), normal stop (DRAFT; authorizes nothing)

**Status: DRAFT, 2026-11-18.** Prospective protocol for review. Nothing here is pinned or frozen; no GPU step is authorized. A
freeze needs the items of §13 done CPU-only under their own GO, and a launch needs a further GO naming a commit. V2's frozen packet,
gates and PASS are not touched. Inputs of this draft: `make_e1_inputs_v3scale.py` → `traces/e1_v3_scale.csv`,
`configs_v3scale/e1_v3_scale/profile.json`, `E1-V3-SCALE-INPUTS.draft.json`. (`configs_v3/`, `SCRIPTS_v3.sha256` and
`make_e1_inputs_v3.py` belong to the lost e1_main_v3 capture; this draft uses the suffix `v3scale` to keep them apart.)

## 1. Question

V1 and V2 exercised the lifecycle controller at 12.25 M kernel rows. The lost capture had about 216 M. V3 asks, at about twice the
V1/V2 size and with a **normal stop**: how long do the flush and the import each take, how much memory and storage do they need, and
do stream preservation, offline recovery and the release measurements hold at that size? V3 is a calibration point with gates, not
a forced-release test and not a step of the E1 measurement itself.

## 2. Workload and identity

| item | value |
|---|---|
| pass / tray | `e1_v3_scale`, `pod4-gb300-10-tray16`, 4 GPUs, one launch |
| config | the V2 config (`configs_fr/e1_v2_forced_release/profile.json`, sha256 7f7e4f66…) with only `name`, `log_dir` and the trace path changed (diff checked at generation); draft sha256 fc91bddbf0fd6b34… |
| trace | `traces/e1_v3_scale.csv` (draft sha256 d9d2dc77b316b786…): the 56 accepted rows of `e1_dry.csv` twice. Waves: b=8 at 0 s, b=48 at 18.627 s, b=8 at 80 s, b=48 at 98.627 s; 112 requests, 32 768 input and 1024 output tokens each. The second pair starts 29 s after V2's measured end of the first (51.2 s) |
| why this trace | every wave is a cell already accepted twice (V1, V2), so iterations and rows are known per wave and the existing checkers apply; only the trace differs from V2 |
| source | to be pinned at freeze: `exp/profiler-lifecycle` commit, trees `alignment` 7b4df784…, `launcher` 5acab289… (unchanged since e32fbab7), submodules vLLM fork 0ab18711 and req-frontend 5c20a259 |
| load-generator executable | `session_runner` sha256 `7f6eb81d45e6b98f3b6ebb56ab20b60ce8aaf4584aa3d4f25e3449b2d05a00c2`, prebuilt contract and build-tool guard as in V2 (`BINARY-PROVENANCE.json`) |
| profiler | Nsight CLI 2025.6.3.541 (sha256 2d53d710…), `QdstrmImporter` (sha256 b30312a6…), identities re-pinned on tray16 at freeze |

The size is 24.50 M rows, 4.3 % below the 25.6 M of the earlier proposal. That proposal named the Phase 0 `quadrant_nsys` recipe
(25.6 M rows); it uses a different load-generator backend, saturated arrivals and no routing hook, so it would change the recipe and
the checkers, not only the size. If 25.6 M matters more than recipe identity, the alternative is four b=48 waves (25.45 M rows
expected); this draft recommends the doubled accepted trace.

## 3. Expected values

Basis: twice the V2 measurement of the same 56 rows. These are expectations with bands, not a model.

| quantity | expected | band / note |
|---|---|---|
| iterations | 4320 | 4200 – 4500 |
| NVTX forward ranges | 17 280 | 4 per iteration |
| kernel rows | 24 501 664 | ± 5 %: 23 276 580 – 25 726 747; four ranks |
| request timings / routing files | 112 / 112 | exact |
| replay duration | ≈ 131 s | 80 s offset + 51 s |
| raw stream | ≈ 0.70 GB | V2: 348 288 111 B (28.4 B/row) |
| report (`.nsys-rep`) | ≈ 0.78 GB | V2: 0.39 GB; Phase 0 at 25.6 M rows: 0.70 GB |
| in-session export (`.sqlite`) | ≈ 3.9 GB | V2: 1.96 GB; Phase 0 at 25.6 M: 3.82 GB |
| run directory | ≈ 6.0 GB | V2: 3.0 GB |
| importer peak memory | unknown | measured 8.7 GB at 12.25 M rows (offline, today); 61 GB was sampled at ≥ 49 % of the 216 M-row import. Planning bound 64 GB; tray16 has 1004 GB, 904 GB available |

## 4. Stop, flush and import: predictions and deadlines

V2's 27–29 s import is **not** scaled. The nearest measurements are the two Phase 0 captures of 25.6 M rows: stop → report 160 s and
163 s, same Nsight version and capture flags, different workload recipe. Other points: 45 s at 5.9 M, 63 s at 6.4 M, 90.8 s at
12.25 M (V1 and V2), 567 s at 97 M. The split between flush and import has been recorded only at 12.25 M (flush 61.9–63.8 s,
import 27.0–28.9 s). At 24.5 M the split is unknown; the flush may contain a fixed part (unverified).

| phase | definition (source) | prediction | deadline | on exceeding it |
|---|---|---|---|---|
| stop accepted | stop request sent → accepted (controller) | < 1 s | 30 s | FAIL |
| flush | stop requested → phase `importing` (controller) and → importer first seen (import probe) | 60 – 140 s | 300 s | soft escalation recorded; FAIL of the phase gate; run continues |
| import | importer first seen → importer gone with a stable report (import probe) | 27 – 110 s | 300 s | FAIL of the phase gate; run continues to the hard deadline |
| stop → report written | controller | 140 – 190 s | **600 s hard** (controller, `release_gpus`) | the controller releases the GPUs; importer untouched; V3 FAIL; trace recovered offline |
| session close after the server stop | controller | < 5 s | 300 s | FAIL |

The predictions are recorded before the run and are not gates. The measured phase durations are the result of V3.

## 5. Independent measurements

| measurement | instrument | rule |
|---|---|---|
| GPU release | `checks/e1_gpu_watch.py`, registered, 2 s interval, stamp before the queries | reported as an interval: last "held" reading window and first "released" reading window (stamp … next stamp − interval), relative to the controller's recorded server SIGTERM. Never as a single figure tighter than the upper edge. Deadline: upper edge ≤ 90 s |
| import state | `checks/e1_import_probe.py`, registered, observer mode (no trigger), 0.5 s interval | importer identified (exactly one, pinned executable, session files); ≥ 2 consecutive intervals of its own CPU/I-O advance; first-seen and gone times; final state `COMPLETED`. Reported progress and worker CPU time are never evidence |
| importer memory | new sampler (§13): `VmRSS`/`VmHWM` of the identified importer, 1 s interval | peak recorded; abort condition at 256 GB |
| storage | new sampler (§13): `du` of the run directory, the stream directory in `/tmp` and every post-run scratch directory, 5 s interval, running through the post-run | peaks recorded; **an unmeasured peak is INCONCLUSIVE for the storage gate** |
| host identity, process census, registry | as V2 | as V2 |

## 6. Stream preservation and recovery

As V2 §6: the controller hard-links the stream when it first sees it and records its final size and SHA-256 after the session
closes; the post-run copies it to durable storage, verifies the copy against the tray original with the complete SHA-256, and only
then imports it offline. The offline import runs under the import probe and the storage and memory samplers. The recovered trace is
checked absolutely (iterations, NVTX ranges, kernel rows, four ranks, capture boundaries) and against the in-session export (the
ten equality checks of FR8b). No script removes a stream; the V1 and V2 originals in `/tmp` stay.

## 7. Importer isolation and server termination

- The importer is a child of the detached Nsight agent, outside the CLI's process tree and group and outside the driver's registered
  tree (probe design §1). Nothing in the controller, the driver or the registry cleanup can reach it by descent, and nothing may
  signal it by pattern.
- **No signal to the Nsight CLI, the agent or the importer** from the controller, the driver or the operator while a report can
  still be produced. Timeouts never escalate to the importer.
- The import probe writes the importer's and the agent's identities (pid, start time, executable, arguments) to its record. The
  closeout adds two checks by executable path and identity, not by name pattern: no importer of this session and no agent of this
  session's CLI alive.
- Only if an importer is still running after the session-close timeout **and** the durable stream copy is verified: the operator
  may end that one identity-verified process with SIGTERM, recorded, V3 then FAIL. Not automatic, never before the durable copy.
- Server termination, normal path: after the stop request has returned, the launcher's ordinary shutdown (SIGTERM to the
  identity-verified API server; in V1 it exited in about one second).
- Server termination, forced (hard deadline or GPU-held ceiling): the frozen `release_gpus` action. Known behaviour from V2: the
  engine is killed at once and the GPUs are free within about 2 s, but the API server waits on the open stop request until the
  identity-verified SIGKILL after 60 s. That is expected and is not by itself a failure; the GPU-release deadline is judged on the
  sampler.

## 8. Resource budgets and hard ceilings

| resource | nominal | hard ceiling | enforced by |
|---|---|---|---|
| reservation | ≈ 50 min | 90 min (pod kept beyond it only while GPUs are not idle; reported as a violation) | operator |
| GPU-held | ≈ 1800 s (startup ≈ 1490 + replay 131 + finalisation ≈ 170) | 2700 s from the launch, any phase, not extendable | controller |
| GPU release after the server SIGTERM | ≈ 2 s | 90 s, upper edge of the sampled interval | gate |
| stop → report | 140 – 190 s | 600 s | controller |
| temporary storage on tray16 `/tmp` | stream 0.7 + retained V1/V2 streams 0.67 + post-run scratch ≈ 4.7 = ≈ 6.1 GB | 10 GB, measured peak | storage sampler, monitor |
| persistent run storage | ≈ 6.0 GB | 8 GB | storage sampler, post-run |
| persistent including archive | ≈ 12.3 GB | 16 GB | closeout |
| host memory | importer unknown (planning 64 GB) | preflight needs ≥ 128 GB available; abort condition importer > 256 GB | preflight, memory sampler |

No operator extension overrides a hard ceiling. `finalize_max_extension_s` stays 120 s and is not planned to be used.

## 9. Gates (prospective; verdict PASS only if every gate passes)

| gate | PASS | FAIL | INCONCLUSIVE |
|---|---|---|---|
| G0 source and executable | launch commit has the pinned trees; executable = pinned sha256 before and after; prebuilt mode; guard armed, nothing blocked | any mismatch, any compile line, any blocked build request | a record missing |
| G1 host identity | lifecycle = run log = reservation node = every sampler host | conflict | a source missing |
| G2 one stop request | one sent, accepted, **returned OK** before the server shutdown | duplicate, or an error return | record missing |
| G3 phases in order | stop_requested → stop_accepted → capture_ended → stream_written → importing → report_written, no escalation, no control line | hard deadline or ceiling fired; any operator control line; `stop_failed` | — |
| G4 phase deadlines | each phase of §4 within its deadline | any exceeded | a phase boundary not observed |
| G5 importer observed | exactly one importer identified; ≥ 2 consecutive importer-specific intervals; final `COMPLETED`; first-seen/gone times recorded | `TERMINATED`; a signal to CLI, agent or importer | `NOT_FOUND`, `AMBIGUOUS`, or activity never established |
| G6 GPU release | upper edge of the sampled interval ≤ 90 s after the recorded SIGTERM; no "held" after the first "released"; census clean | later, or held again | sampler gap > 10 s across the release, or SIGTERM time missing |
| G7 stream durable | final record, durable copy and tray original equal by complete SHA-256 | any difference; stream missing | — |
| G8 offline recovery | offline import rc 0; absolute checks in band; ten equality checks true | any false | tool unavailable |
| G9 scientific completeness | completeness checker per wave (4 waves), extract, Nsight validation, config identity: all PASS | any FAIL | checker cannot be applied (§13) |
| G10 bounds | GPU-held ≤ 2700 s; session close ≤ 300 s; reservation ≤ 90 min | any exceeded | — |
| G11 storage and memory | measured peaks within §8 | a peak above its ceiling | **a peak not measured** |
| G12 closeout | GPUs idle, registry clean, no experiment process, no importer or agent of this session, reservation deleted | anything left | — |

A FAIL or INCONCLUSIVE is final for the launch: no retry, no reinterpretation after the results are seen.

## 10. Procedure (only after a freeze and a GO naming the commit)

Reservation pod and `reservation.json`; negative control; driver-context read-only preflight, all PASS; one launch through a
neutral-path wrapper; monitor every 60 s; closeout CLEAN; delete the reservation with GPUs idle; post-run on tray16 under the
samplers; archive with checksums; report; stop. As V2 §10, with the V3 driver, monitor, closeout and post-run of §13.

## 11. Abort procedure

1. Before the launch: any FAIL in the negative control or preflight, a missing or foreign reservation record, a non-pinned
   executable, a trace/config guard failure: nothing starts.
2. During the run (monitor prints only): startup over 30 min; no stop request 40 min after the launch; engine-error lines before
   shutdown; a GPU process outside the driver's tree; run directory over 8 GB; `/tmp` use over 10 GB or under 50 GB free; importer
   memory over 256 GB; a controller escalation.
3. Operator action: only the pinned registry cleanup (`e1_pids.py cleanup --execute --grace 30`), which signals registered
   processes whose identity (pid, start time, boot id, uid) still matches. It cannot reach the agent or the importer, and they are
   not to be signalled (§7). Nothing is killed by pattern. Uncertain ownership: stop and report.
4. If GPUs are still held 90 s after a release request: G6 has failed; registry cleanup; the reservation pod stays until the GPUs
   read idle.
5. An abort is a FAIL: no retry. The stream is never removed; closeout and post-run still run for the record.

## 12. Not authorized

Any GPU run including V3, a V2 retry, the full E1 retry, the identification partition, fitting, E2–E6, Rust production integration,
profiling database writes, preset changes.

## 13. Open before a freeze (CPU-only work, each needs to be reviewed)

| item | state |
|---|---|
| import probe | written and tested (23 tests, real importer offline); not yet wired into a driver; in-session argument pattern unverified |
| storage sampler, importer memory sampler | not written |
| V3 gate checker with fixtures, including the interval rule of G6 and the unmeasured-peak rule of G11 | not written |
| V3 driver, preflight, monitor, closeout (with the agent/importer checks), post-run | not written; to be derived from the V2 scripts |
| completeness checker on a four-wave trace (wave index 1) | not verified; if it cannot be applied, G9 must be redefined before the freeze, not after the run |
| pins, script manifest, freeze manifest, CPU rehearsal on the V2 evidence, read-only preflight | not done |
| decision: doubled accepted trace (24.5 M rows) or four b=48 waves (25.45 M) | reviewer |
| decision: keep the 60 s server grace as is (analysis §5) | reviewer |
