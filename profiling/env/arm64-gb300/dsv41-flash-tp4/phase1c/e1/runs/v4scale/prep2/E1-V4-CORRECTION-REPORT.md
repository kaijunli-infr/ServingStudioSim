# V4 resource-budget and freeze-readiness correction (CPU-only), 2026-11-19

Review decision on `39561325`: V4 preparation accepted with two required corrections (memory, reservation budget). This report
covers the correction. **No GPU was used, nothing was launched, no reservation pod was created, nothing is frozen.** V4 GPU
execution and full E1 remain NO GO.

| commit | content |
|---|---|
| `890ca1b3` | correction code, tests, fixtures, regenerated inputs, timeline revision 2, memory budget analysis |
| `d04e99d3` | PROPOSED packet revision 2, pins, script / input / freeze manifests (proposed) |
| this commit | evidence (`runs/v4scale/prep2/`) and this report |

Prior records are not modified: the V1, V2 and V3 manifests verify; `runs/v4scale/prep/` (revision 1 evidence) is unchanged.

## A. Memory

| item | result |
|---|---|
| preflight minimum | 448 → **488 GiB** = importer ceiling 128 + runtime floor 64 + 296 for the run; registered in `E1-V4-SCALE-INPUTS.json` (`ceilings`, `memory_budget`); the preflight asserts the sum |
| check | `checks/e1_preflight_memory.py`: exact in kB (488 GiB = 511 705 088 kB), no truncation to whole GiB; cgroup limit against the same value |
| runtime floor 64 GiB, importer ceiling 128 GiB, breach responses | unchanged: same sampler arguments in the driver, same supervisor, same gates (test `test_registered_values_and_what_did_not_change`) |
| boundary fixtures | `tests/fixtures/e1_v4_scale/meminfo_*.txt`: 487 GiB refused; 488 GiB − 1 kB refused; 488 GiB accepted; 488 GiB + 1 kB accepted; 489 GiB accepted. Run by the tests and by the preflight itself |
| on the tray | MemAvailable 860.72 GiB, margin 372.72 GiB; no cgroup limit |

Verification against V3 (`feasibility/E1-V4-MEMORY-BUDGET.md`, from the preserved 1 s memory record):

| V3 measurement | GiB |
|---|---|
| idle before the launch | 858.0 |
| steady after load | 572.1 – 573.2 (285.9 below idle) |
| at the stop request | 562.8 (**295.2 below idle**: what the run holds when the importer starts) |
| minimum during model load (no importer exists then) | 552.0 (306.0 below idle) |
| minimum of the run (during the import; importer peak 15.6) | 547.2 |

488 − 295.2 − 128 = 64.8 ≥ 64: consistent with the V3 measurement, margin 0.8 GiB. 448 gave 24.8.

Unmeasured or uncontrolled concurrent consumers:

1. **Growth over V4's four additional waves.** V3's footprint grew 9.3 GiB between warm-up and the stop request. V4 captures
   160 s longer. At the same mean rate that is about 5 GiB more, which would exceed the 0.8 GiB margin: at the exact extremes
   (488 GiB at the preflight, importer at 128 GiB) MemAvailable would reach about 60 GiB and the supervisor would abort. This
   is a projection from one run, not a measurement. The value was set to 488 as decided and not raised further.
2. The importer at V4's size: predicted 28 – 35 GiB, bounded by the 128 GiB ceiling used in the budget.
3. Other tenants of the node (DaemonSet pods, other sessions of the shared account, CPU pods the scheduler may place): not
   controlled; the preflight is one sample and the runtime floor is the protection.
4. Not concurrent with the server: launcher export and parse, post-run import and export (after the GPU release).
5. `/tmp` on the tray is ext4 on NVMe, not memory-backed: the 15 GB of temporary storage does not count against memory.

## B. Reservation budget (`E1-V4-TIMELINE.md` revision 2)

- Unchanged: 90 min limit, 4680 s supervisor guard, 720 s closeout and deletion budget.
- Launch rule: **eligible iff 4680 − age ≥ 4210**, with 4210 = 20 (check → launcher start; 1 s measured) + 2700 (GPU-held
  ceiling) + 90 (release after the ceiling acts) + 1400 (CPU-only processing after the release: stress prediction 1360 s
  export and parse + 40 s tail; 21 s measured). So age ≤ 470 s.
- Release time is counted once. Gate G10 measures GPU-held up to the release, so a passing run has it inside the 2700 s; the
  enforcement is a trigger at 2700 s followed by a release of up to 90 s, and the budget uses that enforced bound.
- CPU-only processing before the deletion is bounded explicitly: 1400 s in the budget, and absolutely by the guard at 4680 s.
- Immediate launch: 0 + 4210 ≤ 4680. A launch window exists (0 – 470 s). V3 launched at 316 s; with V4's longer driver
  preflight (110 – 116 s in rehearsal) the expected age is about 350 s.
- All extremes together (latest launch, ceiling-long GPU hold, release at its limit, CPU-only at its budget) end at 4680 s.
  The corner reported in revision 1 is closed by narrowing the window from 600 s to 470 s; no ceiling was modified.
- Refusal before GPU startup: the driver evaluates the rule after its preflight and before it starts any observer, the
  supervisor or the launcher; exit 7, `LAUNCH REFUSED` in `run.log`. Real mode cannot override the values (test).

## C. Reservation guard: tests on the actual supervisor and driver

Unit level: `tests/test_e1_v4scale_reservation_scenarios.py` runs the actual supervisor (`e1_supervisor_v4.py` around the
unchanged `e1_supervisor.py`) with its own pinned cleanup (`e1_pids.py cleanup --execute --kill9`) on real registered stand-in
processes, plus an unregistered stand-in for the importer. `tests/test_e1_reservation_guard.py` runs the driver's own
constants line and launch-rule lines. End-to-end: the actual driver, closeout, post-run and archive scripts around mock
processes (`rehearse_e2e_v4scale.sh`), expectations registered in `e1_v4scale_e2e_check.py` before the run.

| case | unit test (actual code, fake clock) | end-to-end scenario (attempt 1) |
|---|---|---|
| launch at the latest permitted instant | age 470.000 s permitted; driver lines at 462 s permitted | `launch_late_permitted`: age 441.4 s at the check, permitted with 28.6 s to spare, run completed; 17 / 17 |
| launch immediately beyond it | age 470.001 s refused; driver lines at 473 s: exit 7, nothing after the rule runs | `launch_refused`: age 500.6 s, exit 7; registry holds the driver only; no observer, supervisor, lifecycle record or launcher line; 4 / 4 |
| GPU-held ceiling reached | 2700 s exactly: no cause; past it without a stop request: registry cleanup; after a stop request: the controller's | `ceiling_before_stop` (rehearsal ceiling 45 s): cause at 45.2 s, cleanup in 0.44 s; 16 / 16 |
| normal completion near the guard | run ends 1 s before, and exactly at, the guard: no response | `complete_near_guard`: ended 396.0 s after creation, rehearsal guard 420 s, no response, gates as registered; 17 / 17 |
| export still running near the guard | guard passes in the CPU-only phase: registry cleanup; importer stand-in alive; stream bytes unchanged | `export_near_guard`: guard at 420.7 s during the export, after the GPU release; importer had completed and was not signalled; preserved stream = controller's final record; 18 / 18 |
| sampler failure | storage sampler silent > 60 s: `monitor_lost`, registry cleanup; also together with the guard | `sampler_lost`: sampler ended through a verified one-record registry; cause after 60.2 s; 16 / 16 |
| supervisor failure | crash before the guard: record incomplete, nothing ended; monitor lines present | `supervisor_lost`: run completed under the controller; G11 INCONCLUSIVE; 12 / 12 |
| delayed closeout or deletion | deadline at 5399 / 5400 / 5401 s; budget arithmetic | fixtures: exactly 90.0 min PASS; 90 min + 1 s FAIL (new); 91 min FAIL; record missing INCONCLUSIVE |
| reservation and storage breach together | same sample: both causes, one cleanup | `guard_and_storage`: both causes present at the decision, one cleanup; 16 / 16 |
| reservation and memory breach together | same sample in the CPU-only phase: both causes, one cleanup. Breach during the import, guard later: one response through the controller, guard recorded as a later cause | covered at unit level |

In every abort outcome the only action is the pinned cleanup of the registered tree, each signal preceded by the identity
check (pid, start time, boot id, uid); the supervisor sends no signal; the importer or its stand-in is not in the registry and
is not signalled; the preserved stream is unchanged. The frozen supervisor and `e1_pids.py` are byte-identical to the V3
manifest (test).

## D. Unchanged experiment design

Trace `63d79100…` and config `4203d307…` are byte-identical to revision 1 (224 requests, eight waves, four copies); expected
8640 iterations and 49 003 328 kernel rows; controller code and policy, 60 s server grace; deadlines 300 / 300 / 600 s; GPU-held
2700 s; storage 15 / 18 / 38 GB; identity policy rules and the scientific completeness requirements; gates G0 – G15. The
identity policy file changes in one value, the pinned hash of `E1-V4-SCALE-INPUTS.json` (the inputs file carries the new
budget); its version stays `v4scale-1`. Early reservation deletion is not enabled and no V4 script calls it (test).

## E. Validation (all at `d04e99d3`, the proposed packet)

| check | result |
|---|---|
| read-only preflight on tray16 | **78 PASS, 0 FAIL** |
| actual V4 configuration through the complete prospective identity checker | identity rehearsal 13 / 13: V4 policy PASS with 20 / 20 trace checks; V1/V2 and V3 policies FAIL on the same records; 7 negative controls FAIL |
| complete V4 post-run on the V4-shaped run directory | verdict FAIL with exactly G8b, G9, G15 not passing, as registered (the capture data are V3's four waves); G13, G14 PASS |
| synthetic fixtures | 118 / 118 (3 positive, 6 boundary, 109 negative) |
| eight-wave completeness fixtures | every expectation met |
| mocked end-to-end | 13 / 13 scenarios, 199 / 199 checks, first recorded attempt |
| guarded CPU-only tests, targeted (tray14) | 940 passed, 1 skipped |
| guarded CPU-only tests, whole tier (tray14) | 5855 passed, 78 skipped, 6 failed: the same failing sections as every earlier whole-tier run |
| after the rehearsals | GPUs 0 MiB, no compute process, no mock or stand-in process, no rehearsal run root, executable = pinned bytes |

Attempts kept (`runs/v4scale/prep2/`):

- `preflight_dry0/`: preflight on the uncommitted tree before the pins existed: 72 PASS, 6 FAIL, all six about pins, manifests
  and uncommitted files, as expected.
- `attempt1/`: preflight, gate rehearsal, end-to-end rehearsal at `d04e99d3`: all met.
- `cpu/`: test logs. The first run of the new scenario tests had one failure (69 passed, 1 failed):
  `test_guard_and_storage_breach_in_the_same_sample` put the run in the capture phase at the guard's age, a state the launch
  rule excludes, and the supervisor correctly reported the GPU-held ceiling instead. The test was changed to the CPU-only
  phase. That first log was overwritten by the rerun (my omission); the failure is recorded here.

## Findings for review

1. **488 GiB has 0.8 GiB of margin over the V3 measurement**, and the growth over V4's longer capture is unmeasured (A.1).
2. **The supervisor gives one response.** If it has already answered a breach through the controller and the run is still
   going at 4680 s, the guard is recorded but not acted on; the reservation then depends on the operator. Shown by a test.
3. **The launch window is 470 s; expected launch age about 350 s.** A slower start is refused safely before any GPU use.
4. **The 720 s closeout and deletion budget depends on the operator.** V3 took 832 s from the stop file to the observed
   deletion, without a deadline. The mechanical parts need about 345 s. Closeout and monitor now print the deadline.
5. **A lost supervisor is not visible in the closeout result** (`supervisor_lost`: closeout CLEAN); gate G11 catches it as
   INCONCLUSIVE and the monitor prints an abort condition during the run.
6. The `guard_and_storage` end-to-end scenario cannot force both causes into one sample; in this attempt both were present.
   The deterministic case is the unit test.
7. Unchanged residual: before a stop request, a failed supervisor is covered only by the operator's monitor.

## Files

`E1-V4-SCALE-PACKET.md` (revision 2, §13 lists the differences), `E1-V4-TIMELINE.md` (revision 2),
`feasibility/E1-V4-MEMORY-BUDGET.md`, `checks/e1_preflight_memory.py`, `checks/e1_reservation_guard.py`,
`tests/test_e1_preflight_memory.py`, `tests/test_e1_reservation_guard.py`, `tests/test_e1_v4scale_reservation_scenarios.py`,
`runs/v4scale/prep2/derivation/` (diffs against the frozen V3 scripts), `V4S-PIN.json`, `SCRIPTS_v4scale.sha256` (90 files),
`INPUTS_v4scale.sha256`, `FREEZE_v4scale.sha256` (proposed).
