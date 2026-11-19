# E1 V4 — CPU-only preparation report (2026-11-19)

Prepared under the review decision on `598d4a2d`. **V4 GPU execution: NO GO. Nothing was launched, no GPU was used, no
reservation pod was created.** The packet and manifests are proposals; the freeze has not been approved. Original V3 verdict
(FAIL / STOP) and the qualified retrospective assessment are unchanged; the frozen V3, V2 and V1 manifests verify.

Commits on `exp/profiler-lifecycle`: `d0677aaa` (code, inputs, tests, analyses), `2fbf06c4` (proposed packet, pins, manifests),
then this evidence. All rehearsals ran at `2fbf06c4` with a clean tracked tree.

## 1. What was asked and where it is

| item | result | where |
|---|---|---|
| A. workload: four exact copies of the accepted trace | 224 requests, 8 waves, b=8 and b=48 kept; deterministic generator with pinned source and output hashes; expected 8640 iterations, 49 003 328 rows | `make_e1_inputs_v4scale.py`, `E1-V4-SCALE-INPUTS.json`, packet §2 – §3 |
| B. prospective enforcement of every ceiling | all eleven limits registered and enforced or judged; the reservation ceiling is now enforced on the tray (new) | packet §4, §5, §8 |
| C. timing feasibility | one worst-case timeline; three durations; no-stop and failed-monitor behaviour; compatibility table; **two findings** | `E1-V4-TIMELINE.md` |
| D. row counts | kernel rows per wave identical in three waves per cell; rank 0 one kernel more per iteration; 216 M consistent under a stated assumption | `feasibility/E1-ROWS-PER-ITERATION-ANALYSIS.md` |
| E. V4 identity and gates | V4-specific policy and checker; G13 – G15 added to the unchanged V3 gates; the complete V4 post-run exercised on the actual V4 config | `checks/`, packet §6 – §7, §3 below |
| F. early reservation release | designed, CPU-tested, **not enabled**; one real conflict found (cleanup evidence) | `E1-V4-RESERVATION-RELEASE-DESIGN.md` |
| G. rehearsals and freeze preparation | all expectations met on the first recorded attempt; proposed packet, pins and manifests written | §2 – §5 below |

## 2. Attempts (all kept)

| attempt | what | result | where |
|---|---|---|---|
| `preflight_dry0` | preflight on the uncommitted tree, before pins and manifests existed, to find script defects early | FAIL as expected: 40 FAIL lines, all of them the missing freeze manifest, pins, script manifest and the identity preflight that needs the pins; every V4-specific check PASS; 622 tests passed | `preflight_dry0/` |
| `attempt1/preflight` | read-only preflight at `2fbf06c4` | **PASS: 75 PASS, 0 FAIL**; 622 tests under the CPU-only guard | `attempt1/preflight/` |
| `attempt1/rehearsal` | identity rehearsal, complete V4 post-run on the V4 configuration, synthetic fixtures, eight-wave fixtures | **expectations met** (§3) | `attempt1/rehearsal/` |
| `attempt1/e2e` | end-to-end on mock processes, five scenarios | **expectations met** (§4) | `attempt1/e2e/` |
| `preflight_final` | read-only preflight at the evidence commit `e418f668` | **PASS: 75 PASS, 0 FAIL** | `preflight_final/` |
| `cpu/` | guarded tests on tray14 | targeted 879 passed, 1 skipped; whole tier 5794 passed, 78 skipped, 6 failed (the same six as in every earlier whole-tier run; unrelated to this work) | `cpu/` |

Failures during development, before the first recorded attempt (not rehearsals; reported for completeness): two of my new unit
tests were wrong (one matched a test file's name in a script, one placed a census before the release it was meant to follow)
and were corrected; the input generator had a variable shadowing bug found by its first local run. No gate, limit or
expectation was changed in response to a result.

## 3. The complete V4 post-run on the actual V4 configuration (`attempt1/rehearsal`)

This is the step V3 lacked.

- **Identity rehearsal** (13 of 13): the V4 checker and policy on the generated V4 config, the V4 trace and the predicted V4
  launch record give PASS with all 20 trace checks, 30 normalized fields, 182 launch and 74 profile fields compared. The V1 / V2
  policy on the same records FAILs on `workload.frontend.path` only, which is the V3 defect; the V3 policy FAILs too. Seven
  negative controls FAIL (V3 config, rows swapped, row removed, streaming on, hard deadline changed, server argument changed,
  trace at an undocumented path). The prediction method reproduces the actual V3 launch record from the V2 one in all fields.
- **Post-run** (`postrun_e1_v4scale.sh`, unchanged, real importer and exporter) on a run directory with the V4 config byte for
  byte, the V4 trace and the predicted launch record, and V3's capture data. Registered before the run: verdict FAIL with exactly
  G8b, G9 and G15 not passing (the data have four of the eight waves). **Met**: 17 results, 14 PASS, 3 FAIL, 0 INCONCLUSIVE.
  G13 PASS (V4 policy, 20 of 20 trace checks), G14 PASS (four-copy trace and V4 config in the run directory), G11 PASS against
  the V4 ceilings, G0 – G7, G8, G10, G12 PASS.
- **Synthetic fixtures**: 117 of 117 met (3 positive, 6 boundary, 108 negative; 9 PASS, 76 FAIL, 32 INCONCLUSIVE verdicts).
  Boundaries: every phase exactly at its deadline; release upper edge exactly 90 s; storage, importer memory and MemAvailable
  exactly at their ceilings; stop → report exactly 600 s with the soft warning; reservation exactly 90 min; GPU-held exactly
  2700 s. Each has its over-the-limit counterpart among the negatives.
- **Eight-wave completeness**: 10 of 10 (the eight-wave structure passes; V3's four waves, V2's two, a missing wave, routing
  and corpus defects, overlapping waves and a short request fail).

Limits of this evidence: the capture data are V3's, so nothing here shows the lifecycle at 49 M rows; the launch record is a
prediction (G13 judges the real one); the wording "four-wave plan" in G9's detail line comes from the unchanged V3 gate code,
the plan it ran is the eight-wave one.

## 4. End-to-end rehearsal on mock processes (`attempt1/e2e`)

The actual V4 driver, closeout, post-run and archive scripts with the real controller, observers, supervisor, registry and
prebuilt-executable contract; mock server, Nsight and importer processes; CUDA hidden. The driver's own preflight passed in
every scenario (74 PASS, 0 FAIL).

| scenario | expectation | result |
|---|---|---|
| success | launch → closeout CLEAN → post-run → archive; then a failing importer is detected | 19 of 19 |
| ambiguous | two importer candidates: recorded INCONCLUSIVE, run continues under the other monitors | 18 of 18 |
| memory_breach | marker → supervisor → release line → controller's server-only stop; importer untouched and finishes | 15 of 15: acknowledged after 0.28 s; breach → SIGTERM 0.6 s; SIGKILL after the unchanged 60 s grace |
| storage_breach | marker → supervisor → identity-verified registry cleanup | 15 of 15 |
| reservation_guard (new) | guard passes during the capture → supervisor → registry cleanup; no breach marker; closeout NOT CLEAN | 16 of 16: cause recorded 420.9 s after the record's creation (rehearsal guard 420 s); cleanup finished 0.43 s later; 13 processes signalled, all in the verified owned tree; no survivor |

Registered gate outcome of a rehearsal launch: FAIL with G0 FAIL, G6 and G10 INCONCLUSIVE, G9 FAIL, G13 INCONCLUSIVE (no launch
record from the mock launcher), G14 FAIL (the rehearsal copy of the config is renamed), G15 INCONCLUSIVE. A rehearsal can never
be V4 evidence. In both cleanup scenarios the registry cleanup reported one lookalike, which it did not signal (rc 2), as in the
V3 rehearsal.

## 5. Proposed freeze set

`E1-V4-SCALE-PACKET.md`; `SCRIPTS_v4scale.sha256` (82 files, sha256 f9463f21eabcb0ec…); `INPUTS_v4scale.sha256`;
`E1-V4-SCALE-INPUTS.json`; `V4S-PIN.json` (code commit `d0677aaa`); `V4S-PIN-identities.json`; `BINARY-PROVENANCE.json`
(unchanged); `FREEZE_v4scale.sha256` (7 entries, sha256 6a7d21094507628b…). The V4 scripts are derived from the frozen V3 ones;
`derivation/` has one diff per file showing every edit beyond the renaming, and the script that produced them.

## 6. Findings for review

1. **Preflight host memory** (timeline §6.1): the retained 448 GiB minimum is not consistent, at the extremes, with the importer
   ceiling, the floor and the server footprint measured in V3 (285 – 295 GiB); 488 GiB would be. Not changed. V4's predicted
   minimum is about 530 GiB, so it is not affected in practice.
2. **Reservation corner** (timeline §5, §6.2): the latest allowed launch together with a ceiling-long GPU hold cannot be
   completed inside 90 min under the existing reservation lifecycle; the guard would end the export. Every single extreme, and
   every predicted case, fits.
3. **V3 sampler lifetimes would have been wrong for V4**: 3400 s is shorter than a V4 run with its 20-minute export; the
   supervisor would have aborted a healthy run. Changed to 5400 s in the V4 driver and listed in the packet.
4. **The reservation guard is new enforcement** and can end a healthy but slow export 78 min after the pod's creation.
5. **Early reservation release** conflicts with the cleanup evidence taken after the export (census, closeout GPU lines); it
   would need gate changes and one CPU-only demonstration on the cluster before it could be enabled.
6. **Rows per iteration depend on the batch size** (1.8 % between b=8 and b=48) and prefill iterations are 10 – 30 % heavier;
   the full-E1 size is consistent with 216 M only under an untested assumption about the other 22 cells.
7. Residual, unchanged from V3: before the stop request, a failed supervisor is covered only by the operator's monitor.

## 7. What is not claimed

No V4 result exists. The predictions for 49 M rows are extrapolations from two controlled sizes. The supervisor's responses
have run only against mock processes. The repeated waves are replicates, not scientific evidence.
