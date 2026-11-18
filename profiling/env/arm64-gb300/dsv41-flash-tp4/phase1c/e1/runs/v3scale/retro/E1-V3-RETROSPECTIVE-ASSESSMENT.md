# E1 V3 — retrospective evidence assessment (CPU-only replay on preserved evidence, 2026-11-18)

Two verdicts, kept separate:

| | verdict | basis |
|---|---|---|
| Original V3 | **FAIL / STOP** | the frozen gates as run on 2026-11-18 22:25Z; G9 FAIL on the configuration-identity step; unchanged |
| Retrospective V3 evidence assessment | **PASS, qualified** | the same post-run with one line changed (the identity step uses the V3-specific policy); 14 of 14 gate results PASS; qualifications in §3 |

No GPU was used. No new capture was made. The replay ran once and met its checks on the first attempt
(`retro_tray16.log`); there is no failed validator attempt to report.

## 1. What was replayed

`retro_e1_v3scale.sh` on tray16, at 1747d00c, on a copy of the archived run directory (removed afterwards); the archive and the
tray-preserved stream were read only and re-verified before and after.

| step | result |
|---|---|
| reproduction: the frozen post-run, unchanged | exit 1, FAIL, exactly G9 not passing, on the same rule (`reproduction/`) |
| corrected post-run | generated from the frozen script by replacing the identity command; the difference is exactly one line (`corrected/postrun_one_line_difference.diff`). Gate code, the three scientific checkers, the inputs, every threshold and the recovery are the frozen ones |
| corrected identity report | PASS: 30 normalized fields (the 29 of the original plus the trace path), 18 of 18 trace checks, no failed field (`corrected/identity_v2_e1_v3_scale.{json,txt}`; the file name is the frozen post-run's) |
| gates | 14 of 14 PASS, none INCONCLUSIVE (`corrected/V3S-GATES.{json,txt}`) |
| difference from the reproduction | one gate: G9 FAIL → PASS. The other thirteen results are identical in status |

## 2. Gate results of the corrected post-run

| gate | result | note |
|---|---|---|
| G0 source and executable | PASS | from the run's own launch and binary records; the executable on disk is still the pinned one |
| G1 host identity | PASS | |
| G2 one stop request | PASS | |
| G3 phases in order | PASS | |
| G4 phase deadlines | PASS | flush 112.0 s, import 53.0 s, stop → report 166.2 s, hard deadline 600 s unextended |
| G5 importer observed | PASS | one importer, own-counter activity, completed, no forbidden signal |
| G6 GPU release interval | PASS | released no later than 2.86 s after the shutdown SIGTERM |
| G7 stream durable | PASS | sha256 764d063d… in every copy |
| G8 / G8b offline recovery | PASS | re-imported in the replay (55 s); 4320 iterations, 24 501 664 rows, ten equality checks |
| G9 scientific completeness | PASS | completeness (four cells, routing records, checksum), extract and Nsight validation exit 0, re-run in the replay; identity PASS under the V3 policy |
| G10 bounds | PASS | GPU-held 1833 s, reservation 60.2 min |
| G11 storage and memory | PASS | run-phase samples from the run; post-run samples from the replay (see §3) |
| G12 closeout | PASS | the original closeout exit code and reservation-deleted record |

## 3. Qualifications

1. **The policy was written after the result was known.** The corrected identity rule is strict and independently checked, but
   it is not a prospective gate. That is why this is an evidence assessment and the original verdict stands.
2. **Only the identity step was changed, and only G9 changed.** No threshold, band, deadline or other checker was touched; the
   one-line difference is recorded. Whether G9 "would have passed" is not claimed; the corrected check passes on the evidence.
3. **Post-run measurements in the replay are the replay's own.** G11's post-run storage and memory samples, and the G8 recovery,
   were re-measured (temporary peak 5.219 GB, offline importer 15.63 GiB). They agree with the original post-run (5.219 GB,
   15.61 GiB), which is preserved. The persistent figure in the replay is the replay root's.
4. **The scientific checkers were re-run on a copy.** Their outputs match the originals in verdict and counts.
5. **Limits of the original evidence still apply**: one run, two cell kinds, mock-only rehearsal of the supervisor's response,
   importer memory sampled once per second.

## 4. What this does not do

It does not change the V3 verdict, does not authorize a V3 or V2 retry or a full E1, and creates no identification partition and
no fit. Disposition is the reviewer's.
