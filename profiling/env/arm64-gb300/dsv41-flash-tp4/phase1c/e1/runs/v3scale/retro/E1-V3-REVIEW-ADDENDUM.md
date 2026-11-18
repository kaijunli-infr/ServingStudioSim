# E1 V3 — review addendum: the configuration-identity defect (2026-11-18)

Separate from, and changing nothing in, the original V3 record. **Original V3 verdict: FAIL / STOP under the frozen gates;
it stays.** The one-launch GO is consumed. The original report (`../run/E1-V3-RUN-REPORT.md`), the gate outputs
(`../run/evidence/cpu/V3S-GATES.json` sha256 1b63ed93…, `V3S-GATES.txt`), the identity report
(`../run/evidence/cpu/identity_v2_e1_v3_scale.{json,txt}`), the frozen packet and its manifests are byte-for-byte as committed at
5a01f642; the replay checks this (§3). The original G9 result is not reinterpreted here.

## 1. The defect

- G9 of the frozen V3 gates requires the three scientific checkers to exit 0 and the configuration-identity report to be PASS.
- The frozen V3 post-run produced that report with the V1/V2 checker and policy (`checks/e1_launch_identity_v2.py`,
  `checks/e1_identity_policy_v1.json`). Their rule for `workload.frontend.path` (`same_content_file`) accepts another path only
  if the file has the pinned hash of the accepted dry-run trace, c6d8e13b….
- The approved V3 workload is that trace repeated twice, d9d2dc77…. It cannot have the pinned hash, so the rule could not hold
  and G9 could not pass for the approved workload, whatever the run did.
- Original result: `profile.workload.frontend.path: rule condition not met: sha256 a=c6d8e13b7229ffd4 b=d9d2dc77b316b786
  pinned=c6d8e13b7229ffd4`; 29 other differing fields normalized; 182 launch and 74 profile fields compared; verdict FAIL.

## 2. How it got into the frozen gate set

- Introduced when the V3 post-run was derived from the V2 one (commit b474de7b): the identity step was copied unchanged.
- Not caught by review of the packet: the packet lists "configuration identity" under G9 without naming the policy's trace pin.
- Not caught by any rehearsal:
  - the gate rehearsal ran the post-run on the preserved V2 run, whose trace is the accepted one (identity PASS);
  - the synthetic fixtures supply the identity verdict as an input;
  - the end-to-end rehearsal had no launch record (identity not evaluable), and G9 was registered there as FAIL for another
    reason, which hid it;
  - the preflight checks trace and config identity by its own means and passed.
- The defect is in the preparation, mine. It was detectable before the run by executing the identity step once against the
  frozen V3 config with a V2 launch record.

## 3. What was verified about the preserved incident (CPU-only, `retro_tray16.log`)

| check | result |
|---|---|
| archive against its `SHA256SUMS` | all 207 files; `SHA256SUMS` sha256 44122927ea6c29995a5af63846bdc03e40c40def73eb221f80acd43c05bcda97 = the committed copy |
| raw stream | sha256 764d063d951d4c4683f1b4bea8336652846459de6b73ffb953a49d30744f8ba3 in the tray original, the archive copy and the archived durable copy = the controller's final record |
| original report, evidence, packet, manifests, gate code, post-run script, V1/V2 identity checker and policy | unchanged since 5a01f642 (`git diff` empty) |
| committed original gate outputs and identity report | byte-identical to the archived originals |
| frozen V3 manifests (freeze, 66 scripts, inputs) and V2 / V1 manifests | verify |
| recovery outputs | offline import of the durable stream rc 0 in both replays (55–56 s); recovered trace 4320 iterations, 17 280 NVTX ranges, 24 501 664 kernel rows, equal to the in-session export on ten checks, as in the original |
| the original failure | reproduced: the frozen post-run, unchanged, on a copy of the archived run: exit 1, FAIL, exactly G9, same rule and hashes |

## 4. The correction (new files only; nothing frozen was edited)

- `checks/e1_identity_policy_v3scale.json`: the V1 policy with one rule replaced. Launch rules, the other profile rules, the
  protected patterns and the worktree roots are identical (a test compares them). `workload.frontend.path` is handled by a new
  rule kind, `derived_trace`.
- `checks/e1_launch_identity_v3scale.py`: imports the unchanged V1/V2 comparison and adds that kind. It is satisfied only if all
  of these hold (18 checks, each reported):
  - both paths are the documented locations (the accepted trace's path; the frozen V3 trace's path under a listed worktree root);
  - the reference file has the pinned hash of the accepted trace;
  - the V3 file has the pinned V3 hash, and that hash is the entry of `INPUTS_v3scale.sha256` and of `E1-V3-SCALE-INPUTS.json`,
    both of which must themselves have the hashes the freeze manifest lists;
  - re-derived from the accepted trace's bytes, without the generator: same header and required fields; 56 source rows, 112 rows;
    rows 1–56 byte-identical in order; rows 57–112 the same rows in order with the wave index renamed once and the arrival time
    plus 80 000.0 ms, other fields identical; no other row or blank line; ids unique; the four waves (b=8 and b=48, twice) with
    their counts and arrival times; the whole file equal to the re-derivation.
  - the policy applies only to `e1_v3_scale` compared with `e1_nsys_val`, and the trace rule must have been exercised.
- It is not a path exemption (content is proved) and not a hash allow-list (the hash is necessary, not sufficient: a test gives
  the checker a tampered trace together with a matching pin and manifest, and it still fails on the derivation).
- Model, server arguments, routing hook, backend, profiler and CUDA settings stay under the unchanged protected patterns.
- V1 and V2 keep their own checker and policy; the old checker rejects the new rule kind, so the V3 policy cannot loosen it.

## 5. Independent verification

`e1_v3scale_identity_independent.py` shares no code with the checkers (`independent/`):

- the generator, run on a scratch copy, reproduces the frozen trace byte for byte; the run directory's `trace.csv` and the file at
  the configured path are the same bytes; the hash is the one in the frozen manifest and inputs file;
- the trace is the 56 accepted rows followed by the same rows with wave index 1 and +80 000.0 ms;
- its own flattening finds 182 launch fields (164 equal, 18 differing) and 74 profile fields (62 equal, 12 differing): the same
  counts as the original report, and the differing fields are exactly the 29 it normalized plus the trace path;
- no field under server, model, Python runtime, profiler or workload settings other than the trace path differs.

Tests (`tests/test_e1_launch_identity_v3scale.py`, 48): 15 trace defects (modified, reordered, omitted, duplicated, unexpected
rows, a third copy, wrong offset, wave not renamed, another cell, header, blank line), format changes, a consistent copy of another
source, hash-or-manifest-alone, unreadable files, undocumented paths, 15 unauthorized configuration changes on the actual V3
records, the original failure reproduced with the frozen checker, the V3 policy refused for other runs and other shapes, and V1 /
V2 passing under their own policy and failing with a changed trace.

## 6. Verdicts (two, separate)

1. **Original V3: FAIL under the frozen gates.** Unchanged.
2. **Retrospective V3 evidence assessment: see `E1-V3-RETROSPECTIVE-ASSESSMENT.md`.** It is an assessment of preserved evidence
   under a policy written after the result was known. It does not replace the original verdict and authorizes nothing.
