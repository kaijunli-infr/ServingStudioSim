# Phase 1C validation protocol (frozen before any data collection)

Version 1.0, 2026-11-13. The sha256 of this file is recorded in `PROTOCOL.sha256`; `tests/test_invariants.py` does not
check it (the protocol governs people, not code), but every Phase 1C analysis report must quote the hash of the
protocol it followed, and a protocol change requires a new version number, Kevin's approval, and applies only to
captures started after the change.

## 1. Frozen artefacts

| artefact | identity | rule |
|---|---|---|
| Phase 1A coefficients | `phase1a/out/coefficients.json`, sha256 78f937c5… | never modified; a revised model writes a new file with a new hash |
| Phase 1A partition | `phase1a/partition/partition.json`, sha256 288e193f… | never modified; its held-out rows remain held-out for every future model version (they were looked at once for Phase 1A and once for Phase 1B and are therefore *reported*, not *validation*, from now on) |
| Phase 1B parameters and predictions | `phase1b/out/params.json` (8dc79573…), `results.json`, per-iteration parquet (canonical checksums in `phase1c/repro/MANIFEST.json`) | never modified; reproducible through `phase1c/repro/verify.py` |
| the three Phase 0 captures | run2, quadrant, salvaged compact tables (hashes in the manifest) | may be used for identification of new parameters only in their calibration-role rows; their held-out rows are not used for anything new |

## 2. Designations

Every new capture receives exactly one designation **before** it runs, written in the approved experiment matrix:

- **identification**: its iterations may be used to estimate parameters, choose model structure, or set bands. Nothing
  estimated from it is validated by it. Its rows get role `identification` in the Phase 1C partition.
- **validation**: its iterations are used once, for a pre-registered comparison, after the model version under test
  is frozen (hash recorded). Role `validation`. No parameter, threshold, band, exclusion rule or class definition may
  be changed in response to a validation result; a change produces a new model version that needs a *new* validation
  capture. A validation capture is never re-used for a later model version.
- **identification + validation half**: two separate vLLM launches (separate runs, different request-order seeds); the
  halves are assigned at capture time by run, never by picking iterations after looking at them.

A capture is called *held-out* only if it carries the validation designation and has not been opened for any purpose
other than the one-shot comparison. Describing an identification capture as held-out, or using a validation capture's
results to refit, is a protocol violation and the affected results are reported as "calibration".

## 3. Partition

New captures are recorded in a **new** partition file `phase1c/partition/partition-1c.json` (role per iteration:
`identification` / `validation` / `excluded`), hash-recorded, with the same leakage guard as Phase 1A
(`Partition.calibration_only` refuses fits on non-identification rows). The Phase 1A partition file is not edited.

## 4. Independence requirement

When a capture identifies a missing or previously unidentifiable parameter (c_kv, rank-2 offset, common-mode fraction,
routing-conditional MoE term, first-chunk profile, Q cap, indexer slope beyond 115k), its validation must come from a
**different** capture: a separate vLLM launch, a different request-order seed and prompt subset, and — for E1/E2 — a
different day. Validation of a parameter on the capture that identified it is reported as in-sample and does not count.

## 5. Pre-registered comparison (one shot per model version)

Metrics, computed by the frozen scripts (`phase1b/seqsim/evaluate.py`, `phase1c/uncertainty/reliability.py`, hashes
recorded) in the four Phase 1B modes with 32 draws, seed 0, plus seeds 1–4 for seed variation:

per class (decode b1-2, b3-16, b17-24, b≥25; prefill 16k chunk by kv0 band; mixed-eager host-bound; mixed-ring other;
mixed-graph) and role: n, signed bias (ms, %), median and p90 absolute error, p90 APE, worst case, p10–p90 coverage,
mean and median interval width, fraction below/above; full-workload span error, Σ walls signed and absolute error,
throughput; lead and host-skew tables by transition.

Acceptance for a model version to be eligible for Phase 1C-2 integration (all on validation captures):

| class | acceptance |
|---|---|
| decode b3-16, b17-24, b≥25 | p90 APE ≤ 5 %, |signed bias| ≤ 2 %, coverage p10–p90 within 70–90 % |
| decode b1-2 | p90 APE ≤ 10 % or explicitly flagged low-confidence in the output (status propagated) |
| prefill 16k chunk, kv0 ≤ 160k | p90 APE ≤ 3 %, coverage 70–90 % |
| mixed-eager host-bound | p90 APE ≤ 10 % or flagged low-confidence |
| mixed-ring other / mixed-graph | p90 APE ≤ 8 % or flagged |
| full workload (span and Σ walls) | |error| ≤ 1 % on every validation pass |
| lead on chains | p90 abs error ≤ 20 ms |

Classes with n < 30 on a validation capture are reported but cannot pass or fail; they remain flagged.

## 6. Reporting

Every analysis report states: protocol version and hash; capture ids, designations, raw-capture SHA256SUMS; model
version (coefficients / params / sources hashes); exclusion counts by rule; all metrics of §5 for every class including
those below n = 30; seed variation; and an explicit list of parameters that were changed since the previous version and
which identification capture supported each change. Negative results are reported with the same tables.

## 7. Prohibited

Tuning on validation rows; widening bands to raise coverage without an identified source of dispersion; global
multipliers; re-running a validation comparison with changed exclusions; editing this file for an in-flight capture.
