# DRAFT — E1 V4 intermediate-scale lifecycle validation (≈ 49 M rows)

**Status: draft for review. Not approved, not frozen, not executable. No scripts, inputs, gates or manifests exist for it. Every
number marked "proposed" needs a decision; every number marked "unchanged" is the V3 value.** GPU execution: NO GO.

## 1. Purpose and non-purpose

- Purpose: one controlled capture at twice V3's size to test the profiler lifecycle and the operational safeguards against
  prospective limits: flush, import, total, GPU-held time, importer memory, storage, stream preservation, offline recovery.
- Non-purpose: scientific identification. The workload repeats two accepted cell kinds; its repeats are replicates within one
  session, not independent evidence. No identification partition, no fit, and no use of this capture in E1's analysis.

## 2. Workload (unchanged recipe, more copies)

| item | value |
|---|---|
| trace | the accepted `traces/e1_dry.csv` (56 requests, sha256 c6d8e13b…) four times: wave index 0 – 3, copies offset by 80 000 ms; 224 requests, 8 waves (b=8 and b=48 at 32k, four each) |
| config | the V2 / V3 config with only name, log directory and trace path changed |
| model, server arguments, routing hook, backend, profiler and CUDA settings, fork, executable | unchanged; pinned as in V3 |
| expected size | 8640 iterations, ≈ 49.0 M kernel rows (E: twice V3) |
| expected capture | ≈ 300 s (E) |

## 3. Predictions (to be registered before the run; not gates) and limits

| quantity | prediction (E) | limit | status of the limit |
|---|---|---|---|
| flush | 203 – 249 s | 300 s | unchanged |
| import | 97 – 118 s | 300 s | unchanged |
| stop → report | 301 – 367 s | 600 s, controller hard deadline, no extension | unchanged |
| GPU-held | 2092 – 2311 s | 2700 s | unchanged |
| reservation | 75 – 81 min | 90 min | unchanged |
| importer peak memory | 28 – 35 GiB | 128 GiB | unchanged |
| MemAvailable | ≈ 530 GiB minimum | preflight ≥ 448 GiB, floor 64 GiB | unchanged |
| temporary storage | 10.4 – 11.4 GB | 15 GB | **proposed** (V3: 10 GB) |
| run root | 12.6 – 14.0 GB | 18 GB | **proposed** (V3: 8 GB) |
| run root + archive | 26.5 – 29.5 GB | 38 GB | **proposed** (V3: 16 GB) |
| iterations / kernel rows | 8640 / 49.0 M | bands of ± 5 % as in V3 | to be generated |

Controller, its termination semantics, the 60 s API-server grace and the supervisor are used unchanged.

## 4. Safeguards (all existing mechanisms)

Reservation pod on the capture tray; preflight in driver context; pinned executable in prebuilt mode; import probe, memory,
storage and GPU samplers with heartbeats; supervisor with the registered abort line; raw-stream preservation by hard link, final
hash after session close, durable copy; offline import and ten-check comparison; registry-based, identity-verified cleanup;
closeout; archive with checksums. One launch, no retry; any accepted control line or supervisor response is a FAIL.

## 5. Gates

The V3 gate set G0 – G12 with regenerated inputs (eight-cell plan, bands, predictions, storage limits of §3). Verdict rule as in
V3: PASS only if every gate passes; FAIL / STOP otherwise; INCONCLUSIVE is not PASS.

## 6. Work required before a freeze could be proposed (CPU-only, each needing its own GO)

1. Trace and config generator for four copies; input manifest.
2. A V4-specific, fail-closed identity policy: the derived-trace proof generalized from "exactly twice" to "exactly four times"
   (rows, order, fields, wave indices, offsets, eight waves, frozen manifest hash), with negative fixtures.
3. Gate inputs and fixtures for eight waves.
4. **A rehearsal of the complete post-run, including the identity step, against the new config** using a preserved launch
   record. This is the step whose absence produced the V3 failure.
5. End-to-end rehearsal on mock processes with the new inputs; preflight; guarded test tier.
6. Packet, manifests, freeze, review.

## 7. What the result would be used for

| outcome | reading |
|---|---|
| all gates pass, flush near 203 s | sub-linear as between V2 and V3; full-E1 limits can be proposed nearer the lower curves, still with a 4.4 × extrapolation |
| all gates pass, flush near 224 – 249 s | proportional or steeper; full-E1 limits must come from the proportional and stress columns |
| flush over 300 s, trace preserved | FAIL; the scaling is steeper than every curve considered here; full E1 not sizable from these data |
| any other gate fails | FAIL / STOP; analysed as an incident, no retry |

One further point does not validate a scaling model; it narrows the range and tests the limits.
