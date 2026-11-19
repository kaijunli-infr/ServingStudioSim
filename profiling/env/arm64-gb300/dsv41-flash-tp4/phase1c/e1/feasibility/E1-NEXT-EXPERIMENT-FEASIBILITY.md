# Full E1 feasibility and next-experiment design (CPU-only analysis, 2026-11-18)

Analysis and draft material only. Nothing here is approved, frozen or executable; no limit, ceiling or controller behaviour was
changed; no identification partition was created and nothing was fitted to scientific data. Labels: **M** measured in a
controlled run, **O** observed once outside a controlled run, **E** computed from M / O (arithmetic in `e1_scaling_model.py`),
**A** assumption without a measurement.

Standing record, unchanged: original V3 verdict FAIL / STOP under the frozen gates; retrospective V3 evidence assessment PASS,
qualified, post hoc and not prospective; the corrected V3 identity policy stays a separate file outside the frozen protocol;
archive, stream hashes, trace and recovery evidence preserved (re-checked in §7).

## 1. Basis

| | rows | flush | import | importer memory | source |
|---|---|---|---|---|---|
| V2 | 12.25 M | 61.9 s | ≈ 28.9 s (E, importer not observed) | 8.7 GiB (offline only) | M |
| V3 | 24.50 M | 112.0 s | 53.0 s | 15.6 GiB (in session and offline) | M |
| e1_main_v3 | ≈ 216 M (plan estimate, never measured) | 865 s | 49 % of the bar after 217 s, then destroyed | 61 GB, one sample at ≥ 49 % | O |

- Two controlled sizes, a factor of two apart, on one workload (b=8 and b=48 at 32k context). That fixes two points; it does not
  validate a line, a proportional law or an exponent. Run-to-run noise is known from one pair only (V1 63.8 s, V2 61.9 s: 3 %).
- The model therefore reports, for each quantity, the spread over three curves through the measured points (power, line,
  proportional) and a separate stress case (exponent 1.15, A). The spread is the model uncertainty.
- Correction to `E1-FULL-READINESS-ANALYSIS.md` §3: it gave V3's startup as ≈ 1640 s. The V3 run report's timeline gives launcher
  start 21:29:07 to measurement begin ≈ 21:54:30, about 1523 s. Here the startup range is 1487 – 1640 s, with 1640 s kept only
  as a conservative upper bound.

## 2. The three options

| | A. no further scaling experiment | B. intermediate, ≈ 49 M rows | C. full E1, ≈ 216 M rows (E) |
|---|---|---|---|
| workload | — (next GPU step would be C) | the accepted 56-request trace four times: 8 waves, 224 requests | `traces/e1_main.csv`: 36 waves, 450 requests, 24 cells |
| rows | 24.5 M (M) | 49.0 M (E: twice V3; V3 was exactly twice V2, M) | 216 M (E); 173 – 259 M if the estimate is off by 20 % (A) |
| distance from the largest controlled size | — | 2 × | 8.8 × (7.1 – 10.6 ×) |

Full table: `E1-SCALING-MODEL.md` (generated). Ranges are min – max over the three curves; the stress case is in brackets.

| quantity | B ≈ 49 M | C ≈ 216 M | C with rows 173 – 259 M | V3 limit (unchanged) |
|---|---|---|---|---|
| flush | 203 – 224 s (249) | 721 – 987 s (1369) | 596 – 1184 s (1686) | 300 s |
| import | 97 – 106 s (118) | 356 – 467 s (648) | 293 – 560 s (798) | 300 s |
| stop → report | 301 – 331 s (367) | 1078 – 1456 s (2017) | 890 – 1745 s (2485) | 600 s |
| GPU-held | 2092 – 2275 s (2311), 35 – 39 min | 3466 – 3997 s (4558), 58 – 76 min | 3278 – 4286 s (5026) | 2700 s |
| GPU-held if the 600 s hard deadline acts | 2599 s | — (deadline too short) | — | 2700 s |
| reservation, sequence as in V3 | 75 – 78 min (81) | 166 – 176 min (220) | 145 – 199 min (257) | 90 min |
| reservation if deleted at verified GPU release | 42 – 45 min (46) | 65 – 74 min (83) | 62 – 79 min (91) | 90 min |
| importer peak memory | 28 – 31 GiB (35) | 98 – 138 GiB (191) | 81 – 165 GiB (235) | 128 GiB |
| MemAvailable minimum | 532 – 535 GiB (528) | 425 – 465 GiB (372) | 398 – 482 GiB (328) | floor 64 GiB |
| temporary storage peak (post-run) | 10.4 – 10.5 GB (11.4) | 41 – 42 GB (57) | 33 – 50 GB (69) | 10 GB |
| run root | 12.6 GB (14.0) | 55 – 56 GB (77) | 44 – 67 GB (95) | 8 GB |
| run root + archive | 26.5 GB (29.5) | 117 GB (162) | 93 – 141 GB (200) | 16 GB |
| export and parse after GPU release | 20 min (23) | 89 – 90 min (125) | 71 – 108 min (154) | — |
| offline recovery: import / export | 112 – 121 s (124) / 260 s (288) | 494 – 623 s (684) / 19 – 20 min (26) | — | — |
| offline recovery: memory | as the in-session importer (M at V3: 15.6 GiB both) | same (A at this size) | | |

Option A has no row of its own: it means these C figures are used for a full E1 packet with no further controlled point.

What is measurement and what is not, in that table:

- M: every V2 / V3 value the curves pass through; the capture duration of the full trace (896 s, e1_main_v3); startup 1487 s
  (V2) and ≈ 1523 s (V3); the fixed parts of the V3 reservation (5.3 min before the launch, 6 min closeout, 8.3 min to deletion).
- O: the 865 s flush. It lies inside the C range (721 – 987 s) if the trace had 216 M rows; with 173 M rows it would imply an
  exponent of 1.05 against V3, with 259 M rows 0.87. The row count is the weak link, not the timing.
- E: everything at 49 M and 216 M.
- A: that the curves' shape holds beyond 24.5 M; that rows per iteration are the same in the 22 cells never captured under the
  controller (this is what "216 M" rests on); that the server's host footprint (285 GiB, M) does not grow with 96k contexts;
  that closeout and reservation handling do not grow with size; that the export and parse time is proportional (two points give
  0.99); the stress exponent itself.

## 3. Sensitivity

- **Model choice.** At 49 M the three curves differ by 10 % for flush (203 / 212 / 224 s) and for importer memory (28 / 29 /
  31 GiB). At 216 M they differ by 37 % for flush and 41 % for memory; with the stress case the spread is a factor of 1.9 for
  both. Extrapolating 8.8 × from two points does not determine a deadline or a memory ceiling to better than a factor of two.
- **Row count.** ± 20 % on the full trace moves every C figure by about ± 20 %, on top of the model spread: flush 596 – 1686 s
  across both.
- **Which limits bind first** (largest size at which each curve stays inside the unchanged V3 limit, M rows):

  | limit | power | line | prop | stress |
  |---|---|---|---|---|
  | flush ≤ 300 s | 78 | 70 | 66 | 58 |
  | import ≤ 300 s | 178 | 150 | 139 | 111 |
  | importer ≤ 128 GiB | 297 | 224 | 201 | 153 |
  | run root ≤ 8 GB | 31 | 31 | 31 | 30 |

  The storage ceilings are passed just above V3's size under every curve. The flush deadline is next, between 58 and 78 M rows.
  So 49 M is close to the largest size that keeps the frozen flush, import, total, GPU-held and memory limits with margin (flush
  margin 1.2 – 1.5 ×), and no size above V3 keeps the frozen storage ceilings.
- **Startup.** The startup range (1487 – 1640 s, about 150 s) is five times the model spread of stop → report at 49 M (30 s);
  GPU-held time at that size is dominated by startup, not by the scaling question.
- **What a 49 M point could discriminate.** Flush predictions are 203 s (exponent 0.86), 224 s (1.0) and 249 s (1.15): steps of
  about 10 %, against 3 % noise known from a single pair. It would separate "sub-linear as in V2 → V3" from "super-linear", not
  pin an exponent. Importer memory likewise (28 / 31 / 35 GiB), limited further by 1 s sampling, which reads a lower bound.

## 4. What an intermediate capture would and would not establish

Two purposes, kept apart:

1. **Profiler lifecycle scaling and operational safety.** This is what B addresses.
2. **Scientific rank-dependence identification.** B adds nothing to this.

| B would establish (beyond V3) | B would not establish |
|---|---|
| a third controlled size on the same workload, so flush, import and importer memory are seen at 1 ×, 2 × and 4 × V2 | a scaling law; full E1 would still be a 4.4 × extrapolation |
| the first run in which a prospective deadline is close: flush expected at 68 – 83 % of 300 s (V3: 37 %) | behaviour of a flush longer than five minutes |
| storage peaks at twice V3 against enforced limits, during the run and the post-run | anything about a 35 GB export |
| stream preservation and offline recovery at 1.3 GB, twice the largest verified stream | recovery of a 5 – 6 GB stream |
| supervisor, samplers and importer probe over a capture twice as long (≈ 300 s, 224 requests) | a supervisor response against a real server (still mock-only) unless one occurs |
| repeatability of rows per iteration and of timing for the same two cells, four times in one session | rows per iteration, completeness or timing for any of the 22 cells not yet captured |
| | any new independent scientific evidence |

- Repeating the accepted trace four times repeats two cell kinds (b=8 and b=48 at 32k). The repeats share a session, a server
  state and a warm-up history; they are replicates of the same condition, not independent evidence, and they contain no new
  batch size or context length. They must not be counted toward identification, and this analysis does not propose to.
- Rank-dependence identification needs the 24-cell design. Only C, or a separately designed capture of new cells, provides it.
- A variant of B built from new cells (for example one wave per batch size at one context length, also ≈ 8 waves) was considered
  and is not recommended as the scaling experiment: its row count could not be predicted from a measurement, so its bands and
  deadlines would not be prospective, and it would mix the two purposes. It is listed as a decision (§6, D5).

## 5. Feasibility and safeguards for option B

Assessed against the mechanisms that exist at dbe4310f, unchanged.

| requirement | status for B | basis |
|---|---|---|
| bounded reservation | feasible: 75 – 81 min expected with the V3 sequence, limit 90 min | E; fixed parts M |
| independently enforced GPU-held ceiling | feasible with the unchanged 2700 s: expected 2092 – 2311 s; 2599 s if the controller's 600 s hard deadline acts after the slowest startup. Before the stop request the supervisor enforces it, after it the controller; the GPU sampler measures it independently | E; enforcement M on mock processes and in controller tests, never yet triggered on a real server |
| prospective flush, import and total deadlines | feasible with the unchanged 300 / 300 / 600 s: flush 203 – 249 s, import 97 – 118 s, total 301 – 367 s. The flush margin is the thinnest (1.2 – 1.5 ×); a flush over 300 s would be a gate FAIL with the trace still preserved if it ends before 600 s | E |
| host-memory headroom | feasible: MemAvailable minimum ≈ 530 GiB against the 64 GiB floor; importer 28 – 35 GiB against 128 GiB | E |
| measured storage peaks and enforced limits | **not feasible under the V3 ceilings**: temporary 10.4 – 11.4 GB (ceiling 10), run root 12.6 – 14.0 GB (8), with archive 26.5 – 29.5 GB (16). New limits would have to be approved. Space is not short (`/tmp` 1163 GB, CephFS 20 771 GB free, M) | E |
| importer-specific monitoring | available: the probe identified and followed the real importer in V3 | M |
| durable stream preservation | available: verified at 0.35 and 0.66 GB, after a forced release and after a normal stop | M |
| offline recovery | available: equal to the in-session export on ten checks at both sizes; expected ≈ 2 min import, ≈ 4.5 min export | M / E |
| identity-verified cleanup | available: registry-based cleanup, closeout CLEAN in V2 and V3 | M |
| configuration identity | **not available yet**: the V3-specific checker proves "exactly twice"; a four-copy trace needs its own policy and, this time, a rehearsal of the complete post-run against the new config before any freeze | this review |
| gates and inputs | not available yet: bands, predictions, the eight-cell plan and the input manifest would have to be generated and frozen | — |

Conclusion: B is feasible with the existing controller, deadlines, GPU-held ceiling and memory limits, **conditional on one
decision outside this analysis**: storage limits above the V3 ceilings. Without that decision B cannot be run as specified.

For C the same assessment gives: every time limit, the GPU-held ceiling, the reservation length, the storage ceilings and
possibly the importer ceiling would have to be set anew from an extrapolation whose spread is a factor of two; holding the
reservation through export and closeout as in V3 would take 2.5 – 3.5 hours.

## 6. Recommendation, risks and decisions

**Recommendation: option B, as a lifecycle and operational-safety experiment only**, and only if the storage limits are
decided. Reasons, from the measurements:

1. It is the largest step that keeps every frozen time and memory limit with margin, so it tests those limits prospectively
   instead of replacing them (§3).
2. It halves the extrapolation to the full trace and gives the first indication of curvature (sub- or super-linear), which is
   what the full-E1 flush deadline and importer ceiling depend on most.
3. It exercises storage enforcement, stream preservation and recovery at twice the verified size at a cost of about 37 min of
   GPU-held time, against 58 – 76 min for a first attempt at C under limits no run has tested.

The case is moderate, not decisive. Against B: it yields no scientific data; one more point does not validate a law; and the
one observation at full size (flush completed in 865 s, import proceeding with 61 GB) suggests the full trace is physically
achievable, the earlier failure having been a launcher bound, not the profiler. If the reviewer prefers to size a full-E1 packet
directly from the stress column and accept a longer GPU hold, option A / C is defensible; the price is that every limit in that
packet is then untested and wide.

Unresolved risks:

| # | risk | applies to |
|---|---|---|
| R1 | the full trace's row count has never been measured; all C figures scale with it | C |
| R2 | flush or import super-linear beyond 24.5 M rows; a forced release during an unfinished flush loses the trace (as in e1_main_v3) | B (small), C |
| R3 | importer memory at full size between 81 and 235 GiB; the approved ceiling is 128 GiB; offline recovery needs the same memory | C |
| R4 | the supervisor's response has only been exercised on mock processes | B, C |
| R5 | a new trace and config bring a new post-run path; V1 and V3 both failed on preparation defects of that kind, not on the capture | B, C |
| R6 | a long flush or a large export stalling a sampler for 60 s would trigger a false abort | C, possibly B |
| R7 | server host footprint with 96k contexts unknown | C |
| R8 | between-session drift (about 5 % in Stage 1.5) limits how well timings from separate runs can be compared | interpretation of B |

Decisions required from review (none taken here):

- D1. Whether to run an intermediate experiment at all (B) or to go to a full-E1 packet (A / C).
- D2. Storage limits for B. Indicative values, stress case plus 25 %: temporary 15 GB, run root 18 GB, run root with archive
  38 GB. These are proposals, not changes.
- D3. Whether the reservation may be deleted at verified GPU release instead of after closeout (irrelevant for B's feasibility,
  decisive for C's reservation length).
- D4. Whether all other V3 limits carry over to B unchanged (the draft assumes so).
- D5. Workload of B: the accepted trace four times (draft), or new cells with a different, explicitly non-scientific purpose.
- D6. Whether a CPU-only measurement of rows per iteration per cell from the preserved V2 / V3 exports is wanted before any
  packet (it would test, for b=8 against b=48 only, the assumption behind the 216 M estimate).
- D7. For C later: deadlines, GPU-held ceiling, importer ceiling and storage limits, after B's result.

A draft protocol for B is in `E1-V4-INTERMEDIATE-PROTOCOL-DRAFT.md`. It is a draft for review, not a packet.

## 7. Checks run for this analysis (CPU-only)

See `checks_tray16.log`: guarded tests of the model, frozen manifests, V3 archive against its `SHA256SUMS`, stream hash,
unchanged V3 record and identity files, GPUs idle before and after.
