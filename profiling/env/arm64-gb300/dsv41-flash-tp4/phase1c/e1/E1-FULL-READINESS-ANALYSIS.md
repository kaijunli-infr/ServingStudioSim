# Full E1 readiness analysis from the V2 and V3 measurements (CPU-only, 2026-11-18)

Analysis only. It is not an execution packet and approves nothing; no identification partition was created and no parameter
was fitted. Every figure is marked **M** (measured, with its source), **E** (an estimate computed from measurements; the
arithmetic is given) or **A** (an assumption that has not been verified).

Target: the full E1 trace `traces/e1_main.csv`, 450 requests in 36 waves over 940.6 s (M, `E1-MAIN-V3-INPUTS.json`), 24 cells
(batch 1–48 × KV 4k / 32k / 96k). Its kernel-row count has never been measured: the one attempt (e1_main_v3) lost its trace. The
earlier plan figure is about 216 M rows (E, from the incident analysis); every size-dependent estimate below inherits that
uncertainty. Scale factor against V3: 216 / 24.5 = 8.8.

## 1. What has been measured

| quantity | V2, 12.25 M rows | V3, 24.50 M rows | other points |
|---|---|---|---|
| flush (stop → importing) | 61.9 s (M; V1: 63.8 s) | 112.0 s (M) | e1_main_v3: 865 s (M: stop 09:37:36, import began 09:52:01; ≈ 216 M rows, E) |
| import | ≈ 28.9 s (E: phase start to report mtime; importer not observed) | 53.0 s (M, importer process observed) | e1_main_v3: 49 % of the progress bar after 217 s, then destroyed (M) |
| stop → report | 90.8 s (M) | 166.2 s (M) | 45 s at 5.9 M; 63 s at 6.4 M; 160 and 163 s at 25.6 M (Phase 0 recipe); 567 s at 97.2 M (Phase 0 recipe) (all M) |
| offline import of the stream | 26 s (M) | 56 s (M) | — |
| raw stream | 348 MB, 28.4 B/row (M) | 662 MB, 27.0 B/row (M) | — |
| report | 392 MB, 32.0 B/row (M) | 792 MB, 32.3 B/row (M) | 2.83 GB at 97.2 M, 29 B/row (M) |
| in-session export | 1.96 GB, 160 B/row (M) | 3.95 GB, 161 B/row (M) | 3.82 GB at 25.6 M (M) |
| run root | 3.2 GB with tables (M) | 6.3 GB (M) | — |
| temporary peak, post-run | 2.6 GB (M) | 5.2 GB (M) | includes 0.7 GB of retained older streams |
| importer resident memory | 4.9–8.7 GiB, offline only (M, 1 s sampling) | 15.6 GiB in session and offline (M) | 61 GB sampled once at ≥ 49 % of the e1_main_v3 import (M) |
| host MemAvailable | not measured | 858 GiB idle → 573 GiB five minutes after launch → 547 GiB minimum, during the import → 861 GiB after release (M) | — |
| GPU release after the server SIGTERM | ≤ 2.12 s (M, forced release) | ≤ 2.86 s (M, normal shutdown) | — |
| server exit after SIGTERM | 60 s, by SIGKILL, with the stop request open (M) | 2.7 s (M, request already answered) | — |
| supervisor | — | 2463 checks over 41 min, no cause, heartbeats < 2 s (M) | — |

## 2. Flush and import at full size

| estimate for ≈ 216 M rows | value | how |
|---|---|---|
| flush | 895 – 990 s (E) | line through the V2 and V3 points: 112.0 + (112.0 − 61.9) / 12.25 M × 191.5 M = 895 s; proportional to V3: 112.0 × 8.82 = 987 s |
| flush, observed once | 865 s (M) | e1_main_v3 |
| import | 430 – 470 s (E) | line through V2 and V3: 430 s; proportional: 467 s. e1_main_v3's 49 % after 217 s gives ≈ 443 s if the bar is linear (A) |
| stop → report | 1300 – 1470 s (E), 22 – 25 min | sum of the above; proportional 166.2 × 8.82 = 1465 s |

- The two-point line and the single large observation agree to within 15 %. That is three sizes of two different traces, and
  the largest was read from a log, not from the controller. It is not a scaling law.
- Whether the flush has a fixed component is not known (A). V1 and V2 gave 62–64 s at one size; V3's 112 s at twice the size is
  1.81 ×, so a fixed part of at most about 12 s is consistent with a line, and a pure proportional law is also within reach.
- The 97.2 M point (567 s total) is lower than a proportional scaling of V3 would give (659 s); it used the Phase 0 recipe.
- The import bar is not evidence of progress (V2); only the importer's own counters are. A linear bar is an assumption.

Consequence: the present deadlines do not fit a full E1. Flush 300 s, import 300 s and stop → report 600 s were sized for
V3; a full E1 needs roughly 900 – 1000 s, 450 – 500 s and 1400 – 1500 s before any margin (E). Choosing the margins is a
packet decision, not made here.

## 3. GPU-held time

| component | value | basis |
|---|---|---|
| startup | 1487 – 1640 s | M: V2 1487 s, V3 ≈ 1640 s (launch to measurement begin) |
| capture | ≈ 900 – 940 s | M: e1_main_v3 replayed 450 requests in 896 s; trace span 940.6 s |
| stop → report | 1300 – 1470 s | E, §2 |
| shutdown | ≈ 3 s | M |
| total | ≈ 3700 – 4050 s, 62 – 68 min | E |

The 2700 s ceiling and a 90-minute reservation with pre-launch checks and post-run inside it do not fit: V3 used 60 minutes of
reservation for 31 minutes of GPU-held time (M). The controller enforces its ceiling and deadlines only after the stop request
(M, controller test); before it the supervisor does (M, rehearsal on mock processes; no real cause has occurred yet).

## 4. Host and importer memory

- Server footprint: MemAvailable fell by 285 GiB within five minutes of the launch and stayed there (M). Whether it grows with
  the larger KV cells of the full trace (96k context) is not known (A: the server arguments are the same, so the footprint is
  expected to be similar).
- Importer: 15.6 GiB at 24.5 M rows (M). Proportional scaling gives about 138 GiB at 216 M (E: 15.6 × 8.82), from the V2-size
  maximum about 153 GiB (E). The only large observation, 61 GB at ≥ 49 % of the import, is below both, and it is not known
  whether memory was still rising at that point (A).
- The importer ceiling approved for V3 is 128 GiB. A full E1 could reach or exceed it (E). Raising it needs separate review; on
  the host side there is room: 547 GiB were available at V3's minimum, so an importer of 150 GiB would leave about 410 GiB (E),
  above the 64 GiB floor.
- The in-session importer's memory rose to its peak and fell to about 7 GiB before it exited (M); the 1 s sampler read a
  lower bound of the true peak.
- NUMA: two nodes of about 479 GiB (M). An importer above roughly 190 GiB would not fit beside the server on one node (E);
  the effect on import time is unknown (A).

## 5. Storage

| item | at ≈ 216 M rows | basis |
|---|---|---|
| raw stream | ≈ 5.8 GB | E: 27.0 B/row (V3) |
| report | ≈ 6.3 – 7.0 GB | E: 29 – 32.3 B/row |
| in-session export | ≈ 35 GB | E: 161 B/row |
| parsed outputs, logs, routing records | ≈ 8 GB | E: 0.88 GB at V3 × 8.82; the routing records scale with requests (450 vs 112), not rows (A) |
| run root | ≈ 55 GB | E: sum |
| post-run scratch (offline report + export) | ≈ 42 GB | E; temporary peak with the stream ≈ 48 GB plus retained older streams |
| archive | ≈ 55 GB more; run root + archive ≈ 110 GB | E |
| free space | `/tmp` 1163 GB, CephFS 20 771 GB | M |

The V3 ceilings (10 GB temporary, 8 GB run root, 16 GB with archive) are an order of magnitude below a full E1. Space is not
the constraint; the ceilings and the controller's disk floors (50 GiB stream directory, 20 GiB report directory) would have to
be restated. Export time has not been measured separately: in V3's post-run the offline export took about 130 s after a 56 s
import (M, from the storage samples); proportional scaling gives about 19 min for the export and 8 min for the import (E).

## 6. Stream preservation and offline recovery

- Preservation by hard link, final hash after session close, durable copy, and offline import equal to the in-session export on
  ten checks: verified at 12.25 M and 24.5 M rows, after a forced release (V2) and a normal stop (V3) (M).
- Not verified at full size (A): that the importer completes on a 5.8 GB stream; that the offline import's memory matches the
  in-session one (it did at V3: 15.6 GiB both, M); that the controller's stream-directory scan and the hard link behave the same
  when the flush lasts 15 minutes. In e1_main_v3 the stream was deleted when the Nsight CLI was interrupted; preservation did
  not exist then, so there is no observation of a preserved stream of that size.
- A forced release during a 15-minute flush, before the stream is announced, has never been exercised (A). V2 forced the
  release 1.2 s into the import, after the stream existed.

## 7. Supervisor and observers at larger size

- Cost does not grow with trace size: one small JSON read, three file stats and two marker checks per second (M: 2463 checks
  in V3). The storage sampler walks the run root every 2 s; the file count grows with requests (about 450 routing arrays against
  112), not with rows (E).
- Heartbeat rule: a sampler silent for 60 s is an abort cause. In V3 the largest gaps were 1.5 s (memory) and 2.9 s (storage)
  (M). Whether a 35 GB export or a 15-minute flush on the same filesystem can stall a sampler for 60 s is unknown (A); a false
  abort during finalisation would cost the run but not the stream.
- The supervisor has never had to act against a real server (A for its response path; M on mock processes only).
- Importer identification worked in the real session (M). The probe's settle rule and the 0.5 s interval are independent of size.
- The GPU sampler's rule (upper edge of the reading window) is independent of size.

## 8. Scientific completeness for the full trace

- The completeness and routing checker has passed on two cells (V1, V2) and four cells of the same two kinds (V3) on real
  data (M), and on a synthetic four-wave structure. The full plan has 24 cells in 36 waves, including batch 1, 2 and 4 with two
  waves each and KV lengths of 4k and 96k; none of those cells has been captured under the lifecycle controller (A).
- The per-cell requirement is at least 400 pure-decode iterations (M: checker rule); whether every cell of the full plan reaches
  it was computed in the plan (`pure_decode_iters_min` in `E1-MAIN-V3-INPUTS.json`) but only observed on the surviving host-side
  records of e1_main_v3, not on a trace (A for the trace side).
- The configuration-identity policy needs a trace rule for the full trace. The V3 defect was exactly an identity rule that had
  not been exercised against the new trace; any next packet has to run every post-run step against the new config in rehearsal.
- V3's row count was exactly twice V2's (24 501 664 = 2 × 12 250 832, M), so rows per iteration are reproducible for these
  two cells. Rows per iteration for the other 22 cells are not measured (A); the 216 M figure depends on them.
- Capture overhead at the full trace size was assessed in Stage 1.5 on small traces only (M there; A here).

## 9. What would have to be decided before a full E1 packet (not decided here)

1. Deadlines for flush, import and stop → report, with stated margins over §2.
2. A GPU-held ceiling and reservation length consistent with §3.
3. An importer-memory ceiling, given §4, and whether it may exceed the 128 GiB approved for V3.
4. Storage ceilings and disk floors consistent with §5.
5. Whether an intermediate size between 24.5 M and 216 M rows is wanted first: the flush, import and memory estimates rest on two
   controlled points and one uncontrolled observation.
6. A trace-identity rule and a rehearsal of the complete post-run against the full config.
