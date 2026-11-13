# Phase 1C-0 report — CPU-only preparation (reproducibility, invariants, uncertainty audit, plans)

Branch `phase1c-prep`, one commit on Phase 1B `29082793` (`phase1b-seqsim`, accepted 2026-11-13 as a CPU-only research
implementation). Scope as approved: freeze Phase 1B, make its outputs reproducible without relying on the tray14 copies,
add sequence-model invariant checks, audit the uncertainty, keep the small-batch and eager classes lower-confidence,
prepare the GPU experiment matrix, freeze the validation protocol, and design (not implement) the Rust integration.
**No GPU capture, Rust change, database write or preset change was made.**

## 1. Freeze

| artefact | identity | state |
|---|---|---|
| Phase 0 | `test-ss` ca9f8ed0 | unchanged |
| Phase 1 planning | `phase1-planning` cc06d9c2 | unchanged |
| Phase 1A | `phase1a-costmodel` a7725255; coefficients 78f937c5…, partition 288e193f… | unchanged (hashes pinned in `tests/test_invariants.py`) |
| Phase 1B | `phase1b-seqsim` 29082793; params 8dc79573…; engine sources hashed in `phase1b/out/results.json["code"]` | unchanged (pinned); predictions reproducible bit-for-bit (section 2) |
| this phase | `phase1c-prep`, directory `profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/` only | `git diff 29082793 HEAD -- . ':!…/phase1c'` is empty (asserted by a test) |

## 2. Reproducibility of the Phase 1B outputs

`repro/MANIFEST.json` records: the three compact-table inputs per file (12 files, sha256 and bytes, plus each
`meta.json` with the extractor's own input hashes), the frozen coefficients / partition / params hashes, the hashes of
all `seqsim/*.py` and `costmodel/*.py` sources, the environment (Python 3.12.13, numpy 2.5.4, pyarrow 26.0.1, aarch64,
Linux 6.14 nvidia-64k), the exact command, seed 0 and 32 draws, the output schema (every parquet column with its
Arrow type), expected row counts (17,686 / 4,750 / 4,687 rows per pass, 17,637 / 4,740 / 4,667 scored), a canonical
content checksum per parquet (encoder-independent: rows sorted by iteration, columns sorted, floats as `repr`), and the
sha256 of `results.json`, `RESULTS.md`, `params.json`. Provenance: the committed Phase 0 `SHA256SUMS` of the derived
artefacts and the raw-capture location (34 GB, sha256-verified in Phase 0).

`repro/verify.py` checks every input hash, regenerates the full Phase 1B run from the frozen inputs into a scratch
directory (fit of the sequence parameters included), and compares: `params.json` sha256, canonical checksums, row
counts and column types of all 12 parquet files, and **every metric** in `results.json["passes"]` with zero tolerance.

Result (`out/logs/verify.log`, `repro/VERIFY.json`): 30 checks, all ok, regeneration 162 s, `VERIFY PASS` — the
regenerated per-iteration outputs are canonically identical and every reported metric is bit-identical.

Second storage location (not tray14): rack 3 `/mnt/shared/home/kevin/dsv41-phase1-mirror/` holds the compact tables
(175 MB) and the Phase 1B parquet outputs (15 MB) with `SHA256SUMS.mirror` (24 files, verified after the copy). The
tables themselves are regenerable from the Phase 0 raw captures with the Phase 1A extractor (hash-recorded).

## 3. Sequence-model invariant checks (`tests/test_invariants.py`, 14 tests, all pass)

| required check | tests |
|---|---|
| collective completion and rank synchronisation | random-skew scalar reference of `critical_path_batch` (completion = max arrival + intrinsic, waits ≥ 0, never a rank sum); engine-level: rank-asymmetric windows → all ranks leave each collective together, waits equal the arrival gaps, the slowest never waits |
| host/GPU ordering and launch-queue depth | with Q = one iteration of launches the host cannot finish launching chunk i before the GPU finished chunk i−1 (per rank); with an unbounded queue it does; GPU order per rank across iterations; random schedules satisfy every ordering rule (host serial, broadcast, f_idle, scheduler one-ahead) |
| cross-iteration host lead | lead = GPU end of i−1 − earliest host start of i; after decode it is positive and capped by wall(i−1) − eps_sched; host-bound eager leaves a GPU idle gap and binds the serial rule |
| critical-path reconstruction | the engine's per-rank ends and waits equal a scalar re-composition of the same segment inputs; decode wall lies between the slowest rank path and the rank sum, exceeding the slowest path only by the carried-over start skew |
| graph versus eager scheduling | decode (CUDA graph) wall independent of the per-launch host cost; host-bound eager wall strictly increasing with it; GPU-bound eager independent; kind thresholds (P + b ≤ 256 graph, ≥ 8192 ring) |
| stream overlap and non-overlap | measured compute/communication overlap shortens the windows by exactly the measured amount and never the collectives; within a rank windows and collectives are serial (wall = Σ windows + Σ collectives for symmetric ranks); across ranks the work is parallel |
| deterministic Monte Carlo replay | same seed → bitwise identical timelines; causality (the first n iterations do not depend on later ones); different seed differs; the median path is seed-independent; the mean of independent symmetric draws lies above the median path by a bounded amount that shrinks with the dispersion |
| missing or unsupported configurations | b outside the tables → status ≠ in-range; kv beyond range → kv-extrapolation; unseen previous-b / host windows → non-measured status; graph-kind mixed without a class → flagged; excluded iteration without a measured timeline or a timeline with the wrong rank count → refused |
| preservation of existing simulator behaviour | frozen input and source hashes unchanged; repository outside `phase1b/` + `phase1c/` identical to 29082793; no repository module references `seqsim`/`phase1b` (opt-in) |
| real frozen inputs | all sequence invariants hold on the real coefficients and parameters in both modes on a mixed synthetic schedule; steady-state decode b=8 within the table's range; chain leads grow then saturate; bitwise replay |

## 4. Uncertainty audit (`uncertainty/RELIABILITY.md`, `uncertainty/FINDINGS.md`)

pred-mc with 32 draws and seeds 0–4 on the three passes; frozen inputs; no parameter changed, no band widened.
Per class: n, signed bias, median and p90 absolute error, p10–p90 coverage, mean and median interval width, fraction
below/above, and the seed-to-seed standard deviation and range of bias and coverage. Headline rows (seed 0, held-out):

| pass | class | n | bias % | median abs err ms | p90 abs err ms | coverage p10–p90 % | width mean / median ms | below / above % | seed std of bias % / coverage pts |
|---|---|---|---|---|---|---|---|---|---|
| run2 | decode b3-16 | 6935 | +0.52 | 0.063 | 0.154 | 57.8 | 0.176 / 0.158 | 30.4 / 11.8 | 0.013 / 0.38 |
| run2 | decode b1-2 | 471 | +2.05 | 0.270 | 0.793 | 84.9 | 1.30 / 1.26 | 8.9 / 6.2 | 0.13 / 1.5 |
| run2 | prefill 16k chunk | 33 | +0.02 | 0.649 | 1.80 | 100 | 8.2 / 8.2 | 0 / 0 | 0.03 / 1.7 |
| run2 | mixed-eager (host-bound) | 3 | −1.76 | 5.29 | 12.6 | 0 | 0.11 / 0.14 | 33 / 67 | 0.004 / 0 |
| quadrant | decode b3-16 | 78 | +2.49 | 0.124 | 0.206 | 17.9 | 0.21 / 0.19 | 82 / 0 | 0.06 / 2.0 |
| quadrant | decode b17-24 | 49 | +1.34 | 0.119 | 0.202 | 51.0 | 0.25 / 0.24 | 47 / 2 | 0.04 / 4.7 |
| quadrant | decode b>=25 | 1071 | +2.19 | 0.231 | 0.416 | 38.5 | 0.42 / 0.37 | 59 / 2 | 0.05 / 0.7 |
| quadrant | decode b1-2 | 53 | +1.44 | 0.235 | 0.582 | 96.2 | 1.59 / 1.58 | 1.9 / 1.9 | 0.45 / 2.3 |
| salvaged | decode b3-16 | 1967 | +0.40 | 0.058 | 0.163 | 69.4 | 0.28 / 0.17 | 24 / 6 | 0.03 / 0.15 |
| salvaged | decode b17-24 | 32 | +4.91 | 0.374 | 0.463 | 0 | 0.13 / 0.14 | 100 / 0 | 0.02 / 0 |

Seed variation is negligible for every class with n ≥ 50 (bias std ≤ 0.13 points, coverage std ≤ 2.3 points); with
n ≤ 14 coverage moves by whole iterations (up to ±50 points) and is not interpretable.

**Interval displacement, decomposed** (variants switch one stochastic component at a time; see FINDINGS.md):

| pass, class (held-out) | V0 median path | V1 + per-(rank,segment) draws | V2 + common-mode dispersion | V3 + MoE band (= pred-mc) |
|---|---|---|---|---|
| quadrant b3-16: bias % / coverage % | −1.18 / — | +2.53 / 2.6 | +2.48 / 11.5 | +2.49 / 17.9 |
| quadrant b≥25 | −0.89 / — | +2.20 / 8.7 | +2.18 / 31.7 | +2.19 / 38.5 |
| quadrant b17-24 | −1.06 / — | +1.30 / 12.2 | +1.23 / 46.9 | +1.34 / 51.0 |
| run2 b3-16 | −2.83 / — | +0.62 / 21.1 | +0.62 / 46.9 | +0.52 / 57.8 |
| run2 b1-2 | +0.48 / — | +1.50 / 76.2 | +1.97 / 87.3 | +2.05 / 84.9 |

The entire displacement between the median path and the stochastic mean (+3.1 to +3.7 points on every decode class)
comes from step V0→V1: independent per-(rank, segment) quantile draws composed through the max-over-ranks barrier.
The common-mode dispersion factor and the MoE routing band change the mean by < 0.1 point; they only widen the band
(coverage 2.6 % → 17.9 % on quadrant b3-16, 21 % → 58 % on run2 b3-16). Routing variability is therefore **not** the
source of the displacement; the MoE band contributes 0.6–2.2 % of relative band width and nothing to the mean. Distribution shift between passes is
visible as the different position of the measured median relative to the two model paths: run2 sits 2.8 % above the
median path and 0.6 % below the independent-draw mean, quadrant sits 0.9–1.2 % above the median path and 2.2–2.5 %
below the independent-draw mean; the salvaged b17-24 rows (table built from quadrant calibration rows) sit 4.9 % below
the stochastic mean. Host-arrival variation: the measured first-collective arrival spread across ranks is 0.016–0.02 ms for b ≥ 3 in every pass (negligible) and 0.45–0.9 ms for b = 1–2 (the rank-lateness pattern, hence the wide b1-2 band). Parameter uncertainty: table sample sizes are 664–2,425 for the main batch sizes and seed variation is ≈ 0, but batch sizes without their own table (b = 5, 10, 14, 15, 26, 28–47; n = 0–5) receive interpolated quantiles whose bands are 10–40× wider (artefact, affects a few hundred quadrant iterations).

Mechanism (`uncertainty/wait_decomposition.py`, FINDINGS.md §3): the Phase 1A tables carry the *measured* mean
collective wait per rank (0.1–0.25 ms per iteration). The median path produces almost none of it (ranks at their medians
are aligned), which is most of its −1 to −3 % bias; independent per-rank draws produce slightly too much at b ≤ 16
(the measured wait is 70–80 % of the way from common-mode to independent) and far too much at b ≥ 24 (25–30 %), which is
exactly the run2 (+0.5 %) vs quadrant (+2.2 %) pattern. Whether the rank-independent fraction depends on b (different
kernel path: 163 vs 86 collectives) or on the pass cannot be separated with these three passes, since every b ≥ 17 row is
from the quadrant pass; experiment E1 removes that confound. No band was widened and nothing was recalibrated on the
frozen held-out rows; the two paths continue to bracket the measurement by ≈ 3 %.

## 5. Small-batch and eager classes

Decode b1–2 and host-bound mixed eager stay **lower-confidence**: they carry `confidence: low` in the proposed response
contract (`schema/seqsim_response.schema.json`) and a `seq_low_confidence` flag in the proposed `CoverageFlags`
extension; the GPU matrix (E1/E3) and the protocol's acceptance table treat them separately. No value was introduced
for rank-2 host lateness or the KV-dependent launch cost; both remain "not identifiable" with the Phase 1B carrying
rules. The inferred-parameter sensitivity analysis of Phase 1B (`phase1b/tools/sensitivity.py`, queue throttle and
delta_launch) is retained unchanged and referenced from the protocol.

## 6. GPU experiment matrix, validation protocol, Rust interface

- `GPU-EXPERIMENT-MATRIX.md`: six captures (E1 decode routing + rank arrival, E2 decode validation, E3 host-bound eager
  at long KV, E4 chunk chains to 160k, E5 graph-path mixed sweep, E6 cold start / first chunk), each with the unresolved
  question, configuration, why the existing traces cannot answer it, measurements and attribution, designation,
  occupancy and reservation, storage budget, stop conditions and the decision it enables; ranked by expected reduction
  of model uncertainty (decode ≈ 79 % of the workload total; the 3 % median/mean spread is the dominant term).
  ≈ 7.5 tray-hours and ≈ 115 GB in total if all are approved. **None is authorised; none was run.**
- `VALIDATION-PROTOCOL.md` (sha256 in `PROTOCOL.sha256`): designations fixed before capture, a new partition file for
  new captures, the Phase 1A held-out rows kept unchanged and no longer counted as validation, independence of
  identification and validation captures, one-shot pre-registered comparisons with acceptance thresholds, reporting
  template, prohibitions.
- `RUST-INTEGRATION-INTERFACE.md` + `schema/*.json`: hook at `IterBatchWorker::start_iteration`, a stateful
  `SequenceModel` trait behind `#[serde(default)] sequence_model: Option<…>` on the DSV4.1 arch config, request/response
  contracts, supported regimes with refusal of everything else, status/uncertainty propagation through `CoverageFlags`
  and new optional cost-log columns, immutable versioned parameter bundles, preset compatibility (public preset untouched;
  it targets B200), and the regression/rollback test list (flag-off bit-identical goldens, contract parity, invariants,
  refusal, kill switch). No Rust code was changed.

## 7. CPU test results and exact commits

| check | result | log |
|---|---|---|
| `phase1c/tests` invariants | 14 passed | `out/logs/pytest_invariants.log` |
| `phase1b/tests` (unchanged sources) | 17 passed, 1 failed **by design on this branch**: `test_phase1a_and_simulator_sources_untouched` asserts that nothing outside `phase1b/` changed since a7725255, and `phase1c/` now exists; it passes on 29082793. The Phase 1C equivalent (`test_repository_outside_phase1b_and_phase1c_is_untouched`) passes. The Phase 1B sources are frozen, so the test was not edited | `out/logs/pytest_phase1b.log` |
| reproducibility verification | 30/30 ok, VERIFY PASS, metrics bit-identical | `out/logs/verify.log`, `repro/VERIFY.json` |
| reliability audit | completed (467 s) | `out/logs/reliability.log` |
| collective-wait decomposition | completed | `out/logs/wait_decomposition.log` |
| repository CPU pytest tier | not re-run in this phase: the diff outside `phase1c/` is empty (asserted) and `phase1c/` is not collected by the repository's pytest configuration; Phase 1B's run on the same tree: 5,063 passed, 2 pre-existing environment failures | Phase 1B report |

Commits: Phase 1B `29082793` (base); Phase 1C-0: `1a648b13` (this directory) followed by a one-line report amendment (`git log 29082793..phase1c-prep`).

**Stop for review.** No GPU execution and no production integration until the experiments are individually approved.
