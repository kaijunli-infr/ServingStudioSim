# Stage 1.5 remediation report (Phase 1.5, CPU-only) — 2026-11-14/15

Branch `phase1c-e1-stage15` (from Stage 1 c7e8ed9b). No GPU was used. Stage 1 evidence is untouched (item A): raw runs,
logs, manifests, SHA256SUMS, the frozen checker `checks/e1_completeness.py`, `E1-STAGE1-REPORT.md`, `runs/e1_dry_*`,
`runs/stage1/`, `configs/`, `traces/`, `INPUTS.sha256` have no diff against c7e8ed9b. The original b=48 point-estimate PASS is
not reinterpreted; the interval criterion is recorded as prospective (E1-OVERHEAD-METHODOLOGY.md §2).

## Deliverables (item G)

| # | deliverable | files |
|---|---|---|
| 1 | this report | `E1-STAGE15-REMEDIATION.md` |
| 2 | versioned checker + regression tests | `checks/e1_completeness_v2.py` (2.0.0), `tests/test_e1_completeness_v2.py` (20 tests); old vs new on Stage 1: `runs/stage15/checker_v1_vs_v2.md` + `runs/stage15/cpu/v{1,2}_{off,on}.{txt,json}` |
| 3 | streaming configuration fix and verification | `patch/apply_stream_optin.py` (launcher `workload.stream`), `patch/test_alignment_stream_optin.py` -> `tests/` (5 tests), `patch/apply_stream_flag_rs.py` + `patch/req-frontend-stream-flag.patch` (`session_runner --stream`, cargo tests green, binary at `$S/bundles/session_runner.stream-flag.2cd7990+patch`), `checks/e1_config_identity.py` + `tests/test_e1_config_identity.py` (5), `make_e1_inputs_v2.py` -> `configs_v2/`, `INPUTS_v2.sha256`, `E1-STAGE15-INPUTS.json`; identity results in `runs/stage15/cpu/config_identity_*` |
| 4 | autotuner investigation | `analysis/AUTOTUNER-INVESTIGATION.md`, `analysis/autotune_cache_diff.py`, `runs/stage15/cpu/autotune_cache_diff_stage1.md` + the two Stage 1 cache files |
| 5 | off/off comparison protocol | `E1-OFFOFF-PROTOCOL.md`, `analysis/e1_pair_compare_v2.py`, `runs/stage15/cpu/pair_v2/` |
| 6 | short Nsight validation protocol | `E1-NSYS-VALIDATION-PROTOCOL.md`, `checks/e1_nsys_validation.py`, `configs_v2/e1_nsys_val/`, `runs/stage15/cpu/nsys_validation_quadrant.{txt,json}` |
| 7 | updated budgets | `E1-STAGE15-BUDGET.md` |
| 8 | GPU authorization request | `E1-STAGE15-GPU-AUTHORIZATION-REQUEST.md` |
| | drivers | `preflight_e1_stage15.sh`, `run_e1_stage15.sh`, `collect_e1_stage15.sh` (Stage 1 scripts unchanged) |

## B. Checker correction — what changed and what it does to Stage 1 verdicts

v2 fixes the two v1 defects reported in Stage 1 §5b and adds structural checks (details and every changed verdict in
`runs/stage15/checker_v1_vs_v2.md`):

* KV tag `kv32k` = 32 768 tokens (v1: 32, cell sub-check silently skipped) -> per-cell minimum now evaluated: PASS on both arms.
* Join: the pre-registered rule itself (every (request, position) emitted in exactly one iteration, increasing, corpus row ==
  route), with ramp vs pure-decode accounting (v1 demanded pure-decode count == corpus rows and failed 54/56 on correct data)
  -> `e1_dry_on` overall FAIL -> PASS. This is the only verdict that changes.
* New: planned-cell-absent FAIL, unplanned-cell FAIL, untagged ids, routing record structure (count, shape, dtype, expert
  range, duplicate top-k, segment coverage/uniqueness, npy[1:] == segment), corpus size and checksum.
* Plan selection (`--plan auto|dry|main`): a draft that merged both plans wrongly demanded a main-plan cell of a dry run; found
  by running on Stage 1 data, fixed, tested.

Tests: 34 new tests pass; full suite 5129 passed / 7 failed, the 7 identical on the frozen base 5c86ae1c (pre-existing,
unrelated: `test_gdn_spec_decode`, `test_l1_single_gemm` energy, public presets).

## C. Confounders

* **Wire mode.** Root cause in `req-frontend/src/backend/client.rs` (`stream: !capturing_routes`). Fix: explicit
  `workload.stream` + `--stream`; all Stage 1.5 configs say `stream: false`; identity check PASS between arms; the check also
  compares launch records after a run (argv/env/server_config) and flagged Stage 1 correctly (wire mode + PATH ordering).
* **Autotuner.** Every launch re-tunes for ~19 min because the loaded cache is ignored in tuning mode for ops that request
  cold-L2 profiling (flashinfer `search_cache`); the two arms also use different cache directories (hook flag in the config
  hash); 86 of 159 winning tactics differed between the two Stage 1 launches. Recipe-preserving remedy: none; the off/off
  control quantifies the effect; archive and diff cache files per launch (driver does). Options needing a code change are
  listed, none recommended now (AUTOTUNER-INVESTIGATION.md §4).
* **Off/off control** prepared as pass 3 with order off, on, off2 (E1-OFFOFF-PROTOCOL.md).

## D. Short Nsight validation

Designed as one bounded hook-on nsys pass on the dry trace (~30 min GPU, ~2.4 GB); checker covers NVTX ownership and
completeness, GPU-window alignment per rank, collective end spread per cell (packet §4.3 thresholds), joins (via v2),
timestamp consistency, export correctness. Run on Phase 0 quadrant data: PASS, 2.9 s. Not run on GPU.

## E. Methodology

Gate A (median ratio <= 1.02) preserved. Gate B pre-registered for future runs: moving-block bootstrap interval with block
length from the integrated autocorrelation time (`L = clip(ceil(5 tau), 10, n/20)`), PASS/INCONCLUSIVE/FAIL, drift
diagnostics; off/off noise flag and excess reporting fixed in advance; no threshold tuned on Stage 1
(E1-OVERHEAD-METHODOLOGY.md). Stage 1 illustration: b8 would be PASS, b48 INCONCLUSIVE — recorded, not adjudicated.

## F. Budget

Per launch ~26 min GPU-held (19 min autotune); Stage 1.5 total ~2 h 05 min nominal, 4 h reservation requested, ~3 GB storage
(E1-STAGE15-BUDGET.md). Full E1 capture excluded.

## Open items for the review

1. Rust `--stream` flag: install the patched binary / re-apply the patch in the launcher tree before any go; decide on
   pushing the req-frontend commit and bumping the gitlink.
2. Stage 1 §5f (`EngineDeadError` at hook-on teardown) not investigated here (not in scope A-G); still worth a look before Stage 2.
3. Authorization: E1-STAGE15-GPU-AUTHORIZATION-REQUEST.md.
