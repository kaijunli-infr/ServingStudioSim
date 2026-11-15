# Checker v1 vs v2 on the preserved Stage 1 artifacts (Phase 1.5, item B)

Inputs (unchanged, read-only): `$S/ServingStudioSim/logs/20261114_dsv41_flash_gb300_tp4_e1/{e1_dry_off,e1_dry_on}/profile`
(raw copy + `SHA256SUMS` in `$S/artifacts/phase1c-e1-dsv41-flash-gb300-tp4-20261114/`), `E1-INPUTS.json`.
Checkers: v1 = `checks/e1_completeness.py` (frozen with the packet, 5c86ae1c; **not modified**), v2 = `checks/e1_completeness_v2.py`
(VERSION 2.0.0, tests `tests/test_e1_completeness_v2.py`, 20 tests). Outputs of both runs are beside this file
(`v1_off.txt`, `v1_on.txt`, `v2_off.txt`, `v2_on.txt`, `*.json`). Commands (CPU, tray14, worktree):

```
uv run --no-sync python $E1/checks/e1_completeness.py    --profile $RUN/e1_dry_on/profile --inputs $E1/E1-INPUTS.json --compare $RUN/e1_dry_off/profile/vllm/dsv41_flash_1017_gb300_tp4_e1_dry_off_metrics.jsonl
uv run --no-sync python $E1/checks/e1_completeness_v2.py --profile $RUN/e1_dry_on/profile --inputs $E1/E1-INPUTS.json --compare $RUN/e1_dry_off/profile --checksum
```

## Verdicts

| run | v1 | v2 |
|---|---|---|
| e1_dry_off (hook off) | PASS (4 PASS, 2 INFO) | PASS (13 PASS, 3 INFO) |
| e1_dry_on (hook on, `--compare` off) | **FAIL (1)**: "requests whose pure-decode iteration count != corpus rows: 54 (join check)" | PASS (32 PASS, 1 INCONCLUSIVE non-gating, 3 INFO) |

## Every changed verdict and why

1. **e1_dry_on overall: FAIL -> PASS.** v1's join sub-check demanded `pure-decode iterations of request == corpus rows (1023)`.
   Every request emits 15 to 95 of its tokens in ramp iterations (iterations with prefill present or with fewer/more decode
   requests than the cell's b), so the count of *pure-decode* iterations is below 1023 for 54 of 56 requests on correct
   data. v2 checks the pre-registered rule itself: for every request and position j in 1..1023 exactly one iteration has
   `output_tokens_before == j` and `emitted_tokens == 1` (57 288 pairs: 0 duplicate, 0 missing, 56/56 position->iteration
   maps strictly increasing), and separately accounts pure + ramp emissions (`positions 1..1023 emitted in pure + ramp == 1023:
   all; ramp-emitted per request min/median/max = 15/95/95`). The Stage 1 report (§5b) had already identified this defect
   and the join rule's validity; the verdict change is the removal of a wrong criterion, not a change of data or rule.
2. **Per-cell pure-decode minimum: not evaluated (v1) -> PASS (v2).** v1 parsed the trace tag `kv32k` as 32 and compared
   it with the plan key 32768, so its per-cell sub-check matched no cell and printed nothing (silent skip, no FAIL). v2
   parses `kv{N}k` as N x 1024 (test `test_tag_parses_kv_in_units_of_1024_tokens`, regression `test_v1_tag_parsing_defect_
   is_reproduced`) and reports `cell b=8 kv=32768 wave=0: 1008 (need >= 400)` and `cell b=48 kv=32768 wave=0: 928 (need
   >= 400)` on both arms. The Stage 1 report had taken these counts from `analysis/e1_pair_compare.py` (§4.2); the numbers
   are identical.
3. **Overhead gate: unchanged.** v1 and v2 both print `PASS` for the pre-registered point gate (b8 ratio 1.0086, b48 1.0195,
   limit 1.02). v2 additionally prints the *prospective* operational criterion (moving-block bootstrap 95 % interval, block
   25 here; b8 [1.0059, 1.0105] PASS, b48 [1.0167, 1.0223] INCONCLUSIVE) marked "reported, not gating". Per instruction A, the
   Stage 1 b=48 point-estimate PASS is **not** reinterpreted; the interval is recorded as a criterion for future runs
   (E1-OVERHEAD-METHODOLOGY.md).
4. **New checks in v2 (no v1 counterpart), all PASS on e1_dry_on:** 56 `.npy` with shape (1024, 40, 6), dtype uint16, expert
   ids < 384, no duplicate expert in any (token, layer) top-6; corpus file size == num_tokens x L x K x 2; 56 segments of
   length 1023, contiguous, non-overlapping, covering [0, 57288); segment names == metrics request ids; npy[1:] == segment for
   all 56 requests (row 0 = last prompt token's forward); every segment row maps to exactly one iteration; corpus fnv1a64
   checksum == manifest. v2 also fails on a planned cell that is absent, on an unplanned cell, on untagged request ids, on
   emitted-token counts != 1024 and on progress entries outside {0, 1} (none of these fired).
5. **Development note (recorded for honesty).** The first v2 draft merged the `main` and `dry` plans and, on the Stage 1 dry
   runs, reported `cell b=8 kv=32768 wave=1: pure-decode iterations 0` as FAIL (a main-plan cell the dry plan never
   contained). That was a checker defect found by running v2 on Stage 1 data; v2 now selects the plan the run executed
   (`--plan auto|dry|main`, auto = the plan whose request count equals the requests seen) and checks every cell of that plan
   (`test_plan_selection_by_request_count_and_explicit_choice`). No Stage 1 data or verdict was touched.

## What this means for the Stage 1 record

Nothing in the Stage 1 report changes: the frozen checker's output, its documented defects (§5b), the pair-comparison
tables and the accepted PASS stand. v2 confirms the data are complete under the pre-registered rules and adds structural
checks on the routing records. For Stage 1.5 / Stage 2 runs v2 is the checker to run (both checkers can still be run side by
side; v1 is expected to keep printing its join FAIL on any correct hook-on run).

## Test status

`tests/test_e1_completeness_v2.py` 20 passed, `tests/test_e1_config_identity.py` 5, `tests/test_alignment_stream_optin.py` 5,
`tests/test_alignment_routes_optin.py` 4 (`pytest_new.log`). Full suite in the worktree: 5129 passed, 7 failed, 39 skipped;
the same 7 (`test_gdn_spec_decode` x1, `test_l1_single_gemm` energy x4, `test_public_presets`, `test_public_sim_presets`) fail
identically on the frozen base clone at 5c86ae1c (`pytest_failures_on_base.log`): pre-existing, unrelated to E1.
