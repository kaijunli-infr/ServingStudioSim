# Short Nsight validation protocol (Phase 1.5 item D) — designed, checker tested on Phase 0 data, NOT authorized to run

Purpose: before the full E1 capture (Stage 2, separate authorization), one bounded nsys pass with the hook on, on the Stage 1
dry trace, to test the six things Stage 1 could not test without nsys. Everything the GPU run would produce is checked by
`checks/e1_nsys_validation.py`, which has been run (CPU) on the Phase 0 `quadrant_nsys` pass and its Phase 1A tables.

## 1. The pass

`configs_v2/e1_nsys_val/profile.json` (sha256 in `INPUTS_v2.sha256`): `profile_kind: nsys` with the Phase 0 nsys block
(`capture_mode: cuda_profiler_api`, `cuda_graph_trace: node`, `cuda_event_trace: false`, `capture_start_delay_seconds: 10`, no
bounded stop — the Phase 0 lesson), hook on (`capture_routed_experts: true`), `stream: false`, openai backend, trace
`traces/e1_dry.csv` (56 requests: b=8 and b=48 at KV 32k, 1024 output tokens, ~73 s), Phase 0 server recipe, devices 0-3.
Identity vs `e1_dry_on_v2`: identical apart from `profile_kind`, `nsys`, `name`, `log_dir` (checked, `runs/stage15/`).
Expected size: ~2160 iterations, i.e. 0.45x the Phase 0 quadrant pass (4750 iterations, 25.6 M kernel rows, 0.70 GB .nsys-rep,
3.82 GB .sqlite, export 4.5 min, parse 8.8 min) -> ~11.7 M kernel rows, ~0.32 GB rep, ~1.8 GB sqlite, export ~2 min, parse ~4 min.

Driver: `run_e1_stage15.sh e1_nsys_val` after passes 1-3 of E1-OFFOFF-PROTOCOL.md (or alone, if only this is authorized);
preflight `preflight_e1_stage15.sh` in front (refuses to start on FAIL).

## 2. What is validated, and how (`checks/e1_nsys_validation.py --profile <dir> --tables <Phase 1A tables> --inputs E1-STAGE15-INPUTS.json`)

| requirement (Kevin, item D) | check | PASS condition |
|---|---|---|
| NVTX ownership and completeness | `parsed.json`: every iteration has 28 NVTX ranges (7 phases x 4 devices); `devices_absent` empty; `iteration_index_by_device` identical across devices; `step_alignment.unpaired_steps_by_device == {}` | all counts 0 / empty |
| per-rank iteration alignment | Phase 1A `iter_rank` table: for every iteration the four ranks' GPU kernel windows `[first_start, last_end]` overlap; GPU first-kernel start spread median/p90/max reported; host-side `forward` NVTX start spread reported | 0 non-overlapping iterations |
| collective end-time spread | `collectives` table via `PassData.collectives(it)`: per pure-decode cell b, `max_rank(end_k) - min_rank(end_k)` over all k; modal K and `n_other_K` | packet §4.3: median <= 0.02 ms and p90 <= 0.05 ms per cell; every pure-decode iteration rank-aligned |
| routing-to-iteration joins | `checks/e1_completeness_v2.py` on the same profile (hook on): unique token<->iteration join, segments == npy[1:], per-cell minima of the dry plan | completeness v2 PASS |
| timestamp consistency across ranks | one nsys session: GPU windows overlap (above); per device the iteration start sequence is strictly increasing; `nvtx_forward_ranges == 4 x iterations` | 0 violations |
| trace completeness and export correctness | `profile_result.validation.ok`, kernel rows and span recorded; `parsed_device_ids == [0,1,2,3]`; `.nsys-rep` and `.sqlite` present and non-empty; tables iteration count == parsed iteration count; JIT-affected iterations after warm-up <= 20 (Phase 0 exclusion rule) | all PASS |

Exit code 1 on any FAIL; JSON report with every number.

## 3. Checker result on Phase 0 `quadrant_nsys` (CPU, 2.9 s; `runs/stage15/nsys_validation_quadrant.txt`)

RESULT PASS. 4750 iterations, 28 ranges each, 0 absent devices, unpaired steps {}, GPU windows overlap for 4750/4750 with
first-kernel start spread median 0.009 ms / p90 0.015 ms / max 3.5 ms, host forward start spread median 0.36 ms / p90 0.71 ms
(host launch skew; the four host-side forward ranges do not overlap in 4287 iterations because a CUDA-graph launch is sub-ms —
reported as INFO, the GPU-side overlap is the check), JIT-affected after warm-up 10, 36 pure-decode cells b=1..48 with
collective end spread medians 0.0001-0.0011 ms and p90 0.0002-0.0086 ms (all far inside 0.02/0.05), modal K 163 (b<=15) /
86 (b>=17), `n_other_K` 0 everywhere, 4726/4726 pure-decode iterations rank-aligned. Routes were skipped (`--skip-routes`:
Phase 0 had no hook). This establishes that the checker runs end to end on real data and that the §4.3 thresholds are
consistent with Phase 0 behaviour; it is not a Stage 1.5 result.

## 4. After the run (CPU)

1. `preflight` log, `run.log`, `profile_result.json`, `parsed.json` sizes recorded in `runs/stage15/e1_nsys_val/`.
2. Phase 1A tables: `cd phase1a && $S/gb300/venv/bin/python -m costmodel.extract --profile <dir> --label e1_nsys_val --out
   $S/phase1a-tables/e1_nsys_val --sha256sums` (CPU, minutes).
3. `checks/e1_nsys_validation.py` as above; `checks/e1_completeness_v2.py --parsed` (writes `jit_excl.txt`).
4. `analysis/e1_identify.py --tables … --label e1_nsys_val` only as a smoke test of the estimator pipeline on a 2-cell pass; no
   estimate from this pass is used for anything (the dry trace has 2 of the 24 E1 cells).
5. Storage: raw copy to `$S/artifacts/phase1c-e1-stage15-<date>/`, SHA256SUMS; the sqlite (~1.8 GB) may be deleted after the
   tables and checker outputs are archived (the rep file is kept).

## 5. Stop rules

Preflight FAIL; nsys fails to start or the launcher reports `validation.ok == false`; disk free below the preflight thresholds;
a GPU pod appears on the tray. Then stop, archive what exists, report. No second attempt without a new go.
