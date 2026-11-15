# Off/off comparison protocol (Phase 1.5 item C) — prepared, NOT authorized to run

Purpose: measure baseline run-to-run variability of (a) pure-decode iteration time and (b) generated tokens between two
launches that differ in nothing but the launch itself, so that the hook-on vs hook-off comparison can be read against it
(E1-OVERHEAD-METHODOLOGY.md §3).

## Passes (configs in `configs_v2/`, hashes in `INPUTS_v2.sha256`, plan in `E1-STAGE15-INPUTS.json`)

| order | pass | kind | hook | stream key | role |
|---|---|---|---|---|---|
| 1 | `e1_dry_off_v2` | workload_metrics, openai | off | `false` (explicit) | baseline, same wire mode as the hook-on arm |
| 2 | `e1_dry_on_v2` | workload_metrics, openai | on (`capture_routed_experts`) | `false` (explicit; also implied by the hook) | hook arm |
| 3 | `e1_dry_off2_v2` | workload_metrics, openai | off | `false` | control: identical to pass 1 apart from `name`, `log_dir` |
| 4 | `e1_nsys_val` | nsys | on | `false` | short Nsight validation (E1-NSYS-VALIDATION-PROTOCOL.md) |

All four use the Stage 1 trace `traces/e1_dry.csv` (sha256 c6d8e13b…, unchanged: 56 requests, cells b=8 and b=48 at KV 32k,
1024 output tokens, trace-timed, ~73 s), the Phase 0 server recipe, and `logs/dsv41_flash_gb300_tp4_e1_stage15/` as run root
(no Stage 1 directory is touched). The order interleaves the hook arm between the two baselines so that a monotone drift of
the tray over the session shows up in control/off rather than being aliased onto on/off.

## Identical configuration (item C.1-C.2)

* Stage 1 confounder: hook-off streamed (`chunk_count` 1024/1025 per request), hook-on did not (`chunk_count` 1), because the
  req-frontend sets `stream: !capturing_routes` (`src/backend/client.rs`). `checks/e1_config_identity.py` on the Stage 1 run
  directories reports exactly that (`runs/stage15/config_identity_stage1_runs.txt`: `effective wire mode a=streaming
  b=non-streaming` FAIL, plus `server env differs in ['PATH']`, ordering only) and nothing else (argv differ only by
  `--enable-return-routed-experts`).
* Fix: `workload.stream` (launcher, `patch/apply_stream_optin.py`, default true, validated boolean, property
  `effective_stream = stream and not capture_routed_experts`; runner passes `--stream false`) and `--stream <true|false>` on
  `session_runner` (`patch/apply_stream_flag_rs.py`, `patch/req-frontend-stream-flag.patch`: clap `ArgAction::Set`,
  default true; `streaming = self.stream && !capturing_routes`; `fold_body` is used for non-streaming with or without a
  routes directory). Tests: `tests/test_alignment_stream_optin.py` (5), req-frontend `cargo test` (all green, 102+7+32+39).
* Identity checks (CPU, done): `e1_dry_off_v2` vs `e1_dry_on_v2` PASS (`runs/stage15/config_identity_v2_off_on.txt`),
  `e1_dry_off_v2` vs `e1_dry_off2_v2` PASS; the check is strict about the file (a `stream` key present on one side only fails
  even when the effective mode matches, `test_implicit_stream_on_one_side_is_a_config_difference`).
* Post-run identity: the same script on the run directories compares `*_launch.json` (argv, server_argv, env, server_config,
  nsys_config, python_runtime, fork_python, fork_git); only the hook flag may differ, `--ignore-env PATH` for the known
  ordering difference. Run it before any statistic is read.
* Launcher tree: the binary must support `--stream` (patched `session_runner`; preflight checks `--help`). The patched binary
  built from gitlink 2cd7990 + patch is at `$S/bundles/session_runner.stream-flag.2cd7990+patch` (sha256 7f6eb81d…); the
  submodule gitlink is **not** bumped in this branch (pushing the req-frontend commit needs approval).

## Measurements and how they are read

1. `checks/e1_completeness_v2.py --inputs E1-STAGE15-INPUTS.json` on every pass (PASS required; `--compare` on pass 2 and 3
   against pass 1).
2. `analysis/e1_pair_compare_v2.py --off <1> --on <2> --control <3>`: per cell on/off and control/off points, tau, L,
   intervals, noise flag, excess with interval; `output_preview` identity for on vs off and control vs off; `chunk_count` sets
   must be equal across all three (same wire mode).
3. `analysis/autotune_cache_diff.py` between the three launches' `autotune_configs.json` (copied by `collect_e1_stage15.sh`):
   how many winning tactics changed between off and off2 (pure re-tune noise) and between off and on.
4. Reading (fixed in E1-OVERHEAD-METHODOLOGY.md §3): Gate A on on/off; Gate B on on/off read with the noise flag from
   control/off; excess reported. No new threshold.

## Budget (detail in E1-STAGE15-BUDGET.md)

Three workload_metrics passes x ~25.5 min GPU-held (22.5 min startup incl. ~19 min autotune + 51 s workload + teardown) =
~77 min; storage ~60 MB (off) + ~110 MB (on) + ~60 MB (off2).

## Stop rules

Any preflight FAIL, any completeness v2 FAIL on pass 1, or a GPU pod scheduled on the tray -> stop and report; no re-runs
without a new go.
