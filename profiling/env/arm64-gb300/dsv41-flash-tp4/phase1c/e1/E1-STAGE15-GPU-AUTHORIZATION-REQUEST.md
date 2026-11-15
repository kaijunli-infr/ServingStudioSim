# Stage 1.5 GPU authorization request (Phase 1.5 item G.8) — awaiting Kevin's decision

Status: Stage 1 accepted (phase1c-e1-stage1 c7e8ed9b). Stage 2 (full E1 capture) remains NO GO and is **not** requested here.
This request covers only the bounded Stage 1.5 passes prepared on branch `phase1c-e1-stage15`. Nothing starts without an
explicit go naming the passes, the tray and the reservation.

## What is requested

| # | pass | GPU-held | purpose |
|---|---|---|---|
| 1 | `e1_dry_off_v2` | ~26 min | hook-off baseline in the same wire mode as the hook arm (stream=false) |
| 2 | `e1_dry_on_v2` | ~26 min | hook-on arm, identical configuration apart from the hook flag |
| 3 | `e1_dry_off2_v2` | ~26 min | off/off control: run-to-run null for timing, tactics and tokens |
| 4 | `e1_nsys_val` | ~30 min (+ ~4 min CPU parse) | short Nsight validation of NVTX ownership, rank alignment, collective end spread, joins, timestamps, export |

Total nominal ~2 h 05 min GPU-held on four GB300s of one rack-10 tray, requested reservation **4 h hard stop**, storage ~3 GB
(budget 20 GB) — E1-STAGE15-BUDGET.md. Options: passes 1-3 only (2 h 30 min reservation) or pass 4 only (1 h 30 min).

## Prerequisites (all CPU, all done unless marked)

* Configs `configs_v2/*` generated, hashed (`INPUTS_v2.sha256`), launcher `--dry-run` validated x4, identity-checked
  (off/on PASS, off/off2 PASS). Trace unchanged from Stage 1.
* Launcher tree: scratch worktree `$S/wt-e1stage1` on `phase1c-e1-stage15` with the Python stream opt-out applied and
  tests green. **Open:** the Rust `--stream` flag is not in the worktree binary (patch reverted after verification; the
  patched binary is at `$S/bundles/session_runner.stream-flag.2cd7990+patch`). Before the go, either install that binary into
  `alignment/load_generator/req-frontend/target/release/` or re-apply `patch/apply_stream_flag_rs.py` and `cargo build`;
  the preflight refuses to start otherwise (check D). Bumping the submodule gitlink needs a push to the req-frontend remote —
  please decide whether that is wanted now or after Stage 1.5.
* Checkers: `checks/e1_completeness_v2.py` (20 tests), `checks/e1_config_identity.py` (5), `checks/e1_nsys_validation.py`
  (run on Phase 0 quadrant: PASS), `analysis/e1_pair_compare_v2.py`, `analysis/autotune_cache_diff.py`.
* Scripts: `preflight_e1_stage15.sh` (smoke-run on tray14, see `runs/stage15/cpu/preflight_stage15_smoke_tray14.log`),
  `run_e1_stage15.sh <pass>`, `collect_e1_stage15.sh`.

## Procedure on the go

1. Laptop: reservation pod on the named tray (as Stage 1), `kubectl` confirms no GPU pod on the node.
2. Tray: `preflight_e1_stage15.sh` must print PREFLIGHT PASS (Stage 1 host checks + Stage 1.5 checks A-F).
3. Passes in the order 1, 2, 3, 4, each `tmux new -d -s e15-<pass> "E1_REPO=$WT bash $E1/run_e1_stage15.sh <pass>"`; after
   each pass: `checks/e1_completeness_v2.py` (PASS required to continue), identity check on the run directories.
4. After pass 3: `analysis/e1_pair_compare_v2.py --off 1 --on 2 --control 3`; after pass 4: Phase 1A extraction +
   `checks/e1_nsys_validation.py`.
5. Cleanup evidence (0 compute apps, 0 MiB, no processes, no tmux), reservation release, `collect_e1_stage15.sh`, report,
   commit, stop for review.

## Stop rules

Any preflight FAIL; completeness v2 FAIL on pass 1; a GPU pod scheduled on the tray; a launch without "Application startup
complete" after 30 min; disk below the preflight thresholds. Then stop, archive, report; no retry without a new go.

## Explicitly not requested

Full E1 capture (`e1_main`), Rust integration into the main branch, model fitting, profiling database writes, preset changes,
any flashinfer/vLLM code change (autotuner option O2).
