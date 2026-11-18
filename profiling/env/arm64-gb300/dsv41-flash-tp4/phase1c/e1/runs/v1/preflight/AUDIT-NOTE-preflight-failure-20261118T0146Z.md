# Audit note: failed driver-context preflight of 2026-11-18 01:45:58–01:46:54Z (5:46 PM PST, 2026-11-17) and its unavailable logs

Written 2026-11-18 before the V1 launch, as required by the V1 GO. Factual record; nothing here changes pinned code or the recipe.

## What happened

- Host pod4-gb300-10-tray16-f3, user dc, tree `wt-lifecycle` at code commit 9d4abd3f, driver context (tmux `e1v1-ctxtest` + tee).
- Started 01:45:58Z, finished 01:46:54Z. Result: **32 PASS, 4 FAIL, `preflight_rc=1`** (PREFLIGHT FAIL).
- The four FAIL lines, as printed at the time:
  - `FAIL  Stage 1.5 preflight (see above)` — its inner line was `FAIL  repo has uncommitted tracked changes`.
  - `FAIL  launcher tree has uncommitted tracked changes`
  - `FAIL  process discovery found candidates (see above; review, never kill by pattern)` — candidate pid 879669.
  - `FAIL  registry plan rc 2 (owned/lookalike/refused; see above)` — `LOOKALIKE pid 879669 uid 1000 bash … -> NOT owned, not touched`; summary `owned 0, stale 0, gone 0, refused 0, lookalikes 1`.
- Passing lines of the same run included the identity preflight (26 checks, 0 FAIL), the test step (50 passed, CPU-only guard line present) and the GPU state census (no compute process, 0 MiB).

## Causes (both operator errors, Claude)

1. `pin_v1.sh` had just rewritten the tracked files `V1-PIN.json` and `V1-PIN-identities.preflight.json`; they were not committed before the preflight, so both clean-tree checks failed.
2. The ssh command that started the preflight also waited for it, and its command text contained the experiment directory path. That waiting shell (pid 879669, started 01:45:51Z) was correctly reported as an independent lookalike. Nothing was signalled.

Neither cause was a defect of the preflight; it stopped as designed. The pins were committed (de4e9ae4) and the preflight was restarted detached; that run (01:48:03–01:48:59Z, 36 PASS, 0 FAIL) is the one recorded in the packet.

## Unavailable logs

The failed attempt's output files were **deleted by the operator (Claude) at about 01:47:5xZ** with an explicit `rm -f` in the scratch script `prep_ctx2.sh`, before the restart. This was a wrong call: failed preflight outputs should have been kept. The deleted files were, under `runs/v1/preflight/`:

- `preflight_v1_tray16_driver_context_20261118T014654Z.log`
- `identity_20261118T014558Z.json`
- `pytest_20261118T014558Z.log`
- `gpu_census_20261118T014558Z.json`

The temporary directory `<run root>/ctxtest` was removed by the wrapper as designed, and the wrapper's summary log `scratch/lifecycle/ctx_v1.log` was overwritten by the next run. They are not recoverable (CephFS, no snapshot taken).

What survives: the FAIL, FOUND and LOOKALIKE lines quoted above, which were captured in the operator session transcript (session_017xskCQhzYqJGMizgpJm2s9) at 01:46–01:48Z, and the scratch script `scratch/lifecycle/prep_ctx2.sh` that shows the deletion. The earlier failed attempts of 2026-11-17 22:30Z and 22:34Z were overwritten the same way and are likewise known only from the transcript and packet §6.

## Rule from now on

Every preflight output, passing or failing, is kept under `runs/v1/preflight/` and committed at closeout.
