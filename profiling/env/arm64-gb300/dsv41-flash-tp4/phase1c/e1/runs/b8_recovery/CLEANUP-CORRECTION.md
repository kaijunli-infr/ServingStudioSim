# Cleanup-safety correction (CPU only, 2026-11-15): discovery separated from termination

Kevin's conditional acceptance of the b=8 recovery packet (9275abbd) required that process discovery and process termination
be separate, that ownership rest on explicit records with identity verification, and that uncertain ownership stop the
procedure. No GPU was used; the frozen workload (`INPUTS_b8.sha256` unchanged), Nsight settings, validation thresholds and
model are untouched.

## What changed

| requirement | implementation |
|---|---|
| 1. explicit owned-PID records, identity incl. start time, verified before signalling | `checks/e1_pids.py`: registry `<run root>/pids/owned.tsv` with pid, start time (`/proc/<pid>/stat` field 22, clock ticks since boot), boot id, uid, label, command; `register` / `spawn` write it; `plan` verifies every record; `cleanup --execute` signals only verified identities |
| 2. no ownership from command line / path / quoted text | `checks/e1_process_census.sh` is discovery only: it prints `FOUND … (candidate; ownership only via the registry)` and never signals; `e1_pids.py` classifies such processes as LOOKALIKE and never signals them |
| 3. exclude the current shell, driver ancestry, ssh session, unrelated processes | the cleanup process, all its ancestors (bash, sshd, tmux, …), pid 1 and other uids are excluded; a registered or owned pid in that set is REFUSED and aborts the whole run with nothing signalled (rc 3); non-owned processes are never candidates for a signal |
| 4. re-verify PID identity immediately before termination | `signal_verified()` re-reads `/proc/<pid>/stat` + status right before `kill()`; a changed start time / uid / boot id → skip ("pid reuse"); zombies count as gone |
| 5. uncertain ownership = stop-and-report | lookalikes present → rc 2 (`NOT CLEAN`, listed, untouched); refusals → rc 3; `closeout_e1.sh` only prints the dry-run plan and the exact `--execute` command |
| 6. positive cases with real owned processes, negative cases with lookalikes | `tests/test_e1_pids_cleanup.py` (real background processes via `subprocess`, registered vs. unregistered) and the real run on tray14 (`cleanup_positive_negative_tray14.log`) |
| 7. test shell and ssh session survive | the pytest `guard` fixture asserts the test process and its parent alive with an unchanged ppid after every case; the tray14 run records the ssh shell pid and start time alive before and after |
| 8. previous failing test preserved | `process_census_positive_tray14.log` (17:37Z) is kept unchanged: the `pgrep -f` pattern of the planted `tail -F` command matched the test shell's own command line and killed it (ssh exit 255). `test_regression_quoted_tail_command_in_shell_cmdline_survives` and the `negative:quoted` process of the tray14 run reproduce that shape: a bash whose argv quotes the planted command text is reported as LOOKALIKE and survives |

The driver `run_e1_b8.sh` gained exactly one functional line, registering its own shell pid at start (`runs/b8_recovery/
driver_diff_vs_9275abbd.patch`, 23 lines: 3 added comment/registration lines, the closing discovery call now passes the
registry path). Pass selection (`e1_b8_recovery` only), the trace guard before launch, the launch command, autotune and
EngineDeadError bookkeeping and the exit code are unchanged. The launcher, vLLM, nsys and `session_runner` are descendants of
the registered driver shell, so `e1_pids.py cleanup` can terminate exactly that tree and nothing else.

## Results

* `tests/test_e1_pids_cleanup.py` + `tests/test_e1_capture_window.py` on tray14 (uv env): **19 passed** (9 cleanup, 10
  capture window; `runs/b8_recovery/pytest_cleanup_capture_window.txt`). Cleanup cases: register/verify; owned tree (bash +
  child sleep) and log watcher terminated while a bash quoting the planted `tail -F` command and a bash carrying the
  experiment marker in argv survive (rc 2); the regression case above; stale record / pid reuse skipped (rc without actions);
  identity re-check immediately before the signal (monkeypatched start time → skip); refusal of the test process, its parent
  and a grandparent (rc 3, the valid owned process is also left alone); dry-run signals nothing; CLI round trip. After each
  case the test process and its parent are alive; no test process is left behind (descendants reaped).
* Real run on tray14 at 18:02:55Z (`cleanup_positive_negative_tray14.log`): registered `tail -F` on the e1_nsys_val run log
  and a registered `bash -c "sleep 600 & wait"` with cwd in the run directory (owned 3 incl. the child); unregistered
  negatives with the quoted tail command and the marker in argv. Discovery: candidates found (rc 1, nothing signalled).
  Dry-run: 3 would be signalled, 2 lookalikes never (rc 2). Execute: signalled 2, skipped 1 (the parent bash exited when its
  child died: "gone before signal"), survivors 0, lookalikes 2 alive (rc 2). The ssh test shell (pid 934165, start 18:02:55)
  is alive before and after. The negatives were then removed through their own registry by verified identity (rc 0) and the
  final discovery is CLEAN.
* Earlier iterations of this correction (same day) found and fixed: zombie children of the test process counted as alive
  (now gone), a registry record with a newline in the command corrupting the file (commands are now single-line; malformed
  lines are ignored with a warning), bash exec-optimising `bash -c "sleep N"` so the quoted text was not in any live cmdline
  (lookalikes now use `; true`), and orphaned test children (the guard now reaps descendants). Leftover test processes from
  those iterations were removed by verified identity, never by pattern.
* `SCRIPTS_b8.sha256` (19 files: recovery scripts, checks, analysis, both test files) is verified by `preflight_e1_b8.sh`;
  manifest sha256 recorded in the commit message.
* Final read-only preflight on tray16: see the last `runs/b8_recovery/preflight_tray16_*.log` (result line below, filled at
  commit time) and `closeout_tray16_*.log` (CLEAN: GPUs 0 MiB, discovery clean, registry plan 0 owned / 0 lookalikes).

PREFLIGHT_RESULT_PLACEHOLDER
