# V1 validation-gate remediation and evidence replay (CPU-only, 2026-11-18)

GO "CPU-only V1 validation-gate remediation and evidence replay". No GPU was used. **Two separate verdicts:**

| | verdict | basis |
|---|---|---|
| 1. Original V1 | **FAIL / STOP** | pinned gates; `V1-GATES.json` unchanged (sha256 `33d1d4025b3de930f8fc88fab03c190ebfd31e348f9ae24c3a65724ab10c1ca6`), report `runs/v1/run/E1-V1-RUN-REPORT.md` unchanged |
| 2. Retrospective V1 evidence assessment | **PASS** | corrected gates (`checks/e1_lifecycle_gates_v2.py` at f88f1aa9) on the preserved artifacts: 25 gates, 0 FAIL, 0 INCONCLUSIVE — `V1-RETRO-GATES.json` |

The retrospective assessment does not replace or upgrade the original verdict. V1 stays recorded as FAIL; the launch is consumed.
The first replay attempt (2bb2157c, 04:20Z) returned FAIL because of two defects in the new validator itself; it is kept in
`attempt1_2bb2157c/` and described in §F.

## A. Incident frozen

- Original report, pinned gates, scripts, results and logs: unchanged. `SCRIPTS_v1.sha256` verifies (30 files); nothing under
  `runs/v1/run/` or `runs/v1/preflight/` was modified; every remediation file is a new file.
- Archive `artifacts/phase1c-e1-v1-dsv41-flash-gb300-tp4-20261118`: `sha256sum -c SHA256SUMS` → all 110 files verify (before and
  after the replay).
- Preserved stream: 347 628 953 bytes, SHA-256 `8f5905d95c61c8e07cf9d557fad97749191146f97c0e1b1a11dc23edb097057f`.
  Durable copy: `<archive>/preserved/dsv41_flash_1017_gb300_tp4_e1_v1_lifecycle.qdstrm` (CephFS, read-only), made at closeout and
  re-verified now: same SHA-256 as the tray16 original `/tmp/dsv41_flash_1017_gb300_tp4_e1_v1_lifecycle.raw-preserve/…qdstrm`.
  The tray original is **retained** (not removed).
- Missing post-run census: `gpu_release_census_postrun.json` was never written, because `postrun_e1_v1.sh` takes it only when
  `lifecycle.json` has a `host` field and the controller writes none. Recorded here; the pinned script is not changed.
- Independently observed GPU-release interval: GPUs last observed **held at 03:22:02Z** (260 943 MiB × 4, 4 compute processes) and
  first observed **released at 03:23:07Z** (0 MiB × 4, 0 compute processes), both by the operator's monitor samples on tray16
  (`operator_monitor_samples.jsonl`, 22 samples 03:06–03:28Z, raw lines in `operator_monitor_samples.raw.txt`; scratch script, not
  pinned). Pinned census samples: released at 03:27:45Z and 03:28:06Z. Controller timestamp, separate: 03:22:46.7Z. The exact release
  instant is not established; it lies in (03:22:02Z, 03:23:07Z].
- Supplementary offline-import diagnostic of 03:48Z: preserved as `runs/v1/run/evidence/supplementary_recovery.json`, explicitly
  **non-pinned** evidence.

## B–E. Corrections (new files; see `SCRIPTS_v1_replay.sha256`)

| gate | pinned behaviour that failed | corrected rule |
|---|---|---|
| **L1** one effective stop | required ≥ 1 suppressed duplicate | one request sent and answered ok; zero or more duplicates, each counted as suppressed; each finalisation phase exactly once; independent of the controller, the server log has exactly one API-server `Stopping profiler` / `Profiler stopped.` pair and one `POST /stop_profile` response (200). Fixture tests for 16 concurrent requests, repeated requests after the report, and a single request: `tests/test_profiler_lifecycle_stop_concurrency.py`; the pinned repeated-call test stays. |
| **L4** preserved stream | compared the final file with a size sampled while Nsight was still appending | finalisation confirmed (report written, Nsight gone, session closed); stream available; size and SHA-256 taken twice around full reads with unchanged size/mtime; not modified after session close; Nsight's own stream gone or the same inode; a durable copy must equal the recorded one. The earlier size sample is reported only. Stability is **not** treated as completeness. |
| **L10 / L10b** recovery | skipped when L4 failed; counts only | mandatory (no tools or no in-session export → not PASS). Offline import, sqlite export, then against the in-session export: kernel rows, NVTX forward ranges, distinct iteration indices and their range, iterations in band, kernel and NVTX capture boundaries, per-process kernel counts, every table's row count, capture metadata (`ANALYSIS_DETAILS`, `META_DATA_CAPTURE`, `TARGET_INFO_SESSION_START_TIME`), export product/schema rows. |
| **S.config_identity** | flat allow-list in the wrong namespace; paths not handled | `checks/e1_launch_identity_v2.py` + `checks/e1_identity_policy_v1.json`: field-by-field; a differing field passes only with a listed rule whose condition holds — output paths (run name, `-o` stem, `log_dir`), worktree source paths (only with the fork head equal on both sides and equal to the pinned head, clean status; the trace path only with both files at the pinned SHA-256), approved lifecycle controls (exact approved values). Protected fields (server argv/config, every env variable except `PYTHONPATH`, all other Nsight argv, capture recipe keys, workload and schedule) can never be normalized; the policy loader rejects a rule on them. Every normalized field is printed with both original values. |
| **L6c** GPU release | host read from `lifecycle.json` (never written); a single timestamp | run host from `run.log`; all independently timestamped observations (pinned census files and JSONL observations) → observed interval (last held, first released]; fails without a released observation after the controller's time, with a held observation after the first released one or after the controller's time, or with an observation from another host. Controller time kept separately. `checks/e1_gpu_watch.py` is a registered-sampler for future runs so the interval is seconds wide. |

Not changed: the lifecycle controller (`alignment/`), the launcher, the recipe, the V1 config and inputs. Two controller-side
improvements are proposed for review, not made: write `host` into `lifecycle.json`, and re-sample `stream.size` when the session closes.

## F. Evidence replay (`replay_e1_v1.sh`, tray16, CPU-only, CUDA hidden)

Attempt 2, f88f1aa9, 04:24:33Z → `V1-RETRO-GATES.{json,txt}`, `identity_v2.{json,txt}`, log `replay_tray16_attempt2.log`:

| comparison (recovered from the preserved stream vs in-session export) | recovered | in-session |
|---|---|---|
| offline import | rc 0, 27 s, report 393 019 035 B | 393 018 929 B (sizes informational) |
| kernel rows | 12 250 832 | 12 250 832 |
| NVTX forward ranges | 8 640 | 8 640 |
| distinct iterations (index range) | 2 160 (181–2342) | 2 160 (181–2342) |
| kernel capture boundaries (ns) | 89 415 511 – 50 605 741 376 | identical |
| NVTX forward boundaries (ns) | 96 784 576 – 50 602 186 272 | identical |
| kernels per process (4 ranks) | 3 062 168 / 3 062 168 / 3 062 168 / 3 064 328 | identical |
| row counts of all 53 tables | — | no difference |
| capture metadata; export product and schema | — | identical |
| export-run rows (file names, paths, export time) | 16 rows | 16 rows; differ by construction, recorded, not judged |

Corrected gates: L1 PASS (1 request, 0 duplicates, server log 1 / 1 / one 200); L4 PASS (size 347 628 953, SHA-256 above, stable,
last modified 6.1 s before session close); L10, L10b PASS; L6c PASS (interval above; controller time inside it); configuration
identity PASS with **30 normalized fields**, all listed with original values in `identity_v2.txt` (4 output-path, 3 worktree-path,
23 approved lifecycle controls across the launch record and the profile config). The 15 unchanged gates PASS as in the original run.
The scientific checker exit codes are taken from the original post-run record, not re-run.

Attempt 1 (2bb2157c, 04:20:35Z) → FAIL on two corrected gates, both defects of the new validator:
- L1 counted any `] Profiler stopped` text and so also a worker's `[wrapper.py:86] Profiler stopped successfully.` line. The accepted
  `e1_nsys_val` server log has the same worker line once. Fixed to count the API server's lines and the HTTP response.
- L10b judged `META_DATA_EXPORT` rows that hold the export's own file paths and wall time. Fixed to judge product and schema rows only.
Both fixes have tests; attempt 1's outputs are kept unchanged.

## Limits of the retrospective assessment

- The lower bound of the GPU-release interval comes from operator monitor samples taken with a scratch script over ssh, not from a
  pinned sampler. Without them the pinned census alone gives only "released by 03:27:45Z".
- "No duplicate finalisation" rests on the controller's record plus the server log; no second independent source exists.
- The corrected gates were written after seeing the V1 data. They have not been exercised prospectively on a fresh run.

## G. Regression (CPU-only, guard active)

All runs through `tests/cpu_only.sh` (devices hidden; guard line "no /dev/nvidia<N> opened; torch.cuda not initialised" in every log; GPUs 0 MiB afterwards).

| run | result | log |
|---|---|---|
| targeted suites (launcher-adjacent, lifecycle, preflight, pinned gates, pid cleanup, guard, corrected gates, identity, stop concurrency), tray14 | **259 passed**, 1 skipped | `pytest_targeted_cpu_only_f88f1aa9_tray14.log` |
| whole tier, tray14 | 5174 passed, 78 skipped, **6 failed** | `pytest_fullsuite_cpu_only_f88f1aa9_tray14.log` |
| targeted suites, tray16 | 258 passed, 1 skipped, **1 failed** | `pytest_targeted_cpu_only_f88f1aa9_tray16.log` |
| whole tier, tray16 | 5173 passed, 78 skipped, **7 failed** | `pytest_fullsuite_cpu_only_f88f1aa9_tray16.log` |

The six whole-tier failures are the same as before this work and outside it: the `cargo`-dependent `test_launcher_corpus` test, the
gdn numerical test, and four `test_l1_single_gemm` energy tests that fail whenever CUDA is hidden. The extra failure on tray16 is
`test_profiler_preflight.py::test_preflight_cli_writes_pins_and_exits_nonzero_on_fail`: that pinned test runs the real location
check against the host's `/tmp`, and tray16's `/tmp` still holds the retained preserved stream
(`locations.stream_leftovers` FAIL). It passes on tray14. The test file is in the pinned V1 manifest and was not changed; its
dependence on the host's `/tmp` is a test-isolation weakness to fix with the next pin.
