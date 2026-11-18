# E1 V2 authorization packet — forced GPU release while the Nsight import is running (CPU-only preparation; authorizes nothing)

**FROZEN 2026-11-18** after the final CPU-only gate rehearsal (§13). The frozen content is listed in §14 and pinned by
`FREEZE_fr.sha256`. Any change to a frozen item needs a new review; a launch needs its own GO naming the commit.

Prepared 2026-11-18 under the GO "Narrow CPU-only controller improvements and V2 readiness preparation" (review of
`exp/profiler-lifecycle` f3b2542e). It asks for **one** GPU launch, `e1_v2_forced_release`, and nothing else. No GPU was used to
prepare it. V2 stands on its own: its gates are new, are frozen by this packet's manifests, and neither read nor change the V1
acceptance record. The two V1 records stay separate and unchanged: original V1 **FAIL / STOP** (pinned gates), retrospective V1
evidence assessment **PASS, qualified** (corrected gates, `runs/v1/replay/`, first failed replay attempt kept).

## 1. Question

V1 showed a normal stop. V2 asks what the e1_main_v3 incident left open: **if the server must be stopped to free the GPUs while
Nsight is still importing, does the trace survive?** Either the importer finishes on its own, or it does not and the trace is
recovered from the preserved raw stream. Both are admissible outcomes; losing the trace, signalling the importer, or failing to
free the GPUs in time is not.

## 2. Exact source, manifests, pins

| item | value |
|---|---|
| branch / code commit | `exp/profiler-lifecycle` code **e32fbab7b39c66894f2ff87a02ecf45139183ee7** (build-tool guard, binary gate checker, rehearsal tooling; on top of 8570b050), pins **565206ee405905826f82db998b1d2a333a589fae** |
| pinned trees (`V2-FR-PIN.json`) | `alignment` 7b4df784a5b6daff7c792e81cbc6503ba2c806ec, `launcher` 5acab28980ef9974bf4cb6dcd676481c357e655d, 17 test files, the CPU-only guard and its runner, and the fixtures by blob id |
| submodules | vLLM fork 0ab18711, req-frontend 5c20a259 |
| load generator binary `session_runner` | pinned bytes **7f6eb81d45e6b98f3b6ebb56ab20b60ce8aaf4584aa3d4f25e3449b2d05a00c2**, restored in the worktree and verified byte-identical to the reference; launched only through the prebuilt contract (§12) |
| identities (`V2-FR-PIN-identities.json`, tray16) | Nsight CLI 2025.6.3.541 sha256 2d53d710…, `QdstrmImporter` sha256 b30312a6…, server interpreter and vLLM import as in V1 |
| inputs (`INPUTS_fr.sha256`) | trace `e1_dry.csv` c6d8e13b…; config `configs_fr/e1_v2_forced_release/profile.json` 7f7e4f66…; source config df949648… |
| scripts, checks, tests (`SCRIPTS_fr.sha256`) | 51 files; the manifest itself is sha256 967a92ddd50acde33f3786abb7b90977f6b30fba56fecf7b975cb88c9faa826c |
| expected values, trigger, limits, outcomes | `E1-V2-FR-INPUTS.json` (sha256 1af3d9ed…, unchanged since 8570b050) |
| freeze manifest | `FREEZE_fr.sha256`: this packet, the three manifests, the inputs, the pins and the binary provenance (§14) |

The V2 config is the V1 config byte for byte apart from `name` and `log_dir` (checked at generation). Model, server arguments,
request schedule, routing, Nsight flags and CUDA environment are those of the accepted `e1_nsys_val` recipe with the approved
no-delay start. Lifecycle bounds are V1's: soft 300 s, hard 600 s, stall 300 s, session close 300 s, max extension 120 s,
GPU-held ceiling 2700 s, stream preserved, no `tmp_dir`.

**Change from the earlier proposal (packet v3 §7).** V2 was sketched there as "hard deadline 20 s so it fires during the import".
V1's measured timeline rules that out: the flush took 64 s before the import started, so a 20 s deadline would fire during the
flush, not the import. V2 therefore uses a phase-based trigger (§3) and leaves every deadline at its V1 value.

## 3. Workload and the deterministic trigger

- One launch, the V1 workload: 56 requests in two waves (b=8 at 0 s, b=48 at 18.6 s), 1024 output tokens each; expected about 2160
  iterations and 12.25 M kernel rows on four ranks. V1 timing: startup about 27 min, stop → import start 64 s, import 27 s.
- **Trigger and independent observer.** `checks/e1_release_probe.py`, a registered background job, polls `lifecycle.json` every
  0.5 s. On the **first** observation of phase `importing` it:
  1. verifies the importer's identity from `/proc`: the recorded Nsight pid is alive with the recorded start time;
  2. takes samples of that process tree (state, CPU ticks) about 0.4 s apart, for at most 3 s, until the import is **confirmed in
     progress**: CPU time or reported progress advanced while the phase stayed `importing`;
  3. appends the one pre-registered line, read from `E1-V2-FR-INPUTS.json`, to `lifecycle.control`:
     `release-gpus V2 forced release during import (pre-registered trigger: first observation of phase importing)`;
  4. waits for the controller's recorded API-server SIGTERM time and samples the importer's identity and state about 0.3 s, 2 s and
     5 s after it.
  It writes the line once and signals nothing. If the stop is already terminal, the importer is not the recorded process, or the
  import ends while it is being confirmed, it writes **nothing** and says so in `probe.json`; that run is INCONCLUSIVE.
- The controller reads the control file once per second and runs its unchanged `release_gpus` action: identity-verified SIGTERM to
  the API server, SIGKILL to recorded server processes still alive after the grace period, never Nsight.
- Expected latency from import start to release request: about 2 s, at most about 5 s, of a 27 s import.

## 4. Controller and preflight changes in this commit (item A/B of the GO)

| change | where | effect |
|---|---|---|
| launch host identity | `lifecycle.py` | `host`, `host_identity` (hostname, boot id, launcher pid, source) written from the launcher process at controller construction; record schema 2 |
| stream observations kept | `lifecycle.py` | every distinct (size, mtime) sample is appended to `stream.observations` with time and phase; `stream.size` stays "latest sample" |
| final stream record | `lifecycle.py` | once, after the Nsight session has closed and the phase is terminal: `stream.final` = size, mtime, sha256 (stat before and after the read) and whether it changed after the last observation; also written as `RETENTION.json` next to the preserved stream |
| completeness wording | `lifecycle.py`, gates | every final record and retention record carries: completeness is **not** established by a stable size or a matching checksum |
| release context | `lifecycle.py` | at a release request: phase, import progress, Nsight alive, report written — evidence only, no decision uses it |
| exact signal time and importer state | `lifecycle.py` (8570b050) | `gpu.api_server_sigterm` = time taken at delivery, pid, outcome, and the action entry carries the same time; `gpu.importer_after_signal` = Nsight identity and `/proc` state right after the SIGTERM and after the termination pass |
| prebuilt load-generator contract | `load_generator/runner.py`, `runner.py`, `launcher/alignment.py` (8570b050) | `--frontend-provenance`: exact executable hash, pinned clean source and build-state record verified before any engine start; refusal on any mismatch; no cargo in this path; hash verified again when the launch ends (§12) |
| build-tool guard | `load_generator/runner.py`, `runner.py`, `launcher/alignment.py` (e32fbab7) | a prebuilt launch arms an interpreter audit hook: any process the launcher starts, from any module or helper, that names cargo, rustc or rustup is refused before it is spawned and recorded in `frontend_binary.json`; the development build function refuses while armed; a blocked request makes the launcher exit non-zero (§12) |
| retained stream vs active session | `preflight.py` | a `*.raw-preserve` directory passes only as *retained inactive*: retention record present, size and sha256 match now, stream not open in any process, recorded Nsight process gone, session-closed time recorded, and its name is not the planned capture's. Nsight's own `nsys-report-*.qdstrm`, a directory without a record, a mismatch, an open stream or a name collision still FAIL |
| adoption | `preflight.py --adopt-retained` | writes the retention record for a stream preserved before records existed, after checking the run's lifecycle record; used once for the retained V1 stream (the stream's sha256 was verified unchanged before and after) |

Unchanged, and covered by the unchanged pinned tests (`test_profiler_lifecycle.py`, 22 tests): the single stop request, importer
isolation, identity-verified termination, soft/hard deadline and GPU-held ceiling behaviour, control-file rules. The stream hash runs
only after the session has closed, so it cannot delay a release.

## 5. Evidence required and the frozen gates (`checks/e1_forced_release_gates.py`, evaluated by `postrun_e1_fr.sh`)

| gate | requirement | evidence |
|---|---|---|
| H | host identity agrees | lifecycle `host` / `host_identity` = run log start line = reservation node (`reservation.json`, pod Running) = host of every GPU observation; missing source INCONCLUSIVE, malformed or conflicting FAIL |
| FR1 | one stop request | `requests_sent == 1`, one `stop_requested` phase, one API-server "Stopping profiler" line; the request may end in an error only at or after the release request |
| FR2 | the registered trigger | exactly one control line, the registered text, result `release requested`, recorded phase `importing`. Read only after the import had ended, or never written because the import was already over: INCONCLUSIVE |
| FR3 | import active when the API server was signalled | (1) phase `importing` observed by the controller (`release_context`) and in at least two probe samples; (2) importer identity verified from `/proc` by the probe (pid and start time equal to the recorded Nsight identity); (3) import **confirmed in progress** immediately before the release (CPU time or progress advanced between probe samples); (4) the exact SIGTERM time lies after the import began and before any report. **If the import completed before the signal, or was not confirmed in progress: INCONCLUSIVE.** A controller record of an earlier phase, or an unverified importer identity: FAIL |
| FR3b | importer checked after the signal | controller's `importer_after_signal` and the probe's `after_signal` samples, both at or after the SIGTERM time, both for the recorded Nsight pid. Either missing: INCONCLUSIVE. The importer's state itself is evidence for the outcome, not a pass condition |
| FR4 | no destructive signal to the importer | every signalled pid is a recorded server process; Nsight's pid or process group is signalled only after a report was written, if at all; the API-server SIGTERM is recorded with its exact time, its pid, outcome `signalled`, and a matching action entry |
| FR5 | GPUs released in time, observed independently | registered sampler `e1_gpu_watch.py` (every 2 s) plus the driver's census: a held observation before the request, the first released observation at most **90 s** after it, none held later; the release is reported as the interval between the last held and first released observation; controller time separately |
| FR6 | session ended | terminal phase (`report_written` or `stop_failed`), Nsight gone, session-closed time recorded |
| FR7 | stream durable and unchanged | controller's final record present; durable copy on the shared filesystem stable over two full reads, equal to the final record and to the tray original. Not a completeness statement |
| FR8, FR8b | recovery and semantic completeness | offline import of the durable copy and sqlite export; recovered trace has iterations in 2100–2250, NVTX = 4 × iterations, kernel rows in 11.64–12.86 M, four processes with kernels, capture boundaries; **where an in-session export exists** it must also equal it in kernel rows, NVTX ranges, iteration indices, capture boundaries, per-process counts, every table's row count and capture metadata |
| FR9 | bounds | GPU-held ≤ 2700 s; release request → session closed ≤ 300 s; no escalation |
| S.* | science, identity, binary | configuration identity under the normalization policy; `binary_unchanged` = `checks/e1_frontend_binary_gate.py`, 16 checks against the pin: driver record (sha256 before and after the launcher, file inode/mtime/size before and after, no cargo `Compiling` line in the run log), launcher record `frontend_binary.json` (prebuilt mode, cargo not consulted, pinned sha256 before and after, build-tool guard installed with nothing blocked), and the executable at post-run time; a missing record is a FAIL; launcher recovery, completeness v2, extract, Nsight validation exit codes; closeout |

Verdict: any FAIL → **FAIL**; no FAIL but a gate not evaluable → **INCONCLUSIVE**; otherwise PASS. FAIL and INCONCLUSIVE are both a
STOP: no retry, no advance. FR7 can never stand in for FR8b. A successful launcher exit decides nothing. A run in which no forced
release took place because the import finished first is INCONCLUSIVE on every forced-release gate and can never be a PASS.
No threshold was changed in this revision: 90 s release deadline, iteration and kernel-row bands, 2700 s ceiling, 300 s session close.

Outcomes, registered in `E1-V2-FR-INPUTS.json`:
- **A** — the importer finishes after the server is gone (`report_written`). The in-session profile is assessed and must equal the
  offline recovery of the preserved stream.
- **B** — the importer ends without a report (`stop_failed`; the launcher exits non-zero by design). The profile is recovered from the
  durable stream through `launcher alignment profile --import-stream` on CPU and assessed by the same scientific checkers. No original
  export exists, so equality is not applicable; the absolute checks and the checkers decide.
- **Neither** — the importer still runs after the 300 s session-close timeout, or the GPUs are not released within 90 s: FAIL (§7).

## 6. Durability and recovery

The controller hard-links the stream when it first sees it, records every size sample, and takes the final size and SHA-256 after
the session closes. `postrun_e1_fr.sh` first copies the preserved stream to `<run>/preserved/` on the shared filesystem, verifies the
copy against the tray original, and only then imports. No script removes the tray original; after an accepted run its removal is an
operator decision, and after a failed one it stays.

## 7. If the server cannot release the GPUs safely

1. The controller's release action is the only automatic one: SIGTERM to the identity-verified API server, 60 s grace, SIGKILL to
   recorded server processes whose identity still matches. A changed identity is skipped and recorded, never signalled.
2. If the sampler still shows memory held 90 s after the request: FR5 fails. The operator runs the pinned verified cleanup of the
   pid registry (`e1_pids.py cleanup --execute`), which signals only registered, identity-matching processes. The preserved stream is
   already hard-linked, so Nsight being in that tree does not endanger the trace. Nothing is ever killed by pattern.
3. If GPUs are still held after that, or a process cannot be attributed: stop and report. The reservation pod is **kept** until the
   GPUs read idle, so the scheduler cannot place a pod on busy GPUs, even if that exceeds the 90-minute reservation; that excess is
   reported as a violation.
4. An importer still running after the session-close timeout is left running by the controller (recorded warning). The GPUs are
   already free at that point. It is ended only through the registry cleanup, after the stream is confirmed preserved.
5. No retry in any of these cases.

## 8. Ceilings

| resource | ceiling | enforced by |
|---|---|---|
| reservation | 90 min hard (exception: §7 item 3) | operator |
| GPU-held | 2700 s from the launch | controller, in any phase, not extendable |
| GPU release after the forced-release request | 90 s | gate FR5 (independent sampler) |
| finalisation | soft 300 s, hard 600 s after the stop request; session close 300 s | controller |
| temporary storage (`/tmp`, tray16) | 5 GB: new stream about 0.35 GB + post-run scratch about 2.4 GB (removed after counting) + the retained V1 stream 0.35 GB | preflight floor, monitor |
| persistent run storage | 5 GB: run directory about 2.7 GB (outcome A) + durable stream 0.35 GB | monitor, post-run |
| persistent including archive | 10 GB | closeout |
| cleanup | registry-verified only; closeout must be CLEAN; reservation deleted after GPUs idle | closeout script |

## 9. Preflight, regression and rehearsal results (CPU-only)

**Exact read-only driver-context preflight `preflight_e1_fr.sh` on tray16** (tmux `e1v2-ctxtest` + tee; every attempt kept in
`runs/v2/preflight/`):

| time (UTC) | commit | result |
|---|---|---|
| 2026-11-18 06:00:54–06:01:48 | d3c5c081 | 45 PASS, **1 FAIL** (`session_runner sha256 95a1526c… != provenance 7f6eb81d…`) |
| 2026-11-18 06:10:43–06:11:35 | 2207cee5 | 45 PASS, **1 FAIL** (same) |
| 2026-11-18 08:29:57–08:30:52 | aca5bd26 (code 8570b050) | **53 PASS, 0 FAIL** |
| 2026-11-18 13:14:35–13:15:35 | 565206ee (code e32fbab7) | **52 PASS, 0 FAIL** (three duplicate lines gone, two new test files pinned; 203 tests) |

The passing run includes: Stage 1.5 chain with `session_runner sha256 == provenance`; the prebuilt contract (`verify-prebuilt`: exact
pinned sha256, clean pinned source, build-state record, cargo not consulted); the launcher dry-run with `--frontend-provenance`;
inputs and the single-source trigger line; pinned trees; 46-file manifest; identity preflight 27 checks / 0 FAIL (retained V1 stream
reported as retained inactive); 165 tests with the CUDA guard line; GPU census (no compute process, 0 MiB); discovery CLEAN; registry
plan clean; storage. The reservation record is reported as not present yet (INFO); the driver refuses to start without it. The 08:30Z run listed three test files twice in the
pinned-tree loop (three repeated PASS lines); e32fbab7 lists each once. The 13:15Z run has the same checks with a 51-file manifest
and 203 tests.

**Regression, guarded CPU-only** (`tests/cpu_only.sh`; guard line in every log; GPUs 0 MiB afterwards), at 8570b050 and again at 565206ee:

| run | result | log (`runs/v2/cpu/`) |
|---|---|---|
| targeted suites, tray14 (19 files) | 351 passed, 1 skipped | `pytest_targeted_cpu_only_8570b050_tray14.log` |
| whole tier, tray14 | 5266 passed, 78 skipped, 6 failed (the same six unrelated tests as before) | `pytest_fullsuite_cpu_only_8570b050_tray14.log` |
| controller/preflight/gate/contract suites inside the tray16 preflight | 165 passed | `runs/v2/preflight/pytest_20261118T0829*.log` |
| targeted suites, tray14 (21 files), 565206ee | 389 passed, 1 skipped | `pytest_targeted_cpu_only_565206ee_tray14.log` |
| whole tier, tray14, 565206ee | 5304 passed, 78 skipped, 6 failed (the same six unrelated tests) | `pytest_fullsuite_cpu_only_565206ee_tray14.log` |
| suites inside the tray16 preflight, 565206ee | 203 passed | `runs/v2/preflight/pytest_20261118T131435Z.log` |

After all of these the executable in the worktree is still the pinned 7f6eb81d….

**Rehearsal of the V2 post-run on real data** (`runs/v2/rehearsal/`, 05:52–05:58Z, at 362e0609): unchanged record; the launcher's
`--import-stream` recovery on a V1-derived fixture gave the V1 counts and the scientific checkers passed. It predates the strengthened
gates. It is kept as it was and is superseded by the final rehearsal in §13, which ran the complete strengthened post-run.

## 10. Procedure (only after a GO naming the commit)

1. Reservation pod from the laptop; write `reservation.json` (pod, node, phase, observed time) into the run directory from `kubectl`.
2. Negative control, then the driver-context preflight `preflight_e1_fr.sh`: all PASS. Every output is kept, failures included.
3. Launch once: `tmux new -d -s e1v2-forced "E1_REPO=$S/wt-lifecycle bash $E1/run_e1_fr.sh"` through a neutral-path wrapper, from a
   commit whose pinned trees equal `V2-FR-PIN.json` (the preflight checks this) and with `sha256sum -c FREEZE_fr.sha256` clean. The
   driver refuses to start without a valid reservation record, registers itself, the GPU sampler and the trigger probe.
4. Monitor every 60 s (`monitor_e1_fr.sh`), no intervention except the abort procedure (§14) and §7.
5. Closeout `closeout_e1_fr.sh` → CLEAN; delete the reservation only with GPUs idle.
6. `postrun_e1_fr.sh` on tray16 → PASS, FAIL or INCONCLUSIVE; archive with checksums; report. Stop.

## 11. Not authorized

V2 and V3 GPU runs, the full E1 retry, the identification partition, fitting, E2–E6, Rust production integration, profiling
database writes, preset changes. This packet is a request, not an authorization.

## 12. Load-generator binary: incident, repair (option 2) and the launch contract

**Incident.** The V1 launcher rebuilt `session_runner` at launch (`runs/v1/ADDENDUM-session-runner-rebuild.md`). V1 ran with a
binary built from the pinned source in the lifecycle worktree (sha256 95a1526c…), not with the pinned bytes (7f6eb81d…).

**Why the launcher compiled although the preflight passed.** `alignment/runner.py::run_profile` called
`load_generator.build_session_runner()` before every capture, unconditionally, and that function runs
`cargo build --release --bin session_runner`. Whether cargo rebuilds depends on its own `target/` state, which no preflight looked at.
The preflight compared the binary's hash; cargo then replaced the binary one minute later, before the server started.

**Repair, done 2026-11-18 08:27Z on CPU (`runs/v2/binary/`):**

| step | result |
|---|---|
| rebuilt V1 executable preserved as incident evidence | `artifacts/phase1c-e1-v1-dsv41-flash-gb300-tp4-20261118-incident-session-runner/` (read-only): the executable, sha256 **95a1526c3a841eebf84d54eafa254cf24590f847fbba233603dc8f0e158aa549**, cargo's dep-info, the 249 `Compiling` lines of the V1 run log, `PROVENANCE.json`, `SHA256SUMS`. It is evidence, not a pin |
| pinned executable restored | copied from the reference worktree `wt-e1stage1`; reference sha256 = pin; after the copy the worktree's executable is sha256 **7f6eb81d45e6b98f3b6ebb56ab20b60ce8aaf4584aa3d4f25e3449b2d05a00c2** and `cmp`-identical to the reference |
| pinned source | req-frontend 5c20a259, clean, not modified |
| cargo build state | **cannot be made compatible with the pinned bytes in this path.** Shown twice: with the pinned bytes in place, the development command `cargo build` re-links (first test) or recompiles (after cargo's own artifact was replaced by the pinned bytes) and puts 95a1526c… back. The build is deterministic for a given path and differs between paths. The pinned bytes were restored after each test |
| build-state record | `target/release/session_runner.build-state.json` (copy in `runs/v2/binary/`): sha256, source commit, origin of the bytes, and the statement that cargo's state here is not compatible and only the prebuilt path may be used |

So requirement "restore or reconstruct compatible Cargo build state" could not be met literally, and nothing was faked to meet it.
Byte identity is instead guaranteed by never running cargo in an authorized launch.

**Launch contract (fail-closed).** With `--frontend-provenance BINARY-PROVENANCE.json` the launcher, before any engine or GPU start:
verifies the executable's full sha256 against the pin; verifies the source checkout is the pinned commit and unmodified; requires the
build-state record, describing these bytes and this source and not older than any source file; and refuses on any failure. It never
invokes cargo. It writes `load_generator/frontend_binary.json` (path, sha256, pinned sha256, source commit, build-state record) and
hashes the executable again when the launch ends; a change makes the launcher exit non-zero. Source equality is never accepted in
place of the executable's hash. The V2 driver always passes the flag and separately refuses to start on a non-pinned executable; the
preflight runs the same check; gate `S.binary_unchanged` combines the driver's and the launcher's records after the run.
Without the flag the launcher behaves as before (development: cargo build). A development build in this worktree therefore replaces
the executable, and the next preflight or authorized launch refuses until the pinned bytes are restored.

**CPU proof (`tests/test_load_generator_prebuilt.py`, 17 tests, CUDA-isolated; any cargo invocation fails the test):**

| case | result |
|---|---|
| valid binary, compatible build state | accepted; record complete; cargo not invoked |
| missing build state | refused ("build state missing"); binary untouched |
| stale cargo fingerprints | no effect on the prebuilt path: accepted, cargo not consulted, bytes unchanged. Stale build state in the contract's sense (a source file newer than the record, or a modified source file): refused |
| incorrect executable hash, identical source | refused ("hash mismatch … not sufficient"); also missing executable, a record for other bytes or another commit, a moved-on checkout, a malformed or truncated pin |
| binary replaced between preflight and launch | preflight check passes, launch check refuses |
| no permitted launch path invokes cargo | the real `run_profile` with a provenance file reaches the engine start without cargo and refuses before it on a mismatch; without the flag cargo is reached exactly once (development path); source check: `cargo` appears in one function, called from one branch |
| complete mocked launch | sha256 before = after = pin, bytes identical, record has both verification times; a binary replaced during the launch is detected and the launcher fails |

**Build-tool guard (e32fbab7).** "Never invokes cargo" was a property of one code path. It is now enforced for the whole launcher
process: `forbid_build_tools()` installs an interpreter audit hook when the prebuilt contract is requested, before the executable is
verified. From then on every process spawn in that interpreter (`subprocess`, `os.system`, `os.popen`, `os.exec*`, `os.posix_spawn`,
the launcher's own process supervisor), whichever module or helper asks for it, is checked; one that names `cargo`, `rustc` or
`rustup` (bare, by path, wrapped in another program, or inside a shell string) is refused before it starts and recorded. The
development build function refuses while the guard is armed. A blocked request is written to `frontend_binary.json` and makes the
launcher exit non-zero even if the helper swallowed the refusal. The guard cannot be removed from a running process. It does not
change the environment or the arguments of the server, the profiler or the load generator.

**CPU proof of the guard (`tests/test_load_generator_prebuilt.py`, now 41 tests; a tripwire `cargo`/`rustc`/`rustup` first on `PATH`
would leave a marker file if anything were really started):**

| case | result |
|---|---|
| every spawn route with the guard armed: argv, absolute path, wrapped (`uv run cargo …`), shell string, `check_output`, `os.system`, `os.popen`, `os.execv`, `os.posix_spawn`, the development build helper, the launcher's process supervisor | each refused and recorded; tripwire never started; other programs still run |
| the real `run_profile` with a provenance file | guard armed with nothing blocked when the engine start is reached; afterwards the development build and the development branch of `run_profile` refuse in that process |
| an indirect helper asks for `cargo build` during a mocked launch and swallows the error | refused; recorded in `frontend_binary.json`; the launcher exits non-zero; executable unchanged |
| static reachability over `launcher/alignment.py`, `launcher/exec.py`, `launcher/process*`, `alignment/runner.py`, `alignment/load_generator/`, `alignment/profiler/` (by called name, an over-approximation) | from the profile entry point the only reachable function that starts a build tool is the development build, whose first statement is the guard's refusal |
| commands that are not build tools (`git … .cargo/registry`, `…/.cargo/bin/uv run …`, the load generator, `git ls-files … Cargo.toml`) | not blocked |
| guard not armed (development path) | inert |

## 13. Final CPU-only gate rehearsal (2026-11-18 13:16–13:24Z, tray16, at 565206ee; `runs/v2/rehearsal_final/`)

`rehearse_e1_fr.sh` ran once, CPU-only (CUDA hidden, GPUs 0 MiB before and after), and met every expectation. It is not V2
evidence. The complete log is `rehearsal_tray16_20261118T132417Z.log`.

**13.1 The complete strengthened post-run on the preserved V1 fixture** (`v1_fixture/`). The fixture is the archived V1 run with
its report and export withheld, so the profile had to be recovered from the retained V1 stream; nothing about a release was added.
`postrun_e1_fr.sh` ran every step: durable copy, identity, launcher recovery, scientific checkers, binary gate, frozen gates.

| result | detail |
|---|---|
| verdict | **FAIL** (exit code 1), 19 gates: 11 PASS, 7 FAIL, 1 INCONCLUSIVE |
| forced-release gates, all FAIL | FR2 (no registered trigger line), FR3 (no release request, no SIGTERM time), FR3b (no importer check after a signal), FR4 (no release signal recorded), FR5 (no release request to measure from), FR9 (no release → session-close interval) |
| the record shows why | import phase recorded; `release_requested_at` null; no API-server SIGTERM record; no `release-gpus` control line; no release signal. With every other gate set aside the verdict is still FAIL |
| gates that do not need a forced release, all PASS on the real data | FR0, FR1, FR6, FR7 (347,628,953 B, sha256 8f5905d9…, durable copy = tray original), FR8 (offline import 27 s), FR8b (2160 iterations, 8640 NVTX ranges, 12,250,832 kernel rows, four processes), configuration identity, launcher recovery, completeness v2, extract, Nsight validation |
| other gates not PASS, each explained | H INCONCLUSIVE (the V1 record is schema 1 without a host field; V1 wrote no reservation record); S.binary_unchanged FAIL (V1 has neither binary record, and its launcher rebuilt the executable) |
| V1 evidence afterwards | retained stream sha256 unchanged; V1 archive re-verified against its `SHA256SUMS`; fixture copy removed |

`e1_fr_rehearsal.py expect-v1` checks this mechanically (`V1-FIXTURE-EXPECTATION.txt`): it fails if the verdict is PASS, if any
forced-release gate passes, if a recovery gate fails on the real data, if the record shows a release request, or if any other gate
is not PASS without a registered reason.

**13.2 Synthetic fixtures through the same command-line validators** (`synthetic/`, 50 fixtures, `SUMMARY.txt`; one gate text per
fixture, the gate JSON for the positive ones). The fixtures use the gate tests' builder with a fake importer and exporter; the
scientific checkers' exit codes are supplied, since they ran on real data in 13.1.

| category | fixtures | required | result |
|---|---|---|---|
| positive: import verified active by controller and probe immediately before the SIGTERM, signal delivery recorded, importer identity intact after it, GPUs released 6 s after the request, complete recovery (outcome A and outcome B); and a release observed at exactly 90.0 s | 3 | every one of the 20 gates PASS | **PASS, 20/20 gates, all three** |
| import completed before the SIGTERM (report written first; release context already terminal; trigger read only in a terminal phase; no release at all) | 4 | INCONCLUSIVE | INCONCLUSIVE ×4, no FAIL, never PASS |
| importer observations missing (no probe record; one sample; not confirmed in progress; no check after the signal by probe or by controller; probe confirmation after the signal) | 6 | INCONCLUSIVE | INCONCLUSIVE ×6 |
| importer observations wrong (identity not verified; another process watched; controller release context missing) | 3 | FAIL | FAIL ×3 |
| late GPU release (90.5 s; 132 s; never; not bracketed; held again; controller time only) | 6 | FAIL at FR5 | FAIL ×6 |
| binary mutation (changed during the launch; not pinned before it; replaced and restored; compiled during the launch; build tool requested; record without guard; records missing) | 7 | FAIL at S.binary_unchanged | FAIL ×7 |
| destructive signal to the importer (SIGINT before a report; SIGKILL to its process group; SIGTERM before the report in outcome A; signal to an unrecorded process; SIGTERM time not recorded) | 5 | FAIL at FR4 | FAIL ×5 |
| incomplete recovery (durable stream missing; stream ≠ final record; no final record; import fails on a truncated stream; too few kernels; NVTX ≠ 4 × iterations; recovered ≠ in-session export; launcher recovery failed; completeness checker failed) | 9 | FAIL | FAIL ×9 |
| recovery tools not supplied | 1 | INCONCLUSIVE | INCONCLUSIVE |
| further stop conditions (importer still running; trigger in the wrong phase; second stop request; host conflict; escalation; configuration identity failed) | 6 | FAIL | FAIL ×6 |

Totals: 3 PASS, 11 INCONCLUSIVE, 36 FAIL; 50 of 50 as registered. The 90.0 s and 90.5 s pair shows the release deadline is exactly
the frozen 90 s. The same matrix runs as a test (`tests/test_e1_fr_rehearsal.py`) inside the preflight.

**13.3 Pinned executable.** `session_runner` was sha256 7f6eb81d45e6b98f3b6ebb56ab20b60ce8aaf4584aa3d4f25e3449b2d05a00c2 before the
rehearsal, after it, and after the regression runs on tray14.

## 14. Freeze record

| frozen item | value |
|---|---|
| source | branch `exp/profiler-lifecycle`; code commit e32fbab7b39c66894f2ff87a02ecf45139183ee7; pin commit 565206ee405905826f82db998b1d2a333a589fae; launch commit: a descendant of the pin commit with the pinned trees of `V2-FR-PIN.json` (checked by the preflight). The commit that carries this packet adds documents and evidence only |
| manifests | `SCRIPTS_fr.sha256` (51 files), `INPUTS_fr.sha256` (trace, config, source config), `V2-FR-PIN.json` (trees, gitlinks, executable), `V2-FR-PIN-identities.json` (profiler, importer, server on tray16), `BINARY-PROVENANCE.json` (executable sha256 7f6eb81d…, source 5c20a259), all listed with this packet and `E1-V2-FR-INPUTS.json` in `FREEZE_fr.sha256` |
| workload | one launch, pass `e1_v2_forced_release`, tray `pod4-gb300-10-tray16`, 4 GPUs; config `configs_fr/e1_v2_forced_release/profile.json` (7f7e4f66…); trace `e1_dry.csv` (c6d8e13b…): 56 requests, two waves (b=8 at 0 s, b=48 at 18.6 s), 1024 output tokens; expected 2160 iterations, 12.25 M kernel rows |
| resource ceilings | §8: reservation 90 min hard; GPU-held 2700 s; GPU release ≤ 90 s after the request; finalisation soft 300 s / hard 600 s; session close 300 s; `/tmp` 5 GB; run directory 5 GB; 10 GB with the archive |
| trigger | §3: the probe's one registered control line on the first observation of phase `importing`, after importer identity and progress are confirmed; text in `E1-V2-FR-INPUTS.json` |
| acceptance gates | §5: H, FR1–FR9 (with FR3b, FR8b), S.* as implemented in `checks/e1_forced_release_gates.py` (sha256 4ebe272c…) and `checks/e1_frontend_binary_gate.py` (sha256 f7bc3189…); verdict rule PASS / FAIL / INCONCLUSIVE; no threshold changed since 2cd7fcea |
| abort procedure | below, with §7 |

**Abort procedure.**

1. *Before the launch.* Any FAIL in the negative control or the preflight, a missing or foreign reservation record, a non-pinned
   executable or a trace/config guard failure: nothing starts (the driver exits 2–6 before any engine start). Report; no launch.
2. *During startup and capture* (monitor every 60 s; it only prints): `ABORT-CONDITION` on startup over 30 min, no stop request
   40 min after the launch, engine-error lines before shutdown, a GPU process outside the driver's tree, run directory over 5 GB,
   `/tmp` under 50 GB free, or a controller escalation. The controller itself releases the GPUs at the 2700 s ceiling in any phase.
3. *Operator abort action.* Only the pinned registry cleanup, `checks/e1_pids.py cleanup --registry <run>/pids/owned.tsv --execute
   --grace 30 --log <run>/cpu/cleanup_<time>.log`, which signals registered processes whose identity (pid, start time, boot id,
   uid) still matches. Nothing is killed by pattern. Uncertain ownership: stop and report.
4. *After the trigger.* §7: if the GPUs are still held 90 s after the request, FR5 has failed; registry cleanup; the reservation
   pod stays until the GPUs read idle.
5. *Always.* An abort is a FAIL / STOP: no retry, no second launch. Any operator control line besides the probe's registered one
   fails FR2. The preserved stream is never removed; closeout and the post-run still run for the record; the reservation is
   deleted only with the GPUs idle.
