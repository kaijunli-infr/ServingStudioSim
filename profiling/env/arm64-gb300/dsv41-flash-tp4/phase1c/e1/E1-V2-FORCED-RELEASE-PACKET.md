# E1 V2 authorization packet — forced GPU release while the Nsight import is running (CPU-only preparation; authorizes nothing)

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
| branch / code commit | `exp/profiler-lifecycle` 362e0609 (controller + preflight changes, validators, V2 scripts), pins d3c5c081, driver/post-run rebuild guard 2207cee5 (pinned trees unchanged) |
| pinned trees (`V2-FR-PIN.json`) | `alignment` 8124376add276be3393ed57a1ebf91f75f4daa12, `launcher` 5a1b662f12e878798eb5c8be04ca445d4c6977b5, test files and fixtures by blob id |
| submodules | vLLM fork 0ab18711, req-frontend 5c20a259 |
| load generator binary `session_runner` | **OPEN, blocks the preflight (§12).** Provenance pin 7f6eb81d…; the binary in the worktree is 95a1526c…, rebuilt from the same source by the V1 launcher |
| identities (`V2-FR-PIN-identities.json`, tray16) | Nsight CLI 2025.6.3.541 sha256 2d53d710…, `QdstrmImporter` sha256 b30312a6…, server interpreter and vLLM import as in V1 |
| inputs (`INPUTS_fr.sha256`) | trace `e1_dry.csv` c6d8e13b…; config `configs_fr/e1_v2_forced_release/profile.json` 7f7e4f66…; source config df949648… |
| scripts, checks, tests (`SCRIPTS_fr.sha256`) | 43 files |
| expected values, trigger, limits, outcomes | `E1-V2-FR-INPUTS.json` |

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
- **Trigger.** `v2_release_probe.sh`, a registered background job, polls `lifecycle.json` every 0.5 s. On the **first** observation
  of phase `importing` it reads `/proc` for the recorded Nsight pid (alive, same start time, child processes), and only if it is
  alive appends the one pre-registered line to `lifecycle.control`:
  `release-gpus V2 forced release during import (pre-registered trigger: first observation of phase importing)`.
  It records what it saw in `probe.json`. It writes once, signals nothing, and writes nothing if the stop is already terminal.
- The controller reads the control file once per second and runs its unchanged `release_gpus` action: identity-verified SIGTERM to
  the API server, SIGKILL to recorded server processes still alive after the grace period, never Nsight.
- Expected latency from import start to release request: at most about 2 s of a 27 s import.

## 4. Controller and preflight changes in this commit (item A/B of the GO)

| change | where | effect |
|---|---|---|
| launch host identity | `lifecycle.py` | `host`, `host_identity` (hostname, boot id, launcher pid, source) written from the launcher process at controller construction; record schema 2 |
| stream observations kept | `lifecycle.py` | every distinct (size, mtime) sample is appended to `stream.observations` with time and phase; `stream.size` stays "latest sample" |
| final stream record | `lifecycle.py` | once, after the Nsight session has closed and the phase is terminal: `stream.final` = size, mtime, sha256 (stat before and after the read) and whether it changed after the last observation; also written as `RETENTION.json` next to the preserved stream |
| completeness wording | `lifecycle.py`, gates | every final record and retention record carries: completeness is **not** established by a stable size or a matching checksum |
| release context | `lifecycle.py` | at a release request: phase, import progress, Nsight alive, report written — evidence only, no decision uses it |
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
| FR2 | the registered trigger | exactly one control line, the registered text, result `release requested`, recorded phase `importing` |
| FR3 | import active at release | controller: `release_context` (phase `importing`, Nsight alive, no report), release time after `importing` began and before any report; **and** the probe's own `/proc` observation of the live Nsight process |
| FR4 | no destructive signal to the importer | every signalled pid is a recorded server process; Nsight's pid or process group is signalled only after a report was written, if at all |
| FR5 | GPUs released in time, observed independently | registered sampler `e1_gpu_watch.py` (every 2 s) plus the driver's census: a held observation before the request, the first released observation at most **90 s** after it, none held later; the release is reported as the interval between the last held and first released observation; controller time separately |
| FR6 | session ended | terminal phase (`report_written` or `stop_failed`), Nsight gone, session-closed time recorded |
| FR7 | stream durable and unchanged | controller's final record present; durable copy on the shared filesystem stable over two full reads, equal to the final record and to the tray original. Not a completeness statement |
| FR8, FR8b | recovery and semantic completeness | offline import of the durable copy and sqlite export; recovered trace has iterations in 2100–2250, NVTX = 4 × iterations, kernel rows in 11.64–12.86 M, four processes with kernels, capture boundaries; **where an in-session export exists** it must also equal it in kernel rows, NVTX ranges, iteration indices, capture boundaries, per-process counts, every table's row count and capture metadata |
| FR9 | bounds | GPU-held ≤ 2700 s; release request → session closed ≤ 300 s; no escalation |
| S.* | science, identity, binary | configuration identity under the normalization policy; `binary_unchanged` (the launcher compiled nothing and left `session_runner` byte-identical); launcher recovery, completeness v2, extract, Nsight validation exit codes; closeout |

Verdict: any FAIL → **FAIL**; no FAIL but a gate not evaluable → **INCONCLUSIVE**; otherwise PASS. FAIL and INCONCLUSIVE are both a
STOP: no retry, no advance. FR7 can never stand in for FR8b. A successful launcher exit decides nothing.

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

## 9. Preflight and rehearsal results (CPU-only)

**Regression, guarded CPU-only** (`tests/cpu_only.sh`; guard line in every log; GPUs 0 MiB afterwards), at 2207cee5:

| run | result | log (`runs/v2/cpu/`) |
|---|---|---|
| targeted suites, tray14 (16 files: launcher-adjacent, pinned lifecycle/preflight/gates, corrected gates, identity, host, retained-stream, forced-release) | 316 passed, 1 skipped | `pytest_targeted_cpu_only_2207cee5_tray14.log` |
| whole tier, tray14 | 5231 passed, 78 skipped, 6 failed (the same six unrelated tests as before: `cargo`-dependent corpus test, gdn numerical test, four energy tests that fail with CUDA hidden) | `pytest_fullsuite_cpu_only_2207cee5_tray14.log` |
| lifecycle/preflight/gate suites, tray16 (retained V1 stream present in `/tmp`) | 130 passed, 0 failed | `pytest_targeted_cpu_only_2207cee5_tray16.log` |

The pinned V1 test that failed on tray16 after the V1 run now passes there, with the V1 stream still in place: it carries a
retention record and is classified as retained inactive. It was not deleted or modified (sha256 verified before and after adoption).

**Read-only driver-context preflight `preflight_e1_fr.sh` on tray16** (tmux `e1v2-ctxtest` + tee; outputs kept in `runs/v2/preflight/`):

| time (UTC) | commit | result |
|---|---|---|
| 2026-11-18 06:00:54–06:01:48 | d3c5c081 | **45 PASS, 1 FAIL** |
| 2026-11-18 06:10:43–06:11:35 | 2207cee5 | **45 PASS, 1 FAIL** |

The one FAIL, both times, is inside the Stage 1.5 chain: `session_runner sha256 95a1526c… != provenance 7f6eb81d…` (§12). Everything
else passes: inputs and trigger line, pinned trees, 43-file manifest, identity preflight 27 checks / 0 FAIL (the retained V1 stream is
reported as retained inactive, INFO), 130 tests with the CUDA guard line, GPU census (no compute process, 0 MiB), discovery CLEAN,
registry plan clean, storage. **The preflight is therefore not passing, and V2 is not launchable as packaged until §12 is decided.**

**Rehearsal of the whole V2 post-run on real data** (`runs/v2/rehearsal/`, CPU-only, tray16, 05:52–05:58Z). A fixture was built from the
real V1 artifacts with the report, export and parsed outputs removed and the lifecycle record rewritten to outcome B. It is a
rehearsal fixture, not V2 evidence. Result: every step ran. Durable copy verified; the launcher's `--import-stream` recovery rc 0 and
the recovered profile has 8640 NVTX ranges and 12 250 832 kernel rows, the V1 values; completeness v2, extract and Nsight validation
PASS on the recovered profile; gates FR1, FR6, FR7, FR8, FR8b PASS. As expected for V1 data, FR2, FR3, FR4, FR5, FR9 FAIL (no forced
release, no sampler), H is INCONCLUSIVE (no host in the V1 record, no reservation record), and the identity check FAILs on the
fixture's scratch `log_dir`, which is the policy correctly rejecting an output outside the run-directory layout. Overall FAIL, as it
must be. The fixture was removed; the V1 archive re-verified.

## 10. Procedure (only after a GO naming the commit)

1. Reservation pod from the laptop; write `reservation.json` (pod, node, phase, observed time) into the run directory from `kubectl`.
2. Negative control, then the driver-context preflight `preflight_e1_fr.sh`: all PASS. Every output is kept, failures included.
3. Launch once: `tmux new -d -s e1v2-forced "E1_REPO=$S/wt-lifecycle bash $E1/run_e1_fr.sh"` through a neutral-path wrapper. The
   driver refuses to start without a valid reservation record, registers itself, the GPU sampler and the trigger probe.
4. Monitor every 60 s (`monitor_e1_fr.sh`), no intervention except §7.
5. Closeout `closeout_e1_fr.sh` → CLEAN; delete the reservation only with GPUs idle.
6. `postrun_e1_fr.sh` on tray16 → PASS, FAIL or INCONCLUSIVE; archive with checksums; report. Stop.

## 11. Not authorized

V2 and V3 GPU runs, the full E1 retry, the identification partition, fitting, E2–E6, Rust production integration, profiling
database writes, preset changes. This packet is a request, not an authorization.

## 12. Open blocker found by the preflight: the load-generator binary pin

**Finding.** The V1 launcher rebuilt `session_runner` at launch. `run.log` of the V1 run shows a full `cargo build` (`Compiling …`)
right after the `start` line, and the binary's mtime is 2026-11-18 02:54:12Z, one second before `launch_started_at`. The lifecycle
worktree had been provisioned on 2026-11-17 with the binary only, without cargo's fingerprints, so cargo treated it as stale. The V1
preflight ran before that and saw the pinned binary; the driver's `start` line also recorded the pinned hash.

| binary | sha256 | built |
|---|---|---|
| pinned (provenance; used by the accepted `e1_nsys_val` and b=8 captures) | 7f6eb81d45e6b98f3b6ebb56ab20b60ce8aaf4584aa3d4f25e3449b2d05a00c2 | 2026-11-15, `wt-e1stage1` |
| what V1 actually ran with; now in `wt-lifecycle` | 95a1526c3a841eebf84d54eafa254cf24590f847fbba233603dc8f0e158aa549 | 2026-11-18 02:53–02:54Z by the V1 launcher |

Both come from the same clean source commit 5c20a259 with the same toolchain (cargo and rustc 1.98.1) and have the same size. The
build embeds its directory: two scratch builds of the same commit in two paths gave two further hashes, and each binary contains its
worktree path. So the bytes differ by build path, not by source. This is a **deviation of the V1 run** that was not known when V1 was
reported: its "all binary hashes match" prerequisite held at preflight time and stopped holding at launch. It is recorded in
`runs/v1/ADDENDUM-session-runner-rebuild.md`; the original V1 report and verdict are unchanged.

**Why it blocks V2.** `preflight_e1_fr.sh` keeps the provenance check, so it fails on the rebuilt binary. Nothing was re-pinned.

**Decision needed** (either makes the preflight pass; both keep the source commit):
1. *Re-pin to the rebuilt binary 95a1526c….* It is what V1 ran with, and cargo's state in this worktree is now consistent, so a
   further launch should not rebuild. Costs: the load generator's bytes differ from the accepted reference captures, although its
   source does not.
2. *Restore the pinned binary 7f6eb81d… together with cargo's build state* from `wt-e1stage1`. Keeps byte identity with the reference
   captures. Risk: cargo's fingerprints include paths, so the launcher may rebuild again; this must be shown on CPU first.

Whichever is chosen, V2 now detects a rebuild: the driver hashes the binary before and after the launcher and counts cargo
`Compiling` lines, and gate `S.binary_unchanged` fails the run if anything was compiled or the binary changed.
