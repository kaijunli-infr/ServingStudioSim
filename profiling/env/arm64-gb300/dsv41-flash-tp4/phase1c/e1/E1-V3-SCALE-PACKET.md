# E1 V3 packet — intermediate trace size (24.5 M kernel rows), normal stop (PROPOSED FOR FREEZE; authorizes nothing)

**Status 2026-11-18: final freeze prepared, awaiting review.** Approved for final freeze preparation at 28dc7f01; this revision adds
the operational safeguards of the following review (deadline semantics, enforced resource limits with a registered supervisor,
importer-uncertainty rules, end-to-end rehearsal of the execution path). Tools, tests, pins and CPU-only rehearsals are complete
(§12). The content listed in §14 is pinned by `FREEZE_v3scale.sha256`, which the preflight verifies. A launch needs its own GO
naming a commit. V2's frozen packet, gates and PASS are untouched. The earlier
`E1-V3-SCALE-PACKET-DRAFT.md` and `E1-V3-SCALE-INPUTS.draft.json` are kept as committed and are superseded by this file.

## 1. Question

At twice the V1/V2 trace size and with a normal stop: how long do the flush and the import each take, how much host memory and
storage do they need, and do stream preservation, offline recovery and the independent measurements hold? V3 is a calibration run
with gates. It is not a forced-release test and not part of the E1 measurement.

## 2. Source, executable, workload, recipe

| item | value |
|---|---|
| branch / code commit | `exp/profiler-lifecycle`, code commit **22422bd2** (`V3S-PIN.json`); launch commit: a descendant with the pinned trees, clean, checked by the preflight and by gate G0 from the driver's `launch_source.json` |
| pinned trees | `alignment` 7b4df784a5b6daff7c792e81cbc6503ba2c806ec and `launcher` 5acab28980ef9974bf4cb6dcd676481c357e655d (both unchanged since e32fbab7: no controller, launcher or shutdown change), 21 test files, the CPU-only guard and its runner, fixtures |
| submodules | vLLM fork 0ab18711335780e062d65994a3243da0281d2081, req-frontend 5c20a259618976ecc360bda84abcc3653a04006e (gitlink = checkout, checked) |
| load-generator executable | `session_runner` sha256 **7f6eb81d45e6b98f3b6ebb56ab20b60ce8aaf4584aa3d4f25e3449b2d05a00c2** (`BINARY-PROVENANCE.json`), launched only through the prebuilt contract with the build-tool guard, as in V2 |
| profiler and server identities | `V3S-PIN-identities.json` (tray16): Nsight CLI 2025.6.3.541 sha256 2d53d710…, `QdstrmImporter` sha256 b30312a6…, server interpreter and vLLM import; equal to the V2 pins |
| pass / tray | `e1_v3_scale`, `pod4-gb300-10-tray16`, 4 GPUs, one launch |
| trace | `traces/e1_v3_scale.csv` sha256 d9d2dc77b316b7862f12e860a49b84010b385946e04f1b2b2ab80a9746710eb8: the 56 accepted rows of `e1_dry.csv` (c6d8e13b…) twice; the second pair is the first with wave index 1 and +80 s. Waves: b=8 at 0 s, b=48 at 18.627 s, b=8 at 80 s, b=48 at 98.627 s; 112 requests, 32 768 input and 1024 output tokens each |
| config | `configs_v3scale/e1_v3_scale/profile.json` sha256 fc91bddbf0fd6b34966948464c8cf2c3dc4887539edb71a5103d2dbf80de71da = the V2 config with only `name`, `log_dir` and the trace path changed (preflight check): model, server arguments, openai backend, routing hook on, stream false, trace-timed arrivals, Nsight flags, no delayed start, no `tmp_dir`, lifecycle bounds soft 300 / hard 600 / stall 300 / session close 300 / max extension 120 / GPU-held ceiling 2700 s, stream preserved |
| controller behaviour | unchanged: 60 s API-server grace (controller default; not overridden), identity-verified server-only termination, no signal to Nsight |
| inputs manifest | `INPUTS_v3scale.sha256` (trace, config, and the V2 config and accepted trace they derive from); `E1-V3-SCALE-INPUTS.json` (expected values, deadlines, limits, ceilings, four-cell plan) |
| scripts manifest | `SCRIPTS_v3scale.sha256`, 64 files |

Not used: the Phase 0 recipe; four b=48-only waves.

## 3. Expected values (twice the V2 measurement of the same 56 rows; bands, not a model)

| quantity | expected | band / note |
|---|---|---|
| iterations | 4320 | 4200 – 4500 |
| NVTX forward ranges | 17 280 | 4 per iteration |
| kernel rows | 24 501 664 | ± 5 %: 23 276 580 – 25 726 747; four ranks |
| per cell (four cells: b=8 and b=48, waves 0 and 1) | pure-decode iterations 1008 / 928 | checker floor 400 per cell (pre-registered rule of `e1_completeness_v2.py`) |
| request timings / routing arrays | 112 / 112 | exact |
| raw stream / report / in-session export | ≈ 0.70 / 0.78 / 3.9 GB | V2: 0.348 / 0.392 / 1.96 GB |
| run root | ≈ 6.3 GB | V2 with tables: 3.16 GB (measured in the rehearsal) |
| post-run scratch | ≈ 3.8 GB | V2 size: 1.92 GB measured in the rehearsal (2 s sampling) |
| importer memory | unknown at this size | same 12.25 M-row stream offline: 8.7 GB, 8.3 GB and 4.9 GB in three samplings (the sampler reads a lower bound of the true peak); 61 GB was sampled at ≥ 49 % of the 216 M-row import |

## 4. Phase clocks, predictions, deadlines

Clocks (wall-clock seconds; one named record each; `checks/e1_v3scale_gates.py`):

| clock | record |
|---|---|
| `t_stop` | lifecycle `stop_requested_at` |
| `t_accepted`, `t_importing`, `t_report` | lifecycle `phase_history` entries `stop_accepted`, `importing`, `report_written` |
| `t_imp_first`, `t_imp_gone` | import probe: `importer.first_seen_at`; first `SETTLING` transition (importer no longer present) |
| `t_sigterm` | lifecycle `actions`: first SIGTERM to the recorded API-server pid |
| `t_closed` | lifecycle `session_closed_at` |

| phase | definition | prediction (recorded, not a gate) | deadline | beyond it |
|---|---|---|---|---|
| stop accepted | `t_accepted − t_stop` | < 1 s | 30 s | G4 FAIL |
| flush | `t_importing − t_stop` | 60 – 140 s | 300 s | G4 FAIL |
| import | `t_imp_gone − t_imp_first` | 27 – 110 s | 300 s | G4 FAIL |
| stop → report | `t_report − t_stop` | 140 – 190 s | 600 s | G4 FAIL; at 600 s the controller's hard deadline releases the GPUs (G3 FAIL), importer untouched, trace recovered offline |
| session close | `t_closed − t_sigterm` | < 5 s | 300 s | G4 and G10 FAIL |

A phase passes at or below its deadline. Deadline semantics, as confirmed in review:

- The flush (300 s), import (300 s) and stop → report (600 s) limits all apply at once. A run with each phase inside its own
  deadline still fails if the total is over 600 s, and the reverse (fixtures in §12).
- The total is measured from the **original** stop request (`t_stop`). Later stop calls are no-ops in the controller and restart
  nothing (controller test).
- The controller's soft deadline, 300 s after the stop request, only records a warning. It is informational: reported in G3 and G4,
  failing neither.
- A soft warning cannot extend the hard deadline. The controller's hard deadline is `t_stop + 600 s` and moves only by an operator
  `extend` line, which V3 does not register. G4 requires the recorded hard deadline to be exactly the registered total from the
  stop request with extension 0 and the policy value 600; G3 fails on any control line. A hard-deadline or ceiling escalation
  fails G3.
- The controller evaluates its hard deadline, its GPU-held ceiling and the control file **only inside its finalisation loop**, that
  is after the stop request (controller test). Before the stop request the bounds are the supervisor's (§5, §10): the GPU-held
  ceiling and the resource limits lead to the pinned registry cleanup.

V2's 27–29 s import is not scaled: the predictions come from the two Phase 0 captures of
25.6 M rows (stop → report 160 s and 163 s; same Nsight version and flags, different workload recipe) and the 12.25 M-row split
(flush 61.9–63.8 s, import 27.0–28.9 s). The split at this size has never been measured; measuring it is the purpose of V3.

## 5. Independent evidence required

| evidence | instrument (registered through `e1_pids.py spawn`; none signals anything) | rule |
|---|---|---|
| GPU release | `checks/e1_gpu_watch.py`, 2 s interval | each observation's stamp precedes its queries and the sampler then sleeps the interval, so a reading was taken within [stamp, next stamp − 2 s]. The release is reported as the last "held" window and the first "released" window relative to `t_sigterm` and judged on the **upper edge** of the latter (≤ 90 s). No tighter figure is claimed. A gap over 10 s or an unavailable reading across the release is INCONCLUSIVE |
| importer state | `checks/e1_import_probe.py`, observer mode, 0.5 s | exactly one importer (pinned executable, this session's stream or report); activity only from its own CPU ticks and I/O counters, two consecutive intervals. Five outcomes are distinguished (§6). Worker CPU time and the reported percentage are never evidence |
| host memory | `checks/e1_mem_watch.py`, 1 s | MemAvailable and the identified importer's resident memory. Bounded policy: floor 64 GiB available, ceiling 128 GiB importer; on a breach the sampler writes `MEMORY-BREACH` and records it (§10) |
| storage | `checks/e1_storage_watch.py`, 2 s, during the run and again during the post-run | temporary = every stream artefact and recovery scratch in `/tmp` (each inode once); persistent = run root (and extract tables in the post-run). Peaks with their times; ceilings 10 GB and 8 GB; `STORAGE-BREACH` on a breach |
| supervision | `checks/e1_supervisor.py`, 1 s, registered in its own registry `pids/supervisor.tsv` | the registered consumer of `MEMORY-BREACH` and `STORAGE-BREACH`; also reads the samplers' own summaries (a breach recorded there without a marker file counts), the age of each sampler's last sample (over 60 s: that monitor is lost) and the GPU-held ceiling before a stop request. One response per run (§10). It sends no signal itself. Its record must be complete, with no response and all monitors effective, for G11 to pass |
| measured means | gate G11 | samples must cover the run (launch → session closed) and the post-run's offline import/export with no gap over 30 s, and the importer must have been identified by the memory sampler in both. **An unmeasured peak is INCONCLUSIVE** |
| host identity | all records | lifecycle = run log = reservation node = every sampler |

## 6. Importer states the protocol distinguishes (gate G5)

| state | evidence | result |
|---|---|---|
| verified activity | importer identified; `ACTIVE` reached (two consecutive intervals of its own counters advancing) | required for PASS |
| present, temporarily inactive | `NO_PROGRESS` intervals with identity intact | recorded with their count; not a failure if `ACTIVE` was reached and the import completed |
| identity unknown or ambiguous | `NOT_FOUND`, `AMBIGUOUS`, or no probe record | INCONCLUSIVE |
| completed | importer gone, report present, non-empty, stable; importer had been seen | required for PASS (completeness itself is G8/G9) |
| terminated | importer gone or replaced without a report, or a zombie | FAIL |

Any recorded signal to the importer, to its agent, to the Nsight CLI before the report, or to a pid that is not a recorded server
process is a G5 FAIL.

Importer uncertainty during the run:

- An importer-specific PASS (G5, the import phase of G4, importer memory in G11) needs exactly one identified importer. Unknown or
  ambiguous identity is INCONCLUSIVE, never PASS and never a guess.
- The run continues in that case only while the independent safety monitors are effective: host MemAvailable (which bounds any
  importer, identified or not), storage, GPU state, and the supervisor that watches their heartbeats. A lost monitor is an abort
  cause (§10).
- No process is ever signalled because it looks like the importer. Without a unique identification by the probe, the importer is
  not signalled by anyone, in any procedure.
- All hard deadlines and cleanup obligations apply unchanged.

## 7. Stream preservation and semantic recovery

The controller hard-links the stream when it first sees it and records its final size and complete SHA-256 after the session closes.
The post-run copies it to durable storage and verifies the copy against the tray original, then (G7) requires final record =
durable copy = tray original over two full reads. G8 imports the durable copy offline and exports it; G8b requires iterations,
NVTX ranges (4 per iteration), kernel rows and four ranks within §3 and capture boundaries present, and equality with the in-session
export in ten respects (rows, ranges, iterations, index range, kernel and NVTX boundaries, per-rank counts, every table's row count,
capture metadata, export product and schema). G9 runs the four-wave completeness and routing-record checker (`--plan dry` with the
four cells of `E1-V3-SCALE-INPUTS.json`, `--checksum`), the extract, the Nsight validation and the configuration identity. No script
removes a stream; the V1 and V2 originals in `/tmp` stay.

## 8. Resource ceilings

| resource | nominal | hard ceiling | enforced / measured by |
|---|---|---|---|
| reservation | ≈ 50 min | 90 min (the pod is kept beyond it only while GPUs are not idle; that is a violation and is reported) | operator; G10 from `reservation.json` and `reservation_deleted.json` |
| GPU-held | ≈ 1800 s | 2700 s from the launch, any phase, not extendable | controller; G10 (controller time and sampler upper edge) |
| GPU release after the server SIGTERM | ≈ 2 s | 90 s (upper edge) | G6 |
| stop → report | 140 – 190 s | 600 s | controller; G4 |
| temporary storage, tray16 `/tmp` | ≈ 5.2 GB peak in the post-run (retained V1/V2 streams 0.70 + new stream 0.70 + scratch ≈ 3.8) | 10 GB measured peak | storage sampler; G11; preflight projection |
| persistent run storage | ≈ 6.3 GB | 8 GB measured peak | storage sampler; G11 |
| persistent including archive | ≈ 12.6 GB | 16 GB | `archive_e1_v3scale.sh` |
| host memory | importer unknown | preflight ≥ 448 GiB available and no smaller cgroup limit; floor 64 GiB available; importer ≤ 128 GiB (lowered from 256) | preflight; memory sampler; supervisor; G11 |

No operator extension overrides a hard ceiling. The only control line registered for V3 is the supervisor's abort line; any accepted
control line fails G3, so an abort is always a FAIL. No ceiling was raised in this revision.

Host memory on tray16 (2026-11-18, idle): MemTotal 958 GiB, MemAvailable 862 GiB (558 GiB of it page cache), no swap, no cgroup
memory limit on the session, two NUMA nodes of about 479 GiB. The server's host footprint under load has never been sampled; V3's
sampler will measure it. The importer ceiling of 128 GiB is twice the largest importer memory ever sampled (61 GB at ≥ 49 % of the
216 M-row import), 15 % of the idle available memory and about a quarter of one NUMA node. With the 64 GiB floor it leaves at least
670 GiB of the idle available memory for the server and the cache; the preflight requires 448 GiB available (128 + 64 + 256 for
the server). The floor, not the importer ceiling, is what bounds an importer that could not be identified.

## 9. Gates (`checks/e1_v3scale_gates.py`; PASS only if every gate passes; exit 0 / 1 / 3)

| gate | PASS | FAIL | INCONCLUSIVE |
|---|---|---|---|
| G0 source and executable | launch record: real mode (a rehearsal launch always fails), pinned trees and gitlinks, clean, descends from the pinned commit; 16 binary checks (pinned bytes before, after and at post-run; same inode/mtime/size; no compile line; prebuilt mode; guard armed, nothing blocked) | any mismatch | launch record, pin or a binary record missing |
| G1 host identity | all sources agree | a conflict | a source without a host |
| G2 one stop request | one sent, returned OK, before `t_sigterm`; one "Stopping profiler" and one "Profiler stopped." line | error return, duplicate, late return, a line missing | server log or `t_sigterm` unavailable |
| G3 phases in order | exactly stop_requested, stop_accepted, capture_ended, stream_written, importing, report_written; no hard-deadline or ceiling escalation, no control line, no release request (a soft-deadline warning is recorded) | anything else | — |
| G4 phase deadlines | every phase of §4 at or below its deadline; controller hard deadline = 600 s from the original stop request, extension 0 | one exceeded; hard deadline extended or not the registered total | a boundary or the deadline record not observed |
| G5 importer observed | §6 | terminated; a forbidden signal | identity unknown or ambiguous; activity never established; completion not observed |
| G6 GPU release interval | upper edge ≤ 90 s; not held again; both censuses released | later; held again; released before the SIGTERM; a census still held | no bracketing observation, missing samples, no SIGTERM time, no census |
| G7 stream durable | §7 | any difference or missing record | — |
| G8 / G8b offline import and recovered trace | §7 | import or export fails; any absolute or equality check false | tools not supplied |
| G9 scientific completeness | four exit codes 0 and identity PASS | one non-zero or identity FAIL | one not run |
| G10 bounds | GPU-held ≤ 2700 s by both measures; session close ≤ 300 s; reservation ≤ 90 min | one exceeded | one not measured |
| G11 storage and memory | measured peaks within §8; no breach file; supervisor record complete, no response, monitors effective | a peak beyond its ceiling; a breach file; a supervisor response; a lost monitor | a peak not measured (§5); supervisor record missing or incomplete |
| G12 closeout | closeout exit 0; reservation-deleted record with no GPU pod left on the node | closeout not clean; a GPU pod left | closeout not run; record missing |

A FAIL or INCONCLUSIVE is final for the launch: no retry, no reinterpretation after the results are seen.

## 10. Abort procedure, supervisor response and host-memory policy

**Automatic response (registered supervisor; one per run; bounded; identity-verified; it sends no signal itself).** Causes: a
memory or storage breach (marker file, or the sampler's own record if the marker is missing); a safety monitor without a sample
for 60 s; the GPU-held ceiling passing before any stop request. All causes present at that moment are recorded together; later
ones are recorded without a second response.

| when | response | who signals | bound |
|---|---|---|---|
| finalisation in progress (stop requested, phase not terminal, no release yet, controller record fresh) | R1: the registered line `release-gpus V3 resource-limit abort (registered supervisor response)` is appended to the control file | the unchanged controller: SIGTERM to the identity-verified API server, SIGKILL to recorded server processes still alive after the 60 s grace; never Nsight | cause seen within 1 s of the marker; line consumed within 1 s (controller poll); acknowledgement awaited 10 s; server gone ≤ 60 s later |
| any other phase (startup, capture, after the report), or R1 not acknowledged within 10 s | R2: the pinned registry cleanup `e1_pids.py cleanup --registry pids/owned.tsv --execute --grace 30 --kill9` | `e1_pids.py`: SIGTERM, after 30 s SIGKILL, only to registered processes and their descendants whose identity (pid, start time, boot id, uid) still matches | cleanup finished within about 45 s |

Neither response can reach the Nsight agent or the importer: they are not registered and not descendants of a registered process,
and the controller never signals Nsight. R2 before the stream exists gives up the trace; R1 and any R2 after the controller has
hard-linked the stream leave it preserved. Either response makes the run a FAIL (G3 or G11).

**Operator procedure.**

1. *Before the launch.* Any FAIL in the negative control or the preflight, a missing or foreign reservation record, a non-pinned
   executable, a trace/config guard failure: nothing starts (the driver exits 2–6 before any engine start).
2. *During the run* (monitor every 60 s; it only prints `ABORT-CONDITION`): startup over 30 min; no stop request 40 min after the
   launch; engine-error lines before shutdown; a GPU process outside the driver's tree; run root over 8 GB; `/tmp` under 50 GB free;
   `MEMORY-BREACH` or `STORAGE-BREACH` present; a supervisor response; the supervisor silent for 30 s; a hard-deadline or ceiling
   escalation (a soft-deadline warning is printed as a note); any accepted control line.
3. *Operator action, if the supervisor has not already acted or is itself silent.* Only the pinned registry cleanup,
   `checks/e1_pids.py cleanup --registry <run>/pids/owned.tsv --execute --grace 30 --log <run>/cpu/cleanup_<time>.jsonl` (and the
   same for `pids/supervisor.tsv` if the supervisor is still alive). Nothing is killed by pattern. Uncertain ownership: stop and
   report.
4. *Host memory.* Preflight: at least 448 GiB available. Run: floor 64 GiB available, ceiling 128 GiB importer; a breach is
   answered by the supervisor as above, which ends the server and frees its memory while the stream stays hard-linked. If
   memory is still critical afterwards: only when the probe identified **exactly one** importer **and** a durable copy of the
   stream has been made and verified by complete SHA-256 may the operator end that importer, by registering that one pid
   (`e1_pids.py register --registry <run>/pids/importer.tsv --label importer <pid>`, identity as recorded by the probe) and running
   the same verified cleanup on that registry. If the importer was not uniquely identified, no importer is signalled: stop, report,
   keep the reservation. Never automatic.
5. *After the server SIGTERM.* If the GPUs are still held 90 s later: G6 has failed; registry cleanup; the reservation pod stays
   until the GPUs read idle.
6. *Forced release by the controller* (hard deadline or ceiling, both evaluated only after the stop request): the frozen
   `release_gpus` action. Known from V2: the engine dies at once and the GPUs are free within about 2 s; the API server waits on
   the open stop request until the identity-verified SIGKILL after the 60 s grace. That behaviour is retained.
7. *Always.* An abort is a FAIL: no retry. A stream that exists is never removed; closeout, post-run and archive still run for the
   record; the reservation is deleted only with the GPUs idle, and `reservation_deleted.json` is written from `kubectl`.

## 11. Procedure (only after acceptance of this packet and a GO naming the commit)

1. Reservation pod from the laptop; write `reservation.json` (pod, node, phase, creation time) into the run directory.
2. Negative control, then the driver-context preflight `preflight_e1_v3scale.sh`: all PASS. Every output kept.
3. Launch once: `tmux new -d -s e1v3-scale "env -u E1_V3S_MODE E1_REPO=$S/wt-lifecycle bash $E1/run_e1_v3scale.sh"` through a
   neutral-path wrapper (`E1_V3S_MODE` must be unset: the preflight reports it, and gate G0 fails on a rehearsal launch).
4. Monitor every 60 s (`monitor_e1_v3scale.sh`); no intervention except §10.
5. Closeout `closeout_e1_v3scale.sh` → CLEAN; delete the reservation with the GPUs idle; write `reservation_deleted.json`.
6. `E1_CLOSEOUT_RC=<closeout exit> postrun_e1_v3scale.sh` on tray16 → PASS, FAIL or INCONCLUSIVE.
7. `archive_e1_v3scale.sh <operator logs>` → VERIFIED; report; stop.

## 12. Tools, tests and CPU-only rehearsal

(Completed by the next commit, after the end-to-end rehearsal of this revision.)

## 14. Freeze record (proposed)

`FREEZE_v3scale.sha256` pins: this packet, `SCRIPTS_v3scale.sha256` (64 files), `INPUTS_v3scale.sha256`,
`E1-V3-SCALE-INPUTS.json`, `V3S-PIN.json`, `V3S-PIN-identities.json`, `BINARY-PROVENANCE.json`. Any change to one of them needs a new
review.

## 15. Not authorized

V3 GPU launch, V2 retry, the full E1 retry, the identification partition, fitting, E2–E6, Rust production integration, profiling
database writes, preset changes.
