# Importer-specific import-state probe: design, tests, evidence (CPU-only, 2026-11-18)

Code: `checks/e1_import_probe.py` (new; the frozen V2 probe `checks/e1_release_probe.py` is untouched). Tests:
`tests/test_e1_import_probe.py` (23). Evidence in this directory. Synthetic fixtures and the preserved V2 stream only; no GPU.

## 1. What the importer actually is (observed)

A CPU-only Nsight session on a pure-CPU target (`synth_a.out`, sampler `synth_procwatch_a.jsonl`, GPUs 0 MiB before and after):

| process | parent | session / group | note |
|---|---|---|---|
| `nsys profile …` (the CLI; the pid the controller records) | the launching shell | the launcher's | CPU time barely moves during the import |
| `nsys-launcher`, `nsys-tee`, the target | the CLI | the launcher's | this is the tree the V2 probe sampled |
| `nsys --start-agent --session-name profile-<CLI pid> …` (the agent) | **pid 1** | **its own** | started 0.6 s after the CLI |
| `QdstrmImporter --version` | the agent | the agent's | lives a few tens of ms |
| `QdstrmImporter --input-file /tmp/nsys-report-XXXX.qdstrm --output-file <report>.nsys-rep --force-overwrite` | **the agent** | the agent's | the importer; holds both files open |

So the importer is neither a descendant of the recorded CLI nor in its process group. This settles hypothesis (f) of the
e1_main_v3 failure analysis ("`QdstrmImporter` is a child of the nsys CLI") as false for this Nsight version, and it explains why
V2's samples of the CLI tree contain no importer. Limits of this observation: one session, plain `profile` mode on a CPU target; the
capture-range mode used with the server has not been observed directly (it needs CUDA). The agent's session name embedding the CLI
pid was seen once.

What the importer's own counters look like (offline import of the preserved V2 stream, 12.25 M rows, `import_final.out`, 26.2 s):

| phase of the import | own CPU ticks | own `rchar` | own `wchar` | report file |
|---|---|---|---|---|
| first ≈ 2 s: read the stream | rising | 0 → 346 MB | ≈ 0 | absent |
| ≈ 2 – 18 s: process | ≈ 100 ticks/s, steady | flat | a few hundred bytes/s, at times flat | absent |
| ≈ 18 – 26 s: write | rising | flat | 0 → 369 MB | appears, grows to 392 MB |

Peak resident memory 8.7 GB. The report does not exist for the first 70 % of the import, and reads stop after 2 s: only the
importer's own CPU time advances throughout. An importer killed at 10 s (`import_killed.out`; this fixture's own child process,
killed through its process handle) leaves **no report file**.

## 2. Design

1. **Identification.** A process of this uid whose `/proc/<pid>/exe` is exactly the pinned importer path and which names this
   session's stream (`--input-file`) or report (`--output-file`) in its arguments, or holds one of them open. Exactly one match is
   required. `--version` invocations, importers of other sessions, other executables with the same name and other users' processes
   do not match. The importer's parent is recorded, with whether it is an Nsight agent whose session name is `profile-<recorded CLI
   pid>`; that link is reported, not required (seen once, §1).
2. **Evidence of activity: the importer's own counters only.** A sampling interval qualifies when the same importer (pid and start
   time) is alive at both ends and its own CPU ticks (`/proc/<pid>/stat`) or its own I/O counters (`/proc/<pid>/io`) advanced.
   Nothing else can qualify: the probe never reads another process's CPU time into the decision, and the report size and the
   controller's reported progress are recorded with their provenance but never counted.
3. **Samples and provenance.** Every sample goes to a JSONL file with its time and the importer's pid, start time, state, ticks,
   I/O counters, parent, executable and match reasons, the report's size and mtime, the phase and the reported progress. The summary
   record names the source of each field (`provenance`), the importer's identity and every state transition with its time.
4. **States.**

   | state | meaning |
   |---|---|
   | `ACTIVE` | importer alive and the last `--min-progress` (default 2) intervals all qualify |
   | `NO_PROGRESS` | importer alive, identity intact, fewer than that many consecutive qualifying intervals: a temporary lack of progress; it can return to `ACTIVE` |
   | `SETTLING` | importer no longer present; waiting three samples for the report to stop changing |
   | `COMPLETED` | importer gone (or never seen) and the report present, non-empty and stable. Whether Nsight has logged "Generated:" is recorded; the exit status is not observable from `/proc` and the record says so |
   | `TERMINATED` | importer gone, replaced (same pid, other start time) or a zombie, and the report missing or empty |
   | `NOT_FOUND` | no process matches and there is no report |
   | `AMBIGUOUS` | more than one process matches; none is chosen |
5. **Fail closed.** "Import active" exists only as `ACTIVE`. In trigger mode the registered line is written once, on the first
   `ACTIVE`, after one more check that the same importer is alive with the pinned executable; in every other case it is withheld with
   a recorded reason (never found, ambiguous, completed, terminated, changed at the final check, or not established within
   `--confirm-s`). Observer mode writes nothing but its own records. The probe never sends a signal.

## 3. Tests (23, `tests/test_e1_import_probe.py`, fake `/proc`)

| requirement | tests |
|---|---|
| identifies the actual importer | exact executable and session files; `--version`, other session, other executable, other uid rejected; match by open files; agent link true / false / unknown; stream path and CLI pid taken from the lifecycle record |
| importer-specific evidence only | V2-shaped tree (CLI flat at 168 ticks, workers ticking, reported progress rising, no importer) → `NOT_FOUND`, trigger withheld; importer present but idle while workers tick, progress and report size rise → `NO_PROGRESS`, never `ACTIVE`; unit test of the interval rule field by field |
| timestamped samples with provenance | sample rows, provenance table, importer record |
| completed / temporary lack of progress / terminated | completed after activity; completed before any activity (trigger withheld); finished before the first observation; `ACTIVE → NO_PROGRESS → ACTIVE`; gone without a report and with an empty report; pid reuse; zombie |
| fails closed | two matching importers; importer changes between the decision and the write; not established within the window; observer mode never writes a control file; "Generated:" recorded but not required; CLI argument validation |
| two consecutive importer intervals before `ACTIVE`; configurable, never zero | dedicated tests |

## 4. On the real importer (preserved V2 stream, tray14, CPU)

| run | result | file |
|---|---|---|
| offline import under the probe | identified by arguments and open stream; `ACTIVE` 1.2 s after the first sighting; `COMPLETED`, report 391 797 430 B; probe exit 0 | `import_final.out`, `import_probe_final.{json,jsonl,log}` |
| same, importer killed at 10 s (the fixture's own child) | `ACTIVE`, then `SETTLING`, then `TERMINATED` (no report); probe exit 1 | `import_killed.out`, `import_probe_killed.*` |
| earlier run with the probe before its last change (CLI pid default) | same as the first | `import_normal.out`, `import_probe_normal.*` |

## 5. Not covered

- The probe has not run inside a real profiling session with the server: identification there relies on the argument pattern seen
  in §1 and on the stream path the controller records. If the importer's arguments differ in capture-range mode, the probe reports
  `NOT_FOUND` and establishes nothing (fail closed), it does not guess.
- `COMPLETED` cannot see the importer's exit status; a complete report is established by the offline validation, as in V2.
- The probe is not wired into any driver; that belongs to a V3 packet freeze.
