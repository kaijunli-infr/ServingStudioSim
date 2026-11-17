# E1 V1 authorization packet — bounded validation of the profiler-lifecycle controller (CPU-only preparation; nothing here authorizes GPU use)

Prepared 2026-11-17 under the GO "CPU-only preparation of a final V1 authorization packet" (provisional acceptance of
`phase1c-e1-audit` 9a28358c and `exp/profiler-lifecycle` 56c28ce3). It asks for **one** GPU launch, `e1_v1_lifecycle`, and nothing
else. Supersedes packet v3 §7 and §10 where they differ (GPU-held limits, V1 recipe details); v3 §1–6, §8–9 and §11–12 stand.
Companion documents on the audit branch: `E1-AUTHORIZATION-PACKET-v3.md`, `runs/e1_main_v3/E1-MAIN-V3-FAILURE-ANALYSIS.md`.
Everything below was produced on CPU; the tray's GPUs were read (`nvidia-smi`), never used.

## 1. Isolation (GO item 1)

| item | value |
|---|---|
| branch | `exp/profiler-lifecycle` only; worktree `$S/wt-lifecycle` on rack 10 (`S=/mnt/shared/home/dc/workspace/serving-studio`) |
| commits | 56c28ce3 (accepted patch) → **ece6dacf** (this packet's code: ceiling, control file, stream discovery, identity preflight, V1 inputs/scripts/gates/tests) → **the packet commit (HEAD of `exp/profiler-lifecycle` at push; named in the README and the final report)** (documentation, pins, CPU preflight results; no code) |
| merges | none. The branch is not merged into `phase1c-e1-audit`, any frozen branch (test-ss, phase1-planning, phase1a-costmodel, phase1b-seqsim, phase1c-prep, phase1c-e1-packet, phase1c-e1-stage1, phase1c-e1-stage15, kevin/dsv41-gb300-main) or any production branch; the frozen heads were re-verified unchanged after the push |
| what V1 executes | the launcher of `exp/profiler-lifecycle` at a commit whose `alignment/` and `launcher/` trees equal the pinned ids of §2 (the preflight compares tree ids, so documentation-only commits on top of ece6dacf are allowed; a code change is not) |
| worktree provisioning (CPU, 2026-11-17 20:09Z) | submodule checkouts cloned with shared objects from the main clone (`alignment/profiler/vllm` at 0ab18711, `alignment/load_generator/req-frontend` at 5c20a259), the 3350 git-ignored vLLM build outputs copied from the accepted Stage 1.5 tree, `session_runner` copied (sha 7f6eb81d… = BINARY-PROVENANCE.json); tracked tree clean |

## 2. Exact pins (GO item 2) — `V1-PIN.json`, `V1-PIN-identities.json`, `INPUTS_v1.sha256`, `SCRIPTS_v1.sha256`

| pinned | value | checked by |
|---|---|---|
| source commit (code) | **ece6dacf** on `exp/profiler-lifecycle` | `preflight_e1_v1.sh` C (tree ids), `pin_v1.sh` |
| tree `alignment/` | `b2be218bbb09bd6b21fc9909b30af2e1850224ed` | same |
| tree `launcher/` | `5a1b662f12e878798eb5c8be04ca445d4c6977b5` | same |
| blobs `tests/test_profiler_lifecycle.py`, `tests/test_profiler_preflight.py`, `tests/test_e1_lifecycle_gates.py`; tree `tests/fixtures` | in `V1-PIN.json` | same |
| submodule gitlinks | vllm fork 0ab18711335780e062d65994a3243da0281d2081; req-frontend 5c20a259618976ecc360bda84abcc3653a04006e | Stage 1.5 preflight D |
| `session_runner` | sha256 7f6eb81d45e6b98f3b6ebb56ab20b60ce8aaf4584aa3d4f25e3449b2d05a00c2 (`--stream` present) | Stage 1.5 preflight D |
| Nsight CLI | `/usr/local/bin/nsys` → `/opt/nvidia/nsight-systems/2025.6.3/target-linux-sbsa-armv8/nsys`, "NVIDIA Nsight Systems version 2025.6.3.541-256337736014v0", sha256 2d53d710fead2076f4e13617b6c6f8b7976965508e82901bc2db5b42f4c30dc4 | identity preflight `profiler.nsys` |
| QdstrmImporter | `/opt/nvidia/nsight-systems/2025.6.3/host-linux-armv8/QdstrmImporter`, sha256 b30312a66165fdf1aee0da6469c4800fa792e385c74f93d04ce9a188375abb8b, same install root as the CLI | `profiler.importer`, `profiler.importer_matches_cli` |
| server interpreter | `$S/gb300/fork-env-v0.31.0/venv/bin/python` → `/mnt/shared/home/dc/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/bin/python3.12`, sha256 1b3b0c6017252761abf8f14dfd14a889c690a0dd473e199fcf81766ddc1c379d | `server.identity` |
| vLLM fork | vllm 0.31.1.dev17+g0ab187113 imported from `$S/wt-lifecycle/alignment/profiler/vllm` (fork head 0ab18711) | `server.vllm_import`, `server.identity` |
| Nsight recipe | prefix `nsys profile --trace-fork-before-exec=true --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --cuda-event-trace=false --force-overwrite=true -o <stem> --capture-range=cudaProfilerApi --capture-range-end=stop`, then the server argv of the accepted `e1_nsys_val` launch record (identical, incl. `--enable-return-routed-experts --profiler-config.profiler=cuda`); env identical (no `TMPDIR`) | `recipe.*` checks against `…/e1_nsys_val/profile/vllm/*_launch.json`; `test_v1_config_expands_to_the_accepted_nsys_val_recipe` |
| inputs | trace `traces/e1_dry.csv` c6d8e13b7229ffd40cfede127bd57ad345aa17213b7590d39f9b2ef697f60158 (56 rows); config `configs_v1/e1_v1_lifecycle/profile.json` **5461ff34f017101563602e57b504285be563fa3b955d7182ff76df5ebec4c3e3**; source `configs_v2/e1_nsys_val/profile.json` df949648… (accepted a62e40bd) | `INPUTS_v1.sha256` |
| scripts, checks, tests, fixtures | `SCRIPTS_v1.sha256`, 26 files, file sha256 a50440ea4caacff1c0485a6b4bda7b2dc31a18b6f092eea024232386138e6c73 | preflight C |
| weights, corpus, caches, driver 595.71.05, venv versions | unchanged Stage 1 checks | Stage 1 host preflight |

## 3. The one-launch V1 workload and its expected trace coverage (GO item 3)

Recipe = `e1_nsys_val` (accepted 2026-11-15) with the profiler armed before the first request (as `e1_b8_recovery` and `e1_main_v3`):
one vLLM TP4 server, hook on (`capture_routed_experts`), openai backend, `stream: false`, warm-up, trace-timed replay of the frozen
dry-run trace. Config differences from the accepted source: `name`, `log_dir`, the removed 10 s start delay, and the lifecycle
bounds (`finalize_*`, `gpu_held_ceiling_s`, `preserve_raw_stream`) — nothing else (`E1-V1-INPUTS.json`, config identity check).

| wave | requests | KV | output | trace arrival | expected iterations (accepted single-wave captures) | expected pure-decode iterations |
|---|---|---|---|---|---|---|
| `e1_b8_kv32k_w0` | 8 | 32 768 | 1024 | 0.0 s | 1040 (e1_b8_recovery) | 1008 |
| `e1_b48_kv32k_w0` | 48 | 32 768 | 1024 | 18.627 s (after the b=8 wave ends at ≈ 9 s) | 1120 (e1_nsys_val) | 928 |
| **total** | **56** | | | replay ≈ 51 s | **≈ 2160** (acceptance band 2100–2250; NVTX forward ranges = 4 × iterations) | 1936 |

Expected trace size: kernel rows ≈ 12.25 M (5 887 244 + 6 363 588; band ± 5 %), `.nsys-rep` ≈ 0.39 GB, `.sqlite` ≈ 2.0 GB, raw
`.qdstrm` ≈ 0.3 GB in `/tmp` (preserved by hard link, same bytes), routing 56 npy, 56 request timings. Stop-to-report from the
calibration of six accepted captures (45 s @ 5.9 M rows, 63 s @ 6.4 M, 160–163 s @ 25.6 M, 567 s @ 97 M; the lost 216 M-row capture
> 1230 s): **≈ 90 s predicted, 180 s worst case** against a 300 s soft and 600 s hard deadline.

Nominal timeline: startup 22–25 min (autotune ≈ 19 min) → warm-up + capture ≈ 3 min → stop/flush/import ≈ 1.5 min → server stop,
GPUs free ≈ 27–28 min after the launch → sqlite export + parse ≈ 4 min on CPU → driver exit ≈ 32 min. The two probe lines are
written by `v1_control_probe.sh` while the stop runs (§8). Every Nsight capture so far announced its stream ("Generating
'/tmp/nsys-report-xxxx.qdstrm'"), so the `stream_written` phase and the hard-link preservation are exercised by V1.

## 4. Ceilings (GO item 4)

| ceiling | value | enforced by | on breach |
|---|---|---|---|
| **hard reservation ceiling** | **90 min** from pod Running to pod deleted (nominal ≈ 45 min) | operator (laptop) | registry cleanup (`e1_pids.py cleanup --execute`), then delete the pod; the run is a failed attempt |
| **GPU-held ceiling** | **2700 s (45 min)** from the launch (`launch_started_at`) to GPUs released; `nsys.gpu_held_ceiling_s: 2700` | **controller** (release action in any phase; `extend` cannot move it); plus operator abort if `nvidia-smi` still shows memory 46 min after the launch | controller: SIGTERM to the identity-verified API server, grace 60 s, SIGKILL recorded survivors; importer left alone; stream preserved; gate L6 FAIL → STOP |
| finalisation | soft 300 s / hard 600 s after the stop request; stall 300 s; max extension 120 s; session close 300 s | controller | escalation record / release action / warning; gate L6b, L8 → STOP |
| stage limits for the operator | startup "Application startup complete" ≤ 30 min; stop request ≤ 10 min after "measurement begins" | operator via `monitor_e1_v1.sh` ABORT-CONDITION lines | registry cleanup, never by pattern |
| **temporary-storage ceiling** | **5 GB in `/tmp`** (tray root NVMe: stream ≈ 0.3 GB + its hard link = same bytes); floors: `/tmp` ≥ 100 GB free at preflight, 50 GiB warning at the stop | preflight, controller, monitor | ABORT-CONDITION; the preserved directory `/tmp/<name>.raw-preserve/` is removed by the operator only after `postrun_e1_v1.sh` PASS |
| **persistent-storage ceiling** | **5 GB run directory** (`logs/dsv41_flash_gb300_tp4_e1_v1/e1_v1_lifecycle`, expected ≈ 2.9 GB); **10 GB** on CephFS including the archive copy; report-dir floor 20 GiB at the stop; CephFS ≥ 1000 GB at preflight | monitor, collect step | ABORT-CONDITION / no archive copy until reviewed |

## 5. The controller changes neither the frozen workload nor the profiling recipe (GO item 5)

Evidence, all CPU:

1. `tests/test_profiler_lifecycle.py::test_launch_recipe_is_unchanged_against_the_frozen_e1_main_v3_launch` and
   `tests/test_profiler_preflight.py::test_v1_config_expands_to_the_accepted_nsys_val_recipe`: the server argv, the nsys flags and
   the nsys configuration fields that the V1 config expands to are identical to the accepted launch records (output stem excepted).
2. `alignment.profiler.preflight` checks `recipe.server_argv`, `recipe.nsys_prefix`, `recipe.nsys_executable`, `recipe.env` against
   `e1_nsys_val`'s launch record on the tray — PASS in the CPU preflight of §6.
3. `checks/e1_config_identity.py` between `configs_v2/e1_nsys_val` and `configs_v1/e1_v1_lifecycle` with the allow-list `name`,
   `log_dir`, `nsys.capture_start_delay_seconds` and the eight lifecycle keys — PASS; the same check runs post-run on the launch record
   (`postrun_e1_v1.sh` §4 and §5).
4. By construction: the controller is created after `Popen` and acts only after the workload has ended (`/stop_profile`); before
   that it only reads the server log for pids. `ignore_sighup()` is set in the launcher after the children exist (own session). No
   `TMPDIR` is set (no `tmp_dir`), so the server environment is byte-identical to the accepted launch (`recipe.env`). The request
   schedule, trace, backend, hook flag and `session_runner` binary are unchanged (`INPUTS_v1.sha256`, Stage 1.5 preflight D).

## 6. Read-only preflight of process identities (GO item 6) — `alignment/profiler/preflight.py`, run by `preflight_e1_v1.sh` D

What it verifies without starting anything: the Nsight CLI (resolved path, version string, sha256) and the `QdstrmImporter` it will
use (path, sha256, same install root) against the pins; the server interpreter (realpath, sha256) and the vLLM module it imports
(version, file under the checked-out fork source root, fork git head) via a CPU-only probe with `CUDA_VISIBLE_DEVICES=""`; the
recipe (§5); the cleanup primitives on this host (`procident`: own identity readable, ancestry resolution, `signal_verified` refuses
the own pid and pid 1, detects a stale start time, `nvidia-smi --query-compute-apps` works); the controller's log parsers against the
accepted `e1_nsys_val` server log (API/EngineCore/Worker pids found; stop_accepted → capture_ended → stream_written → import_start →
import_progress → stop_returned found in order); the policy (`validate()`, ceiling explicit); and the locations (`lifecycle.json`,
`lifecycle.control`, the run's `log_dir` absent; `/tmp` free and without own `nsys-report-*.qdstrm` / `*.raw-preserve` leftovers;
report-dir free space).

CPU preflight result on **pod4-gb300-10-tray16-f3** at **2026-11-17 22:36:33–22:37:25Z** (driver context: tmux `e1v1-*` + tee into the run root, the
same way the driver runs it): **32 PASS, 0 FAIL (Stage 1/1.5 chain PASS; identity preflight 26 checks, 0 FAIL; 43 tests passed on the tray; discovery CLEAN; registry plan clean)** — log `runs/v1/preflight/preflight_v1_tray16_driver_context_20261117T223725Z.log`, identity record
`runs/v1/preflight/identity_20261117T223633Z.json`, pytest log `runs/v1/preflight/pytest_20261117T223633Z.log`. Two earlier attempts the same evening are recorded in the final report, not kept as files: 22:30Z 1 FAIL (`config trace path`: the generated config still pointed at the main clone's copy of the trace; fixed by pointing it at this tree's `traces/e1_dry.csv`, same bytes) and 22:34Z 2 FAIL (`launcher tree has uncommitted tracked changes` from that fix before it was committed; the code commit was amended, the pins regenerated). The passing run is on the committed tree ece6dacf.

## 7. Proof requirements (GO item 7) — evaluated by `checks/e1_lifecycle_gates.py` from `lifecycle.json`, any FAIL = STOP

| proof | gate | evidence field(s) | pass condition |
|---|---|---|---|
| stop-request idempotence | L1 | `http.requests_sent`, `http.duplicate_requests_suppressed`, `http.status` | exactly one POST left the launcher (`requests_sent == 1`); the runner's `finally` path re-requested and was suppressed (`≥ 1`); `status == ok` |
| phase ordering | L2, L2b | `phase_history[]` (`phase`, `at`), `stop_requested_at` | phases strictly increasing in `stop_requested → stop_accepted → capture_ended → stream_written → importing → report_written`, timestamps monotone, final `report_written` |
| report completeness | L3, L3b | `report.path/size`, `profile_result.validation.ok/kernel_rows/nvtx_forward_ranges` | non-empty `.nsys-rep`; validation ok; rows in 11.64–12.86 M; NVTX/4 in 2100–2250 (both waves) |
| preserved-stream recovery | L4, **L10** | `stream.path/size/preserved_path/preserved_size/discovered_by`; offline `QdstrmImporter -i <preserved> -o <scratch>` | hard link made and its size equals the stream's; the offline import of the preserved stream (run by `postrun_e1_v1.sh` **before** anything is deleted) produces a report within ± 1 % of the in-session report's size |
| no destructive signal to the importer | L5, L5b | `actions[]` (pid, action, reason, at), `nsys.pid`, `server.api_server` | the only signals are `SIGTERM` (reason `shutdown`) to the identity-verified API server after `report_written`; none to the nsys pid before the report |
| GPU-held / finalisation bounds | L6, L6b | `launch_started_at`, `gpu.released_at`, `stop_requested_at`, `report.written_at`, `policy` | held ≤ 2700 s; stop → report ≤ 600 s |
| control file | L7, L8c | `control_commands[]`, `deadlines.hard_extension_s` | exactly the two probe lines, both accepted; `hard_extension_s` equals the sum of accepted `extend`s (60) |
| no escalation, clean warnings, session closed | L8, L8b, L9 | `escalations[]`, `warnings[]`, `nsys.alive`, `server.alive`, `session_closed_at` | empty; no hard-link or descendant warning; nsys and server gone |
| scientific gates (unchanged thresholds) | S.* | exit codes of `e1_completeness_v2.py --plan dry`, `costmodel.extract`, `e1_nsys_validation.py`, `e1_config_identity.py` (launch record vs accepted run), recipe check on the launch record | all 0 |

Unit evidence for the same properties (CPU): `test_repeated_stop_calls_send_exactly_one_http_request`, phase ordering in every
scenario test, `test_import_failure_with_recoverable_raw_stream_then_offline_import`,
`test_raw_stream_found_by_descriptor_scan_is_preserved_before_nsight_announces_it`, `test_hard_deadline_during_import_stops_the_server_not_the_importer`.

## 8. Control-file commands: tested, every action recorded (GO item 8)

CPU (`test_control_file_every_command_is_recorded_with_its_result`, `test_control_file_release_gpus_with_reason_is_an_identity_verified_server_stop`,
`test_gpu_held_ceiling_binds_in_any_phase_and_no_extension_moves_it`):

| line written | recorded result |
|---|---|
| `note operator watching the import` | `noted` |
| `extend 60 V1 control-file test` | `hard deadline extended by 60 s (operator)` (+ "GPU-held ceiling unchanged at …" when a ceiling is set) |
| `extend abc not a number` | `rejected: seconds not a number` |
| `extend 60` | `rejected: usage is 'extend <seconds> <reason>'` |
| `extend 1000 would exceed the cap` | `rejected: total extension 1060 s would exceed max_extension_s 300` |
| `release-gpus` | `rejected: a reason is required` (nothing signalled) |
| `frobnicate the importer` | `rejected: unknown command` |
| `note partial` … ` line completed` (two writes) | consumed once, as `note partial line completed` → `noted` |
| `release-gpus operator: budget exhausted` | `release requested`; `actions`: SIGTERM to the API server pid with identity, reason `operator: release-gpus …`; `gpu.released_at` set; nsys never signalled |

Every record carries `at`, `command`, `phase`, `result`; releases add to `actions[]`, deadline changes to `deadlines`, ceiling/hard/soft
events to `escalations[]`; `lifecycle.json` is rewritten atomically after each. On the GPU run, `v1_control_probe.sh` (a registered
background job started by `run_e1_v1.sh` through `e1_pids.py spawn`) writes `note V1 operator probe: control file read` and
`extend 60 V1 operator probe: extension logged` once `lifecycle.json` shows `stop_requested_at`; gate L7 requires exactly these two
records, accepted, and L8c that the 60 s is the whole extension.

## 9. Any gate failure is a STOP (GO item 9)

* Pre-launch: any `FAIL` line in `preflight_e1_v1.sh` (Stage 1/1.5 chain, inputs, pinned trees, identity preflight, test suites,
  discovery, registry plan, storage) → the driver exits before `uv run` (rc 3/4) and nothing is launched.
* Post-run: `postrun_e1_v1.sh` folds the scientific checkers' exit codes into `checks/e1_lifecycle_gates.py`; **one FAIL anywhere →
  `V1 ACCEPTANCE: STOP`** (exit 1): the run directory and the preserved stream are kept for review, nothing is relaunched, V2 is not
  prepared, and the stop is reported with the gate table (`cpu/V1-GATES.txt`, `.json`). A STOP is not a retry permission: a new GO
  naming the corrected commit is required for any further GPU use.
* The only acceptable V1 outcome is PASS on every gate; "partially informative" results are recorded but do not advance the programme.

## 10. Reconciling the 90-minute and 100-minute GPU-held limits (GO item 10)

Packet v2 and v3 §6 said "GPU-held ≤ 90 min" (a planning worst case: startup 25 + capture 16.5 + finalisation 46); v3 §7 listed
"GPU-held total 100 min, operator, `release-gpus`" (an abort trigger). Two numbers for one quantity, neither enforced by code, and the
per-stage operator limits of v3 §7 (startup 30 + workload 25 + finalisation hard 50 = 105 min) exceed both.

Decision for this packet and for any future full retry (not requested here):

1. **One GPU-held quantity**: launch (`launch_started_at`, server spawn) → `gpu.released_at` (server processes gone and no compute
   pid). It is recorded in `lifecycle.json` and enforced by the controller through `nsys.gpu_held_ceiling_s`, in any phase, with the
   same release action as the hard deadline; the operator's `extend` cannot move it (`max_extension_s` only shapes the finalisation
   hard deadline).
2. **V1: ceiling 2700 s (45 min)**, nominal 27–28 min, enforced by the controller; the reservation ceiling 90 min is the outer bound.
3. **Full retry (e1_main_v4, when and if authorized): ceiling 100 min enforced by the controller; the 90-minute figure is retired
   as a limit and kept only as the planning expectation** (≈ 63 min nominal, ≤ 90 min expected worst case, reported after the run).
   The stage limits are made consistent with it: startup ≤ 30 (operator), stop request ≤ 25 min after measurement begins
   (operator), finalisation hard deadline 50 min (controller), ceiling 100 min (controller). Whichever comes first binds:
   with nominal startup (23–25) and capture (16.5) the 50-min hard deadline binds at ≈ 90–92 min GPU-held, inside the ceiling; if
   startup or capture run long, the ceiling binds first and the finalisation budget shrinks accordingly — the operator sees this
   coming in `deadlines.gpu_held_ceiling_at` vs `deadlines.hard_at`. Proposed v4 config values: `finalize_soft_deadline_s 1800`,
   `finalize_hard_deadline_s 3000`, `finalize_stall_timeout_s 900`, `finalize_max_extension_s 1200`, `gpu_held_ceiling_s 6000`,
   `preserve_raw_stream true`. This item is a decision for review, not an authorization request; the full retry is not prepared.

## 11. Procedures

1. **Reservation** (laptop): pod `dsv41-gb300-reserve-tray16` Running, only GPU pod on the node; hard 90 min.
2. **Preflight** (tray16, dc, neutral-path wrapper, driver context): `E1_REPO=$S/wt-lifecycle bash $E1/preflight_e1_v1.sh` → PASS
   required; identity record and pytest log land in `<run root>/preflight/`.
3. **GO** naming the commit (trees = §2) and this packet; then **launch**: `tmux new -d -s e1v1-lifecycle "E1_REPO=$S/wt-lifecycle bash
   $E1/run_e1_v1.sh"` through a neutral-path wrapper in `$S/scratch/`. One launch; no retry without a new GO.
4. **Monitor** every 60 s (laptop, read-only ssh, neutral path): `bash $E1/monitor_e1_v1.sh` — phase, import %, `hard_at`,
   `ceiling_at`, preserved size, GPU memory, foreign processes, `/tmp` and run-dir sizes; ABORT-CONDITION lines are reviewed, the
   controller acts on its own bounds; the operator intervenes only through `lifecycle.control` (`release-gpus <reason>`) or, outside
   the controller, through the registry (`e1_pids.py cleanup --execute`), never by pattern.
5. **Closeout** (tray16): `bash $E1/closeout_e1_v1.sh` → CLEAN; then delete the pod (laptop).
6. **Post-run** (any tray, CPU): `E1_REPO=$S/wt-lifecycle bash $E1/postrun_e1_v1.sh` → PASS or STOP (§9); only after PASS the
   operator removes `/tmp/<name>.raw-preserve/`; archive the run directory with `SHA256SUMS` (≤ 5 GB) and record the result on the
   branch.
7. **Recovery** (`stop_failed`, dead launcher, or a preserved stream without a report): `python -m launcher alignment profile
   <profile.json> --import-stream /tmp/<name>.raw-preserve/<name>.qdstrm` on any tray (CPU; implies `--resume`), then the normal
   finalisation; nothing GPU-side is repeated.

## 12. Not authorized and still open

Not authorized and not run: V1–V3 GPU runs, the full E1 retry, the identification partition, model fitting, Rust production
integration, profiling-database writes, preset changes. Open until V1/V2 run: hypotheses (a)–(c) of the failure analysis (workers
blocked in `cudaProfilerStop` until the import completes; Nsight's target in its own process group; the 09:58:47 SIGTERM source).
V1 proves the controller's observable behaviour, the probe path and the recipe invariance on a small capture; it does not prove
what a release during `importing` does to a large trace (that is V2, with its own GO and packet addendum).
