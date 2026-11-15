# E1 b=8 recovery capture: authorization packet (CPU preparation only; GPU execution needs a separate GO)

Status 2026-11-15: prepared and verified on CPU. Nothing in this packet has been run on a GPU. The accepted `e1_nsys_val`
b=48 result (commit a62e40bd) and every Stage 1.5 trace, report, script and hash are unchanged; this packet only adds files.

Kevin's decision of 2026-11-15: `e1_nsys_val` accepted as a successful b=48 trace validation and a partial completion of the
two-cell validation; full E1 NO GO; GO for the CPU-only preparation of one b=8 recovery capture; NO GO for any new GPU run,
model refitting, E2–E6, Rust production integration, profiling database writes or preset changes.

## 1. What the recovery is

One nsys pass, `e1_b8_recovery`, hook on, on the eight b=8 requests of the Stage 1.5 dry trace and nothing else, with the
profiler armed before the first request is released. It supplies the b=8 half of the pre-registered two-cell Nsight validation
(`E1-NSYS-VALIDATION-PROTOCOL.md`) that the 10 s delayed start of `e1_nsys_val` skipped. b=48 is not re-run.

| item | `e1_nsys_val` (run, accepted) | `e1_b8_recovery` (prepared) |
|---|---|---|
| trace | `traces/e1_dry.csv`, 56 requests (8 b=8 at 0 s, 48 b=48 at 18.6 s), sha c6d8e13b… | `traces/e1_b8_recovery.csv` = header + rows 1–8 of `e1_dry.csv`, byte-identical (8 b=8 at 0 s), sha 398c317e… |
| config | `configs_v2/e1_nsys_val/profile.json`, sha df949648… | `configs_v2/e1_b8_recovery/profile.json`, sha 1f8f9738…; differs in exactly four keys: `name`, `log_dir`, `workload.frontend.path`, `nsys.capture_start_delay_seconds` (removed) |
| nsys block | Phase 0 `diverse_nsys_run2` block: `cuda_profiler_api`, graph trace `node`, event trace off, **delayed start 10 s**, no bounded stop | Phase 0 `quadrant_nsys` block: `cuda_profiler_api`, graph trace `node`, event trace off, **no delay**, no bounded stop |
| server block, env, fork venv, hook, `stream: false`, openai backend, warmup, corpus, tokenizer | Stage 1.5 | identical (checked by `checks/e1_config_identity.py`) |
| launcher tree / `session_runner` | worktree `phase1c-e1-stage15`, req-frontend 5c20a259, binary sha 7f6eb81d… | same tree, same gitlink, same binary (preflight re-verifies) |
| expected iterations | 2160 (1120 captured) | 1040, all captured: 2 prefill-only, 15 mixed, 15 decode ramp, 1008 pure b=8 |

Documented deviations from the accepted Stage 1.5 configuration (item 6 of the preparation list): the 48 b=48 requests are
absent from the trace, and the nsys delayed start is removed. Everything else is identical.

## 2. Census of the 1,120 captured `e1_nsys_val` iterations (item 2)

`analysis/e1_capture_window.py census --run <e1_nsys_val>` (`runs/b8_recovery/census_e1_nsys_val.md/.json`), classifying
each EngineCore iteration by phase and by the cells of the decode requests in flight (request-id tags in
`decode_request_progress`):

| captured iterations (parsed.json, index 1223..2342, contiguous) | n | request ownership |
|---|---|---|
| prefill-only (16k chunks of the b=48 wave's first request, no decode request in flight) | 2 | none in flight (iterations 1223, 1224 at 18.67 s on the replay clock) |
| mixed: a 16k prefill chunk plus 1..47 decoding b=48 requests (`stage: mixed`) | 95 | b=48 cell only |
| decode ramp: 1..47 decoding b=48 requests, no prefill | 95 | b=48 cell only |
| pure decode b=48 | 928 | b=48 cell only |
| **total** | **1120** | 1118 b=48-owned + 2 un-owned prefill chunks; 0 b=8; 0 overrun |

The 192 non-pure iterations are therefore the b=48 wave's own ramp (its 47 later requests prefill two chunks each while the
earlier ones decode: 2 + 2×47 = 96 prefill iterations, 95 decode-ramp iterations in between). The wave ends with all 48
requests finishing in the same iteration (equal output length), so there is no ramp-down. JIT module loads occurred in four
captured iterations, 1271, 1287, 1303, 1319, all `mixed` b=48 ramp iterations (the checker reported 3 because it ignores the
first 50 captured iterations); no pure-decode iteration is JIT-affected. The parser's `stage` field agrees: prefill 2, mixed
95, decode 1023. The capture window (armed 10.0 s after replay start) opened in the idle gap between the b=8 wave (last
iteration 1220 ended at 9.055 s) and the b=48 wave (first iteration 1223 at 18.673 s), which is why no b=8 iteration and no
partially captured iteration exists.

## 3. Capture-window proof (items 3 and 4)

Timing model, from `alignment/runner.py::run_profile`: `measurement_ready()` runs after the server has drained; without a
delay it calls `/start_profile` synchronously and only then takes `replay_start_monotonic_ns` and lets the load generator
release the first request. With a delay d, a timer arms the profiler d seconds after that instant. Every EngineCore iteration
record carries `observed_start/end_monotonic_ns` in the same CLOCK_MONOTONIC domain as `replay_start_monotonic_ns`, so each
pass's iterations are placed on the replay clock exactly and any window can be checked against them on CPU.

Measured b=8 wave position in the four Stage 1.5 passes (`runs/b8_recovery/capture_window_simulation.md`):

| pass | first iteration starts | first pure b=8 starts | last pure b=8 ends | wave end | pure b=8 n |
|---|---|---|---|---|---|
| e1_dry_off_v2 | 0.037 s | 3.608 s | 8.818 s | 8.887 s | 1008 |
| e1_dry_on_v2 | 0.035 s | 3.617 s | 8.981 s | 9.053 s | 1008 |
| e1_dry_off2_v2 | 0.033 s | 3.612 s | 8.835 s | 8.904 s | 1008 |
| e1_nsys_val (under nsys) | 0.033 s | 3.723 s | 8.984 s | 9.055 s | 1008 |

Simulated coverage of the pure b=8 iterations for candidate windows (an iteration counts as captured only when it lies
entirely inside the window; an iteration that the window start cuts through is "straddling" and would reach the parser
without its 28 NVTX ranges, failing that gate):

| delay | pure b=8 captured (all four passes) | straddling | gate |
|---|---|---|---|
| **none (profiler armed before the first request)** | **1008 / 1008** | **0** | **PASS** |
| 0.0 s | 1008 / 1008 | 0 | PASS (equivalent: the first iteration starts 33 ms after replay start) |
| 1.0 – 3.6 s | 1008 / 1008 | 1 mixed prefill iteration cut | FAIL (28-range gate at risk) |
| 3.7 s | 990–1008 | 1 | FAIL (loses pure iterations in three of four passes) |
| 4.0 s | 932–955 | 1 | FAIL |
| 10.0 s (as run) | 0 / 1008 | 0 | FAIL (reproduces the e1_nsys_val outcome; b=48 928/928) |

Decision: no delayed start. It is the only window that captures the whole b=8 wave deterministically, independent of prefill
speed under nsys (the first pure iteration moved by 0.1 s under nsys), and it is exactly the Phase 0 `quadrant_nsys` nsys block
that produced 4750 clean iterations. A 2–3 s delay would keep all pure iterations but cuts one prefill-chunk iteration in
every pass; it is not proposed. The extra content captured by starting at the beginning is 32 ramp iterations (~3.7 s).

Reconciliation of clocks: capture start happens in a drained server (the Stage 1.5 passes show 33–37 ms between replay start
and the first iteration, all 17 prefill chunks of the wave after it); warmup runs before `measurement_ready()` and is outside
the measured replay (`measurement_log_offset`); the end of capture is `/stop_profile` after the replay and the idle wait, so the
last b=8 iteration (ending ~9.0 s) is inside the window with margin.

Regression test (item 5): `tests/test_e1_capture_window.py`, 10 CPU tests: synthetic two-wave timeline with delayed-start
boundary cases (window at exactly an iteration start is inclusive; one microsecond later straddles; delay in the idle gap
misses the first wave entirely; delay inside a prefill chunk straddles it but keeps all pure iterations; bounded duration cut
at and just before an iteration end; delay after the last wave captures nothing; overrun/untagged kinds; CLI round trip), plus
a real-data test (skipped where the Stage 1.5 run root is absent) asserting for all four passes 1008 pure b=8 iterations
between 3.5 and 4.0 s and 9.5 s, 0 captured with the 10 s delay, 928 b=48 captured, 1008 captured without delay, and the
census of 1120 / 928 / 0. Result on tray14: `10 passed in 8.79s`.

## 4. Workload identity (item 6)

* Trace rows: `cmp` of `traces/e1_b8_recovery.csv` against `head -9 traces/e1_dry.csv` is exact (preflight check).
* Prompts: req-frontend seeds each independent request's synthetic prompt from its ordinal in the trace
  (`executor/independent.rs`: `TokenProvider::new(pool, request_ordinal * 9973)`) over a token pool whose size is
  `max(2 × max prompt, request count, 100 M)` = 100 M tokens for both traces, built from the same corpus (`enwik9`) with the same
  tokenizer; the b=8 requests are ordinals 0..7 in both traces, so their prompts are identical by construction.
* Post-run confirmation: `checks/e1_workload_identity.py` compares the eight requests' load-generator records (id, input_len,
  prompt_len, output target, arrival, actual output length, finish reason, server prompt/completion/cached token counts)
  against `e1_dry_on_v2` and `e1_nsys_val`, and reports routed-experts array shapes (must agree) and element agreement
  (informational). Any field difference is a FAIL.
* Server identity: `checks/e1_config_identity.py` on the launch records with the four allowed keys (`postrun_e1_b8.sh` §6).

## 5. Execution limits, abort conditions, cleanup (items 7, 9)

| ceiling | value | basis |
|---|---|---|
| reservation (pod `dsv41-gb300-reserve-tray16`, 4 GPUs) | **60 min hard**, from apply to delete | nominal ~30 min: startup 22–25 min (autotune ~19 min, re-tunes every launch), replay ~10 s, stop/flush ~1 min, sqlite export ~1 min, teardown ~2 min, checks 3 min; `e1_nsys_val` used 31 min 53 s for a 51 s replay |
| GPU-held | ≤ 30 min, one launch, all four GPUs of tray16 (`cuda_visible_devices 0,1,2,3`), ~257 GB per GPU | Stage 1.5 |
| startup | stop if "Application startup complete" is not in the server log 30 min after launch | Stage 1.5 rule |
| storage (new raw on CephFS) | rep ≤ 0.6 GB, sqlite ≤ 3 GB, total ≤ 5 GB; CephFS > 500 GB and tray root > 100 GB free (preflight) | expected ~5.9 M kernel rows (1040 iterations × ~5.6 k rows + prefill), rep ~0.2 GB, sqlite ~1.0 GB (171 B/row measured on `e1_nsys_val`), parquet/json ~0.1 GB |
| passes | exactly one; the driver accepts only `e1_b8_recovery`; the trace cannot produce b=48; no `e1_main` | `run_e1_b8.sh` refuses any other argument and re-checks the trace immediately before launch |

Abort conditions (stop, archive what exists, run `closeout_e1.sh`, report; no retry without a new go): preflight FAIL; startup
> 30 min; any GPU process or GPU pod on tray16 that is not the launch or the reservation; `EngineDeadError`-family line before
the shutdown trigger; `validation.ok == false` or export failure; disk below the preflight thresholds; reservation hard stop.

Process cleanup (item 9): `checks/e1_process_census.sh` enumerates vllm/EngineCore, nsys, `session_runner`, launcher, E1
driver/preflight/postrun shells, `tail`/`less` watchers on experiment logs, nvidia-smi loops and `watch`, python monitors,
any process whose command line or cwd references the experiment directories, experiment-named tmux sessions (`e1-*`, `e15-*`,
`e1b8-*`; other users' sessions on a shared tray are reported as INFO) and GPU compute apps; exits 1 if anything is alive. It
runs in the preflight (must be clean before the go), at the end of the driver and in `closeout_e1.sh` (with per-GPU memory and
the reservation-delete reminder). Positive test on tray14 (`runs/b8_recovery/process_census_positive_tray14.log`): a planted
`tail -F` on a run log and a `sleep` with its cwd in the run directory were both reported (NOT CLEAN), killed by pid, and the
census returned CLEAN. Read-only closeout on tray16 at 17:36Z: CLEAN (16 PASS). Note from the positive test: a shell whose own
command line quotes the planted command text matches `pgrep -f` of that text; kill by explicit pid only.

## 6. Pre-GPU verification already done (CPU, 2026-11-15)

| check | result | evidence |
|---|---|---|
| frozen remote heads unchanged | test-ss ca9f8ed0, phase1-planning cc06d9c2, phase1a-costmodel a7725255, phase1b-seqsim 29082793, phase1c-prep 2e53072d, phase1c-e1-packet 5c86ae1c, phase1c-e1-stage1 c7e8ed9b, kevin/dsv41-gb300-main 58f5d779; phase1c-e1-stage15 a62e40bd before this packet | `git ls-remote` from rack 3 tray01 |
| recovery inputs | generated by `make_e1_inputs_b8.py` from the hashed Stage 1.5 inputs; `INPUTS_b8.sha256` verifies | §1 |
| `preflight_e1_b8.sh` on tray16 (read-only, 17:35Z) | PREFLIGHT PASS: 16 recovery checks + 32 nested Stage 1.5 / Stage 1 checks, 0 FAIL; launcher dry-run of the config OK; binary 7f6eb81d… == provenance; gitlink/checkout 5c20a259 clean; nsys block == Phase 0 quadrant_nsys; config identity vs e1_nsys_val with the four allowed keys; census clean; GPUs 0 MiB | `runs/b8_recovery/preflight_tray16_20261115T173501Z.log` |
| capture-window tests | 10 passed | §3 |
| simulation / census | §2, §3 | `runs/b8_recovery/` |

To be re-done immediately before a GPU go (same as for `e1_nsys_val`): frozen heads, bundle verify, binary hash, full
preflight on tray16, reservation pod Running with no other GPU pod on the node, free space, and the capture configuration
hash against this packet.

## 7. Exact procedure for the GPU pass (not to be executed without a separate GO)

```
# laptop: reservation (socks proxy to Rancher must be up), then verify no other GPU pod on the node
HTTPS_PROXY=socks5://127.0.0.1:11080 kubectl --request-timeout=90s -n inference-bench apply -f k8s/dsv41-gb300/reserve-tray16.yaml
HTTPS_PROXY=socks5://127.0.0.1:11080 kubectl --request-timeout=90s get pods -A -o wide --field-selector spec.nodeName=pod4-gb300-10-tray16
# tray16 as dc: preflight (read-only), then the one pass in tmux
export HOME=/mnt/shared/home/dc; S=$HOME/workspace/serving-studio; WT=$S/wt-e1stage1; E1=$WT/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1
E1_REPO=$WT bash $E1/preflight_e1_b8.sh
tmux new -d -s e1b8-recovery "E1_REPO=$WT bash $E1/run_e1_b8.sh"
# watch: $S/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15/e1_b8_recovery/run.log ("Application startup complete" < 30 min; rc=)
# any tray as dc, after rc=0: gates (capture window, completeness v2, extract, nsys validation, workload identity)
E1_REPO=$WT bash $E1/postrun_e1_b8.sh
# tray16: closeout (GPUs, full process census, tmux), then on the laptop delete the reservation and re-check the node
bash $E1/closeout_e1.sh e1_b8_recovery
HTTPS_PROXY=socks5://127.0.0.1:11080 kubectl --request-timeout=90s -n inference-bench delete pod dsv41-gb300-reserve-tray16
# collect small artifacts into runs/b8_recovery/, raw copy + SHA256SUMS.b8 into the Stage 1.5 archive
bash $E1/collect_e1_b8.sh
```

Hashes the preflight pins: config 1f8f97389a4c8bf8875e4359b3353d6e2af2c07718753eecd50cfff073bd789d, trace
398c317e093e0192ad6dc7e26cceb78825163f4ab9957a787aabc1c1629bdc77 (`INPUTS_b8.sha256`), `session_runner`
7f6eb81d45e6b98f3b6ebb56ab20b60ce8aaf4584aa3d4f25e3449b2d05a00c2 (`BINARY-PROVENANCE.json`), req-frontend 5c20a259 / tree
06a4a61e, launcher tree = the `phase1c-e1-stage15` head carrying this packet.

## 8. Frozen acceptance criteria (item 10: thresholds unchanged)

1. Capture window (`postrun_e1_b8.sh` §0): parsed iterations contiguous, every one present in the metrics, owners only the
   b=8 cell; pure b=8 captured = all pure b=8 iterations of the run (expected 1008) and ≥ 400.
2. `checks/e1_nsys_validation.py` with its defaults: export `validation.ok`, `nvtx_forward_ranges == 4 × iterations`, devices
   [0,1,2,3], 28 NVTX ranges per iteration, no absent device, identical iteration index across devices, strictly increasing
   starts per device, four-rank GPU windows overlap in every iteration, collective end spread b=8 median ≤ 0.02 ms and p90 ≤
   0.05 ms with every pure iteration rank-aligned, JIT-affected iterations after warm-up ≤ 20, tables == parsed iterations.
3. `checks/e1_completeness_v2.py --inputs E1-B8-RECOVERY-INPUTS.json`: plan = the b=8 cell (8 requests, ≥ 400 pure), no
   unplanned cell, all 8 requests emit 1024 tokens, routes join (unique token↔iteration, segments == npy[1:]).
4. `checks/e1_workload_identity.py` vs `e1_dry_on_v2` and `e1_nsys_val`: PASS.
5. Phase 1A tables extracted with input hashes; launch-record identity vs `e1_nsys_val` with the four allowed keys.
6. Required measurements to report for b=8 (observed, separated from hypotheses): NVTX completeness and correlation ownership,
   four-rank iteration alignment, collective end-time spread, timestamp consistency, routing-to-iteration joins, JIT exclusions,
   per-rank GPU execution and collective-wait distributions, kernel identities and launch characteristics
   (`analysis/e1_nsys_rank_kernels.py --cells 8`, Phase 0 quadrant as cross-session reference, hypothesis material only).
7. Report the actual reservation duration, GPU occupancy, storage, every gate result and any deviation; stop for review.

Nothing from this pass feeds model coefficients, the Rust path, the database or presets.

## 9. Source provenance (fork prepared; production submodule reference unchanged)

* Bundle `patch/request-factory-e1-stage15-stream-flag.bundle`, sha256
  2d6e197ce4630bcd729c934d9da04132c346e06cb3fe2c0f5383f65f34df485e, unchanged; contains 5c20a259 on 2cd7990 (upstream).
* Fork created 2026-11-15 16:18Z with `gh` as kaijunli-infr on rack 3 tray01: `https://github.com/kaijunli-infr/request-factory`
  (public fork of `uw-syfi/request-factory`, where the account has READ permission only). Branch `e1-stage15-stream-flag` was
  pushed from the bundle's objects on top of a fresh upstream clone; `git ls-remote` shows
  `5c20a259618976ecc360bda84abcc3653a04006e refs/heads/e1-stage15-stream-flag`.
* Clean-checkout reproducibility verified: `git clone -b e1-stage15-stream-flag https://github.com/kaijunli-infr/request-factory`
  gives HEAD 5c20a259, tree 06a4a61e9c7a751ac2e1f9b5fc50fbee655c65d6 (== the submodule source tree of the approved binary).
  The staging directory used for this was removed afterwards.
* `.gitmodules` on every branch still points at `https://github.com/uw-syfi/request-factory.git`
  (branch `glm52-tp4-streaming-fix`); the stage15 gitlink 5c20a259 is now fetchable from the fork. Changing the experimental
  `.gitmodules` URL is a separate step, not done.

## 10. Files

| path | what |
|---|---|
| `E1-B8-RECOVERY-PACKET.md` | this packet |
| `make_e1_inputs_b8.py` → `traces/e1_b8_recovery.csv`, `configs_v2/e1_b8_recovery/profile.json`, `INPUTS_b8.sha256`, `E1-B8-RECOVERY-INPUTS.json` | deterministic derivation from the hashed Stage 1.5 inputs |
| `analysis/e1_capture_window.py` | census of a pass's iterations; delayed-start simulation; coverage gate |
| `tests/test_e1_capture_window.py` (repo `tests/`) | CPU regression tests incl. delayed-start boundary cases and the real-data reproduction |
| `checks/e1_process_census.sh` | experiment-owned process enumeration (preflight, driver end, closeout) |
| `checks/e1_workload_identity.py` | per-request identity of the b=8 requests between runs |
| `preflight_e1_b8.sh`, `run_e1_b8.sh`, `postrun_e1_b8.sh`, `closeout_e1.sh`, `collect_e1_b8.sh` | preflight, one-pass driver, gates, closeout, collection |
| `runs/b8_recovery/` | census and simulation outputs, preflight and closeout logs from tray16, process-census positive test |
