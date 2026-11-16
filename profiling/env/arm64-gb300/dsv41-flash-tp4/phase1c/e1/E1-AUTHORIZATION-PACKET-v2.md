# E1 authorization packet v2 — full decode routing / rank-arrival capture (`e1_main_v3`), prepared 2026-11-16, CPU only

Supersedes `E1-AUTHORIZATION-PACKET.md` (5c86ae1c, accepted for Stage 1 only) for the capture procedure; the scientific design
(cells, estimators, contrasts, protocol) is unchanged and quoted by reference. Basis: Stage 1 (c7e8ed9b), Stage 1.5 passes 1–4
(a62e40bd), the b=8 recovery (223aff6d) and the CPU audit `audit/E1-READINESS-AUDIT.md`. **Nothing in this packet is authorized
to run**; it is the request. Branch `phase1c-e1-audit`; exact revisions and hashes in §9.

## 0. What changes against v1, and why

| v1 | v2 | reason (evidence) |
|---|---|---|
| nsys `capture_start_delay_seconds: 10` | **no delay**: `/start_profile` is called synchronously before the first request | the 10 s delay dropped the whole b=8 wave in `e1_nsys_val`; the simulation shows it would drop the first E1 wave (b=1, 4 k, wave 0) entirely (`audit/schedule/E1-SCHEDULE-SIM.md`); the no-delay pass captured 1040/1040 iterations |
| two dry launches (`e1_dry_off`, `e1_dry_on`) + `e1_main` | **one launch** (`e1_main_v3`) | the dry pair was run twice (Stage 1, Stage 1.5); the hook-overhead question is answered to the resolvable level (b=8 +1.5 %; b=48 not resolvable at 2 % from one pair) |
| streaming default of the req-frontend | `stream: false` explicit (Stage 1.5 fix, fork 5c20a259, binary 7f6eb81d…) | token identity under the openai backend |
| config `configs/e1_main` (paths in the main clone) | `configs_v3/e1_main_v3/profile.json`, derived mechanically from the accepted `configs_v2/e1_nsys_val` (`make_e1_inputs_v3.py`: only name, log_dir, trace path change, delay removed; server block, environment and workload options verbatim; nsys block == Phase 0 quadrant) | the accepted recipe is the one that passed all gates twice |
| reservation 5 h, abort startup > 60 min | reservation **4 h**, abort startup > **30 min**, workload not finished 25 min after measurement begins, GPUs held 20 min after the workload | measured startup 23 min 17 s (b=8 run), simulated capture 16.4 min |
| cleanup by pattern | registry-based identity-verified cleanup (`checks/e1_pids.py`), discovery separated from termination, own invocation pipeline excluded, independent lookalikes = STOP | Stage 1.5 incident and GO attempt 1 (496b9233), both corrected and tested (21 tests) |
| storage budget 80 GB, ~42 GB expected | unchanged (~43 GB from measured rates) | 216 M kernel rows × 149 B/row sqlite, 29 B/row rep |

## 1. Workload (unchanged design; frozen trace)

`traces/e1_main.csv` sha256 2e147402b5a24f9e2078bf12e3623691c2fd6d5c8378bab09d7e86ee23bcc2ef: 450 requests, 36 waves, b ∈ {1, 2, 4, 8,
16, 24, 32, 48} × KV ∈ {4 k, 32 k, 96 k}, 1024 generated tokens each, 2 waves for b ≤ 8, trace-timed arrivals (next wave at
previous estimate × 1.5 + 5 s), span 940.6 s planned. Plan per wave in `E1-INPUTS.json` (frozen) and `E1-MAIN-V3-INPUTS.json`
(v3 provenance). Pure-decode iterations per wave ≥ 736 by plan (min at b=48, 96 k), ≥ 400 everywhere; total 38 104 iterations,
≈ 35 624 pure.

Revised simulation (`analysis/e1_schedule_sim.py`, `audit/schedule/`): wave durations scaled by the measured nsys-on ratio
(0.9856: b=8 32 k wave 9.02 s vs 9.1 planned, b=48 32 k wave 32.04 s vs 32.7); simulated span 895.6 s; **0 of 36 waves overlap
the previous one** (smallest gap ≈ 2.6 s plus the 1.5× margin inside each estimate); profiler armed before the first request
captures every wave; a 10 s delay would miss wave `b1 kv4k w0` completely.

Hook on (`capture_routed_experts: true`), openai backend, `stream: false`, `temperature 0`, `ignore_eos`, prompts from distinct
enwik9 offsets (request_ordinal × 9973 seeding, so the b=8 32 k wave 0 and b=48 32 k wave 0 requests are byte-identical to the
accepted passes — used as a workload-identity cross-check after the run).

## 2. Routing-versus-timing measurement strategy (explicit)

Everything is measured in one launch, one nsys session, one hook-on capture, so routing and timing come from the same iterations:

1. **Timing per rank and collective** from nsys (Phase 1A `collectives` / `iter_rank` tables via the frozen extractor): arrival,
   completion, in-collective time per (iteration, rank, k); segments between collectives. Wait and intrinsic by the Phase 1A proxy
   everywhere, and additionally by the direct (simultaneous-completion) definition on the b ≥ 17 path where the audit showed the two
   agree (B.2). The K = 163 path (b ≤ 16) keeps the proxy only; this is recorded as a modelling assumption in every output.
2. **Routing per token and layer** from the hook (`routed_experts/*.npy`, `token_corpus`), joined to iterations by
   `checks/e1_completeness_v2.py` (unique token↔iteration join; segments == npy[1:]; every segment row maps to exactly one iteration).
3. **Per-cell estimators** (frozen `analysis/e1_identify.py`, cells = (b, KV, phase ∈ {pure, mixed}) with the modal K): W_meas, W_med,
   W_ind, W_cm, f_ind with bootstrap CI, ρ, c_common, first-arrival spread and last-rank histogram, collective end spread, MoE
   cross-rank correlation, R² of wall and MoE time on per-iteration routing statistics. Plus `analysis/e1_rank_dependence.py` per b
   (coefficient-free composition, tails, critical-path shares, autocorrelation).
4. **Pre-registered contrasts** (packet v1 §8, unchanged thresholds): regime (b ≥ 24 vs b ≤ 16 at each KV: Δ < −0.2), KV (range > 0.2
   at two or more b), phase (|Δ| > 0.2 at two or more cells), pass (|Δ| > 0.15 vs frozen rows). The audit already shows Δ ≈ −0.47
   between b=8 and b=48 at KV 32 k across two sessions; E1 tests it inside one launch at three KV.
5. **What is separated**: routing-driven common-mode variation (MoE group time correlation, R² on routing) from rank-independent
   variation (ρ, f_ind), and both from host arrival (first-collective spread, lateness). **What is not**: hook overhead (needs an
   off arm; not part of E1), tactic drift between sessions (single session by design).
6. **Serial correlation**: the audit found within-wave autocorrelation of iteration time (lag-1 0.57 at b=8); per-cell CIs will be
   reported both with the frozen i.i.d. bootstrap (pre-registered) and with a block bootstrap (block = 20 iterations) as a
   sensitivity line. No threshold changes.

## 3. Identification and validation partitions

* E1 (`e1_main_v3`) carries the designation **identification** (VALIDATION-PROTOCOL §2–3), fixed here before the capture. After the
  gates pass, `make_partition_1c.py --tables phase1a-tables/e1_main_v3 --label e1_main_v3 --designation identification --out
  phase1c/partition/partition-1c.json` writes every non-excluded iteration as `identification` and JIT / kind-mismatch iterations as
  `excluded`. The Phase 1A partition (`phase1a/partition/partition.json`, sha 288e193f…, pinned by `phase1c/tests/test_invariants.py`)
  is not edited; its held-out rows are not read (the frozen estimator refuses frozen tables without `--calibration-only`).
* The accepted Stage 1.5 passes (`e1_nsys_val`, `e1_b8_recovery`) are **not** E1 rows and are not fitted on; they stay what they are
  (trace-validation evidence) and may serve as a cross-session reference line only.
* **Validation** is E2 (separate launch, different day, different request-order seed and prompt subset, production backend
  `vllm_tokens`, no routes, designation `validation`), compared one-shot against a model version whose hash is recorded before E2
  runs. Nothing estimated from E1 is validated by E1. The Phase 1A/1B coefficients, predictions and the Phase 1C-0 uncertainty tables
  stay frozen (`audit/FROZEN-EVIDENCE.sha256`).

## 4. GPU reservation and storage (measured basis)

| phase | estimate | basis |
|---|---|---|
| startup (model load, graph capture, FlashInfer autotune, warm-up) | 23–25 min | 23 min 17 s measured 2026-11-16; Stage 1: 25 min |
| capture (36 waves) | 16.4 min (895.6 s + warm-up) | schedule simulation |
| nsys stop + report finalisation (6.2 GB rep) | ≤ 20 min | Phase 0 run2 2.7 GB: ≈ 10 min |
| **GPU-held** | **≈ 60 min**, maximum 90 min | |
| sqlite export (CPU, GPUs free, launcher still holds the pass) | ≈ 23 min | 37 s per 5.9 M rows measured |
| parse (CPU) | ≈ 63 min (linear; Phase 0 run2 rate gives ≈ 65 min) | 103 s per 5.9 M rows measured |
| **reservation** | **4 h hard limit**, nominal ≈ 2 h 45 min including closeout | |

Storage (`audit/schedule/e1_schedule_sim.json`): 216 M kernel rows → `.nsys-rep` 6.2 GB, `.sqlite` 32.1 GB, parsed.json + kernels
parquet + host_timeline 2.6 GB, routes + corpus ≈ 0.5 GB, logs/metrics ≈ 0.2 GB: **≈ 43 GB raw, ceiling 80 GB** (abort if the run
directory exceeds it). CephFS free 20.6 TB (preflight requires > 1000 GB), tray16 root 1165 GB free (nsys temporaries; > 100 GB
required). Raw copy to `artifacts/phase1c-e1-main-dsv41-flash-gb300-tp4-<date>/run/e1_main_v3/` with `SHA256SUMS`
(`collect_e1_main_v3.sh`); committed to Git only the small files (configs, logs, provenance, checker outputs, tables' meta).

## 5. Pre-launch requirements (all must pass; any failure = STOP without GPU execution, no retry under the same GO)

1. HEAD is exactly the authorized commit (clean tree); `SCRIPTS_v3.sha256` (25 files) and `INPUTS_v3.sha256` verify; frozen evidence
   manifests verify (`audit/FROZEN-EVIDENCE*.sha256`).
2. req-frontend gitlink and checkout 5c20a259, tree 06a4a61e, `session_runner` sha 7f6eb81d… == `BINARY-PROVENANCE.json`.
3. tray16: 4 × 0 MiB, 0 compute apps, no tmux server, no experiment processes; node `nvidia.com/gpu` 0/4 allocated, no GPU pod;
   reservation `dsv41-gb300-reserve-tray16` applied and Running, the only GPU-requesting pod on the node.
4. `preflight_e1_main_v3.sh` run **in the exact driver context** (tmux `e1main-*`, `2>&1 | tee <run root>/<pass>/preflight.log`,
   launched from a neutral-path wrapper that exits immediately): all PASS; own pipeline and own tmux session reported as INFO;
   discovery CLEAN; registry plan clean. Evidence that the v3 preflight passes in that context on the committed tree:
   `audit/preflight/preflight_v3_tray16_driver_context_20261116T073003Z.log` (21 PASS, 0 FAIL).
5. Negative control immediately before: an independent lookalike (scratch registry) must make the same preflight STOP, then be
   removed only through verified cleanup of its scratch registry (as done on 2026-11-16 05:32Z for the b=8 GO).
6. Profiler-arming configuration: `nsys` block == Phase 0 quadrant block, no `capture_start_delay_seconds`, no bounded stop; the
   driver's guard re-checks the 450-row trace sha, the config's trace path and the absence of a delay immediately before launch.
7. Registry path `logs/dsv41_flash_gb300_tp4_e1_main/pids/owned.tsv` absent (empty) before launch; cleanup protections (own-ancestry
   refusal, PID-reuse re-verification, lookalike STOP) are the tested `e1_pids.py` of 1c8aa34d.
8. ≥ 1000 GB CephFS free, ≥ 100 GB root free.

## 6. Launch, monitoring, abort and cleanup procedure

```
# laptop: reservation (socks proxy to Rancher), then confirm it is the only GPU pod on the node
HTTPS_PROXY=socks5://127.0.0.1:11080 kubectl --request-timeout=90s -n inference-bench apply -f k8s/dsv41-gb300/reserve-tray16.yaml
# tray16 as dc, through neutral-path wrappers in $S/scratch/ (their command lines name no experiment path):
#   ctx wrapper:   tmux new -d -s e1main-ctxtest "E1_REPO=$WT bash $E1/preflight_e1_main_v3.sh 2>&1 | tee <run root>/ctxtest/preflight.log ..."
#   negative control, then the real context preflight (both read-only)
#   launch wrapper: tmux new -d -s e1main-v3 "E1_REPO=$WT bash $E1/run_e1_main_v3.sh"
# monitoring every 60 s (laptop, ssh to tray16, neutral path): bash $E1/monitor_e1.sh <run root> e1_main_v3 30 25 20 80
#   + every 5 min: kubectl GPU pods on the node (must stay = the reservation)
# after rc=0: bash $E1/closeout_e1.sh e1_main_v3 ; kubectl delete pod dsv41-gb300-reserve-tray16 ; postrun_e1_main_v3.sh (any tray, CPU) ; collect_e1_main_v3.sh
```

Abort conditions (operator acts; the monitor only prints `ABORT-CONDITION` lines): startup > 30 min without "Application startup
complete"; a foreign GPU process (compute app not descended from the registered driver); engine error lines before the shutdown
trigger; workload not finished 25 min after "measurement begins"; GPUs still held 20 min after "workload done"; run directory >
80 GB; CephFS < 500 GB or root < 100 GB free; a GPU pod other than the reservation on the node; reservation older than 4 h.
Abort action: `python3 -I checks/e1_pids.py cleanup --registry <run root>/pids/owned.tsv --execute --grace 30 --kill9` (identity-
verified, deepest first; refuses own ancestry; lookalikes → STOP and report), then closeout, then pod deletion. Never by pattern.

Cleanup verification (closeout_e1.sh): 4 × 0 MiB and 0 compute apps; 16 process-discovery checks PASS; registry plan: driver GONE,
nothing owned alive, 0 lookalikes; no tmux server; reservation pod deleted and node allocation 0/4; raw copy checksummed.

## 7. Post-run gates and analysis (`postrun_e1_main_v3.sh`)

0. capture window: parsed iterations contiguous and equal to the metrics iterations, every b's pooled pure-decode count ≥ the plan's
   summed minimum, no straddling (profiler armed before the first request). 1. completeness v2 (36 cells, joins, npy shapes, corpus).
2. frozen extractor → `phase1a-tables/e1_main_v3`. 3. nsys validation checker, frozen thresholds (28 NVTX ranges per iteration, 4
devices, strictly increasing starts, GPU windows overlap, collective end spread median ≤ 0.02 ms and p90 ≤ 0.05 ms per cell, JIT ≤
20 after warm-up). 4. workload identity of the two shared waves against the accepted passes. 5. frozen estimators with metrics and
corpus. 6. rank/kernel reports (refs: run2 for b ≤ 16, quadrant for b ≥ 24) and the rank-dependence audit per b. 7. configuration
identity vs `e1_nsys_val` (allowed: name, log_dir, trace path, delay; the launch-record checker will again report the nsys output
path and the delay — enumerated in the audit C.2). Then, on instruction only, `make_partition_1c.py --designation identification`.

## 8. Proposed acceptance criteria and stop conditions

Acceptance (all required; none may be relaxed after the run):
* rc = 0; `validation.ok`; export checksummed; parsed = metrics iterations, contiguous; first captured iteration is the first measured.
* every one of the 36 waves present; pure-decode iterations per wave ≥ 400 and per b ≥ the plan's minimum; 0 overrun iterations.
* NVTX: 4 × iterations forward ranges; 0 iterations without 28 ranges; 0 absent devices; 0 index disagreements; unpaired steps {}.
* collective end spread per pure-decode cell: median ≤ 0.02 ms, p90 ≤ 0.05 ms; all pure-decode iterations rank-aligned (modal K).
* routing joins: 0 duplicate, 0 missing; 450 npy files (1024, 40, 6) uint16; corpus 450 × 1023 tokens; segments == npy[1:].
* JIT-affected iterations after warm-up ≤ 20; remaining clean iterations per cell ≥ 400.
* workload identity of the shared waves vs the accepted passes: PASS.
* raw ≤ 80 GB; GPU-held ≤ 90 min; reservation ≤ 4 h; cleanup verification complete.
Stop conditions during the run: §6. Scientific outcome is reported as the pre-registered contrasts with their thresholds (v1 §8); a
failed gate is reported with the evidence preserved, never repaired by changing thresholds.

## 9. Exact source revisions and hashes

| item | value |
|---|---|
| launcher tree / scripts | `phase1c-e1-audit` at the commit carrying this packet (recorded in the GO); derived from accepted 223aff6d; frozen heads unchanged |
| `SCRIPTS_v3.sha256` | 25 files (drivers, preflights, checks, analysis incl. audit tools, tests); manifest sha recorded in the commit message and `hash_scripts_v3.sh` output |
| `INPUTS_v3.sha256` | `traces/e1_main.csv` 2e147402…, `configs_v3/e1_main_v3/profile.json` 8bd51ebe0f8e3363…, sources `configs_v2/e1_nsys_val/profile.json` df949648…, `configs_v2/e1_b8_recovery/profile.json` 1f8f9738…, `traces/e1_b8_recovery.csv` 398c317e…, `traces/e1_dry.csv` c6d8e13b… |
| req-frontend | gitlink 5c20a259618976ecc360bda84abcc3653a04006e (fork kaijunli-infr/request-factory `e1-stage15-stream-flag`), tree 06a4a61e, `session_runner` 7f6eb81d45e6b98f3b6ebb56ab20b60ce8aaf4584aa3d4f25e3449b2d05a00c2; `.gitmodules` unchanged (uw-syfi) |
| engine | vLLM fork 0ab18711 (`servingstudio-alignment-v0.31.0`), driver 595.71.05, nsys 2025.6.3.541, model DeepSeek-V4.1-Flash-1017 |
| frozen evidence | `audit/FROZEN-EVIDENCE.sha256` (246 entries), `audit/FROZEN-EVIDENCE-EXTERNAL.sha256` (11) |
| thresholds | `checks/e1_nsys_validation.py` defaults and v1 §4.3 / §8 — unchanged |

## 10. Unresolved scientific risks

1. **Session effect on f_ind** (audit B.4: 0.33 at b=48 in Stage 1.5 vs 0.23 in Phase 0): E1 is one session; the regime contrast is
   internal to it, but the absolute level will carry a session term that only a second launch (E2 or a repeat) can bound.
2. **Hook overhead at b=48** is not resolvable at 2 % from one on/off pair (Stage 1.5) and E1 has no off arm; any effect of the hook on
   rank timing is confounded with "hook-on" in all E1 cells. Mitigation: the hook's GPU cost is < 1 % (two copy kernels); host-side
   effects are unmeasured.
3. **Per-launch tactic re-selection** (audit C.3: 51–67 % of tactics differ between any two launches; 1–3 % GPU-time effect at b=48
   in the one traced pair): E1 measures one tactic draw; cross-session comparisons of absolute times are limited to that precision.
4. **Serial correlation within waves** (ACF lag-1 0.57 at b=8): the frozen i.i.d. bootstrap CIs understate uncertainty; the block
   bootstrap sensitivity is reported alongside, thresholds unchanged.
5. **Intrinsic/wait separation on the b ≤ 16 path** relies on the fastest-rank proxy; the direct definition is invalid there (audit
   B.2). A mixture weight estimated at b ≤ 16 inherits this assumption.
6. **Mixed-phase cells** have n ≈ 1–96 iterations per wave; the phase contrast is only testable at b ≥ 16 (n ≥ 50 rule).
7. **Rare stalls** (one 5.3 ms rank-3 lateness in 928 b=48 iterations): E1's tails will include such events; they are reported, not
   excluded, unless the JIT rule applies.
8. **Export/parse scaling**: 216 M rows is 37 × the largest Stage 1.5 export; parse time may be super-linear (Phase 0 run2 at 97 M
   rows: 29 min). The reservation covers up to 4 h; if the parse exceeds 3 h it is stopped and reported, the rep/sqlite are kept.
9. **Capture volume on the tray root** (nsys temporaries ≈ rep size): 1165 GB free; stop rule at 100 GB.

## 11. Decisions requested with a GO

* GO for exactly one launch of `e1_main_v3` on rack 10 tray16 at the named commit, reservation ≤ 4 h, GPU-held ≤ 90 min, raw ≤ 80 GB,
  under §5–§8; no retry after any refusal; stop for review after the single pass with the §7 gates and the pre-registered contrasts.
* Confirmation that the partition write (`partition-1c.json`, identification) is part of the post-run on a passing result, or is to
  wait for a separate instruction.
