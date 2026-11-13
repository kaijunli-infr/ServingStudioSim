# Proposed opt-in Rust integration interface for the sequence model (design only; no code change)

Status: proposal for review (Phase 1C-2 is not authorised). The design references the repository as surveyed on
2026-11-13 (branch `phase1c-prep` = `29082793` + this directory); file:line references are to that tree.

## 0. What exists today (survey)

- The seven-layer cost model produces one number per iteration in L1–L4 and the L5 worker advances the clock by it:
  `IterBatchWorker::start_iteration` (`simulator/src/worker/workers/iter/iter_batch_worker.rs:163-176`) calls
  `self.execution.evaluate_iteration(&self.input, iteration, now)` and sets `now + cost`;
  `UnifiedIterExecution::evaluate_iteration` (`simulator/src/worker/execution/unified_iter_execution.rs:69-77`) →
  `CostBuffers::run_iter` (`simulator/src/worker/cost_buffers.rs:372-411`) → `IterwiseUnifiedModel::eval_iter`
  (`simulator/src/arch/contract.rs:87-122`) → `LeafMetrics { m: Metrics4, coverage: CoverageFlags, backend_index }`.
- The batch input is `UnifiedArchInput { groups: Vec<ArchGroupInput>, tokens_per_source_rank }` with
  `ArchGroupInput { batch_tokens, prefill_tokens, decode_tokens, prefill_chunk_pairs: Vec<(u32,u32)>, decode_kv_lens, total_kv_len }`
  (`contract.rs:17-31, 79-83`). There is no decode/prefill/mixed enum; the kind is implied by `prefill_tokens > 0`.
- There is **no host-lead, scheduler or CPU term**, no plugin/hook mechanism, no per-iteration Python callback (PyO3
  is used only for L1 kernel lookups through `profiling.perf_api`, `simulator/src/timing/bridge/core.rs:494-572`), and no
  p10/p90 or confidence fields. The only timing-override knobs are `gpu_time_multiplier` / `prefill_gpu_time_multiplier`
  (`simulator/src/worker/config.rs:288-348`).
- Unsupported inputs are refused by `check_input(&UnifiedArchInput) -> Result<(), String>`; coverage is carried as
  `CoverageFlags { EXTRAPOLATED, JIT, NO_COVERAGE }` (`simulator/src/timing/cache/interp.rs:102-108`) and rolled up per
  iteration; the run-level status is `RunSummary.cause`.
- Presets are validated against the Rust-authoritative `Registry` (`launcher/schema/validate.py`); arch options are
  `#[param]` fields of `ArchConfig::DeepseekV41Vllm` (`simulator/src/arch/config.rs:375-405`), e.g. the opt-in
  counterfactual `#[serde(default)] decoder_swa_bounded_replay: bool` (`:404`). The public preset
  `presets/public/DeepSeek-V4.1-Flash/deepseek_v41_vllm.yaml` targets **NVIDIA B200**, not GB300; presets do not reference
  `profiling/env/<platform>/`.
- Per-iteration output rows: `CostLogEntry` (`simulator/src/log/rows.rs:199`: worker_id, iter_id, batch_id, wall_start_ms,
  total_time_ms, …) and `GroupInputLog` (`:157`), written to `raw/cost_log/*.parquet`; run summary `RunSummary`
  (`simulator/src/sim/run.rs:92-112`). Golden tests compare `tests/golden/<metric>/<GPU>.json` flat maps
  (`launcher/golden.py`); there is no DeepSeek/GB300 golden and no general flag-off-parity harness (precedents:
  `config.rs:279`, `cost_buffers.rs:113,285`, `tests/session_execution_v2_plan_equality.rs`).

## 1. Hook point and shape

A new trait in the L5 worker, implemented by an adapter and consulted only when the flag is on:

```rust
// simulator/src/worker/sequence_model.rs (new; design only)
pub struct SeqRequest<'a> { pub iteration: u64, pub now: Time, pub input: &'a UnifiedArchInput, pub previous: [IterKindB; 2] }
pub struct SeqCost { pub gpu: Time, pub host_lead: Time, pub gpu_idle_gap: Time, pub p10: Option<Time>, pub p90: Option<Time>,
                     pub status: SeqStatus, pub confidence: SeqConfidence, pub flags: CoverageFlags, pub kind: IterKind }
pub trait SequenceModel: Send {
    fn check_input(&self, input: &UnifiedArchInput) -> Result<(), String>;   // refuse unsupported shapes up front
    fn next_iteration(&mut self, req: SeqRequest) -> Result<SeqCost, String>;  // stateful: keeps the per-rank timelines
    fn version(&self) -> &str;                                                 // model_version, checked against the preset
}
```

Interception at `IterBatchWorker::start_iteration`: when `self.sequence_model.is_some()`, the worker calls
`next_iteration` and advances the clock by `cost.gpu + cost.gpu_idle_gap` (the GPU-visible iteration time on the
critical path including host-bound idling), records `host_lead` and the band in the cost log, and merges `flags` into
the iteration's `CoverageFlags`. The existing `evaluate_iteration` path is **not executed** in that case (not scaled,
not blended: no multipliers), and is byte-for-byte unchanged when the flag is off. `PipelineStageExecution` is out of
scope (TP4 single stage).

The model is **stateful per worker** (one instance per simulated worker/TP group): the sequence rules need the previous
two iterations and the per-rank host/GPU timelines, which the worker does not hold. Resetting a worker resets the model.

## 2. Input and output contracts

Formalised as JSON Schema (`schema/seqsim_request.schema.json`, `schema/seqsim_response.schema.json`) so that the
Python reference (`phase1b/seqsim`) and any future Rust port speak the same contract and can be parity-tested.

Mapping from `UnifiedArchInput` to the Phase 1B `Feat`:

| Feat field | from |
|---|---|
| `stage` | `decode` if every group has `prefill_tokens == 0`, else `mixed` |
| `b` | number of decode requests (`len(decode_kv_lens)` summed over groups) |
| `P` | Σ `prefill_tokens` |
| `chunks` | `prefill_chunk_pairs` as `[prefix_len, append_len]` (kv0 = prefix_len for a 16k chunk) |
| `kv_max`, `kv_sum` | max / Σ over `decode_kv_lens` |
| `is_chunk16k`, `kv0`, `mixed_kind` | derived as in `costmodel.data` (graph ≤ 256 tokens, ring ≥ 8192) |
| `prev_stage`, `b_{i-2}` | from `previous` |

Outputs (response schema): `gpu_ms` (median path or mean of draws), `gpu_p10_ms`/`gpu_p90_ms`, `host_lead_ms`,
`gpu_idle_gap_ms`, per-rank offsets, `status`, `kv_status`, `confidence`, `flags`, `message`.

## 3. Supported regimes (v1) and refusal

| regime | support |
|---|---|
| TP4, EP as in the captured recipe, DSV4.1-Flash, GB300, vLLM v0.31 recipe (chunk 16384, graph cap 256) | supported |
| decode b 1–48, kv ≤ 131,063 per request | supported; b 1–2 **low confidence**; b outside the tables → `extrapolation` status |
| 16k prefill chunks, kv0 ≤ 98k (KVW range) | supported; beyond → `kv-extrapolation` flag |
| eager mixed (257 ≤ P + b < 8192) | supported, **low confidence** (host-bound) |
| ring mixed (P + b ≥ 8192, not a 16k chunk) | supported, status inferred |
| graph-kind mixed (P + b ≤ 256) | supported as class medians; missing class → `unsupported` |
| anything else (other TP/EP, other GPU, pipeline stages, speculative decoding, other chunk sizes) | `check_input` → `Err`, run refused at start, never a silent fallback to the L4 model |

Rule: the sequence model never falls back to the L1–L4 number for an iteration it cannot cost; it refuses the input.
A per-run option `on_unsupported: refuse | replay_l4_with_flag` may be added later, but `refuse` is the default.

## 4. Uncertainty and status propagation

- `CoverageFlags` gains `SEQ_EXTRAPOLATED`, `SEQ_INFERRED`, `SEQ_KV_EXTRAPOLATED`, `SEQ_LOW_CONFIDENCE`, `SEQ_UNSUPPORTED`
  (bit positions after `NO_COVERAGE`), rolled up exactly like the existing flags to the iteration and run level, so
  the run summary reports the fraction of iterations with each flag.
- `CostLogEntry` gains optional columns `seq_gpu_p10_ms`, `seq_gpu_p90_ms`, `seq_host_lead_ms`, `seq_gpu_idle_gap_ms`,
  `seq_status`, `seq_confidence`, `seq_kind` (null when the flag is off; the parquet schema version increments).
- `RunSummary` gains `seq_model_version: Option<String>`, `seq_flagged_iterations: Option<FlagCounts>`,
  `seq_latency_band_ms: Option<(f64, f64)>` (p10/p90 of the end-to-end span under `mode=mc`).
- Request-level metrics (TTFT, TPOT) inherit per-iteration bands only in `mode=mc` by running the DES on the mean
  path and attaching the iteration bands; full propagation of draws through the DES is out of scope for v1 and said so.

## 5. Versioned parameters

- A parameter bundle directory: `coefficients.json` (Phase 1A), `params.json` (Phase 1B), `MODEL.json`
  `{model_version, contract_version, coefficients_sha256, params_sha256, sources_sha256, supported_regimes, low_confidence_classes}`.
- The preset names the bundle; the loader verifies every hash and refuses on mismatch; `RunMeta` records the bundle
  hashes and `model_version`; the cost log manifest records them too.
- Model versions are immutable; a revised model is a new bundle with a new `model_version` and a new validation capture
  under the protocol.

## 6. Preset compatibility

A new optional arch key (Rust-authoritative, surfaces automatically in the launcher Registry):

```yaml
arch:
  type: deepseek_v41_vllm
  sequence_model:            # absent or enabled: false -> existing behaviour, bit-identical
    enabled: true
    bundle: profiling/env/arm64-gb300/dsv41-flash-tp4/model/1b-29082793   # MODEL.json inside
    mode: median             # median | mc
    draws: 32
    seed: 0
```

implemented as `#[serde(default)] sequence_model: Option<SequenceModelConfig>` on `ArchConfig::DeepseekV41Vllm`
(precedent `decoder_swa_bounded_replay`, `config.rs:404`), `cache_key = false` for `draws`/`seed`/`mode` is **wrong**:
they change results, so all fields are part of the cache key. Public presets are not touched (not authorised, and the
public DSV4.1 preset targets B200); a GB300 preset would be a new non-public file under `presets/` with `gpu: NVIDIA GB300`.

Implementation choice for v1: the adapter calls the Python reference through the existing PyO3 bridge pattern
(`PyModule::import("seqsim.bridge")`, one call per iteration, vectorised draws inside Python; ≈ 1–2 ms per iteration
for 32 draws, i.e. ≈ 35 s for the 17.7k-iteration run2 schedule) behind the contract above; a Rust port is a later,
parity-tested replacement.

## 7. Regression and rollback tests

| test | content |
|---|---|
| flag-off parity (golden) | record `raw/cost_log` and `summary.json` of the DSV4.1 preset runs **before** the change; after the change with `sequence_model` absent the parquet cost log and summary are bit-identical (new optional columns null; a schema-tolerant comparison that fails on any numeric difference) |
| flag-off parity on all existing goldens | `tests/golden/**` unchanged; `just test-cpu` and `cargo test -p simulator --lib` green |
| contract parity | a fixed synthetic schedule (CPU, deterministic) run through the Rust adapter and through `phase1b/seqsim` directly: identical `gpu_ms`, `host_lead_ms`, status, flags (tolerance 0 in `median` mode; bitwise for `mc` with the same seed) |
| sequence invariants through the adapter | `phase1c/tests/test_invariants.py` invariants re-checked on the Rust side's recorded timeline (monotone GPU order, lead definition, scheduler rule) |
| refusal | unsupported configurations (TP2, other GPU, other arch, speculative decoding) fail `check_input` at run start with a message; no iteration is costed |
| hash and version | bundle hash mismatch, contract version mismatch, model_version mismatch → refused |
| rollback | removing the key (or `enabled: false`) restores the previous behaviour with no other change; an environment kill switch `SERVINGSTUDIO_DISABLE_SEQUENCE_MODEL=1` forces `enabled: false` and logs it; the parameter bundle of the previous version stays in the tree (immutable) |
| determinism | two runs with the same preset and seed produce identical cost logs |

## 8. Out of scope for v1 (stated)

Pipeline stages, EP/DP topologies other than the captured recipe, request-level band propagation through the DES,
energy, and any arch other than `deepseek_v41_vllm` on GB300.
