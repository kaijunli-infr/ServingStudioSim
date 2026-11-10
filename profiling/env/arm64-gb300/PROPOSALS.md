# GB300 follow-up proposals (nothing here is implemented)

Written after the Phase 2B smoke test (`SMOKE_RESULTS.md`). Each section is a
proposal for approval, with the evidence it rests on.

## A. Gate the `fa3` attention backends to SM90

**Evidence.** FlashInfer builds its FA3 path only for Hopper: the wrappers pick
it when `is_sm90a_supported` holds, the JIT target is `sm90a`, and on GB300 the
smoke probe compiled the kernel for 60 s and then failed with `no kernel image
is available for execution on the device`. The tracked `profile.db` has fa3
rows for H200 only (prefill 1755, decode 1296); B200 has none. The GB300 Llama
3.1 8B dry run would attempt fa3 for all 270 prefill and 324 decode specs.

**Patch** (three files, one line each, plus tests):

```diff
--- a/profiling/kernels/flashinfer_attn_prefill.py
+++ b/profiling/kernels/flashinfer_attn_prefill.py
@@ _SUPPORTS = {
     "fa3": BackendSupport(
         compute=frozenset({DType.BF16, DType.FP8_E4M3}),
         kv=frozenset({DType.BF16, DType.FP8_E4M3}),
+        # FlashInfer builds FA3 for sm90a only (is_sm90a_supported); on
+        # SM100/SM103 the cubin cannot launch (GB300 smoke test, 2026-11-10).
+        sm_targets=frozenset({"sm_90a"}),
     ),
```

The same three added lines in `profiling/kernels/flashinfer_attn_decode.py`
and `profiling/kernels/flashinfer_attn_rect.py`. Tests to add in
`tests/test_backend_capability.py`:

```python
def test_fa3_is_hopper_only():
    for kind in ("flashinfer_attn_prefill", "flashinfer_attn_decode", "flashinfer_attn_rect"):
        assert backend_supports(kind, "fa3", DType.BF16, DType.BF16, gpu="NVIDIA H200")
        for gpu in ("NVIDIA B200", "NVIDIA B300", "NVIDIA GB300"):
            assert not backend_supports(kind, "fa3", DType.BF16, DType.BF16, gpu=gpu)
    # The FP8-attention gap on Blackwell becomes visible instead of a runtime failure.
    assert supported_backends("flashinfer_attn_prefill", DType.FP8_E4M3, gpu="NVIDIA GB300") == []
    assert supported_backends("flashinfer_attn_prefill", DType.FP8_E4M3, gpu="NVIDIA H200") == ["fa3"]
```

Existing tests that call `supported_backends` without a GPU keep passing (no
GPU means no device filter).

**Effect.** The launcher's dry-run `options` column drops fa3 on GB300/B200, so
no fa3 rows are attempted; an explicit backend map that still lists fa3 fails
validation with `backend 'fa3' unsupported … at dtype=bf16, gpu=NVIDIA GB300`.
H200 behaviour is unchanged. One thing to confirm before applying: the Llama
arch's default attention list is `["fa2", "fa3"]` (`llama3_dense.rs`); check
that the JIT fill filters that default through `Role.options` rather than
attempting every default backend, otherwise the Rust default needs the same
device awareness.

## B. Is the `torch` kv_cache_append backend equivalent to `vllm_cuda`?

Both runners live in `profiling/runners/attention/kv_cache_append.py` and share
every input: the same `[num_tokens, num_kv_heads, head_dim]` K and V rows, the
same random `slot_mapping` over `max(256, ceil(T / block_size))` pages, the
same logical cache shape `[num_blocks, block_size, num_kv_heads, head_dim]`
with NHD or HND strides, and the same `bytes_accessed` formula.

| aspect | `torch` | `vllm_cuda` (`reshape_and_cache_flash_kernel`) | equivalent for Llama BF16 TP1? |
| --- | --- | --- | --- |
| destination | `cache[slot // bs, slot % bs] = row` for K and for V | identical mapping inside one kernel | yes |
| dtype | requires `input_dtype == kv_dtype` | same-dtype copy, or bf16 → fp8 with `k_scale`/`v_scale` | yes: the TP1 rows are bf16/bf16, where vLLM copies raw values and ignores the scales |
| scale granularity | tensor only | tensor (per-head via diffkv variant) | yes: the rows use tensor |
| head_dim_v | symmetric or asymmetric | symmetric only | yes: Llama is symmetric |
| padding slots (`slot < 0`) | not handled | skipped | not exercised by the profiler (randperm has no negatives) |
| launches | two (`index_put_` for K, for V) | one fused kernel | **no** |
| timing | `Timer.cuda_event` around both launches, warm cache, no flush | `Timer.cupti` on the one kernel, L2 flushed before each launch | **no** |

So the write contract is the same for every spec the Llama TP1 fill needs
(bf16 in, bf16 cache, NHD, tensor scale, head_dim 128): a `torch` row would
describe the same memory traffic. The measured time is not the same quantity:
two generic scatter launches without a cache flush versus one fused kernel
with a cold L2. Expect `torch` to read higher at small `num_tokens` (two launch
floors) and closer at large `num_tokens` (bandwidth-bound either way); the
exact ratio is unknown until both run on the same GPU, which needs the vLLM
env. Recommendation: do not silently substitute. Either (1) build the vLLM
arm64 env (the alignment fork's `.venv`; the heavier but faithful route), or
(2) if the 70 rows are to be filled now, use the `torch` backend for those
rows only, label their provenance, and treat them as an upper bound to be
replaced. Option (2) needs an arch change (`kv_cache_append_backends` for the
Llama block) or a per-run backend map, both outside the current approval.

## C. Smallest useful BF16 TP1 batch (for approval, not started)

Target: `presets/predict_llama3_8b_gb300_iter.json`; dry run on 2026-11-10
lists 1533 missing specs across 13 roles (CPU-only, `timing-predict --dry-run
--no-gpu`, log `logs/dryrun-gb300-llama8b-20261110-0647.log` on rack 3).

| kernel kind / backend | roles | specs | smoke-validated | env |
| --- | --- | --- | --- | --- |
| single_gemm torch_linear | qkv, o_proj, up_gate, down, lm_head | 680 | yes | host (arm64 venv) |
| rms_norm flashinfer | input_norm | 63 | yes | host |
| elementwise triton | embedding, activation | 126 | **no** (Triton 3.6 bundles a CUDA 12.8 `ptxas`; whether it emits sm_103 code is unverified) | host |
| flashinfer_attn_prefill fa2 | prefill | 270 | yes | host |
| flashinfer_attn_decode fa2 | decode | 324 | yes | host |
| kv_cache_append vllm_cuda | kv_cache_append | 70 | no: needs `vllm_env`, absent on arm64 | – |
| flashinfer_attn_{prefill,decode} fa3 | (same roles) | 594 attempts | fails cleanly (60 s JIT the first time, fast afterwards) | – |

Proposed batch: the 1463 host-runnable rows, in two steps.

1. **Probe (≈ 2 min GPU):** `profiling run elementwise --backend triton` on two
   specs with `--fresh`. Pass = correct Triton compile for sm_103 and a time
   near the B200 row. If Triton cannot target sm_103, stop and report (the fix
   would be pointing Triton at the CUDA 13.1 `ptxas`, `TRITON_PTXAS_PATH`, an
   env change, not a kernel change).
2. **Fill (≈ 30–60 min wall, one GPU, GPU busy ≈ 15 min):** per kind and
   backend, `profiling run <kind> --backend <fa2|torch_linear|flashinfer|triton>
   --gpu-name "NVIDIA GB300" --specs <kind>.json`, without `--fresh`, so rows
   are inserted into `$GB300_ROOT/db/profile-gb300.db` (private copy; the
   tracked DB is untouched). Running per backend means fa3 is never attempted,
   with no code change. Spec lists come from the simulator binary's
   `cost-tree --kernel-configs` for the preset's arch block (to be confirmed
   on the exported file before the run; the dry-run report only carries counts).
   Timing basis from the smoke test: ≈ 0.7 s per GEMM spec, ≈ 0.5 s per
   attention spec plus the one-time 19 s fa2 prefill JIT, ≈ 4 s start-up per
   worker invocation.

Outputs: `$GB300_ROOT/db/profile-gb300.db` (+ tens of MB), artifacts under
`$GB300_ROOT/artifacts/tp1-fill-<stamp>/`, manifests before and after, the
same 2 Hz GPU sampler. Cache growth expected under 200 MB (one fa2 head
configuration; GEMM and norm need no JIT). Validation: `profiling count-missing`
per kind reaches 0 for the five kinds; the dry run then reports exactly 70
missing (kv_cache_append); 5 % of rows spot-checked against the B200/H200
ratios seen in the smoke test (GEMM ≤ B200 time, norm near B200, fa2 decode
near H200 bandwidth × 1.7); outlier flags reviewed. Recovery: the fill is
idempotent (existing rows are skipped), so an interrupted run is simply
restarted. Rollback: delete the private DB copy; nothing else is written.

Remaining after the batch: the 70 kv_cache_append rows (section B decision)
and, if fa3 stays unguarded, the 594 fa3 attempts the launcher would make on a
later full fill (section A). Prediction for GB300 cannot run until the 70 rows
exist.
