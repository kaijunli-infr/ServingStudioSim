# GB300 Llama 3.1 8B BF16 TP1 fill (2026-11-10)

Two batches: the first fill of 1166 rows (this section through
"Smallest separate plan"), then Phase 2C, which closed the remaining 70
kv_cache_append rows and the fa3 gap (last section). Final state: **0 / 1236
specs missing** for Llama 3.1 8B BF16 TP1 on `NVIDIA GB300`.

## First batch

Run directory: `$GB300_ROOT/artifacts/tp1/run-20261110-075407` (`run-latest`),
log `$GB300_ROOT/logs/tp1-fill-run1.log`. Source: fork commit `e897db56`
(worktree `ServingStudioSim-gb300`, clean). Device: rack 3 tray12 physical GPU 1,
`GPU-9574ced0-46be-d5b7-0098-01fa1e84dccb`, driver 610.57.04, power limit
1400 W, max clocks 2070 / 3996 MHz; pre and post manifests show the same GPU
state. Software: torch 2.10.0+cu130, triton 3.6.0, flashinfer 0.6.11.post3,
CUPTI 13.0.85 (cu13 wheel), CUDA 13.1 toolkit for JIT. Database: the private
copy `$GB300_ROOT/db/profile-gb300.db` only; the tracked `profiling/profile.db`
checksum is unchanged. Timing provenance: every row carries `run_key` and
`profiler_run_at`, and `_profile_run` records each run's environment.

## Totals

| item | value |
| --- | --- |
| wall time, start to `tp1 fill done` | 23 min 7 s (07:54:07 to 08:17:14 UTC), budget 60 min |
| step wall (fills + validations + probe + repeat) | 23.1 min; the two GEMM batches took 512 s each, everything else under 130 s each |
| rows inserted | 1166: single_gemm 680 (340 torch_linear + 340 torch), rms_norm 63, elementwise 126, flashinfer_attn_prefill fa2 135, flashinfer_attn_decode fa2 162 |
| failures / skips | 0 failed specs, 0 rows flagged `is_outlier`, 0 retries, every batch verdict PASS |
| GPU | 2755 samples at 2 Hz: utilization > 0 in 95 %, ≥ 50 % in 90 %, mean 89 %; power peak 1406 W, mean while busy 1205 W; SM clock fell to 1312 MHz under the large GEMMs (power-limited); peak memory 68.5 GB |
| cache growth | 43.7 MB (Triton 27.7 MB for the elementwise kernels, driver PTX cache +16 MB); total private caches excluding uv: 639 MB |
| DB growth | 74.0 MB to 74.2 MB |
| leftovers | no process on the GPU after the run |

The approved count of 1,463 counted fa3 attempts inside the 270 prefill and
324 decode figures (the dry run counts rows per backend). The fa2-only,
host-runnable set is 1166 rows, all filled.

## Probe (before any fill)

- Identity: one device, `NVIDIA GB300`, cc (10, 3), UUID as expected, exact
  ones-matmul.
- Triton on SM103: for capability ≥ 100 Triton 3.6 selects its bundled
  `ptxas-blackwell` (CUDA 12.9.86, not the 12.8 `ptxas`), target `sm_103a`;
  the 15 PTX files it wrote to the private Triton cache all carry
  `.target sm_103a`. The profiler's own `elementwise_fan_kernel` (fan-in 1 and
  2) and `elementwise_zero_kernel` produced byte-exact results against torch
  at 16.8 MB, 29.4 MB and 262 kB outputs (0 mismatches). First launches
  including autotune: 1.7 s, 0.9 s, 1.0 s.
- CUPTI Kernel10 reading, independent cross-check: torch.profiler (Kineto,
  its own CUPTI client, 50 launches with an L2-flushing `zero_()` between
  them) against the profiler's `Timer.cupti` rows for the same GEMMs:

  | m (n 6144, k 4096) | Kineto p50 | profiler row | difference |
  | --- | --- | --- | --- |
  | 64 | 13.9 µs | 11.5 µs (smoke) / 11.6 µs (fill) | Kineto +20 % at a 12 µs kernel; the two tools flush differently |
  | 2048 | 55.5 µs | 53.1 µs | +4.5 % |
  | 4096 | 107.4 µs | 109.9 µs | −2.3 % |

  A misread record would not agree to a few percent. CUDA 12 compatibility
  of the same source: `g++ -fsyntax-only` against Triton's bundled CUPTI 12.8
  header passes and selects `CUpti_ActivityKernel9`; the four path-ranking
  unit tests pass; the x86 cu12 layout is still the first candidate on a CUDA
  12 torch.

## Per-kind results

Times are CUPTI kernel time with an L2 flush before each launch (the
profiler's standard). Reference ratios are GB300 time over the tracked row at
identical arguments.

| kind / backend | specs | rows | time µs min / p50 / p90 / max | peak | vs B200 (p50, n) | vs H200 (p50, n) | monotonic violations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single_gemm torch_linear | 340 | 340 | 9.2 / 214 / 2752 / 40969 | 2098 TFLOPS (n 4096, k 4096, m 3584) | 0.76 (68) | 0.40 (340) | 0 |
| single_gemm torch | 340 | 340 | 9.2 / 215 / 2751 / 41548 | 2096 TFLOPS | – | 0.40 (340) | 1 (n 4096 k 14336, m 1280 → 1408: −16 %, a cuBLAS kernel switch) |
| rms_norm flashinfer | 63 | 63 | 3.1 / 11.3 / 33.8 / 149 | 7.6 TB/s | – | 0.62 (63) | 0 |
| elementwise triton | 126 | 126 | 2.3 / 17.8 / 107 / 767 | 8.5 TB/s | 0.97 (89) | 0.57 (126) | 0 |
| flashinfer_attn_prefill fa2 | 135 | 135 | 8.4 / 380 / 4993 / 74735 | 471 TFLOPS (prefix 49152, append 32768) | – | 0.88 (135) | 0 |
| flashinfer_attn_decode fa2 | 162 | 162 | 6.1 / 28.8 / 1211 / 2567 | 7.1 TB/s (batch 1, 4.2 M tokens) | – | 0.71 (162) | 0 |

GEMM peak per projection (torch_linear): qkv (n 6144) 2034 TFLOPS at m 3072;
o_proj (n 4096) 2098 at m 3584; up_gate (n 28672) 1925 at m 512; down
(k 14336) 1972 at m 1024; lm_head (n 128256) 1744 at m 5120 (3.08 ms). At
m 1 the rows read 10–155 µs. `torch` (`torch.mm`) and `torch_linear`
(`F.linear`) agree: median ratio 1.001, range 0.88–1.14 over 340 shapes.

**GEMM FLOP accounting** (`profiling/runners/gemm/torch.py`): `flops = 2 * m *
n * k`, `tflops = flops / (time_ms / 1000) / 1e12`, `bytes = (m*k + k*n +
m*n) * 2` for bf16; `Timer.cupti(..., interval_union=True)` so a split-K
pair counts as one GPU-active interval. The 2500 TFLOPS dense BF16 spec peak
is therefore reached at 84 % (2098) at a power-limited clock.

## Repeatability (`--fresh` re-measurement of a stride sample, not inserted)

| kind / backend | pairs | fresh / stored p50 | abs. rel. diff p50 / max |
| --- | --- | --- | --- |
| single_gemm torch_linear | 23 | 0.997 | 0.4 % / 3.6 % (m 64, n 28672) |
| rms_norm flashinfer | 5 | 0.998 | 0.3 % / 0.6 % |
| elementwise triton | 10 | 1.000 | 0.4 % / 18.7 % (two of ten: 52 MB copy +18.7 %, 14.7 → 7.3 MB +17.5 %) |
| flashinfer_attn_prefill fa2 | 11 | 0.999 | 0.1 % / 0.3 % |
| flashinfer_attn_decode fa2 | 13 | 1.001 | 0.3 % / 2.6 % |

The two elementwise outliers are the only repeatability finding. The Triton
kernels are `@autotune`d over five block sizes and the tuner re-benchmarks in
every fresh process, so two processes can settle on different configurations
for the same shape; the stored row is whichever won in the fill process. Eight
of ten samples repeat within 1 %. If this matters for the small-kernel rows,
the remedy is to pin the autotune choice (a runner change, not proposed here).

## Remaining missing configurations

Dry run against the private database after the fill:
`total: 367 / 1533 specs missing`.

| role | missing | why |
| --- | --- | --- |
| unified.attn.attn.kv_cache_append (vllm_cuda) | 70 / 70 | needs `vllm_env`, which does not exist on arm64 (plan below) |
| unified.attn.attn.prefill (fa3 half) | 135 / 270 | FA3 is Hopper-only; the Llama arch still lists it (PROPOSALS.md §A) |
| unified.attn.attn.decode (fa3 half) | 162 / 324 | same |
| every other role | 0 | filled |

Prediction for GB300 stays blocked on all 367: the simulator refuses a
missing row under `--no-gpu`. The 297 fa3 rows need no measurement, only the
§A gate (or a per-run backend map without fa3); the 70 kv_cache_append rows
need the plan below.

## Smallest separate plan for the 70 vLLM kv_cache_append rows (as proposed; executed in Phase 2C below)

Goal: measure `reshape_and_cache_flash_kernel` (vLLM's CUDA kernel) on the
GB300 for the 70 specs (8 KV heads, head_dim 128, block 16, bf16/bf16, NHD,
tensor scale, num_tokens 1 … 65536), through the existing `vllm_cuda` runner,
with no substitution.

1. **Do not build the alignment fork's full vLLM** (`alignment/profiler/vllm`
   at e1423a4d: an x86 cu128 build with FlashInfer, DeepGEMM and FlashMLA; a
   multi-hour arm64 build and a new runtime stack). The runner imports only
   `vllm._custom_ops.reshape_and_cache_flash` and
   `vllm.utils.torch_utils.create_kv_caches_with_random_flash`.
2. **Minimal `vllm_env` for arm64:** a second small uv project under
   `profiling/env/arm64-gb300/vllm/` whose venv installs the same cu130 torch
   plus a vLLM wheel with the compiled `_C` ops. Preferred source: the
   official vLLM aarch64 CUDA wheel of the release the fork is based on
   (upstream `04730e8`; check `pip download vllm==<that release>
   --platform manylinux_2_28_aarch64`). If no arm64 wheel exists for that
   release, build only `csrc/cache_kernels.cu` from the fork's checkout as a
   tiny torch extension exposing `reshape_and_cache_flash`, registered as
   `vllm._custom_ops` shim; the kernel source is 300 lines and has no other
   dependency. Both routes keep the fork's kernel version (the `--env`
   `VIBESIM_VLLM_FORK_ROOT` pointer selects the checkout).
3. **Point the profiler at it** with `VIBESIM_VLLM_FORK_ROOT=<that project
   dir>` so `_VLLM_FORK_PYTHON` resolves to the new venv (no source change),
   and probe: `profiling run kv_cache_append --backend vllm_cuda --fresh` on
   2 specs; verify the written cache against the torch reference in the same
   runner module (`check_packed_append` style, or `smoke_checks.py`-style
   fp32-free byte comparison), and that CUPTI sees exactly one
   `reshape_and_cache_flash_kernel` per launch.
4. **Fill:** the 70 specs, one `profiling run` without `--fresh`, about 2
   minutes of GPU time; validate with `tp1_validate.py batch`; repeat 7 with
   `--fresh`.
5. **Budget:** CPU-only env build 20–60 min (wheel) or 1–2 h (extension),
   GPU under 10 min, disk under 5 GB in `$GB300_ROOT`. Nothing touches shared
   environments or the tracked database.

Then the dry run reports 297 missing, all fa3, and the §A gate decision
unblocks prediction.

## Phase 2C: fa3 gate and the 70 vLLM kv_cache_append rows (2026-11-10)

Approved scope: (A) make FA3 Hopper-only and filter every kernel's declared
backends by the device at build time; (B) run the fork's own
`reshape_and_cache_flash` kernel on arm64 through a minimal shim and fill the
70 rows. Nothing else was measured.

### A. FA3 gate and device filter (commits `83551cdc`, `87fccb80`)

- `fa3` in `flashinfer_attn_{prefill,decode,rect}` declares
  `sm_targets={"sm_90a"}`; `BackendSupport.allows_compute_capability` already
  excluded a declared `sm_targets` set on other architectures.
- `Kernel::build` (Rust, `simulator/src/timing/kernels/engine.rs`) now asks
  `perf_api.supported_backend_subset(kind, backends, compute, kv, gpu_name)`
  and keeps only the backends that run the kernel's dtypes on that GPU. An
  empty subset is `BuildError::NoSupportedBackend`, a hard error naming the
  kernel, dtypes, GPU and declared list. The reduction is logged:
  `kernel unified.attn.attn.prefill (flashinfer_attn_prefill) backends ["fa2",
  "fa3"] -> ["fa2"]: the rest cannot run compute=bf16, kv=bf16 on NVIDIA GB300`.
  Enumerate mode (`cost-trees`) is untouched, so spec export still lists fa3
  and the GPU-specific filter happens only where rows are looked up.
- Checks: 189 pytest (capability + launcher), 574 Rust `timing::` tests; H200
  dry run unchanged (0 / 1533 missing, 270 / 324 attention rows = fa2 + fa3);
  GB300 dry run 70 / 1236 missing before the fill (attention 135 / 162 rows =
  fa2 only). Forcing `"backends": {"main": {"attn": ["fa3"]}}` on the GB300
  preset fails at the binary with `… has no backend that runs compute=bf16,
  kv=bf16 on NVIDIA GB300: declared ["fa3"], none supported on this device`
  (`$GB300_ROOT/artifacts/phase2c/predict_gb300_force_fa3.json`).
- FP8: the Llama arch hardcodes attention compute to bf16, so FP8 attention
  cannot be requested from the preset. At registry level
  `supported_backends("flashinfer_attn_prefill", fp8, fp8, gpu="NVIDIA GB300")`
  is `[]` (tested), and the shim refuses fp8 KV caches with an explicit
  `NotImplementedError` (below). No silent substitution path exists.

### B. The vLLM shim (commits `9f25a680`, `3b17315d`; `vllm-shim/README.md`)

- Source: fork `serendipity-zk/vllm` @ `e1423a4d` (the repository's pin),
  `csrc/libtorch_stable/cache_kernels.cu` compiled **unmodified** (sha256
  pinned in `build_shim.py`, byte-identical to upstream `04730e82`), with a
  40-line binding file that registers the same op schema (the `ops.def`
  literal is checked verbatim against the fork's `torch_bindings.cpp`) via the
  libtorch stable ABI. `vllm/_custom_ops.py` and `vllm/utils/torch_utils.py`
  carry the fork's own lines for the two functions the runner imports.
- Why not the official aarch64 wheel: every published arm64 vLLM wheel pins a
  different torch (2.11 / 2.13 / 2.14) than this env (2.10.0+cu130), so the
  `_C` library cannot load here, and the pinned fork is not a release.
- Build: `cpp_extension.load`, 16 s, into `$GB300_ROOT/vllm-shim/build`
  (4.8 MB total, `manifest.json` with every input hash). Gotchas recorded in
  the shim README: include root must be `csrc/` only; `-DUSE_CUDA
  -DTORCH_TARGET_VERSION=0x020A000000000000ULL`; link `-lcuda` from the
  toolkit stubs; `ninja` from the env venv on PATH; run the probe with
  `python -P` (the repository copy of `vllm/` would otherwise shadow the
  installed shim); Kineto and the repository's CUPTI extension cannot share a
  process.
- `gb300.env` sets `VIBESIM_VLLM_FORK_ROOT=$GB300_ROOT/vllm-shim`, whose
  `.venv` is a symlink to the env venv, so `profiling.exec.env.vllm_env`
  resolves without a source change.

### B. Probe, fill, validation (`run_kv_fill.sh`, run `run-20261110-100407-kv`)

Run directory `$GB300_ROOT/artifacts/tp1/run-20261110-100407-kv`
(`run-latest-kv`), log `$GB300_ROOT/logs/kv-fill-run7.log` (runs 1–6 stopped at
the probe gate on driver/probe bugs, all fixed in the driver before any fill;
none wrote rows). Same GPU as the first batch (tray12 physical GPU 1,
`GPU-9574ced0…`, UUID asserted; 0 compute apps before and after; power 177 W
idle, 596 W peak, SM clock 2070 MHz busy; the same 111–129 MiB foreign context
with no visible process as during the first batch). Source: worktree at
`3b17315d` plus the driver/probe fixes committed next (`profiler_git_hash`
`3b17315d…-dirty`). Wall: 40 s for the whole run, 10.3 s of fill.

| gate | result |
| --- | --- |
| shim identity | `NVIDIA GB300`, cc 10.3, 152 SMs, uuid match |
| correctness, 64 and 8192 tokens | key/value caches **bitwise equal** to the torch reference (`[num_blocks, block, heads, head_dim]`, NHD, bf16); rows changed = tokens × 8 KV heads |
| kernel identity (Kineto, child process) | exactly one `vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0>` per launch |
| CUPTI | matched 1 kernel per iteration; 64 tokens 3.37 µs, 8192 tokens 41.85 µs; `Timer.cupti` 3.45 / 42.00 µs; warm CUDA events 13.69 / 28.38 µs (launch-bound at small sizes, as expected: CUPTI reports kernel time) |
| fp8 KV cache | explicit `NotImplementedError` (the shim builds no `convert_fp8`) |
| worker vs probe (`--fresh`, through the profiler) | 3.44 µs vs 3.37 (ratio 1.022), 41.74 vs 41.85 (0.997); gate 0.8–1.25 |

Fill: `kv_cache_append` / `vllm_cuda`, 70 specs → 70 rows, status ok for all.
`tp1_validate.py batch`: **PASS**, time min 2.89 µs, p50 18.9 µs, p90 70.9 µs,
max 317.8 µs (65536 tokens); 0 monotonic violations, 0 outliers; vs H200
`vllm_cuda` over the 70 common rows GB300/H200 p50 0.657, max 1.049.
Repeat (`specs[3::7]`, 10 specs, `--fresh`, not inserted): **PASS**, fresh/stored
p50 1.0015, max |Δ| 2.4 %. Cache delta 0 bytes, tracked `profiling/profile.db`
unchanged. `_profile_run` id 361: cuda 13.0 (torch's), driver 610.57.04,
`backend_version` NULL (the shim is not an installed distribution; the H200
`vllm_cuda` rows carry NULL too).

### Final validation (CPU only, private DB)

| check | result |
| --- | --- |
| dry run `predict_llama3_8b_gb300_iter` | **0 / 1236 specs missing** (`logs/dryrun-phase2c-gb300-final.log`) |
| GB300 rows in `$GB300_ROOT/db/profile-gb300.db` | single_gemm 680 (torch 340, torch_linear 340), rms_norm 63, elementwise 126, flashinfer_attn_prefill 135 (fa2), flashinfer_attn_decode 162 (fa2), kv_cache_append 70 (vllm_cuda) = **1236**; file 74,207,232 B, sha256 `f2dae6e9…0365b` |
| end-to-end `timing-predict --no-gpu` | rc 0, 31 s; `logs/predict_llama3_8b_gb300_iter/`; cost manifest shows every slot on `NVIDIA GB300` with backends triton / flashinfer / torch+torch_linear / vllm_cuda / fa2 / fa2 |
| per-kernel breakdown | prefill iter (2048 tokens) 21.18 ms: up_gate_proj 39 %, down_proj 20 %, attention prefill 18 % (fa2), qkv 8 %, o_proj 6 %, kv_cache_append 2.0 %; decode iter (64 requests, 38400 cached tokens) 4.17 ms: up_gate 27 %, attention decode 27 % (fa2), down 17 %, kv_cache_append 2.7 % |
| public API, local (`public_api_check.py`) | health ok; catalog lists `NVIDIA GB300` with 1236 rows and the B300 spec peaks (bf16 2500 TFLOPS dense, fp8 5000, nvfp4 13125, HBM 8000 GB/s, NVLink 900 GB/s one-way); `kv_cache_append` rows 70, num_tokens 1 … 65536, 2.89 … 317.8 µs, provenance present (`$GB300_ROOT/artifacts/phase2c/public_api_check.json`) |

### Remaining unsupported paths (by design, not gaps)

- `fa3` on GB300: Hopper-only build, filtered at build time; forcing it is an
  explicit error.
- FP8 attention and FP8 KV caches on GB300: not requestable from the Llama
  arch, `[]` at registry level, refused by the shim. Not profiled.
- `convert_fp8` and every other vLLM op: not built; only
  `reshape_and_cache_flash` exists on arm64.
- TP2 / TP4 / TP8 and FP8 GEMM profiling: not started (not approved).
- The private DB is not merged into `profiling/profile.db`; nothing is deployed.
