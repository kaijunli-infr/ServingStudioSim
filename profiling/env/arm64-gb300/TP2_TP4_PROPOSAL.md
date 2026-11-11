# Proposal: GB300 Llama 3.1 8B BF16 TP2 and TP4 profiling

Status 2026-11-10 evening: phase A (the 1854 single-GPU rows) was approved and is done, see `TP24_RESULTS.md`. Phases B and C (the 38 all-reduce rows) were approved and are done, see `ALLREDUCE_RESULTS.md`. The TP2/TP4 rows were merged on 2026-11-11 with a separate `nccl_graph` all-reduce backend, see `INTEGRATION_RESULTS.md`.

Prepared 2026-11-10 on CPU only (`simulator cost-trees --kernel-configs` and
`timing-predict --dry-run --no-gpu` against the merged database; inputs and
logs in `$GB300_ROOT/artifacts/tp-proposal/`). No GPU was used. Nothing here
runs without a new approval.

## Summary

| | TP2 | TP4 | both |
| --- | --- | --- | --- |
| arch | `llama3_dense_tp`, `tp_size` 2 | `llama3_dense_tp`, `tp_size` 4 | |
| GPUs per replica | 2 | 4 | |
| specs the dry run needs | 1255 | 1255 | |
| already in the database (TP1 rows reused) | 309 | 293 | |
| **missing** | **946** | **962** | **1892 unique rows** (16 shared) |
| of which single-GPU kernels | 927 | 943 | 1854 |
| of which communication (`all_reduce`) | 19 | 19 | 38 |

**Correction (2026-11-10, later the same day).** The first version of this
document reported that the arm64 torch wheel has `USE_NCCL=OFF` and called it
a blocker. That was wrong. The check had run in a damaged environment: a
`uv run` spawned by the analyzer had replaced this env's torch with the
CPU-only `2.10.0+cpu` wheel at 10:06 UTC (after every GPU measurement of the
day; see README "Environment incident"). The real `torch 2.10.0+cu130` wheel
is built with NCCL. See "Communication kernels".

## Missing configurations

Per role, from the dry run (`dryrun-tp2.log`, `dryrun-tp4.log`); a tensor-
parallel rank holds 1/N of the heads and of the MLP width, so most shapes
are new:

| role | kind | TP2 missing | TP4 missing | why |
| --- | --- | --- | --- | --- |
| embedding, input_norm, post_norm, final_norm | elementwise, rms_norm | 0 | 0 | full hidden size on every rank; TP1 rows reused |
| lm_head | single_gemm | 0 / 136 | 0 / 136 | not sharded; TP1 rows reused |
| qkv_proj | single_gemm | 136 | 136 | n = 3072 (TP2), 1536 (TP4) instead of 6144 |
| o_proj | single_gemm | 136 | 136 | k = 2048, 1024 instead of 4096 |
| up_gate_proj | single_gemm | 136 | 136 | n = 14336, 7168 instead of 28672 |
| down_proj | single_gemm | 136 | 136 | k = 7168, 3584 instead of 14336 |
| activation | elementwise | 16 / 63 | 32 / 63 | narrower intermediate |
| attn.prefill | flashinfer_attn_prefill | 135 | 135 | 16 / 4 query / KV heads (TP2), 8 / 2 (TP4) instead of 32 / 8 |
| attn.decode | flashinfer_attn_decode | 162 | 162 | same |
| attn.kv_cache_append | kv_cache_append | 70 | 70 | 4, 2 KV heads instead of 8 |
| attn_block.tp_allreduce | all_reduce | 19 | 19 | `num_gpus` 2, 4; `fabric` nvlink; 4 KiB … 1 GiB |
| mlp_block.tp_allreduce | all_reduce | 0 (same specs) | 0 | shares the attention block's 19 |

Unique new rows after de-duplication across roles and across TP2 / TP4
(the profiler writes two backends per GEMM cell):

| kind | backend(s) | TP2 | TP4 | both |
| --- | --- | --- | --- | --- |
| single_gemm | torch, torch_linear | 544 | 544 | 1088 |
| flashinfer_attn_prefill | fa2 | 135 | 135 | 270 |
| flashinfer_attn_decode | fa2 | 162 | 162 | 324 |
| kv_cache_append | vllm_cuda (the existing shim) | 70 | 70 | 140 |
| elementwise | triton | 16 | 32 | 32 (TP4's contain TP2's) |
| rms_norm | flashinfer | 0 | 0 | 0 |
| all_reduce | nccl | 19 | 19 | 38 |
| **total** | | **946** | **962** | **1892** |

fa3 does not appear: the device gate reduces attention to fa2 on GB300.
The kv_cache_append rows use the same kernel and shim as TP1 with fewer
heads; no new build.

## Communication kernels

The TP arch declares one collective, `all_reduce` with backend `nccl`
(`ALLREDUCE_BACKENDS` in `simulator/src/arch/llama3_dense_tp.rs`): after
`o_proj` and after `down_proj`, every rank all-reduces the full hidden-state
buffer. The runner (`profiling/runners/comm/nccl.py`) spawns one process per
GPU with `torch.multiprocessing`, joins a `torch.distributed` group on the
NCCL backend and times `dist.all_reduce` (50 warm-ups, 100 timed calls per
size, rank 0's CUDA events). Rows are keyed by `(num_gpus, fabric,
message_size_bytes)`; the database has them for H200 (2, 4, 8 GPUs) and B200
(4, 8 GPUs), 19 sizes each.

**NCCL is available; the existing runner needs no change.** Checked on CPU in
the restored environment (`torch 2.10.0+cu130`, aarch64 wheel from the
PyTorch cu130 index):

```
torch.distributed.is_nccl_available() -> True        torch.__config__: USE_CUDA=ON, USE_NCCL=1
from torch._C._distributed_c10d import ProcessGroupNCCL   -> ok
ldd torch/lib/libtorch_cuda.so -> libnccl.so.2 => …/site-packages/nvidia/nccl/lib/libnccl.so.2
nvidia-nccl-cu13 2.28.9 (the version torch 2.10.0+cu130 requires)
```

Independent confirmation from the published wheel and the PyTorch sources
(read without installing anything): `libtorch_python.so` of
`torch-2.10.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl` contains the
`ProcessGroupNCCL` bindings (the `+cpu` wheel does not), `libtorch_cuda.so`
references `libnccl.so.2`, `ncclCommInitRank` and `ncclAllReduce`; the v2.10.0
build scripts (`.ci/manywheel/build_cuda.sh`, `build_common.sh`) set
`USE_NCCL=1`, `USE_SYSTEM_NCCL=1`, `USE_STATIC_NCCL=0` with an rpath to the
pip NCCL for aarch64 CUDA wheels, and the builder image carries NCCL
`v2.28.9-1`. Why the earlier check said OFF: it queried the CPU-only wheel.

What that means for the options discussed earlier:

| option | status |
| --- | --- |
| NCCL-enabled arm64 torch | **already installed**: the env's own wheel. No other wheel, container or source build is needed. |
| source build of torch with NCCL | not needed. For the record: it would need NVPL, ACL, cuDNN / cuSPARSELt / NVSHMEM headers, a 25–40 GB build tree, version-tag care (`2.10.0+cu130`), and would change the stack every accepted row was measured on. |
| direct-NCCL (ctypes) runner | not needed. For the record: about 150 lines after vLLM's `pynccl` wrapper (unique id from rank 0 over a gloo group or a pipe, `ncclCommInitRank`, `ncclAllReduce` on the current stream), plus a correctness probe and a protocol cross-check; it would time raw NCCL without the `torch.distributed` call overhead the x86 rows include. |
| existing runner (`profiling/runners/comm/nccl.py`, `TorchMpLauncher`) | usable unchanged, pending a GPU probe. |

Not yet verified, because it needs GPUs and the all-reduce GPU work is not
approved: that a 2- and 4-rank NCCL group actually initialises on a GB300
tray and returns correct sums. Things to watch at that probe: NCCL on this
stack is 2.28.9 (the x86 rows were measured with 2.27.5); reports from other
GB200 sites of NCCL init crashing on a deep topology search (workaround
`ulimit -s 8192`) and of hangs with `NVreg_CoherentGPUMemoryMode=driver`
(both from vendor notes, not reproduced here); the `nvshmem` backend is not
declared by the Llama TP arch and is not proposed.

**Recommended plan for the 38 all-reduce rows** (needs approval; lowest risk
because it adds no code and no packages):

1. Probe, about 2 minutes, GPUs 2 and 3: the existing runner with `--fresh`
   on 2 sizes (64 KiB, 64 MiB) at `num_gpus` 2; check that every rank holds
   the exact sum (a 20-line check using the runner's own launcher), record
   NCCL's version and transport from `NCCL_DEBUG=INFO`.
2. Fill TP2 (19 rows, GPUs 2 and 3), validate (monotonic in size, bus
   bandwidth below the 900 GB/s one-way NVLink peak, ratio against the H200
   and B200 `nccl` rows), repeat 5 sizes with `--fresh`.
3. Same for TP4 (19 rows, GPUs 0–3) when the whole tray is free.
4. Only if the probe fails for a reason inside torch's NCCL backend: fall
   back to the direct-NCCL runner as a separately approved change.

## Topology

- One tray is the unit: 4 GB300 GPUs under one Grace host. `nvidia-smi nvlink
  -s` on tray12 shows 18 links at 53.125 GB/s on each of the 4 GPUs, all up.
  TP2 and TP4 groups fit inside one tray, which is what `fabric = nvlink`
  with `num_gpus` 2 and 4 means in the existing rows (GPUs of one node).
- No cross-tray (rack-level NVLink switch) measurement is proposed: the arch
  asks only for `num_gpus` 2 and 4, and nothing in the profiler launches
  ranks across hosts.
- The single-GPU rows are per-rank kernels: they are measured on one GPU,
  exactly like TP1. Tensor parallelism changes their shapes, not how they
  are measured.

## GPU allocation

| phase | GPUs | which | exclusive time |
| --- | --- | --- | --- |
| A. single-GPU fill (1854 rows) | 1 | tray12 GPU 1 (`GPU-9574ced0…`), the GPU all TP1 rows came from | about 45 min |
| B. all_reduce TP2 (19 rows) | 2 | tray12 GPUs 2 and 3 | about 5 min |
| C. all_reduce TP4 (19 rows) | 4 | tray12 GPUs 0–3 | about 5 min |

- Phase C needs the whole tray idle. The 90–125 MiB that GPUs 0 and 1 show
  while idle is memory the driver keeps after earlier compute runs (appendix),
  not a live workload, so it does not block phase C.
- Phase B on GPUs 2 and 3 keeps GPU 1 free for single-GPU work.
- tray12 is shared by convention. A container of another session
  (`vo-…`, SGLang) was started on it at 14:25 UTC on 2026-11-10 and was gone
  20 minutes later; `rack3_free` and `nvidia-smi` are to be re-checked
  immediately before each phase, as for TP1.
- Same gates as TP1 before any fill: UUID assertion, no compute apps,
  manifests before and after, 2 Hz GPU sampler, probe then worker-vs-probe
  check, validation and a `--fresh` repeat sample per batch, private database
  only.

## Estimated runtime

From the TP1 campaign on the same GPU (1166 rows in 23 min; 1236 with the
KV rows):

| batch | rows | basis | estimate |
| --- | --- | --- | --- |
| single_gemm, 2 backends | 1088 | TP1: 680 rows in 1024 s; sharded shapes are at most the TP1 size | 20–27 min |
| attention prefill + decode, fa2 | 594 | TP1: 297 rows in under 4.5 min; fewer heads | 6–9 min |
| kv_cache_append | 140 | TP1: 70 rows in 10 s plus 30 s of gates | 2 min |
| elementwise | 32 | TP1: 126 rows in under 130 s | 1 min |
| probes, validation, repeats, manifests | – | TP1: about 4 min | 5 min |
| **phase A total** | 1854 | | **35–45 min, budget 75 min** |
| all_reduce, per group size | 19 | 150 calls per size; 1 GiB at 2.5–4.5 ms on B200 / H200; group start about 1 min | 3–5 min each |

## Disk

| item | estimate |
| --- | --- |
| database growth | under 0.5 MB (1236 rows added 0.2 MB) |
| JIT caches (Triton, FlashInfer, driver) | under 100 MB (TP1 added 44 MB; the fa2 and Triton kernels are already compiled) |
| run artifacts (manifests, sampler, specs, results) | under 50 MB |
| all-reduce through the existing runner | no new packages (`nvidia-nccl-cu13` 2.28.9 is part of the env) |
| **phase A + B + C** | **under 0.5 GB** in `$GB300_ROOT` (now 12 GB; the shared volume has 25 TB free) |

## What an approval would need to say

1. Phase A (single-GPU, 1854 rows, tray12 GPU 1, 75 min budget): go or no.
2. The all-reduce probe and fills through the existing runner (plan above).
3. For phases B and C: which GPUs and when.
4. Whether TP2 / TP4 presets (`predict_llama3_8b_gb300_tp{2,4}_iter.json`)
   should be added to `presets/` once the rows exist. They are not added now:
   a public preset without rows would fail the preset tier.

## Appendix: the idle memory on tray12 GPUs 0 and 1 (read-only investigation, 2026-11-10)

Observation: with nothing running, `memory.used` reads about 123 MiB on GPU 0
and 90–130 MiB on GPU 1, 3 MiB on GPUs 2 and 3, utilization 0 %, and
`nvidia-smi` lists no compute process.

**Whose it is.** On GPU 1 it is left behind by this project's own profiler
runs, not created by another process:

| time (UTC, 2026-11-10) | GPU 1 `memory.used` idle | what had just run |
| --- | --- | --- |
| 06:25, 06:30, 06:40 (before each smoke attempt) | 3 MiB | RustFS already running since 11-09 02:41 |
| 06:26, 06:32 (after attempts 1 and 2) | 3 MiB | plain torch checks used 6.9 GB; the profiler steps failed |
| 06:43 (after attempt 3) | **129 MiB** | the first run in which profiler workers (CUPTI timer) succeeded |
| 10:04 before the KV fill | 115 MiB | TP1 fill |
| 14:28 | 111 MiB | KV fill |
| 15:27 after the TP2/TP4 fill | 93 MiB | TP2/TP4 fill |

The value appears with the first successful profiler run and changes after
each later one. GPU 0's 123 MiB has the same signature (no process, constant
over a 30 s sample) and presumably the same kind of origin from someone
else's earlier work; that was not traced. GPUs 2 and 3, unused in this
period, read 3 MiB.

**RustFS.** `rustfs server` (pid 34241, user kevin, `~/workspace/rustfs-kv`,
started 2026-11-09 02:41 UTC, one node of a four-tray object store) does have
the NVIDIA devices open: `libcuda` and the UCX CUDA transports of its NIXL
plugin are mapped, it holds descriptors on `/dev/nvidia0`–`3` and
`nvidia-uvm`, and the kernel's per-GPU UVM threads for all four GPUs were
created 4 s after it started. But:

- It did not create the memory: GPU 1 read 3 MiB for 28 hours of RustFS
  uptime, until this project's run. The same RustFS on tray14 (same start
  time, same threads) sits on four GPUs at 3 MiB; on tray09 three GPUs read
  about 122 MiB and one 77 MiB.
- It is not using the GPU: its one CUDA-library thread has accumulated 0 CPU
  ticks and 3 context switches since start, GPU 0 showed 0 % utilization and
  a constant 123 MiB over 60 samples, and its NIXL transfer thread did not
  advance in 30 s.
- It is busy on the CPU: about 5 cores (1995 ticks in 20 s), 11 established
  connections on port 9100 from three other trays, a stage log written every
  few seconds. That is object-store traffic over the network, not GPU work.

**Not established:** which driver component keeps the memory after the
profiler's processes exit, and whether RustFS's open UVM registration is what
prevents its release (plausible: tray14's GPUs have had no compute use since
its RustFS started, so there is nothing to retain; a plain torch context
alone left nothing on tray12). Settling that would take a deliberate
experiment on an idle GPU, which was not done.

**Consequences.** No effect on measurements is visible (probe, worker and
repeat timings agree within a few percent across runs with 3, 129, 115 and
111 MiB present). Nothing was reset or terminated, and nothing needs to be.
`run_kv_fill.sh`-style gates that require "no compute apps" remain correct;
a gate on `memory.used == 0` would be wrong on these trays. Worth knowing for
the KV-store work: `launch-k3-hicache.sh` refuses to start when the tray's
GPUs hold more than 2000 MiB in total, which a running profiler fill exceeds.
