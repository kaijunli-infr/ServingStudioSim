# Proposal: GB300 Llama 3.1 8B BF16 TP2 and TP4 profiling (not approved, not started)

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

**One blocker needs a decision before any TP run** (section "Communication
kernels"): the arm64 torch wheel in this environment is built with
`USE_NCCL=OFF`, so the all-reduce runner the Llama TP arch declares cannot
start. The 1854 single-GPU rows are not affected.

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

**Blocker.** In this environment (`torch 2.10.0+cu130`, the aarch64 wheel from
the PyTorch cu130 index):

```
torch.distributed.is_nccl_available() -> False        torch.__config__: USE_NCCL=OFF
nvidia-nccl-cu13 2.28.9 is installed (libnccl.so.2), but torch is not linked to it
```

So `dist.init_process_group(backend="nccl")` cannot work here. The `nvshmem`
all-reduce runner does not avoid this: its launcher also creates a
`cpu:gloo,cuda:nccl` group. The x86 reference stack uses torch's NCCL
(`nvidia-nccl-cu12` 2.27.5 in the repository lock).

Options, none taken:

1. **A torch build with NCCL for arm64.** Find an aarch64 CUDA 13 torch 2.10
   wheel built with `USE_NCCL=ON` (to be checked: NVIDIA's own PyTorch
   builds), or build torch from source (hours, tens of GB; changes the stack
   every accepted TP1 row was measured on). Keeps the runner unchanged.
2. **Call NCCL directly.** A small runner that drives `ncclAllReduce` from
   the installed `libnccl.so.2` through ctypes, with the same launch, warm-up
   and timing protocol. Same library call torch's NCCL backend makes, but it
   is a new runner, and NCCL 2.28.9 instead of the 2.27.5 behind the x86
   rows. It would be registered so that the row is honest about its path
   (your call whether as `nccl` or a new backend name).
3. **Defer the collectives.** Fill the 1854 single-GPU rows now; TP2 / TP4
   prediction stays blocked on 19 rows each until 1 or 2 is decided.

Recommendation: option 3 now, option 2 as a separately approved step with a
correctness probe (sum equality on every rank) and a cross-check of its
timing protocol against an existing x86 `nccl` row before any fill. This
needs your decision; option 2 falls under "new kernel implementations or
backend substitutions", which is not approved.

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

- Phase C needs the whole tray idle. GPUs 0 and 1 currently carry about
  120 MiB each that no process claims (see the final report; the likely owner
  is the `rustfs` KV store of the KVSim work). It did not disturb single-GPU
  timing, but an all-reduce shares the links, so phase C should run with the
  owner's agreement or after that store is stopped by its owner.
- Phase B on GPUs 2 and 3 avoids those two GPUs.
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
| option 2 runner, if approved | no new packages (`nvidia-nccl-cu13` is installed) |
| option 1, if chosen | 3–5 GB for a wheel; 30 GB or more for a source build |
| **phase A + B + C** | **under 0.5 GB** in `$GB300_ROOT` (now 12 GB; the shared volume has 25 TB free) |

## What an approval would need to say

1. Phase A (single-GPU, 1854 rows, tray12 GPU 1, 75 min budget): go or no.
2. The NCCL route: option 1, 2 or 3.
3. For phases B and C: which GPUs, and whether the 120 MiB contexts on GPUs 0
   and 1 must be gone first.
4. Whether TP2 / TP4 presets (`predict_llama3_8b_gb300_tp{2,4}_iter.json`)
   should be added to `presets/` once the rows exist. They are not added now:
   a public preset without rows would fail the preset tier.

## Appendix: the idle memory on tray12 GPUs 0 and 1 (read-only investigation, 2026-11-10)

Observation: `memory.used` is 123 MiB on GPU 0 and 111 MiB on GPU 1 (GPU 1
read 129 MiB during the first TP1 batch and 115 MiB before the KV fill), 3 MiB
on GPUs 2 and 3, utilization 0 %, and `nvidia-smi` lists no compute process
(`--query-compute-apps`, the process table and `pmon` are all empty).

What was looked at, without touching anything:

- The only user process holding NVIDIA device files that this account can
  see is `rustfs server` (pid 34241, user kevin, started 2026-11-09 02:41
  UTC, binary `~/workspace/rustfs-kv/bin/rustfs`, listening on port 9100):
  it has `libcuda.so`, `libcudart.so.13.4` and the UCX CUDA transports of its
  NIXL plugin mapped, open descriptors on `/dev/nvidia0`–`3`, `nvidiactl` and
  `nvidia-uvm`, and no `CUDA_VISIBLE_DEVICES`. It belongs to the KV-store
  work in `~/workspace/rustfs-kv`, not to this project.
- The container `inkling-rl-vllm` (up 5 weeks, all GPUs requested) runs only
  `sleep infinity`; it is not the holder.
- `nvidia-persistenced` and the kernel threads are the usual ones; persistence
  mode is enabled, MIG is off, accounting mode is off.
- The amount changes slowly (129, 115, 111 MiB over six hours), which fits a
  live process more than a leaked context.

Conclusion: `rustfs` is the probable owner, **not proven**. Not explained: why
it is absent from the compute-app list, and why only GPUs 0 and 1 show memory
when it has all four open. Processes of other users cannot be inspected from
this account, so one of those cannot be ruled out.

Non-disruptive ways to settle it (for the user or the owning session):
`sudo lsof /dev/nvidia1` lists every holder; or note `memory.used` the next
time that store is stopped or restarted for its own reasons. No GPU reset and
no process termination is proposed, and none was done. It had no measurable
effect on the single-GPU rows (probe, worker and repeat timings agree within
2.4 %).
