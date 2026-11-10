# Phase 2E: what an all-reduce row times, and what the simulator should charge (2026-11-10)

**Finding.** The 43 to 50 µs small-message floor is the host cost of one eager
`torch.distributed.all_reduce` call. It is not GPU or link time. The NCCL
kernel itself runs for 7 to 14 µs at these sizes, and the same collective
replayed from a CUDA graph costs 14 to 20 µs. The stored small-message rows are
therefore 2 to 4 times what a graph-replayed serving step pays, and they are
the only rows in the model that include host launch time.

Nothing in any profile database or in the simulator was changed.

## Stage A: inspection (CPU-only)

**Simulator model.**

- `attn_block_tp` and `mlp_block_tp` each compile to
  `Sum(norm, GEMMs, attention or activation, …, tp_allreduce)`. A layer is the
  sum of both blocks, the iteration is that sum × 32. The all-reduce leaf is
  `time_ms` interpolated linearly over `message_size_bytes` on the
  `(num_gpus, fabric)` curve, with the full tokens × hidden × 2 bytes.
- The tree is a strict serial sum: no overlap of communication and
  computation, no stream model, no host term.
- Every compute leaf is a CUPTI kernel duration (`Timer.cupti`), that is GPU
  execution only. Wall time is the kernel sum × `gpu_time_multiplier`, default
  1.0, documented as "no inter-kernel overhead (kernel-folded time IS the wall
  time)". This is the timing of GPU-bound back-to-back execution, which is what
  a CUDA-graph replay delivers. Several kernels model vLLM's graph capture
  explicitly (`fa4_varlen_attention`, `moe_sink_gate_select`).
- The repository's newer collective runners (`flashinfer_trtllm.py`,
  `flashinfer_mnnvl.py`) already time **CUDA-graph replay**, with the comment
  "Match vLLM's serving path". The plain `all_reduce`/nccl runner is the older
  method and the odd one out.

**Why host gaps are inside the CUDA-event interval.** The runner records an
event, calls `dist.all_reduce` 100 times, records a second event. Each call
does its host work (Python, the process-group layer, `ncclAllReduce` enqueue)
and returns without waiting for the GPU, because `TORCH_NCCL_BLOCKING_WAIT` is
unset. When the kernel is shorter than that host work, the GPU finishes each
collective and idles until the next launch. The second event fires after the
last kernel, so the interval equals the time the slowest rank's host needed to
issue 100 calls. A collective also cannot finish before every rank has
launched its kernel, so the slowest host sets the pace for all.

**Existing raw data.** Phase 2D has per-call host stamps and the event total.
It shows that the host accounts for 75 to 97 % of the interval but contains no
GPU-side kernel timestamps, so it cannot give the execution time. That needed
the experiment below.

## Stage B: GPU diagnostic

tray12, TP2 on GPUs 2 and 3, TP4 on GPUs 0 to 3, rank r on the r-th GPU, clean
worktree at `884772f4`. Five fresh rank groups per stage for methods A, S and
B, one further group for C. Same `dist.all_reduce` (SUM), same bf16 buffers,
same NCCL 2.28.9; SUM verified on every rank, 0 mismatches. GPU time 110 s of
the 10 minutes allowed (TP2 18:05:39 to 18:06:25, TP4 18:07:12 to 18:08:16).
Disk 2 MB. Guards passed before and after with an identical package
fingerprint; all 14 preflight checks clean; no timeout; both databases have
unchanged checksums.

Methods (`allreduce_semantics.py`):

- **A, runner:** the stored-row method, three batches of 100, plus the host
  time inside each call and between calls.
- **S, synchronised:** one call followed by `torch.cuda.synchronize()`, host
  clock. One isolated eager collective.
- **B, kernel:** start and duration of every `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`
  launch from torch.profiler (CUPTI). A kernel that starts before its peers
  waits for them, so the figure used is the median of the rank that waited
  least in each group ("peers ready"). The profiler slows the host to 105 to
  150 µs per call; that widens the waits, not the execution.
- **C, graph replay:** `dist.all_reduce` captured in a CUDA graph. Capture
  succeeded on every rank. A replay after refilling the buffer gave the correct
  sum on every rank (0 mismatches), so the collective really is in the graph.
  Timed with the repository's own `_time_cuda_graph` (10 collectives per
  graph, host wall over replays, slowest rank).

Times in µs: median over the five groups, range in brackets.

| TP | bytes | stored row | A runner | host time inside call | S synchronised | B kernel, peers ready | C graph replay | stored / C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 8 KiB | 57.3 | 47.1 (38.0 to 47.6) | 44.7 (36.2 to 46.0) | 69.6 (61.4 to 70.9) | 8.1 (7.4 to 8.2) | 14.2 | 4.0 |
| 2 | 512 KiB | 46.2 | 48.6 (38.0 to 49.6) | 46.2 (36.4 to 47.5) | 71.9 (65.2 to 73.0) | 9.3 (9.3 to 9.3) | 14.5 | 3.2 |
| 2 | 8 MiB | 54.6 | 47.3 (43.8 to 49.2) | 44.9 (35.0 to 45.4) | 85.8 (82.8 to 87.3) | 41.5 (40.8 to 41.8) | 46.4 | 1.2 |
| 4 | 8 KiB | 46.6 | 47.4 (44.6 to 50.8) | 35.2 (32.4 to 42.7) | 69.2 (67.5 to 73.8) | 12.4 (11.8 to 12.5) | 19.3 | 2.4 |
| 4 | 512 KiB | 43.3 | 66.5 (48.8 to 72.8) | 36.6 (33.3 to 47.2) | 71.5 (70.1 to 79.2) | 14.0 (13.7 to 55.6) | 20.5 | 2.1 |
| 4 | 8 MiB | 45.8 | 50.0 (46.8 to 140.3) | 35.2 (34.8 to 43.5) | 87.4 (86.1 to 91.3) | 41.7 (41.7 to 41.9) | 48.3 | 0.9 |

"Host time inside call" is rank 0's; at TP4 the other ranks' medians reach 49
to 51 µs, and the runner time follows the slowest rank. The time between calls
is 0.2 to 0.4 µs.

What the table shows:

- **Source of the floor.** Method A equals the host time inside the call of
  the slowest rank. The kernel needs 7 to 14 µs up to 512 KiB. The floor is
  host launch cost.
- **Variation across groups.** Host-side figures (A, S, in-call) move by 20 to
  30 % between groups, as in Phase 2D. The kernel time with peers ready moves
  by about 10 % or less (one TP4 group aside, where every rank waited). Method C was
  run in one group per stage, so its group-to-group variation is not measured.
- **Crossover.** At 8 MiB the kernel needs about 41 µs and graph replay 46 to
  48 µs, the same as the stored rows. From there up the rows are GPU-bound and
  the method no longer matters. The 14 rows from 16 MiB up are unaffected.
- **Isolated eager call.** 61 to 91 µs, more than the runner figure, because
  the launch-to-completion latency is no longer hidden by back-to-back calls.

## Is the simulator's per-all-reduce cost right?

- **For graph-replayed execution: pessimistic, by 2 to 4 times below 8 MiB.**
  To my knowledge vLLM replays decode steps from CUDA graphs and issues its TP
  all-reduce inside them through its own NCCL wrapper; that was not verified
  in this phase, which measured capture of `torch.distributed`'s call.
- **For eager execution: roughly realistic for the collective alone, but
  inconsistent.** An eager step pays about 45 µs of host time per all-reduce,
  and a comparable host cost for every other kernel launch, which the model
  does not charge. Only the all-reduce leaf carries it.
- **Never optimistic** at these sizes, except against an isolated synchronised
  call (61 to 91 µs), which no serving loop issues per collective.

Effect on predictions if the small-message cost were the graph-replay figure
(arithmetic on the Phase 2D compute totals, 64 all-reduces per iteration; no
database was changed):

| case | TP1 | TP2 now | TP2 with graph cost | TP4 now | TP4 with graph cost |
| --- | --- | --- | --- | --- | --- |
| decode, 1 request (8 KiB) | 3,375 | 6,210 | about 3,450 | 4,987 | about 3,240 |
| decode, 64 requests (512 KiB) | 4,169 | 5,665 | about 3,630 | 4,951 | about 3,490 |

With the stored rows TP1 is the fastest decode configuration; with graph-replay
costs TP4 and, at 64 requests, TP2 are faster than TP1. The ranking of TP
against TP1 for decode therefore depends on this methodology choice, far more
than on the ±10 % noise examined in Phase 2D.

## Integration policy

Recommended:

1. **Integrate now what is method-independent:** the 1,854 single-GPU rows and
   the 14 all-reduce rows from 16 MiB up.
2. **Do not integrate the 24 small-message rows in either existing form**
   (stored or Phase 2D medians). Documented uncertainty does not cover them:
   the issue is a 2 to 4 times methodology offset, not noise.
3. **Re-measure those 24 rows with graph-replay timing**, the method the
   repository's other collective runners already use, keeping
   `dist.all_reduce`, the tensors and NCCL unchanged. This needs a decision
   and approval: it is a change to the `all_reduce`/nccl runner's timing for
   small messages (or a GB300-only variant), a short GPU run, and replacement
   of 24 private rows. The existing H200 and B200 rows use the old method, so
   cross-GPU comparisons of small-message all-reduce would mix methods until
   those are redone upstream.
4. **Until then,** TP2/TP4 predictions on GB300 are usable for large prefill
   and should not be published for decode or small prefill.

Alternatives: integrate everything as is and document that small-message TP
all-reduce is an eager-launch upper bound (consistent with H200 and B200 rows,
pessimistic for served decode); or hold all 1,892 rows until item 3 is done.

Not established: why groups settle at different host levels; the graph-replay
figure's variation across groups; whether vLLM's own wrapper replays at the
same cost as `torch.distributed`'s call.

## Artifacts

- `$GB300_ROOT/artifacts/phase2e/tp2-run-20261110-180539/`, `tp4-run-*/`
  (`tp2-latest`, `tp4-latest`): `raw/timing-group<g>-rank<r>.json` (batch
  times, per-call host times, synchronised latencies, every kernel record) and
  `raw/graph-group1-rank<r>.json`; manifests, guards, isolation logs.
- `semantics.json` there and committed as `phase2e/semantics.json`.
- Phase 2D raw samples are untouched under `$GB300_ROOT/artifacts/phase2d/`.
- Harness: `allreduce_semantics.py`, `run_allreduce_semantics.sh`.
