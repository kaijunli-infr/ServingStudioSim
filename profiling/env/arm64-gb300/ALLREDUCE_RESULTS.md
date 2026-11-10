# GB300 TP2/TP4 all-reduce probe and fill, 2026-11-10

The 38 `all_reduce`/nccl rows a Llama 3.1 8B BF16 TP2 or TP4 replica needs are
measured, in the private database only. With them the dry run reports 0 missing
specs for TP1 (1236), TP2 (1255) and TP4 (1255). The tracked
`profiling/profile.db` is unchanged (sha256 `41c9cbe5…bd82af`).

Two things to know before using the rows:

- **The small-message rows are noisy.** Below 16 MiB the all-reduce time is a
  latency floor of 35 to 60 µs that varies by about 20 % between rank groups and
  shows single-row spikes of 2 to 3 times in roughly 1 of 12 measurements. Four
  stored rows were such spikes and were re-measured (see "Re-measured rows").
- **The floor is 1.5 to 1.9 times the H200 floor.** From 16 MiB up, GB300 is
  faster than H200 (0.56 times at 1 GiB). The driver's validator flagged the
  TP2 median ratio (1.54, limit 1.5) and stopped; the stage was continued after
  review with the reason written into the run log.

## Runs

tray12, rack 3. TP2 on GPUs 2 and 3, TP4 on GPUs 0 to 3, rank r on the r-th GPU.
Driver `run_allreduce_fill.sh`, existing runner `profiling/runners/comm/nccl.py`
unchanged. GPU window 16:42:54 to 16:50:33 UTC (7 min 39 s of the 30 allowed;
231 s inside drivers). Disk: 15 MB of artifacts, the private DB grew by 8 KiB.

| run dir under `$GB300_ROOT/artifacts/allreduce` | commit | outcome |
| --- | --- | --- |
| `tp2-run-20261110-164254` | `09d2006f` | probe pass, 19 rows written, validator ANOMALY (ratio to H200), stopped |
| `tp2-run-20261110-164451` | `14cc56e7` | 3 fresh repeats for diagnosis, no row written |
| `tp2-run-20261110-164655` | `48a816d5` | 2 rows re-measured with `--force`, 3 fresh repeats, done |
| `tp4-run-20261110-164806` | `48a816d5` | probe pass, 19 rows written, validator PASS, repeats flagged, stopped |
| `tp4-run-20261110-164941` | `48a816d5` | 2 rows re-measured with `--force`, 3 fresh repeats, done |

All five ran from a clean worktree. Environment guard (`env_guard.py`) passed
before and after every run with an identical package fingerprint: torch
2.10.0+cu130, CUDA 13.0, NCCL 2.28.9, `is_nccl_available()` true,
`UV_NO_SYNC=1`. All 30 isolation checks show no compute process on any tray12
GPU and the expected UUIDs. No timeout fired, nothing was killed.

## Probe

| | TP2 | TP4 |
| --- | --- | --- |
| rank to GPU | rank 0 GPU 2, rank 1 GPU 3 | rank r GPU r |
| SUM mismatches at 8 KiB and 64 MiB, every rank | 0 | 0 |
| process group ready (spawn, import, TCP rendezvous) | 2.6 to 3.0 s | 4.1 s |
| first collective (NCCL communicator creation) | 21.9 s cold, 1.25 s warm | 1.9 s |
| steady state at 8 KiB | 41 to 46 µs | 40 µs |
| steady state at 64 MiB | 148 µs, 452 GB/s bus | 184 µs, 547 GB/s bus |

The cold 21.9 s is NCCL's first kernel load on this machine (`kernels 20.91`
in its init timings); every later group needed about 0.5 s for it. The cache
directory under `$GB300_ROOT/cache` did not grow, so where that is cached is
not established.

Topology from `NCCL_DEBUG=INFO`: one node, every channel `via P2P/CUMEM`
(direct CUDA peer-to-peer over the NVLink switch; no shared-memory or network
path), ring order 0 1 (2 3), `MNNVL 0`. NVLS is reported available; whether
NCCL used it for a given size is not logged at this level. This matches what the simulator
assumes: `attn_block_tp` and `mlp_block_tp` each cost one all-reduce of the
full tokens × hidden × 2 bytes buffer over `tp_size` ranks on fabric `nvlink`,
and the runner times `dist.all_reduce` of exactly that buffer, one rank per GPU.

Watch items from the proposal: stack limit is 8192 KiB and the driver runs with
`NVreg_CoherentGPUMemoryMode=driver`; neither a crash nor a hang occurred.

## Rows

Time per all-reduce in µs; bus bandwidth in GB/s.

| bytes | TP2 | TP4 | TP4 / TP2 | TP2 bus | TP4 bus | TP2 / H200 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 KiB | 57.4 | 51.8 | 0.90 | 0.1 | 0.1 | 1.88 |
| 64 KiB | 52.9 | 49.2 | 0.93 | 1.2 | 2.0 | 1.63 |
| 512 KiB | 46.2 | 43.3 | 0.94 | 11.3 | 18.2 | 1.44 |
| 4 MiB | 61.8 | 43.9 | 0.71 | 67.9 | 143 | 1.74 |
| 16 MiB | 59.8 | 68.3 | 1.14 | 281 | 368 | 0.78 |
| 64 MiB | 157 | 188 | 1.20 | 428 | 535 | 0.62 |
| 256 MiB | 526 | 711 | 1.35 | 511 | 566 | 0.58 |
| 1 GiB | 1849 | 2680 | 1.45 | 581 | 601 | 0.56 |

Scaling: at 1 GiB TP4 takes 1.45 times TP2, against 1.5 for a ring at equal
link bandwidth (H200: 1.39). The floor does not grow with rank count. TP4
against B200 at identical arguments has a median ratio of 1.22; the full
distributions are in `validate.json` in the run dirs.

## Repeatability

Three fresh rank groups per stage, 19 specs each, compared with the stored rows
(final run of each stage, after the re-measurement).

| range | TP2 median diff | TP2 worst | TP4 median diff | TP4 worst |
| --- | --- | --- | --- | --- |
| 16 MiB and up (7 sizes) | 2.2 to 2.4 % | 61 % (one 32 MiB spike) | 0.6 to 3.1 % | 36 % (one 64 MiB spike) |
| below 16 MiB (12 sizes) | 7 to 20 % | 52 % | 13 to 27 % | 2.5 times (1 MiB spikes in two repeats) |

The validator's rule (median under 5 %, worst under 25 %) is not met in the
floor range. Typical fresh values there: TP2 44 to 52 µs, TP4 46 to 55 µs.
Stored TP2 rows from 16 MiB to 64 MiB sit 5 to 12 % above their repeats.
The cause of the floor spread is not established. The timed window is 100
back-to-back calls, so one stall of a few milliseconds on the host accounts
for a spike of this size; tray12 also ran a CPU-busy object store at the time.

## Re-measured rows

Rule (`allreduce_spikes.py`): a stored row is re-measured when it differs by
more than 25 % from the median of three fresh repeats.

| stage | bytes | first value | fresh repeats | stored now |
| --- | --- | --- | --- | --- |
| TP2 | 128 KiB | 76.1 µs | 47.3, 36.8, 47.4 | 61.0 µs |
| TP2 | 4 MiB | 139.6 µs | 45.5, 38.5, 48.0 | 61.8 µs |
| TP4 | 16 KiB | 89.8 µs | 48.1, 53.2, 46.4 | 52.0 µs |
| TP4 | 64 KiB | 73.3 µs | 47.3, 46.9, 47.7 | 49.2 µs |

The two TP2 values are still about 30 % above their later repeats. They were
not re-measured a second time.

## Predictions

`timing-predict --no-gpu` for the two cases of `predict_llama3_8b_iter_cases.json`
against the private DB, from the main checkout's own CPU venv so the GB300 venv
cannot be touched. Outputs in `$GB300_ROOT/artifacts/allreduce/predict/`.
Times in µs per iteration; compute is the per-rank kernel sum.

| case | TP | total | per-rank compute | all-reduce | communication share |
| --- | --- | --- | --- | --- | --- |
| prefill, 2048 tokens | 1 | 21,180 | 21,180 | 0 | 0 % |
| | 2 | 15,549 | 11,724 | 3,825 | 24.6 % |
| | 4 | 12,142 | 7,769 | 4,373 | 36.0 % |
| decode, 64 requests × 600 cached | 1 | 4,169 | 4,169 | 0 | 0 % |
| | 2 | 5,665 | 2,706 | 2,959 | 52.2 % |
| | 4 | 4,951 | 2,182 | 2,769 | 55.9 % |

Each iteration has 64 all-reduces (two per layer): 16 MiB each in the prefill
case, 512 KiB in the decode case.

Plausibility:

- Prefill gets faster with TP (1.36 times at TP2, 1.74 at TP4); per-rank
  compute falls to 55 % and 37 % of TP1.
- Decode gets slower than TP1 at both TP sizes, because 64 floor-bound
  all-reduces cost about 2.9 ms against 4.2 ms of TP1 compute. That is the
  expected shape for a small model, and the same rows on H200 give about 2 ms.
- The decode all-reduce cost at TP4 is below TP2 only because the stored
  512 KiB rows are 43.3 and 46.2 µs, a difference inside the floor noise. The
  decode totals carry roughly ±10 % from this.
- The fa2 prefill kernel is not faster at TP4 (2,712 µs) than at TP2 (2,646).
  Those are single-GPU rows from the earlier fill, not all-reduce rows.

## Provenance

- Private DB: `$GB300_ROOT/db/profile-gb300.db`, sha256 `9474a224…2be1af`,
  74,510,336 bytes, 3128 GB300 rows (3090 + 38).
- Run records: `09d2006f` (17 TP2 rows), `48a816d5` (2 TP2 and 19 TP4 rows);
  CUDA 13.0, driver 610.57.04. NCCL 2.28.9 (the x86 rows used 2.27.5).
- GPUs: `GPU-0294efb9…`, `GPU-9574ced0…`, `GPU-d2cef419…`, `GPU-098db872…`;
  manifests, probe reports, NCCL logs, GPU samples and validator reports are in
  each run dir.
