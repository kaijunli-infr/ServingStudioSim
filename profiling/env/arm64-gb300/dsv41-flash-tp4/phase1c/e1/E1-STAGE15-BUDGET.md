# Stage 1.5 GPU occupancy and storage budget (Phase 1.5 item F) — separate from the full E1 capture

Basis: Stage 1 measurements (two dry passes on tray16, 2026-11-14) and the Phase 0 `quadrant_nsys` pass (2026-11-12).

## Per-launch anatomy (measured, Stage 1)

| phase | hook off | hook on |
|---|---|---|
| server start -> model loaded | 1 min 44 s | 1 min 43 s |
| first CUDA-graph capture + KV init | ~1 min | ~1 min |
| **FlashInfer autotune (re-tunes every launch; AUTOTUNER-INVESTIGATION.md)** | **19 min 08 s** | **18 min 58 s** |
| second graph capture, startup complete | ~40 s | ~40 s |
| workload (56 requests, trace-timed) | 50.4 s | 50.8 s |
| corpus packing, teardown, launcher exit | ~2 min | ~2 min |
| **GPU-held per pass (launch -> rc)** | **25 min 23 s** | **25 min 21 s** |

Peak GPU memory 256 849 MiB per GPU; all four GPUs for the whole pass.

## Stage 1.5 passes

| # | pass | GPU-held estimate | basis | storage |
|---|---|---|---|---|
| 1 | `e1_dry_off_v2` (stream=false) | 26 min | Stage 1 hook-off + non-streaming (no effect on the server) | ~60 MB |
| 2 | `e1_dry_on_v2` | 26 min | Stage 1 hook-on | ~110 MB (routes.u16 27 MB + 56 npy 27 MB) |
| 3 | `e1_dry_off2_v2` (control) | 26 min | = pass 1 | ~60 MB |
| 4 | `e1_nsys_val` | 30 min GPU-held (26 + nsys stop/flush ~1 min + sqlite export ~2 min while the server is down but the launcher still holds the pass); parse ~4 min CPU after | quadrant_nsys scaled by 2160/4750 iterations: rep 0.70 GB -> ~0.32 GB; sqlite 3.82 GB -> ~1.8 GB; export 4.5 -> ~2 min; parse 8.8 -> ~4 min | ~2.4 GB (+ host_timeline/parsed/kernels parquet ~0.25 GB) |
| | preflight + between-pass checks (CPU, GPUs idle but reserved) | 4 x ~3 min | Stage 1 | - |
| | Phase 1A table extraction for pass 4 (CPU) | ~5 min | quadrant extraction | ~0.1 GB |
| | cleanup verification and reservation release | 5 min | Stage 1 (19:20 -> 19:27) | - |
| **total, nominal** | | **~2 h 05 min GPU-held, ~2 h 25 min reservation** | | **~3 GB** |
| **requested reservation** | | **4 h (hard stop; ~1.6x margin on the reservation, as in Stage 1 where 5 h were requested and 1 h 14 min used)** | | **budget 20 GB on CephFS (free 21 TB), tray root for nsys temp needs > 100 GB free (1.3 TB)** |

Options if only part is authorized: passes 1-3 alone = ~1 h 25 min nominal (2 h 30 min reservation); pass 4 alone = ~40 min
nominal (1 h 30 min reservation; its hook-on/off comparison then relies on Stage 1 or on passes 1-3 later).

## Autotuner startup

Included in every line above (~19 min of every 26). Not removable without a recipe change (AUTOTUNER-INVESTIGATION.md §4);
no cache pre-warming is assumed. If a launch exceeds 30 min without "Application startup complete", stop it (Stage 1 engine
ready timeout is 3600 s; the preflight prints the time to watch).

## Artifact generation and cleanup

* Collected into the repo (small): preflight/run logs, profile_result, launch json, drive summaries, load_generator
  summaries, token_corpus manifest + provenance, checker and pair-comparison outputs, the three `autotune_configs.json`
  copies (~0.1 MB each), nsys validation JSON. Raw run directories rsync'ed to `$S/artifacts/phase1c-e1-stage15-<date>/run/`
  with SHA256SUMS (`collect_e1_stage15.sh`).
* Cleanup after the last pass: `nvidia-smi` 0 compute apps / 0 MiB on all four GPUs, no vllm/nsys/session_runner processes,
  no tmux server left by the driver, reservation pod deleted, time recorded in the report (as in Stage 1 §6).

## Not in this budget

The full E1 capture (Stage 2: `e1_main`, ~1 h 25 min GPU + 20 min autotune per launch, ~15-20 GB) stays under its own
authorization (packet §9); nothing here pre-commits GPU time to it.
