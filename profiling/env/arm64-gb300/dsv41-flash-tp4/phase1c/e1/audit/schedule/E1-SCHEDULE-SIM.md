# Revised full-E1 schedule, capture window and storage (simulation from accepted measurements)

Wave durations = packet plan x 0.9856 (measured nsys-on waves: b=8 32k 9.022 s vs planned 9.1 s; b=48 32k 32.04 s vs 32.7 s).
Trace span 895.6 s (packet 940.6 s); waves overlapping the previous wave: 0 of 36; iterations 38104; kernel rows 215.9 M.

| b | KV | wave | arrival s | sim duration s | sim end s | gap to next arrival s | iterations | kernel rows M |
|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 0 | 0.0 | 4.93 | 4.93 | 7.6 | 1025 | 5.8 |
| 1 | 4096 | 1 | 12.5 | 4.93 | 17.43 | 7.6 | 1025 | 5.8 |
| 2 | 4096 | 0 | 25.0 | 4.73 | 29.73 | 7.4 | 1025 | 5.8 |
| 2 | 4096 | 1 | 37.1 | 4.73 | 41.83 | 7.4 | 1025 | 5.8 |
| 4 | 4096 | 0 | 49.2 | 4.83 | 54.03 | 7.7 | 1025 | 5.8 |
| 4 | 4096 | 1 | 61.7 | 4.83 | 66.53 | 7.6 | 1025 | 5.8 |
| 8 | 4096 | 0 | 74.1 | 5.72 | 79.82 | 8.0 | 1026 | 5.81 |
| 8 | 4096 | 1 | 87.8 | 5.72 | 93.52 | 8.0 | 1026 | 5.81 |
| 16 | 4096 | 0 | 101.5 | 7.29 | 108.79 | 8.7 | 1028 | 5.82 |
| 24 | 4096 | 0 | 117.5 | 9.17 | 126.67 | 9.8 | 1030 | 5.85 |
| 32 | 4096 | 0 | 136.5 | 10.45 | 146.95 | 10.6 | 1032 | 5.86 |
| 48 | 4096 | 0 | 157.5 | 12.81 | 170.31 | 11.7 | 1036 | 5.89 |
| 1 | 32768 | 0 | 182.0 | 5.13 | 187.13 | 7.7 | 1026 | 5.81 |
| 1 | 32768 | 1 | 194.8 | 5.13 | 199.93 | 7.8 | 1026 | 5.81 |
| 2 | 32768 | 0 | 207.7 | 5.42 | 213.12 | 7.7 | 1028 | 5.82 |
| 2 | 32768 | 1 | 220.8 | 5.42 | 226.22 | 7.8 | 1028 | 5.82 |
| 4 | 32768 | 0 | 234.0 | 6.51 | 240.51 | 8.4 | 1032 | 5.84 |
| 4 | 32768 | 1 | 248.9 | 6.51 | 255.41 | 8.4 | 1032 | 5.84 |
| 8 | 32768 | 0 | 263.8 | 8.97 | 272.77 | 9.6 | 1040 | 5.89 |
| 8 | 32768 | 1 | 282.4 | 8.97 | 291.37 | 9.6 | 1040 | 5.89 |
| 16 | 32768 | 0 | 301.0 | 13.8 | 314.8 | 12.2 | 1056 | 5.98 |
| 24 | 32768 | 0 | 327.0 | 18.92 | 345.92 | 14.9 | 1072 | 6.09 |
| 32 | 32768 | 0 | 360.8 | 23.46 | 384.26 | 17.2 | 1088 | 6.18 |
| 48 | 32768 | 0 | 401.5 | 32.23 | 433.73 | 21.9 | 1120 | 6.36 |
| 1 | 98304 | 0 | 455.6 | 6.11 | 461.71 | 8.1 | 1030 | 5.83 |
| 1 | 98304 | 1 | 469.8 | 6.11 | 475.91 | 8.2 | 1030 | 5.83 |
| 2 | 98304 | 0 | 484.1 | 7.2 | 491.3 | 8.8 | 1036 | 5.86 |
| 2 | 98304 | 1 | 500.1 | 7.2 | 507.3 | 8.8 | 1036 | 5.86 |
| 4 | 98304 | 0 | 516.1 | 10.25 | 526.35 | 10.2 | 1048 | 5.93 |
| 4 | 98304 | 1 | 536.6 | 10.25 | 546.85 | 10.2 | 1048 | 5.93 |
| 8 | 98304 | 0 | 557.1 | 16.36 | 573.46 | 13.5 | 1072 | 6.07 |
| 8 | 98304 | 1 | 587.0 | 16.36 | 603.36 | 13.6 | 1072 | 6.07 |
| 16 | 98304 | 0 | 617.0 | 28.58 | 645.58 | 19.9 | 1120 | 6.34 |
| 24 | 98304 | 0 | 665.5 | 41.2 | 706.7 | 26.4 | 1168 | 6.64 |
| 32 | 98304 | 0 | 733.1 | 53.13 | 786.23 | 32.7 | 1216 | 6.91 |
| 48 | 98304 | 0 | 818.9 | 76.68 | 895.58 | nan | 1312 | 7.45 |

## Capture window

* profiler start delay 0.0 s: waves starting before the profiler is armed: none; partially captured 0, fully missed 0.
* profiler start delay 10.0 s: waves starting before the profiler is armed: ['b1 kv4096 w0']; partially captured 0, fully missed 1.

## Storage and post-processing (measured rates: sqlite 149 B/row, rep 29 B/row, export 6.3 s/Mrow, parse 17.5 s/Mrow)

| .nsys-rep | .sqlite | parsed.json + kernels parquet + host_timeline | routes + corpus | raw total | export | parse |
|---|---|---|---|---|---|---|
| 6.2 GB | 32.1 GB | 3.6 GB | ~0.44 GB | ~42.6 GB | ~22.6 min | ~63.0 min (linear scaling; Phase 0 run2 at 97 M rows took 29 min, i.e. 18 s/Mrow, so this is an upper estimate) |

GPU-held estimate: startup 23.3 min (measured on the b=8 pass) + warm-up and capture 16.4 min + nsys stop/report finalisation ≤ 20 min = ~59.7 min; export and parse are CPU-only on the tray while the launcher still holds the pass.
