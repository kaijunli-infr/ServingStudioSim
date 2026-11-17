# e1_main_v3 — surviving-artifact analysis (host-side only; no Nsight trace exists)

Archive: `/mnt/shared/home/dc/workspace/serving-studio/artifacts/phase1c-e1-main-dsv41-flash-gb300-tp4-20261116/run/e1_main_v3` (frozen, SHA256SUMS verified). Measured window (CLOCK_MONOTONIC ns): (2814954689276700, 2815851225663737).

All latency numbers below are **host-observed under an active Nsight capture and with routing return enabled**; they are routing-instrumented, profiler-perturbed host timings, not clean GPU timings, and must not be treated as validation data for any model.

## 1. Completeness by cell and wave

| b | KV | wave | planned | replayed | routing files | request timings | pure-decode iterations (host) | contiguous | ≥ 400 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4k | 0 | 1 | 1 | 1 | 1 | 1023 | True | True |
| 1 | 4k | 1 | 1 | 1 | 1 | 1 | 1023 | True | True |
| 1 | 32k | 0 | 1 | 1 | 1 | 1 | 1023 | True | True |
| 1 | 32k | 1 | 1 | 1 | 1 | 1 | 1023 | True | True |
| 1 | 96k | 0 | 1 | 1 | 1 | 1 | 1023 | True | True |
| 1 | 96k | 1 | 1 | 1 | 1 | 1 | 1023 | True | True |
| 2 | 4k | 0 | 2 | 2 | 2 | 2 | 1021 | True | True |
| 2 | 4k | 1 | 2 | 2 | 2 | 2 | 1021 | True | True |
| 2 | 32k | 0 | 2 | 2 | 2 | 2 | 1020 | True | True |
| 2 | 32k | 1 | 2 | 2 | 2 | 2 | 1020 | True | True |
| 2 | 96k | 0 | 2 | 2 | 2 | 2 | 1016 | True | True |
| 2 | 96k | 1 | 2 | 2 | 2 | 2 | 1016 | True | True |
| 4 | 4k | 0 | 4 | 4 | 4 | 4 | 1021 | True | True |
| 4 | 4k | 1 | 4 | 4 | 4 | 4 | 1021 | True | True |
| 4 | 32k | 0 | 4 | 4 | 4 | 4 | 1016 | True | True |
| 4 | 32k | 1 | 4 | 4 | 4 | 4 | 1016 | True | True |
| 4 | 96k | 0 | 4 | 4 | 4 | 4 | 1004 | True | True |
| 4 | 96k | 1 | 4 | 4 | 4 | 4 | 1004 | True | True |
| 8 | 4k | 0 | 8 | 8 | 8 | 8 | 1020 | True | True |
| 8 | 4k | 1 | 8 | 8 | 8 | 8 | 1020 | True | True |
| 8 | 32k | 0 | 8 | 8 | 8 | 8 | 1008 | True | True |
| 8 | 32k | 1 | 8 | 8 | 8 | 8 | 1008 | True | True |
| 8 | 96k | 0 | 8 | 8 | 8 | 8 | 980 | True | True |
| 8 | 96k | 1 | 8 | 8 | 8 | 8 | 980 | True | True |
| 16 | 4k | 0 | 16 | 16 | 16 | 16 | 1018 | True | True |
| 16 | 32k | 0 | 16 | 16 | 16 | 16 | 992 | True | True |
| 16 | 96k | 0 | 16 | 16 | 16 | 16 | 932 | True | True |
| 24 | 4k | 0 | 24 | 24 | 24 | 24 | 1016 | True | True |
| 24 | 32k | 0 | 24 | 24 | 24 | 24 | 976 | True | True |
| 24 | 96k | 0 | 24 | 24 | 24 | 24 | 884 | True | True |
| 32 | 4k | 0 | 32 | 32 | 32 | 32 | 1014 | True | True |
| 32 | 32k | 0 | 32 | 32 | 32 | 32 | 960 | True | True |
| 32 | 96k | 0 | 32 | 32 | 32 | 32 | 836 | True | True |
| 48 | 4k | 0 | 48 | 48 | 48 | 48 | 1010 | True | True |
| 48 | 32k | 0 | 48 | 48 | 48 | 48 | 928 | True | True |
| 48 | 96k | 0 | 48 | 48 | 48 | 48 | 740 | True | True |

## 2. Request / token / iteration reconciliation

* requests: planned 450, replayed 450, all SUCCESS True, all finish=length True, all 1024 output tokens True, prefix-cache hits 0.
* routing files: 450 (450 with shape (1024, 40, 6) uint16 and ids < 384; bad 0).
* request timing records in the measured window: 450 (missing 0, unexpected 0).
* iteration records: 38108 in the server log, 38108 in the measured window (indices 1308–39485, 35 index gaps overall); pure-decode 36830, prefill-only 108, mixed 1170, empty 0; pure-decode iterations not attributable to a plan cell: 0.
* decode tokens scheduled in the window: 460350 vs expected 450 × 1023 = 460350; prefill tokens 20275200 vs planned 20275200; window wall 896.2 s.
* pure-decode iterations by batch size: {'1': 6229, '2': 6180, '3': 75, '4': 6130, '5': 48, '6': 51, '7': 57, '8': 6048, '9': 32, '10': 33, '11': 35, '12': 32, '13': 33, '14': 32, '15': 38, '16': 2966, '17': 24, '18': 24, '19': 27, '20': 24, '21': 24, '22': 25, '23': 29, '24': 2892, '25': 16, '26': 16, '27': 18, '28': 16, '29': 16, '30': 17, '31': 20, '32': 2818, '33': 8, '34': 8, '35': 9, '36': 8, '37': 8, '38': 8, '39': 9, '40': 8, '41': 8, '42': 8, '43': 9, '44': 8, '45': 8, '46': 9, '47': 11, '48': 2678}.

## 3. Routing statistics (per-token top-6 of 384 routed experts, 40 layers)

* pooled over all 460 800 tokens: mean per-layer entropy 7.430 bits (uniform = 8.585), mean max/mean expert load 30.31, (layer, expert) pairs never routed 37; consecutive-token expert overlap (sampled) p50 0.882.

| b | KV | wave | tokens | mean layer entropy (bits) [95 % CI, request bootstrap] | mean max/mean load [95 % CI] | mean JSD vs pooled (bits) |
|---|---|---|---|---|---|---|
| 1 | 4k | 0 | 1024 | 6.359 [nan, nan] | 45.64 [nan, nan] | 0.2317 |
| 1 | 4k | 1 | 1024 | 5.972 [nan, nan] | 39.69 [nan, nan] | 0.2967 |
| 1 | 32k | 0 | 1024 | 6.173 [nan, nan] | 48.96 [nan, nan] | 0.2525 |
| 1 | 32k | 1 | 1024 | 6.568 [nan, nan] | 41.34 [nan, nan] | 0.1625 |
| 1 | 96k | 0 | 1024 | 5.649 [nan, nan] | 51.41 [nan, nan] | 0.2720 |
| 1 | 96k | 1 | 1024 | 5.899 [nan, nan] | 39.43 [nan, nan] | 0.3437 |
| 2 | 4k | 0 | 2048 | 6.368 [5.813, 6.368] | 33.66 [33.66, 42.25] | 0.2504 |
| 2 | 4k | 1 | 2048 | 6.459 [5.750, 6.459] | 38.74 [38.74, 48.65] | 0.2098 |
| 2 | 32k | 0 | 2048 | 6.670 [6.298, 6.670] | 38.17 [38.17, 46.38] | 0.1450 |
| 2 | 32k | 1 | 2048 | 6.557 [5.953, 6.557] | 32.14 [32.14, 45.33] | 0.1917 |
| 2 | 96k | 0 | 2048 | 6.811 [6.184, 6.811] | 34.82 [34.82, 47.84] | 0.1762 |
| 2 | 96k | 1 | 2048 | 6.515 [5.595, 6.515] | 35.27 [35.27, 50.58] | 0.1994 |
| 4 | 4k | 0 | 4096 | 6.928 [6.311, 7.014] | 34.72 [33.66, 43.00] | 0.1021 |
| 4 | 4k | 1 | 4096 | 6.964 [6.303, 6.986] | 30.93 [29.71, 38.30] | 0.0975 |
| 4 | 32k | 0 | 4096 | 6.764 [6.239, 6.764] | 35.16 [32.64, 44.22] | 0.1321 |
| 4 | 32k | 1 | 4096 | 7.158 [6.474, 7.158] | 26.95 [26.10, 39.88] | 0.1469 |
| 4 | 96k | 0 | 4096 | 6.699 [6.107, 6.717] | 33.53 [29.08, 45.44] | 0.1596 |
| 4 | 96k | 1 | 4096 | 6.758 [6.107, 6.758] | 31.34 [30.23, 43.69] | 0.1641 |
| 8 | 4k | 0 | 8192 | 7.185 [6.794, 7.162] | 29.85 [25.33, 40.31] | 0.0867 |
| 8 | 4k | 1 | 8192 | 7.134 [6.760, 7.103] | 31.24 [27.03, 37.10] | 0.0721 |
| 8 | 32k | 0 | 8192 | 7.252 [6.876, 7.228] | 28.66 [26.52, 33.71] | 0.0784 |
| 8 | 32k | 1 | 8192 | 7.109 [6.803, 7.088] | 33.54 [33.14, 37.59] | 0.0664 |
| 8 | 96k | 0 | 8192 | 6.955 [6.520, 7.008] | 32.65 [30.65, 37.71] | 0.0745 |
| 8 | 96k | 1 | 8192 | 7.204 [6.763, 7.177] | 29.73 [27.75, 34.93] | 0.0613 |
| 16 | 4k | 0 | 16384 | 7.301 [7.097, 7.289] | 32.96 [30.57, 36.62] | 0.0464 |
| 16 | 32k | 0 | 16384 | 7.146 [6.914, 7.151] | 32.15 [30.44, 34.81] | 0.0470 |
| 16 | 96k | 0 | 16384 | 7.318 [7.061, 7.303] | 28.40 [26.11, 33.32] | 0.0481 |
| 24 | 4k | 0 | 24576 | 7.176 [6.985, 7.193] | 32.51 [31.19, 34.85] | 0.0368 |
| 24 | 32k | 0 | 24576 | 7.192 [6.997, 7.228] | 31.96 [29.52, 34.35] | 0.0417 |
| 24 | 96k | 0 | 24576 | 7.243 [7.041, 7.279] | 31.29 [28.85, 33.68] | 0.0391 |
| 32 | 4k | 0 | 32768 | 7.319 [7.156, 7.348] | 30.47 [29.29, 32.25] | 0.0266 |
| 32 | 32k | 0 | 32768 | 7.282 [7.138, 7.292] | 30.81 [28.79, 32.92] | 0.0208 |
| 32 | 96k | 0 | 32768 | 7.438 [7.282, 7.431] | 29.54 [28.17, 31.41] | 0.0287 |
| 48 | 4k | 0 | 49152 | 7.323 [7.211, 7.330] | 31.90 [30.49, 33.47] | 0.0204 |
| 48 | 32k | 0 | 49152 | 7.383 [7.278, 7.384] | 29.67 [28.60, 31.17] | 0.0165 |
| 48 | 96k | 0 | 49152 | 7.415 [7.295, 7.423] | 30.26 [29.09, 31.81] | 0.0174 |

Wave repeatability (w0 vs w1 of the same cell; different prompts): mean per-layer JSD against a permutation null of the same split sizes.

| b | KV | JSD w0 vs w1 (bits) | null p50 | null p97.5 |
|---|---|---|---|---|
| 1 | 4k | 0.3155 | 0.3155 | 0.3155 |
| 1 | 32k | 0.2924 | 0.2924 | 0.2924 |
| 1 | 96k | 0.4647 | 0.4647 | 0.4647 |
| 2 | 4k | 0.4511 | 0.4406 | 0.4511 |
| 2 | 32k | 0.2424 | 0.2424 | 0.3462 |
| 2 | 96k | 0.3407 | 0.3681 | 0.3875 |
| 4 | 4k | 0.1561 | 0.1584 | 0.2226 |
| 4 | 32k | 0.2443 | 0.2232 | 0.2759 |
| 4 | 96k | 0.2814 | 0.2310 | 0.3092 |
| 8 | 4k | 0.1053 | 0.1233 | 0.1804 |
| 8 | 32k | 0.1259 | 0.1293 | 0.1555 |
| 8 | 96k | 0.1363 | 0.1338 | 0.1762 |

Uncertainty: the request is the independent unit (prompts are distinct enwik9 slices); CIs are percentile bootstraps over requests (B = 400), so cells with 1–2 requests have no meaningful interval. Marginals are pooled over 1024 tokens per request, so token-level autocorrelation within a request is inside the unit, not ignored. Resampling with replacement duplicates requests, which biases the pooled entropy down and the max-load ratio up, so the point estimate can sit at an interval edge: read the intervals as dispersion across requests, not as coverage intervals. For b = 1 and b = 2 the permutation null below has one or two distinct splits.

## 4. Host-side timing distributions (profiler-perturbed; routing-instrumented)

| b | KV | wave | pure-decode iters | elapsed ms p50 | p10 | p90 | mean | sd | first-quarter p50 | last-quarter p50 | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4k | 0 | 1023 | 4.20 | 4.00 | 4.90 | 4.35 | 0.49 | 4.20 | 4.73 | 4.7 |
| 1 | 4k | 1 | 1023 | 4.46 | 4.30 | 5.01 | 4.58 | 0.32 | 4.44 | 4.72 | 5.0 |
| 1 | 32k | 0 | 1023 | 4.45 | 4.23 | 5.00 | 4.55 | 0.44 | 4.50 | 4.31 | 5.0 |
| 1 | 32k | 1 | 1023 | 4.37 | 4.18 | 4.99 | 4.49 | 0.34 | 4.23 | 4.41 | 4.9 |
| 1 | 96k | 0 | 1023 | 4.21 | 3.96 | 4.68 | 4.28 | 0.32 | 4.23 | 4.45 | 4.7 |
| 1 | 96k | 1 | 1023 | 4.42 | 3.96 | 4.79 | 4.39 | 0.36 | 4.01 | 4.46 | 4.8 |
| 2 | 4k | 0 | 1021 | 4.16 | 3.96 | 4.99 | 4.33 | 0.43 | 4.21 | 4.28 | 4.9 |
| 2 | 4k | 1 | 1021 | 4.12 | 3.94 | 4.99 | 4.31 | 0.44 | 4.29 | 4.11 | 4.9 |
| 2 | 32k | 0 | 1020 | 4.22 | 4.08 | 5.08 | 4.44 | 1.07 | 4.16 | 4.58 | 4.9 |
| 2 | 32k | 1 | 1020 | 4.25 | 4.06 | 5.00 | 4.43 | 0.39 | 4.50 | 4.26 | 4.9 |
| 2 | 96k | 0 | 1016 | 4.37 | 4.07 | 4.77 | 4.42 | 0.63 | 4.14 | 4.54 | 4.8 |
| 2 | 96k | 1 | 1016 | 4.43 | 4.06 | 4.86 | 4.45 | 0.31 | 4.11 | 4.75 | 4.9 |
| 4 | 4k | 0 | 1021 | 4.17 | 4.10 | 4.62 | 4.28 | 0.58 | 4.19 | 4.14 | 4.9 |
| 4 | 4k | 1 | 1021 | 4.15 | 4.08 | 4.38 | 4.20 | 0.24 | 4.15 | 4.16 | 4.8 |
| 4 | 32k | 0 | 1016 | 4.29 | 4.18 | 4.59 | 4.35 | 0.27 | 4.29 | 4.26 | 4.9 |
| 4 | 32k | 1 | 1016 | 4.34 | 4.23 | 4.44 | 4.35 | 0.14 | 4.25 | 4.34 | 4.9 |
| 4 | 96k | 0 | 1004 | 4.32 | 4.24 | 4.61 | 4.37 | 0.19 | 4.30 | 4.37 | 4.8 |
| 4 | 96k | 1 | 1004 | 4.34 | 4.28 | 4.41 | 4.35 | 0.14 | 4.35 | 4.34 | 4.8 |
| 8 | 4k | 0 | 1020 | 4.57 | 4.50 | 4.65 | 4.58 | 0.13 | 4.56 | 4.59 | 5.4 |
| 8 | 4k | 1 | 1020 | 4.55 | 4.48 | 4.64 | 4.59 | 0.63 | 4.57 | 4.53 | 5.4 |
| 8 | 32k | 0 | 1008 | 4.77 | 4.70 | 4.84 | 4.83 | 1.54 | 4.78 | 4.77 | 5.4 |
| 8 | 32k | 1 | 1008 | 4.72 | 4.65 | 4.81 | 4.73 | 0.10 | 4.74 | 4.72 | 5.3 |
| 8 | 96k | 0 | 980 | 4.73 | 4.67 | 4.80 | 4.76 | 0.90 | 4.73 | 4.74 | 5.2 |
| 8 | 96k | 1 | 980 | 4.78 | 4.72 | 4.85 | 4.78 | 0.06 | 4.77 | 4.79 | 5.2 |
| 16 | 4k | 0 | 1018 | 5.65 | 5.55 | 5.74 | 5.65 | 0.18 | 5.63 | 5.63 | 6.6 |
| 16 | 32k | 0 | 992 | 5.64 | 5.53 | 5.74 | 5.65 | 0.43 | 5.65 | 5.65 | 6.4 |
| 16 | 96k | 0 | 932 | 5.79 | 5.70 | 5.88 | 5.88 | 2.56 | 5.80 | 5.76 | 6.2 |
| 24 | 4k | 0 | 1016 | 6.92 | 6.80 | 7.04 | 6.95 | 0.82 | 6.96 | 6.88 | 8.1 |
| 24 | 32k | 0 | 976 | 6.97 | 6.87 | 7.08 | 7.04 | 1.88 | 6.97 | 7.01 | 7.9 |
| 24 | 96k | 0 | 884 | 7.09 | 6.93 | 7.23 | 7.09 | 0.19 | 7.15 | 6.97 | 7.2 |
| 32 | 4k | 0 | 1014 | 7.58 | 7.45 | 7.71 | 7.58 | 0.18 | 7.56 | 7.54 | 9.0 |
| 32 | 32k | 0 | 960 | 7.55 | 7.41 | 7.71 | 7.56 | 0.18 | 7.65 | 7.47 | 8.5 |
| 32 | 96k | 0 | 836 | 7.86 | 7.72 | 7.98 | 7.96 | 2.89 | 7.86 | 7.79 | 7.7 |
| 48 | 4k | 0 | 1010 | 8.47 | 8.30 | 8.67 | 8.51 | 0.96 | 8.62 | 8.37 | 10.3 |
| 48 | 32k | 0 | 928 | 8.61 | 8.44 | 8.76 | 8.67 | 1.79 | 8.67 | 8.48 | 9.7 |
| 48 | 96k | 0 | 740 | 8.75 | 8.60 | 8.89 | 8.75 | 0.20 | 8.71 | 8.72 | 7.8 |

EngineCore request timings (TTFT = first schedule to first token; TPOT over 1023 decode steps):

| b | KV | wave | n | TTFT ms p50 | TTFT p90 | TPOT ms p50 | TPOT p90 | client total ms p50 | client − engine ms p50 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4k | 0 | 1 | 320.3 | 320.3 | 4.630 | 4.630 | 5069.4 | 12.5 |
| 1 | 4k | 1 | 1 | 201.4 | 201.4 | 4.895 | 4.895 | 5220.6 | 12.1 |
| 1 | 32k | 0 | 1 | 455.9 | 455.9 | 4.850 | 4.850 | 5437.5 | 20.0 |
| 1 | 32k | 1 | 1 | 457.7 | 457.7 | 4.774 | 4.774 | 5361.3 | 20.1 |
| 1 | 96k | 0 | 1 | 1401.5 | 1401.5 | 4.611 | 4.611 | 6158.2 | 39.7 |
| 1 | 96k | 1 | 1 | 1402.5 | 1402.5 | 4.710 | 4.710 | 6258.1 | 36.8 |
| 2 | 4k | 0 | 2 | 203.7 | 210.6 | 4.882 | 4.961 | 5307.8 | 109.4 |
| 2 | 4k | 1 | 2 | 206.1 | 211.7 | 4.856 | 4.936 | 5284.6 | 110.8 |
| 2 | 32k | 0 | 2 | 677.7 | 855.4 | 4.979 | 5.148 | 5795.4 | 24.5 |
| 2 | 32k | 1 | 2 | 676.6 | 853.9 | 4.976 | 5.145 | 5791.9 | 24.8 |
| 2 | 96k | 0 | 2 | 2093.0 | 2646.5 | 5.440 | 5.969 | 7708.8 | 50.6 |
| 2 | 96k | 1 | 2 | 2090.8 | 2644.5 | 5.458 | 5.986 | 7724.7 | 50.8 |
| 4 | 4k | 0 | 4 | 432.6 | 444.9 | 5.037 | 5.188 | 5603.5 | 18.2 |
| 4 | 4k | 1 | 4 | 424.6 | 427.3 | 4.852 | 5.096 | 5507.6 | 119.9 |
| 4 | 32k | 0 | 4 | 1226.1 | 1557.2 | 5.303 | 5.963 | 6797.0 | 145.5 |
| 4 | 32k | 1 | 4 | 1336.3 | 1561.0 | 5.226 | 5.884 | 6716.7 | 33.8 |
| 4 | 96k | 0 | 4 | 3749.4 | 5069.7 | 6.580 | 8.317 | 10553.1 | 72.8 |
| 4 | 96k | 1 | 4 | 3760.8 | 5066.3 | 6.560 | 8.300 | 10543.7 | 72.3 |
| 8 | 4k | 0 | 8 | 432.7 | 635.5 | 5.463 | 5.738 | 6245.0 | 224.0 |
| 8 | 4k | 1 | 8 | 426.9 | 655.1 | 5.509 | 5.783 | 6287.3 | 224.8 |
| 8 | 32k | 0 | 8 | 2008.9 | 3191.4 | 6.656 | 7.911 | 9092.6 | 275.1 |
| 8 | 32k | 1 | 8 | 2000.0 | 3182.0 | 6.559 | 7.812 | 8986.2 | 276.0 |
| 8 | 96k | 0 | 8 | 6242.0 | 10069.5 | 9.743 | 13.524 | 16544.2 | 334.5 |
| 8 | 96k | 1 | 8 | 6243.1 | 10071.0 | 9.776 | 13.554 | 16581.3 | 336.1 |
| 16 | 4k | 0 | 16 | 658.3 | 976.1 | 6.854 | 7.279 | 7925.8 | 251.9 |
| 16 | 32k | 0 | 16 | 3780.5 | 6457.1 | 9.430 | 11.965 | 13709.2 | 282.0 |
| 16 | 96k | 0 | 16 | 11784.3 | 20141.2 | 16.581 | 24.507 | 29184.9 | 438.4 |
| 24 | 4k | 0 | 24 | 867.9 | 1473.8 | 8.631 | 9.046 | 9968.9 | 263.1 |
| 24 | 32k | 0 | 24 | 5564.5 | 9679.0 | 12.904 | 16.782 | 19047.9 | 282.5 |
| 24 | 96k | 0 | 24 | 17371.1 | 30185.5 | 23.176 | 35.295 | 41438.6 | 359.7 |
| 32 | 4k | 0 | 32 | 1129.7 | 1786.9 | 9.639 | 10.286 | 11261.1 | 278.0 |
| 32 | 32k | 0 | 32 | 7353.2 | 12906.3 | 15.351 | 20.562 | 23340.8 | 283.6 |
| 32 | 96k | 0 | 32 | 22945.0 | 39984.8 | 29.407 | 45.680 | 53382.5 | 357.0 |
| 48 | 4k | 0 | 48 | 1542.7 | 2641.9 | 11.480 | 12.370 | 13614.7 | 320.9 |
| 48 | 32k | 0 | 48 | 10933.1 | 19371.1 | 20.217 | 28.086 | 31900.8 | 285.0 |
| 48 | 96k | 0 | 48 | 33873.9 | 59833.6 | 40.949 | 65.506 | 76346.8 | 579.7 |

## 5. Differences from the frozen workload plan

* arrival deviation (actual submit offset − planned arrival): p50 1.6 ms, p90 2.3 ms, max 3.1 ms over 450 requests.
* consecutive waves overlapping in time: 0 (plan: 0).
* cells whose observed pure-decode wave blocks differ from the plan: 0 [].
* waves with submit spread > 1 s: [].

## 6. Autotune tactic cache of this launch vs earlier launches

| A | B | common keys | differing | fraction |
|---|---|---|---|---|
| main_v3 | b8_recovery | 160 | 84 | 0.525 |
| main_v3 | nsys_val | 160 | 76 | 0.475 |
| main_v3 | on_v2 | 160 | 103 | 0.644 |
| main_v3 | off_v2 | 160 | 96 | 0.600 |
| main_v3 | off2_v2 | 160 | 95 | 0.594 |
| b8_recovery | nsys_val | 160 | 82 | 0.512 |
| b8_recovery | on_v2 | 160 | 94 | 0.588 |
| b8_recovery | off_v2 | 160 | 96 | 0.600 |
| b8_recovery | off2_v2 | 160 | 100 | 0.625 |
| nsys_val | on_v2 | 160 | 103 | 0.644 |
| nsys_val | off_v2 | 160 | 98 | 0.613 |
| nsys_val | off2_v2 | 160 | 103 | 0.644 |
| on_v2 | off_v2 | 160 | 88 | 0.550 |
| on_v2 | off2_v2 | 160 | 94 | 0.588 |
| off_v2 | off2_v2 | 160 | 93 | 0.581 |

## 7. Measurements that cannot be reconstructed without the Nsight trace

* per-rank kernel timings and the pure-decode iteration decomposition into kernel families (needs CUPTI kernel rows)
* f_ind (fraction of iteration time that is rank-independent) by b / KV / phase, and its frozen + block-bootstrap CIs
* four-rank NVTX / collective alignment, collective end spread, modal-K rank alignment per pure-decode cell
* critical-path ranks, wait vs intrinsic decomposition, straggler attribution
* autotune tactic-dependent GPU time per kernel family (tactic *choices* survive in the autotune cache; their GPU time does not)
* the K=163 decomposition proxy and every pre-registered contrast that is defined on kernel-level quantities
* JIT-affected iteration detection from kernel names (only host-side elapsed outliers can hint at it)
* routing-vs-timing correlations at kernel granularity (routing survives; per-iteration GPU time does not — only host-observed elapsed)
* the Stage 1.5 comparisons that require identical instrumentation (host-side elapsed here includes nsys capture overhead of unknown size)

No coefficient was fitted and no identification partition was written. Machine-readable results: `surviving.json`.
