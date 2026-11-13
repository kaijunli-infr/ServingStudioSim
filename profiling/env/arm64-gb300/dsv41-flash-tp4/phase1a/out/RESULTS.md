# Phase 1A calibration and held-out results
coefficients sha256 `78f937c500ff9f205db425a1573789d182df22cf67e8b84de39b22a7ae01c1c9`; partition sha256 `288e193f46323f4d4361913a08c47afa9c4d9f98cc4c4be0d5b88f9629304b7a`; git rev `cc06d9c26758e000f3ea9d66c8efff2eb76feffd`
Error metrics: n, median signed error (ms), median |error| (ms), p90 |error| (ms), p90 APE (%), bias (mean signed error, ms, and % of Σ measured), worst |error| (ms). APE is defined where the measured value is >= 0.05 ms.  Roles: calibration (fitted), heldout (never fitted), crosscheck (salvaged original capture, never fitted).
`alt_*` columns: the alternative composition — deterministic median path for decode (vs the stochastic expectation used as the prediction), standalone max(host, GPU) with lead = 0 for mixed iterations (vs the lead-aware prediction).

## Pass `diverse_nsys_run2` (17637 evaluated iterations, 49 excluded)

### Slowest-rank iteration latency by role and class
| key | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | bias_pct | worst_abs_err_ms | slowest_rank_match_pct | alt_median_err_ms | alt_p90_ape_pct | alt_bias_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calibration | chunk16k kv0=0k | 22 | -4.817 | 4.817 | 11.042 | 4.8 | -5.961 | -2.64 | 11.867 | — | -4.817 | 4.8 | -2.64 |
| calibration | chunk16k kv0=16k | 14 | 1.970 | 1.992 | 2.528 | 1.1 | 1.698 | +0.76 | 2.771 | — | 1.970 | 1.1 | +0.76 |
| calibration | chunk16k kv0=32k | 9 | 1.008 | 1.008 | 1.521 | 0.7 | 1.073 | +0.47 | 1.621 | — | 1.008 | 0.7 | +0.47 |
| calibration | chunk16k kv0=64k | 6 | 1.097 | 1.097 | 1.750 | 0.7 | 1.242 | +0.52 | 2.016 | — | 1.097 | 0.7 | +0.52 |
| calibration | chunk16k kv0=80k | 2 | 0.365 | 0.365 | 0.642 | 0.3 | 0.365 | +0.15 | 0.712 | — | 0.365 | 0.3 | +0.15 |
| calibration | chunk16k kv0=96k | 2 | 3.057 | 3.057 | 3.407 | 1.4 | 3.057 | +1.24 | 3.494 | — | 3.057 | 1.4 | +1.24 |
| calibration | decode b=1 | 620 | -0.078 | 0.312 | 0.480 | 11.0 | -0.143 | -3.07 | 19.834 | 1 | -0.151 | 10.8 | -4.64 |
| calibration | decode b=11 | 1282 | 0.069 | 0.073 | 0.171 | 2.7 | 0.043 | +0.66 | 5.272 | 0 | -0.133 | 3.4 | -2.45 |
| calibration | decode b=12 | 643 | 0.061 | 0.066 | 0.150 | 2.4 | 0.012 | +0.18 | 5.124 | 4 | -0.156 | 3.7 | -3.15 |
| calibration | decode b=13 | 279 | 0.069 | 0.072 | 0.140 | 2.2 | -0.034 | -0.52 | 5.394 | 3 | -0.126 | 3.2 | -3.52 |
| calibration | decode b=15 | 5 | 0.436 | 0.473 | 2.122 | 22.4 | -0.294 | -4.15 | 3.198 | 0 | -0.062 | 23.3 | -11.16 |
| calibration | decode b=16 | 13 | 0.063 | 0.063 | 0.090 | 1.4 | 0.050 | +0.79 | 0.133 | 0 | -0.063 | 2.2 | -1.21 |
| calibration | decode b=2 | 1266 | 0.006 | 0.187 | 0.604 | 12.0 | -0.136 | -2.99 | 31.114 | 0 | -0.249 | 17.1 | -8.59 |
| calibration | decode b=3 | 562 | 0.049 | 0.058 | 0.115 | 2.5 | -0.001 | -0.01 | 4.674 | 0 | -0.095 | 3.6 | -3.09 |
| calibration | decode b=4 | 1120 | 0.054 | 0.059 | 0.103 | 2.3 | -0.025 | -0.53 | 33.021 | 0 | -0.092 | 3.7 | -3.65 |
| calibration | decode b=6 | 1167 | 0.050 | 0.060 | 0.116 | 2.3 | 0.027 | +0.53 | 5.408 | 0 | -0.172 | 5.2 | -3.77 |
| calibration | decode b=7 | 1103 | 0.046 | 0.053 | 0.102 | 2.0 | 0.037 | +0.70 | 4.029 | 1 | -0.164 | 4.6 | -3.31 |
| calibration | decode b=8 | 1038 | 0.049 | 0.064 | 0.122 | 2.4 | 0.031 | +0.59 | 4.573 | 4 | -0.174 | 5.2 | -3.69 |
| calibration | decode b=9 | 1014 | 0.065 | 0.069 | 0.150 | 2.3 | 0.051 | +0.78 | 4.504 | 0 | -0.122 | 3.0 | -2.09 |
| calibration | mixed-eager | 14 | -6.252 | 6.252 | 32.738 | 18.0 | -11.195 | -5.85 | 43.399 | — | 0.209 | 17.8 | +4.05 |
| calibration | mixed-ring | 4 | -2.840 | 2.840 | 19.174 | 11.3 | -7.520 | -4.24 | 25.703 | — | 21.289 | 39.5 | +15.35 |
| calibration | mixed-ring multi-chunk | 6 | 1.878 | 3.473 | 4.325 | 1.9 | 1.331 | +0.57 | 4.480 | — | 1.878 | 1.9 | +0.57 |
| heldout | chunk16k kv0=0k | 4 | -8.605 | 8.605 | 9.894 | 4.3 | -6.447 | -2.84 | 10.067 | — | -8.605 | 4.3 | -2.84 |
| heldout | chunk16k kv0=112k | 1 | 0.706 | 0.706 | 0.706 | 0.3 | 0.706 | +0.28 | 0.706 | — | 0.706 | 0.3 | +0.28 |
| heldout | chunk16k kv0=16k | 4 | 2.159 | 2.159 | 2.576 | 1.2 | 1.998 | +0.90 | 2.577 | — | 2.159 | 1.2 | +0.90 |
| heldout | chunk16k kv0=32k | 4 | 1.236 | 1.236 | 1.642 | 0.7 | 1.178 | +0.52 | 1.778 | — | 1.236 | 0.7 | +0.52 |
| heldout | chunk16k kv0=48k | 11 | 1.854 | 1.854 | 2.397 | 1.0 | 1.669 | +0.72 | 2.584 | — | 1.854 | 1.0 | +0.72 |
| heldout | chunk16k kv0=64k | 4 | 1.303 | 1.303 | 1.647 | 0.7 | 1.369 | +0.58 | 1.766 | — | 1.303 | 0.7 | +0.58 |
| heldout | chunk16k kv0=80k | 3 | 0.594 | 0.594 | 0.887 | 0.4 | 0.658 | +0.27 | 0.960 | — | 0.594 | 0.4 | +0.27 |
| heldout | chunk16k kv0=96k | 2 | 3.645 | 3.645 | 3.692 | 1.5 | 3.645 | +1.49 | 3.703 | — | 3.645 | 1.5 | +1.49 |
| heldout | decode b=1 | 155 | -0.027 | 0.295 | 0.471 | 10.1 | -0.164 | -3.52 | 19.697 | 1 | -0.099 | 10.6 | -5.08 |
| heldout | decode b=10 | 1366 | 0.019 | 0.042 | 0.099 | 1.5 | -0.004 | -0.07 | 5.447 | 1 | -0.177 | 3.9 | -3.05 |
| heldout | decode b=11 | 1169 | 0.068 | 0.073 | 0.167 | 2.6 | 0.069 | +1.06 | 4.008 | 1 | -0.134 | 3.4 | -2.06 |
| heldout | decode b=12 | 633 | 0.059 | 0.064 | 0.150 | 2.3 | 0.022 | +0.34 | 4.691 | 3 | -0.157 | 3.7 | -3.00 |
| heldout | decode b=13 | 277 | 0.063 | 0.065 | 0.135 | 2.1 | 0.003 | +0.04 | 5.020 | 4 | -0.131 | 3.2 | -2.97 |
| heldout | decode b=14 | 204 | 0.273 | 0.279 | 0.371 | 5.9 | 0.117 | +1.80 | 5.349 | 0 | -0.072 | 2.5 | -3.49 |
| heldout | decode b=15 | 6 | 0.444 | 0.444 | 0.514 | 8.2 | 0.461 | +7.27 | 0.530 | 0 | -0.054 | 1.1 | -0.58 |
| heldout | decode b=16 | 13 | 0.040 | 0.049 | 3.500 | 32.9 | -0.632 | -9.04 | 4.458 | 0 | -0.087 | 33.8 | -10.85 |
| heldout | decode b=2 | 316 | 0.024 | 0.196 | 0.623 | 12.4 | -0.110 | -2.42 | 1.838 | 0 | -0.231 | 17.4 | -8.06 |
| heldout | decode b=3 | 141 | 0.045 | 0.053 | 0.111 | 2.4 | 0.002 | +0.04 | 1.662 | 0 | -0.099 | 3.4 | -3.03 |
| heldout | decode b=4 | 294 | 0.055 | 0.058 | 0.099 | 2.2 | -0.002 | -0.04 | 3.337 | 0 | -0.091 | 3.8 | -3.18 |
| heldout | decode b=5 | 662 | -0.093 | 0.093 | 0.194 | 3.8 | -0.109 | -2.18 | 4.662 | 0 | -0.278 | 7.4 | -5.87 |
| heldout | decode b=6 | 670 | 0.061 | 0.065 | 0.127 | 2.5 | 0.041 | +0.80 | 3.913 | 0 | -0.160 | 4.7 | -3.51 |
| heldout | decode b=7 | 514 | 0.047 | 0.051 | 0.115 | 2.2 | -0.001 | -0.01 | 14.447 | 1 | -0.163 | 4.3 | -3.99 |
| heldout | decode b=8 | 637 | 0.059 | 0.066 | 0.122 | 2.4 | 0.050 | +0.95 | 0.614 | 2 | -0.165 | 4.8 | -3.34 |
| heldout | decode b=9 | 349 | 0.070 | 0.071 | 0.144 | 2.2 | 0.054 | +0.83 | 4.624 | 1 | -0.117 | 3.0 | -2.04 |
| heldout | mixed-eager | 3 | -16.985 | 16.985 | 20.738 | 9.8 | -15.880 | -7.76 | 21.676 | — | -3.584 | 6.4 | -3.08 |
| heldout | mixed-ring | 1 | -4.902 | 4.902 | 4.902 | 2.4 | -4.902 | -2.37 | 4.902 | — | -1.498 | 0.7 | -0.73 |
| heldout | mixed-ring multi-chunk | 3 | -1.602 | 2.360 | 3.880 | 1.7 | -1.167 | -0.51 | 4.260 | — | -1.602 | 1.7 | -0.51 |

### Slowest-rank iteration latency by role and regime
| key | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | bias_pct | worst_abs_err_ms | slowest_rank_match_pct | alt_median_err_ms | alt_p90_ape_pct | alt_bias_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calibration | b<=16 | 10112 | 0.053 | 0.071 | 0.239 | 5.4 | -0.008 | -0.15 | 33.021 | 1 | -0.146 | 6.5 | -3.76 |
| calibration | mixed-eager | 14 | -6.252 | 6.252 | 32.738 | 18.0 | -11.195 | -5.85 | 43.399 | — | 0.209 | 17.8 | +4.05 |
| calibration | mixed-ring | 65 | 0.898 | 2.251 | 9.077 | 4.0 | -1.623 | -0.72 | 25.703 | — | 0.985 | 4.2 | +0.23 |
| heldout | b<=16 | 7406 | 0.044 | 0.066 | 0.182 | 3.3 | 0.007 | +0.12 | 19.697 | 1 | -0.159 | 5.4 | -3.38 |
| heldout | mixed-eager | 3 | -16.985 | 16.985 | 20.738 | 9.8 | -15.880 | -7.76 | 21.676 | — | -3.584 | 6.4 | -3.08 |
| heldout | mixed-ring | 37 | 1.236 | 1.766 | 4.516 | 2.1 | 0.333 | +0.14 | 10.067 | — | 1.236 | 1.7 | +0.18 |

### Decode latency by role, regime and KV band
| key | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | bias_pct | worst_abs_err_ms | slowest_rank_match_pct | alt_median_err_ms | alt_p90_ape_pct | alt_bias_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calibration | b<=16 | kv32-96k | 4287 | 0.051 | 0.059 | 0.114 | 2.1 | 0.015 | +0.28 | 33.021 | 1 | -0.141 | 4.6 | -3.25 |
| calibration | b<=16 | kv<32k | 2295 | 0.024 | 0.184 | 0.534 | 11.2 | -0.117 | -2.55 | 31.114 | 0 | -0.168 | 15.3 | -6.59 |
| calibration | b<=16 | kv>=96k | 3530 | 0.064 | 0.068 | 0.148 | 2.5 | 0.035 | +0.59 | 5.408 | 2 | -0.147 | 4.1 | -2.91 |
| heldout | b<=16 | kv32-96k | 1879 | 0.029 | 0.059 | 0.131 | 2.5 | -0.019 | -0.34 | 14.447 | 1 | -0.167 | 5.9 | -3.71 |
| heldout | b<=16 | kv<32k | 574 | 0.032 | 0.191 | 0.523 | 11.1 | -0.107 | -2.34 | 19.697 | 0 | -0.160 | 15.1 | -6.39 |
| heldout | b<=16 | kv>=96k | 4953 | 0.052 | 0.065 | 0.165 | 2.8 | 0.030 | +0.50 | 5.683 | 1 | -0.156 | 4.4 | -3.00 |

### Latency by prediction status (extrapolation / inferred cases reported separately)
| key | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | bias_pct | worst_abs_err_ms | slowest_rank_match_pct | alt_median_err_ms | alt_p90_ape_pct | alt_bias_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calibration | in-range | - | 34 | -5.001 | 5.001 | 11.758 | 5.1 | -7.573 | -3.50 | 43.399 | — | -4.225 | 5.0 | -2.16 |
| calibration | inferred | - | 45 | 1.216 | 1.595 | 3.839 | 1.7 | -0.106 | -0.05 | 33.988 | — | 1.496 | 1.9 | +3.02 |
| calibration | measured | in-range | 10112 | 0.053 | 0.071 | 0.239 | 5.4 | -0.008 | -0.15 | 33.021 | 1 | -0.146 | 6.5 | -3.76 |
| heldout | extrapolation | - | 4 | -8.140 | 10.286 | 20.269 | 9.7 | -8.592 | -3.78 | 21.676 | — | 0.826 | 5.8 | -1.21 |
| heldout | in-range | - | 5 | -8.979 | 8.979 | 9.837 | 4.4 | -8.232 | -3.75 | 10.067 | — | -7.720 | 4.3 | -2.95 |
| heldout | inferred | - | 31 | 1.370 | 1.602 | 2.577 | 1.2 | 1.297 | +0.56 | 4.260 | — | 1.370 | 1.2 | +0.56 |
| heldout | interpolated | in-range | 2227 | 0.001 | 0.060 | 0.208 | 3.8 | -0.023 | -0.38 | 5.447 | 0 | -0.194 | 6.3 | -3.75 |
| heldout | interpolated | kv-extrapolation | 5 | 0.218 | 0.268 | 2.531 | 24.5 | -0.612 | -8.46 | 4.035 | 0 | -0.126 | 25.5 | -13.23 |
| heldout | measured | in-range | 5174 | 0.059 | 0.068 | 0.173 | 3.0 | 0.021 | +0.36 | 19.697 | 1 | -0.147 | 4.7 | -3.20 |

### Kernel-group accuracy (slowest-rank group time)
| key | group | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| calibration | b<=16 | fmha | 10112 | -0.000 | 0.001 | 0.004 | 1.0 | 0.000 | 0.049 | 0.405 |
| calibration | b<=16 | indexer | 10112 | -0.008 | 0.010 | 0.033 | 8.4 | -0.012 | 0.128 | 0.352 |
| calibration | b<=16 | sparse_meta | 10112 | -0.002 | 0.002 | 0.013 | 3.2 | -0.004 | 0.104 | 0.390 |
| calibration | b<=16 | moe | 10112 | -0.013 | 0.028 | 0.073 | 3.8 | -0.014 | 0.223 | 1.987 |
| calibration | b<=16 | comm | 10112 | 0.030 | 0.048 | 0.198 | 18.2 | -0.036 | 32.897 | 0.767 |
| calibration | b<=16 | other | 10112 | -0.012 | 0.015 | 0.064 | 1.9 | -0.020 | 0.209 | 3.391 |
| calibration | mixed-eager | fmha | 14 | -0.361 | 0.423 | 0.827 | 30.7 | -0.197 | 1.269 | 3.537 |
| calibration | mixed-eager | indexer | 14 | -0.032 | 0.056 | 0.246 | 17.1 | -0.017 | 0.444 | 1.135 |
| calibration | mixed-eager | sparse_meta | 14 | 0.022 | 0.062 | 0.109 | 14.7 | -0.007 | 0.341 | 0.894 |
| calibration | mixed-eager | moe | 14 | -0.137 | 0.565 | 1.610 | 18.0 | 0.251 | 1.755 | 12.669 |
| calibration | mixed-eager | comm | 14 | -154.012 | 154.012 | 174.328 | 97.6 | -144.624 | 192.106 | 159.505 |
| calibration | mixed-eager | other | 14 | 0.210 | 0.383 | 1.184 | 19.0 | -0.045 | 5.220 | 10.622 |
| calibration | mixed-ring | fmha | 10 | 1.327 | 1.396 | 3.030 | 7.3 | 0.650 | 5.067 | 41.238 |
| calibration | mixed-ring | indexer | 10 | -0.289 | 0.491 | 1.260 | 10.4 | -0.389 | 1.694 | 11.752 |
| calibration | mixed-ring | sparse_meta | 10 | -0.130 | 0.130 | 0.317 | 5.4 | -0.164 | 0.378 | 8.504 |
| calibration | mixed-ring | moe | 10 | -0.355 | 0.681 | 2.096 | 6.7 | -0.738 | 2.655 | 48.058 |
| calibration | mixed-ring | comm | 10 | -0.310 | 0.898 | 64.088 | 72.6 | -17.551 | 71.606 | 40.122 |
| calibration | mixed-ring | other | 10 | -0.782 | 0.812 | 1.826 | 3.1 | -0.614 | 2.537 | 77.126 |
| calibration | mixed-ring | chunk16k | fmha | 55 | -0.022 | 0.320 | 0.549 | 1.4 | -0.043 | 0.960 | 40.547 |
| calibration | mixed-ring | chunk16k | indexer | 55 | -0.143 | 0.287 | 0.936 | 6.1 | -0.210 | 1.511 | 14.784 |
| calibration | mixed-ring | chunk16k | sparse_meta | 55 | -0.002 | 0.042 | 0.130 | 1.5 | -0.021 | 0.482 | 8.598 |
| calibration | mixed-ring | chunk16k | moe | 55 | -0.032 | 0.229 | 0.629 | 1.3 | 0.021 | 0.762 | 48.573 |
| calibration | mixed-ring | chunk16k | comm | 55 | -0.415 | 0.538 | 3.797 | 8.7 | -1.157 | 7.455 | 40.227 |
| calibration | mixed-ring | chunk16k | other | 55 | -0.113 | 0.391 | 1.206 | 1.5 | -0.208 | 2.650 | 77.493 |
| heldout | b<=16 | fmha | 7406 | -0.000 | 0.001 | 0.007 | 1.5 | -0.000 | 0.033 | 0.484 |
| heldout | b<=16 | indexer | 7406 | -0.017 | 0.017 | 0.043 | 11.1 | -0.018 | 0.111 | 0.389 |
| heldout | b<=16 | sparse_meta | 7406 | -0.003 | 0.003 | 0.015 | 3.8 | -0.005 | 0.055 | 0.398 |
| heldout | b<=16 | moe | 7406 | -0.024 | 0.040 | 0.112 | 4.4 | -0.028 | 0.246 | 2.848 |
| heldout | b<=16 | comm | 7406 | 0.033 | 0.046 | 0.108 | 13.9 | -0.006 | 19.867 | 0.773 |
| heldout | b<=16 | other | 7406 | -0.031 | 0.031 | 0.190 | 5.2 | -0.064 | 0.391 | 3.472 |
| heldout | mixed-eager | fmha | 3 | -0.524 | 0.524 | 0.541 | 44.1 | -0.390 | 0.546 | 1.234 |
| heldout | mixed-eager | indexer | 3 | -0.094 | 0.094 | 0.100 | 12.5 | -0.069 | 0.101 | 0.802 |
| heldout | mixed-eager | sparse_meta | 3 | 0.004 | 0.020 | 0.086 | 9.6 | 0.029 | 0.102 | 0.559 |
| heldout | mixed-eager | moe | 3 | 1.780 | 1.780 | 1.958 | 23.0 | 0.888 | 2.003 | 8.681 |
| heldout | mixed-eager | comm | 3 | -176.322 | 176.322 | 181.833 | 97.9 | -171.988 | 183.210 | 180.259 |
| heldout | mixed-eager | other | 3 | -15.746 | 15.746 | 18.127 | 72.7 | -11.436 | 18.722 | 22.523 |
| heldout | mixed-ring | fmha | 4 | -1.046 | 2.237 | 4.703 | 9.8 | -1.513 | 5.432 | 45.563 |
| heldout | mixed-ring | indexer | 4 | -0.869 | 0.869 | 1.455 | 10.9 | -0.705 | 1.595 | 13.023 |
| heldout | mixed-ring | sparse_meta | 4 | -0.073 | 0.082 | 0.153 | 1.8 | -0.072 | 0.153 | 8.504 |
| heldout | mixed-ring | moe | 4 | -0.078 | 0.245 | 1.355 | 4.1 | -0.399 | 1.787 | 48.547 |
| heldout | mixed-ring | comm | 4 | 0.157 | 0.548 | 52.026 | 53.6 | -18.236 | 74.004 | 39.656 |
| heldout | mixed-ring | other | 4 | -0.341 | 0.364 | 0.687 | 1.0 | -0.063 | 0.816 | 76.293 |
| heldout | mixed-ring | chunk16k | fmha | 33 | -0.129 | 0.235 | 0.592 | 1.5 | -0.134 | 0.992 | 40.485 |
| heldout | mixed-ring | chunk16k | indexer | 33 | -0.308 | 0.429 | 0.902 | 4.1 | -0.306 | 1.157 | 22.629 |
| heldout | mixed-ring | chunk16k | sparse_meta | 33 | -0.006 | 0.035 | 0.102 | 1.2 | -0.016 | 0.120 | 8.704 |
| heldout | mixed-ring | chunk16k | moe | 33 | 0.315 | 0.416 | 0.725 | 1.5 | 0.233 | 0.968 | 48.259 |
| heldout | mixed-ring | chunk16k | comm | 33 | -0.044 | 0.269 | 0.738 | 1.8 | -0.395 | 10.713 | 39.857 |
| heldout | mixed-ring | chunk16k | other | 33 | 0.254 | 0.547 | 1.390 | 1.7 | 0.252 | 2.788 | 78.942 |

Note: in host-bound eager iterations the measured `comm` group is dominated by twoshot kernels spinning while the host launches (host-induced wait, 150–190 ms); the model's `comm` is the intrinsic cost plus the GPU-bound wait term, so that row is not a modelling target. `comm` for decode = intrinsic + predicted synchronization wait.

### Per-rank GPU busy accuracy
| key | rank | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| calibration | b<=16 | 0 | 10112 | 0.044 | 0.064 | 0.170 | 3.6 | -0.002 | 29.636 | 5.034 |
| calibration | b<=16 | 1 | 10112 | 0.053 | 0.069 | 0.238 | 5.5 | 0.001 | 33.017 | 5.040 |
| calibration | b<=16 | 2 | 10112 | 0.056 | 0.072 | 0.237 | 5.9 | 0.007 | 33.004 | 5.037 |
| calibration | b<=16 | 3 | 10112 | 0.050 | 0.067 | 0.232 | 5.4 | 0.002 | 33.015 | 5.036 |
| calibration | mixed-eager | 0 | 14 | -147.112 | 147.112 | 172.321 | 87.2 | -136.801 | 188.355 | 187.333 |
| calibration | mixed-eager | 1 | 14 | -151.147 | 151.147 | 173.445 | 87.8 | -139.278 | 190.068 | 189.507 |
| calibration | mixed-eager | 2 | 14 | 2.026 | 2.544 | 5.984 | 18.3 | -0.031 | 23.772 | 33.200 |
| calibration | mixed-eager | 3 | 14 | -135.480 | 135.480 | 168.290 | 86.3 | -114.376 | 173.014 | 177.006 |
| calibration | mixed-ring | 0 | 65 | 1.840 | 2.093 | 4.170 | 1.8 | -1.473 | 70.062 | 221.851 |
| calibration | mixed-ring | 1 | 65 | 1.939 | 2.167 | 4.716 | 2.1 | -1.630 | 72.781 | 221.900 |
| calibration | mixed-ring | 2 | 65 | 1.631 | 1.951 | 3.148 | 1.4 | 0.866 | 42.072 | 221.759 |
| calibration | mixed-ring | 3 | 65 | 2.122 | 2.205 | 3.686 | 1.7 | -0.245 | 64.985 | 221.838 |
| heldout | b<=16 | 0 | 7406 | 0.039 | 0.062 | 0.167 | 3.1 | 0.008 | 20.287 | 6.187 |
| heldout | b<=16 | 1 | 7406 | 0.046 | 0.066 | 0.184 | 3.4 | 0.012 | 19.713 | 6.188 |
| heldout | b<=16 | 2 | 7406 | 0.043 | 0.064 | 0.171 | 3.1 | 0.022 | 5.918 | 6.186 |
| heldout | b<=16 | 3 | 7406 | 0.041 | 0.064 | 0.181 | 3.3 | 0.012 | 5.686 | 6.185 |
| heldout | mixed-eager | 0 | 3 | -155.684 | 155.684 | 174.948 | 87.2 | -109.869 | 179.763 | 191.894 |
| heldout | mixed-eager | 1 | 3 | -157.177 | 157.177 | 176.244 | 88.8 | -161.578 | 181.010 | 192.335 |
| heldout | mixed-eager | 2 | 3 | 1.294 | 3.134 | 140.568 | 75.1 | -56.833 | 174.926 | 32.891 |
| heldout | mixed-eager | 3 | 3 | -152.573 | 152.573 | 169.117 | 88.2 | -159.017 | 173.253 | 187.314 |
| heldout | mixed-ring | 0 | 37 | 2.240 | 2.358 | 3.770 | 1.6 | -0.249 | 73.884 | 231.009 |
| heldout | mixed-ring | 1 | 37 | 2.329 | 2.622 | 4.002 | 1.7 | -0.144 | 74.949 | 230.979 |
| heldout | mixed-ring | 2 | 37 | 2.210 | 2.331 | 3.370 | 1.5 | 1.989 | 5.095 | 230.982 |
| heldout | mixed-ring | 3 | 37 | 2.476 | 2.623 | 3.662 | 1.6 | 0.441 | 68.046 | 230.977 |

### Per-rank GPU idle accuracy
| key | rank | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| calibration | b<=16 | 0 | 10112 | 0.000 | 0.006 | 0.089 | 15.8 | -0.033 | 33.005 | 0.153 |
| calibration | b<=16 | 1 | 10112 | -0.000 | 0.005 | 0.015 | 9.6 | -0.010 | 29.111 | 0.149 |
| calibration | b<=16 | 2 | 10112 | -0.000 | 0.007 | 0.134 | 26.0 | -0.018 | 31.164 | 0.156 |
| calibration | b<=16 | 3 | 10112 | 0.000 | 0.005 | 0.015 | 9.4 | -0.013 | 30.958 | 0.152 |
| calibration | mixed-eager | 0 | 14 | -7.556 | 7.556 | 39.338 | 130.8 | -12.420 | 53.213 | 8.745 |
| calibration | mixed-eager | 1 | 14 | -4.843 | 4.843 | 30.240 | 121.7 | -11.031 | 59.928 | 5.965 |
| calibration | mixed-eager | 2 | 14 | -162.675 | 162.675 | 177.788 | 99.3 | -151.836 | 204.342 | 163.901 |
| calibration | mixed-eager | 3 | 14 | -8.928 | 8.928 | 105.635 | 135.7 | -34.969 | 183.527 | 10.157 |
| calibration | mixed-ring | 0 | 65 | -0.038 | 0.199 | 6.020 | 83.5 | -1.704 | 10.849 | 1.226 |
| calibration | mixed-ring | 1 | 65 | -0.011 | 0.138 | 5.409 | 82.7 | -1.501 | 8.714 | 1.133 |
| calibration | mixed-ring | 2 | 65 | -0.003 | 0.237 | 9.680 | 88.8 | -4.413 | 78.194 | 1.229 |
| calibration | mixed-ring | 3 | 65 | -0.002 | 0.225 | 7.576 | 86.0 | -2.624 | 44.506 | 1.231 |
| heldout | b<=16 | 0 | 7406 | 0.000 | 0.006 | 0.017 | 10.7 | -0.013 | 7.004 | 0.157 |
| heldout | b<=16 | 1 | 7406 | -0.000 | 0.006 | 0.016 | 9.9 | -0.005 | 5.609 | 0.156 |
| heldout | b<=16 | 2 | 7406 | 0.000 | 0.006 | 0.019 | 11.3 | -0.019 | 14.478 | 0.159 |
| heldout | b<=16 | 3 | 7406 | 0.000 | 0.006 | 0.015 | 9.2 | -0.008 | 20.252 | 0.158 |
| heldout | mixed-eager | 0 | 3 | -9.537 | 9.537 | 143.712 | 97.3 | -65.162 | 177.256 | 10.725 |
| heldout | mixed-eager | 1 | 3 | -9.532 | 9.532 | 23.491 | 94.7 | -14.943 | 26.981 | 10.654 |
| heldout | mixed-eager | 2 | 3 | -168.481 | 168.481 | 190.202 | 99.4 | -121.220 | 195.632 | 169.707 |
| heldout | mixed-eager | 3 | 3 | -13.239 | 13.239 | 32.180 | 137.6 | -16.473 | 36.915 | 14.468 |
| heldout | mixed-ring | 0 | 37 | 0.022 | 0.092 | 1.299 | 41.5 | -0.635 | 9.198 | 1.167 |
| heldout | mixed-ring | 1 | 37 | -0.046 | 0.085 | 0.283 | 20.1 | -0.549 | 8.726 | 1.168 |
| heldout | mixed-ring | 2 | 37 | 0.060 | 0.104 | 3.177 | 50.2 | -2.797 | 80.152 | 1.166 |
| heldout | mixed-ring | 3 | 37 | 0.059 | 0.092 | 3.060 | 49.1 | -0.823 | 8.959 | 1.170 |

### Decode KV-independence test (residual slope vs KV; ms per 100k tokens)
| label | role | regime | n | variable | slope_ms_per_100k | slope_se | x_range_100k | resid_std_ms |
|---|---|---|---|---|---|---|---|---|
| diverse_nsys_run2 | calibration | b<=16 | 10112 | kv_sum | +0.0315 | 0.0044 | (0.14197, 6.0579) | 0.635 |
| diverse_nsys_run2 | calibration | b<=16 | 10112 | kv_max | +0.1025 | 0.0139 | (0.12615, 1.31063) | 0.635 |
| diverse_nsys_run2 | heldout | b<=16 | 7406 | kv_sum | +0.0282 | 0.0040 | (0.142, 6.05806) | 0.433 |
| diverse_nsys_run2 | heldout | b<=16 | 7406 | kv_max | +0.0867 | 0.0129 | (0.12618, 1.31069) | 0.433 |

### Composition-rule check on held-out iterations (measured own segments → completion = max arrival + intrinsic)
- decode: n=400, median error +0.034 ms, p90 |error| 0.038 ms, worst 0.045 ms (median wall 6.38 ms; Σ over ranks / wall = 3.86)
- chunk: n=33, median error -0.019 ms, p90 |error| 0.078 ms, worst 0.120 ms (median wall 232.18 ms; Σ over ranks / wall = 3.97)

### Mixed-iteration host critical path (max over ranks of the forward window)
- role calibration, kind eager, measured bound **host**: n=14, forward-window median err -0.2 ms, p90 APE 5.0 %, bound predicted correctly 100 %, launch count median err +1, wall median err -6.3 ms (p90 APE 18.0 %), slowest host rank counts {'0': 1, '2': 12, '3': 1}
- role calibration, kind ring, measured bound **gpu**: n=60, forward-window median err +5.1 ms, p90 APE 10.8 %, bound predicted correctly 100 %, launch count median err +3, wall median err +1.0 ms (p90 APE 3.3 %), slowest host rank counts {'0': 7, '1': 6, '2': 35, '3': 12}
- role calibration, kind ring, measured bound **host**: n=5, forward-window median err +1.5 ms, p90 APE 6.2 %, bound predicted correctly 60 %, launch count median err -3, wall median err -3.9 ms (p90 APE 11.1 %), slowest host rank counts {'2': 4, '3': 1}
- role heldout, kind eager, measured bound **host**: n=3, forward-window median err -4.5 ms, p90 APE 6.7 %, bound predicted correctly 100 %, launch count median err -6, wall median err -17.0 ms (p90 APE 9.8 %), slowest host rank counts {'2': 3}
- role heldout, kind ring, measured bound **gpu**: n=35, forward-window median err -3.1 ms, p90 APE 11.7 %, bound predicted correctly 100 %, launch count median err +5, wall median err +1.2 ms (p90 APE 1.7 %), slowest host rank counts {'0': 5, '1': 14, '2': 13, '3': 3}
- role heldout, kind ring, measured bound **host**: n=2, forward-window median err -21.2 ms, p90 APE 15.3 %, bound predicted correctly 50 %, launch count median err +2, wall median err -0.6 ms (p90 APE 2.3 %), slowest host rank counts {'1': 1, '2': 1}

### Mixed iterations by kind: lead-aware vs standalone composition
- role calibration, kind eager, measured bound **host**, n=14 (previous stage {'decode': 10, 'mixed': 4}): lead-aware median err -6.3 ms, p90 APE 18.0 %, bias -5.8 %, worst 43.4 ms; standalone median err +0.2 ms, p90 APE 17.8 %, bias +4.0 %; host lead median 5.4 ms (max 108.1); all-reduce kind predicted correctly 100 %
- role calibration, kind ring, measured bound **gpu**, n=60 (previous stage {'decode': 18, 'mixed': 42}): lead-aware median err +1.0 ms, p90 APE 3.3 %, bias -0.5 %, worst 11.9 ms; standalone median err +1.0 ms, p90 APE 3.3 %, bias -0.5 %; host lead median 39.2 ms (max 129.6); all-reduce kind predicted correctly 100 %
- role calibration, kind ring, measured bound **host**, n=5 (previous stage {'mixed': 2, 'decode': 3}): lead-aware median err -3.9 ms, p90 APE 11.1 %, bias -4.3 %, worst 25.7 ms; standalone median err +5.9 ms, p90 APE 37.0 %, bias +10.4 %; host lead median 4.8 ms (max 90.6); all-reduce kind predicted correctly 100 %
- role heldout, kind eager, measured bound **host**, n=3 (previous stage {'decode': 2, 'mixed': 1}): lead-aware median err -17.0 ms, p90 APE 9.8 %, bias -7.8 %, worst 21.7 ms; standalone median err -3.6 ms, p90 APE 6.4 %, bias -3.1 %; host lead median 5.4 ms (max 17.9); all-reduce kind predicted correctly 100 %
- role heldout, kind ring, measured bound **gpu**, n=35 (previous stage {'mixed': 33, 'decode': 2}): lead-aware median err +1.2 ms, p90 APE 1.7 %, bias +0.2 %, worst 10.1 ms; standalone median err +1.2 ms, p90 APE 1.7 %, bias +0.2 %; host lead median 93.8 ms (max 127.2); all-reduce kind predicted correctly 100 %
- role heldout, kind ring, measured bound **host**, n=2 (previous stage {'mixed': 1, 'decode': 1}): lead-aware median err -0.6 ms, p90 APE 2.3 %, bias -0.3 %, worst 4.9 ms; standalone median err +1.1 ms, p90 APE 1.4 %, bias +0.5 %; host lead median 51.9 ms (max 100.4); all-reduce kind predicted correctly 100 %

### Full-workload prediction
measured GPU span 134.46 s (Σ iteration walls 134.58 s, 49 excluded JIT iterations = 10.09 s); predicted iterations cover 92.6 % of the span.
- all: n=17637, Σ measured 124.488 s, Σ predicted 124.162 s, error -0.26 %; by stage: decode 17518 it. 97.918/97.946 s, chunk16k 88 it. 20.190/20.253 s, mixed-ring 14 it. 2.962/2.993 s, mixed-eager 17 it. 3.091/3.296 s
- calibration: n=10191, Σ measured 72.128 s, Σ predicted 71.785 s, error -0.48 %; by stage: decode 10112 it. 54.709/54.790 s, chunk16k 55 it. 12.474/12.557 s, mixed-ring 10 it. 2.077/2.099 s, mixed-eager 14 it. 2.525/2.681 s
- heldout: n=7446, Σ measured 52.360 s, Σ predicted 52.377 s, error +0.03 %; by stage: decode 7406 it. 43.208/43.156 s, chunk16k 33 it. 7.716/7.695 s, mixed-ring 4 it. 0.885/0.894 s, mixed-eager 3 it. 0.567/0.614 s

### Prediction status counts
- calibration | in-range: 34
- calibration | inferred: 45
- calibration | measured: 10112
- heldout | extrapolation: 4
- heldout | in-range: 5
- heldout | inferred: 31
- heldout | interpolated: 2232
- heldout | measured: 5174

## Pass `quadrant_nsys` (4740 evaluated iterations, 10 excluded)

### Slowest-rank iteration latency by role and class
| key | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | bias_pct | worst_abs_err_ms | slowest_rank_match_pct | alt_median_err_ms | alt_p90_ape_pct | alt_bias_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calibration | decode b=1 | 206 | -0.292 | 0.292 | 0.542 | 10.8 | -0.285 | -5.96 | 1.618 | 1 | -0.365 | 12.2 | -7.49 |
| calibration | decode b=10 | 1 | 0.193 | 0.193 | 0.193 | 3.0 | 0.193 | +3.02 | 0.193 | 0 | -0.003 | 0.1 | -0.05 |
| calibration | decode b=12 | 103 | 0.166 | 0.166 | 0.241 | 3.8 | 0.166 | +2.61 | 0.282 | 17 | -0.050 | 2.1 | -0.79 |
| calibration | decode b=17 | 1 | 0.510 | 0.510 | 0.510 | 7.0 | 0.510 | +6.96 | 0.510 | 0 | 0.331 | 4.5 | +4.52 |
| calibration | decode b=19 | 1 | 0.265 | 0.265 | 0.265 | 3.5 | 0.265 | +3.50 | 0.265 | 0 | 0.086 | 1.1 | +1.13 |
| calibration | decode b=2 | 2 | 0.184 | 0.221 | 0.369 | 9.2 | 0.184 | +4.35 | 0.405 | 0 | -0.071 | 6.3 | -1.68 |
| calibration | decode b=20 | 1 | 0.257 | 0.257 | 0.257 | 3.4 | 0.257 | +3.40 | 0.257 | 0 | 0.078 | 1.0 | +1.03 |
| calibration | decode b=23 | 1 | 0.028 | 0.028 | 0.028 | 0.4 | 0.028 | +0.36 | 0.028 | 0 | -0.151 | 1.9 | -1.94 |
| calibration | decode b=24 | 190 | 0.110 | 0.111 | 0.203 | 2.7 | 0.095 | +1.23 | 0.288 | 53 | -0.069 | 2.6 | -1.08 |
| calibration | decode b=25 | 210 | 0.100 | 0.101 | 0.177 | 2.2 | 0.095 | +1.16 | 0.227 | 0 | -0.078 | 2.0 | -1.01 |
| calibration | decode b=27 | 103 | 0.112 | 0.135 | 0.249 | 3.0 | -0.208 | -2.41 | 9.975 | 63 | -0.076 | 3.3 | -4.59 |
| calibration | decode b=28 | 1 | 0.115 | 0.115 | 0.115 | 1.4 | 0.115 | +1.37 | 0.115 | 100 | -0.098 | 1.2 | -1.17 |
| calibration | decode b=29 | 1 | 0.176 | 0.176 | 0.176 | 2.1 | 0.176 | +2.07 | 0.176 | 100 | -0.064 | 0.8 | -0.75 |
| calibration | decode b=3 | 102 | 0.055 | 0.056 | 0.088 | 1.9 | 0.048 | +1.04 | 0.324 | 0 | -0.088 | 2.8 | -2.07 |
| calibration | decode b=32 | 2 | 0.441 | 0.441 | 0.458 | 5.4 | 0.441 | +5.17 | 0.462 | 100 | 0.127 | 1.7 | +1.48 |
| calibration | decode b=34 | 2 | -0.784 | 0.890 | 1.517 | 14.0 | -0.784 | -7.85 | 1.674 | 100 | -1.145 | 17.1 | -11.46 |
| calibration | decode b=39 | 2 | 0.169 | 0.169 | 0.208 | 2.2 | 0.169 | +1.76 | 0.218 | 50 | -0.303 | 3.6 | -3.16 |
| calibration | decode b=4 | 1 | -0.004 | 0.004 | 0.004 | 0.1 | -0.004 | -0.10 | 0.004 | 0 | -0.150 | 3.2 | -3.23 |
| calibration | decode b=40 | 1 | 0.650 | 0.650 | 0.650 | 7.0 | 0.650 | +7.04 | 0.650 | 100 | 0.155 | 1.7 | +1.68 |
| calibration | decode b=41 | 1 | 0.026 | 0.026 | 0.026 | 0.3 | 0.026 | +0.26 | 0.026 | 100 | -0.491 | 4.9 | -4.93 |
| calibration | decode b=42 | 1 | 0.168 | 0.168 | 0.168 | 1.7 | 0.168 | +1.69 | 0.168 | 100 | -0.371 | 3.7 | -3.74 |
| calibration | decode b=43 | 2 | 0.205 | 0.205 | 0.258 | 2.6 | 0.205 | +2.05 | 0.272 | 100 | -0.357 | 4.1 | -3.57 |
| calibration | decode b=44 | 2 | 0.358 | 0.358 | 0.374 | 3.8 | 0.358 | +3.59 | 0.378 | 100 | -0.226 | 2.4 | -2.27 |
| calibration | decode b=45 | 1 | 0.391 | 0.391 | 0.391 | 3.9 | 0.391 | +3.89 | 0.391 | 100 | -0.216 | 2.2 | -2.15 |
| calibration | decode b=46 | 3 | 0.546 | 0.546 | 0.562 | 5.6 | 0.522 | +5.20 | 0.566 | 67 | -0.084 | 1.6 | -1.08 |
| calibration | decode b=47 | 5 | 0.518 | 0.781 | 3.040 | 21.1 | -0.395 | -3.57 | 4.542 | 60 | -0.135 | 21.2 | -9.47 |
| calibration | decode b=48 | 2425 | 0.208 | 0.212 | 0.416 | 4.3 | 0.196 | +1.97 | 5.874 | 53 | -0.080 | 2.5 | -0.93 |
| calibration | decode b=6 | 103 | 0.141 | 0.141 | 0.183 | 3.7 | 0.142 | +2.82 | 0.228 | 0 | -0.080 | 2.4 | -1.58 |
| calibration | decode b=7 | 1 | 0.185 | 0.185 | 0.185 | 3.6 | 0.185 | +3.63 | 0.185 | 0 | -0.025 | 0.5 | -0.50 |
| calibration | mixed-eager | 1 | -3.382 | 3.382 | 3.382 | 1.7 | -3.382 | -1.72 | 3.382 | — | 4.213 | 2.1 | +2.15 |
| calibration | mixed-eager multi-chunk | 2 | -21.056 | 21.056 | 29.062 | 20.5 | -21.056 | -12.04 | 31.063 | — | 29.618 | 41.6 | +16.93 |
| calibration | mixed-graph | 4 | -0.597 | 2.060 | 2.925 | 8.3 | -0.299 | -0.82 | 2.925 | — | -0.597 | 8.3 | -0.82 |
| calibration | mixed-ring multi-chunk | 5 | 2.189 | 2.264 | 10.917 | 4.6 | -1.457 | -0.65 | 15.472 | — | 2.189 | 4.6 | -0.65 |
| heldout | decode b=1 | 52 | -0.339 | 0.339 | 0.550 | 10.9 | -0.398 | -8.12 | 5.268 | 0 | -0.412 | 12.3 | -9.61 |
| heldout | decode b=10 | 1 | 0.151 | 0.151 | 0.151 | 2.4 | 0.151 | +2.36 | 0.151 | 0 | -0.044 | 0.7 | -0.69 |
| heldout | decode b=12 | 24 | 0.188 | 0.188 | 0.240 | 3.8 | 0.174 | +2.75 | 0.286 | 17 | -0.028 | 1.4 | -0.67 |
| heldout | decode b=15 | 1 | 0.470 | 0.470 | 0.470 | 7.4 | 0.470 | +7.41 | 0.470 | 0 | -0.028 | 0.4 | -0.44 |
| heldout | decode b=2 | 1 | -0.141 | 0.141 | 0.141 | 3.1 | -0.141 | -3.08 | 0.141 | 0 | -0.396 | 8.7 | -8.68 |
| heldout | decode b=23 | 1 | -0.008 | 0.008 | 0.008 | 0.1 | -0.008 | -0.10 | 0.008 | 100 | -0.187 | 2.4 | -2.39 |
| heldout | decode b=24 | 48 | 0.102 | 0.109 | 0.185 | 2.4 | 0.092 | +1.19 | 0.251 | 65 | -0.077 | 2.6 | -1.12 |
| heldout | decode b=25 | 52 | 0.091 | 0.091 | 0.166 | 2.1 | 0.093 | +1.14 | 0.203 | 0 | -0.086 | 2.0 | -1.04 |
| heldout | decode b=26 | 129 | 0.078 | 0.084 | 0.185 | 2.3 | 0.022 | +0.27 | 6.356 | 52 | -0.108 | 2.7 | -1.98 |
| heldout | decode b=27 | 25 | 0.105 | 0.115 | 0.533 | 5.9 | -0.321 | -3.67 | 7.135 | 32 | -0.083 | 7.3 | -5.82 |
| heldout | decode b=29 | 1 | 0.151 | 0.151 | 0.151 | 1.8 | 0.151 | +1.77 | 0.151 | 100 | -0.089 | 1.0 | -1.04 |
| heldout | decode b=3 | 26 | 0.068 | 0.068 | 0.093 | 2.0 | 0.064 | +1.38 | 0.123 | 0 | -0.075 | 2.6 | -1.73 |
| heldout | decode b=30 | 127 | 0.279 | 0.279 | 0.356 | 4.2 | 0.281 | +3.32 | 0.465 | 59 | 0.014 | 1.3 | +0.19 |
| heldout | decode b=33 | 1 | -0.195 | 0.195 | 0.195 | 2.1 | -0.195 | -2.10 | 0.195 | 0 | -0.533 | 5.7 | -5.74 |
| heldout | decode b=34 | 1 | 0.108 | 0.108 | 0.108 | 1.2 | 0.108 | +1.19 | 0.108 | 100 | -0.253 | 2.8 | -2.78 |
| heldout | decode b=36 | 126 | 0.279 | 0.279 | 0.354 | 3.9 | 0.278 | +3.04 | 0.445 | 59 | -0.126 | 2.4 | -1.39 |
| heldout | decode b=37 | 1 | 0.194 | 0.194 | 0.194 | 2.1 | 0.194 | +2.08 | 0.194 | 0 | -0.233 | 2.5 | -2.50 |
| heldout | decode b=47 | 1 | -4.058 | 4.058 | 4.058 | 27.5 | -4.058 | -27.54 | 4.058 | 100 | -4.711 | 32.0 | -31.98 |
| heldout | decode b=48 | 607 | 0.214 | 0.215 | 0.409 | 4.2 | 0.217 | +2.19 | 0.623 | 53 | -0.074 | 2.5 | -0.72 |
| heldout | decode b=5 | 1 | -0.037 | 0.037 | 0.037 | 0.7 | -0.037 | -0.75 | 0.037 | 0 | -0.222 | 4.5 | -4.49 |
| heldout | decode b=6 | 24 | 0.140 | 0.140 | 0.170 | 3.4 | 0.143 | +2.83 | 0.212 | 0 | -0.082 | 2.1 | -1.57 |
| heldout | decode b=7 | 1 | 0.174 | 0.174 | 0.174 | 3.4 | 0.174 | +3.40 | 0.174 | 0 | -0.036 | 0.7 | -0.71 |
| heldout | mixed-eager multi-chunk | 1 | -3.161 | 3.161 | 3.161 | 1.6 | -3.161 | -1.58 | 3.161 | — | 3.960 | 2.0 | +1.97 |
| heldout | mixed-ring multi-chunk | 1 | 1.423 | 1.423 | 1.423 | 0.6 | 1.423 | +0.64 | 1.423 | — | 1.423 | 0.6 | +0.64 |

### Slowest-rank iteration latency by role and regime
| key | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | bias_pct | worst_abs_err_ms | slowest_rank_match_pct | alt_median_err_ms | alt_p90_ape_pct | alt_bias_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calibration | b17-24 | 194 | 0.111 | 0.111 | 0.206 | 2.7 | 0.099 | +1.28 | 0.510 | 52 | -0.068 | 2.7 | -1.04 |
| calibration | b<=16 | 519 | 0.063 | 0.149 | 0.438 | 8.9 | -0.041 | -0.81 | 1.618 | 4 | -0.097 | 10.3 | -3.66 |
| calibration | b>=25 | 2762 | 0.191 | 0.197 | 0.408 | 4.2 | 0.172 | +1.76 | 9.975 | 50 | -0.080 | 2.5 | -1.09 |
| calibration | mixed-eager | 3 | -11.048 | 11.048 | 27.060 | 18.8 | -15.165 | -8.33 | 31.063 | — | 4.213 | 37.2 | +11.62 |
| calibration | mixed-graph | 4 | -0.597 | 2.060 | 2.925 | 8.3 | -0.299 | -0.82 | 2.925 | — | -0.597 | 8.3 | -0.82 |
| calibration | mixed-ring | 5 | 2.189 | 2.264 | 10.917 | 4.6 | -1.457 | -0.65 | 15.472 | — | 2.189 | 4.6 | -0.65 |
| heldout | b17-24 | 49 | 0.098 | 0.106 | 0.183 | 2.4 | 0.090 | +1.17 | 0.251 | 65 | -0.081 | 2.6 | -1.14 |
| heldout | b<=16 | 131 | 0.073 | 0.148 | 0.428 | 8.6 | -0.082 | -1.60 | 5.268 | 3 | -0.087 | 10.1 | -4.47 |
| heldout | b>=25 | 1071 | 0.208 | 0.213 | 0.384 | 4.1 | 0.185 | +1.98 | 7.135 | 51 | -0.078 | 2.5 | -1.01 |
| heldout | mixed-eager | 1 | -3.161 | 3.161 | 3.161 | 1.6 | -3.161 | -1.58 | 3.161 | — | 3.960 | 2.0 | +1.97 |
| heldout | mixed-ring | 1 | 1.423 | 1.423 | 1.423 | 0.6 | 1.423 | +0.64 | 1.423 | — | 1.423 | 0.6 | +0.64 |

### Decode latency by role, regime and KV band
| key | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | bias_pct | worst_abs_err_ms | slowest_rank_match_pct | alt_median_err_ms | alt_p90_ape_pct | alt_bias_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calibration | b17-24 | kv<32k | 194 | 0.111 | 0.111 | 0.206 | 2.7 | 0.099 | +1.28 | 0.510 | 52 | -0.068 | 2.7 | -1.04 |
| calibration | b<=16 | kv<32k | 519 | 0.063 | 0.149 | 0.438 | 8.9 | -0.041 | -0.81 | 1.618 | 4 | -0.097 | 10.3 | -3.66 |
| calibration | b>=25 | kv<32k | 2762 | 0.191 | 0.197 | 0.408 | 4.2 | 0.172 | +1.76 | 9.975 | 50 | -0.080 | 2.5 | -1.09 |
| heldout | b17-24 | kv<32k | 49 | 0.098 | 0.106 | 0.183 | 2.4 | 0.090 | +1.17 | 0.251 | 65 | -0.081 | 2.6 | -1.14 |
| heldout | b<=16 | kv<32k | 131 | 0.073 | 0.148 | 0.428 | 8.6 | -0.082 | -1.60 | 5.268 | 3 | -0.087 | 10.1 | -4.47 |
| heldout | b>=25 | kv<32k | 1071 | 0.208 | 0.213 | 0.384 | 4.1 | 0.185 | +1.98 | 7.135 | 51 | -0.078 | 2.5 | -1.01 |

### Latency by prediction status (extrapolation / inferred cases reported separately)
| key | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | bias_pct | worst_abs_err_ms | slowest_rank_match_pct | alt_median_err_ms | alt_p90_ape_pct | alt_bias_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calibration | extrapolation | - | 1 | -1.195 | 1.195 | 1.195 | 3.2 | -1.195 | -3.20 | 1.195 | — | -1.195 | 3.2 | -3.20 |
| calibration | extrapolation | in-range | 4 | 0.261 | 0.261 | 0.436 | 5.9 | 0.265 | +3.50 | 0.510 | 0 | 0.082 | 3.7 | +1.14 |
| calibration | in-range | - | 4 | -7.215 | 7.215 | 14.145 | 6.1 | -7.563 | -3.48 | 15.472 | — | -2.748 | 5.3 | -1.93 |
| calibration | inferred | - | 4 | 2.226 | 3.175 | 22.970 | 16.1 | -5.631 | -2.81 | 31.063 | — | 3.175 | 32.7 | +9.09 |
| calibration | interpolated | in-range | 20 | 0.245 | 0.305 | 0.575 | 5.8 | 0.203 | +2.15 | 1.674 | 85 | -0.212 | 4.3 | -2.77 |
| calibration | measured | in-range | 3451 | 0.161 | 0.178 | 0.405 | 4.4 | 0.136 | +1.51 | 9.975 | 43 | -0.084 | 2.9 | -1.30 |
| calibration | measured-class | - | 3 | 0.000 | 2.925 | 2.925 | 8.5 | 0.000 | +0.00 | 2.925 | — | 0.000 | 8.5 | +0.00 |
| heldout | extrapolation | in-range | 1 | -0.008 | 0.008 | 0.008 | 0.1 | -0.008 | -0.10 | 0.008 | 100 | -0.187 | 2.4 | -2.39 |
| heldout | in-range | - | 1 | -3.161 | 3.161 | 3.161 | 1.6 | -3.161 | -1.58 | 3.161 | — | 3.960 | 2.0 | +1.97 |
| heldout | inferred | - | 1 | 1.423 | 1.423 | 1.423 | 0.6 | 1.423 | +0.64 | 1.423 | — | 1.423 | 0.6 | +0.64 |
| heldout | interpolated | in-range | 388 | 0.230 | 0.230 | 0.345 | 4.0 | 0.191 | +2.21 | 6.356 | 56 | -0.079 | 2.4 | -1.09 |
| heldout | measured | in-range | 862 | 0.158 | 0.172 | 0.404 | 4.5 | 0.137 | +1.53 | 7.135 | 43 | -0.081 | 2.8 | -1.29 |

### Kernel-group accuracy (slowest-rank group time)
| key | group | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| calibration | b17-24 | fmha | 194 | -0.000 | 0.001 | 0.002 | 0.4 | 0.000 | 0.019 | 0.539 |
| calibration | b17-24 | indexer | 194 | -0.001 | 0.001 | 0.010 | 3.4 | -0.003 | 0.019 | 0.276 |
| calibration | b17-24 | sparse_meta | 194 | -0.001 | 0.002 | 0.015 | 3.4 | -0.004 | 0.030 | 0.431 |
| calibration | b17-24 | moe | 194 | -0.008 | 0.049 | 0.114 | 3.2 | -0.011 | 0.322 | 3.489 |
| calibration | b17-24 | comm | 194 | 0.129 | 0.129 | 0.160 | 28.0 | 0.098 | 0.195 | 0.602 |
| calibration | b17-24 | other | 194 | -0.008 | 0.014 | 0.064 | 1.7 | -0.017 | 0.093 | 3.726 |
| calibration | b<=16 | fmha | 519 | 0.000 | 0.002 | 0.004 | 0.9 | 0.000 | 0.012 | 0.400 |
| calibration | b<=16 | indexer | 519 | 0.001 | 0.015 | 0.034 | 9.4 | 0.003 | 0.063 | 0.345 |
| calibration | b<=16 | sparse_meta | 519 | -0.001 | 0.004 | 0.009 | 2.4 | -0.001 | 0.030 | 0.382 |
| calibration | b<=16 | moe | 519 | -0.024 | 0.036 | 0.060 | 4.8 | -0.014 | 0.117 | 1.543 |
| calibration | b<=16 | comm | 519 | 0.026 | 0.079 | 0.472 | 27.6 | -0.101 | 1.652 | 0.761 |
| calibration | b<=16 | other | 519 | 0.001 | 0.017 | 0.057 | 1.7 | 0.005 | 0.111 | 3.385 |
| calibration | b>=25 | fmha | 2762 | -0.001 | 0.005 | 0.104 | 16.7 | 0.021 | 0.115 | 0.725 |
| calibration | b>=25 | indexer | 2762 | -0.001 | 0.003 | 0.007 | 2.4 | -0.001 | 0.026 | 0.291 |
| calibration | b>=25 | sparse_meta | 2762 | -0.002 | 0.002 | 0.015 | 3.9 | -0.005 | 0.129 | 0.386 |
| calibration | b>=25 | moe | 2762 | -0.009 | 0.116 | 0.242 | 4.7 | -0.011 | 0.491 | 5.094 |
| calibration | b>=25 | comm | 2762 | 0.207 | 0.209 | 0.265 | 38.8 | 0.166 | 9.982 | 0.726 |
| calibration | b>=25 | other | 2762 | -0.005 | 0.020 | 0.059 | 1.4 | -0.012 | 0.356 | 4.165 |
| calibration | mixed-eager | fmha | 3 | 0.299 | 0.299 | 0.541 | 2.8 | 0.255 | 0.601 | 19.589 |
| calibration | mixed-eager | indexer | 3 | 0.164 | 0.167 | 0.225 | 12.5 | 0.030 | 0.240 | 3.103 |
| calibration | mixed-eager | sparse_meta | 3 | -0.027 | 0.046 | 0.270 | 6.7 | -0.102 | 0.326 | 3.899 |
| calibration | mixed-eager | moe | 3 | 0.066 | 0.237 | 0.477 | 2.6 | -0.078 | 0.537 | 26.836 |
| calibration | mixed-eager | comm | 3 | -101.267 | 101.267 | 123.300 | 91.0 | -87.971 | 128.808 | 120.460 |
| calibration | mixed-eager | other | 3 | -0.138 | 0.260 | 0.415 | 1.3 | -0.110 | 0.453 | 37.048 |
| calibration | mixed-ring | fmha | 5 | -1.341 | 1.341 | 2.226 | 4.5 | -1.230 | 2.656 | 48.226 |
| calibration | mixed-ring | indexer | 5 | 0.268 | 0.316 | 0.731 | 9.0 | 0.133 | 0.795 | 7.285 |
| calibration | mixed-ring | sparse_meta | 5 | 0.037 | 0.037 | 0.072 | 0.9 | 0.037 | 0.081 | 8.188 |
| calibration | mixed-ring | moe | 5 | 0.288 | 0.288 | 0.673 | 1.4 | 0.396 | 0.734 | 48.482 |
| calibration | mixed-ring | comm | 5 | -0.280 | 0.357 | 4.880 | 10.4 | -1.494 | 7.632 | 40.092 |
| calibration | mixed-ring | other | 5 | 0.368 | 0.955 | 2.788 | 3.5 | -0.369 | 3.633 | 75.952 |
| heldout | b17-24 | fmha | 49 | -0.000 | 0.001 | 0.003 | 0.5 | -0.000 | 0.004 | 0.539 |
| heldout | b17-24 | indexer | 49 | -0.000 | 0.001 | 0.011 | 4.0 | -0.002 | 0.018 | 0.275 |
| heldout | b17-24 | sparse_meta | 49 | -0.002 | 0.002 | 0.017 | 3.9 | -0.005 | 0.023 | 0.432 |
| heldout | b17-24 | moe | 49 | -0.007 | 0.046 | 0.095 | 2.8 | -0.015 | 0.223 | 3.489 |
| heldout | b17-24 | comm | 49 | 0.130 | 0.130 | 0.174 | 31.2 | 0.102 | 0.200 | 0.601 |
| heldout | b17-24 | other | 49 | -0.008 | 0.012 | 0.062 | 1.6 | -0.019 | 0.090 | 3.726 |
| heldout | b<=16 | fmha | 131 | 0.000 | 0.002 | 0.004 | 0.9 | 0.001 | 0.051 | 0.400 |
| heldout | b<=16 | indexer | 131 | 0.002 | 0.017 | 0.036 | 10.2 | 0.003 | 0.060 | 0.345 |
| heldout | b<=16 | sparse_meta | 131 | -0.001 | 0.004 | 0.009 | 2.3 | -0.001 | 0.016 | 0.382 |
| heldout | b<=16 | moe | 131 | -0.022 | 0.034 | 0.056 | 5.3 | -0.013 | 0.144 | 1.536 |
| heldout | b<=16 | comm | 131 | 0.025 | 0.083 | 0.471 | 27.3 | -0.140 | 5.408 | 0.742 |
| heldout | b<=16 | other | 131 | 0.003 | 0.016 | 0.050 | 1.6 | 0.005 | 0.259 | 3.379 |
| heldout | b>=25 | fmha | 1071 | -0.003 | 0.006 | 0.097 | 15.5 | 0.008 | 0.114 | 0.698 |
| heldout | b>=25 | indexer | 1071 | -0.002 | 0.003 | 0.007 | 2.4 | -0.001 | 0.028 | 0.288 |
| heldout | b>=25 | sparse_meta | 1071 | -0.002 | 0.004 | 0.021 | 5.4 | -0.003 | 0.110 | 0.389 |
| heldout | b>=25 | moe | 1071 | 0.022 | 0.090 | 0.222 | 4.4 | 0.009 | 0.461 | 4.927 |
| heldout | b>=25 | comm | 1071 | 0.214 | 0.215 | 0.294 | 41.3 | 0.174 | 7.212 | 0.715 |
| heldout | b>=25 | other | 1071 | -0.007 | 0.030 | 0.102 | 2.5 | -0.008 | 0.311 | 4.141 |
| heldout | mixed-eager | fmha | 1 | 0.446 | 0.446 | 0.446 | 2.3 | 0.446 | 0.446 | 19.750 |
| heldout | mixed-eager | indexer | 1 | 0.148 | 0.148 | 0.148 | 4.7 | 0.148 | 0.148 | 3.149 |
| heldout | mixed-eager | sparse_meta | 1 | -0.094 | 0.094 | 0.094 | 2.4 | -0.094 | 0.094 | 3.977 |
| heldout | mixed-eager | moe | 1 | 0.268 | 0.268 | 0.268 | 1.0 | 0.268 | 0.268 | 26.871 |
| heldout | mixed-eager | comm | 1 | -92.783 | 92.783 | 92.783 | 82.9 | -92.783 | 92.783 | 111.974 |
| heldout | mixed-eager | other | 1 | 0.867 | 0.867 | 0.867 | 2.4 | 0.867 | 0.867 | 36.011 |
| heldout | mixed-ring | fmha | 1 | -2.054 | 2.054 | 2.054 | 4.2 | -2.054 | 2.054 | 48.893 |
| heldout | mixed-ring | indexer | 1 | -0.791 | 0.791 | 0.791 | 9.6 | -0.791 | 0.791 | 8.237 |
| heldout | mixed-ring | sparse_meta | 1 | 0.105 | 0.105 | 0.105 | 1.3 | 0.105 | 0.105 | 8.118 |
| heldout | mixed-ring | moe | 1 | 0.720 | 0.720 | 0.720 | 1.5 | 0.720 | 0.720 | 48.167 |
| heldout | mixed-ring | comm | 1 | 0.143 | 0.143 | 0.143 | 0.4 | 0.143 | 0.143 | 39.669 |
| heldout | mixed-ring | other | 1 | 0.915 | 0.915 | 0.915 | 1.2 | 0.915 | 0.915 | 75.498 |

Note: in host-bound eager iterations the measured `comm` group is dominated by twoshot kernels spinning while the host launches (host-induced wait, 150–190 ms); the model's `comm` is the intrinsic cost plus the GPU-bound wait term, so that row is not a modelling target. `comm` for decode = intrinsic + predicted synchronization wait.

### Per-rank GPU busy accuracy
| key | rank | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| calibration | b17-24 | 0 | 194 | 0.110 | 0.111 | 0.204 | 2.7 | 0.100 | 0.518 | 7.557 |
| calibration | b17-24 | 1 | 194 | 0.101 | 0.102 | 0.195 | 2.6 | 0.088 | 0.488 | 7.556 |
| calibration | b17-24 | 2 | 194 | 0.107 | 0.107 | 0.204 | 2.7 | 0.098 | 0.504 | 7.558 |
| calibration | b17-24 | 3 | 194 | 0.103 | 0.106 | 0.199 | 2.7 | 0.093 | 0.495 | 7.556 |
| calibration | b<=16 | 0 | 519 | 0.075 | 0.096 | 0.256 | 4.6 | 0.026 | 1.898 | 4.466 |
| calibration | b<=16 | 1 | 519 | 0.067 | 0.147 | 0.446 | 9.3 | -0.045 | 1.633 | 4.790 |
| calibration | b<=16 | 2 | 519 | 0.075 | 0.173 | 0.563 | 12.6 | -0.034 | 1.755 | 4.782 |
| calibration | b<=16 | 3 | 519 | 0.064 | 0.149 | 0.474 | 10.0 | -0.030 | 1.624 | 4.779 |
| calibration | b>=25 | 0 | 2762 | 0.205 | 0.209 | 0.420 | 4.4 | 0.192 | 9.960 | 9.739 |
| calibration | b>=25 | 1 | 2762 | 0.205 | 0.209 | 0.423 | 4.4 | 0.191 | 9.957 | 9.740 |
| calibration | b>=25 | 2 | 2762 | 0.199 | 0.204 | 0.411 | 4.3 | 0.207 | 0.925 | 9.735 |
| calibration | b>=25 | 3 | 2762 | 0.208 | 0.213 | 0.424 | 4.4 | 0.201 | 9.956 | 9.735 |
| calibration | mixed-eager | 0 | 3 | -97.376 | 97.376 | 118.134 | 64.0 | -83.611 | 123.323 | 180.886 |
| calibration | mixed-eager | 1 | 3 | -93.934 | 93.934 | 122.255 | 65.0 | -84.674 | 129.335 | 185.820 |
| calibration | mixed-eager | 2 | 3 | -6.049 | 6.049 | 24.829 | 19.0 | -10.047 | 29.523 | 103.645 |
| calibration | mixed-eager | 3 | 3 | -17.895 | 17.895 | 80.308 | 42.1 | -36.001 | 95.911 | 103.301 |
| calibration | mixed-graph | 0 | 4 | 1.633 | 5.967 | 8.668 | 48.9 | 0.817 | 8.668 | 31.391 |
| calibration | mixed-graph | 1 | 4 | 0.020 | 0.040 | 3.324 | 11.1 | 1.183 | 4.731 | 32.259 |
| calibration | mixed-graph | 2 | 4 | -3.178 | 6.357 | 12.976 | 49.4 | -3.953 | 15.813 | 21.153 |
| calibration | mixed-graph | 3 | 4 | 0.137 | 0.275 | 16.589 | 152.5 | 5.895 | 23.581 | 32.467 |
| calibration | mixed-ring | 0 | 5 | 1.825 | 1.927 | 5.398 | 2.4 | 0.501 | 6.500 | 220.272 |
| calibration | mixed-ring | 1 | 5 | 2.734 | 2.771 | 5.232 | 2.3 | 1.361 | 5.613 | 220.345 |
| calibration | mixed-ring | 2 | 5 | 2.118 | 2.345 | 4.956 | 2.2 | 0.698 | 6.002 | 219.979 |
| calibration | mixed-ring | 3 | 5 | 3.160 | 3.160 | 4.330 | 2.0 | 2.816 | 5.023 | 220.058 |
| heldout | b17-24 | 0 | 49 | 0.111 | 0.114 | 0.196 | 2.6 | 0.094 | 0.238 | 7.556 |
| heldout | b17-24 | 1 | 49 | 0.093 | 0.095 | 0.170 | 2.3 | 0.079 | 0.233 | 7.564 |
| heldout | b17-24 | 2 | 49 | 0.104 | 0.104 | 0.183 | 2.4 | 0.091 | 0.239 | 7.560 |
| heldout | b17-24 | 3 | 49 | 0.099 | 0.100 | 0.181 | 2.4 | 0.084 | 0.233 | 7.560 |
| heldout | b<=16 | 0 | 131 | 0.083 | 0.103 | 0.365 | 8.5 | -0.010 | 5.812 | 4.450 |
| heldout | b<=16 | 1 | 131 | 0.072 | 0.146 | 0.450 | 8.9 | -0.087 | 5.281 | 4.764 |
| heldout | b<=16 | 2 | 131 | 0.086 | 0.174 | 0.550 | 12.4 | -0.086 | 5.385 | 4.770 |
| heldout | b<=16 | 3 | 131 | 0.076 | 0.149 | 0.484 | 10.0 | -0.026 | 1.428 | 4.758 |
| heldout | b>=25 | 0 | 1071 | 0.225 | 0.227 | 0.397 | 4.4 | 0.203 | 7.112 | 9.579 |
| heldout | b>=25 | 1 | 1071 | 0.226 | 0.229 | 0.400 | 4.3 | 0.204 | 7.114 | 9.579 |
| heldout | b>=25 | 2 | 1071 | 0.162 | 0.164 | 0.379 | 4.0 | 0.172 | 6.367 | 9.572 |
| heldout | b>=25 | 3 | 1071 | 0.221 | 0.223 | 0.401 | 4.2 | 0.207 | 7.128 | 9.576 |
| heldout | mixed-eager | 0 | 1 | -84.068 | 84.068 | 84.068 | 43.5 | -84.068 | 84.068 | 193.169 |
| heldout | mixed-eager | 1 | 1 | -89.971 | 89.971 | 89.971 | 45.3 | -89.971 | 89.971 | 198.472 |
| heldout | mixed-eager | 2 | 1 | 4.655 | 4.655 | 4.655 | 4.5 | 4.655 | 4.655 | 104.399 |
| heldout | mixed-eager | 3 | 1 | -73.368 | 73.368 | 73.368 | 40.1 | -73.368 | 73.368 | 182.883 |
| heldout | mixed-ring | 0 | 1 | 1.089 | 1.089 | 1.089 | 0.5 | 1.089 | 1.089 | 220.667 |
| heldout | mixed-ring | 1 | 1 | 1.865 | 1.865 | 1.865 | 0.8 | 1.865 | 1.865 | 220.687 |
| heldout | mixed-ring | 2 | 1 | 1.548 | 1.548 | 1.548 | 0.7 | 1.548 | 1.548 | 220.668 |
| heldout | mixed-ring | 3 | 1 | 2.447 | 2.447 | 2.447 | 1.1 | 2.447 | 2.447 | 220.676 |

### Per-rank GPU idle accuracy
| key | rank | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| calibration | b17-24 | 0 | 194 | 0.000 | 0.004 | 0.027 | 14.5 | -0.004 | 0.048 | 0.159 |
| calibration | b17-24 | 1 | 194 | 0.000 | 0.004 | 0.009 | 6.0 | -0.002 | 0.043 | 0.158 |
| calibration | b17-24 | 2 | 194 | 0.000 | 0.003 | 0.008 | 5.1 | -0.001 | 0.040 | 0.162 |
| calibration | b17-24 | 3 | 194 | 0.000 | 0.003 | 0.008 | 5.1 | -0.001 | 0.041 | 0.163 |
| calibration | b<=16 | 0 | 519 | -0.002 | 0.010 | 0.601 | 40.5 | -0.009 | 1.505 | 0.157 |
| calibration | b<=16 | 1 | 519 | 0.001 | 0.005 | 0.013 | 8.5 | -0.001 | 0.355 | 0.147 |
| calibration | b<=16 | 2 | 519 | 0.001 | 0.007 | 0.102 | 37.3 | -0.056 | 1.938 | 0.152 |
| calibration | b<=16 | 3 | 519 | -0.002 | 0.006 | 0.016 | 9.6 | -0.017 | 1.099 | 0.153 |
| calibration | b>=25 | 0 | 2762 | 0.000 | 0.004 | 0.018 | 10.1 | -0.006 | 3.202 | 0.161 |
| calibration | b>=25 | 1 | 2762 | 0.000 | 0.004 | 0.010 | 6.1 | -0.004 | 3.196 | 0.160 |
| calibration | b>=25 | 2 | 2762 | 0.000 | 0.004 | 0.017 | 9.4 | -0.023 | 9.725 | 0.164 |
| calibration | b>=25 | 3 | 2762 | 0.000 | 0.004 | 0.010 | 6.3 | -0.012 | 6.993 | 0.164 |
| calibration | mixed-eager | 0 | 3 | -0.987 | 0.987 | 10.858 | 99.3 | -4.572 | 13.326 | 2.175 |
| calibration | mixed-eager | 1 | 3 | -5.099 | 5.099 | 7.782 | 92.1 | -4.337 | 8.453 | 6.221 |
| calibration | mixed-eager | 2 | 3 | -102.910 | 102.910 | 126.106 | 99.0 | -78.268 | 131.905 | 104.136 |
| calibration | mixed-eager | 3 | 3 | -35.630 | 35.630 | 102.064 | 98.5 | -52.098 | 118.672 | 36.859 |
| calibration | mixed-ring | 0 | 5 | 0.166 | 0.191 | 6.293 | 78.3 | -2.127 | 9.137 | 1.023 |
| calibration | mixed-ring | 1 | 5 | 0.103 | 0.134 | 6.262 | 79.1 | -2.148 | 9.086 | 1.020 |
| calibration | mixed-ring | 2 | 5 | 0.207 | 0.261 | 6.985 | 80.2 | -2.348 | 9.996 | 1.019 |
| calibration | mixed-ring | 3 | 5 | 0.222 | 0.303 | 8.842 | 80.6 | -2.936 | 13.273 | 1.007 |
| heldout | b17-24 | 0 | 49 | -0.001 | 0.004 | 0.033 | 17.0 | -0.006 | 0.046 | 0.161 |
| heldout | b17-24 | 1 | 49 | -0.002 | 0.004 | 0.009 | 5.6 | -0.002 | 0.029 | 0.160 |
| heldout | b17-24 | 2 | 49 | -0.001 | 0.004 | 0.010 | 6.4 | -0.003 | 0.039 | 0.163 |
| heldout | b17-24 | 3 | 49 | -0.001 | 0.003 | 0.008 | 4.7 | -0.000 | 0.010 | 0.164 |
| heldout | b<=16 | 0 | 131 | -0.003 | 0.011 | 0.699 | 44.2 | -0.016 | 1.206 | 0.160 |
| heldout | b<=16 | 1 | 131 | 0.001 | 0.005 | 0.014 | 9.0 | -0.000 | 0.045 | 0.146 |
| heldout | b<=16 | 2 | 131 | 0.001 | 0.006 | 0.021 | 15.8 | -0.048 | 0.762 | 0.152 |
| heldout | b<=16 | 3 | 131 | -0.001 | 0.005 | 0.017 | 10.6 | -0.049 | 4.167 | 0.152 |
| heldout | b>=25 | 0 | 1071 | 0.000 | 0.004 | 0.016 | 8.9 | -0.004 | 2.550 | 0.161 |
| heldout | b>=25 | 1 | 1071 | -0.000 | 0.004 | 0.009 | 6.1 | -0.003 | 1.659 | 0.161 |
| heldout | b>=25 | 2 | 1071 | 0.000 | 0.004 | 0.010 | 6.8 | -0.014 | 7.145 | 0.164 |
| heldout | b>=25 | 3 | 1071 | 0.001 | 0.004 | 0.012 | 7.2 | -0.007 | 6.313 | 0.164 |
| heldout | mixed-eager | 0 | 1 | -5.091 | 5.091 | 5.091 | 81.1 | -5.091 | 5.091 | 6.280 |
| heldout | mixed-eager | 1 | 1 | 0.154 | 0.154 | 0.154 | 15.9 | 0.154 | 0.154 | 0.969 |
| heldout | mixed-eager | 2 | 1 | -94.977 | 94.977 | 94.977 | 98.7 | -94.977 | 94.977 | 96.202 |
| heldout | mixed-eager | 3 | 1 | -15.335 | 15.335 | 15.335 | 92.6 | -15.335 | 15.335 | 16.564 |
| heldout | mixed-ring | 0 | 1 | 0.168 | 0.168 | 0.168 | 16.4 | 0.168 | 0.168 | 1.021 |
| heldout | mixed-ring | 1 | 1 | 0.115 | 0.115 | 0.115 | 11.4 | 0.115 | 0.115 | 1.008 |
| heldout | mixed-ring | 2 | 1 | 0.202 | 0.202 | 0.202 | 19.7 | 0.202 | 0.202 | 1.024 |
| heldout | mixed-ring | 3 | 1 | 0.211 | 0.211 | 0.211 | 20.7 | 0.211 | 0.211 | 1.018 |

### Decode KV-independence test (residual slope vs KV; ms per 100k tokens)
| label | role | regime | n | variable | slope_ms_per_100k | slope_se | x_range_100k | resid_std_ms |
|---|---|---|---|---|---|---|---|---|
| quadrant_nsys | calibration | b17-24 | 194 | kv_sum | -0.9170 | 0.2224 | (0.68783, 0.98235) | 0.096 |
| quadrant_nsys | calibration | b17-24 | 194 | kv_max | +9.1876 | 9.8965 | (0.03985, 0.04222) | 0.096 |
| quadrant_nsys | calibration | b<=16 | 519 | kv_sum | +0.9887 | 0.0546 | (0.03966, 0.49212) | 0.264 |
| quadrant_nsys | calibration | b<=16 | 519 | kv_max | +91.3026 | 17.6769 | (0.03966, 0.04222) | 0.264 |
| quadrant_nsys | calibration | b>=25 | 2762 | kv_sum | +0.2859 | 0.0193 | (0.60692, 2.1775) | 0.433 |
| quadrant_nsys | calibration | b>=25 | 2762 | kv_max | +6.9798 | 1.0860 | (0.03591, 0.06142) | 0.433 |
| quadrant_nsys | heldout | b17-24 | 49 | kv_sum | -0.6064 | 0.7501 | (0.92595, 0.98139) | 0.086 |
| quadrant_nsys | heldout | b17-24 | 49 | kv_max | -20.9390 | 17.4593 | (0.03987, 0.04221) | 0.086 |
| quadrant_nsys | heldout | b<=16 | 131 | kv_sum | +1.3063 | 0.2549 | (0.00128, 0.60354) | 0.532 |
| quadrant_nsys | heldout | b<=16 | 131 | kv_max | +129.5264 | 6.6633 | (0.00128, 0.04222) | 0.532 |
| quadrant_nsys | heldout | b>=25 | 1071 | kv_sum | +0.2777 | 0.0294 | (0.63566, 2.17798) | 0.367 |
| quadrant_nsys | heldout | b>=25 | 1071 | kv_max | +3.7201 | 1.3716 | (0.03724, 0.06142) | 0.367 |

### Composition-rule check on held-out iterations (measured own segments → completion = max arrival + intrinsic)
- decode: n=400, median error +0.000 ms, p90 |error| 0.003 ms, worst 0.041 ms (median wall 9.64 ms; Σ over ranks / wall = 3.95)

### Mixed-iteration host critical path (max over ranks of the forward window)
- role calibration, kind eager, measured bound **host**: n=3, forward-window median err +1.9 ms, p90 APE 1.1 %, bound predicted correctly 67 %, launch count median err +15, wall median err -11.0 ms (p90 APE 18.8 %), slowest host rank counts {'2': 3}
- role calibration, kind ring, measured bound **gpu**: n=5, forward-window median err +14.3 ms, p90 APE 10.1 %, bound predicted correctly 100 %, launch count median err -4, wall median err +2.2 ms (p90 APE 4.6 %), slowest host rank counts {'2': 2, '3': 3}
- role heldout, kind eager, measured bound **host**: n=1, forward-window median err +2.3 ms, p90 APE 1.1 %, bound predicted correctly 100 %, launch count median err +15, wall median err -3.2 ms (p90 APE 1.6 %), slowest host rank counts {'2': 1}
- role heldout, kind ring, measured bound **gpu**: n=1, forward-window median err +18.1 ms, p90 APE 9.1 %, bound predicted correctly 100 %, launch count median err +12, wall median err +1.4 ms (p90 APE 0.6 %), slowest host rank counts {'2': 1}

### Mixed iterations by kind: lead-aware vs standalone composition
- role calibration, kind eager, measured bound **host**, n=3 (previous stage {'mixed': 2, 'decode': 1}): lead-aware median err -11.0 ms, p90 APE 18.8 %, bias -8.3 %, worst 31.1 ms; standalone median err +4.2 ms, p90 APE 37.2 %, bias +11.6 %; host lead median 7.6 ms (max 127.7); all-reduce kind predicted correctly 100 %
- role calibration, kind graph, measured bound **n/a**, n=3 (previous stage {'mixed': 1, 'decode': 2}): lead-aware median err +0.0 ms, p90 APE 8.5 %, bias +0.0 %, worst 2.9 ms; standalone median err +0.0 ms, p90 APE 8.5 %, bias +0.0 %; host lead median 8.2 ms (max 12.9); all-reduce kind predicted correctly 100 %
- role calibration, kind ring, measured bound **gpu**, n=5 (previous stage {'decode': 2, 'mixed': 3}): lead-aware median err +2.2 ms, p90 APE 4.6 %, bias -0.6 %, worst 15.5 ms; standalone median err +2.2 ms, p90 APE 4.6 %, bias -0.6 %; host lead median 31.3 ms (max 79.7); all-reduce kind predicted correctly 100 %
- role heldout, kind eager, measured bound **host**, n=1 (previous stage {'decode': 1}): lead-aware median err -3.2 ms, p90 APE 1.6 %, bias -1.6 %, worst 3.2 ms; standalone median err +4.0 ms, p90 APE 2.0 %, bias +2.0 %; host lead median 7.1 ms (max 7.1); all-reduce kind predicted correctly 100 %
- role heldout, kind ring, measured bound **gpu**, n=1 (previous stage {'mixed': 1}): lead-aware median err +1.4 ms, p90 APE 0.6 %, bias +0.6 %, worst 1.4 ms; standalone median err +1.4 ms, p90 APE 0.6 %, bias +0.6 %; host lead median 105.3 ms (max 105.3); all-reduce kind predicted correctly 100 %

### Full-workload prediction
measured GPU span 49.84 s (Σ iteration walls 49.51 s, 10 excluded JIT iterations = 5.08 s); predicted iterations cover 89.1 % of the span.
- all: n=4740, Σ measured 44.427 s, Σ predicted 45.036 s, error +1.37 %; by stage: mixed-graph 4 it. 0.145/0.146 s, decode 4726 it. 42.852/42.188 s, mixed-ring 6 it. 1.341/1.346 s, mixed-eager 4 it. 0.698/0.747 s
- calibration: n=3487, Σ measured 32.930 s, Σ predicted 33.350 s, error +1.27 %; by stage: mixed-graph 4 it. 0.145/0.146 s, mixed-ring 5 it. 1.117/1.125 s, mixed-eager 3 it. 0.501/0.546 s, decode 3475 it. 31.586/31.113 s
- heldout: n=1253, Σ measured 11.497 s, Σ predicted 11.687 s, error +1.65 %; by stage: decode 1251 it. 11.266/11.074 s, mixed-ring 1 it. 0.223/0.222 s, mixed-eager 1 it. 0.197/0.201 s

### Prediction status counts
- calibration | extrapolation: 5
- calibration | in-range: 4
- calibration | inferred: 4
- calibration | interpolated: 20
- calibration | measured: 3451
- calibration | measured-class: 3
- heldout | extrapolation: 1
- heldout | in-range: 1
- heldout | inferred: 1
- heldout | interpolated: 388
- heldout | measured: 862

## Pass `diverse_nsys_salvaged` (4667 evaluated iterations, 20 excluded)

### Slowest-rank iteration latency by role and class
| key | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | bias_pct | worst_abs_err_ms | slowest_rank_match_pct | alt_median_err_ms | alt_p90_ape_pct | alt_bias_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| crosscheck | chunk16k kv0=0k | 14 | -7.629 | 7.629 | 10.214 | 4.4 | -5.518 | -2.45 | 11.295 | — | -7.629 | 4.4 | -2.45 |
| crosscheck | chunk16k kv0=112k | 1 | 0.678 | 0.678 | 0.678 | 0.3 | 0.678 | +0.27 | 0.678 | — | 0.678 | 0.3 | +0.27 |
| crosscheck | chunk16k kv0=16k | 10 | 1.811 | 1.811 | 2.275 | 1.0 | 1.853 | +0.83 | 2.425 | — | 1.811 | 1.0 | +0.83 |
| crosscheck | chunk16k kv0=32k | 9 | 1.019 | 1.019 | 1.110 | 0.5 | 0.865 | +0.38 | 1.221 | — | 1.019 | 0.5 | +0.38 |
| crosscheck | chunk16k kv0=48k | 8 | 1.860 | 1.860 | 2.177 | 0.9 | 1.742 | +0.75 | 2.375 | — | 1.860 | 0.9 | +0.75 |
| crosscheck | chunk16k kv0=64k | 8 | 1.298 | 1.298 | 1.738 | 0.7 | 1.311 | +0.55 | 1.850 | — | 1.298 | 0.7 | +0.55 |
| crosscheck | chunk16k kv0=80k | 4 | 0.537 | 0.537 | 0.939 | 0.4 | 0.542 | +0.22 | 1.015 | — | 0.537 | 0.4 | +0.22 |
| crosscheck | chunk16k kv0=96k | 4 | 3.204 | 3.204 | 3.489 | 1.4 | 3.153 | +1.28 | 3.532 | — | 3.204 | 1.4 | +1.28 |
| crosscheck | decode b=10 | 896 | 0.027 | 0.043 | 0.105 | 1.6 | 0.014 | +0.22 | 5.722 | 1 | -0.169 | 3.8 | -2.77 |
| crosscheck | decode b=11 | 1027 | 0.038 | 0.053 | 0.120 | 1.9 | 0.020 | +0.31 | 5.548 | 0 | -0.164 | 3.7 | -2.79 |
| crosscheck | decode b=12 | 436 | 0.019 | 0.039 | 0.098 | 1.5 | -0.004 | -0.07 | 5.067 | 2 | -0.198 | 4.2 | -3.39 |
| crosscheck | decode b=13 | 298 | -0.041 | 0.051 | 0.181 | 2.7 | -0.132 | -2.01 | 7.363 | 2 | -0.236 | 5.7 | -4.96 |
| crosscheck | decode b=14 | 243 | 0.178 | 0.179 | 0.258 | 4.1 | 0.114 | +1.75 | 5.810 | 1 | -0.167 | 3.9 | -3.54 |
| crosscheck | decode b=15 | 145 | 0.403 | 0.410 | 0.506 | 8.0 | 0.297 | +4.56 | 4.988 | 0 | -0.095 | 2.9 | -3.09 |
| crosscheck | decode b=16 | 173 | 0.035 | 0.042 | 0.117 | 1.9 | -0.019 | -0.30 | 4.846 | 1 | -0.092 | 2.6 | -2.28 |
| crosscheck | decode b=17 | 66 | 0.370 | 0.373 | 0.453 | 6.1 | 0.287 | +3.80 | 4.669 | 74 | 0.191 | 3.7 | +1.43 |
| crosscheck | decode b=4 | 164 | 0.057 | 0.068 | 0.114 | 2.4 | 0.034 | +0.75 | 0.634 | 0 | -0.089 | 4.5 | -2.42 |
| crosscheck | decode b=5 | 1 | -4.975 | 4.975 | 4.975 | 50.3 | -4.975 | -50.32 | 4.975 | 0 | -5.161 | 52.2 | -52.20 |
| crosscheck | decode b=6 | 108 | 0.039 | 0.049 | 0.105 | 2.1 | 0.016 | +0.31 | 1.023 | 0 | -0.183 | 4.9 | -3.98 |
| crosscheck | decode b=7 | 786 | 0.069 | 0.075 | 0.146 | 2.8 | 0.048 | +0.92 | 5.018 | 1 | -0.142 | 4.6 | -3.10 |
| crosscheck | decode b=8 | 105 | 0.077 | 0.081 | 0.165 | 3.3 | -0.046 | -0.87 | 5.418 | 0 | -0.146 | 3.7 | -5.09 |
| crosscheck | decode b=9 | 145 | 0.071 | 0.071 | 0.127 | 2.0 | 0.040 | +0.61 | 4.382 | 0 | -0.116 | 2.6 | -2.25 |
| crosscheck | mixed-eager | 2 | -17.735 | 17.735 | 19.002 | 8.9 | -17.735 | -8.38 | 19.319 | — | -13.395 | 6.9 | -6.33 |
| crosscheck | mixed-graph | 1 | 19.988 | 19.988 | 19.988 | 123.9 | 19.988 | +123.86 | 19.988 | — | 19.988 | 123.9 | +123.86 |
| crosscheck | mixed-ring | 1 | -51.892 | 51.892 | 51.892 | 26.1 | -51.892 | -26.11 | 51.892 | — | 5.753 | 2.9 | +2.89 |
| crosscheck | mixed-ring multi-chunk | 12 | -0.412 | 2.831 | 3.966 | 1.8 | -1.379 | -0.60 | 13.799 | — | -0.412 | 1.8 | -0.60 |

### Slowest-rank iteration latency by role and regime
| key | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | bias_pct | worst_abs_err_ms | slowest_rank_match_pct | alt_median_err_ms | alt_p90_ape_pct | alt_bias_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| crosscheck | b17-24 | 66 | 0.370 | 0.373 | 0.453 | 6.1 | 0.287 | +3.80 | 4.669 | 74 | 0.191 | 3.7 | +1.43 |
| crosscheck | b<=16 | 4527 | 0.043 | 0.059 | 0.158 | 2.8 | 0.022 | +0.36 | 7.363 | 1 | -0.161 | 4.2 | -3.14 |
| crosscheck | mixed-eager | 2 | -17.735 | 17.735 | 19.002 | 8.9 | -17.735 | -8.38 | 19.319 | — | -13.395 | 6.9 | -6.33 |
| crosscheck | mixed-graph | 1 | 19.988 | 19.988 | 19.988 | 123.9 | 19.988 | +123.86 | 19.988 | — | 19.988 | 123.9 | +123.86 |
| crosscheck | mixed-ring | 71 | 1.052 | 1.783 | 9.376 | 4.1 | -1.120 | -0.49 | 51.892 | — | 1.081 | 3.5 | -0.13 |

### Decode latency by role, regime and KV band
| key | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | bias_pct | worst_abs_err_ms | slowest_rank_match_pct | alt_median_err_ms | alt_p90_ape_pct | alt_bias_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| crosscheck | b17-24 | kv>=96k | 66 | 0.370 | 0.373 | 0.453 | 6.1 | 0.287 | +3.80 | 4.669 | 74 | 0.191 | 3.7 | +1.43 |
| crosscheck | b<=16 | kv32-96k | 824 | 0.033 | 0.062 | 0.157 | 2.7 | -0.014 | -0.24 | 5.418 | 0 | -0.186 | 4.9 | -4.08 |
| crosscheck | b<=16 | kv<32k | 165 | 0.057 | 0.068 | 0.114 | 2.5 | 0.004 | +0.09 | 4.975 | 0 | -0.089 | 4.6 | -3.06 |
| crosscheck | b<=16 | kv>=96k | 3538 | 0.044 | 0.058 | 0.161 | 2.8 | 0.031 | +0.50 | 7.363 | 1 | -0.158 | 3.9 | -2.94 |

### Latency by prediction status (extrapolation / inferred cases reported separately)
| key | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | bias_pct | worst_abs_err_ms | slowest_rank_match_pct | alt_median_err_ms | alt_p90_ape_pct | alt_bias_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| crosscheck | extrapolation | - | 3 | 2.672 | 2.672 | 16.525 | 99.3 | 7.779 | +4.52 | 19.988 | — | 2.672 | 99.3 | +4.52 |
| crosscheck | extrapolation | in-range | 66 | 0.370 | 0.373 | 0.453 | 6.1 | 0.287 | +3.80 | 4.669 | 74 | 0.191 | 3.7 | +1.43 |
| crosscheck | in-range | - | 13 | -9.616 | 9.616 | 15.681 | 7.3 | -10.106 | -4.47 | 19.319 | — | -9.616 | 5.8 | -4.17 |
| crosscheck | inferred | - | 58 | 1.294 | 1.406 | 3.424 | 1.4 | 0.225 | +0.10 | 51.892 | — | 1.298 | 1.4 | +0.53 |
| crosscheck | interpolated | in-range | 1140 | 0.043 | 0.059 | 0.189 | 2.9 | 0.031 | +0.48 | 5.810 | 1 | -0.169 | 3.9 | -3.00 |
| crosscheck | measured | in-range | 3387 | 0.044 | 0.059 | 0.148 | 2.7 | 0.019 | +0.32 | 7.363 | 1 | -0.158 | 4.2 | -3.18 |

### Kernel-group accuracy (slowest-rank group time)
| key | group | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| crosscheck | b17-24 | fmha | 66 | 0.020 | 0.020 | 0.022 | 4.3 | 0.020 | 0.023 | 0.518 |
| crosscheck | b17-24 | indexer | 66 | -0.054 | 0.054 | 0.055 | 16.7 | -0.054 | 0.056 | 0.329 |
| crosscheck | b17-24 | sparse_meta | 66 | 0.001 | 0.001 | 0.003 | 0.7 | 0.001 | 0.004 | 0.429 |
| crosscheck | b17-24 | moe | 66 | 0.243 | 0.243 | 0.306 | 9.6 | 0.231 | 0.354 | 3.239 |
| crosscheck | b17-24 | comm | 66 | 0.138 | 0.138 | 0.179 | 32.5 | 0.061 | 4.746 | 0.594 |
| crosscheck | b17-24 | other | 66 | 0.054 | 0.054 | 0.064 | 1.8 | 0.052 | 0.069 | 3.664 |
| crosscheck | b<=16 | fmha | 4527 | 0.001 | 0.002 | 0.018 | 3.7 | -0.001 | 0.075 | 0.498 |
| crosscheck | b<=16 | indexer | 4527 | -0.018 | 0.019 | 0.042 | 10.6 | -0.018 | 0.076 | 0.401 |
| crosscheck | b<=16 | sparse_meta | 4527 | -0.013 | 0.013 | 0.025 | 6.0 | -0.013 | 0.127 | 0.412 |
| crosscheck | b<=16 | moe | 4527 | -0.016 | 0.042 | 0.101 | 3.9 | -0.018 | 0.251 | 2.943 |
| crosscheck | b<=16 | comm | 4527 | 0.043 | 0.049 | 0.123 | 15.5 | 0.026 | 7.180 | 0.778 |
| crosscheck | b<=16 | other | 4527 | -0.037 | 0.038 | 0.161 | 4.4 | -0.060 | 0.331 | 3.462 |
| crosscheck | mixed-eager | fmha | 2 | -0.464 | 0.464 | 0.474 | 31.3 | -0.464 | 0.476 | 1.516 |
| crosscheck | mixed-eager | indexer | 2 | -0.002 | 0.007 | 0.008 | 1.0 | -0.002 | 0.009 | 0.810 |
| crosscheck | mixed-eager | sparse_meta | 2 | 0.059 | 0.059 | 0.062 | 11.9 | 0.059 | 0.063 | 0.525 |
| crosscheck | mixed-eager | moe | 2 | 1.562 | 1.562 | 1.884 | 21.6 | 1.562 | 1.965 | 9.112 |
| crosscheck | mixed-eager | comm | 2 | -180.565 | 180.565 | 184.565 | 97.8 | -180.565 | 185.565 | 184.735 |
| crosscheck | mixed-eager | other | 2 | 1.173 | 1.173 | 1.205 | 19.8 | 1.173 | 1.213 | 6.083 |
| crosscheck | mixed-ring | fmha | 13 | -1.646 | 2.391 | 5.363 | 11.4 | -1.760 | 5.518 | 45.653 |
| crosscheck | mixed-ring | indexer | 13 | -0.663 | 0.663 | 1.462 | 7.6 | -0.692 | 1.775 | 11.740 |
| crosscheck | mixed-ring | sparse_meta | 13 | -0.060 | 0.060 | 0.182 | 2.5 | -0.073 | 0.240 | 8.516 |
| crosscheck | mixed-ring | moe | 13 | 0.083 | 0.175 | 0.883 | 1.8 | 0.111 | 2.065 | 48.343 |
| crosscheck | mixed-ring | comm | 13 | 0.105 | 0.278 | 6.004 | 12.8 | -6.136 | 72.730 | 39.708 |
| crosscheck | mixed-ring | other | 13 | -0.209 | 0.454 | 1.343 | 2.2 | -0.380 | 1.973 | 77.304 |
| crosscheck | mixed-ring | chunk16k | fmha | 58 | -0.086 | 0.301 | 0.700 | 1.7 | -0.130 | 1.050 | 40.519 |
| crosscheck | mixed-ring | chunk16k | indexer | 58 | -0.171 | 0.397 | 1.221 | 6.4 | -0.281 | 2.103 | 19.379 |
| crosscheck | mixed-ring | chunk16k | sparse_meta | 58 | -0.021 | 0.043 | 0.228 | 2.6 | -0.053 | 0.322 | 8.691 |
| crosscheck | mixed-ring | chunk16k | moe | 58 | 0.086 | 0.291 | 0.664 | 1.4 | 0.173 | 1.234 | 48.457 |
| crosscheck | mixed-ring | chunk16k | comm | 58 | -0.236 | 0.373 | 2.135 | 5.1 | -0.788 | 12.655 | 40.049 |
| crosscheck | mixed-ring | chunk16k | other | 58 | 0.069 | 0.550 | 1.615 | 2.0 | -0.064 | 5.298 | 78.741 |

Note: in host-bound eager iterations the measured `comm` group is dominated by twoshot kernels spinning while the host launches (host-induced wait, 150–190 ms); the model's `comm` is the intrinsic cost plus the GPU-bound wait term, so that row is not a modelling target. `comm` for decode = intrinsic + predicted synchronization wait.

### Per-rank GPU busy accuracy
| key | rank | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| crosscheck | b17-24 | 0 | 66 | 0.364 | 0.372 | 0.442 | 6.1 | 0.284 | 4.600 | 7.303 |
| crosscheck | b17-24 | 1 | 66 | 0.355 | 0.356 | 0.439 | 6.1 | 0.274 | 4.608 | 7.302 |
| crosscheck | b17-24 | 2 | 66 | 0.361 | 0.367 | 0.444 | 6.1 | 0.346 | 0.473 | 7.303 |
| crosscheck | b17-24 | 3 | 66 | 0.358 | 0.358 | 0.439 | 6.1 | 0.354 | 0.465 | 7.300 |
| crosscheck | b<=16 | 0 | 4527 | 0.039 | 0.057 | 0.153 | 2.7 | 0.022 | 5.828 | 6.305 |
| crosscheck | b<=16 | 1 | 4527 | 0.044 | 0.060 | 0.155 | 2.7 | 0.025 | 7.369 | 6.305 |
| crosscheck | b<=16 | 2 | 4527 | 0.033 | 0.054 | 0.122 | 2.1 | 0.013 | 7.358 | 6.301 |
| crosscheck | b<=16 | 3 | 4527 | 0.042 | 0.057 | 0.152 | 2.7 | 0.040 | 7.343 | 6.301 |
| crosscheck | mixed-eager | 0 | 2 | -176.839 | 176.839 | 180.862 | 88.2 | -176.839 | 181.867 | 201.200 |
| crosscheck | mixed-eager | 1 | 2 | -177.989 | 177.989 | 181.380 | 88.7 | -177.989 | 182.228 | 201.214 |
| crosscheck | mixed-eager | 2 | 2 | -163.931 | 163.931 | 164.977 | 88.3 | -163.931 | 165.239 | 185.937 |
| crosscheck | mixed-eager | 3 | 2 | 5.574 | 5.574 | 5.834 | 32.0 | 5.574 | 5.898 | 18.566 |
| crosscheck | mixed-graph | 0 | 1 | 18.398 | 18.398 | 18.398 | 119.2 | 18.398 | 18.398 | 15.439 |
| crosscheck | mixed-graph | 1 | 1 | 19.179 | 19.179 | 19.179 | 124.3 | 19.179 | 19.179 | 15.434 |
| crosscheck | mixed-graph | 2 | 1 | -0.871 | 0.871 | 0.871 | 5.7 | -0.871 | 0.871 | 15.347 |
| crosscheck | mixed-graph | 3 | 1 | 19.092 | 19.092 | 19.092 | 124.5 | 19.092 | 19.092 | 15.329 |
| crosscheck | mixed-ring | 0 | 71 | 1.940 | 2.153 | 3.341 | 1.5 | 0.019 | 74.329 | 227.385 |
| crosscheck | mixed-ring | 1 | 71 | 2.093 | 2.338 | 3.703 | 1.7 | 0.282 | 74.490 | 227.397 |
| crosscheck | mixed-ring | 2 | 71 | 2.003 | 2.192 | 4.099 | 1.7 | 0.152 | 75.788 | 227.313 |
| crosscheck | mixed-ring | 3 | 71 | 2.393 | 2.471 | 3.537 | 1.5 | 1.965 | 6.268 | 227.294 |

### Per-rank GPU idle accuracy
| key | rank | n | median_err_ms | median_abs_err_ms | p90_abs_err_ms | p90_ape_pct | bias_ms | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| crosscheck | b17-24 | 0 | 66 | 0.001 | 0.004 | 0.023 | 12.7 | -0.002 | 0.075 | 0.158 |
| crosscheck | b17-24 | 1 | 66 | 0.001 | 0.004 | 0.009 | 5.8 | -0.002 | 0.077 | 0.158 |
| crosscheck | b17-24 | 2 | 66 | 0.002 | 0.003 | 0.010 | 6.3 | -0.048 | 3.270 | 0.160 |
| crosscheck | b17-24 | 3 | 66 | -0.000 | 0.005 | 0.011 | 7.5 | -0.054 | 3.510 | 0.163 |
| crosscheck | b<=16 | 0 | 4527 | 0.002 | 0.006 | 0.015 | 9.8 | -0.004 | 7.162 | 0.159 |
| crosscheck | b<=16 | 1 | 4527 | 0.001 | 0.006 | 0.015 | 9.5 | -0.003 | 3.776 | 0.159 |
| crosscheck | b<=16 | 2 | 4527 | 0.000 | 0.006 | 0.017 | 10.2 | -0.013 | 4.196 | 0.162 |
| crosscheck | b<=16 | 3 | 4527 | 0.000 | 0.006 | 0.015 | 9.5 | -0.016 | 4.267 | 0.162 |
| crosscheck | mixed-eager | 0 | 2 | -8.397 | 8.397 | 11.123 | 89.8 | -8.397 | 11.804 | 9.586 |
| crosscheck | mixed-eager | 1 | 2 | -8.450 | 8.450 | 10.546 | 90.1 | -8.450 | 11.069 | 9.573 |
| crosscheck | mixed-eager | 2 | 2 | -23.623 | 23.623 | 23.892 | 95.1 | -23.623 | 23.959 | 24.849 |
| crosscheck | mixed-eager | 3 | 2 | -191.871 | 191.871 | 193.463 | 99.4 | -191.871 | 193.861 | 193.100 |
| crosscheck | mixed-ring | 0 | 71 | 0.038 | 0.150 | 2.692 | 69.4 | -0.681 | 8.777 | 1.150 |
| crosscheck | mixed-ring | 1 | 71 | -0.023 | 0.111 | 2.757 | 71.1 | -0.796 | 10.146 | 1.146 |
| crosscheck | mixed-ring | 2 | 71 | 0.081 | 0.169 | 3.914 | 76.1 | -0.864 | 11.483 | 1.145 |
| crosscheck | mixed-ring | 3 | 71 | 0.067 | 0.168 | 6.294 | 83.7 | -2.156 | 75.482 | 1.162 |

### Decode KV-independence test (residual slope vs KV; ms per 100k tokens)
| label | role | regime | n | variable | slope_ms_per_100k | slope_se | x_range_100k | resid_std_ms |
|---|---|---|---|---|---|---|---|---|
| diverse_nsys_salvaged | crosscheck | b17-24 | 66 | kv_sum | +50.0958 | 23.0421 | (4.29788, 4.30893) | 0.619 |
| diverse_nsys_salvaged | crosscheck | b17-24 | 66 | kv_max | +851.6290 | 391.7156 | (1.22906, 1.22971) | 0.619 |
| diverse_nsys_salvaged | crosscheck | b<=16 | 4527 | kv_sum | +0.0117 | 0.0069 | (0.61473, 5.12963) | 0.372 |
| diverse_nsys_salvaged | crosscheck | b<=16 | 4527 | kv_max | +0.0648 | 0.0254 | (0.31965, 1.26446) | 0.372 |

### Composition-rule check on held-out iterations (measured own segments → completion = max arrival + intrinsic)
- decode: n=400, median error +0.035 ms, p90 |error| 0.041 ms, worst 0.046 ms (median wall 6.47 ms; Σ over ranks / wall = 3.87)
- chunk: n=58, median error -0.038 ms, p90 |error| 0.081 ms, worst 0.106 ms (median wall 229.61 ms; Σ over ranks / wall = 3.97)

### Mixed-iteration host critical path (max over ranks of the forward window)
- role crosscheck, kind eager, measured bound **host**: n=2, forward-window median err -13.0 ms, p90 APE 6.7 %, bound predicted correctly 100 %, launch count median err -5, wall median err -17.7 ms (p90 APE 8.9 %), slowest host rank counts {'3': 2}
- role crosscheck, kind ring, measured bound **gpu**: n=70, forward-window median err +0.0 ms, p90 APE 11.6 %, bound predicted correctly 100 %, launch count median err +3, wall median err +1.1 ms (p90 APE 3.5 %), slowest host rank counts {'0': 20, '1': 8, '2': 16, '3': 26}
- role crosscheck, kind ring, measured bound **host**: n=1, forward-window median err -2.7 ms, p90 APE 1.3 %, bound predicted correctly 100 %, launch count median err -4, wall median err -51.9 ms (p90 APE 26.1 %), slowest host rank counts {'3': 1}

### Mixed iterations by kind: lead-aware vs standalone composition
- role crosscheck, kind eager, measured bound **host**, n=2 (previous stage {'decode': 2}): lead-aware median err -17.7 ms, p90 APE 8.9 %, bias -8.4 %, worst 19.3 ms; standalone median err -13.4 ms, p90 APE 6.9 %, bias -6.3 %; host lead median 4.3 ms (max 4.3); all-reduce kind predicted correctly 100 %
- role crosscheck, kind graph, measured bound **n/a**, n=1 (previous stage {'mixed': 1}): lead-aware median err +20.0 ms, p90 APE 123.9 %, bias +123.9 %, worst 20.0 ms; standalone median err +20.0 ms, p90 APE 123.9 %, bias +123.9 %; host lead median 109.4 ms (max 109.4); all-reduce kind predicted correctly 100 %
- role crosscheck, kind ring, measured bound **gpu**, n=70 (previous stage {'decode': 9, 'mixed': 61}): lead-aware median err +1.1 ms, p90 APE 3.5 %, bias -0.2 %, worst 13.8 ms; standalone median err +1.1 ms, p90 APE 3.5 %, bias -0.2 %; host lead median 96.0 ms (max 132.7); all-reduce kind predicted correctly 100 %
- role crosscheck, kind ring, measured bound **host**, n=1 (previous stage {'mixed': 1}): lead-aware median err -51.9 ms, p90 APE 26.1 %, bias -26.1 %, worst 51.9 ms; standalone median err +5.8 ms, p90 APE 2.9 %, bias +2.9 %; host lead median 57.6 ms (max 57.6); all-reduce kind predicted correctly 100 %

### Full-workload prediction
measured GPU span 62.18 s (Σ iteration walls 62.20 s, 20 excluded JIT iterations = 16.95 s); predicted iterations cover 72.8 % of the span.
- all: n=4667, Σ measured 45.253 s, Σ predicted 45.277 s, error +0.05 %; by stage: decode 4593 it. 28.575/28.455 s, chunk16k 58 it. 13.409/13.420 s, mixed-ring 13 it. 2.870/2.938 s, mixed-eager 2 it. 0.388/0.423 s, mixed-graph 1 it. 0.036/0.016 s
- crosscheck: n=4667, Σ measured 45.253 s, Σ predicted 45.277 s, error +0.05 %; by stage: decode 4593 it. 28.575/28.455 s, chunk16k 58 it. 13.409/13.420 s, mixed-ring 13 it. 2.870/2.938 s, mixed-eager 2 it. 0.388/0.423 s, mixed-graph 1 it. 0.036/0.016 s

### Prediction status counts
- crosscheck | extrapolation: 69
- crosscheck | in-range: 13
- crosscheck | inferred: 58
- crosscheck | interpolated: 1140
- crosscheck | measured: 3387

## Long-context prefill (16k chunks)
Calibration fit on 55 clean 16k chunks (kv0 0–98262, chains [1, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27]): wall = 223.7 + 0.2001·kv0_k ms (rmse 3.64); slowest-rank busy = 219.9 + 0.2462·kv0_k; quadratic term at max kv0 = +26.36 ms (rmse 3.00).
| group | a_ms | s_ms_per_1k | rmse_ms |
|---|---|---|---|
| fmha | 41.141 | -0.0115 | 0.365 |
| indexer | 10.836 | +0.2429 | 0.528 |
| sparse_meta | 8.540 | +0.0032 | 0.097 |
| moe | 48.488 | +0.0013 | 0.370 |
| comm | 41.669 | -0.0301 | 1.760 |
| other | 76.480 | +0.0603 | 0.697 |

### Held-out chunks by KV offset bucket (chain-level and band hold-outs)
| kv0_bucket_k | n | chains | status | wall_median_err_ms | wall_p90_ape | wall_worst_ms | indexer_median_err_ms | indexer_p90_ape | model_wall_median_err_ms | model_wall_p90_ape |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 | [3, 9, 12, 21] | ['in-range'] | -5.073 | 2.8 | 6.568 | -0.109 | 4.7 | -8.605 | 4.3 |
| 16 | 4 | [3, 9, 12, 21] | ['in-range'] | 4.241 | 2.1 | 4.761 | -0.075 | 3.6 | 2.159 | 1.2 |
| 32 | 4 | [3, 9, 12, 21] | ['in-range'] | 1.887 | 1.0 | 2.557 | -0.485 | 3.7 | 1.236 | 0.7 |
| 48 | 11 | [1, 2, 3, 9, 10, 12, 14, 17, 20, 21, 27] | ['in-range'] | 1.184 | 0.7 | 1.865 | 0.113 | 3.3 | 1.854 | 1.0 |
| 64 | 4 | [3, 9, 12, 21] | ['in-range'] | -0.844 | 0.5 | 1.324 | 0.157 | 1.8 | 1.303 | 0.7 |
| 80 | 3 | [9, 12, 21] | ['in-range'] | -3.225 | 1.4 | 3.445 | -0.423 | 2.8 | 0.594 | 0.4 |
| 96 | 2 | [9, 12] | ['extrapolation', 'in-range'] | -1.879 | 0.8 | 1.915 | 0.789 | 3.0 | 3.645 | 1.5 |
| 112 | 1 | [12] | ['extrapolation'] | -6.479 | 2.6 | 6.479 | -0.850 | 2.1 | 0.706 | 0.3 |

(`wall`/`indexer` = the dedicated 16k-chunk linear-in-kv0 fit; `model_wall` = the full mixed/prefill component model.)

### Slopes (ms per 1k KV) fitted independently on held-out and salvaged data
| fit | n | a_ms | s_ms_per_1k | kv0_range |
|---|---|---|---|---|
| heldout_pooled_wall | 33 | 221.902 | +0.2377 | (0, 114653) |
| heldout_chain3_wall | 5 | 225.662 | +0.1431 | (0, 65512) |
| heldout_chain9_wall | 7 | 222.517 | +0.2274 | (0, 98244) |
| heldout_chain12_wall | 8 | 222.723 | +0.2391 | (0, 114653) |
| heldout_chain21_wall | 6 | 218.582 | +0.2982 | (2176, 84036) |
| salvaged_all_wall | 58 | 222.609 | +0.2282 | (0, 114625) |
| salvaged_chain1_wall | 5 | 225.240 | +0.1517 | (0, 65519) |
| salvaged_chain2_wall | 4 | 217.827 | +0.3110 | (15484, 64620) |
| salvaged_chain3_wall | 5 | 226.376 | +0.1334 | (0, 65507) |
| salvaged_chain9_wall | 7 | 218.573 | +0.2898 | (2779, 100992) |
| salvaged_chain10_wall | 7 | 223.610 | +0.2152 | (0, 98224) |
| salvaged_chain12_wall | 8 | 223.397 | +0.2340 | (0, 114625) |
| salvaged_chain14_wall | 5 | 225.114 | +0.1550 | (0, 65492) |
| salvaged_chain17_wall | 7 | 223.786 | +0.2137 | (0, 98244) |
| heldout_pooled_indexer | 33 | 11.043 | +0.2420 | (0, 114653) |
| heldout_chain3_indexer | 5 | 11.114 | +0.2369 | (0, 65512) |
| heldout_chain9_indexer | 7 | 11.023 | +0.2365 | (0, 98244) |
| heldout_chain12_indexer | 8 | 10.978 | +0.2447 | (0, 114653) |
| heldout_chain21_indexer | 6 | 11.074 | +0.2455 | (2176, 84036) |
| salvaged_all_indexer | 58 | 10.918 | +0.2429 | (0, 114625) |
| salvaged_chain1_indexer | 5 | 10.848 | +0.2377 | (0, 65519) |
| salvaged_chain2_indexer | 4 | 11.715 | +0.2253 | (15484, 64620) |
| salvaged_chain3_indexer | 5 | 10.495 | +0.2444 | (0, 65507) |
| salvaged_chain9_indexer | 7 | 11.187 | +0.2430 | (2779, 100992) |
| salvaged_chain10_indexer | 7 | 11.665 | +0.2304 | (0, 98224) |
| salvaged_chain12_indexer | 8 | 11.441 | +0.2403 | (0, 114625) |
| salvaged_chain14_indexer | 5 | 10.354 | +0.2552 | (0, 65492) |
| salvaged_chain17_indexer | 7 | 11.006 | +0.2382 | (0, 98244) |
| calibration_group_fmha | — | 41.141 | -0.0115 | — |
| calibration_group_indexer | — | 10.836 | +0.2429 | — |
| calibration_group_sparse_meta | — | 8.540 | +0.0032 | — |
| calibration_group_moe | — | 48.488 | +0.0013 | — |
| calibration_group_comm | — | 41.669 | -0.0301 | — |
| calibration_group_other | — | 76.480 | +0.0603 | — |

## MoE statistical model (decode)
| b | role | n | status | meas_p10_ms | meas_p50_ms | meas_p90_ms | pred_central_ms | pred_low_ms | pred_high_ms | central_err_pct | E_mean | heldout_E_mean | heldout_routing_central_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | calibration | 826 | in-range | 0.841 | 0.861 | 0.891 | 1.039 | 1.039 | 1.039 | +20.7 | 6.0 | 6.0 | 1.039 |
| 1 | heldout | 207 | in-range | 0.841 | 0.859 | 0.893 | 1.039 | 1.039 | 1.039 | +20.9 | 6.0 | 6.0 | 1.039 |
| 2 | calibration | 1268 | in-range | 1.023 | 1.032 | 1.070 | 1.186 | 1.178 | 1.194 | +14.9 | 11.4 | 11.5 | 1.189 |
| 2 | heldout | 317 | in-range | 1.023 | 1.032 | 1.069 | 1.186 | 1.178 | 1.194 | +14.9 | 11.4 | 11.5 | 1.189 |
| 3 | calibration | 664 | in-range | 1.467 | 1.508 | 1.539 | 1.460 | 1.442 | 1.479 | -3.1 | 21.2 | 21.5 | 1.467 |
| 3 | heldout | 167 | in-range | 1.464 | 1.512 | 1.539 | 1.460 | 1.442 | 1.479 | -3.4 | 21.2 | 21.5 | 1.467 |
| 4 | calibration | 1121 | in-range | 1.429 | 1.457 | 1.482 | 1.460 | 1.442 | 1.479 | +0.3 | 21.2 | 21.5 | 1.467 |
| 4 | heldout | 294 | in-range | 1.430 | 1.456 | 1.482 | 1.460 | 1.442 | 1.479 | +0.3 | 21.2 | 21.5 | 1.467 |
| 5 | heldout | 663 | in-range | 1.725 | 1.798 | 1.850 | 1.942 | 1.901 | 1.979 | +8.0 | 37.6 | 38.6 | 1.964 |
| 6 | calibration | 1270 | in-range | 1.895 | 1.942 | 1.981 | 1.942 | 1.901 | 1.979 | +0.0 | 37.6 | 38.6 | 1.964 |
| 6 | heldout | 694 | in-range | 1.885 | 1.929 | 1.969 | 1.942 | 1.901 | 1.979 | +0.7 | 37.6 | 38.6 | 1.964 |
| 7 | calibration | 1104 | in-range | 1.986 | 2.022 | 2.054 | 1.942 | 1.901 | 1.979 | -3.9 | 37.6 | 38.6 | 1.964 |
| 7 | heldout | 515 | in-range | 1.970 | 2.016 | 2.049 | 1.942 | 1.901 | 1.979 | -3.6 | 37.6 | 38.6 | 1.964 |
| 8 | calibration | 1038 | in-range | 1.915 | 1.963 | 2.015 | 1.942 | 1.901 | 1.979 | -1.1 | 37.6 | 38.6 | 1.964 |
| 8 | heldout | 637 | in-range | 1.908 | 1.953 | 2.006 | 1.942 | 1.901 | 1.979 | -0.5 | 37.6 | 38.6 | 1.964 |
| 9 | calibration | 1014 | in-range | 2.954 | 3.015 | 3.070 | 2.763 | 2.680 | 2.824 | -8.3 | 63.6 | 65.8 | 2.809 |
| 9 | heldout | 349 | in-range | 2.950 | 3.011 | 3.059 | 2.763 | 2.680 | 2.824 | -8.2 | 63.6 | 65.8 | 2.809 |
| 10 | calibration | 1 | in-range | 2.998 | 2.998 | 2.998 | 2.763 | 2.680 | 2.824 | -7.8 | 63.6 | 65.8 | 2.809 |
| 10 | heldout | 1367 | in-range | 2.978 | 3.035 | 3.090 | 2.763 | 2.680 | 2.824 | -9.0 | 63.6 | 65.8 | 2.809 |
| 11 | calibration | 1282 | in-range | 2.855 | 2.945 | 3.018 | 2.763 | 2.680 | 2.824 | -6.2 | 63.6 | 65.8 | 2.809 |
| 11 | heldout | 1169 | in-range | 2.849 | 2.939 | 3.014 | 2.763 | 2.680 | 2.824 | -6.0 | 63.6 | 65.8 | 2.809 |
| 12 | calibration | 746 | in-range | 2.849 | 2.916 | 3.000 | 2.763 | 2.680 | 2.824 | -5.2 | 63.6 | 65.8 | 2.809 |
| 12 | heldout | 657 | in-range | 2.856 | 2.921 | 2.990 | 2.763 | 2.680 | 2.824 | -5.4 | 63.6 | 65.8 | 2.809 |
| 13 | calibration | 279 | in-range | 2.806 | 2.859 | 2.920 | 2.763 | 2.680 | 2.824 | -3.4 | 63.6 | 65.8 | 2.809 |
| 13 | heldout | 277 | in-range | 2.808 | 2.861 | 2.920 | 2.763 | 2.680 | 2.824 | -3.4 | 63.6 | 65.8 | 2.809 |
| 14 | heldout | 204 | in-range | 2.733 | 2.803 | 2.875 | 2.763 | 2.680 | 2.824 | -1.4 | 63.6 | 65.8 | 2.809 |
| 15 | calibration | 5 | in-range | 2.781 | 2.818 | 2.859 | 2.763 | 2.680 | 2.824 | -1.9 | 63.6 | 65.8 | 2.809 |
| 15 | heldout | 7 | in-range | 2.796 | 2.812 | 2.852 | 2.763 | 2.680 | 2.824 | -1.8 | 63.6 | 65.8 | 2.809 |
| 16 | calibration | 13 | in-range | 2.758 | 2.795 | 2.841 | 2.763 | 2.680 | 2.824 | -1.2 | 63.6 | 65.8 | 2.809 |
| 16 | heldout | 13 | in-range | 2.733 | 2.800 | 2.817 | 2.763 | 2.680 | 2.824 | -1.3 | 63.6 | 65.8 | 2.809 |
| 17 | calibration | 1 | in-range | 3.145 | 3.145 | 3.145 | 3.468 | 3.382 | 3.563 | +10.3 | 84.2 | 87.9 | 3.546 |
| 19 | calibration | 1 | in-range | 3.381 | 3.381 | 3.381 | 3.468 | 3.382 | 3.563 | +2.6 | 84.2 | 87.9 | 3.546 |
| 20 | calibration | 1 | in-range | 3.396 | 3.396 | 3.396 | 3.468 | 3.382 | 3.563 | +2.1 | 84.2 | 87.9 | 3.546 |
| 23 | calibration | 1 | in-range | 3.639 | 3.639 | 3.639 | 3.468 | 3.382 | 3.563 | -4.7 | 84.2 | 87.9 | 3.546 |
| 23 | heldout | 1 | in-range | 3.676 | 3.676 | 3.676 | 3.468 | 3.382 | 3.563 | -5.6 | 84.2 | 87.9 | 3.546 |
| 24 | calibration | 190 | in-range | 3.406 | 3.475 | 3.567 | 3.468 | 3.382 | 3.563 | -0.2 | 84.2 | 87.9 | 3.546 |
| 24 | heldout | 48 | in-range | 3.391 | 3.480 | 3.542 | 3.468 | 3.382 | 3.563 | -0.3 | 84.2 | 87.9 | 3.546 |
| 25 | calibration | 210 | in-range | 3.633 | 3.702 | 3.772 | 4.101 | 3.991 | 4.193 | +10.8 | 101.3 | 105.7 | 4.193 |
| 25 | heldout | 52 | in-range | 3.654 | 3.703 | 3.769 | 4.101 | 3.991 | 4.193 | +10.8 | 101.3 | 105.7 | 4.193 |
| 26 | heldout | 129 | in-range | 3.692 | 3.773 | 3.852 | 4.101 | 3.991 | 4.193 | +8.7 | 101.3 | 105.7 | 4.193 |
| 27 | calibration | 103 | in-range | 3.794 | 3.865 | 3.940 | 4.101 | 3.991 | 4.193 | +6.1 | 101.3 | 105.7 | 4.193 |
| 27 | heldout | 25 | in-range | 3.818 | 3.858 | 3.917 | 4.101 | 3.991 | 4.193 | +6.3 | 101.3 | 105.7 | 4.193 |
| 28 | calibration | 1 | in-range | 3.946 | 3.946 | 3.946 | 4.101 | 3.991 | 4.193 | +3.9 | 101.3 | 105.7 | 4.193 |
| 29 | calibration | 1 | in-range | 4.013 | 4.013 | 4.013 | 4.101 | 3.991 | 4.193 | +2.2 | 101.3 | 105.7 | 4.193 |
| 29 | heldout | 1 | in-range | 4.083 | 4.083 | 4.083 | 4.101 | 3.991 | 4.193 | +0.4 | 101.3 | 105.7 | 4.193 |
| 30 | heldout | 127 | in-range | 3.968 | 4.049 | 4.109 | 4.101 | 3.991 | 4.193 | +1.3 | 101.3 | 105.7 | 4.193 |
| 32 | calibration | 2 | in-range | 3.989 | 4.002 | 4.015 | 4.101 | 3.991 | 4.193 | +2.5 | 101.3 | 105.7 | 4.193 |
| 33 | heldout | 1 | in-range | 4.671 | 4.671 | 4.671 | 4.670 | 4.569 | 4.764 | -0.0 | 115.4 | 120.7 | 4.781 |
| 34 | calibration | 2 | in-range | 4.418 | 4.562 | 4.707 | 4.670 | 4.569 | 4.764 | +2.4 | 115.4 | 120.7 | 4.781 |
| 34 | heldout | 1 | in-range | 4.385 | 4.385 | 4.385 | 4.670 | 4.569 | 4.764 | +6.5 | 115.4 | 120.7 | 4.781 |
| 36 | heldout | 126 | in-range | 4.336 | 4.428 | 4.487 | 4.670 | 4.569 | 4.764 | +5.5 | 115.4 | 120.7 | 4.781 |
| 37 | heldout | 1 | in-range | 4.583 | 4.583 | 4.583 | 4.670 | 4.569 | 4.764 | +1.9 | 115.4 | 120.7 | 4.781 |
| 39 | calibration | 2 | in-range | 4.979 | 5.004 | 5.029 | 4.670 | 4.569 | 4.764 | -6.7 | 115.4 | 120.7 | 4.781 |
| 40 | calibration | 1 | in-range | 4.437 | 4.437 | 4.437 | 4.670 | 4.569 | 4.764 | +5.3 | 115.4 | 120.7 | 4.781 |
| 41 | calibration | 1 | in-range | 5.314 | 5.314 | 5.314 | 5.201 | 5.101 | 5.305 | -2.1 | 127.7 | 133.9 | 5.333 |
| 42 | calibration | 1 | in-range | 5.071 | 5.071 | 5.071 | 5.201 | 5.101 | 5.305 | +2.6 | 127.7 | 133.9 | 5.333 |
| 43 | calibration | 2 | in-range | 5.281 | 5.317 | 5.354 | 5.201 | 5.101 | 5.305 | -2.2 | 127.7 | 133.9 | 5.333 |
| 44 | calibration | 2 | in-range | 5.147 | 5.217 | 5.286 | 5.201 | 5.101 | 5.305 | -0.3 | 127.7 | 133.9 | 5.333 |
| 45 | calibration | 1 | in-range | 5.401 | 5.401 | 5.401 | 5.201 | 5.101 | 5.305 | -3.7 | 127.7 | 133.9 | 5.333 |
| 46 | calibration | 3 | in-range | 5.228 | 5.335 | 5.380 | 5.201 | 5.101 | 5.305 | -2.5 | 127.7 | 133.9 | 5.333 |
| 47 | calibration | 5 | in-range | 5.141 | 5.301 | 5.355 | 5.201 | 5.101 | 5.305 | -1.9 | 127.7 | 133.9 | 5.333 |
| 47 | heldout | 1 | in-range | 5.376 | 5.376 | 5.376 | 5.201 | 5.101 | 5.305 | -3.3 | 127.7 | 133.9 | 5.333 |
| 48 | calibration | 2425 | in-range | 4.897 | 5.102 | 5.313 | 5.201 | 5.101 | 5.305 | +1.9 | 127.7 | 133.9 | 5.333 |
| 48 | heldout | 607 | in-range | 4.899 | 5.111 | 5.318 | 5.201 | 5.101 | 5.305 | +1.8 | 127.7 | 133.9 | 5.333 |

t_moe(b) = 0.878 + 0.02107·E(b) + 0.03400·b ms; R² 0.984, rmse 0.166 ms over 35 batch sizes. Range = p10–p90 of E(b) propagated through β; per-iteration routing is not predicted.

## Parameter classification
| component | parameter | status | evidence |
|---|---|---|---|
| 1 decode | per-batch per-rank compute windows (K+1 per b), first-kernel offsets | measured | medians over 13587 calibration iterations, 35 batch sizes |
| 1 decode | per-call collective intrinsic cost (mhc / oneshot / twoshot / all-gather) per b | measured | min over ranks per call, median over iterations |
| 1 decode | multi-stream overlap, intra-window idle per rank | measured | Σ durations − union; window span − union |
| 1 decode | values at non-calibrated batch sizes | inferred | linear interpolation between neighbouring calibrated b of the same regime |
| 1 decode | KV independence | assumed within observed range | kv_max <= 131063; tested in evaluate (residual vs kv) |
| 1 decode | stream assignment of overlapping kernels | not identifiable | graph-executor stream ids are not program streams (gap G6); overlap carried as a measured term |
| 2 mixed | launch count L(P, b, KV) | measured | linear fit, rmse 12.3 launches |
| 2 mixed | per-rank per-launch host cost c_r | measured | 14 host-bound eager calibration iterations |
| 2 mixed | per-launch CUDA API cost c_api (unthrottled) | measured |  |
| 2 mixed | GPU tail after the last launch | measured |  |
| 2 mixed | cost of KV-dependent launches c_kv | inferred | inferred (de-confounded from launch back-pressure; the host is not on the critical path in these iterations) |
| 2 mixed | graph-path mixed iterations (P+b <= 256) | measured class medians only | 2 (P,b) classes |
| 2 mixed | cause of rank-2 host slowness | not identifiable | gap G4 |
| 3 moe | E(b) distinct experts per layer | measured (corpus pass) | 110691 calibration tokens |
| 3 moe | α, β, γ of t_moe(b) = α + β·E(b) + γ·b | inferred | fit through batch-level kernel medians, R² 0.984 |
| 3 moe | per-iteration routing | not identifiable | not captured in timing passes (gap G3) |
| 4 prefill/longctx | per-group per-rank compute vs (T, Σt², KV, b) | measured | 87 calibration mixed iterations |
| 4 prefill/longctx | indexer / sparse-metadata KV slope | measured | 55 calibration 16k chunks, kv0 <= 98262.0 |
| 4 prefill/longctx | chunk-length × KV interaction (t != 16k at kv0 > 0) | inferred structure | gap G1: only a handful of multi-chunk calibration rows |
| 4 prefill/longctx | kv0 beyond the deepest calibration chunk | extrapolation (flagged) | no silent prediction |
| 5 collectives | eager/ring intrinsic per call vs payload tokens | measured | twoshot: n=20, nccl_allreduce: n=70, nccl_allgather: n=90 |
| 5 collectives | payload bytes | inferred | tokens × hidden (not in trace) |
| 5 collectives | waiting time | derived | composition (rank skew) or host model; never a collective parameter |
| 6 composition | completion = max_r arrival + intrinsic | verified | reconstruction error reported in evaluate |
| 7 host | inter-iteration GPU gap | measured | ≈ 0 in all classes (GPU-saturated, host runs ahead) |
| 7 host | rank-specific decode launch lateness (b <= 2) | measured (effect) | cause inferred: rank-0 sampling/bookkeeping |
| 7 host | back-pressure inflation of launch API time | measured | evidence table |
