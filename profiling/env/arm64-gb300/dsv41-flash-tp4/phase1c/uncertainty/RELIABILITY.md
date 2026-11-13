# Reliability of the Phase 1B predictions (uncertainty audit, Phase 1C-0)

pred-mc with 32 draws, seeds 0..4; frozen coefficients/partition/params (hashes in ../repro/MANIFEST.json). Interval = p10-p90 of the draws. Errors are predicted mean minus measured. No parameter was changed.

## diverse_nsys_run2

### pred-mc (seed 0) with seed variation

| class | n | bias_pct | median_abs_err_ms | p90_abs_err_ms | cover_p10_p90_pct | width_mean_ms | width_median_ms | below_pct | above_pct | bias_pct_seed_std | cover_seed_std | cover_seed_min | cover_seed_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | 1886 | 1.762 | 0.286 | 0.794 | 85.631 | 1.300 | 1.252 | 8.006 | 6.363 | 0.155 | 0.520 | 85.631 | 86.957 |
| calibration | decode b3-16 | 8226 | 0.697 | 0.063 | 0.132 | 61.999 | 0.170 | 0.167 | 34.841 | 3.161 | 0.010 | 0.630 | 60.637 | 61.999 |
| calibration | mixed-eager (host-bound) | 14 | 0.749 | 7.458 | 19.076 | 14.286 | 1.221 | 0.179 | 57.143 | 28.571 | 0.034 | 3.194 | 7.143 | 14.286 |
| calibration | mixed-ring (other) | 10 | 2.203 | 4.020 | 13.464 | 50.000 | 6.325 | 6.820 | 30.000 | 20.000 | 0.024 | 0.000 | 50.000 | 50.000 |
| calibration | prefill 16k chunk | 55 | 0.275 | 1.180 | 3.278 | 89.091 | 7.374 | 7.302 | 9.091 | 1.818 | 0.053 | 4.186 | 83.636 | 92.727 |
| heldout | decode b1-2 | 471 | 2.051 | 0.270 | 0.793 | 84.926 | 1.298 | 1.256 | 8.917 | 6.157 | 0.127 | 1.549 | 84.926 | 88.323 |
| heldout | decode b3-16 | 6935 | 0.522 | 0.063 | 0.154 | 57.751 | 0.176 | 0.158 | 30.411 | 11.839 | 0.013 | 0.380 | 56.842 | 57.751 |
| heldout | mixed-eager (host-bound) | 3 | -1.757 | 5.293 | 12.588 | 0.000 | 0.111 | 0.141 | 33.333 | 66.667 | 0.004 | 0.000 | 0.000 | 0.000 |
| heldout | mixed-ring (other) | 4 | -0.283 | 4.181 | 5.756 | 50.000 | 5.298 | 6.061 | 25.000 | 25.000 | 0.163 | 13.693 | 25.000 | 50.000 |
| heldout | prefill 16k chunk | 33 | 0.020 | 0.649 | 1.801 | 100.000 | 8.218 | 8.180 | 0.000 | 0.000 | 0.029 | 1.660 | 96.970 | 100.000 |

### pred-det (median path, no interval)

| class | n | bias_pct | median_abs_err_ms | p90_abs_err_ms |
|---|---|---|---|---|
| calibration | decode b1-2 | 1886 | 0.266 | 0.279 | 0.807 |
| calibration | decode b3-16 | 8226 | -2.702 | 0.136 | 0.231 |
| calibration | mixed-eager (host-bound) | 14 | 0.850 | 7.549 | 18.876 |
| calibration | mixed-ring (other) | 10 | 2.183 | 3.724 | 13.662 |
| calibration | prefill 16k chunk | 55 | 0.251 | 0.686 | 3.474 |
| heldout | decode b1-2 | 471 | 0.476 | 0.271 | 0.740 |
| heldout | decode b3-16 | 6935 | -2.829 | 0.153 | 0.261 |
| heldout | mixed-eager (host-bound) | 3 | -1.687 | 5.293 | 12.417 |
| heldout | mixed-ring (other) | 4 | -0.361 | 4.389 | 5.874 |
| heldout | prefill 16k chunk | 33 | -0.008 | 0.458 | 1.307 |

## quadrant_nsys

### pred-mc (seed 0) with seed variation

| class | n | bias_pct | median_abs_err_ms | p90_abs_err_ms | cover_p10_p90_pct | width_mean_ms | width_median_ms | below_pct | above_pct | bias_pct_seed_std | cover_seed_std | cover_seed_min | cover_seed_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | 208 | 4.157 | 0.246 | 0.592 | 97.115 | 1.613 | 1.612 | 1.923 | 0.962 | 0.246 | 1.158 | 96.154 | 99.038 |
| calibration | decode b17-24 | 194 | 1.448 | 0.116 | 0.226 | 51.546 | 0.250 | 0.250 | 47.938 | 0.515 | 0.007 | 1.748 | 49.485 | 54.124 |
| calibration | decode b3-16 | 311 | 2.248 | 0.124 | 0.208 | 26.045 | 0.175 | 0.180 | 73.633 | 0.322 | 0.016 | 1.390 | 23.794 | 27.331 |
| calibration | decode b>=25 | 2762 | 1.949 | 0.207 | 0.426 | 43.736 | 0.398 | 0.369 | 54.200 | 2.064 | 0.017 | 0.191 | 43.338 | 43.809 |
| calibration | mixed-eager (host-bound) | 3 | 11.216 | 4.529 | 51.526 | 0.000 | 2.986 | 1.683 | 66.667 | 33.333 | 0.098 | 18.257 | 0.000 | 33.333 |
| calibration | mixed-graph | 4 | -0.313 | 1.700 | 2.715 | 0.000 | 0.002 | 0.002 | 50.000 | 50.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| calibration | mixed-ring (other) | 5 | 1.293 | 5.581 | 8.485 | 20.000 | 7.754 | 8.173 | 60.000 | 20.000 | 0.036 | 10.954 | 0.000 | 20.000 |
| heldout | decode b1-2 | 53 | 1.437 | 0.235 | 0.582 | 96.226 | 1.591 | 1.578 | 1.887 | 1.887 | 0.446 | 2.311 | 92.453 | 98.113 |
| heldout | decode b17-24 | 49 | 1.340 | 0.119 | 0.202 | 51.020 | 0.247 | 0.244 | 46.939 | 2.041 | 0.043 | 4.698 | 46.939 | 59.184 |
| heldout | decode b3-16 | 78 | 2.489 | 0.124 | 0.206 | 17.949 | 0.212 | 0.190 | 82.051 | 0.000 | 0.057 | 2.027 | 14.103 | 19.231 |
| heldout | decode b>=25 | 1071 | 2.189 | 0.231 | 0.416 | 38.469 | 0.424 | 0.371 | 59.384 | 2.148 | 0.050 | 0.739 | 38.469 | 40.336 |
| heldout | mixed-eager (host-bound) | 1 | 0.950 | 1.906 | 1.906 | 0.000 | 1.378 | 1.378 | 100.000 | 0.000 | 0.082 | 54.772 | 0.000 | 100.000 |
| heldout | mixed-ring (other) | 1 | 2.028 | 4.496 | 4.496 | 100.000 | 7.946 | 7.946 | 0.000 | 0.000 | 0.129 | 44.721 | 0.000 | 100.000 |

### pred-det (median path, no interval)

| class | n | bias_pct | median_abs_err_ms | p90_abs_err_ms |
|---|---|---|---|---|
| calibration | decode b1-2 | 208 | 0.714 | 0.392 | 0.705 |
| calibration | decode b17-24 | 194 | -0.948 | 0.073 | 0.203 |
| calibration | decode b3-16 | 311 | -1.305 | 0.073 | 0.123 |
| calibration | decode b>=25 | 2762 | -0.924 | 0.110 | 0.246 |
| calibration | mixed-eager (host-bound) | 3 | 11.348 | 4.529 | 52.006 |
| calibration | mixed-graph | 4 | -0.314 | 1.700 | 2.715 |
| calibration | mixed-ring (other) | 5 | 1.330 | 5.809 | 7.723 |
| heldout | decode b1-2 | 53 | -1.100 | 0.375 | 0.727 |
| heldout | decode b17-24 | 49 | -1.056 | 0.074 | 0.199 |
| heldout | decode b3-16 | 78 | -1.179 | 0.068 | 0.116 |
| heldout | decode b>=25 | 1071 | -0.893 | 0.101 | 0.230 |
| heldout | mixed-eager (host-bound) | 1 | 1.225 | 2.457 | 2.457 |
| heldout | mixed-ring (other) | 1 | 2.149 | 4.765 | 4.765 |

## diverse_nsys_salvaged

### pred-mc (seed 0) with seed variation

| class | n | bias_pct | median_abs_err_ms | p90_abs_err_ms | cover_p10_p90_pct | width_mean_ms | width_median_ms | below_pct | above_pct | bias_pct_seed_std | cover_seed_std | cover_seed_min | cover_seed_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calibration | decode b17-24 | 34 | 3.324 | 0.385 | 0.467 | 0.000 | 0.133 | 0.134 | 97.059 | 2.941 | 0.019 | 0.000 | 0.000 | 0.000 |
| calibration | decode b3-16 | 2560 | 0.622 | 0.059 | 0.153 | 68.047 | 0.271 | 0.161 | 24.492 | 7.461 | 0.030 | 0.397 | 67.305 | 68.320 |
| calibration | mixed-eager (host-bound) | 1 | -4.124 | 8.664 | 8.664 | 0.000 | 0.126 | 0.126 | 0.000 | 100.000 | 0.002 | 0.000 | 0.000 | 0.000 |
| calibration | mixed-graph | 1 | 123.896 | 19.994 | 19.994 | 0.000 | 0.000 | 0.000 | 100.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| calibration | mixed-ring (other) | 10 | -0.711 | 1.987 | 5.477 | 70.000 | 7.129 | 6.805 | 0.000 | 30.000 | 0.069 | 8.367 | 60.000 | 80.000 |
| calibration | prefill 16k chunk | 33 | 0.145 | 0.979 | 2.439 | 90.909 | 7.101 | 7.467 | 9.091 | 0.000 | 0.066 | 0.000 | 90.909 | 90.909 |
| heldout | decode b17-24 | 32 | 4.912 | 0.374 | 0.463 | 0.000 | 0.132 | 0.135 | 100.000 | 0.000 | 0.018 | 0.000 | 0.000 | 0.000 |
| heldout | decode b3-16 | 1967 | 0.396 | 0.058 | 0.163 | 69.446 | 0.284 | 0.167 | 24.250 | 6.304 | 0.028 | 0.146 | 69.192 | 69.548 |
| heldout | mixed-eager (host-bound) | 1 | -5.496 | 11.721 | 11.721 | 0.000 | 0.185 | 0.185 | 0.000 | 100.000 | 0.005 | 0.000 | 0.000 | 0.000 |
| heldout | mixed-ring (other) | 3 | -1.423 | 4.662 | 5.530 | 33.333 | 6.589 | 6.085 | 0.000 | 66.667 | 0.149 | 14.907 | 33.333 | 66.667 |
| heldout | prefill 16k chunk | 25 | -0.153 | 1.088 | 2.252 | 96.000 | 7.797 | 8.104 | 0.000 | 4.000 | 0.028 | 1.789 | 92.000 | 96.000 |

### pred-det (median path, no interval)

| class | n | bias_pct | median_abs_err_ms | p90_abs_err_ms |
|---|---|---|---|---|
| calibration | decode b17-24 | 34 | 0.912 | 0.197 | 0.283 |
| calibration | decode b3-16 | 2560 | -2.752 | 0.157 | 0.256 |
| calibration | mixed-eager (host-bound) | 1 | -4.026 | 8.459 | 8.459 |
| calibration | mixed-graph | 1 | 123.895 | 19.994 | 19.994 |
| calibration | mixed-ring (other) | 10 | -0.643 | 1.933 | 5.448 |
| calibration | prefill 16k chunk | 33 | 0.181 | 0.367 | 2.229 |
| heldout | decode b17-24 | 32 | 2.492 | 0.200 | 0.279 |
| heldout | decode b3-16 | 1967 | -2.916 | 0.153 | 0.248 |
| heldout | mixed-eager (host-bound) | 1 | -5.399 | 11.513 | 11.513 |
| heldout | mixed-ring (other) | 3 | -1.340 | 4.813 | 5.136 |
| heldout | prefill 16k chunk | 25 | -0.154 | 0.676 | 2.093 |

## Interval displacement decomposition (decode classes)

Variants differ only in which stochastic component is switched on; V0 is the median path, V3 is the reported pred-mc.

### quadrant_nsys

| class | variant | n | bias_pct | pred_median_ms | meas_median_ms | cover_p10_p90_pct | width_median_ms | below_pct | above_pct |
|---|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | V0 median path (pred-det) | 208 | 0.714 | 4.428 | 4.783 | 0.000 | 0.000 | 52.404 | 47.596 |
| calibration | decode b1-2 | V1 per-(rank,segment) draws only | 208 | 3.083 | 4.935 | 4.783 | 94.231 | 1.225 | 3.846 | 1.923 |
| calibration | decode b1-2 | V2 draws + common-mode dispersion | 208 | 4.015 | 4.977 | 4.783 | 96.635 | 1.540 | 1.923 | 1.442 |
| calibration | decode b1-2 | V3 draws + dispersion + MoE band (= pred-mc) | 208 | 4.157 | 4.976 | 4.783 | 97.115 | 1.612 | 1.923 | 0.962 |
| calibration | decode b1-2 | V4 draws + MoE band, no dispersion | 208 | 3.198 | 4.942 | 4.783 | 94.231 | 1.263 | 4.327 | 1.442 |
| calibration | decode b17-24 | V0 median path (pred-det) | 194 | -0.948 | 7.665 | 7.726 | 0.000 | 0.000 | 21.649 | 78.351 |
| calibration | decode b17-24 | V1 per-(rank,segment) draws only | 194 | 1.394 | 7.846 | 7.726 | 10.309 | 0.040 | 79.897 | 9.794 |
| calibration | decode b17-24 | V2 draws + common-mode dispersion | 194 | 1.393 | 7.845 | 7.726 | 46.392 | 0.233 | 53.093 | 0.515 |
| calibration | decode b17-24 | V3 draws + dispersion + MoE band (= pred-mc) | 194 | 1.448 | 7.852 | 7.726 | 51.546 | 0.250 | 47.938 | 0.515 |
| calibration | decode b17-24 | V4 draws + MoE band, no dispersion | 194 | 1.442 | 7.850 | 7.726 | 26.804 | 0.130 | 70.619 | 2.577 |
| calibration | decode b3-16 | V0 median path (pred-det) | 311 | -1.305 | 4.966 | 5.042 | 0.000 | 0.000 | 8.039 | 91.961 |
| calibration | decode b3-16 | V1 per-(rank,segment) draws only | 311 | 2.307 | 5.185 | 5.042 | 2.572 | 0.053 | 96.463 | 0.965 |
| calibration | decode b3-16 | V2 draws + common-mode dispersion | 311 | 2.305 | 5.183 | 5.042 | 20.900 | 0.174 | 78.457 | 0.643 |
| calibration | decode b3-16 | V3 draws + dispersion + MoE band (= pred-mc) | 311 | 2.248 | 5.183 | 5.042 | 26.045 | 0.180 | 73.633 | 0.322 |
| calibration | decode b3-16 | V4 draws + MoE band, no dispersion | 311 | 2.242 | 5.183 | 5.042 | 3.537 | 0.075 | 95.498 | 0.965 |
| calibration | decode b>=25 | V0 median path (pred-det) | 2762 | -0.924 | 9.861 | 9.910 | 0.000 | 0.000 | 29.797 | 70.203 |
| calibration | decode b>=25 | V1 per-(rank,segment) draws only | 2762 | 1.943 | 10.150 | 9.910 | 5.539 | 0.062 | 90.768 | 3.693 |
| calibration | decode b>=25 | V2 draws + common-mode dispersion | 2762 | 1.932 | 10.147 | 9.910 | 42.433 | 0.358 | 55.612 | 1.955 |
| calibration | decode b>=25 | V3 draws + dispersion + MoE band (= pred-mc) | 2762 | 1.949 | 10.149 | 9.910 | 43.736 | 0.369 | 54.200 | 2.064 |
| calibration | decode b>=25 | V4 draws + MoE band, no dispersion | 2762 | 1.947 | 10.151 | 9.910 | 14.482 | 0.151 | 83.418 | 2.100 |
| heldout | decode b1-2 | V0 median path (pred-det) | 53 | -1.100 | 4.428 | 4.832 | 0.000 | 0.000 | 49.057 | 50.943 |
| heldout | decode b1-2 | V1 per-(rank,segment) draws only | 53 | 1.595 | 4.963 | 4.832 | 90.566 | 1.234 | 5.660 | 3.774 |
| heldout | decode b1-2 | V2 draws + common-mode dispersion | 53 | 2.108 | 4.971 | 4.832 | 94.340 | 1.550 | 3.774 | 1.887 |
| heldout | decode b1-2 | V3 draws + dispersion + MoE band (= pred-mc) | 53 | 1.437 | 4.931 | 4.832 | 96.226 | 1.578 | 1.887 | 1.887 |
| heldout | decode b1-2 | V4 draws + MoE band, no dispersion | 53 | 1.408 | 4.923 | 4.832 | 92.453 | 1.184 | 3.774 | 3.774 |
| heldout | decode b17-24 | V0 median path (pred-det) | 49 | -1.056 | 7.665 | 7.739 | 0.000 | 0.000 | 14.286 | 85.714 |
| heldout | decode b17-24 | V1 per-(rank,segment) draws only | 49 | 1.295 | 7.848 | 7.739 | 12.245 | 0.042 | 77.551 | 10.204 |
| heldout | decode b17-24 | V2 draws + common-mode dispersion | 49 | 1.233 | 7.839 | 7.739 | 46.939 | 0.224 | 51.020 | 2.041 |
| heldout | decode b17-24 | V3 draws + dispersion + MoE band (= pred-mc) | 49 | 1.340 | 7.855 | 7.739 | 51.020 | 0.244 | 46.939 | 2.041 |
| heldout | decode b17-24 | V4 draws + MoE band, no dispersion | 49 | 1.318 | 7.850 | 7.739 | 30.612 | 0.133 | 67.347 | 2.041 |
| heldout | decode b3-16 | V0 median path (pred-det) | 78 | -1.179 | 4.966 | 5.043 | 0.000 | 0.000 | 6.410 | 93.590 |
| heldout | decode b3-16 | V1 per-(rank,segment) draws only | 78 | 2.532 | 5.185 | 5.043 | 2.564 | 0.054 | 96.154 | 1.282 |
| heldout | decode b3-16 | V2 draws + common-mode dispersion | 78 | 2.475 | 5.192 | 5.043 | 11.538 | 0.170 | 87.179 | 1.282 |
| heldout | decode b3-16 | V3 draws + dispersion + MoE band (= pred-mc) | 78 | 2.489 | 5.185 | 5.043 | 17.949 | 0.190 | 82.051 | 0.000 |
| heldout | decode b3-16 | V4 draws + MoE band, no dispersion | 78 | 2.381 | 5.181 | 5.043 | 2.564 | 0.076 | 96.154 | 1.282 |
| heldout | decode b>=25 | V0 median path (pred-det) | 1071 | -0.893 | 9.861 | 9.752 | 0.000 | 0.000 | 29.785 | 70.215 |
| heldout | decode b>=25 | V1 per-(rank,segment) draws only | 1071 | 2.201 | 10.145 | 9.752 | 8.683 | 0.065 | 87.488 | 3.828 |
| heldout | decode b>=25 | V2 draws + common-mode dispersion | 1071 | 2.180 | 10.121 | 9.752 | 31.746 | 0.350 | 65.640 | 2.614 |
| heldout | decode b>=25 | V3 draws + dispersion + MoE band (= pred-mc) | 1071 | 2.189 | 10.123 | 9.752 | 38.469 | 0.371 | 59.384 | 2.148 |
| heldout | decode b>=25 | V4 draws + MoE band, no dispersion | 1071 | 2.200 | 10.141 | 9.752 | 20.542 | 0.157 | 77.311 | 2.148 |

### diverse_nsys_run2

| class | variant | n | bias_pct | pred_median_ms | meas_median_ms | cover_p10_p90_pct | width_median_ms | below_pct | above_pct |
|---|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | V0 median path (pred-det) | 1886 | 0.266 | 4.483 | 4.432 | 0.000 | 0.000 | 63.203 | 36.797 |
| calibration | decode b1-2 | V1 per-(rank,segment) draws only | 1886 | 1.198 | 4.526 | 4.432 | 76.723 | 0.944 | 12.725 | 10.551 |
| calibration | decode b1-2 | V2 draws + common-mode dispersion | 1886 | 1.756 | 4.560 | 4.432 | 86.108 | 1.236 | 7.476 | 6.416 |
| calibration | decode b1-2 | V3 draws + dispersion + MoE band (= pred-mc) | 1886 | 1.762 | 4.563 | 4.432 | 85.631 | 1.252 | 8.006 | 6.363 |
| calibration | decode b1-2 | V4 draws + MoE band, no dispersion | 1886 | 1.201 | 4.529 | 4.432 | 76.458 | 0.937 | 13.043 | 10.498 |
| calibration | decode b3-16 | V0 median path (pred-det) | 8226 | -2.702 | 5.079 | 5.244 | 0.000 | 0.000 | 1.726 | 98.274 |
| calibration | decode b3-16 | V1 per-(rank,segment) draws only | 8226 | 0.777 | 5.285 | 5.244 | 19.013 | 0.055 | 71.238 | 9.750 |
| calibration | decode b3-16 | V2 draws + common-mode dispersion | 8226 | 0.777 | 5.280 | 5.244 | 55.312 | 0.157 | 41.332 | 3.355 |
| calibration | decode b3-16 | V3 draws + dispersion + MoE band (= pred-mc) | 8226 | 0.697 | 5.279 | 5.244 | 61.999 | 0.167 | 34.841 | 3.161 |
| calibration | decode b3-16 | V4 draws + MoE band, no dispersion | 8226 | 0.696 | 5.282 | 5.244 | 31.169 | 0.081 | 60.686 | 8.145 |
| heldout | decode b1-2 | V0 median path (pred-det) | 471 | 0.476 | 4.483 | 4.412 | 0.000 | 0.000 | 63.482 | 36.518 |
| heldout | decode b1-2 | V1 per-(rank,segment) draws only | 471 | 1.500 | 4.523 | 4.412 | 76.221 | 0.938 | 12.102 | 11.677 |
| heldout | decode b1-2 | V2 draws + common-mode dispersion | 471 | 1.970 | 4.570 | 4.412 | 87.261 | 1.251 | 6.794 | 5.945 |
| heldout | decode b1-2 | V3 draws + dispersion + MoE band (= pred-mc) | 471 | 2.051 | 4.565 | 4.412 | 84.926 | 1.256 | 8.917 | 6.157 |
| heldout | decode b1-2 | V4 draws + MoE band, no dispersion | 471 | 1.458 | 4.528 | 4.412 | 76.645 | 0.937 | 13.163 | 10.191 |
| heldout | decode b3-16 | V0 median path (pred-det) | 6935 | -2.829 | 6.307 | 6.393 | 0.000 | 0.000 | 1.759 | 98.241 |
| heldout | decode b3-16 | V1 per-(rank,segment) draws only | 6935 | 0.620 | 6.522 | 6.393 | 21.139 | 0.055 | 61.629 | 17.231 |
| heldout | decode b3-16 | V2 draws + common-mode dispersion | 6935 | 0.617 | 6.520 | 6.393 | 46.921 | 0.148 | 39.841 | 13.237 |
| heldout | decode b3-16 | V3 draws + dispersion + MoE band (= pred-mc) | 6935 | 0.522 | 6.510 | 6.393 | 57.751 | 0.158 | 30.411 | 11.839 |
| heldout | decode b3-16 | V4 draws + MoE band, no dispersion | 6935 | 0.516 | 6.512 | 6.393 | 38.198 | 0.096 | 47.455 | 14.348 |

## Phase 1A decode table: median path vs independent vs common-mode draws, measured walls and first-collective arrival spread by batch size

independent = per-(rank, segment) quantile draws (engine behaviour); common-mode = same quantile on all ranks. Measured columns are the per-pass wall distributions of all non-excluded decode iterations of that b.

| b | n_table | median_path_ms | indep_mean_ms | common_mean_ms | indep_p10_p90_width_ms | common_p10_p90_width_ms | run2_n | run2_meas_median_ms | run2_meas_p10_p90_width_ms | quad_n | quad_meas_median_ms | quad_meas_p10_p90_width_ms | salv_n | salv_meas_median_ms | run2_arrival_spread_median_ms | quad_arrival_spread_median_ms | moe_wall_band_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 826 | 4.428 | 4.518 | 4.332 | 1.056 | 1.081 | 775 | 4.573 | 0.851 | 258 | 4.809 | 0.590 | 0 | nan | 0.625 | 0.899 | 0.000 |
| 2 | 1268 | 4.161 | 4.417 | 4.268 | 0.973 | 0.985 | 1582 | 4.404 | 0.833 | 3 | 4.453 | 0.437 | 0 | nan | 0.456 | 0.534 | 0.271 |
| 3 | 664 | 4.523 | 4.667 | 4.550 | 0.039 | 0.049 | 703 | 4.618 | 0.123 | 128 | 4.609 | 0.070 | 0 | nan | 0.018 | 0.018 | 0.625 |
| 4 | 1121 | 4.504 | 4.650 | 4.540 | 0.036 | 0.043 | 1414 | 4.595 | 0.122 | 1 | 4.654 | 0.000 | 164 | 4.593 | 0.016 | 0.031 | 0.635 |
| 5 | 0 | 4.727 | 4.912 | 4.779 | 0.046 | 0.058 | 662 | 5.005 | 0.196 | 1 | 4.949 | 0.000 | 1 | 9.887 | 0.019 | 0.018 | 1.113 |
| 6 | 1270 | 4.961 | 5.182 | 5.033 | 0.063 | 0.078 | 1837 | 5.129 | 0.157 | 127 | 5.042 | 0.082 | 108 | 5.144 | 0.017 | 0.018 | 1.187 |
| 7 | 1104 | 5.073 | 5.283 | 5.140 | 0.062 | 0.077 | 1617 | 5.237 | 0.130 | 2 | 5.104 | 0.009 | 786 | 5.215 | 0.019 | 0.017 | 1.225 |
| 8 | 1038 | 5.021 | 5.244 | 5.093 | 0.063 | 0.077 | 1675 | 5.191 | 0.162 | 0 | nan | nan | 105 | 5.168 | 0.016 | nan | 1.240 |
| 9 | 1014 | 6.387 | 6.574 | 6.401 | 0.058 | 0.080 | 1363 | 6.507 | 0.159 | 0 | nan | nan | 145 | 6.504 | 0.019 | nan | 2.003 |
| 10 | 0 | 6.367 | 6.563 | 6.382 | 0.058 | 0.082 | 1366 | 6.544 | 0.149 | 2 | 6.391 | 0.033 | 896 | 6.536 | 0.018 | 0.022 | 2.017 |
| 11 | 1282 | 6.351 | 6.553 | 6.367 | 0.058 | 0.085 | 2451 | 6.484 | 0.187 | 0 | nan | nan | 1027 | 6.515 | 0.019 | nan | 2.031 |
| 12 | 746 | 6.302 | 6.519 | 6.323 | 0.057 | 0.087 | 1276 | 6.458 | 0.171 | 127 | 6.349 | 0.159 | 436 | 6.500 | 0.018 | 0.014 | 2.021 |
| 13 | 279 | 6.264 | 6.458 | 6.284 | 0.057 | 0.082 | 556 | 6.393 | 0.144 | 0 | nan | nan | 298 | 6.500 | 0.018 | nan | 1.994 |
| 14 | 0 | 6.274 | 6.620 | 6.517 | 0.904 | 0.980 | 204 | 6.346 | 0.180 | 0 | nan | nan | 243 | 6.441 | 0.017 | nan | 1.983 |
| 15 | 5 | 6.309 | 6.810 | 6.771 | 1.786 | 1.911 | 11 | 6.370 | 0.201 | 1 | 6.337 | 0.000 | 145 | 6.404 | 0.017 | 0.026 | 1.973 |
| 16 | 13 | 6.237 | 6.362 | 6.240 | 0.053 | 0.075 | 26 | 6.317 | 0.120 | 0 | nan | nan | 173 | 6.328 | 0.016 | nan | 1.965 |
| 17 | 190 | 7.658 | 7.838 | 7.672 | 0.045 | 0.066 | 0 | nan | nan | 1 | 7.327 | 0.000 | 66 | 7.468 | nan | 0.016 | 2.162 |
| 19 | 190 | 7.658 | 7.838 | 7.672 | 0.045 | 0.066 | 0 | nan | nan | 1 | 7.573 | 0.000 | 0 | nan | nan | 0.014 | 2.162 |
| 20 | 190 | 7.658 | 7.838 | 7.672 | 0.045 | 0.066 | 0 | nan | nan | 1 | 7.580 | 0.000 | 0 | nan | nan | 0.022 | 2.162 |
| 23 | 190 | 7.658 | 7.838 | 7.672 | 0.045 | 0.066 | 0 | nan | nan | 2 | 7.828 | 0.029 | 0 | nan | nan | 0.011 | 2.162 |
| 24 | 190 | 7.658 | 7.838 | 7.672 | 0.045 | 0.066 | 0 | nan | nan | 238 | 7.728 | 0.232 | 0 | nan | nan | 0.018 | 2.162 |
| 25 | 210 | 8.064 | 8.243 | 8.081 | 0.048 | 0.069 | 0 | nan | nan | 262 | 8.145 | 0.166 | 0 | nan | nan | 0.016 | 2.043 |
| 26 | 0 | 8.145 | 8.333 | 8.165 | 0.047 | 0.068 | 0 | nan | nan | 129 | 8.253 | 0.212 | 0 | nan | nan | 0.019 | 2.069 |
| 27 | 103 | 8.237 | 8.427 | 8.264 | 0.050 | 0.072 | 0 | nan | nan | 128 | 8.314 | 0.309 | 0 | nan | nan | 0.018 | 2.094 |
| 28 | 0 | 8.322 | 8.541 | 8.379 | 0.137 | 0.154 | 0 | nan | nan | 1 | 8.420 | 0.000 | 0 | nan | nan | 0.019 | 2.113 |
| 29 | 0 | 8.408 | 8.656 | 8.494 | 0.241 | 0.261 | 0 | nan | nan | 2 | 8.484 | 0.020 | 0 | nan | nan | 0.023 | 2.132 |
| 30 | 0 | 8.494 | 8.771 | 8.609 | 0.347 | 0.373 | 0 | nan | nan | 127 | 8.480 | 0.147 | 0 | nan | nan | 0.018 | 2.151 |
| 32 | 0 | 8.668 | 9.002 | 8.842 | 0.556 | 0.594 | 0 | nan | nan | 2 | 8.541 | 0.033 | 0 | nan | nan | 0.023 | 2.187 |
| 33 | 0 | 8.757 | 9.117 | 8.959 | 0.663 | 0.702 | 0 | nan | nan | 1 | 9.290 | 0.000 | 0 | nan | nan | 0.021 | 1.874 |
| 34 | 0 | 8.846 | 9.233 | 9.076 | 0.769 | 0.813 | 0 | nan | nan | 3 | 9.100 | 1.426 | 0 | nan | nan | 0.022 | 1.888 |
| 36 | 0 | 9.026 | 9.464 | 9.312 | 0.980 | 1.030 | 0 | nan | nan | 126 | 9.152 | 0.169 | 0 | nan | nan | 0.016 | 1.917 |
| 37 | 0 | 9.116 | 9.580 | 9.431 | 1.087 | 1.142 | 0 | nan | nan | 1 | 9.349 | 0.000 | 0 | nan | nan | 0.019 | 1.931 |
| 39 | 0 | 9.296 | 9.812 | 9.669 | 1.299 | 1.363 | 0 | nan | nan | 2 | 9.600 | 0.078 | 0 | nan | nan | 0.017 | 1.958 |
| 40 | 0 | 9.387 | 9.929 | 9.788 | 1.406 | 1.472 | 0 | nan | nan | 1 | 9.231 | 0.000 | 0 | nan | nan | 0.014 | 1.971 |
| 41 | 0 | 9.477 | 10.045 | 9.908 | 1.512 | 1.581 | 0 | nan | nan | 1 | 9.969 | 0.000 | 0 | nan | nan | 0.022 | 1.859 |
| 42 | 0 | 9.568 | 10.162 | 10.028 | 1.618 | 1.692 | 0 | nan | nan | 1 | 9.939 | 0.000 | 0 | nan | nan | 0.030 | 1.871 |
| 43 | 0 | 9.659 | 10.279 | 10.148 | 1.724 | 1.801 | 0 | nan | nan | 2 | 10.016 | 0.107 | 0 | nan | nan | 0.024 | 1.883 |
| 44 | 0 | 9.749 | 10.395 | 10.268 | 1.830 | 1.909 | 0 | nan | nan | 2 | 9.976 | 0.032 | 0 | nan | nan | 0.022 | 1.894 |
| 45 | 0 | 9.840 | 10.512 | 10.388 | 1.936 | 2.019 | 0 | nan | nan | 1 | 10.057 | 0.000 | 0 | nan | nan | 0.020 | 1.906 |
| 46 | 0 | 9.931 | 10.630 | 10.508 | 2.042 | 2.129 | 0 | nan | nan | 3 | 10.015 | 0.091 | 0 | nan | nan | 0.015 | 1.917 |
| 47 | 5 | 10.022 | 10.747 | 10.628 | 2.148 | 2.238 | 0 | nan | nan | 6 | 10.176 | 5.084 | 0 | nan | nan | 0.018 | 1.928 |
| 48 | 2425 | 9.853 | 10.143 | 9.850 | 0.067 | 0.106 | 0 | nan | nan | 3032 | 9.933 | 0.371 | 0 | nan | nan | 0.018 | 1.884 |
