# Phase 1B sequence-simulation results

params sha256 `8dc79573c1481c7fc2cecb5a095a110dd5fb0d4a09e1e6274f32bd42196a61b8`; coefficients sha256 `78f937c500ff9f205db425a1573789d182df22cf67e8b84de39b22a7ae01c1c9`; partition sha256 `288e193f46323f4d4361913a08c47afa9c4d9f98cc4c4be0d5b88f9629304b7a`; seed 0; draws 32

Modes: recon-gpu (measured segments + measured host times), recon-seq (measured segments and host durations, host starts from the sequence rules), pred-det (fully predicted, median path), pred-mc (fully predicted, stochastic draws). Metrics: n, signed bias (ms, % of measured sum), median abs error, p90 APE (measured >= 0.05 ms), worst abs error.

## Pass `diverse_nsys_run2`

### recon-gpu (17637 evaluated of 17686 iterations, 20.2 s, draws 1)

#### Iteration latency by role and class
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | 1886 | 0.020 | 0.426 | 0.019 | 0.019 | 0.551 | 0.034 | 4.432 |
| calibration | decode b3-16 | 8226 | 0.033 | 0.592 | 0.033 | 0.033 | 0.715 | 0.049 | 5.244 |
| calibration | mixed-eager (host-bound) | 14 | -0.117 | -0.061 | -0.004 | 0.004 | 0.273 | 0.832 | 198.524 |
| calibration | mixed-ring (other) | 10 | 0.001 | 0.000 | -0.002 | 0.015 | 0.020 | 0.059 | 221.437 |
| calibration | prefill 16k chunk | 55 | -0.099 | -0.043 | -0.030 | 0.049 | 0.127 | 1.472 | 227.228 |
| heldout | decode b1-2 | 471 | 0.020 | 0.428 | 0.020 | 0.020 | 0.551 | 0.031 | 4.412 |
| heldout | decode b3-16 | 6935 | 0.034 | 0.572 | 0.034 | 0.034 | 0.668 | 0.050 | 6.393 |
| heldout | mixed-eager (host-bound) | 3 | -0.307 | -0.150 | -0.002 | 0.002 | 0.372 | 0.916 | 202.606 |
| heldout | mixed-ring (other) | 4 | 0.076 | 0.034 | 0.063 | 0.063 | 0.066 | 0.167 | 227.144 |
| heldout | prefill 16k chunk | 33 | -0.149 | -0.064 | -0.029 | 0.029 | 0.036 | 3.436 | 232.178 |


#### By status (extrapolation / inferred / kv-extrapolation reported separately)
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | measured | measured | 1886 | 0.020 | 0.426 | 0.019 | 0.019 | 0.551 | 0.034 | 4.432 |
| calibration | decode b3-16 | measured | measured | 8226 | 0.033 | 0.592 | 0.033 | 0.033 | 0.715 | 0.049 | 5.244 |
| calibration | mixed-eager (host-bound) | measured | measured | 14 | -0.117 | -0.061 | -0.004 | 0.004 | 0.273 | 0.832 | 198.524 |
| calibration | mixed-ring (other) | measured | measured | 10 | 0.001 | 0.000 | -0.002 | 0.015 | 0.020 | 0.059 | 221.437 |
| calibration | prefill 16k chunk | measured | measured | 55 | -0.099 | -0.043 | -0.030 | 0.049 | 0.127 | 1.472 | 227.228 |
| heldout | decode b1-2 | measured | measured | 471 | 0.020 | 0.428 | 0.020 | 0.020 | 0.551 | 0.031 | 4.412 |
| heldout | decode b3-16 | measured | measured | 6935 | 0.034 | 0.572 | 0.034 | 0.034 | 0.668 | 0.050 | 6.393 |
| heldout | mixed-eager (host-bound) | measured | measured | 3 | -0.307 | -0.150 | -0.002 | 0.002 | 0.372 | 0.916 | 202.606 |
| heldout | mixed-ring (other) | measured | measured | 4 | 0.076 | 0.034 | 0.063 | 0.063 | 0.066 | 0.167 | 227.144 |
| heldout | prefill 16k chunk | measured | measured | 33 | -0.149 | -0.064 | -0.029 | 0.029 | 0.036 | 3.436 | 232.178 |


#### Long-context 16k chunks by KV offset
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | kv0 0k | measured | 23 | -0.184 | -0.082 | 0.005 | 0.075 | 0.253 | 1.472 | 224.828 |
| calibration | kv0 16k | measured | 14 | -0.044 | -0.020 | -0.043 | 0.043 | 0.036 | 0.084 | 222.855 |
| calibration | kv0 32k | measured | 8 | -0.015 | -0.007 | -0.016 | 0.021 | 0.016 | 0.056 | 228.509 |
| calibration | kv0 64k | measured | 6 | -0.037 | -0.015 | -0.037 | 0.037 | 0.024 | 0.065 | 237.721 |
| calibration | kv0 80k | measured | 2 | -0.065 | -0.027 | -0.065 | 0.065 | 0.032 | 0.082 | 243.541 |
| calibration | kv0 96k | measured | 2 | -0.044 | -0.018 | -0.044 | 0.044 | 0.029 | 0.079 | 245.807 |
| heldout | kv0 0k | measured | 4 | -0.966 | -0.426 | -0.230 | 0.252 | 1.107 | 3.436 | 228.741 |
| heldout | kv0 112k | measured | 1 | -0.031 | -0.012 | -0.031 | 0.031 | 0.012 | 0.031 | 253.092 |
| heldout | kv0 16k | measured | 4 | -0.040 | -0.018 | -0.038 | 0.038 | 0.029 | 0.069 | 222.705 |
| heldout | kv0 32k | measured | 4 | -0.046 | -0.020 | -0.045 | 0.045 | 0.028 | 0.070 | 228.553 |
| heldout | kv0 48k | measured | 11 | -0.039 | -0.017 | -0.021 | 0.021 | 0.037 | 0.128 | 232.215 |
| heldout | kv0 64k | measured | 4 | -0.028 | -0.012 | -0.030 | 0.030 | 0.020 | 0.047 | 237.623 |
| heldout | kv0 80k | measured | 3 | -0.045 | -0.019 | -0.029 | 0.029 | 0.030 | 0.085 | 243.497 |
| heldout | kv0 96k | measured | 2 | -0.002 | -0.001 | -0.002 | 0.007 | 0.003 | 0.009 | 245.212 |


#### Host lead (GPU end of i-1 minus earliest host start of i) by transition
| key | n | meas_lead_median_ms | pred_lead_median_ms | lead_bias_ms | lead_median_abs_err_ms | lead_p90_abs_err_ms | lead_worst_abs_err_ms | meas_host_skew_median_ms | pred_host_skew_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| decode | decode | calibration | 10079 | 4.581 | 318.174 | 275.637 | 312.304 | 457.207 | 481.201 | 0.013 | 0.013 |
| decode | decode | heldout | 7373 | 5.602 | 196.394 | 208.952 | 191.683 | 399.086 | 481.160 | 0.011 | 0.011 |
| decode | eager | calibration | 10 | 5.077 | 273.024 | 221.494 | 267.557 | 298.876 | 307.729 | 0.029 | 0.029 |
| decode | eager | heldout | 2 | 5.399 | 283.937 | 278.539 | 278.539 | 291.149 | 294.301 | 0.019 | 0.019 |
| decode | ring | calibration | 21 | 2.596 | 197.347 | 186.351 | 194.539 | 307.771 | 319.562 | 0.223 | 0.223 |
| decode | ring | heldout | 3 | 1.867 | 166.834 | 184.631 | 171.062 | 289.518 | 319.132 | 0.554 | 0.554 |
| mixed | decode | calibration | 32 | 15.850 | 224.254 | 185.396 | 209.461 | 306.530 | 321.582 | 16.174 | 16.174 |
| mixed | decode | heldout | 33 | 18.208 | 240.931 | 199.602 | 215.988 | 306.456 | 318.308 | 18.561 | 18.561 |
| mixed | eager | calibration | 4 | 46.539 | 280.188 | 232.280 | 213.894 | 285.374 | 307.676 | 19.084 | 19.084 |
| mixed | eager | heldout | 1 | 17.931 | 300.296 | 282.365 | 282.365 | 282.365 | 282.365 | 18.449 | 18.449 |
| mixed | ring | calibration | 44 | 73.682 | 226.653 | 136.553 | 172.473 | 274.179 | 318.504 | 22.324 | 22.324 |
| mixed | ring | heldout | 34 | 96.306 | 237.701 | 130.871 | 165.380 | 233.469 | 318.437 | 31.642 | 31.642 |


#### Full workload
- span: measured 134.465 s, predicted 134.946 s, error +0.36 % (abs 0.481 s); end drift +481.2 ms, max |drift| 481.2 ms; 49 excluded (JIT) iterations replayed from their measured timelines
- throughput (tok/s): measured decode 964.0, prefill 14398.4; predicted decode 960.5, prefill 14347.1
- Σ walls all: n=17637, measured 124.488 s, predicted 125.029 s, signed bias +0.43 %, abs error 0.541 s, Σ|per-iteration err| 0.569 s; by kind: decode 17518 it. 97.946/98.500 s, eager 17 it. 3.296/3.293 s, ring 102 it. 23.246/23.236 s
- Σ walls calibration: n=10191, measured 72.128 s, predicted 72.431 s, signed bias +0.42 %, abs error 0.303 s, Σ|per-iteration err| 0.319 s; by kind: decode 10112 it. 54.790/55.100 s, eager 14 it. 2.681/2.680 s, ring 65 it. 14.656/14.651 s
- Σ walls heldout: n=7446, measured 52.360 s, predicted 52.598 s, signed bias +0.45 %, abs error 0.238 s, Σ|per-iteration err| 0.250 s; by kind: decode 7406 it. 43.156/43.400 s, eager 3 it. 0.614/0.613 s, ring 37 it. 8.589/8.585 s
- host_start binding constraint counts (draw-summed): {'serial': 77, 'scheduler': 17608, 'first': 1}

### recon-seq (17637 evaluated of 17686 iterations, 20.3 s, draws 1)

#### Iteration latency by role and class
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | 1886 | 0.020 | 0.426 | 0.019 | 0.019 | 0.551 | 0.034 | 4.432 |
| calibration | decode b3-16 | 8226 | 0.033 | 0.592 | 0.033 | 0.033 | 0.715 | 0.049 | 5.244 |
| calibration | mixed-eager (host-bound) | 14 | 3.139 | 1.639 | 3.843 | 3.843 | 2.525 | 5.411 | 198.524 |
| calibration | mixed-ring (other) | 10 | 1.236 | 0.589 | -0.002 | 0.028 | 2.845 | 6.805 | 221.437 |
| calibration | prefill 16k chunk | 55 | 1.674 | 0.733 | -0.014 | 0.065 | 2.616 | 8.401 | 227.228 |
| heldout | decode b1-2 | 471 | 0.020 | 0.428 | 0.020 | 0.020 | 0.551 | 0.031 | 4.412 |
| heldout | decode b3-16 | 6935 | 0.034 | 0.572 | 0.034 | 0.034 | 0.668 | 0.050 | 6.393 |
| heldout | mixed-eager (host-bound) | 3 | 2.774 | 1.355 | 3.876 | 3.876 | 1.977 | 4.092 | 202.606 |
| heldout | mixed-ring (other) | 4 | 1.322 | 0.592 | 0.136 | 0.136 | 1.716 | 4.994 | 227.144 |
| heldout | prefill 16k chunk | 33 | 0.380 | 0.163 | -0.023 | 0.031 | 0.051 | 8.043 | 232.178 |


#### By status (extrapolation / inferred / kv-extrapolation reported separately)
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | measured | measured | 1886 | 0.020 | 0.426 | 0.019 | 0.019 | 0.551 | 0.034 | 4.432 |
| calibration | decode b3-16 | measured | measured | 8226 | 0.033 | 0.592 | 0.033 | 0.033 | 0.715 | 0.049 | 5.244 |
| calibration | mixed-eager (host-bound) | measured | measured | 14 | 3.139 | 1.639 | 3.843 | 3.843 | 2.525 | 5.411 | 198.524 |
| calibration | mixed-ring (other) | measured | measured | 10 | 1.236 | 0.589 | -0.002 | 0.028 | 2.845 | 6.805 | 221.437 |
| calibration | prefill 16k chunk | measured | measured | 55 | 1.674 | 0.733 | -0.014 | 0.065 | 2.616 | 8.401 | 227.228 |
| heldout | decode b1-2 | measured | measured | 471 | 0.020 | 0.428 | 0.020 | 0.020 | 0.551 | 0.031 | 4.412 |
| heldout | decode b3-16 | measured | measured | 6935 | 0.034 | 0.572 | 0.034 | 0.034 | 0.668 | 0.050 | 6.393 |
| heldout | mixed-eager (host-bound) | measured | measured | 3 | 2.774 | 1.355 | 3.876 | 3.876 | 1.977 | 4.092 | 202.606 |
| heldout | mixed-ring (other) | measured | measured | 4 | 1.322 | 0.592 | 0.136 | 0.136 | 1.716 | 4.994 | 227.144 |
| heldout | prefill 16k chunk | measured | measured | 33 | 0.380 | 0.163 | -0.023 | 0.031 | 0.051 | 8.043 | 232.178 |


#### Long-context 16k chunks by KV offset
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | kv0 0k | measured | 23 | 4.055 | 1.795 | 5.277 | 5.277 | 2.861 | 8.401 | 224.828 |
| calibration | kv0 16k | measured | 14 | -0.044 | -0.020 | -0.043 | 0.043 | 0.036 | 0.084 | 222.855 |
| calibration | kv0 32k | measured | 8 | -0.015 | -0.007 | -0.016 | 0.021 | 0.016 | 0.056 | 228.509 |
| calibration | kv0 64k | measured | 6 | -0.037 | -0.015 | -0.037 | 0.037 | 0.024 | 0.065 | 237.721 |
| calibration | kv0 80k | measured | 2 | -0.065 | -0.027 | -0.065 | 0.065 | 0.032 | 0.082 | 243.541 |
| calibration | kv0 96k | measured | 2 | -0.044 | -0.018 | -0.044 | 0.044 | 0.029 | 0.079 | 245.807 |
| heldout | kv0 0k | measured | 4 | 3.402 | 1.501 | 2.775 | 2.775 | 2.847 | 8.043 | 228.741 |
| heldout | kv0 112k | measured | 1 | -0.031 | -0.012 | -0.031 | 0.031 | 0.012 | 0.031 | 253.092 |
| heldout | kv0 16k | measured | 4 | -0.040 | -0.018 | -0.038 | 0.038 | 0.029 | 0.069 | 222.705 |
| heldout | kv0 32k | measured | 4 | -0.046 | -0.020 | -0.045 | 0.045 | 0.028 | 0.070 | 228.553 |
| heldout | kv0 48k | measured | 11 | -0.039 | -0.017 | -0.021 | 0.021 | 0.037 | 0.128 | 232.215 |
| heldout | kv0 64k | measured | 4 | -0.028 | -0.012 | -0.030 | 0.030 | 0.020 | 0.047 | 237.623 |
| heldout | kv0 80k | measured | 3 | -0.045 | -0.019 | -0.029 | 0.029 | 0.030 | 0.085 | 243.497 |
| heldout | kv0 96k | measured | 2 | -0.002 | -0.001 | -0.002 | 0.007 | 0.003 | 0.009 | 245.212 |


#### Host lead (GPU end of i-1 minus earliest host start of i) by transition
| key | n | meas_lead_median_ms | pred_lead_median_ms | lead_bias_ms | lead_median_abs_err_ms | lead_p90_abs_err_ms | lead_worst_abs_err_ms | meas_host_skew_median_ms | pred_host_skew_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| decode | decode | calibration | 10079 | 4.581 | 4.605 | -0.395 | 0.042 | 2.331 | 9.279 | 0.013 | 0.000 |
| decode | decode | heldout | 7373 | 5.602 | 5.632 | -0.105 | 0.044 | 0.123 | 12.844 | 0.011 | 0.000 |
| decode | eager | calibration | 10 | 5.077 | 4.034 | -1.219 | 1.292 | 1.517 | 1.548 | 0.029 | 0.000 |
| decode | eager | heldout | 2 | 5.399 | 4.086 | -1.313 | 1.313 | 1.323 | 1.325 | 0.019 | 0.000 |
| decode | ring | calibration | 21 | 2.596 | 2.737 | 0.832 | 0.901 | 4.162 | 5.210 | 0.223 | 0.000 |
| decode | ring | heldout | 3 | 1.867 | 1.374 | 3.009 | 5.524 | 5.530 | 5.532 | 0.554 | 0.000 |
| mixed | decode | calibration | 32 | 15.850 | 18.088 | 1.400 | 2.873 | 15.282 | 28.945 | 16.174 | 14.706 |
| mixed | decode | heldout | 33 | 18.208 | 19.749 | 3.084 | 2.056 | 8.245 | 28.686 | 18.561 | 15.457 |
| mixed | eager | calibration | 4 | 46.539 | 47.929 | 4.705 | 5.418 | 9.249 | 10.810 | 19.084 | 24.867 |
| mixed | eager | heldout | 1 | 17.931 | 6.764 | -11.167 | 11.167 | 11.167 | 11.167 | 18.449 | 0.182 |
| mixed | ring | calibration | 44 | 73.682 | 69.062 | 0.831 | 8.575 | 22.438 | 29.121 | 22.324 | 20.352 |
| mixed | ring | heldout | 34 | 96.306 | 103.507 | 4.474 | 10.470 | 27.115 | 28.985 | 31.642 | 30.103 |


#### Full workload
- span: measured 134.465 s, predicted 135.397 s, error +0.69 % (abs 0.933 s); end drift +932.6 ms, max |drift| 932.6 ms; 49 excluded (JIT) iterations replayed from their measured timelines
- throughput (tok/s): measured decode 964.0, prefill 14398.4; predicted decode 957.3, prefill 14299.2
- Σ walls all: n=17637, measured 124.488 s, predicted 125.216 s, signed bias +0.58 %, abs error 0.728 s, Σ|per-iteration err| 0.737 s; by kind: decode 17518 it. 97.946/98.500 s, eager 17 it. 3.296/3.348 s, ring 102 it. 23.246/23.368 s
- Σ walls calibration: n=10191, measured 72.128 s, predicted 72.586 s, signed bias +0.64 %, abs error 0.458 s, Σ|per-iteration err| 0.465 s; by kind: decode 10112 it. 54.790/55.100 s, eager 14 it. 2.681/2.725 s, ring 65 it. 14.656/14.761 s
- Σ walls heldout: n=7446, measured 52.360 s, predicted 52.630 s, signed bias +0.52 %, abs error 0.270 s, Σ|per-iteration err| 0.272 s; by kind: decode 7406 it. 43.156/43.400 s, eager 3 it. 0.614/0.623 s, ring 37 it. 8.589/8.607 s
- host_start binding constraint counts (draw-summed): {'serial': 1930, 'scheduler': 15755, 'first': 1}

### pred-det (17637 evaluated of 17686 iterations, 23.1 s, draws 1)

#### Iteration latency by role and class
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | 1886 | 0.012 | 0.266 | 0.119 | 0.279 | 15.871 | 31.046 | 4.432 |
| calibration | decode b3-16 | 8226 | -0.152 | -2.702 | -0.136 | 0.136 | 4.210 | 33.162 | 5.244 |
| calibration | mixed-eager (host-bound) | 14 | 1.627 | 0.850 | 2.841 | 7.549 | 11.947 | 22.868 | 198.524 |
| calibration | mixed-ring (other) | 10 | 4.582 | 2.183 | 1.684 | 3.724 | 7.167 | 32.760 | 221.437 |
| calibration | prefill 16k chunk | 55 | 0.572 | 0.251 | 0.203 | 0.686 | 1.539 | 8.587 | 227.228 |
| heldout | decode b1-2 | 471 | 0.022 | 0.476 | 0.122 | 0.271 | 15.246 | 19.770 | 4.412 |
| heldout | decode b3-16 | 6935 | -0.167 | -2.829 | -0.152 | 0.153 | 4.690 | 14.652 | 6.393 |
| heldout | mixed-eager (host-bound) | 3 | -3.455 | -1.687 | -1.460 | 5.293 | 5.833 | 14.199 | 202.606 |
| heldout | mixed-ring (other) | 4 | -0.806 | -0.361 | -1.306 | 4.389 | 2.671 | 6.058 | 227.144 |
| heldout | prefill 16k chunk | 33 | -0.019 | -0.008 | 0.024 | 0.458 | 0.564 | 10.947 | 232.178 |


#### By status (extrapolation / inferred / kv-extrapolation reported separately)
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | measured | in-range | 1886 | 0.012 | 0.266 | 0.119 | 0.279 | 15.871 | 31.046 | 4.432 |
| calibration | decode b3-16 | measured | in-range | 8226 | -0.152 | -2.702 | -0.136 | 0.136 | 4.210 | 33.162 | 5.244 |
| calibration | mixed-eager (host-bound) | in-range | in-range | 12 | 0.984 | 0.487 | 2.841 | 5.343 | 4.857 | 21.867 | 198.809 |
| calibration | mixed-eager (host-bound) | inferred | in-range | 2 | 5.486 | 4.266 | 5.486 | 17.382 | 13.775 | 22.868 | 128.594 |
| calibration | mixed-ring (other) | in-range | in-range | 2 | 9.016 | 4.473 | 9.016 | 9.016 | 5.533 | 11.540 | 201.551 |
| calibration | mixed-ring (other) | inferred | in-range | 8 | 3.473 | 1.638 | 0.423 | 2.790 | 7.977 | 32.760 | 224.261 |
| calibration | prefill 16k chunk | in-range | in-range | 20 | 1.498 | 0.661 | 2.236 | 3.133 | 2.227 | 8.587 | 225.707 |
| calibration | prefill 16k chunk | inferred | in-range | 35 | 0.043 | 0.019 | 0.048 | 0.421 | 0.498 | 3.975 | 227.924 |
| heldout | decode b1-2 | measured | in-range | 471 | 0.022 | 0.476 | 0.122 | 0.271 | 15.246 | 19.770 | 4.412 |
| heldout | decode b3-16 | interpolated | in-range | 2227 | -0.207 | -3.387 | -0.187 | 0.188 | 6.209 | 4.822 | 6.496 |
| heldout | decode b3-16 | interpolated | kv-extrapolation | 5 | 0.434 | 6.004 | -0.070 | 0.119 | 15.108 | 2.546 | 6.400 |
| heldout | decode b3-16 | measured | in-range | 4703 | -0.149 | -2.563 | -0.140 | 0.140 | 3.957 | 14.652 | 6.339 |
| heldout | mixed-eager (host-bound) | extrapolation | kv-extrapolation | 2 | -4.453 | -2.163 | -4.453 | 9.746 | 6.227 | 14.199 | 205.854 |
| heldout | mixed-eager (host-bound) | in-range | in-range | 1 | -1.460 | -0.721 | -1.460 | 1.460 | 0.721 | 1.460 | 202.606 |
| heldout | mixed-ring (other) | in-range | in-range | 1 | 5.445 | 2.638 | 5.445 | 5.445 | 2.638 | 5.445 | 206.401 |
| heldout | mixed-ring (other) | inferred | in-range | 3 | -2.890 | -1.261 | -3.334 | 3.334 | 2.440 | 6.058 | 228.735 |
| heldout | prefill 16k chunk | extrapolation | kv-extrapolation | 2 | 1.077 | 0.432 | 1.077 | 1.440 | 0.938 | 2.517 | 249.171 |
| heldout | prefill 16k chunk | in-range | in-range | 3 | -3.747 | -1.635 | -0.228 | 0.228 | 3.824 | 10.947 | 229.682 |
| heldout | prefill 16k chunk | inferred | in-range | 28 | 0.302 | 0.130 | 0.185 | 0.517 | 0.470 | 2.609 | 232.196 |


#### Long-context 16k chunks by KV offset
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | kv0 0k | in-range | 20 | 1.498 | 0.661 | 2.236 | 3.133 | 2.227 | 8.587 | 225.707 |
| calibration | kv0 0k | inferred | 3 | 0.009 | 0.004 | -0.421 | 0.490 | 0.382 | 0.937 | 219.408 |
| calibration | kv0 16k | inferred | 14 | 0.031 | 0.014 | 0.368 | 0.435 | 0.500 | 3.975 | 222.855 |
| calibration | kv0 32k | inferred | 8 | -0.314 | -0.137 | -0.413 | 0.413 | 0.238 | 0.692 | 228.509 |
| calibration | kv0 64k | inferred | 6 | 0.175 | 0.074 | 0.042 | 0.218 | 0.283 | 0.942 | 237.721 |
| calibration | kv0 80k | inferred | 2 | -0.718 | -0.295 | -0.718 | 0.718 | 0.412 | 1.076 | 243.541 |
| calibration | kv0 96k | inferred | 2 | 1.969 | 0.801 | 1.969 | 1.969 | 0.947 | 2.412 | 245.807 |
| heldout | kv0 0k | in-range | 3 | -3.747 | -1.635 | -0.228 | 0.228 | 3.824 | 10.947 | 229.682 |
| heldout | kv0 0k | inferred | 1 | -0.178 | -0.081 | -0.178 | 0.178 | 0.081 | 0.178 | 219.169 |
| heldout | kv0 112k | extrapolation | 1 | -0.363 | -0.143 | -0.363 | 0.363 | 0.143 | 0.363 | 253.092 |
| heldout | kv0 16k | inferred | 4 | 0.439 | 0.197 | 0.575 | 0.701 | 0.455 | 1.021 | 222.705 |
| heldout | kv0 32k | inferred | 4 | -0.212 | -0.093 | -0.141 | 0.347 | 0.349 | 0.968 | 228.553 |
| heldout | kv0 48k | inferred | 11 | 0.468 | 0.201 | 0.652 | 0.718 | 0.494 | 1.348 | 232.215 |
| heldout | kv0 64k | inferred | 4 | 0.311 | 0.131 | 0.273 | 0.273 | 0.247 | 0.692 | 237.623 |
| heldout | kv0 80k | inferred | 3 | -0.426 | -0.175 | -0.458 | 0.458 | 0.259 | 0.673 | 243.497 |
| heldout | kv0 96k | extrapolation | 1 | 2.517 | 1.026 | 2.517 | 2.517 | 1.026 | 2.517 | 245.250 |
| heldout | kv0 96k | inferred | 1 | 2.609 | 1.064 | 2.609 | 2.609 | 1.064 | 2.609 | 245.173 |


#### Host lead (GPU end of i-1 minus earliest host start of i) by transition
| key | n | meas_lead_median_ms | pred_lead_median_ms | lead_bias_ms | lead_median_abs_err_ms | lead_p90_abs_err_ms | lead_worst_abs_err_ms | meas_host_skew_median_ms | pred_host_skew_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| decode | decode | calibration | 10079 | 4.581 | 4.398 | -0.241 | 0.150 | 0.379 | 27.535 | 0.013 | 0.000 |
| decode | decode | heldout | 7373 | 5.602 | 5.492 | -0.186 | 0.152 | 0.282 | 18.976 | 0.011 | 0.000 |
| decode | eager | calibration | 10 | 5.077 | 3.921 | -1.541 | 1.504 | 1.964 | 2.388 | 0.029 | 0.000 |
| decode | eager | heldout | 2 | 5.399 | 3.940 | -1.459 | 1.459 | 1.476 | 1.480 | 0.019 | 0.000 |
| decode | ring | calibration | 21 | 2.596 | 2.557 | 0.059 | 1.390 | 3.800 | 4.964 | 0.223 | 0.000 |
| decode | ring | heldout | 3 | 1.867 | 1.211 | 1.207 | 2.193 | 4.555 | 5.145 | 0.554 | 0.000 |
| mixed | decode | calibration | 32 | 15.850 | 14.637 | -0.379 | 4.121 | 9.328 | 22.310 | 16.174 | 13.561 |
| mixed | decode | heldout | 33 | 18.208 | 16.152 | 3.886 | 2.602 | 17.727 | 71.179 | 18.561 | 14.190 |
| mixed | eager | calibration | 4 | 46.539 | 38.190 | 1.671 | 11.691 | 20.589 | 22.373 | 19.084 | 23.053 |
| mixed | eager | heldout | 1 | 17.931 | 10.501 | -7.430 | 7.430 | 7.430 | 7.430 | 18.449 | 9.376 |
| mixed | ring | calibration | 44 | 73.682 | 84.393 | 1.412 | 12.446 | 22.803 | 55.373 | 22.324 | 28.445 |
| mixed | ring | heldout | 34 | 96.306 | 98.697 | -1.384 | 18.753 | 25.010 | 43.249 | 31.642 | 42.666 |


#### Full workload
- span: measured 134.465 s, predicted 132.386 s, error -1.55 % (abs 2.078 s); end drift -2078.4 ms, max |drift| 2278.5 ms; 49 excluded (JIT) iterations replayed from their measured timelines
- throughput (tok/s): measured decode 964.0, prefill 14398.4; predicted decode 979.1, prefill 14624.5
- Σ walls all: n=17637, measured 124.488 s, predicted 122.200 s, signed bias -1.84 %, abs error 2.288 s, Σ|per-iteration err| 3.833 s; by kind: decode 17518 it. 97.946/95.573 s, eager 17 it. 3.296/3.308 s, ring 102 it. 23.246/23.319 s
- Σ walls calibration: n=10191, measured 72.128 s, predicted 71.004 s, signed bias -1.56 %, abs error 1.124 s, Σ|per-iteration err| 2.347 s; by kind: decode 10112 it. 54.790/53.566 s, eager 14 it. 2.681/2.704 s, ring 65 it. 14.656/14.734 s
- Σ walls heldout: n=7446, measured 52.360 s, predicted 51.196 s, signed bias -2.22 %, abs error 1.164 s, Σ|per-iteration err| 1.487 s; by kind: decode 7406 it. 43.156/42.007 s, eager 3 it. 0.614/0.604 s, ring 37 it. 8.589/8.585 s
- host_start binding constraint counts (draw-summed): {'serial': 171, 'scheduler': 17514, 'first': 1}

### pred-mc (17637 evaluated of 17686 iterations, 25.3 s, draws 32)

#### Iteration latency by role and class
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | 1886 | 0.081 | 1.762 | 0.166 | 0.286 | 18.016 | 31.096 | 4.432 |
| calibration | decode b3-16 | 8226 | 0.039 | 0.697 | 0.056 | 0.063 | 2.375 | 33.019 | 5.244 |
| calibration | mixed-eager (host-bound) | 14 | 1.435 | 0.749 | 2.636 | 7.458 | 12.148 | 22.477 | 198.524 |
| calibration | mixed-ring (other) | 10 | 4.624 | 2.203 | 1.735 | 4.020 | 7.065 | 32.471 | 221.437 |
| calibration | prefill 16k chunk | 55 | 0.629 | 0.275 | 0.370 | 1.180 | 1.446 | 8.875 | 227.228 |
| heldout | decode b1-2 | 471 | 0.094 | 2.051 | 0.168 | 0.270 | 17.514 | 19.140 | 4.412 |
| heldout | decode b3-16 | 6935 | 0.031 | 0.522 | 0.043 | 0.063 | 2.688 | 14.442 | 6.393 |
| heldout | mixed-eager (host-bound) | 3 | -3.597 | -1.757 | -1.673 | 5.293 | 5.913 | 14.412 | 202.606 |
| heldout | mixed-ring (other) | 4 | -0.633 | -0.283 | -0.883 | 4.181 | 2.617 | 5.986 | 227.144 |
| heldout | prefill 16k chunk | 33 | 0.047 | 0.020 | 0.192 | 0.649 | 0.754 | 6.626 | 232.178 |


#### By status (extrapolation / inferred / kv-extrapolation reported separately)
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | measured | in-range | 1886 | 0.081 | 1.762 | 0.166 | 0.286 | 18.016 | 31.096 | 4.432 |
| calibration | decode b3-16 | measured | in-range | 8226 | 0.039 | 0.697 | 0.056 | 0.063 | 2.375 | 33.019 | 5.244 |
| calibration | mixed-eager (host-bound) | in-range | in-range | 12 | 0.812 | 0.402 | 2.636 | 5.161 | 4.773 | 22.054 | 198.809 |
| calibration | mixed-eager (host-bound) | inferred | in-range | 2 | 5.175 | 4.024 | 5.175 | 17.302 | 13.587 | 22.477 | 128.594 |
| calibration | mixed-ring (other) | in-range | in-range | 2 | 8.828 | 4.380 | 8.828 | 8.828 | 5.439 | 11.352 | 201.551 |
| calibration | mixed-ring (other) | inferred | in-range | 8 | 3.573 | 1.685 | 0.464 | 3.142 | 7.885 | 32.471 | 224.261 |
| calibration | prefill 16k chunk | in-range | in-range | 20 | 1.306 | 0.576 | 1.811 | 2.440 | 1.995 | 8.875 | 225.707 |
| calibration | prefill 16k chunk | inferred | in-range | 35 | 0.242 | 0.105 | 0.302 | 0.920 | 0.775 | 4.150 | 227.924 |
| heldout | decode b1-2 | measured | in-range | 471 | 0.094 | 2.051 | 0.168 | 0.270 | 17.514 | 19.140 | 4.412 |
| heldout | decode b3-16 | interpolated | in-range | 2227 | -0.007 | -0.123 | -0.004 | 0.058 | 3.582 | 4.651 | 6.496 |
| heldout | decode b3-16 | interpolated | kv-extrapolation | 5 | 0.792 | 10.953 | 0.260 | 0.260 | 19.164 | 2.923 | 6.400 |
| heldout | decode b3-16 | measured | in-range | 4703 | 0.048 | 0.828 | 0.060 | 0.065 | 2.453 | 14.442 | 6.339 |
| heldout | mixed-eager (host-bound) | extrapolation | kv-extrapolation | 2 | -4.560 | -2.215 | -4.560 | 9.852 | 6.316 | 14.412 | 205.854 |
| heldout | mixed-eager (host-bound) | in-range | in-range | 1 | -1.673 | -0.826 | -1.673 | 1.673 | 0.826 | 1.673 | 202.606 |
| heldout | mixed-ring (other) | in-range | in-range | 1 | 5.222 | 2.530 | 5.222 | 5.222 | 2.530 | 5.222 | 206.401 |
| heldout | mixed-ring (other) | inferred | in-range | 3 | -2.584 | -1.128 | -3.141 | 3.141 | 2.398 | 5.986 | 228.735 |
| heldout | prefill 16k chunk | extrapolation | kv-extrapolation | 2 | 1.058 | 0.425 | 1.058 | 1.058 | 0.679 | 1.818 | 249.171 |
| heldout | prefill 16k chunk | in-range | in-range | 3 | -2.612 | -1.140 | -0.608 | 0.608 | 2.355 | 6.626 | 229.682 |
| heldout | prefill 16k chunk | inferred | in-range | 28 | 0.260 | 0.112 | 0.268 | 0.745 | 0.665 | 2.384 | 232.196 |


#### Long-context 16k chunks by KV offset
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | kv0 0k | in-range | 20 | 1.306 | 0.576 | 1.811 | 2.440 | 1.995 | 8.875 | 225.707 |
| calibration | kv0 0k | inferred | 3 | 0.055 | 0.025 | -0.242 | 0.655 | 0.443 | 1.063 | 219.408 |
| calibration | kv0 16k | inferred | 14 | 0.728 | 0.326 | 1.048 | 1.291 | 1.118 | 4.150 | 222.855 |
| calibration | kv0 32k | inferred | 8 | -0.507 | -0.222 | -0.288 | 0.343 | 0.540 | 1.686 | 228.509 |
| calibration | kv0 64k | inferred | 6 | -0.004 | -0.002 | -0.004 | 0.378 | 0.469 | 1.180 | 237.721 |
| calibration | kv0 80k | inferred | 2 | -0.620 | -0.255 | -0.620 | 1.217 | 0.702 | 1.837 | 243.541 |
| calibration | kv0 96k | inferred | 2 | 1.714 | 0.697 | 1.714 | 1.714 | 0.805 | 2.040 | 245.807 |
| heldout | kv0 0k | in-range | 3 | -2.612 | -1.140 | -0.608 | 0.608 | 2.355 | 6.626 | 229.682 |
| heldout | kv0 0k | inferred | 1 | -1.258 | -0.574 | -1.258 | 1.258 | 0.574 | 1.258 | 219.169 |
| heldout | kv0 112k | extrapolation | 1 | 0.298 | 0.118 | 0.298 | 0.298 | 0.118 | 0.298 | 253.092 |
| heldout | kv0 16k | inferred | 4 | 1.022 | 0.458 | 0.791 | 0.791 | 0.938 | 2.384 | 222.705 |
| heldout | kv0 32k | inferred | 4 | -0.638 | -0.279 | -0.581 | 0.612 | 0.645 | 1.734 | 228.553 |
| heldout | kv0 48k | inferred | 11 | 0.541 | 0.232 | 0.534 | 0.588 | 0.549 | 1.324 | 232.215 |
| heldout | kv0 64k | inferred | 4 | -0.042 | -0.018 | 0.214 | 0.647 | 0.448 | 1.245 | 237.623 |
| heldout | kv0 80k | inferred | 3 | -0.276 | -0.113 | -0.842 | 1.029 | 0.427 | 1.044 | 243.497 |
| heldout | kv0 96k | extrapolation | 1 | 1.818 | 0.741 | 1.818 | 1.818 | 0.741 | 1.818 | 245.250 |
| heldout | kv0 96k | inferred | 1 | 2.057 | 0.839 | 2.057 | 2.057 | 0.839 | 2.057 | 245.173 |


#### Host lead (GPU end of i-1 minus earliest host start of i) by transition
| key | n | meas_lead_median_ms | pred_lead_median_ms | lead_bias_ms | lead_median_abs_err_ms | lead_p90_abs_err_ms | lead_worst_abs_err_ms | meas_host_skew_median_ms | pred_host_skew_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| decode | decode | calibration | 10079 | 4.581 | 4.621 | -0.083 | 0.073 | 0.410 | 27.659 | 0.013 | 0.000 |
| decode | decode | heldout | 7373 | 5.602 | 5.729 | 0.000 | 0.073 | 0.210 | 18.361 | 0.011 | 0.000 |
| decode | eager | calibration | 10 | 5.077 | 4.106 | -1.362 | 1.301 | 1.788 | 2.319 | 0.029 | 0.000 |
| decode | eager | heldout | 2 | 5.399 | 4.153 | -1.245 | 1.245 | 1.263 | 1.267 | 0.019 | 0.000 |
| decode | ring | calibration | 21 | 2.596 | 2.737 | 0.251 | 1.584 | 4.018 | 5.135 | 0.223 | 0.000 |
| decode | ring | heldout | 3 | 1.867 | 1.434 | 1.400 | 1.970 | 4.658 | 5.330 | 0.554 | 0.000 |
| mixed | decode | calibration | 32 | 15.850 | 14.637 | -0.331 | 4.121 | 9.328 | 22.658 | 16.174 | 13.561 |
| mixed | decode | heldout | 33 | 18.208 | 16.152 | 3.833 | 2.602 | 17.743 | 70.604 | 18.561 | 14.190 |
| mixed | eager | calibration | 4 | 46.539 | 38.386 | 1.858 | 11.869 | 20.422 | 21.982 | 19.084 | 23.053 |
| mixed | eager | heldout | 1 | 17.931 | 10.501 | -7.430 | 7.430 | 7.430 | 7.430 | 18.449 | 9.376 |
| mixed | ring | calibration | 44 | 73.682 | 83.963 | 1.587 | 11.864 | 22.869 | 54.798 | 22.324 | 28.445 |
| mixed | ring | heldout | 34 | 96.306 | 99.235 | -0.695 | 18.940 | 23.690 | 43.306 | 31.642 | 42.666 |


#### Full workload
- span: measured 134.465 s, predicted 135.477 s, error +0.75 % (abs 1.013 s); end drift +1012.6 ms, max |drift| 1012.6 ms; 49 excluded (JIT) iterations replayed from their measured timelines
- throughput (tok/s): measured decode 964.0, prefill 14398.4; predicted decode 956.8, prefill 14290.8
- Σ walls all: n=17637, measured 124.488 s, predicted 125.309 s, signed bias +0.66 %, abs error 0.821 s, Σ|per-iteration err| 2.685 s; by kind: decode 17518 it. 97.946/98.678 s, eager 17 it. 3.296/3.305 s, ring 102 it. 23.246/23.326 s
- Σ walls calibration: n=10191, measured 72.128 s, predicted 72.702 s, signed bias +0.80 %, abs error 0.575 s, Σ|per-iteration err| 1.764 s; by kind: decode 10112 it. 54.790/55.264 s, eager 14 it. 2.681/2.701 s, ring 65 it. 14.656/14.737 s
- Σ walls heldout: n=7446, measured 52.360 s, predicted 52.606 s, signed bias +0.47 %, abs error 0.246 s, Σ|per-iteration err| 0.922 s; by kind: decode 7406 it. 43.156/43.414 s, eager 3 it. 0.614/0.604 s, ring 37 it. 8.589/8.588 s
- host_start binding constraint counts (draw-summed): {'serial': 22239, 'scheduler': 543681, 'first': 32}

#### Uncertainty coverage (fraction of measured walls inside the predicted band)
| key | n | cover_p10_p90_pct | cover_p05_p95_pct | band_p10_p90_median_ms | band_rel_median_pct | below_pct | above_pct |
|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | 1886 | 85.631 | 93.425 | 1.252 | 27.794 | 8.006 | 6.363 |
| calibration | decode b3-16 | 8226 | 61.999 | 72.526 | 0.167 | 2.975 | 34.841 | 3.161 |
| calibration | mixed-eager (host-bound) | 14 | 14.286 | 14.286 | 0.179 | 0.090 | 57.143 | 28.571 |
| calibration | mixed-ring (other) | 10 | 50.000 | 60.000 | 6.820 | 3.842 | 30.000 | 20.000 |
| calibration | prefill 16k chunk | 55 | 89.091 | 92.727 | 7.302 | 3.176 | 9.091 | 1.818 |
| heldout | decode b1-2 | 471 | 84.926 | 93.631 | 1.256 | 27.513 | 8.917 | 6.157 |
| heldout | decode b3-16 | 6935 | 57.751 | 67.541 | 0.158 | 2.763 | 30.411 | 11.839 |
| heldout | mixed-eager (host-bound) | 3 | 0.000 | 0.000 | 0.141 | 0.066 | 33.333 | 66.667 |
| heldout | mixed-ring (other) | 4 | 50.000 | 50.000 | 6.061 | 2.643 | 25.000 | 25.000 |
| heldout | prefill 16k chunk | 33 | 100.000 | 100.000 | 8.180 | 3.583 | 0.000 | 0.000 |


## Pass `quadrant_nsys`

### recon-gpu (4740 evaluated of 4750 iterations, 3.5 s, draws 1)

#### Iteration latency by role and class
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | 208 | 0.021 | 0.435 | 0.020 | 0.020 | 0.555 | 0.033 | 4.783 |
| calibration | decode b17-24 | 194 | -0.001 | -0.011 | -0.001 | 0.001 | 0.035 | 0.005 | 7.726 |
| calibration | decode b3-16 | 311 | 0.032 | 0.603 | 0.032 | 0.032 | 0.755 | 0.044 | 5.042 |
| calibration | decode b>=25 | 2762 | -0.000 | -0.003 | -0.000 | 0.001 | 0.025 | 0.009 | 9.910 |
| calibration | mixed-eager (host-bound) | 3 | -0.381 | -0.210 | -0.006 | 0.006 | 0.433 | 1.133 | 196.173 |
| calibration | mixed-graph | 4 | -0.000 | -0.001 | -0.000 | 0.002 | 0.005 | 0.002 | 36.723 |
| calibration | mixed-ring (other) | 5 | 2.393 | 1.064 | 0.126 | 0.126 | 2.917 | 11.511 | 221.369 |
| heldout | decode b1-2 | 53 | 0.022 | 0.450 | 0.023 | 0.023 | 0.569 | 0.034 | 4.832 |
| heldout | decode b17-24 | 49 | -0.001 | -0.011 | -0.001 | 0.001 | 0.035 | 0.004 | 7.739 |
| heldout | decode b3-16 | 78 | 0.031 | 0.585 | 0.031 | 0.031 | 0.749 | 0.041 | 5.043 |
| heldout | decode b>=25 | 1071 | -0.000 | -0.001 | -0.000 | 0.001 | 0.024 | 0.006 | 9.752 |
| heldout | mixed-eager (host-bound) | 1 | -0.001 | -0.001 | -0.001 | 0.001 | 0.001 | 0.001 | 200.613 |
| heldout | mixed-ring (other) | 1 | 0.061 | 0.028 | 0.061 | 0.061 | 0.028 | 0.061 | 221.700 |


#### By status (extrapolation / inferred / kv-extrapolation reported separately)
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | measured | measured | 208 | 0.021 | 0.435 | 0.020 | 0.020 | 0.555 | 0.033 | 4.783 |
| calibration | decode b17-24 | measured | measured | 194 | -0.001 | -0.011 | -0.001 | 0.001 | 0.035 | 0.005 | 7.726 |
| calibration | decode b3-16 | measured | measured | 311 | 0.032 | 0.603 | 0.032 | 0.032 | 0.755 | 0.044 | 5.042 |
| calibration | decode b>=25 | measured | measured | 2762 | -0.000 | -0.003 | -0.000 | 0.001 | 0.025 | 0.009 | 9.910 |
| calibration | mixed-eager (host-bound) | measured | measured | 3 | -0.381 | -0.210 | -0.006 | 0.006 | 0.433 | 1.133 | 196.173 |
| calibration | mixed-graph | measured | measured | 4 | -0.000 | -0.001 | -0.000 | 0.002 | 0.005 | 0.002 | 36.723 |
| calibration | mixed-ring (other) | measured | measured | 5 | 2.393 | 1.064 | 0.126 | 0.126 | 2.917 | 11.511 | 221.369 |
| heldout | decode b1-2 | measured | measured | 53 | 0.022 | 0.450 | 0.023 | 0.023 | 0.569 | 0.034 | 4.832 |
| heldout | decode b17-24 | measured | measured | 49 | -0.001 | -0.011 | -0.001 | 0.001 | 0.035 | 0.004 | 7.739 |
| heldout | decode b3-16 | measured | measured | 78 | 0.031 | 0.585 | 0.031 | 0.031 | 0.749 | 0.041 | 5.043 |
| heldout | decode b>=25 | measured | measured | 1071 | -0.000 | -0.001 | -0.000 | 0.001 | 0.024 | 0.006 | 9.752 |
| heldout | mixed-eager (host-bound) | measured | measured | 1 | -0.001 | -0.001 | -0.001 | 0.001 | 0.001 | 0.001 | 200.613 |
| heldout | mixed-ring (other) | measured | measured | 1 | 0.061 | 0.028 | 0.061 | 0.061 | 0.028 | 0.061 | 221.700 |


#### Host lead (GPU end of i-1 minus earliest host start of i) by transition
| key | n | meas_lead_median_ms | pred_lead_median_ms | lead_bias_ms | lead_median_abs_err_ms | lead_p90_abs_err_ms | lead_worst_abs_err_ms | meas_host_skew_median_ms | pred_host_skew_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| decode | decode | calibration | 3466 | 8.081 | 19.702 | 9.012 | 12.446 | 20.845 | 26.547 | 0.014 | 0.014 |
| decode | decode | heldout | 1249 | 7.831 | 6.442 | 4.971 | 5.315 | 19.958 | 26.542 | 0.014 | 0.014 |
| decode | eager | calibration | 1 | 7.595 | 27.565 | 19.970 | 19.970 | 19.970 | 19.970 | 0.091 | 0.091 |
| decode | eager | heldout | 1 | 7.122 | 28.205 | 21.083 | 21.083 | 21.083 | 21.083 | 0.119 | 0.119 |
| decode | graph | calibration | 2 | 10.543 | 31.325 | 20.782 | 20.782 | 21.014 | 21.072 | 2.362 | 2.362 |
| decode | ring | calibration | 2 | -170.765 | -165.319 | 5.445 | 5.623 | 9.979 | 11.068 | 0.081 | 0.081 |
| mixed | decode | calibration | 9 | 12.242 | 35.200 | 19.550 | 21.060 | 23.677 | 26.556 | 12.700 | 12.700 |
| mixed | decode | heldout | 1 | 2.765 | 23.253 | 20.487 | 20.487 | 20.487 | 20.487 | 3.110 | 3.110 |
| mixed | eager | calibration | 2 | 66.798 | 84.461 | 17.663 | 17.663 | 22.814 | 24.102 | 37.491 | 37.491 |
| mixed | graph | calibration | 1 | 1.049 | 22.659 | 21.610 | 21.610 | 21.610 | 21.610 | 1.362 | 1.362 |
| mixed | ring | calibration | 3 | 50.349 | 61.300 | 10.980 | 10.951 | 11.084 | 11.117 | 31.803 | 31.803 |
| mixed | ring | heldout | 1 | 105.286 | 116.459 | 11.173 | 11.173 | 11.173 | 11.173 | 60.341 | 60.341 |
| prefill | decode | heldout | 1 | 5.972 | 5.779 | -0.193 | 0.193 | 0.193 | 0.193 | 6.338 | 6.338 |


#### Full workload
- span: measured 49.839 s, predicted 49.852 s, error +0.03 % (abs 0.013 s); end drift +12.6 ms, max |drift| 28.2 ms; 10 excluded (JIT) iterations replayed from their measured timelines
- throughput (tok/s): measured decode 3555.6, prefill 3575.0; predicted decode 3554.7, prefill 3574.1
- Σ walls all: n=4740, measured 44.427 s, predicted 44.455 s, signed bias +0.06 %, abs error 0.028 s, Σ|per-iteration err| 0.036 s; by kind: decode 4726 it. 42.188/42.204 s, eager 4 it. 0.747/0.746 s, graph 4 it. 0.146/0.146 s, ring 6 it. 1.346/1.359 s
- Σ walls calibration: n=3487, measured 32.930 s, predicted 32.955 s, signed bias +0.07 %, abs error 0.024 s, Σ|per-iteration err| 0.031 s; by kind: decode 3475 it. 31.113/31.127 s, eager 3 it. 0.546/0.545 s, graph 4 it. 0.146/0.146 s, ring 5 it. 1.125/1.137 s
- Σ walls heldout: n=1253, measured 11.497 s, predicted 11.500 s, signed bias +0.03 %, abs error 0.004 s, Σ|per-iteration err| 0.005 s; by kind: decode 1251 it. 11.074/11.078 s, eager 1 it. 0.201/0.201 s, ring 1 it. 0.222/0.222 s
- host_start binding constraint counts (draw-summed): {'serial': 834, 'scheduler': 3915, 'first': 1}

### recon-seq (4740 evaluated of 4750 iterations, 3.5 s, draws 1)

#### Iteration latency by role and class
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | 208 | 0.021 | 0.435 | 0.020 | 0.020 | 0.555 | 0.033 | 4.783 |
| calibration | decode b17-24 | 194 | -0.001 | -0.011 | -0.001 | 0.001 | 0.035 | 0.005 | 7.726 |
| calibration | decode b3-16 | 311 | 0.032 | 0.603 | 0.032 | 0.032 | 0.755 | 0.044 | 5.042 |
| calibration | decode b>=25 | 2762 | -0.000 | -0.003 | -0.000 | 0.001 | 0.025 | 0.009 | 9.910 |
| calibration | mixed-eager (host-bound) | 3 | 0.936 | 0.514 | -0.005 | 0.498 | 1.398 | 3.312 | 196.173 |
| calibration | mixed-graph | 4 | -0.000 | -0.001 | -0.000 | 0.002 | 0.005 | 0.002 | 36.723 |
| calibration | mixed-ring (other) | 5 | 2.632 | 1.170 | 0.174 | 0.174 | 3.067 | 10.754 | 221.369 |
| heldout | decode b1-2 | 53 | 0.022 | 0.450 | 0.023 | 0.023 | 0.569 | 0.034 | 4.832 |
| heldout | decode b17-24 | 49 | -0.001 | -0.011 | -0.001 | 0.001 | 0.035 | 0.004 | 7.739 |
| heldout | decode b3-16 | 78 | 0.031 | 0.585 | 0.031 | 0.031 | 0.749 | 0.041 | 5.043 |
| heldout | decode b>=25 | 1071 | -0.000 | -0.001 | -0.000 | 0.001 | 0.024 | 0.006 | 9.752 |
| heldout | mixed-eager (host-bound) | 1 | 2.311 | 1.152 | 2.311 | 2.311 | 1.152 | 2.311 | 200.613 |
| heldout | mixed-ring (other) | 1 | 0.061 | 0.028 | 0.061 | 0.061 | 0.028 | 0.061 | 221.700 |


#### By status (extrapolation / inferred / kv-extrapolation reported separately)
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | measured | measured | 208 | 0.021 | 0.435 | 0.020 | 0.020 | 0.555 | 0.033 | 4.783 |
| calibration | decode b17-24 | measured | measured | 194 | -0.001 | -0.011 | -0.001 | 0.001 | 0.035 | 0.005 | 7.726 |
| calibration | decode b3-16 | measured | measured | 311 | 0.032 | 0.603 | 0.032 | 0.032 | 0.755 | 0.044 | 5.042 |
| calibration | decode b>=25 | measured | measured | 2762 | -0.000 | -0.003 | -0.000 | 0.001 | 0.025 | 0.009 | 9.910 |
| calibration | mixed-eager (host-bound) | measured | measured | 3 | 0.936 | 0.514 | -0.005 | 0.498 | 1.398 | 3.312 | 196.173 |
| calibration | mixed-graph | measured | measured | 4 | -0.000 | -0.001 | -0.000 | 0.002 | 0.005 | 0.002 | 36.723 |
| calibration | mixed-ring (other) | measured | measured | 5 | 2.632 | 1.170 | 0.174 | 0.174 | 3.067 | 10.754 | 221.369 |
| heldout | decode b1-2 | measured | measured | 53 | 0.022 | 0.450 | 0.023 | 0.023 | 0.569 | 0.034 | 4.832 |
| heldout | decode b17-24 | measured | measured | 49 | -0.001 | -0.011 | -0.001 | 0.001 | 0.035 | 0.004 | 7.739 |
| heldout | decode b3-16 | measured | measured | 78 | 0.031 | 0.585 | 0.031 | 0.031 | 0.749 | 0.041 | 5.043 |
| heldout | decode b>=25 | measured | measured | 1071 | -0.000 | -0.001 | -0.000 | 0.001 | 0.024 | 0.006 | 9.752 |
| heldout | mixed-eager (host-bound) | measured | measured | 1 | 2.311 | 1.152 | 2.311 | 2.311 | 1.152 | 2.311 | 200.613 |
| heldout | mixed-ring (other) | measured | measured | 1 | 0.061 | 0.028 | 0.061 | 0.061 | 0.028 | 0.061 | 221.700 |


#### Host lead (GPU end of i-1 minus earliest host start of i) by transition
| key | n | meas_lead_median_ms | pred_lead_median_ms | lead_bias_ms | lead_median_abs_err_ms | lead_p90_abs_err_ms | lead_worst_abs_err_ms | meas_host_skew_median_ms | pred_host_skew_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| decode | decode | calibration | 3466 | 8.081 | 8.092 | -0.203 | 0.053 | 0.182 | 26.971 | 0.014 | 0.000 |
| decode | decode | heldout | 1249 | 7.831 | 7.858 | -0.142 | 0.043 | 0.152 | 4.308 | 0.014 | 0.000 |
| decode | eager | calibration | 1 | 7.595 | 7.497 | -0.098 | 0.098 | 0.098 | 0.098 | 0.091 | 0.000 |
| decode | eager | heldout | 1 | 7.122 | 7.600 | 0.478 | 0.478 | 0.478 | 0.478 | 0.119 | 0.000 |
| decode | graph | calibration | 2 | 10.543 | 9.720 | -0.823 | 0.823 | 1.154 | 1.237 | 2.362 | 0.365 |
| decode | ring | calibration | 2 | -170.765 | 6.028 | 176.792 | 176.792 | 317.213 | 352.319 | 0.081 | 1.899 |
| mixed | decode | calibration | 9 | 12.242 | 14.092 | 5.334 | 3.496 | 17.298 | 25.925 | 12.700 | 11.278 |
| mixed | decode | heldout | 1 | 2.765 | 1.555 | -1.211 | 1.211 | 1.211 | 1.211 | 3.110 | 2.373 |
| mixed | eager | calibration | 2 | 66.798 | 78.090 | 11.292 | 11.292 | 14.911 | 15.816 | 37.491 | 38.325 |
| mixed | graph | calibration | 1 | 1.049 | 5.751 | 4.703 | 4.703 | 4.703 | 4.703 | 1.362 | 0.776 |
| mixed | ring | calibration | 3 | 50.349 | 66.216 | 14.252 | 15.185 | 15.731 | 15.867 | 31.803 | 35.344 |
| mixed | ring | heldout | 1 | 105.286 | 116.582 | 11.296 | 11.296 | 11.296 | 11.296 | 60.341 | 57.828 |
| prefill | decode | heldout | 1 | 5.972 | 4.858 | -1.114 | 1.114 | 1.114 | 1.114 | 6.338 | 5.462 |


#### Full workload
- span: measured 49.839 s, predicted 49.549 s, error -0.58 % (abs 0.291 s); end drift -290.9 ms, max |drift| 337.9 ms; 10 excluded (JIT) iterations replayed from their measured timelines
- throughput (tok/s): measured decode 3555.6, prefill 3575.0; predicted decode 3576.5, prefill 3596.0
- Σ walls all: n=4740, measured 44.427 s, predicted 44.462 s, signed bias +0.08 %, abs error 0.035 s, Σ|per-iteration err| 0.042 s; by kind: decode 4726 it. 42.188/42.204 s, eager 4 it. 0.747/0.752 s, graph 4 it. 0.146/0.146 s, ring 6 it. 1.346/1.360 s
- Σ walls calibration: n=3487, measured 32.930 s, predicted 32.960 s, signed bias +0.09 %, abs error 0.029 s, Σ|per-iteration err| 0.035 s; by kind: decode 3475 it. 31.113/31.127 s, eager 3 it. 0.546/0.549 s, graph 4 it. 0.146/0.146 s, ring 5 it. 1.125/1.138 s
- Σ walls heldout: n=1253, measured 11.497 s, predicted 11.502 s, signed bias +0.05 %, abs error 0.006 s, Σ|per-iteration err| 0.007 s; by kind: decode 1251 it. 11.074/11.078 s, eager 1 it. 0.201/0.203 s, ring 1 it. 0.222/0.222 s
- host_start binding constraint counts (draw-summed): {'serial': 275, 'scheduler': 4474, 'first': 1}

### pred-det (4740 evaluated of 4750 iterations, 5.2 s, draws 1)

#### Iteration latency by role and class
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | 208 | 0.034 | 0.714 | 0.055 | 0.392 | 15.462 | 1.411 | 4.783 |
| calibration | decode b17-24 | 194 | -0.073 | -0.948 | -0.061 | 0.073 | 2.574 | 0.338 | 7.726 |
| calibration | decode b3-16 | 311 | -0.070 | -1.305 | -0.073 | 0.073 | 2.444 | 0.461 | 5.042 |
| calibration | decode b>=25 | 2762 | -0.090 | -0.924 | -0.072 | 0.110 | 2.470 | 10.155 | 9.910 |
| calibration | mixed-eager (host-bound) | 3 | 20.655 | 11.348 | 2.619 | 4.529 | 36.888 | 63.875 | 196.173 |
| calibration | mixed-graph | 4 | -0.115 | -0.314 | -0.593 | 1.700 | 7.804 | 2.933 | 36.723 |
| calibration | mixed-ring (other) | 5 | 2.991 | 1.330 | 5.334 | 5.809 | 3.414 | 8.285 | 221.369 |
| heldout | decode b1-2 | 53 | -0.054 | -1.100 | -0.036 | 0.375 | 16.192 | 3.002 | 4.832 |
| heldout | decode b17-24 | 49 | -0.082 | -1.056 | -0.074 | 0.074 | 2.532 | 0.312 | 7.739 |
| heldout | decode b3-16 | 78 | -0.063 | -1.179 | -0.068 | 0.068 | 2.341 | 0.217 | 5.043 |
| heldout | decode b>=25 | 1071 | -0.084 | -0.893 | -0.070 | 0.101 | 2.387 | 7.316 | 9.752 |
| heldout | mixed-eager (host-bound) | 1 | 2.457 | 1.225 | 2.457 | 2.457 | 1.225 | 2.457 | 200.613 |
| heldout | mixed-ring (other) | 1 | 4.765 | 2.149 | 4.765 | 4.765 | 2.149 | 4.765 | 221.700 |


#### By status (extrapolation / inferred / kv-extrapolation reported separately)
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | measured | in-range | 208 | 0.034 | 0.714 | 0.055 | 0.392 | 15.462 | 1.411 | 4.783 |
| calibration | decode b17-24 | extrapolation | in-range | 4 | 0.093 | 1.226 | 0.089 | 0.119 | 3.782 | 0.338 | 7.576 |
| calibration | decode b17-24 | measured | in-range | 190 | -0.077 | -0.993 | -0.062 | 0.071 | 2.563 | 0.309 | 7.727 |
| calibration | decode b3-16 | interpolated | in-range | 1 | 0.003 | 0.050 | 0.003 | 0.003 | 0.050 | 0.003 | 6.370 |
| calibration | decode b3-16 | measured | in-range | 310 | -0.070 | -1.310 | -0.073 | 0.073 | 2.445 | 0.461 | 5.042 |
| calibration | decode b>=25 | interpolated | in-range | 19 | -0.267 | -2.779 | -0.209 | 0.209 | 4.273 | 2.027 | 9.949 |
| calibration | decode b>=25 | measured | in-range | 2743 | -0.089 | -0.912 | -0.072 | 0.110 | 2.460 | 10.155 | 9.910 |
| calibration | mixed-eager (host-bound) | in-range | in-range | 2 | -0.955 | -0.471 | -0.955 | 3.574 | 2.077 | 4.529 | 202.945 |
| calibration | mixed-eager (host-bound) | inferred | in-range | 1 | 63.875 | 45.570 | 63.875 | 63.875 | 45.570 | 63.875 | 140.168 |
| calibration | mixed-graph | extrapolation | in-range | 1 | -1.195 | -3.201 | -1.195 | 1.195 | 3.201 | 1.195 | 37.320 |
| calibration | mixed-graph | measured-class | in-range | 3 | 0.245 | 0.674 | 0.008 | 2.206 | 8.119 | 2.933 | 36.126 |
| calibration | mixed-ring (other) | in-range | in-range | 2 | -0.773 | -0.334 | -0.773 | 6.107 | 2.826 | 6.880 | 231.431 |
| calibration | mixed-ring (other) | inferred | in-range | 3 | 5.501 | 2.493 | 5.809 | 5.809 | 3.545 | 8.285 | 221.014 |
| heldout | decode b1-2 | measured | in-range | 53 | -0.054 | -1.100 | -0.036 | 0.375 | 16.192 | 3.002 | 4.832 |
| heldout | decode b17-24 | extrapolation | in-range | 1 | -0.180 | -2.298 | -0.180 | 0.180 | 2.298 | 0.180 | 7.846 |
| heldout | decode b17-24 | measured | in-range | 48 | -0.080 | -1.030 | -0.070 | 0.074 | 2.547 | 0.312 | 7.735 |
| heldout | decode b3-16 | interpolated | in-range | 2 | -0.128 | -2.259 | -0.128 | 0.128 | 4.011 | 0.217 | 5.680 |
| heldout | decode b3-16 | measured | in-range | 76 | -0.061 | -1.149 | -0.068 | 0.068 | 2.175 | 0.187 | 5.043 |
| heldout | decode b>=25 | interpolated | in-range | 386 | -0.086 | -0.998 | -0.071 | 0.085 | 2.279 | 6.535 | 8.481 |
| heldout | decode b>=25 | measured | in-range | 685 | -0.082 | -0.841 | -0.069 | 0.112 | 2.457 | 7.316 | 9.904 |
| heldout | mixed-eager (host-bound) | in-range | in-range | 1 | 2.457 | 1.225 | 2.457 | 2.457 | 1.225 | 2.457 | 200.613 |
| heldout | mixed-ring (other) | inferred | in-range | 1 | 4.765 | 2.149 | 4.765 | 4.765 | 2.149 | 4.765 | 221.700 |


#### Host lead (GPU end of i-1 minus earliest host start of i) by transition
| key | n | meas_lead_median_ms | pred_lead_median_ms | lead_bias_ms | lead_median_abs_err_ms | lead_p90_abs_err_ms | lead_worst_abs_err_ms | meas_host_skew_median_ms | pred_host_skew_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| decode | decode | calibration | 3466 | 8.081 | 8.110 | -0.182 | 0.115 | 0.358 | 27.064 | 0.014 | 0.000 |
| decode | decode | heldout | 1249 | 7.831 | 7.656 | -0.155 | 0.103 | 0.309 | 7.463 | 0.014 | 0.000 |
| decode | eager | calibration | 1 | 7.595 | 7.635 | 0.040 | 0.040 | 0.040 | 0.040 | 0.091 | 0.000 |
| decode | eager | heldout | 1 | 7.122 | 7.544 | 0.423 | 0.423 | 0.423 | 0.423 | 0.119 | 0.000 |
| decode | graph | calibration | 2 | 10.543 | 8.602 | -1.941 | 1.941 | 3.063 | 3.343 | 2.362 | 0.512 |
| decode | ring | calibration | 2 | -170.765 | 3.996 | 174.761 | 175.524 | 315.333 | 350.285 | 0.081 | 0.325 |
| mixed | decode | calibration | 9 | 12.242 | 14.728 | 5.132 | 3.129 | 18.333 | 25.381 | 12.700 | 11.306 |
| mixed | decode | heldout | 1 | 2.765 | 1.451 | -1.314 | 1.314 | 1.314 | 1.314 | 3.110 | 2.215 |
| mixed | eager | calibration | 2 | 66.798 | 46.614 | -20.184 | 26.952 | 43.099 | 47.136 | 37.491 | 40.232 |
| mixed | graph | calibration | 1 | 1.049 | 5.751 | 4.703 | 4.703 | 4.703 | 4.703 | 1.362 | 0.776 |
| mixed | ring | calibration | 3 | 50.349 | 40.217 | -16.393 | 10.534 | 24.918 | 28.514 | 31.803 | 29.541 |
| mixed | ring | heldout | 1 | 105.286 | 66.917 | -38.368 | 38.368 | 38.368 | 38.368 | 60.341 | 59.146 |
| prefill | decode | heldout | 1 | 5.972 | 4.285 | -1.687 | 1.687 | 1.687 | 1.687 | 6.338 | 4.467 |


#### Full workload
- span: measured 49.839 s, predicted 49.212 s, error -1.26 % (abs 0.627 s); end drift -627.6 ms, max |drift| 635.8 ms; 10 excluded (JIT) iterations replayed from their measured timelines
- throughput (tok/s): measured decode 3555.6, prefill 3575.0; predicted decode 3601.0, prefill 3620.6
- Σ walls all: n=4740, measured 44.427 s, predicted 44.131 s, signed bias -0.67 %, abs error 0.296 s, Σ|per-iteration err| 0.821 s; by kind: decode 4726 it. 42.188/41.808 s, eager 4 it. 0.747/0.811 s, graph 4 it. 0.146/0.146 s, ring 6 it. 1.346/1.366 s
- Σ walls calibration: n=3487, measured 32.930 s, predicted 32.729 s, signed bias -0.61 %, abs error 0.202 s, Σ|per-iteration err| 0.639 s; by kind: decode 3475 it. 31.113/30.835 s, eager 3 it. 0.546/0.608 s, graph 4 it. 0.146/0.146 s, ring 5 it. 1.125/1.140 s
- Σ walls heldout: n=1253, measured 11.497 s, predicted 11.402 s, signed bias -0.82 %, abs error 0.094 s, Σ|per-iteration err| 0.182 s; by kind: decode 1251 it. 11.074/10.973 s, eager 1 it. 0.201/0.203 s, ring 1 it. 0.222/0.226 s
- host_start binding constraint counts (draw-summed): {'serial': 25, 'scheduler': 4724, 'first': 1}

### pred-mc (4740 evaluated of 4750 iterations, 5.9 s, draws 32)

#### Iteration latency by role and class
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | 208 | 0.199 | 4.157 | 0.201 | 0.246 | 13.173 | 0.994 | 4.783 |
| calibration | decode b17-24 | 194 | 0.112 | 1.448 | 0.116 | 0.116 | 2.960 | 0.537 | 7.726 |
| calibration | decode b3-16 | 311 | 0.120 | 2.248 | 0.121 | 0.124 | 3.549 | 0.311 | 5.042 |
| calibration | decode b>=25 | 2762 | 0.190 | 1.949 | 0.203 | 0.207 | 4.370 | 10.148 | 9.910 |
| calibration | mixed-eager (host-bound) | 3 | 20.416 | 11.216 | 2.501 | 4.529 | 36.546 | 63.276 | 196.173 |
| calibration | mixed-graph | 4 | -0.115 | -0.313 | -0.593 | 1.700 | 7.805 | 2.934 | 36.723 |
| calibration | mixed-ring (other) | 5 | 2.908 | 1.293 | 4.339 | 5.581 | 3.759 | 9.460 | 221.369 |
| heldout | decode b1-2 | 53 | 0.070 | 1.437 | 0.088 | 0.235 | 13.195 | 2.713 | 4.832 |
| heldout | decode b17-24 | 49 | 0.104 | 1.340 | 0.117 | 0.119 | 2.645 | 0.273 | 7.739 |
| heldout | decode b3-16 | 78 | 0.133 | 2.489 | 0.124 | 0.124 | 3.607 | 0.644 | 5.043 |
| heldout | decode b>=25 | 1071 | 0.205 | 2.189 | 0.227 | 0.231 | 4.471 | 7.129 | 9.752 |
| heldout | mixed-eager (host-bound) | 1 | 1.906 | 0.950 | 1.906 | 1.906 | 0.950 | 1.906 | 200.613 |
| heldout | mixed-ring (other) | 1 | 4.496 | 2.028 | 4.496 | 4.496 | 2.028 | 4.496 | 221.700 |


#### By status (extrapolation / inferred / kv-extrapolation reported separately)
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | measured | in-range | 208 | 0.199 | 4.157 | 0.201 | 0.246 | 13.173 | 0.994 | 4.783 |
| calibration | decode b17-24 | extrapolation | in-range | 4 | 0.287 | 3.796 | 0.283 | 0.283 | 6.257 | 0.537 | 7.576 |
| calibration | decode b17-24 | measured | in-range | 190 | 0.108 | 1.400 | 0.116 | 0.116 | 2.859 | 0.298 | 7.727 |
| calibration | decode b3-16 | interpolated | in-range | 1 | 0.188 | 2.945 | 0.188 | 0.188 | 2.945 | 0.188 | 6.370 |
| calibration | decode b3-16 | measured | in-range | 310 | 0.120 | 2.246 | 0.121 | 0.122 | 3.550 | 0.311 | 5.042 |
| calibration | decode b>=25 | interpolated | in-range | 19 | 0.255 | 2.659 | 0.335 | 0.434 | 6.727 | 1.626 | 9.949 |
| calibration | decode b>=25 | measured | in-range | 2743 | 0.190 | 1.944 | 0.202 | 0.207 | 4.323 | 10.148 | 9.910 |
| calibration | mixed-eager (host-bound) | in-range | in-range | 2 | -1.014 | -0.500 | -1.014 | 3.515 | 2.071 | 4.529 | 202.945 |
| calibration | mixed-eager (host-bound) | inferred | in-range | 1 | 63.276 | 45.143 | 63.276 | 63.276 | 45.143 | 63.276 | 140.168 |
| calibration | mixed-graph | extrapolation | in-range | 1 | -1.195 | -3.201 | -1.195 | 1.195 | 3.201 | 1.195 | 37.320 |
| calibration | mixed-graph | measured-class | in-range | 3 | 0.245 | 0.675 | 0.008 | 2.206 | 8.120 | 2.934 | 36.126 |
| calibration | mixed-ring (other) | in-range | in-range | 2 | -1.343 | -0.580 | -1.343 | 5.681 | 2.835 | 7.024 | 231.431 |
| calibration | mixed-ring (other) | inferred | in-range | 3 | 5.742 | 2.603 | 5.581 | 5.581 | 3.952 | 9.460 | 221.014 |
| heldout | decode b1-2 | measured | in-range | 53 | 0.070 | 1.437 | 0.088 | 0.235 | 13.195 | 2.713 | 4.832 |
| heldout | decode b17-24 | extrapolation | in-range | 1 | 0.017 | 0.211 | 0.017 | 0.017 | 0.211 | 0.017 | 7.846 |
| heldout | decode b17-24 | measured | in-range | 48 | 0.106 | 1.364 | 0.118 | 0.119 | 2.653 | 0.273 | 7.735 |
| heldout | decode b3-16 | interpolated | in-range | 2 | 0.048 | 0.845 | 0.048 | 0.087 | 1.967 | 0.135 | 5.680 |
| heldout | decode b3-16 | measured | in-range | 76 | 0.135 | 2.535 | 0.124 | 0.124 | 3.621 | 0.644 | 5.043 |
| heldout | decode b>=25 | interpolated | in-range | 386 | 0.224 | 2.592 | 0.269 | 0.270 | 4.621 | 6.364 | 8.481 |
| heldout | decode b>=25 | measured | in-range | 685 | 0.194 | 1.988 | 0.201 | 0.206 | 4.359 | 7.129 | 9.904 |
| heldout | mixed-eager (host-bound) | in-range | in-range | 1 | 1.906 | 0.950 | 1.906 | 1.906 | 0.950 | 1.906 | 200.613 |
| heldout | mixed-ring (other) | inferred | in-range | 1 | 4.496 | 2.028 | 4.496 | 4.496 | 2.028 | 4.496 | 221.700 |


#### Host lead (GPU end of i-1 minus earliest host start of i) by transition
| key | n | meas_lead_median_ms | pred_lead_median_ms | lead_bias_ms | lead_median_abs_err_ms | lead_p90_abs_err_ms | lead_worst_abs_err_ms | meas_host_skew_median_ms | pred_host_skew_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| decode | decode | calibration | 3466 | 8.081 | 8.385 | 0.074 | 0.199 | 0.500 | 27.354 | 0.014 | 0.000 |
| decode | decode | heldout | 1249 | 7.831 | 8.211 | 0.118 | 0.209 | 0.473 | 7.221 | 0.014 | 0.000 |
| decode | eager | calibration | 1 | 7.595 | 7.754 | 0.159 | 0.159 | 0.159 | 0.159 | 0.091 | 0.000 |
| decode | eager | heldout | 1 | 7.122 | 8.096 | 0.974 | 0.974 | 0.974 | 0.974 | 0.119 | 0.000 |
| decode | graph | calibration | 2 | 10.543 | 9.322 | -1.222 | 1.861 | 2.839 | 3.083 | 2.362 | 0.512 |
| decode | ring | calibration | 2 | -170.765 | 4.341 | 175.106 | 175.468 | 315.553 | 350.574 | 0.081 | 0.325 |
| mixed | decode | calibration | 9 | 12.242 | 14.728 | 5.117 | 3.129 | 18.333 | 25.381 | 12.700 | 11.306 |
| mixed | decode | heldout | 1 | 2.765 | 2.631 | -0.135 | 0.135 | 0.135 | 0.135 | 3.110 | 2.215 |
| mixed | eager | calibration | 2 | 66.798 | 46.914 | -19.884 | 26.653 | 42.560 | 46.537 | 37.491 | 40.232 |
| mixed | graph | calibration | 1 | 1.049 | 5.751 | 4.703 | 4.703 | 4.703 | 4.703 | 1.362 | 0.776 |
| mixed | ring | calibration | 3 | 50.349 | 41.537 | -15.539 | 10.389 | 24.012 | 27.417 | 31.803 | 29.541 |
| mixed | ring | heldout | 1 | 105.286 | 67.786 | -37.500 | 37.500 | 37.500 | 37.500 | 60.341 | 59.146 |
| prefill | decode | heldout | 1 | 5.972 | 4.285 | -1.687 | 1.687 | 1.687 | 1.687 | 6.338 | 4.467 |


#### Full workload
- span: measured 49.839 s, predicted 50.448 s, error +1.22 % (abs 0.609 s); end drift +608.9 ms, max |drift| 609.7 ms; 10 excluded (JIT) iterations replayed from their measured timelines
- throughput (tok/s): measured decode 3555.6, prefill 3575.0; predicted decode 3512.7, prefill 3531.8
- Σ walls all: n=4740, measured 44.427 s, predicted 45.373 s, signed bias +2.13 %, abs error 0.946 s, Σ|per-iteration err| 1.202 s; by kind: decode 4726 it. 42.188/43.052 s, eager 4 it. 0.747/0.810 s, graph 4 it. 0.146/0.146 s, ring 6 it. 1.346/1.366 s
- Σ walls calibration: n=3487, measured 32.930 s, predicted 33.632 s, signed bias +2.13 %, abs error 0.701 s, Σ|per-iteration err| 0.898 s; by kind: decode 3475 it. 31.113/31.739 s, eager 3 it. 0.546/0.607 s, graph 4 it. 0.146/0.146 s, ring 5 it. 1.125/1.139 s
- Σ walls heldout: n=1253, measured 11.497 s, predicted 11.741 s, signed bias +2.13 %, abs error 0.245 s, Σ|per-iteration err| 0.304 s; by kind: decode 1251 it. 11.074/11.313 s, eager 1 it. 0.201/0.203 s, ring 1 it. 0.222/0.226 s
- host_start binding constraint counts (draw-summed): {'serial': 1630, 'scheduler': 150338, 'first': 32}

#### Uncertainty coverage (fraction of measured walls inside the predicted band)
| key | n | cover_p10_p90_pct | cover_p05_p95_pct | band_p10_p90_median_ms | band_rel_median_pct | below_pct | above_pct |
|---|---|---|---|---|---|---|---|
| calibration | decode b1-2 | 208 | 97.115 | 99.519 | 1.612 | 33.616 | 1.923 | 0.962 |
| calibration | decode b17-24 | 194 | 51.546 | 61.856 | 0.250 | 3.228 | 47.938 | 0.515 |
| calibration | decode b3-16 | 311 | 26.045 | 36.013 | 0.180 | 3.233 | 73.633 | 0.322 |
| calibration | decode b>=25 | 2762 | 43.736 | 53.729 | 0.369 | 3.720 | 54.200 | 2.064 |
| calibration | mixed-eager (host-bound) | 3 | 0.000 | 0.000 | 1.683 | 0.858 | 66.667 | 33.333 |
| calibration | mixed-graph | 4 | 0.000 | 0.000 | 0.002 | 0.006 | 50.000 | 50.000 |
| calibration | mixed-ring (other) | 5 | 20.000 | 20.000 | 8.173 | 3.692 | 60.000 | 20.000 |
| heldout | decode b1-2 | 53 | 96.226 | 96.226 | 1.578 | 32.871 | 1.887 | 1.887 |
| heldout | decode b17-24 | 49 | 51.020 | 61.224 | 0.244 | 3.154 | 46.939 | 2.041 |
| heldout | decode b3-16 | 78 | 17.949 | 30.769 | 0.190 | 3.562 | 82.051 | 0.000 |
| heldout | decode b>=25 | 1071 | 38.469 | 47.993 | 0.371 | 3.806 | 59.384 | 2.148 |
| heldout | mixed-eager (host-bound) | 1 | 0.000 | 0.000 | 1.378 | 0.687 | 100.000 | 0.000 |
| heldout | mixed-ring (other) | 1 | 100.000 | 100.000 | 7.946 | 3.584 | 0.000 | 0.000 |


## Pass `diverse_nsys_salvaged`

### recon-gpu (4667 evaluated of 4687 iterations, 5.5 s, draws 1)

#### Iteration latency by role and class
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b17-24 | 34 | -0.003 | -0.037 | -0.003 | 0.003 | 0.063 | 0.006 | 7.483 |
| calibration | decode b3-16 | 2560 | 0.034 | 0.565 | 0.035 | 0.035 | 0.647 | 0.047 | 6.467 |
| calibration | mixed-eager (host-bound) | 1 | 0.002 | 0.001 | 0.002 | 0.002 | 0.001 | 0.002 | 210.094 |
| calibration | mixed-graph | 1 | -0.002 | -0.012 | -0.002 | 0.002 | 0.012 | 0.002 | 16.138 |
| calibration | mixed-ring (other) | 10 | 0.029 | 0.013 | 0.014 | 0.072 | 0.065 | 0.176 | 225.657 |
| calibration | prefill 16k chunk | 33 | 0.058 | 0.025 | -0.047 | 0.055 | 0.059 | 4.979 | 228.400 |
| heldout | decode b17-24 | 32 | -0.003 | -0.042 | -0.003 | 0.003 | 0.073 | 0.007 | 7.465 |
| heldout | decode b3-16 | 1967 | 0.035 | 0.559 | 0.035 | 0.035 | 0.635 | 0.048 | 6.484 |
| heldout | mixed-eager (host-bound) | 1 | -0.001 | -0.000 | -0.001 | 0.001 | 0.000 | 0.001 | 213.248 |
| heldout | mixed-ring (other) | 3 | 0.076 | 0.033 | 0.100 | 0.100 | 0.061 | 0.146 | 223.834 |
| heldout | prefill 16k chunk | 25 | -0.044 | -0.019 | -0.062 | 0.066 | 0.046 | 0.697 | 232.245 |


#### By status (extrapolation / inferred / kv-extrapolation reported separately)
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b17-24 | measured | measured | 34 | -0.003 | -0.037 | -0.003 | 0.003 | 0.063 | 0.006 | 7.483 |
| calibration | decode b3-16 | measured | measured | 2560 | 0.034 | 0.565 | 0.035 | 0.035 | 0.647 | 0.047 | 6.467 |
| calibration | mixed-eager (host-bound) | measured | measured | 1 | 0.002 | 0.001 | 0.002 | 0.002 | 0.001 | 0.002 | 210.094 |
| calibration | mixed-graph | measured | measured | 1 | -0.002 | -0.012 | -0.002 | 0.002 | 0.012 | 0.002 | 16.138 |
| calibration | mixed-ring (other) | measured | measured | 10 | 0.029 | 0.013 | 0.014 | 0.072 | 0.065 | 0.176 | 225.657 |
| calibration | prefill 16k chunk | measured | measured | 33 | 0.058 | 0.025 | -0.047 | 0.055 | 0.059 | 4.979 | 228.400 |
| heldout | decode b17-24 | measured | measured | 32 | -0.003 | -0.042 | -0.003 | 0.003 | 0.073 | 0.007 | 7.465 |
| heldout | decode b3-16 | measured | measured | 1967 | 0.035 | 0.559 | 0.035 | 0.035 | 0.635 | 0.048 | 6.484 |
| heldout | mixed-eager (host-bound) | measured | measured | 1 | -0.001 | -0.000 | -0.001 | 0.001 | 0.000 | 0.001 | 213.248 |
| heldout | mixed-ring (other) | measured | measured | 3 | 0.076 | 0.033 | 0.100 | 0.100 | 0.061 | 0.146 | 223.834 |
| heldout | prefill 16k chunk | measured | measured | 25 | -0.044 | -0.019 | -0.062 | 0.066 | 0.046 | 0.697 | 232.245 |


#### Long-context 16k chunks by KV offset
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | kv0 0k | measured | 13 | 0.218 | 0.097 | -0.044 | 0.070 | 0.437 | 4.979 | 224.064 |
| calibration | kv0 16k | measured | 6 | -0.052 | -0.023 | -0.064 | 0.064 | 0.036 | 0.081 | 223.060 |
| calibration | kv0 32k | measured | 5 | -0.039 | -0.017 | -0.038 | 0.038 | 0.032 | 0.079 | 228.400 |
| calibration | kv0 64k | measured | 5 | -0.046 | -0.019 | -0.050 | 0.050 | 0.037 | 0.094 | 237.661 |
| calibration | kv0 80k | measured | 2 | -0.049 | -0.020 | -0.049 | 0.049 | 0.025 | 0.063 | 243.394 |
| calibration | kv0 96k | measured | 2 | -0.047 | -0.019 | -0.047 | 0.047 | 0.019 | 0.047 | 245.685 |
| heldout | kv0 0k | measured | 3 | 0.050 | 0.022 | -0.075 | 0.471 | 0.282 | 0.697 | 230.518 |
| heldout | kv0 112k | measured | 1 | -0.037 | -0.015 | -0.037 | 0.037 | 0.015 | 0.037 | 253.149 |
| heldout | kv0 16k | measured | 3 | -0.052 | -0.023 | -0.055 | 0.055 | 0.026 | 0.059 | 223.044 |
| heldout | kv0 32k | measured | 3 | -0.061 | -0.027 | -0.073 | 0.073 | 0.044 | 0.108 | 229.175 |
| heldout | kv0 48k | measured | 8 | -0.060 | -0.026 | -0.066 | 0.066 | 0.038 | 0.101 | 232.257 |
| heldout | kv0 64k | measured | 3 | -0.049 | -0.020 | -0.040 | 0.040 | 0.029 | 0.076 | 237.986 |
| heldout | kv0 80k | measured | 2 | -0.057 | -0.023 | -0.057 | 0.057 | 0.029 | 0.073 | 243.805 |
| heldout | kv0 96k | measured | 2 | -0.068 | -0.028 | -0.068 | 0.068 | 0.029 | 0.071 | 246.206 |


#### Host lead (GPU end of i-1 minus earliest host start of i) by transition
| key | n | meas_lead_median_ms | pred_lead_median_ms | lead_bias_ms | lead_median_abs_err_ms | lead_p90_abs_err_ms | lead_worst_abs_err_ms | meas_host_skew_median_ms | pred_host_skew_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| decode | decode | calibration | 2582 | 5.738 | 73.505 | 69.645 | 67.627 | 130.318 | 148.019 | 0.012 | 0.012 |
| decode | decode | heldout | 1990 | 5.770 | 93.800 | 85.139 | 87.903 | 134.519 | 148.083 | 0.012 | 0.012 |
| decode | eager | calibration | 1 | 4.339 | 23.012 | 18.672 | 18.672 | 18.672 | 18.672 | 0.046 | 0.046 |
| decode | eager | heldout | 1 | 4.341 | 28.553 | 24.211 | 24.211 | 24.211 | 24.211 | 0.040 | 0.040 |
| decode | ring | calibration | 7 | 2.921 | 59.698 | 76.616 | 56.586 | 134.662 | 144.923 | 0.421 | 0.421 |
| decode | ring | heldout | 2 | 1.901 | 81.678 | 79.777 | 79.777 | 112.698 | 120.929 | 2.466 | 2.466 |
| mixed | decode | calibration | 11 | 15.346 | 58.833 | 59.500 | 38.748 | 126.099 | 144.606 | 15.719 | 15.719 |
| mixed | decode | heldout | 9 | 16.264 | 131.731 | 100.188 | 118.184 | 145.538 | 148.105 | 16.695 | 16.695 |
| mixed | graph | calibration | 1 | 109.375 | 147.269 | 37.894 | 37.894 | 37.894 | 37.894 | 16.464 | 16.464 |
| mixed | ring | calibration | 36 | 99.511 | 139.406 | 60.630 | 38.380 | 144.699 | 144.974 | 22.201 | 22.201 |
| mixed | ring | heldout | 26 | 99.714 | 146.929 | 62.461 | 38.025 | 122.282 | 144.884 | 45.409 | 45.409 |


#### Full workload
- span: measured 62.176 s, predicted 62.324 s, error +0.24 % (abs 0.148 s); end drift +148.1 ms, max |drift| 148.1 ms; 20 excluded (JIT) iterations replayed from their measured timelines
- throughput (tok/s): measured decode 768.6, prefill 20080.8; predicted decode 766.7, prefill 20033.1
- Σ walls all: n=4667, measured 45.253 s, predicted 45.411 s, signed bias +0.35 %, abs error 0.158 s, Σ|per-iteration err| 0.169 s; by kind: decode 4593 it. 28.455/28.612 s, eager 2 it. 0.423/0.423 s, graph 1 it. 0.016/0.016 s, ring 71 it. 16.358/16.359 s
- Σ walls calibration: n=2639, measured 25.943 s, predicted 26.033 s, signed bias +0.35 %, abs error 0.090 s, Σ|per-iteration err| 0.098 s; by kind: decode 2594 it. 15.884/15.972 s, eager 1 it. 0.210/0.210 s, graph 1 it. 0.016/0.016 s, ring 43 it. 9.833/9.835 s
- Σ walls heldout: n=2028, measured 19.310 s, predicted 19.378 s, signed bias +0.35 %, abs error 0.068 s, Σ|per-iteration err| 0.072 s; by kind: decode 1999 it. 12.571/12.640 s, eager 1 it. 0.213/0.213 s, ring 28 it. 6.526/6.525 s
- host_start binding constraint counts (draw-summed): {'serial': 86, 'scheduler': 4600, 'first': 1}

### recon-seq (4667 evaluated of 4687 iterations, 5.5 s, draws 1)

#### Iteration latency by role and class
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b17-24 | 34 | -0.003 | -0.037 | -0.003 | 0.003 | 0.063 | 0.006 | 7.483 |
| calibration | decode b3-16 | 2560 | 0.034 | 0.565 | 0.035 | 0.035 | 0.647 | 0.047 | 6.467 |
| calibration | mixed-eager (host-bound) | 1 | 4.284 | 2.039 | 4.284 | 4.284 | 2.039 | 4.284 | 210.094 |
| calibration | mixed-graph | 1 | -0.002 | -0.012 | -0.002 | 0.002 | 0.012 | 0.002 | 16.138 |
| calibration | mixed-ring (other) | 10 | 0.029 | 0.013 | 0.014 | 0.072 | 0.065 | 0.176 | 225.657 |
| calibration | prefill 16k chunk | 33 | 0.980 | 0.427 | -0.044 | 0.070 | 2.347 | 7.663 | 228.400 |
| heldout | decode b17-24 | 32 | -0.003 | -0.042 | -0.003 | 0.003 | 0.073 | 0.007 | 7.465 |
| heldout | decode b3-16 | 1967 | 0.035 | 0.559 | 0.035 | 0.035 | 0.635 | 0.048 | 6.484 |
| heldout | mixed-eager (host-bound) | 1 | 4.673 | 2.192 | 4.673 | 4.673 | 2.192 | 4.673 | 213.248 |
| heldout | mixed-ring (other) | 3 | 1.414 | 0.623 | 0.146 | 0.146 | 1.419 | 4.115 | 223.834 |
| heldout | prefill 16k chunk | 25 | 0.128 | 0.055 | -0.062 | 0.066 | 0.046 | 8.464 | 232.245 |


#### By status (extrapolation / inferred / kv-extrapolation reported separately)
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b17-24 | measured | measured | 34 | -0.003 | -0.037 | -0.003 | 0.003 | 0.063 | 0.006 | 7.483 |
| calibration | decode b3-16 | measured | measured | 2560 | 0.034 | 0.565 | 0.035 | 0.035 | 0.647 | 0.047 | 6.467 |
| calibration | mixed-eager (host-bound) | measured | measured | 1 | 4.284 | 2.039 | 4.284 | 4.284 | 2.039 | 4.284 | 210.094 |
| calibration | mixed-graph | measured | measured | 1 | -0.002 | -0.012 | -0.002 | 0.002 | 0.012 | 0.002 | 16.138 |
| calibration | mixed-ring (other) | measured | measured | 10 | 0.029 | 0.013 | 0.014 | 0.072 | 0.065 | 0.176 | 225.657 |
| calibration | prefill 16k chunk | measured | measured | 33 | 0.980 | 0.427 | -0.044 | 0.070 | 2.347 | 7.663 | 228.400 |
| heldout | decode b17-24 | measured | measured | 32 | -0.003 | -0.042 | -0.003 | 0.003 | 0.073 | 0.007 | 7.465 |
| heldout | decode b3-16 | measured | measured | 1967 | 0.035 | 0.559 | 0.035 | 0.035 | 0.635 | 0.048 | 6.484 |
| heldout | mixed-eager (host-bound) | measured | measured | 1 | 4.673 | 2.192 | 4.673 | 4.673 | 2.192 | 4.673 | 213.248 |
| heldout | mixed-ring (other) | measured | measured | 3 | 1.414 | 0.623 | 0.146 | 0.146 | 1.419 | 4.115 | 223.834 |
| heldout | prefill 16k chunk | measured | measured | 25 | 0.128 | 0.055 | -0.062 | 0.066 | 0.046 | 8.464 | 232.245 |


#### Long-context 16k chunks by KV offset
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | kv0 0k | measured | 13 | 2.558 | 1.138 | 0.955 | 1.704 | 3.039 | 7.663 | 224.064 |
| calibration | kv0 16k | measured | 6 | -0.052 | -0.023 | -0.064 | 0.064 | 0.036 | 0.081 | 223.060 |
| calibration | kv0 32k | measured | 5 | -0.039 | -0.017 | -0.038 | 0.038 | 0.032 | 0.079 | 228.400 |
| calibration | kv0 64k | measured | 5 | -0.046 | -0.019 | -0.050 | 0.050 | 0.037 | 0.094 | 237.661 |
| calibration | kv0 80k | measured | 2 | -0.049 | -0.020 | -0.049 | 0.049 | 0.025 | 0.063 | 243.394 |
| calibration | kv0 96k | measured | 2 | -0.047 | -0.019 | -0.047 | 0.047 | 0.019 | 0.047 | 245.685 |
| heldout | kv0 0k | measured | 3 | 1.488 | 0.656 | -0.075 | 3.925 | 3.277 | 8.464 | 230.518 |
| heldout | kv0 112k | measured | 1 | -0.037 | -0.015 | -0.037 | 0.037 | 0.015 | 0.037 | 253.149 |
| heldout | kv0 16k | measured | 3 | -0.052 | -0.023 | -0.055 | 0.055 | 0.026 | 0.059 | 223.044 |
| heldout | kv0 32k | measured | 3 | -0.061 | -0.027 | -0.073 | 0.073 | 0.044 | 0.108 | 229.175 |
| heldout | kv0 48k | measured | 8 | -0.060 | -0.026 | -0.066 | 0.066 | 0.038 | 0.101 | 232.257 |
| heldout | kv0 64k | measured | 3 | -0.049 | -0.020 | -0.040 | 0.040 | 0.029 | 0.076 | 237.986 |
| heldout | kv0 80k | measured | 2 | -0.057 | -0.023 | -0.057 | 0.057 | 0.029 | 0.073 | 243.805 |
| heldout | kv0 96k | measured | 2 | -0.068 | -0.028 | -0.068 | 0.068 | 0.029 | 0.071 | 246.206 |


#### Host lead (GPU end of i-1 minus earliest host start of i) by transition
| key | n | meas_lead_median_ms | pred_lead_median_ms | lead_bias_ms | lead_median_abs_err_ms | lead_p90_abs_err_ms | lead_worst_abs_err_ms | meas_host_skew_median_ms | pred_host_skew_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| decode | decode | calibration | 2582 | 5.738 | 5.779 | 0.046 | 0.039 | 0.109 | 6.618 | 0.012 | 0.000 |
| decode | decode | heldout | 1990 | 5.770 | 5.826 | 0.047 | 0.041 | 0.109 | 2.563 | 0.012 | 0.000 |
| decode | eager | calibration | 1 | 4.339 | 2.862 | -1.477 | 1.477 | 1.477 | 1.477 | 0.046 | 0.000 |
| decode | eager | heldout | 1 | 4.341 | 2.813 | -1.528 | 1.528 | 1.528 | 1.528 | 0.040 | 0.000 |
| decode | ring | calibration | 7 | 2.921 | 2.764 | 2.013 | 2.028 | 5.145 | 6.516 | 0.421 | 0.000 |
| decode | ring | heldout | 2 | 1.901 | 4.223 | 2.322 | 2.322 | 3.730 | 4.082 | 2.466 | 0.000 |
| mixed | decode | calibration | 11 | 15.346 | 24.038 | 10.415 | 5.148 | 32.915 | 38.498 | 15.719 | 14.380 |
| mixed | decode | heldout | 9 | 16.264 | 22.731 | 10.840 | 3.153 | 37.520 | 39.836 | 16.695 | 12.714 |
| mixed | graph | calibration | 1 | 109.375 | 112.781 | 3.406 | 3.406 | 3.406 | 3.406 | 16.464 | 14.428 |
| mixed | ring | calibration | 36 | 99.511 | 108.677 | 8.059 | 8.595 | 10.408 | 11.092 | 22.201 | 20.794 |
| mixed | ring | heldout | 26 | 99.714 | 107.469 | 7.798 | 8.611 | 10.558 | 12.595 | 45.409 | 42.599 |


#### Full workload
- span: measured 62.176 s, predicted 62.524 s, error +0.56 % (abs 0.348 s); end drift +348.1 ms, max |drift| 348.1 ms; 20 excluded (JIT) iterations replayed from their measured timelines
- throughput (tok/s): measured decode 768.6, prefill 20080.8; predicted decode 764.3, prefill 19969.1
- Σ walls all: n=4667, measured 45.253 s, predicted 45.459 s, signed bias +0.46 %, abs error 0.206 s, Σ|per-iteration err| 0.224 s; by kind: decode 4593 it. 28.455/28.612 s, eager 2 it. 0.423/0.432 s, graph 1 it. 0.016/0.016 s, ring 71 it. 16.358/16.398 s
- Σ walls calibration: n=2639, measured 25.943 s, predicted 26.068 s, signed bias +0.48 %, abs error 0.125 s, Σ|per-iteration err| 0.132 s; by kind: decode 2594 it. 15.884/15.972 s, eager 1 it. 0.210/0.214 s, graph 1 it. 0.016/0.016 s, ring 43 it. 9.833/9.865 s
- Σ walls heldout: n=2028, measured 19.310 s, predicted 19.391 s, signed bias +0.42 %, abs error 0.081 s, Σ|per-iteration err| 0.092 s; by kind: decode 1999 it. 12.571/12.640 s, eager 1 it. 0.213/0.218 s, ring 28 it. 6.526/6.533 s
- host_start binding constraint counts (draw-summed): {'serial': 107, 'scheduler': 4579, 'first': 1}

### pred-det (4667 evaluated of 4687 iterations, 8.0 s, draws 1)

#### Iteration latency by role and class
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b17-24 | 34 | 0.069 | 0.912 | 0.183 | 0.197 | 3.830 | 4.081 | 7.483 |
| calibration | decode b3-16 | 2560 | -0.168 | -2.752 | -0.157 | 0.157 | 4.150 | 4.535 | 6.467 |
| calibration | mixed-eager (host-bound) | 1 | -8.459 | -4.026 | -8.459 | 8.459 | 4.026 | 8.459 | 210.094 |
| calibration | mixed-graph | 1 | 19.994 | 123.895 | 19.994 | 19.994 | 123.895 | 19.994 | 16.138 |
| calibration | mixed-ring (other) | 10 | -1.452 | -0.643 | -0.910 | 1.933 | 2.421 | 5.709 | 225.657 |
| calibration | prefill 16k chunk | 33 | 0.416 | 0.181 | -0.190 | 0.367 | 0.919 | 8.329 | 228.400 |
| heldout | decode b17-24 | 32 | 0.186 | 2.492 | 0.200 | 0.200 | 3.776 | 0.289 | 7.465 |
| heldout | decode b3-16 | 1967 | -0.183 | -2.916 | -0.153 | 0.153 | 3.918 | 7.551 | 6.484 |
| heldout | mixed-eager (host-bound) | 1 | -11.513 | -5.399 | -11.513 | 11.513 | 5.399 | 11.513 | 213.248 |
| heldout | mixed-ring (other) | 3 | -3.039 | -1.340 | -4.813 | 4.813 | 2.276 | 5.217 | 223.834 |
| heldout | prefill 16k chunk | 25 | -0.360 | -0.154 | 0.098 | 0.676 | 0.851 | 12.230 | 232.245 |


#### By status (extrapolation / inferred / kv-extrapolation reported separately)
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b17-24 | extrapolation | in-range | 34 | 0.069 | 0.912 | 0.183 | 0.197 | 3.830 | 4.081 | 7.483 |
| calibration | decode b3-16 | interpolated | in-range | 612 | -0.178 | -2.714 | -0.166 | 0.166 | 3.850 | 2.698 | 6.524 |
| calibration | decode b3-16 | measured | in-range | 1948 | -0.165 | -2.765 | -0.153 | 0.154 | 4.228 | 4.535 | 6.422 |
| calibration | mixed-eager (host-bound) | in-range | in-range | 1 | -8.459 | -4.026 | -8.459 | 8.459 | 4.026 | 8.459 | 210.094 |
| calibration | mixed-graph | extrapolation | in-range | 1 | 19.994 | 123.895 | 19.994 | 19.994 | 123.895 | 19.994 | 16.138 |
| calibration | mixed-ring (other) | inferred | in-range | 10 | -1.452 | -0.643 | -0.910 | 1.933 | 2.421 | 5.709 | 225.657 |
| calibration | prefill 16k chunk | in-range | in-range | 8 | 1.364 | 0.599 | -0.224 | 1.891 | 2.604 | 8.329 | 228.662 |
| calibration | prefill 16k chunk | inferred | in-range | 25 | 0.113 | 0.049 | -0.168 | 0.352 | 0.451 | 2.293 | 228.400 |
| heldout | decode b17-24 | extrapolation | in-range | 32 | 0.186 | 2.492 | 0.200 | 0.200 | 3.776 | 0.289 | 7.465 |
| heldout | decode b3-16 | interpolated | in-range | 528 | -0.182 | -2.786 | -0.159 | 0.159 | 3.707 | 5.115 | 6.523 |
| heldout | decode b3-16 | measured | in-range | 1439 | -0.183 | -2.967 | -0.150 | 0.150 | 3.965 | 7.551 | 6.462 |
| heldout | mixed-eager (host-bound) | in-range | in-range | 1 | -11.513 | -5.399 | -11.513 | 11.513 | 5.399 | 11.513 | 213.248 |
| heldout | mixed-ring (other) | in-range | in-range | 1 | -4.813 | -2.055 | -4.813 | 4.813 | 2.055 | 4.813 | 234.193 |
| heldout | mixed-ring (other) | inferred | in-range | 2 | -2.152 | -0.964 | -2.152 | 3.065 | 2.139 | 5.217 | 223.168 |
| heldout | prefill 16k chunk | extrapolation | kv-extrapolation | 2 | 0.586 | 0.234 | 0.586 | 0.985 | 0.588 | 1.571 | 250.112 |
| heldout | prefill 16k chunk | in-range | in-range | 2 | -7.518 | -3.255 | -7.518 | 7.518 | 4.878 | 12.230 | 230.980 |
| heldout | prefill 16k chunk | inferred | in-range | 21 | 0.231 | 0.099 | 0.129 | 0.575 | 0.495 | 2.441 | 232.245 |


#### Long-context 16k chunks by KV offset
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | kv0 0k | in-range | 8 | 1.364 | 0.599 | -0.224 | 1.891 | 2.604 | 8.329 | 228.662 |
| calibration | kv0 0k | inferred | 5 | -0.023 | -0.010 | -0.279 | 0.832 | 0.451 | 1.014 | 219.524 |
| calibration | kv0 16k | inferred | 6 | 0.025 | 0.011 | 0.115 | 0.274 | 0.268 | 0.652 | 223.060 |
| calibration | kv0 32k | inferred | 5 | -0.318 | -0.139 | -0.341 | 0.341 | 0.158 | 0.367 | 228.400 |
| calibration | kv0 64k | inferred | 5 | 0.260 | 0.110 | 0.239 | 0.239 | 0.233 | 0.746 | 237.661 |
| calibration | kv0 80k | inferred | 2 | -0.566 | -0.233 | -0.566 | 0.566 | 0.303 | 0.781 | 243.394 |
| calibration | kv0 96k | inferred | 2 | 2.105 | 0.857 | 2.105 | 2.105 | 0.919 | 2.293 | 245.685 |
| heldout | kv0 0k | in-range | 2 | -7.518 | -3.255 | -7.518 | 7.518 | 4.878 | 12.230 | 230.980 |
| heldout | kv0 0k | inferred | 1 | 0.098 | 0.045 | 0.098 | 0.098 | 0.045 | 0.098 | 219.009 |
| heldout | kv0 112k | extrapolation | 1 | -0.400 | -0.158 | -0.400 | 0.400 | 0.158 | 0.400 | 253.149 |
| heldout | kv0 16k | inferred | 3 | 0.230 | 0.103 | 0.210 | 0.210 | 0.210 | 0.532 | 223.044 |
| heldout | kv0 32k | inferred | 3 | -0.793 | -0.346 | -1.100 | 1.100 | 0.492 | 1.135 | 229.175 |
| heldout | kv0 48k | inferred | 8 | 0.564 | 0.243 | 0.711 | 0.711 | 0.403 | 1.196 | 232.257 |
| heldout | kv0 64k | inferred | 3 | 0.202 | 0.085 | 0.129 | 0.129 | 0.204 | 0.575 | 237.986 |
| heldout | kv0 80k | inferred | 2 | -0.556 | -0.228 | -0.556 | 0.556 | 0.377 | 1.011 | 243.805 |
| heldout | kv0 96k | extrapolation | 1 | 1.571 | 0.636 | 1.571 | 1.571 | 0.636 | 1.571 | 247.076 |
| heldout | kv0 96k | inferred | 1 | 2.441 | 0.995 | 2.441 | 2.441 | 0.995 | 2.441 | 245.337 |


#### Host lead (GPU end of i-1 minus earliest host start of i) by transition
| key | n | meas_lead_median_ms | pred_lead_median_ms | lead_bias_ms | lead_median_abs_err_ms | lead_p90_abs_err_ms | lead_worst_abs_err_ms | meas_host_skew_median_ms | pred_host_skew_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| decode | decode | calibration | 2582 | 5.738 | 5.554 | -0.155 | 0.153 | 0.255 | 6.484 | 0.012 | 0.000 |
| decode | decode | heldout | 1990 | 5.770 | 5.699 | -0.149 | 0.146 | 0.245 | 3.956 | 0.012 | 0.000 |
| decode | eager | calibration | 1 | 4.339 | 2.686 | -1.653 | 1.653 | 1.653 | 1.653 | 0.046 | 0.000 |
| decode | eager | heldout | 1 | 4.341 | 2.574 | -1.767 | 1.767 | 1.767 | 1.767 | 0.040 | 0.000 |
| decode | ring | calibration | 7 | 2.921 | 2.491 | 0.504 | 1.910 | 5.642 | 6.261 | 0.421 | 0.000 |
| decode | ring | heldout | 2 | 1.901 | 1.894 | -0.007 | 3.995 | 4.001 | 4.003 | 2.466 | 0.000 |
| mixed | decode | calibration | 11 | 15.346 | 23.525 | 15.211 | 16.191 | 38.498 | 47.391 | 15.719 | 15.380 |
| mixed | decode | heldout | 9 | 16.264 | 31.274 | 21.908 | 3.006 | 53.635 | 108.829 | 16.695 | 11.395 |
| mixed | graph | calibration | 1 | 109.375 | 124.023 | 14.648 | 14.648 | 14.648 | 14.648 | 16.464 | 0.020 |
| mixed | ring | calibration | 36 | 99.511 | 98.727 | 1.945 | 14.107 | 26.758 | 47.270 | 22.201 | 43.332 |
| mixed | ring | heldout | 26 | 99.714 | 86.344 | -6.129 | 18.919 | 35.682 | 40.381 | 45.409 | 42.699 |


#### Full workload
- span: measured 62.176 s, predicted 61.691 s, error -0.78 % (abs 0.485 s); end drift -485.3 ms, max |drift| 522.4 ms; 20 excluded (JIT) iterations replayed from their measured timelines
- throughput (tok/s): measured decode 768.6, prefill 20080.8; predicted decode 774.6, prefill 20238.8
- Σ walls all: n=4667, measured 45.253 s, predicted 44.453 s, signed bias -1.77 %, abs error 0.800 s, Σ|per-iteration err| 0.981 s; by kind: decode 4593 it. 28.455/27.674 s, eager 2 it. 0.423/0.403 s, graph 1 it. 0.016/0.036 s, ring 71 it. 16.358/16.339 s
- Σ walls calibration: n=2639, measured 25.943 s, predicted 25.526 s, signed bias -1.61 %, abs error 0.417 s, Σ|per-iteration err| 0.562 s; by kind: decode 2594 it. 15.884/15.457 s, eager 1 it. 0.210/0.202 s, graph 1 it. 0.016/0.036 s, ring 43 it. 9.833/9.832 s
- Σ walls heldout: n=2028, measured 19.310 s, predicted 18.927 s, signed bias -1.99 %, abs error 0.383 s, Σ|per-iteration err| 0.419 s; by kind: decode 1999 it. 12.571/12.217 s, eager 1 it. 0.213/0.202 s, ring 28 it. 6.526/6.507 s
- host_start binding constraint counts (draw-summed): {'serial': 94, 'scheduler': 4592, 'first': 1}

### pred-mc (4667 evaluated of 4687 iterations, 8.6 s, draws 32)

#### Iteration latency by role and class
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b17-24 | 34 | 0.253 | 3.324 | 0.367 | 0.385 | 6.328 | 3.890 | 7.483 |
| calibration | decode b3-16 | 2560 | 0.038 | 0.622 | 0.040 | 0.059 | 2.737 | 4.322 | 6.467 |
| calibration | mixed-eager (host-bound) | 1 | -8.664 | -4.124 | -8.664 | 8.664 | 4.124 | 8.664 | 210.094 |
| calibration | mixed-graph | 1 | 19.994 | 123.896 | 19.994 | 19.994 | 123.896 | 19.994 | 16.138 |
| calibration | mixed-ring (other) | 10 | -1.606 | -0.711 | -1.296 | 1.987 | 2.425 | 5.573 | 225.657 |
| calibration | prefill 16k chunk | 33 | 0.332 | 0.145 | -0.254 | 0.979 | 1.061 | 8.365 | 228.400 |
| heldout | decode b17-24 | 32 | 0.367 | 4.912 | 0.374 | 0.374 | 6.275 | 0.479 | 7.465 |
| heldout | decode b3-16 | 1967 | 0.025 | 0.396 | 0.040 | 0.058 | 2.828 | 7.373 | 6.484 |
| heldout | mixed-eager (host-bound) | 1 | -11.721 | -5.496 | -11.721 | 11.721 | 5.496 | 11.721 | 213.248 |
| heldout | mixed-ring (other) | 3 | -3.228 | -1.423 | -4.662 | 4.662 | 2.452 | 5.746 | 223.834 |
| heldout | prefill 16k chunk | 25 | -0.359 | -0.153 | 0.221 | 1.088 | 0.916 | 11.177 | 232.245 |


#### By status (extrapolation / inferred / kv-extrapolation reported separately)
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | decode b17-24 | extrapolation | in-range | 34 | 0.253 | 3.324 | 0.367 | 0.385 | 6.328 | 3.890 | 7.483 |
| calibration | decode b3-16 | interpolated | in-range | 612 | 0.046 | 0.703 | 0.034 | 0.058 | 3.402 | 2.511 | 6.524 |
| calibration | decode b3-16 | measured | in-range | 1948 | 0.035 | 0.594 | 0.041 | 0.060 | 2.547 | 4.322 | 6.422 |
| calibration | mixed-eager (host-bound) | in-range | in-range | 1 | -8.664 | -4.124 | -8.664 | 8.664 | 4.124 | 8.664 | 210.094 |
| calibration | mixed-graph | extrapolation | in-range | 1 | 19.994 | 123.896 | 19.994 | 19.994 | 123.896 | 19.994 | 16.138 |
| calibration | mixed-ring (other) | inferred | in-range | 10 | -1.606 | -0.711 | -1.296 | 1.987 | 2.425 | 5.573 | 225.657 |
| calibration | prefill 16k chunk | in-range | in-range | 8 | 1.179 | 0.518 | -0.355 | 2.192 | 2.404 | 8.365 | 228.662 |
| calibration | prefill 16k chunk | inferred | in-range | 25 | 0.061 | 0.026 | 0.165 | 0.907 | 0.775 | 2.321 | 228.400 |
| heldout | decode b17-24 | extrapolation | in-range | 32 | 0.367 | 4.912 | 0.374 | 0.374 | 6.275 | 0.479 | 7.465 |
| heldout | decode b3-16 | interpolated | in-range | 528 | 0.035 | 0.528 | 0.037 | 0.052 | 3.047 | 4.991 | 6.523 |
| heldout | decode b3-16 | measured | in-range | 1439 | 0.021 | 0.345 | 0.044 | 0.060 | 2.761 | 7.373 | 6.462 |
| heldout | mixed-eager (host-bound) | in-range | in-range | 1 | -11.721 | -5.496 | -11.721 | 11.721 | 5.496 | 11.721 | 213.248 |
| heldout | mixed-ring (other) | in-range | in-range | 1 | -4.662 | -1.991 | -4.662 | 4.662 | 1.991 | 4.662 | 234.193 |
| heldout | mixed-ring (other) | inferred | in-range | 2 | -2.510 | -1.125 | -2.510 | 3.236 | 2.343 | 5.746 | 223.168 |
| heldout | prefill 16k chunk | extrapolation | kv-extrapolation | 2 | 0.729 | 0.292 | 0.729 | 1.487 | 0.837 | 2.217 | 250.112 |
| heldout | prefill 16k chunk | in-range | in-range | 2 | -7.042 | -3.049 | -7.042 | 7.042 | 4.472 | 11.177 | 230.980 |
| heldout | prefill 16k chunk | inferred | in-range | 21 | 0.174 | 0.075 | 0.240 | 1.065 | 0.584 | 2.276 | 232.245 |


#### Long-context 16k chunks by KV offset
| key | n | bias_ms | bias_pct | median_err_ms | median_abs_err_ms | p90_ape_pct | worst_abs_err_ms | meas_median_ms |
|---|---|---|---|---|---|---|---|---|
| calibration | kv0 0k | in-range | 8 | 1.179 | 0.518 | -0.355 | 2.192 | 2.404 | 8.365 | 228.662 |
| calibration | kv0 0k | inferred | 5 | 0.073 | 0.033 | -0.312 | 0.563 | 0.919 | 2.220 | 219.524 |
| calibration | kv0 16k | inferred | 6 | 0.252 | 0.112 | 0.831 | 1.135 | 0.575 | 1.289 | 223.060 |
| calibration | kv0 32k | inferred | 5 | -0.467 | -0.204 | -0.888 | 0.888 | 0.416 | 0.979 | 228.400 |
| calibration | kv0 64k | inferred | 5 | 0.142 | 0.060 | 0.316 | 0.350 | 0.225 | 0.638 | 237.661 |
| calibration | kv0 80k | inferred | 2 | -1.435 | -0.589 | -1.435 | 1.435 | 0.633 | 1.564 | 243.394 |
| calibration | kv0 96k | inferred | 2 | 2.069 | 0.842 | 2.069 | 2.069 | 0.925 | 2.321 | 245.685 |
| heldout | kv0 0k | in-range | 2 | -7.042 | -3.049 | -7.042 | 7.042 | 4.472 | 11.177 | 230.980 |
| heldout | kv0 0k | inferred | 1 | 0.205 | 0.094 | 0.205 | 0.205 | 0.094 | 0.205 | 219.009 |
| heldout | kv0 112k | extrapolation | 1 | -0.758 | -0.299 | -0.758 | 0.758 | 0.299 | 0.758 | 253.149 |
| heldout | kv0 16k | inferred | 3 | 0.595 | 0.266 | 0.420 | 0.420 | 0.448 | 1.143 | 223.044 |
| heldout | kv0 32k | inferred | 3 | -1.264 | -0.552 | -1.184 | 1.184 | 0.634 | 1.521 | 229.175 |
| heldout | kv0 48k | inferred | 8 | 0.455 | 0.196 | 0.891 | 1.080 | 0.581 | 1.356 | 232.257 |
| heldout | kv0 64k | inferred | 3 | 0.125 | 0.052 | 0.240 | 0.436 | 0.229 | 0.571 | 237.986 |
| heldout | kv0 80k | inferred | 2 | -0.415 | -0.170 | -0.415 | 0.843 | 0.482 | 1.258 | 243.805 |
| heldout | kv0 96k | extrapolation | 1 | 2.217 | 0.897 | 2.217 | 2.217 | 0.897 | 2.217 | 247.076 |
| heldout | kv0 96k | inferred | 1 | 2.276 | 0.928 | 2.276 | 2.276 | 0.928 | 2.276 | 245.337 |


#### Host lead (GPU end of i-1 minus earliest host start of i) by transition
| key | n | meas_lead_median_ms | pred_lead_median_ms | lead_bias_ms | lead_median_abs_err_ms | lead_p90_abs_err_ms | lead_worst_abs_err_ms | meas_host_skew_median_ms | pred_host_skew_median_ms |
|---|---|---|---|---|---|---|---|---|---|
| decode | decode | calibration | 2582 | 5.738 | 5.791 | 0.052 | 0.063 | 0.178 | 6.904 | 0.012 | 0.000 |
| decode | decode | heldout | 1990 | 5.770 | 5.871 | 0.056 | 0.067 | 0.178 | 3.834 | 0.012 | 0.000 |
| decode | eager | calibration | 1 | 4.339 | 2.891 | -1.448 | 1.448 | 1.448 | 1.448 | 0.046 | 0.000 |
| decode | eager | heldout | 1 | 4.341 | 2.782 | -1.559 | 1.559 | 1.559 | 1.559 | 0.040 | 0.000 |
| decode | ring | calibration | 7 | 2.921 | 2.725 | 0.697 | 2.063 | 5.699 | 6.576 | 0.421 | 0.000 |
| decode | ring | heldout | 2 | 1.901 | 2.080 | 0.179 | 3.968 | 4.111 | 4.147 | 2.466 | 0.000 |
| mixed | decode | calibration | 11 | 15.346 | 23.525 | 15.155 | 16.191 | 38.498 | 47.800 | 15.719 | 15.380 |
| mixed | decode | heldout | 9 | 16.264 | 31.274 | 21.935 | 3.006 | 53.755 | 109.429 | 16.695 | 11.484 |
| mixed | graph | calibration | 1 | 109.375 | 124.332 | 14.957 | 14.957 | 14.957 | 14.957 | 16.464 | 0.021 |
| mixed | ring | calibration | 36 | 99.511 | 100.096 | 2.169 | 14.679 | 27.128 | 47.522 | 22.201 | 42.617 |
| mixed | ring | heldout | 26 | 99.714 | 86.819 | -5.950 | 18.963 | 34.964 | 40.214 | 45.409 | 42.699 |


#### Full workload
- span: measured 62.176 s, predicted 62.630 s, error +0.73 % (abs 0.454 s); end drift +454.3 ms, max |drift| 458.4 ms; 20 excluded (JIT) iterations replayed from their measured timelines
- throughput (tok/s): measured decode 768.6, prefill 20080.8; predicted decode 763.0, prefill 19935.2
- Σ walls all: n=4667, measured 45.253 s, predicted 45.395 s, signed bias +0.31 %, abs error 0.142 s, Σ|per-iteration err| 0.645 s; by kind: decode 4593 it. 28.455/28.622 s, eager 2 it. 0.423/0.403 s, graph 1 it. 0.016/0.036 s, ring 71 it. 16.358/16.334 s
- Σ walls calibration: n=2639, measured 25.943 s, predicted 26.055 s, signed bias +0.43 %, abs error 0.112 s, Σ|per-iteration err| 0.364 s; by kind: decode 2594 it. 15.884/15.990 s, eager 1 it. 0.210/0.201 s, graph 1 it. 0.016/0.036 s, ring 43 it. 9.833/9.827 s
- Σ walls heldout: n=2028, measured 19.310 s, predicted 19.340 s, signed bias +0.16 %, abs error 0.030 s, Σ|per-iteration err| 0.280 s; by kind: decode 1999 it. 12.571/12.632 s, eager 1 it. 0.213/0.202 s, ring 28 it. 6.526/6.507 s
- host_start binding constraint counts (draw-summed): {'serial': 3008, 'scheduler': 146944, 'first': 32}

#### Uncertainty coverage (fraction of measured walls inside the predicted band)
| key | n | cover_p10_p90_pct | cover_p05_p95_pct | band_p10_p90_median_ms | band_rel_median_pct | below_pct | above_pct |
|---|---|---|---|---|---|---|---|
| calibration | decode b17-24 | 34 | 0.000 | 0.000 | 0.134 | 1.801 | 97.059 | 2.941 |
| calibration | decode b3-16 | 2560 | 68.047 | 75.977 | 0.161 | 2.813 | 24.492 | 7.461 |
| calibration | mixed-eager (host-bound) | 1 | 0.000 | 0.000 | 0.126 | 0.060 | 0.000 | 100.000 |
| calibration | mixed-graph | 1 | 0.000 | 0.000 | 0.000 | 0.002 | 100.000 | 0.000 |
| calibration | mixed-ring (other) | 10 | 70.000 | 100.000 | 6.805 | 3.116 | 0.000 | 30.000 |
| calibration | prefill 16k chunk | 33 | 90.909 | 90.909 | 7.467 | 3.045 | 9.091 | 0.000 |
| heldout | decode b17-24 | 32 | 0.000 | 0.000 | 0.135 | 1.801 | 100.000 | 0.000 |
| heldout | decode b3-16 | 1967 | 69.446 | 76.563 | 0.167 | 2.764 | 24.250 | 6.304 |
| heldout | mixed-eager (host-bound) | 1 | 0.000 | 0.000 | 0.185 | 0.087 | 0.000 | 100.000 |
| heldout | mixed-ring (other) | 3 | 33.333 | 66.667 | 6.085 | 2.735 | 0.000 | 66.667 |
| heldout | prefill 16k chunk | 25 | 96.000 | 96.000 | 8.104 | 3.566 | 0.000 | 4.000 |


## Sequence parameters (calibration only)

- a_host per rank (ms): [0.224, 0.251, 0.26, 0.253]; f_idle 0.543 ms; delta_launch 0.020 ms (inferred)
- queue depth Q = 968 launches; evidence {"n_rank_rows": 192, "implied_p50": 436.101012789793, "implied_p90": 827.4256934106021, "implied_max": 967.529253868864, "saturated_n": 58, "saturated_p50": 806.2392778314542, "saturated_p90": 957.8809538035774, "rule": "Q = max over calibration ring rank-rows of (GPU_end - host_end_r) * launches_r / wall (lower bound of the hard cap); nominal CUDA queue 1024"}
- h_pre (ms per rank): {decode: [2.61, 2.54, 2.8, 2.68], ring: [7.98, 7.99, 8.13, 7.88], eager: [5.8, 5.59, 6.01, 5.91], graph: [5.76, 5.66, 6.16, 5.91]}
- p_post (ms per rank): {decode: [1.12, 1.1, 1.23, 1.16], ring: [1.87, 1.84, 1.79, 1.77], eager: [1.56, 1.47, 1.57, 1.52], graph: [1.58, 1.5, 1.45, 1.38]}

### eps_sched (host_start_0(i) - GPU_end(i-2) after a decode iteration), ms

| kind | b_prev2 | n | median_ms | p10_ms | p90_ms |
|---|---|---|---|---|---|
| decode | 1 | 410 | 2.387 | 0.406 | 2.849 |
| decode | 2 | 635 | 0.496 | 0.453 | 2.784 |
| decode | 3 | 331 | 0.458 | 0.423 | 0.545 |
| decode | 4 | 549 | 0.529 | 0.458 | 0.656 |
| decode | 6 | 382 | 0.563 | 0.537 | 0.624 |
| decode | 7 | 391 | 0.625 | 0.575 | 0.673 |
| decode | 8 | 266 | 0.628 | 0.573 | 0.689 |
| decode | 9 | 437 | 0.641 | 0.594 | 0.693 |
| decode | 11 | 74 | 0.652 | 0.626 | 0.716 |
| decode | 12 | 61 | 0.748 | 0.705 | 0.794 |
| decode | 20 | 1 | 1.212 | 1.212 | 1.212 |
| decode | 23 | 1 | 1.698 | 1.698 | 1.698 |
| decode | 24 | 92 | 1.040 | 1.011 | 1.171 |
| decode | 25 | 108 | 1.074 | 1.027 | 1.155 |
| decode | 27 | 52 | 1.090 | 1.052 | 1.131 |
| decode | 28 | 1 | 1.373 | 1.373 | 1.373 |
| decode | 43 | 1 | 1.577 | 1.577 | 1.577 |
| decode | 44 | 1 | 2.130 | 2.130 | 2.130 |
| decode | 46 | 3 | 1.560 | 1.356 | 1.878 |
| decode | 47 | 2 | 3.075 | 1.424 | 4.727 |
| decode | 48 | 1212 | 1.744 | 1.627 | 1.879 |
| ring | * | 6 | 3.810 | 3.340 | 7.229 |
| graph | * | 1 | 2.028 | 2.028 | 2.028 |
| eager | * | 1 | 2.320 | 2.320 | 2.320 |


### Parameter status

- eps_sched[kind, b_prev2]: measured (host_start_0(i) - GPU_end(i-2) after a decode iteration; per b table, MIN_N rule)
- a_host: measured (host_start_r(i) - host_end_r(i-1) after mixed iterations, per rank)
- f_idle: measured (first kernel - host start when the GPU was idle; n=4)
- delta_launch: inferred (individual launch -> GPU start latency, 20 us; not separately identifiable, second order)
- h_pre / p_post: measured (forward start - host start; host end - forward end; per kind and rank)
- hostwin_dec / n_launch_dec: measured per b (decode host forward window, launches)
- queue_depth: measured lower bound (max implied depth on calibration ring rows)
- segment profile: measured (median normalized compute window per rank and segment, quantiles for stochastic draws)
- launch distribution over segments: assumed uniform (launch index proportional to segment index); not identifiable per segment from the API trace
- rank-2 host lateness: not identifiable (gap G4): carried as the measured per-rank c_host of Phase 1A, no cause modelled
- c_kv (KV-dependent launch cost): not identifiable: Phase 1A value 0 kept; launch count still grows with KV16 (measured)
- dispersion: measured (calibration quantiles of wall / reference; used only to widen the stochastic band, mean-1 factor)
- mixed segment jitter: not sampled: per-segment jitter is common-mode across ranks; independent sampling over-predicted ring walls by 6 %
