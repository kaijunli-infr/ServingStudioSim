| pass | wave start | batch | iterations | kernel rows | rows / iteration (all) | pure decode: iterations, mean, min – max | partial-batch decode: n, mean | prefill or mixed: n, mean, min – max | rows per rank 0 – 3 |
|---|---|---|---|---|---|---|---|---|---|
| e1_v2_forced_release | 0.12 s | 8 | 1040 | 5887244 | 5660.81 | 1008, 5638.98, 5637 – 5645 | 15, 5615.4 | 17, 6995.59, 6229 – 7233 | 1472591, 1471551, 1471551, 1471551 |
| e1_v2_forced_release | 18.75 s | 48 | 1120 | 6363588 | 5681.77 | 928, 5537.25, 5533 – 5541 | 95, 5567.65 | 97, 7176.22, 6229 – 7301 | 1591737, 1590617, 1590617, 1590617 |
| e1_v3_scale | 0.11 s | 8 | 1040 | 5887244 | 5660.81 | 1008, 5638.98, 5637 – 5645 | 15, 5615.4 | 17, 6995.59, 6229 – 7233 | 1472591, 1471551, 1471551, 1471551 |
| e1_v3_scale | 18.74 s | 48 | 1120 | 6363588 | 5681.77 | 928, 5537.25, 5533 – 5541 | 95, 5567.65 | 97, 7176.22, 6229 – 7301 | 1591737, 1590617, 1590617, 1590617 |
| e1_v3_scale | 80.1 s | 8 | 1040 | 5887244 | 5660.81 | 1008, 5638.98, 5637 – 5645 | 15, 5615.4 | 17, 6995.59, 6229 – 7233 | 1472591, 1471551, 1471551, 1471551 |
| e1_v3_scale | 98.74 s | 48 | 1120 | 6363588 | 5681.77 | 928, 5537.25, 5533 – 5541 | 95, 5567.65 | 97, 7176.22, 6229 – 7301 | 1591737, 1590617, 1590617, 1590617 |

Pure-decode rows per iteration, by rank (value: number of iterations):

- b=8 (3 waves, identical): rank 0: {1410: 759, 1412: 249}; rank 1: {1409: 759, 1411: 249}; rank 2: {1409: 759, 1411: 249}; rank 3: {1409: 759, 1411: 249}
- b=48 (3 waves, identical): rank 0: {1384: 435, 1386: 493}; rank 1: {1383: 435, 1385: 493}; rank 2: {1383: 435, 1385: 493}; rank 3: {1383: 435, 1385: 493}

Checks: {"sum_of_waves_equals_recovered_trace_rows": {"e1_v2_forced_release": true, "e1_v3_scale": true}, "waves_of_a_cell_identical_in_rows_iterations_and_sequence": {"b8": true, "b48": true}, "waves_per_cell": {"b8": ["e1_v2_forced_release @ 0.12 s", "e1_v3_scale @ 0.11 s", "e1_v3_scale @ 80.1 s"], "b48": ["e1_v2_forced_release @ 18.75 s", "e1_v3_scale @ 18.74 s", "e1_v3_scale @ 98.74 s"]}}

Full E1 consistency (assumption-based bracket, not a measurement): {"basis": "ASSUMPTION: every cell's pure-decode iterations have between the lowest and the highest measured rows per iteration (b=48 and b=8 at 32k), and its prefill-carrying iterations between the lowest and highest measured for such iterations; iterations per wave = OUT_LEN + b x prefill chunks (1, 2, 6 for 4k, 32k, 96k), which reproduces the two measured cells", "iteration_formula_check_on_measured_cells": {"b8": {"formula_iterations": 1040, "measured_iterations": 1040}, "b48": {"formula_iterations": 1120, "measured_iterations": 1120}}, "decode_rows_per_iteration_bracket": [5537.25, 5638.98], "prefill_rows_per_iteration_bracket": [6229, 7301], "waves": 36, "iterations": 38214, "kernel_rows_low": 212534334, "kernel_rows_high": 217731709, "earlier_estimate": 216000000, "by_kv": {"4096": {"iterations": 12438, "rows_low": 68976078, "rows_high": 70386936}, "32768": {"iterations": 12588, "rows_low": 69910428, "rows_high": 71482086}, "98304": {"iterations": 13188, "rows_low": 73647828, "rows_high": 75862686}}}
