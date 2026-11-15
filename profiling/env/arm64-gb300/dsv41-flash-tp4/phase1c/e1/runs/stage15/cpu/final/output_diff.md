# Output differences between runs (same trace, greedy, temperature 0)

Finest available comparison: `output_preview` (first 100 characters) plus token counts; full token sequences are not logged by the pre-registered load-generator configuration.

| run | requests | all SUCCESS | output_len values | finish_reason | chunk_count values | median duration ms |
|---|---|---|---|---|---|---|
| off | 56 | True | {1024: 56} | {'length': 56} | {1: 56} | 31811.3 |
| on | 56 | True | {1024: 56} | {'length': 56} | {1: 56} | 31570.0 |
| off2 | 56 | True | {1024: 56} | {'length': 56} | {1: 56} | 31866.8 |

| pair | paired | preview identical | preview differs | len differs | completion_tokens differs | chunk_count differs | differing requests (first 6) |
|---|---|---|---|---|---|---|---|
| off vs on | 56 | 17 | 39 | 0 | 0 | 0 | e1_b48_kv32k_w0_r02, e1_b48_kv32k_w0_r03, e1_b48_kv32k_w0_r04, e1_b48_kv32k_w0_r05, e1_b48_kv32k_w0_r07, e1_b48_kv32k_w0_r08 |
| off vs off2 | 56 | 22 | 34 | 0 | 0 | 0 | e1_b48_kv32k_w0_r00, e1_b48_kv32k_w0_r02, e1_b48_kv32k_w0_r03, e1_b48_kv32k_w0_r04, e1_b48_kv32k_w0_r05, e1_b48_kv32k_w0_r07 |
| on vs off2 | 56 | 25 | 31 | 0 | 0 | 0 | e1_b48_kv32k_w0_r00, e1_b48_kv32k_w0_r03, e1_b48_kv32k_w0_r04, e1_b48_kv32k_w0_r05, e1_b48_kv32k_w0_r07, e1_b48_kv32k_w0_r08 |

## Preview identity per cell

| pair | cell | identical / paired |
|---|---|---|
| off vs on | e1_b48_kv32k_w0 | 13 / 48 |
| off vs on | e1_b8_kv32k_w0 | 4 / 8 |
| off vs off2 | e1_b48_kv32k_w0 | 19 / 48 |
| off vs off2 | e1_b8_kv32k_w0 | 3 / 8 |
| on vs off2 | e1_b48_kv32k_w0 | 21 / 48 |
| on vs off2 | e1_b8_kv32k_w0 | 4 / 8 |
