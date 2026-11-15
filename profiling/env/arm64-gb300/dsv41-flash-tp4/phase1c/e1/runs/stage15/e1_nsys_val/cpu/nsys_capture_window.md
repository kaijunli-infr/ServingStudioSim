# nsys capture window vs the dry trace

| pass | metrics iterations | median pure-decode iteration ms b=8 | b=48 |
|---|---|---|---|
| e1_dry_off_v2 | 2160 | 4.718 | 9.023 |
| e1_dry_on_v2 | 2160 | 4.788 | 8.488 |
| e1_dry_off2_v2 | 2160 | 4.704 | 9.027 |
| e1_nsys_val | 2160 | 4.684 | 8.527 |

nsys pass metrics iterations 181..2342 (2160); engine time (sum of observed_elapsed_ms) 38.40 s
pure b=8 iterations 198..2328 (1010): engine time 3.65 s .. 38.34 s
pure b=48 iterations 1320..2247 (928): engine time 29.85 s .. 38.40 s
parsed (captured) iterations 1223..2342 (1120): engine time at first parsed iteration 8.44 s
metrics iterations before the first parsed iteration: 1040 (pure b=8 among them: 1008 of 1010; pure b=48: 0 of 928)

trace `/mnt/shared/home/dc/workspace/serving-studio/wt-e1stage1/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c/e1/traces/e1_dry.csv`: 56 requests, columns ['id', 'input_len', 'output_len', 'arrival_time']
trace group b8: n=8, arrival_time min 0 max 0
trace group b48: n=48, arrival_time min 18627 max 18627
