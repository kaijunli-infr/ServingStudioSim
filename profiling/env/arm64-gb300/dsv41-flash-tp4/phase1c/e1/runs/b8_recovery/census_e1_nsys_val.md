# Iteration census: e1_nsys_val

Replay span 51.197 s; EngineCore metrics iterations 2160 (index 181..2342).

| kind (all metrics iterations) | n |
|---|---|
| decode_ramp_b48 | 95 |
| decode_ramp_b8 | 15 |
| mixed_b48 | 95 |
| mixed_b8 | 15 |
| prefill_only | 4 |
| pure_b48 | 928 |
| pure_b8 | 1008 |

| wave | first iteration (start s) | first pure (start s) | last pure (end s) | wave end s | pure n | owned n |
|---|---|---|---|---|---|---|
| b8 | 181 (0.033) | 198 (3.723) | 1205 (8.984) | 9.055 | 1008 | 1040 |
| b48 | 1223 (18.673) | 1320 (40.387) | 2247 (49.953) | 50.713 | 928 | 1120 |

## Captured (parsed.json): 1120 iterations, index 1223..2342, contiguous=True, missing in metrics=[]

nsys block: `{"executable": "/usr/local/bin/nsys", "capture_mode": "cuda_profiler_api", "cuda_graph_trace": "node", "cuda_event_trace": false, "capture_start_delay_seconds": 10.0}`; first captured iteration starts at 18.6727 s on the replay clock; previous metrics iteration {'index': 1220, 'end_s': 9.0553, 'kind': 'decode_ramp_b8'}

| kind | captured n |
|---|---|
| decode_ramp_b48 | 95 |
| mixed_b48 | 95 |
| prefill_only | 2 |
| pure_b48 | 928 |

| request ownership (cells of the decode requests in flight) | n |
|---|---|
| - | 2 |
| b48_kv32k | 1118 |

parser stage field: {'prefill': 2, 'mixed': 95, 'decode': 1023}
JIT-affected captured iterations (index, kind): [(1271, 'mixed_b48'), (1287, 'mixed_b48'), (1303, 'mixed_b48'), (1319, 'mixed_b48')]
