# Iteration census: e1_b8_recovery

Replay span 9.613 s; EngineCore metrics iterations 1040 (index 85..1124).

| kind (all metrics iterations) | n |
|---|---|
| decode_ramp_b8 | 15 |
| mixed_b8 | 15 |
| prefill_only | 2 |
| pure_b8 | 1008 |

| wave | first iteration (start s) | first pure (start s) | last pure (end s) | wave end s | pure n | owned n |
|---|---|---|---|---|---|---|
| b8 | 85 (0.038) | 102 (3.615) | 1109 (8.948) | 9.022 | 1008 | 1040 |

## Captured (parsed.json): 1040 iterations, index 85..1124, contiguous=True, missing in metrics=[]

nsys block: `{"executable": "/usr/local/bin/nsys", "capture_mode": "cuda_profiler_api", "cuda_graph_trace": "node", "cuda_event_trace": false}`; first captured iteration starts at 0.0384 s on the replay clock; previous metrics iteration None

| kind | captured n |
|---|---|
| decode_ramp_b8 | 15 |
| mixed_b8 | 15 |
| prefill_only | 2 |
| pure_b8 | 1008 |

| request ownership (cells of the decode requests in flight) | n |
|---|---|
| - | 2 |
| b8_kv32k | 1038 |

parser stage field: {'prefill': 2, 'mixed': 15, 'decode': 1023}
JIT-affected captured iterations (index, kind): []
