# E1 identification statistics — e1_b8_recovery

| b | KV | phase | n | K | wall med ms | f_ind | CI90 | rho | c_common | arr. spread p50/p90 ms | last rank hist | coll. end spread p50/p90 ms | MoE xrank corr | R2 wall~routing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 32768 | pure | 1008 | 163 | 5.283 | 0.79 | 0.77–0.80 | 0.30 | 0.23 | 0.015/0.021 | [7, 51, 544, 406] | 0.0017/0.0082 | 0.68 | 0.06 |

## Frozen tables (calibration rows only, via the committed coefficients)

| b | status | n | K | f_ind |
|---|---|---|---|---|
| 8 | measured | 1038 | 163 | 0.77 |

## Decision statistics

```
{
 "regime": {},
 "kv": {},
 "pass": {
  "8": {
   "f_frozen": 0.7748,
   "frozen_sources": [
    "diverse_nsys_run2"
   ],
   "f_e1": {
    "32768": 0.7913
   },
   "delta_max_abs": 0.0165
  }
 }
}
```
