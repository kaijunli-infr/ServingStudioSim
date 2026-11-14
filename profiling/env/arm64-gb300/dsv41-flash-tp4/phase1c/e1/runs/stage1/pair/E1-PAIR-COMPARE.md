# E1 Stage 1 dry-run pair: routing hook OFF vs ON

off: `/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/20261114_dsv41_flash_gb300_tp4_e1/e1_dry_off/profile`
on:  `/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/20261114_dsv41_flash_gb300_tp4_e1/e1_dry_on/profile`

## Engine iteration time, pure-decode iterations (observed_elapsed_ms)

| cell | n off | n on | median off | median on | ratio on/off | boot 95 % CI | mean off | mean on | std off | std on | p10/p90 off | p10/p90 on | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b8_kv32768_w0 | 1008 | 1008 | 4.711 | 4.751 | 1.0086 | [1.0075, 1.0097] | 4.711 | 4.738 | 0.135 | 0.193 | 4.654/4.771 | 4.632/4.815 | PASS |
| b48_kv32768_w0 | 928 | 928 | 8.627 | 8.795 | 1.0195 | [1.0180, 1.0209] | 8.634 | 8.796 | 0.164 | 0.159 | 8.496/8.771 | 8.657/8.921 | INCONCLUSIVE |

pure-decode iterations per cell vs planned minimum: b8_kv32768_w0: off 1008 / on 1008 (need >= 400), b48_kv32768_w0: off 928 / on 928 (need >= 400) -> PASS

**Overhead gate (median ratio <= 1.02 per cell, bootstrap CI must also lie <= 1.02 for PASS): INCONCLUSIVE**

## Request-level comparison (replay.jsonl, paired by request id)

requests: off 56, on 56, paired 56, unpaired 0

| cell | n | TTFT med off/on (ms) | TTFT p90 off/on | total dur med off/on (ms) | dur p90 off/on | dur ratio on/off | out tokens off/on (all == OUT_LEN?) | output_preview identical |
|---|---|---|---|---|---|---|---|---|
| b48_kv32768_w0 | 48 | 11209.7/31904.9 | 19644.8/32155.5 | 31526.3/31907.4 | 31779.4/32157.9 | 1.0121 | [1024]/[1024] (yes) | 15/48 |
| b8_kv32768_w0 | 8 | 2266.6/8997.7 | 3454.4/9021.5 | 8868.1/9000.8 | 8891.0/9024.2 | 1.0150 | [1024]/[1024] (yes) | 6/8 |

token counts == OUT_LEN for every request on both sides and all requests paired: PASS
output_preview text identical off vs on for every paired request: DIFFERENT (previews are text prefixes, not token ids; identity is reported, not gated)

## Throughput (engine record: emitted tokens / summed pure-decode iteration time)

| cell | tok/s off | tok/s on | ratio on/off |
|---|---|---|---|
| b8_kv32768_w0 | 1698.1 | 1688.6 | 0.9944 |
| b48_kv32768_w0 | 5559.1 | 5456.8 | 0.9816 |

replay summary.json (common):
```
{
 "off": {
  "attempted_steps": 56,
  "success_steps": 56,
  "failed_steps": 0,
  "target_output_tokens": 57344,
  "actual_output_tokens": 57344,
  "output_mismatch_steps": 0,
  "output_token_delta": 0,
  "total_duration_ms_sum": 1583008.7500000002,
  "total_duration_ms_avg": 28268.01339285715,
  "total_duration_ms_p50": 31459.05,
  "total_duration_ms_p90": 31771.155,
  "total_duration_ms_max": 31819.05,
  "ttft_ms_avg": 9931.77267857143,
  "ttft_ms_p50": 9418.43,
  "ttft_ms_p90": 19286.015,
  "ttft_ms_max": 21689.77,
  "ttft_token_id_steps": 56,
  "ttft_text_fallback_steps": 0,
  "run_duration_ms": 50447.87263870239,
  "request_throughput_per_s": 1.1100567193598199,
  "output_token_throughput_per_s": 1136.6980806244555,
  "tpot_measured_steps": 56,
  "tpot_ms_avg": 17.922478005865102,
  "tpot_ms_p50": 18.1686265884653,
  "tpot_ms_p90": 27.38487781036168,
  "tpot_ms_max": 29.8688954056696,
  "completion_amortized_tpot_measured_steps": 56,
  "completion_amortized_tpot_ms_avg": 17.923988968021224,
  "completion_amortized_tpot_ms_p50": 18.17006842619746,
  "completion_amortized_tpot_ms_p90": 27.386578690127074,
  "completion_amortized_tpot_ms_max": 29.87068426197459,
  "context_overflow_steps": 0
 },
 "on": {
  "attempted_steps": 56,
  "success_steps": 56,
  "failed_steps": 0,
  "target_output_tokens": 57344,
  "actual_output_tokens": 57344,
  "output_mismatch_steps": 0,
  "output_token_delta": 0,
  "total_duration_ms_sum": 1602281.7700000003,
  "total_duration_ms_avg": 28612.17446428572,
  "total_duration_ms_p50": 31840.765,
  "total_duration_ms_p90": 32150.1,
  "total_duration_ms_max": 32204.58,
  "ttft_ms_avg": 28609.44410714286,
  "ttft_ms_p50": 31837.98,
  "ttft_ms_p90": 32147.53,
  "ttft_ms_max": 32201.93,
  "ttft_token_id_steps": 56,
  "ttft_text_fallback_steps": 0,
  "run_duration_ms": 50833.640575408936,
  "request_throughput_per_s": 1.1016326858771222,
  "output_token_throughput_per_s": 1128.071870338173,
  "tpot_measured_steps": 0,
  "tpot_ms_avg": null,
  "tpot_ms_p50": null,
  "tpot_ms_p90": null,
  "tpot_ms_max": null,
  "completion_amortized_tpot_measured_steps": 56,
  "completion_amortized_tpot_ms_avg": 0.0026689708141322236,
  "completion_amortized_tpot_ms_p50": 0.002624633431086324,
  "completion_amortized_tpot_ms_p90": 0.0031280547409569,
  "completion_amortized_tpot_ms_max": 0.003636363636361218,
  "context_overflow_steps": 0
 }
}
```

## Verdicts

```
{
 "overhead_gate": "INCONCLUSIVE",
 "pure_decode_counts_vs_plan": "PASS",
 "token_counts": "PASS",
 "output_preview_identity": "DIFFERENT"
}
```
