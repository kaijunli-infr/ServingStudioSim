# E1 Stage 1 dry-run pair: routing hook OFF vs ON

off: `/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15/e1_dry_off_v2/profile`
on:  `/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15/e1_dry_on_v2/profile`

## Engine iteration time, pure-decode iterations (observed_elapsed_ms)

| cell | n off | n on | median off | median on | ratio on/off | boot 95 % CI | mean off | mean on | std off | std on | p10/p90 off | p10/p90 on | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b8_kv32768_w0 | 1008 | 1008 | 4.718 | 4.788 | 1.0149 | [1.0137, 1.0161] | 4.717 | 4.788 | 0.167 | 0.093 | 4.658/4.776 | 4.726/4.843 | PASS |
| b48_kv32768_w0 | 928 | 928 | 9.023 | 8.488 | 0.9406 | [0.9393, 0.9417] | 9.027 | 8.489 | 0.155 | 0.144 | 8.886/9.153 | 8.367/8.601 | PASS |

pure-decode iterations per cell vs planned minimum: b8_kv32768_w0: off 1008 / on 1008 (need >= 400), b48_kv32768_w0: off 928 / on 928 (need >= 400) -> PASS

**Overhead gate (median ratio <= 1.02 per cell, bootstrap CI must also lie <= 1.02 for PASS): PASS**

## Request-level comparison (replay.jsonl, paired by request id)

requests: off 56, on 56, paired 56, unpaired 0

| cell | n | TTFT med off/on (ms) | TTFT p90 off/on | total dur med off/on (ms) | dur p90 off/on | dur ratio on/off | out tokens off/on (all == OUT_LEN?) | output_preview identical |
|---|---|---|---|---|---|---|---|---|
| b48_kv32768_w0 | 48 | 31867.7/31627.6 | 32117.3/31880.5 | 31869.4/31630.3 | 32119.0/31883.3 | 0.9925 | [1024]/[1024] (yes) | 13/48 |
| b8_kv32768_w0 | 8 | 8862.4/9030.8 | 8886.1/9054.8 | 8864.4/9033.4 | 8887.7/9057.6 | 1.0191 | [1024]/[1024] (yes) | 4/8 |

token counts == OUT_LEN for every request on both sides and all requests paired: PASS
output_preview text identical off vs on for every paired request: DIFFERENT (previews are text prefixes, not token ids; identity is reported, not gated)

## Throughput (engine record: emitted tokens / summed pure-decode iteration time)

| cell | tok/s off | tok/s on | ratio on/off |
|---|---|---|---|
| b8_kv32768_w0 | 1696.0 | 1670.7 | 0.9851 |
| b48_kv32768_w0 | 5317.5 | 5654.1 | 1.0633 |

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
  "total_duration_ms_sum": 1599368.51,
  "total_duration_ms_avg": 28560.151964285713,
  "total_duration_ms_p50": 31802.8,
  "total_duration_ms_p90": 32110.875,
  "total_duration_ms_max": 32159.72,
  "ttft_ms_avg": 28558.404285714285,
  "ttft_ms_p50": 31801.2,
  "ttft_ms_p90": 32109.22,
  "ttft_ms_max": 32157.64,
  "ttft_token_id_steps": 56,
  "ttft_text_fallback_steps": 0,
  "run_duration_ms": 50788.989543914795,
  "request_throughput_per_s": 1.1026011838959602,
  "output_token_throughput_per_s": 1129.0636123094632,
  "tpot_measured_steps": 0,
  "tpot_ms_avg": null,
  "tpot_ms_p50": null,
  "tpot_ms_p90": null,
  "tpot_ms_max": null,
  "completion_amortized_tpot_measured_steps": 56,
  "completion_amortized_tpot_ms_avg": 0.0017083857003212294,
  "completion_amortized_tpot_ms_p50": 0.0016275659824037674,
  "completion_amortized_tpot_ms_p90": 0.0020625610948188395,
  "completion_amortized_tpot_ms_max": 0.002649071358747925,
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
  "total_duration_ms_sum": 1589287.1300000001,
  "total_duration_ms_avg": 28380.127321428572,
  "total_duration_ms_p50": 31562.58,
  "total_duration_ms_p90": 31874.985,
  "total_duration_ms_max": 31926.99,
  "ttft_ms_avg": 28377.557857142852,
  "ttft_ms_p50": 31560.045,
  "ttft_ms_p90": 31872.405,
  "ttft_ms_max": 31924.5,
  "ttft_token_id_steps": 56,
  "ttft_text_fallback_steps": 0,
  "run_duration_ms": 50556.05435371399,
  "request_throughput_per_s": 1.1076813789343132,
  "output_token_throughput_per_s": 1134.2657320287367,
  "tpot_measured_steps": 0,
  "tpot_ms_avg": null,
  "tpot_ms_p50": null,
  "tpot_ms_p90": null,
  "tpot_ms_max": null,
  "completion_amortized_tpot_measured_steps": 56,
  "completion_amortized_tpot_ms_avg": 0.0025116952939530615,
  "completion_amortized_tpot_ms_p50": 0.0024877810361664263,
  "completion_amortized_tpot_ms_p90": 0.002922776148581498,
  "completion_amortized_tpot_ms_max": 0.003968719452589923,
  "context_overflow_steps": 0
 }
}
```

## Verdicts

```
{
 "overhead_gate": "PASS",
 "pure_decode_counts_vs_plan": "PASS",
 "token_counts": "PASS",
 "output_preview_identity": "DIFFERENT"
}
```
