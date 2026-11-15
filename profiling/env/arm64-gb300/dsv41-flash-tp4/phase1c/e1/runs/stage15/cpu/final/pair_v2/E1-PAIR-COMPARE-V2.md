# E1 paired comparison v2 (serial-correlation-aware)

off: `/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15/e1_dry_off_v2/profile`
on: `/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15/e1_dry_on_v2/profile`
control: `/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/dsv41_flash_gb300_tp4_e1_stage15/e1_dry_off2_v2/profile`

## Pure-decode iteration time: diagnostics and overhead statistics

| cell | arm | n | median | mean | std | acf1 | acf2 | acf5 | acf10 | tau_int | block L | first/second half median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b8_kv32768_w0 | off | 1008 | 4.718 | 4.717 | 0.167 | -0.128 | 0.015 | -0.009 | 0.01 | 1.0 | 10 | 4.723/4.712 |
| b8_kv32768_w0 | on | 1008 | 4.788 | 4.788 | 0.093 | 0.016 | 0.035 | 0.029 | 0.041 | 1.83 | 10 | 4.779/4.795 |
| b8_kv32768_w0 | control | 1008 | 4.704 | 4.704 | 0.195 | -0.233 | 0.034 | 0.004 | -0.001 | 1.0 | 10 | 4.697/4.712 |
| b48_kv32768_w0 | off | 928 | 9.023 | 9.027 | 0.155 | 0.13 | 0.272 | 0.065 | 0.149 | 6.24 | 32 | 9.066/8.983 |
| b48_kv32768_w0 | on | 928 | 8.488 | 8.489 | 0.144 | 0.153 | 0.143 | 0.024 | 0.038 | 2.2 | 11 | 8.504/8.470 |
| b48_kv32768_w0 | control | 928 | 9.027 | 9.032 | 0.142 | 0.021 | 0.15 | -0.031 | 0.108 | 1.34 | 10 | 9.018/9.042 |

## Overhead gates per cell

| cell | on/off median ratio | pre-registered point gate (<= 1.02) | block-bootstrap 95 % CI | prospective criterion | control/off | control CI (noise flag) | excess on vs control [CI] |
|---|---|---|---|---|---|---|---|
| b8_kv32768_w0 | 1.0149 | PASS | [1.0136, 1.0162] | PASS | 0.9971 | [0.9956, 0.9988] (quiet) | 1.0178 [1.0158, 1.0200] |
| b48_kv32768_w0 | 0.9406 | PASS | [0.9379, 0.9432] | PASS | 1.0004 | [0.9980, 1.0030] (quiet) | 0.9402 [0.9365, 0.9436] |

**Pre-registered point gate: PASS. Prospective block-bootstrap criterion: PASS.**

## Requests

paired 56; output_preview identical 17/56; chunk_count values off [1] on [1] (equal sets = same wire mode); median per-request duration ratio on/off 0.9938

control vs off: output_preview identical 22/56 (run-to-run token determinism)

## Verdicts

```
{
 "point_gate": "PASS",
 "prospective_block_bootstrap": "PASS"
}
```
