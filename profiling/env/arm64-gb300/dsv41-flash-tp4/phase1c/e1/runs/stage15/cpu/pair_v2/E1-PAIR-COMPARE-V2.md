# E1 paired comparison v2 (serial-correlation-aware)

off: `/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/20261114_dsv41_flash_gb300_tp4_e1/e1_dry_off/profile`
on: `/mnt/shared/home/dc/workspace/serving-studio/ServingStudioSim/logs/20261114_dsv41_flash_gb300_tp4_e1/e1_dry_on/profile`

## Pure-decode iteration time: diagnostics and overhead statistics

| cell | arm | n | median | mean | std | acf1 | acf2 | acf5 | acf10 | tau_int | block L | first/second half median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b8_kv32768_w0 | off | 1008 | 4.711 | 4.711 | 0.135 | -0.095 | 0.043 | 0.008 | 0.032 | 1.0 | 10 | 4.704/4.717 |
| b8_kv32768_w0 | on | 1008 | 4.751 | 4.738 | 0.193 | -0.171 | 0.07 | 0.078 | 0.066 | 1.0 | 10 | 4.756/4.742 |
| b48_kv32768_w0 | off | 928 | 8.627 | 8.634 | 0.164 | 0.042 | 0.247 | -0.009 | 0.134 | 1.58 | 10 | 8.610/8.640 |
| b48_kv32768_w0 | on | 928 | 8.795 | 8.796 | 0.159 | 0.165 | 0.198 | 0.103 | 0.13 | 5.32 | 27 | 8.819/8.770 |

## Overhead gates per cell

| cell | on/off median ratio | pre-registered point gate (<= 1.02) | block-bootstrap 95 % CI | prospective criterion | control/off | control CI (noise flag) | excess on vs control [CI] |
|---|---|---|---|---|---|---|---|
| b8_kv32768_w0 | 1.0086 | PASS | [1.0066, 1.0100] | PASS | - | - | - |
| b48_kv32768_w0 | 1.0195 | PASS | [1.0169, 1.0220] | INCONCLUSIVE | - | - | - |

**Pre-registered point gate: PASS. Prospective block-bootstrap criterion: INCONCLUSIVE.**

## Requests

paired 56; output_preview identical 21/56; chunk_count values off [1024, 1025] on [1] (equal sets = same wire mode); median per-request duration ratio on/off 1.0137

## Verdicts

```
{
 "point_gate": "PASS",
 "prospective_block_bootstrap": "INCONCLUSIVE"
}
```
