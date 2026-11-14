# phase1c/e1 — E1 authorization packet (decode routing and rank-arrival capture)

Everything needed to review, and later run, GPU experiment E1 of `../GPU-EXPERIMENT-MATRIX.md` under the frozen
`../VALIDATION-PROTOCOL.md`. **Nothing here has been executed on a GPU, and the launcher patch is not applied to any
branch.** Read `E1-AUTHORIZATION-PACKET.md` first; its section 12 lists the decisions requested with the go.

| path | what |
|---|---|
| `E1-AUTHORIZATION-PACKET.md` | the packet (ten review items, deviations, verbatim matrix / protocol excerpts) |
| `make_e1_inputs.py` → `traces/`, `configs/`, `E1-INPUTS.json`, `INPUTS.sha256` | deterministic request traces and launcher profile configs |
| `preflight_e1_host.sh` | read-only host preflight (run on the tray as dc) |
| `run_e1.sh <pass>` | the driver; refuses to start unless the preflight passes; one pass per invocation |
| `checks/e1_completeness.py` | correctness / completeness checks, dry-run overhead comparison |
| `analysis/e1_identify.py` | pre-registered estimators (no fitting) |
| `make_partition_1c.py`, `collect_e1.sh` | post-capture bookkeeping (partition file, artifact collection) |
| `patch/` | the 46-line launcher opt-in for recording routes beside a timing pass, its tests and applier |
| `out/` | CPU smoke outputs on frozen calibration rows, tray16 preflight log |
| `SHA256SUMS` | hashes of every file above (`sha256sum -c SHA256SUMS`) |

Regenerate / re-check on tray14 (CPU only):

```
export HOME=/mnt/shared/home/dc; S=$HOME/workspace/serving-studio; D=$S/ServingStudioSim/profiling/env/arm64-gb300/dsv41-flash-tp4
cd $D/phase1c/e1 && python3 make_e1_inputs.py --out . && sha256sum -c INPUTS.sha256
$S/gb300/venv/bin/python analysis/e1_identify.py --tables $S/phase1a-tables/quadrant_nsys --label quadrant_nsys --calibration-only $D/phase1a/partition/partition.json --out out/smoke_quadrant --boot 50
$S/gb300/venv/bin/python checks/e1_completeness.py --profile $S/artifacts/phase0-dsv41-flash-gb300-tp4-20261112/run/quadrant_nsys/profile --parsed --out out/smoke_quadrant_completeness.json
```
