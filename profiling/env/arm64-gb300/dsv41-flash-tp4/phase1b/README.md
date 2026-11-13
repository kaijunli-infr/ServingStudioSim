# Phase 1B — opt-in TP4 sequence simulator (`seqsim`)

CPU-only sequence simulation of DSV41-flash TP4 iterations on GB300, built on the unchanged Phase 1A
`costmodel` package (`../phase1a`). Nothing here launches vLLM, touches a GPU, writes a profiling
database or changes a preset. The path is opt-in: it is a separate Python package with its own CLI;
the Rust simulator and every Phase 0 / Phase 1A artifact are untouched (tested, see
`tests/test_seqsim.py::test_phase1a_and_simulator_sources_untouched`). See `PHASE1B-REPORT.md`.

## What it models

Each rank has a **host timeline** and a **GPU timeline**; iterations are coupled by explicit rules
with measured parameters (`seqsim/params.py`):

| rule | parameter(s) | status |
|---|---|---|
| host serial order per rank: `host_start_r(i) >= host_end_r(i-1) + a_host` | `a_host[r]` | measured |
| scheduler one-step-ahead: `host_start_0(i) >= GPU_end(i-2) + eps_sched[kind_i, b_{i-2}]` | `eps_sched` table | measured |
| scheduler broadcast: `host_start_r(i) >= host_start_0(i)` | — | structural |
| first kernel `>= host_start + f_idle`; forward launches start at `host_start + h_pre[kind][r]` | `f_idle`, `h_pre` | measured |
| launch queue: launch j completes `>=` GPU completion of launch `j - Q` | `Q` | measured lower bound |
| host end = last launch + `p_post[kind][r]` | `p_post` | measured |
| decode (CUDA graph): one launch block, GPU path from the Phase 1A per-batch tables (stochastic draws) | Phase 1A `decode` | measured |
| eager / ring prefill and mixed: per-rank launch-paced compute windows (Phase 1A prefill coefficients × measured segment profile), collectives with explicit barriers (`completion = max_r arrival + intrinsic`) | Phase 1A `prefill`, `collectives`, `mixed`; `profile[kind]` | measured / inferred |
| MoE routing uncertainty: Phase 1A corpus band applied to the MoE share of the decode path (stochastic mode) | Phase 1A `moe` | statistical, no per-iteration routing |
| KV dependence: indexer/sparse-meta via Phase 1A coefficients; `kv-extrapolation` status beyond kv_max 131,063 (decode) or KVW 98.2 (prefill) | Phase 1A ranges | flagged |

Iteration completion is the critical path (max over ranks of the tail window end). No rank sum, no
global multiplier anywhere (asserted by a test on the engine source).

## Layout

```
phase1b/
  seqsim/__init__.py    puts ../phase1a on sys.path and imports costmodel (unchanged)
  seqsim/params.py      SeqParams.fit (calibration role only, leakage guard) / eps_sched / decode_host
  seqsim/engine.py      Engine.run(schedule, measured, recon) -> per-iteration IterRecord (vectorised over draws)
  seqsim/evaluate.py    modes recon-gpu / recon-seq / pred-det / pred-mc, metrics, coverage, workload, markdown
  seqsim/run.py         CLI (fit params -> params.json + PARAMS.sha256, evaluate all passes -> results.json, RESULTS.md)
  tests/test_seqsim.py  18 deterministic CPU tests (synthetic data)
  tools/sensitivity.py  sensitivity of the inferred parameters (queue throttle, delta_launch) on one pass
  out/                  params.json, PARAMS.sha256, results.json, RESULTS.md, logs/ (committed);
                        iterations_<pass>_<mode>.parquet (git-ignored repo-wide, regenerable, 15 MB)
```

## Reproduce (rack 10 tray14, `export HOME=/mnt/shared/home/dc`)

```bash
PY=$HOME/workspace/serving-studio/gb300/venv/bin/python
cd $HOME/workspace/serving-studio/ServingStudioSim/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1b
$PY -m seqsim.run --tables $HOME/workspace/serving-studio/phase1a-tables --partition ../phase1a/partition \
    --coeffs ../phase1a/out/coefficients.json --out out --draws 32 --seed 0 --git-rev $(git rev-parse HEAD)
$PY -m pytest -q tests
```

The compact tables are the Phase 1A ones (hashes in `../phase1a/out/coefficients.json`). The whole
run (3 passes × 4 modes, 32 draws) takes about 3 minutes on one CPU core. Runs are deterministic for
a given `--seed`.

## Using the engine

```python
import seqsim
from seqsim.engine import Engine
from seqsim.params import SeqParams
import json
eng = Engine(json.load(open("../phase1a/out/coefficients.json")), SeqParams.load("out/params.json"), draws=32, seed=0, stochastic=True, moe_band=True)
recs = eng.run(schedule)          # schedule: list of costmodel.data.Feat (structural features only); the first iteration needs a Measured anchor
for r in recs:
    r.t0, r.t1, r.wall, r.host_start, r.host_end, r.rank_end, r.rank_wait, r.lead, r.status, r.kv_status, r.binding
```

`recon="gpu"` / `recon="seq"` replay measured segments (engine validation); excluded (JIT) iterations
are always replayed from their measured timelines so the sequence stays aligned.
