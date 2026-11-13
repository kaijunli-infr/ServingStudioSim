# Phase 1A — DSV41 GB300 TP4 CPU-only cost model (`costmodel`)

CPU-only implementation and offline calibration of the Phase 1 cost-model proposal
(`phase1-planning` @ cc06d9c2), using only the frozen Phase 0 captures. Nothing here
launches vLLM, touches a GPU, writes a production profiling database or changes a public
preset. See `PHASE1A-REPORT.md` for results, provenance and the readiness recommendation.

## Layout

```
phase1a/
  costmodel/            pure-Python package (numpy / pandas / pyarrow only)
    groups.py           kernel-name → group (fmha, indexer, sparse_meta, moe, comm, other) and comm kind
    intervals.py        union / windowed-union / overlap accounting (compute vs comm, intra-rank idle)
    extract.py          frozen parsed.json + kernels.parquet + host_timeline.json → compact tables (CLI)
    partition.py        freeze the calibration / held-out partition (hash-locked) + leakage guard
    data.py             Feat / PassData: structural features, per-rank segments, host lead
    compose.py          TP4 per-rank ordered timelines → collective sync → critical path (module 5/6)
    decode.py           module 1: decode GPU execution per regime (≤16 / 17–24 / ≥25), stochastic composition
    collectives.py      module 5: collective intrinsic cost vs payload (eager twoshot, NCCL ring/all-gather)
    longctx.py          module 4: prefill compute per group/rank incl. indexer / sparse-meta KV dependence; 16k-chunk fit
    moe.py              module 3: MoE expert-GEMM statistical model from the routing corpus (central + uncertainty)
    mixed.py            module 2/7: mixed-eager host/GPU timeline, launch count, per-rank host cost, lead-aware wall
    host.py             module 7 evidence: inter-iteration gaps, decode host, launch back-pressure
    calibrate.py        fit everything on the calibration role only → out/coefficients.json (CLI)
    predict.py          Predictor(coefficients) → per-iteration prediction with status
    evaluate.py         held-out / calibration / cross-check metrics → out/results.json + RESULTS.md (CLI)
  tests/test_costmodel.py   15 CPU unit tests (synthetic data; no captures needed)
  partition/            frozen partition (partition.json + PARTITION.sha256)
  out/                  coefficients.json (+ sha256), results.json, RESULTS.md, logs/
  explore1.py, explore2.py, debug_eager.py   exploration scripts kept for the record (not part of the package)
```

Compact tables (`iterations.parquet`, `iter_rank.parquet`, `collectives.parquet`, `meta.json` per
pass) are **not** committed (164 MB); they live on rack 10 CephFS at
`/mnt/shared/home/dc/workspace/serving-studio/phase1a-tables/<label>/` and their sha256 are
recorded in `out/coefficients.json` → `provenance.tables`. They are reproducible from the frozen
captures with `extract.py` (hash-verified against the capture `SHA256SUMS`).

## Reproduce (rack 10 tray14, `export HOME=/mnt/shared/home/dc`)

```bash
PY=$HOME/workspace/serving-studio/gb300/venv/bin/python
REPO=$HOME/workspace/serving-studio/ServingStudioSim
P1=$REPO/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1a
A=$HOME/workspace/serving-studio/artifacts/phase0-dsv41-flash-gb300-tp4-20261112   # frozen Phase 0 artifact
TAB=$HOME/workspace/serving-studio/phase1a-tables
cd $P1

# 1. extract compact tables (≈2–4 min per pass, CPU only, verifies input hashes against $A/SHA256SUMS)
#    capture dirs: run/diverse_nsys_run2, run/quadrant_nsys, and run/diverse_nsys (= label diverse_nsys_salvaged)
$PY -m costmodel.extract --profile $A/run/diverse_nsys_run2/profile --label diverse_nsys_run2 --excl ../phase0/diverse_nsys_run2/jit_excl.txt \
    --out $TAB/diverse_nsys_run2 --sha256sums $A/SHA256SUMS --sha-prefix run/diverse_nsys_run2/profile/
$PY -m costmodel.extract --profile $A/run/quadrant_nsys/profile --label quadrant_nsys \
    --out $TAB/quadrant_nsys --sha256sums $A/SHA256SUMS --sha-prefix run/quadrant_nsys/profile/
$PY -m costmodel.extract --profile $A/run/diverse_nsys/profile --label diverse_nsys_salvaged \
    --out $TAB/diverse_nsys_salvaged --sha256sums $A/SHA256SUMS --sha-prefix run/diverse_nsys/profile/

# 2. freeze the partition (already frozen: partition/PARTITION.sha256 = 288e193f…; sources are the
#    Phase 1 planning outputs ../phase1/out/validation_split.json and routing_stats.json, hashes recorded)
#    $PY -c 'from costmodel.partition import materialize as m; m("../phase1/out/validation_split.json", "../phase1/out/routing_stats.json", "partition")'

# 3. calibrate (calibration role only; refuses held-out ids) and 4. evaluate
$PY -m costmodel.calibrate --tables $TAB --partition partition \
    --corpus $A/run/diverse_corpus/profile/token_corpus --out out --git-rev $(git rev-parse HEAD)
$PY -m costmodel.evaluate  --tables $TAB --partition partition --coeffs out/coefficients.json --out out

# 5. tests
$PY -m pytest -q tests
```

Determinism: `--seed` (default 0) controls the MoE bootstrap and the decode Monte-Carlo
composition (`MC_SEED`); two runs on the same tables produce byte-identical
`coefficients.json`.

## Using the predictor

```python
from costmodel.predict import Predictor
from costmodel.data import Feat
pr = Predictor.load("out/coefficients.json")
p = pr.predict(Feat(iteration=0, stage="decode", excluded=False, b=8, kv_max=60000, kv_sum=400000,
                    P=0, chunks=0, wall_ns=0, t0_ns=0, t1_ns=0, regime="b<=16", mixed_kind="",
                    is_chunk16k=False, kv0=0, chain=-1, role="", cls="", lead_ns=0, prev_stage=""))
p["wall_ns"], p["status"], p["kv_status"], p["rank_busy_ns"], p["rank_wait_ns"], p["groups_ns"]
```

Every prediction carries a `status` (`measured`, `interpolated`, `in-range`, `inferred`,
`extrapolation`, `measured-class`) and, for decode, a `kv_status`; nothing is silently
extrapolated. There is no global latency multiplier anywhere in the package.
