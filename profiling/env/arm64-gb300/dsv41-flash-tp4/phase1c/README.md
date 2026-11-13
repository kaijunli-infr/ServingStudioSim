# Phase 1C-0 — CPU-only preparation for validating the Phase 1B sequence model

Branch `phase1c-prep`, built on Phase 1B `29082793` (frozen). Nothing here runs on a GPU, launches vLLM, writes a
profiling database, changes a preset or touches Rust code. The Phase 1B engine (`../phase1b/seqsim`) is used as a
black box and its sources, parameters and predictions are pinned by hash (`tests/test_invariants.py`,
`repro/MANIFEST.json`). See `PHASE1C0-REPORT.md`.

```
phase1c/
  PHASE1C0-REPORT.md              report: what was done, results, exact commits, CPU tests
  repro/                          reproducibility of the Phase 1B outputs
    MANIFEST.json                 inputs (tables per file, coefficients, partition, params, sources) with sha256; environment;
                                  exact command and seed; output schema, row counts, canonical content checksums
    make_manifest.py              writes MANIFEST.json from the frozen inputs/outputs
    verify.py / regenerate.sh     regenerate everything into a scratch dir and compare (canonical checksums, params hash, every metric)
    VERIFY.json                   result of the last verification (PASS)
    common.py                     canonical parquet checksum (encoder-independent), hashing, environment
  tests/test_invariants.py        14 deterministic CPU invariant checks of the sequence model (synthetic + real frozen inputs)
  uncertainty/
    reliability.py                uncertainty audit: per class and seed, interval displacement decomposition, table-draw analysis
    RELIABILITY.md / reliability.json   generated tables (5 seeds × 32 draws × 3 passes)
    FINDINGS.md                   interpretation of the audit
  GPU-EXPERIMENT-MATRIX.md        six independently approvable captures (none authorised)
  VALIDATION-PROTOCOL.md          frozen protocol (hash in PROTOCOL.sha256)
  RUST-INTEGRATION-INTERFACE.md   design of the opt-in Rust hook (no code)
  schema/                         JSON Schemas of the request/response contract
  out/logs/                       verify.log, reliability.log, pytest logs
```

## Reproduce (rack 10 tray14, `export HOME=/mnt/shared/home/dc`)

```bash
PY=$HOME/workspace/serving-studio/gb300/venv/bin/python
T=$HOME/workspace/serving-studio/phase1a-tables           # or the rack-3 mirror /mnt/shared/home/kevin/dsv41-phase1-mirror/phase1a-tables
cd $HOME/workspace/serving-studio/ServingStudioSim/profiling/env/arm64-gb300/dsv41-flash-tp4/phase1c
$PY -m pytest -q tests                                    # 14 invariant tests, < 2 s
TABLES=$T PY=$PY repro/regenerate.sh                     # regenerate Phase 1B outputs and verify (~4 min)
$PY uncertainty/reliability.py --tables $T --seeds 5     # reliability tables (~7 min)
```

Second storage location of the inputs that are not committed (compact tables, 175 MB) and of the Phase 1B per-iteration
parquet outputs (15 MB): rack 3 `/mnt/shared/home/kevin/dsv41-phase1-mirror/` with `SHA256SUMS.mirror` (24 files,
verified). Primary copies: tray14 `phase1a-tables/` and `phase1b/out/`.
