# session_runner provenance repair (CPU-only, 2026-11-18)

GO "CPU-only session_runner provenance repair and V2 readiness correction", option 2. No GPU was used.

| | sha256 | where |
|---|---|---|
| pinned executable (accepted reference; `BINARY-PROVENANCE.json`) | 7f6eb81d45e6b98f3b6ebb56ab20b60ce8aaf4584aa3d4f25e3449b2d05a00c2 | restored to `wt-lifecycle/alignment/load_generator/req-frontend/target/release/session_runner`; reference copy unchanged in `wt-e1stage1` |
| executable the V1 run used (rebuilt by the V1 launcher) | 95a1526c3a841eebf84d54eafa254cf24590f847fbba233603dc8f0e158aa549 | preserved, read-only, in `artifacts/phase1c-e1-v1-dsv41-flash-gb300-tp4-20261118-incident-session-runner/` with its provenance and checksums; **not re-pinned** |

Files here: `restore_tray14_20261118T0827Z.log` (preserve, restore, first cargo test), `cargo_state_reconstruction_tray14_20261118T0828Z.log`
(second cargo test), `cargo_after_restore.txt`, `cargo_after_reconstruction.txt`, `rebuilt-v1-executable-PROVENANCE.json`,
`rebuilt-v1-executable-SHA256SUMS`, `session_runner.build-state.json` (the record the launch contract requires).

Findings:
1. The launcher compiled because `run_profile` called `build_session_runner()` (a `cargo build`) before every capture; the preflight
   only hashed the binary.
2. Cargo's build state in the lifecycle worktree cannot be made compatible with the pinned bytes: every development `cargo build`
   there yields 95a1526c… again (deterministic for the path, different between paths). Two attempts are logged. The pinned bytes
   were restored after each.
3. Byte identity is therefore enforced by a launch path that never runs cargo and refuses to start unless the executable's hash, its
   source and its build-state record all match (packet §12; 17 CPU tests).

The V1 records are unchanged: original V1 FAIL / STOP; retrospective evidence assessment PASS, qualified (one qualification being
that V1 ran with the rebuilt executable).
