# Addendum to the V1 run record: the load generator was rebuilt at launch (found 2026-11-18 06:01Z)

This addendum adds a fact that was not known when `runs/v1/run/E1-V1-RUN-REPORT.md` was written. It does not change that report,
the pinned gates, the original verdict (V1 **FAIL / STOP**) or the retrospective assessment record (`runs/v1/replay/`).

## Fact

The V1 run did not execute the pinned `session_runner` binary. The launcher's own `cargo build` step rebuilt it at launch.

| evidence | value |
|---|---|
| V1 driver `start` line (02:53:23Z), hash taken before the launcher | `session_runner=7f6eb81d45e6b98f` (the pinned binary) |
| V1 `run.log` lines 4 onward, directly after the `start` line | `Compiling proc-macro2 v1.0.106`, … (a full cargo build of the load generator) |
| binary mtime in `wt-lifecycle` | 2026-11-18 02:54:12Z (`launch_started_at` is 02:54:13.264Z) |
| binary sha256 in `wt-lifecycle` since then | 95a1526c3a841eebf84d54eafa254cf24590f847fbba233603dc8f0e158aa549 |
| pinned sha256 (`BINARY-PROVENANCE.json`; still the binary in `wt-e1stage1`) | 7f6eb81d45e6b98f3b6ebb56ab20b60ce8aaf4584aa3d4f25e3449b2d05a00c2 |
| source of both | req-frontend 5c20a259618976ecc360bda84abcc3653a04006e, clean in both worktrees |
| toolchain | cargo 1.98.1 (797e8a9bc), rustc 1.98.1 (48a229cea), the provenance toolchain |
| size | 28 759 968 bytes, both |
| difference | 273 451 differing bytes; each binary contains its own worktree path; two scratch builds of the same commit in two other paths gave two further, different hashes |

## Cause

The lifecycle worktree was provisioned on 2026-11-17 by copying the binary alone, without cargo's `target/` build state. The launcher
runs `cargo build` before a capture; cargo found no fingerprints and rebuilt. Preparation error of the operator (Claude).

## What it means

- V1's prerequisite "all source, binary, recipe, input and script hashes match" was true at the preflight (02:51Z) and at the driver's
  own preflight, and false from 02:54:12Z on. The run used a load generator built from the pinned **source**, not the pinned **bytes**.
- The V1 trace-side results do not depend on the load generator's bytes for their internal consistency checks (56/56 requests, joins,
  iteration and kernel counts), and those counts equal the sum of the two accepted single-wave captures. That is an observation, not a
  proof that the two binaries behave identically.
- The retrospective evidence assessment did not check this either. Its "PASS, qualified" carries this as one more qualification.

## Consequence for V2

The read-only V2 preflight fails on this hash and stays failing until the pin is decided (packet §12). V2's driver and post-run now
detect a rebuild at launch (gate `S.binary_unchanged`).
