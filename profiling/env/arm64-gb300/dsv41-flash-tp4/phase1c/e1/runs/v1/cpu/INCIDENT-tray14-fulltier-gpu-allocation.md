# Unintended GPU allocation during the whole-tier test run on tray14 (2026-11-17, ≈ 20:41–20:43Z)

**What this was: a test run, not a profiling experiment.** No Nsight process, no vLLM server, no trace, no profile
directory, no lifecycle controller and no V1 artifact were involved. The GPU was touched by `torch` inside the pytest
process of the repository's whole test tier. It was also **not authorized**: the standing rule is that any GPU start is
shown (matrix, duration, trays) and gets an explicit go, and the GO of 2026-11-17 authorized CPU-only work. This is
recorded as a rule breach of the operator (Claude), unintended, with the remediation below.

## Facts (from `runs/v1/cpu/pytest_fullsuite_tray14_pre-amend-8eda99fc_code-trees-identical.log` and the tray)

| item | value |
|---|---|
| host | pod4-gb300-10-tray14-f3 (rack 10, Kubernetes node; GPU 0 of 4) |
| command | `uv run --no-sync python -m pytest -q` of the whole tier in `wt-lifecycle` at the pre-amend tree 8eda99fc (code trees identical to ece6dacf), through the scratch runner `run_tests_wl.sh`, **without** `CUDA_VISIBLE_DEVICES=""` and without a marker filter |
| process | pid 1736374, `gb300/venv/bin/python3` (the pytest process itself; uid 1000 = dc) |
| allocation | 684 MiB on GPU 0 (`nvidia-smi` compute-apps inside the run; per-GPU memory.used 697 / 3 / 3 / 3 MiB), observed at 20:43:09Z by `test_census_and_plan_pass_from_inside_the_driver_context`, which therefore reported a compute app and failed |
| duration | within the 122.73 s run; after the run `nvidia-smi` on tray14 showed 0 MiB on every GPU and no compute process (re-checked 2026-11-18 01:36Z: 0 0 0 0 MiB, no compute process) |
| other GPU users on the tray | none during the run (the census listed only pid 1736374); tmux sessions `kvsim`, `kvsim-watchdog` of another session exist on the tray and were not touched |
| outcome | 5158 passed, 63 skipped, 3 failed, 4 errors; the three failures and four errors are the `cargo`-dependent tests, the gdn numerical test and the census test above |

## Cause

`tests/conftest.py` auto-detects CUDA (`torch.cuda.is_available()`) and un-skips the `gpu` tier when a device is visible,
and several unmarked kernel tests choose `cuda` themselves when it is available. The project's own CPU recipe
(`just test-cpu`) filters with `-m "not gpu and not agent and not bench and not presets"`; my scratch runner did neither
that nor hide the devices. The earlier, targeted suites (lifecycle, preflight, gates, pids, launcher-adjacent) never
import a kernel test and did not allocate.

## Remediation (this commit)

1. `tests/cpu_only_guard.py` — pytest plugin: sets `CUDA_VISIBLE_DEVICES=""` before torch is imported and, at session
   end, fails the run with exit status 3 if the pytest process has an open `/dev/nvidia<N>` device node or
   `torch.cuda.is_initialized()` is true; prints one summary line either way. `tests/test_cpu_only_guard.py` (4 tests)
   covers the detector, the verdict and the end-to-end exit status.
2. `tests/cpu_only.sh` — the isolated CPU-only command (`bash tests/cpu_only.sh [pytest args]`): hides the devices and
   loads the guard. `preflight_e1_v1.sh` E runs the suites only through it and requires the guard's "CUDA hidden" line;
   the scratch runner on the rack now execs it.
3. The whole tier was re-run CPU-only through the guard (`runs/v1/cpu/pytest_fullsuite_cpu_only_tray14_*.log`): GPU
   tests are skipped by the tier logic, the guard reports no device node opened, and `nvidia-smi` showed 0 MiB on every
   GPU throughout.
4. The preflight's launcher dry-run also runs with `CUDA_VISIBLE_DEVICES=""`; the identity preflight's interpreter probe
   already did. The V1 driver itself (`run_e1_v1.sh`) still unsets the variable — it is the only script that may use GPUs,
   and only under a GO.
