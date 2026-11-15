# Stage 1 `EngineDeadError` at hook-on teardown — log inspection (Phase 1.5 prerequisite 4, CPU)

Source: `logs/20261114_dsv41_flash_gb300_tp4_e1/e1_dry_on/profile/vllm/dsv41_flash_1017_gb300_tp4_e1_dry_on_server.log`
(lines 7700-7751) and the hook-off log (lines 7774-7791), `e1_dry_on/run.log`. Nothing was re-run.

## Timeline (hook on, 2026-11-14 UTC)

| time | event |
|---|---|
| 19:19:58 | last engine iterations (2339-2342, 1-2 generation requests), last `POST /v1/completions 200 OK`, last `VibeSimAlignmentRequestTiming` |
| 19:19:59 | `Running: 0 reqs, Waiting: 0 reqs`; launcher `GET /load` (idle poll) |
| 19:19:5x | launcher `workload done … reached_idle: True`, routed-experts directory complete (run.log line 171) |
| 19:20:01 | `[shutdown] EngineCore: trigger received signal=SIGINT`; API server `stopping engine client mode=abort timeout=0s`; EngineCore `request processing complete; starting resource teardown`, `exiting busy loop`; process manager `send sigterm to process EngineCore` then, with timeout 0 s, `force killing remaining process EngineCore pid 4088490` |
| 19:20:01 | `AsyncLLM output_handler failed` — `output_handler` was awaiting `engine_core.get_output_async()` (`async_llm.py:827`); `core_client.py:1215` raises `EngineDeadError` because the EngineCore it was reading from had just been force-killed |
| 19:20:01 | API server continues its normal shutdown (`engine client stopped`, HTTP server shutdown, `Application shutdown complete`); `resource_tracker: 1 leaked shared_memory object` |
| 19:20:10 | launcher rc=0 |

Hook off (18:53:01) shows the identical shutdown sequence (`mode=abort timeout=0s`, SIGTERM, immediate force kill) without
the `output_handler` error, and the same leaked shared-memory warning.

## Most specific supported explanation

The error is a **teardown race inside the launcher's abort-mode shutdown**, not an engine failure during the workload:

1. The launcher stops the server in abort mode with a 0 s grace period, so the process manager force-kills EngineCore the
   moment SIGTERM is sent (both arms, both logs).
2. The API server's `output_handler` coroutine is a permanent loop waiting on the EngineCore output socket. Whether it
   observes the socket dying before it is cancelled by the API server's own shutdown is a matter of microseconds; in the
   hook-on run it did (and logged `EngineDeadError`), in the hook-off run it did not.
3. Everything that produces data had finished before the shutdown trigger: last request completed 19:19:58, engine idle at
   19:19:59, the load generator reported `reached_idle: True` and the routed-experts directory complete, and the
   completeness checks (v1 and v2) found 56/56 requests, 2160 iterations and a complete, unique token-to-iteration join.

Unsupported by the logs: an EngineCore crash or hang (the EngineCore logged its own orderly `exiting busy loop` before the
kill), a hook-specific cause (the sequence and the kill are identical in the hook-off arm), data loss (checked).

Residual uncertainty: the race itself is only inferred from ordering; the fork's shutdown code was not modified or tested here.

## Abort conditions for Stage 1.5 (kept explicit because the explanation is inferred)

A run is **aborted / not accepted** if any of these holds:

* `EngineDeadError`, `EngineCore encountered an issue` or `output_handler failed` appears in the server log **before** the
  `[shutdown] … trigger received` line (i.e. during startup or the workload), or the launcher exits rc != 0;
* the load generator does not report `reached_idle: True`, or any request has `status != SUCCESS` in `replay.jsonl`;
* `checks/e1_completeness_v2.py` reports any FAIL (metrics count, 56 requests x 1024 tokens, join, routes for the hook arm);
* the server log contains an EngineCore traceback other than the post-kill `EngineDeadError` in `output_handler`.

If the only occurrence is the post-shutdown `output_handler` line (as in Stage 1), the run is accepted and the line is
recorded in the report. `run_e1_stage15.sh` prints the count and position of these lines after each pass.
