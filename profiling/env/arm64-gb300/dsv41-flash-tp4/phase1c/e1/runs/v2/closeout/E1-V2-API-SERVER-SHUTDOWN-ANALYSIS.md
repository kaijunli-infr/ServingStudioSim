# Why the API server outlived its SIGTERM in V2 (CPU-only analysis, 2026-11-18)

Sources: the V2 and V1 server logs and lifecycle records (run directory / archive), the probe record, the vLLM fork source at
0ab18711 and uvicorn 0.54.0 in the serving environment, and one CPU-only fixture (`uvicorn_fixture.py`, result
`uvicorn_fixture.json`). No GPU was used. **No shutdown behaviour was changed**; §5 lists options only.

## 1. Timeline, SIGTERM to SIGKILL (UTC, 2026-11-18)

| time | event | source |
|---|---|---|
| 15:55:00 | API server: "Stopping profiler..."; Worker_TP0: "Profiler stopped successfully." No "Profiler stopped." from the API server, no `POST /stop_profile … 200` line, ever | server log |
| 15:56:00 | EngineCore: "No available shared memory broadcast block found in 60 seconds" | server log |
| 15:56:01.90 | controller phase `importing` | lifecycle |
| 15:56:02.34 / 02.79 | probe: API server, EngineCore and four workers alive (state S) | `probe.json` |
| 15:56:03.074 | SIGTERM to API server pid 1556645, identity verified | lifecycle `actions` |
| 15:56:03 | API server: "shutdown triggered"; "stopping engine client mode=abort timeout=0s"; "send sigterm to process EngineCore"; "force killing remaining process EngineCore"; "engine client stopped"; "signalling HTTP server shutdown"; "shutting down FastAPI HTTP server"; uvicorn: "Shutting down"; "**Waiting for connections to close. (CTRL+C to force quit)**" | server log |
| 15:56:03.54 | probe: EngineCore and the four workers are zombies; API server alive | `probe.json` |
| 15:56:04.09 – 05.19 | sampler reads the GPUs free | `gpu_observations.jsonl` (addendum §2) |
| 15:56:30.82 | report file last written | file mtime |
| 15:56:03 → 15:57:03 | no further line from the API server; the import bar runs to 100 % | server log |
| 15:57:03.114 | SIGKILL to pid 1556645 after the 60 s grace, identity re-checked | lifecycle `actions` |
| 15:57:03.118 | the launcher's stop request returns "Remote end closed connection without response" | lifecycle `http` |
| 15:57:04.12 | Nsight session closed; "Generated: …nsys-rep" | lifecycle, server log |

V1 for comparison (normal stop): "Stopping profiler..." 03:21:12 → "Profiler stopped." and `POST /stop_profile 200` at 03:22:42
(import ended 03:22:43) → SIGTERM 03:22:45.43 → "Waiting for application shutdown." → "Application shutdown complete." → GPUs
released 03:22:46.7. In V1 no request was outstanding at the SIGTERM and the server exited within about a second.

## 2. Findings backed by evidence

1. **The SIGTERM was delivered and handled at once.** The server's own handler logged "shutdown triggered" in the same second and
   ran its shutdown sequence. Signal handling is not the cause.
2. **The engine was stopped immediately and the GPUs were released by that, not by the API server's exit.** Engine shutdown ran
   with `timeout=0` (abort), the EngineCore was force-killed, engine and workers were dead 0.47 s after the SIGTERM, and the sampler
   saw the GPUs free at its next reading.
3. **The server then waited in uvicorn's connection wait, for the whole 60 s.** Its last line is uvicorn's "Waiting for connections
   to close"; the following lines of a normal shutdown ("Waiting for application shutdown.", "Application shutdown complete.") never
   appear. In uvicorn 0.54.0 (`server.py`, `_wait_tasks_to_complete`) that loop runs while any connection is open and is bounded
   only by `timeout_graceful_shutdown`, which the vLLM launcher does not set (`uvicorn.Config(app, **uvicorn_kwargs)`; no occurrence
   of `timeout_graceful_shutdown` under `vllm/entrypoints`), so the wait is unbounded.
4. **Exactly one request was outstanding: the launcher's `/stop_profile`.** It was sent at 15:55:00 and had not returned (no
   "Profiler stopped." line); its client got a connection error at the instant of the SIGKILL. The last completion response and a
   `GET /load` precede the stop request in the log, and no other request line follows it.
5. **A stop request stays open until Nsight has finished processing.** In V1 it returned at 03:22:42, 90 s after it was sent and
   within a second of the end of the import. Any server shutdown requested during the flush or the import therefore meets an
   outstanding request.
6. **Reproduced without vLLM or a GPU.** `uvicorn_fixture.py` builds the same structure as the launcher (server without its own
   signal handlers, loop-level SIGTERM handler, cancel the serve task, `await server.shutdown()`):

   | case | exit after SIGTERM | log tail | client |
   |---|---|---|---|
   | one request awaiting forever, default config | none within 12 s; needed SIGKILL | "Waiting for connections to close." | "Remote end closed connection without response" at the kill |
   | no request in flight | 0.16 s | "Application shutdown complete." | — |
   | one request awaiting forever, `timeout_graceful_shutdown=3` | 3.17 s | "Cancel 1 running task(s), timeout graceful shutdown exceeded" | HTTP 500 after 3.1 s |
   | one request awaiting forever, client closes its socket 2 s after SIGTERM | none within 12 s; needed SIGKILL | "Waiting for background tasks to complete." | — |

   The first case shows the V2 log signature and the V2 client error exactly.
7. **Process reaping did not hold the server up.** The dead engine and workers were unreaped zombies for at least 5 s, but the
   server was by then in the HTTP wait, which does not depend on them; zombies hold no GPU memory.

## 3. Findings from reading the code (not exercised by a test)

- The `/stop_profile` handler awaits `engine_client.stop_profile()`, which awaits a future that only an engine reply resolves
  (`core_client._call_utility_async`). The client's shutdown closes sockets and cancels its output task; the code read does not fail
  or cancel pending utility futures. With the engine killed, nothing resolves the future, so the handler never returns. This fits
  findings 3–5 but the cancellation path was read, not run.

## 4. Hypotheses (consistent with the evidence, not shown)

- **Why the stop request is open during the flush and the import:** the profiler-stop call blocks in one or more workers until
  Nsight has finished (only Worker_TP0 logs completion at once; the EngineCore reports a blocked broadcast after 60 s; the request
  returns at the end of the import in V1). Which call blocks, and in which workers, is not logged.
- **Why the Nsight session closed one second after the SIGKILL:** the CLI waits for the target process to exit before it ends the
  session; the report had been complete since 15:56:30.
- **Whether the flush duration (62–64 s in V1 and V2) contains a fixed 60 s wait** related to the "60 seconds" broadcast message:
  suggested by two runs of one size only; not tested.

## 5. Options for a later decision (none applied)

| option | effect | cost / risk |
|---|---|---|
| keep as is | server gone 60 s after a forced release; GPUs free within about 2 s regardless | the session stays open 60 s longer than necessary |
| shorter `server_term_grace_s` in the controller policy | earlier SIGKILL of the same verified pid | controller behaviour change; needs its own review and tests |
| `timeout_graceful_shutdown` in the server's uvicorn config | server exits by itself after the bound | change in the vLLM fork (pinned submodule) |
| cancel the pending stop request server-side on shutdown | clean exit | change in the vLLM fork |

Closing the client's connection is not an option: the fixture's fourth case shows the server still waits for the handler task.
