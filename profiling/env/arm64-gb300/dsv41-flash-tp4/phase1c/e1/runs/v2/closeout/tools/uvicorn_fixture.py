#!/usr/bin/env python3
"""CPU-only fixture: does a uvicorn server shut down on SIGTERM while one request is still awaiting something that never returns?

Mirrors the vLLM launcher's structure (vllm/entrypoints/launchers/launcher.py at 0ab18711): a server whose own signal handlers are
disabled, a loop-level SIGTERM handler that sets an event, a task that then cancels the serve task, and `await server.shutdown()`.
It imports uvicorn only (no vLLM, no torch, no CUDA). Cases:
  pending        one request awaiting forever, default config (timeout_graceful_shutdown=None)   <- the V2 situation
  idle           no request in flight
  pending_bound  one request awaiting forever, timeout_graceful_shutdown=3
  pending_client_closes  one request awaiting forever, default config; the client closes its socket 2 s after the SIGTERM
Usage: uvicorn_fixture.py serve <case> <port>     (child)      uvicorn_fixture.py run     (driver: runs the three cases)
"""
import asyncio, http.client, json, os, signal, socket, subprocess, sys, threading, time


async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        while True:
            m = await receive()
            if m["type"] == "lifespan.startup": await send({"type": "lifespan.startup.complete"})
            elif m["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"}); return
    if scope["path"] == "/hang":
        await asyncio.get_running_loop().create_future()  # like awaiting an engine reply that will never come
    await send({"type": "http.response.start", "status": 200, "headers": []}); await send({"type": "http.response.body", "body": b"ok"})


async def serve(case, port):
    import uvicorn

    class NoSignalServer(uvicorn.Server):
        def install_signal_handlers(self): pass
        def capture_signals(self):
            import contextlib
            return contextlib.nullcontext()

    kw = {"timeout_graceful_shutdown": 3} if case == "pending_bound" else {}
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info", **kw); config.load()
    server = NoSignalServer(config); loop = asyncio.get_running_loop(); ev = asyncio.Event()
    server_task = loop.create_task(server.serve())
    loop.add_signal_handler(signal.SIGTERM, ev.set)

    async def handle():
        await ev.wait(); print(f"[fixture] SIGTERM handled at {time.time():.3f}", flush=True)
        server.should_exit = True; server_task.cancel()

    loop.create_task(handle())
    try:
        await server_task
    except asyncio.CancelledError:
        print("[fixture] shutting down HTTP server", flush=True); await server.shutdown()
    print(f"[fixture] serve() returned at {time.time():.3f}", flush=True)


def run():
    import uvicorn
    results = {"uvicorn": uvicorn.__version__, "python": sys.version.split()[0], "cases": {}}
    for case in ("pending", "idle", "pending_bound", "pending_client_closes"):
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "serve", case, str(port)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(100):
            try: socket.create_connection(("127.0.0.1", port), timeout=0.2).close(); break
            except OSError: time.sleep(0.1)
        client = {}
        def req():
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=60); c.request("POST", "/hang"); client["conn"] = c; client["status"] = c.getresponse().status
            except Exception as e: client["error"] = f"{type(e).__name__}: {e}"
            client["returned_after_sigterm_s"] = round(time.time() - t_sig, 2) if "t_sig" in globals() else None
        if case != "idle":
            threading.Thread(target=req, daemon=True).start(); time.sleep(1.0)
        global t_sig
        t_sig = time.time(); p.send_signal(signal.SIGTERM)
        if case == "pending_client_closes":
            time.sleep(2.0); client["conn"].sock.shutdown(socket.SHUT_RDWR); client["conn"].sock.close(); client["closed_by_client_s_after_sigterm"] = round(time.time() - t_sig, 2)
        try:
            p.wait(timeout=12); exited, killed = round(time.time() - t_sig, 2), False
        except subprocess.TimeoutExpired:
            exited, killed = None, True; p.kill(); p.wait()  # this driver's own child
        time.sleep(0.3); out = p.stdout.read()
        results["cases"][case] = {"exited_s_after_sigterm": exited, "needed_sigkill_after_12_s": killed, "returncode": p.returncode, "client": {k: v for k, v in client.items() if k != "conn"},
                                  "log": [l for l in out.splitlines() if "Shutting down" in l or "Waiting" in l or "fixture" in l or "Cancel" in l or "shutdown complete" in l or "Finished" in l]}
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "serve": asyncio.run(serve(sys.argv[2], int(sys.argv[3])))
    else: run()
