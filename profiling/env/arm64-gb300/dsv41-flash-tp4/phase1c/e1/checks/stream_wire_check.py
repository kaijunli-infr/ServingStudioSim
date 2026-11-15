#!/usr/bin/env python3
"""CPU verification of session_runner's wire mode (Phase 1.5 prerequisite 2): run the binary against a local mock
OpenAI-compatible server twice (--stream true / --stream false) and check what the server received and how the client logged it.

    python checks/stream_wire_check.py --bin <session_runner> --tokenizer <dir or tokenizer.json> --out <dir>

PASS conditions: with --stream false every /v1/completions body has "stream": false and the client logs chunk_count 1;
with --stream true every body has "stream": true and chunk_count > 1; all requests SUCCESS in both runs. Exit 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RECEIVED: list[dict] = []
OUT_TOKENS = 12


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "mock", "object": "model"}], "running": 0, "waiting": 0}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        RECEIVED.append({"path": self.path, "stream": req.get("stream"), "max_tokens": req.get("max_tokens"), "model": req.get("model")})
        if self.path.rstrip("/") != "/v1/completions":
            self.send_response(404); self.end_headers(); return
        n_out = min(int(req.get("max_tokens") or OUT_TOKENS), OUT_TOKENS)
        usage = {"prompt_tokens": 64, "completion_tokens": n_out, "total_tokens": 64 + n_out}
        if req.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            for i in range(n_out):
                chunk = {"id": "cmpl-mock", "object": "text_completion", "choices": [{"index": 0, "text": " word", "finish_reason": None if i < n_out - 1 else "length"}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode()); self.wfile.flush()
            self.wfile.write(f"data: {json.dumps({'id': 'cmpl-mock', 'object': 'text_completion', 'choices': [], 'usage': usage})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        else:
            body = json.dumps({"id": "cmpl-mock", "object": "text_completion", "model": req.get("model"),
                               "choices": [{"index": 0, "text": " word" * n_out, "finish_reason": "length"}], "usage": usage}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


def run_once(a, port, stream: str, out):
    trace = os.path.join(out, "trace.csv")
    with open(trace, "w") as fh:
        fh.write("id,input_len,output_len,arrival_time\n")
        for i in range(4):
            fh.write(f"r{i},64,{OUT_TOKENS},0\n")
    log = os.path.join(out, f"replay_stream_{stream}.jsonl")
    argv = [a.bin, "--trace", trace, "--input-file-format", "text-generation-independent", "--text-file", a.text_file, "--tokenizer", a.tokenizer,
            "--base-url", f"http://127.0.0.1:{port}/v1", "--backend", "openai", "--model", "mock", "--log-path", log,
            "--summary-path", os.path.join(out, f"summary_stream_{stream}.json"), "--arrival-mode", "saturated", "--max-model-len", "4096",
            "--timeline", "false", "--stream", stream]
    before = len(RECEIVED)
    p = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    open(os.path.join(out, f"session_runner_stream_{stream}.log"), "w").write(p.stdout + p.stderr)
    recs = [json.loads(l) for l in open(log) if l.strip()] if os.path.exists(log) else []
    return p.returncode, RECEIVED[before:], recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True); ap.add_argument("--tokenizer", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--text-file")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.text_file:
        a.text_file = os.path.join(a.out, "corpus.txt")
        open(a.text_file, "w").write(("The quick brown fox jumps over the lazy dog while the five boxing wizards jump quickly. " * 400))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H); port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    rep = {"bin": a.bin, "sha256": hashlib.sha256(open(a.bin, "rb").read()).hexdigest(), "runs": {}}
    fails = []
    for stream, want_stream, want_chunks in (("false", False, lambda c: c == 1), ("true", True, lambda c: c > 1)):
        rc, got, recs = run_once(a, port, stream, a.out)
        comps = [g for g in got if g["path"].rstrip("/") == "/v1/completions"]
        wire = sorted({g["stream"] for g in comps}, key=str)
        chunks = sorted({r["outcome"].get("chunk_count") for r in recs})
        status = sorted({r["outcome"].get("status") for r in recs})
        ok = rc == 0 and len(comps) >= 4 and wire == [want_stream] and all(want_chunks(c) for c in chunks) and status == ["SUCCESS"] and len(recs) == 4
        rep["runs"][stream] = {"rc": rc, "completions_received": len(comps), "stream_field_values": wire, "chunk_counts": chunks, "statuses": status, "records": len(recs), "ok": ok}
        print(f"{'PASS' if ok else 'FAIL'}  --stream {stream}: rc={rc} completions={len(comps)} body.stream={wire} chunk_count={chunks} status={status} records={len(recs)}")
        if not ok:
            fails.append(stream)
    srv.shutdown()
    json.dump(rep, open(os.path.join(a.out, "stream_wire_check.json"), "w"), indent=1)
    print("RESULT", "PASS" if not fails else f"FAIL {fails}", f"binary sha256 {rep['sha256'][:16]}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
