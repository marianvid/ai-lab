#!/usr/bin/env python3
"""HTTP host that manages one Higgs worker and exposes AI-Lab speech JSON."""
from __future__ import annotations

import argparse
import json
import signal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from ai_lab.speech.higgs_backend import HiggsBackend


class Handler(BaseHTTPRequestHandler):
    backend = None

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path != "/v1/audio/speech/generations":
            self._json(404, {"error": {"message": "not found"}})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 16384:
                raise ValueError("speech request is empty or too large")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("speech request must be an object")
            self._json(200, self.backend.generate(body))
        except (ValueError, TypeError) as error:
            self._json(400, {"error": {"message": str(error)}})
        except Exception as error:
            self._json(500, {"error": {"message": str(error)}})

    def _json(self, status, body):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-binary", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--worker-port", type=int, required=True)
    parser.add_argument("--mem-fraction-static", type=float, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    backend = HiggsBackend(args.worker_binary, args.model_path,
                           args.worker_port, args.mem_fraction_static)
    Handler.backend = backend
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    signal.signal(signal.SIGTERM, lambda *_: Thread(
        target=server.shutdown, daemon=True).start())
    try:
        server.serve_forever()
    finally:
        backend.close()
        server.server_close()


if __name__ == "__main__":
    main()
