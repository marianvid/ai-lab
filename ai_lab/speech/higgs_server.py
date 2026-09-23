#!/usr/bin/env python3
"""HTTP host that manages one Higgs worker and exposes AI-Lab speech JSON."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from ai_lab.speech.contract import MAX_REQUEST_BYTES
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
            if not 0 < size <= MAX_REQUEST_BYTES:
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
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ui-port", type=int, required=True)
    args = parser.parse_args()
    backend = HiggsBackend(args.worker_binary, args.model_path,
                           args.worker_port, args.mem_fraction_static,
                           args.max_parallel)
    Handler.backend = backend
    playground_root = Path(__file__).resolve().parents[1] / "native_ui" / "sglang_omni"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(playground_root), env.get("PYTHONPATH", ""))))
    try:
        playground = subprocess.Popen([
            sys.executable, "-m", "playground.higgs.app", "--api-base",
            f"http://127.0.0.1:{args.worker_port}", "--port", str(args.ui_port)],
            env=env)
    except Exception:
        backend.close()
        raise
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if playground.poll() is not None:
                raise RuntimeError("Higgs playground exited during startup")
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.ui_port}/healthz", timeout=2) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(0.5)
        else:
            raise TimeoutError("Higgs playground did not become ready")
        server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    except Exception:
        if playground.poll() is None:
            playground.terminate()
            try:
                playground.wait(timeout=10)
            except subprocess.TimeoutExpired:
                playground.kill()
                playground.wait(timeout=5)
        backend.close()
        raise
    signal.signal(signal.SIGTERM, lambda *_: Thread(
        target=server.shutdown, daemon=True).start())
    try:
        server.serve_forever()
    finally:
        if playground.poll() is None:
            playground.terminate()
            try:
                playground.wait(timeout=10)
            except subprocess.TimeoutExpired:
                playground.kill()
                playground.wait(timeout=5)
        backend.close()
        server.server_close()


if __name__ == "__main__":
    main()
