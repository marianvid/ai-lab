#!/usr/bin/env python3
"""Isolated HTTP host for the Khala music generator."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ai_lab.music.khala_backend import KhalaBackend


class Handler(BaseHTTPRequestHandler):
    backend = None

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path != "/v1/audio/music/generations":
            self._json(404, {"error": {"message": "not found"}})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                raise ValueError("music request is empty or too large")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("music request must be an object")
            self._json(200, self.backend.generate(body))
        except (ValueError, TypeError) as error:
            self._json(400, {"error": {"message": str(error)}})
        except Exception as error:
            self._json(500, {"error": {"message": str(error)}})

    def _json(self, status: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-script", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--default-bucket", type=int, required=True)
    parser.add_argument("--maximum-bucket", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    Handler.backend = KhalaBackend(
        args.generator_script, args.model_path, args.output_root,
        args.default_bucket, args.maximum_bucket)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
