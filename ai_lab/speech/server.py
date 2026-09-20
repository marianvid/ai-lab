#!/usr/bin/env python3
"""Isolated HTTP host for configurable speech backends."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ai_lab.speech.contract import PATH, validate_payload
from ai_lab.speech.kokoro_backend import KokoroBackend
from ai_lab.speech.qwen import MODES, QwenTtsBackend
from ai_lab.speech.voxcpm_backend import VoxCpmBackend


class Handler(BaseHTTPRequestHandler):
    backend = None

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path != PATH:
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

    def _json(self, status: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("qwen", "kokoro", "voxcpm"), default="qwen")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(MODES))
    parser.add_argument("--language-code")
    parser.add_argument("--default-voice")
    parser.add_argument("--repo-id")
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.backend == "qwen":
        if not args.mode:
            parser.error("Qwen speech requires --mode")
        Handler.backend = QwenTtsBackend(args.model_path, args.mode)
    elif args.backend == "kokoro":
        if not all((args.language_code, args.default_voice, args.repo_id)):
            parser.error("Kokoro requires language, voice and repo settings")
        Handler.backend = KokoroBackend(
            args.model_path, args.language_code, args.default_voice,
            args.repo_id)
    else:
        Handler.backend = VoxCpmBackend(
            args.model_path, args.cfg_value, args.inference_timesteps)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
