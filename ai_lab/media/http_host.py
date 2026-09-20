"""Small JSON HTTP host shared by isolated media backends."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    backend = None
    max_body_bytes = 65536
    request_path = "/v1/audio/music/generations"

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path != self.request_path:
            self._json(404, {"error": {"message": "not found"}})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= self.max_body_bytes:
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


def serve(backend, port: int, max_body_bytes: int = 65536,
          request_path: str = "/v1/audio/music/generations") -> None:
    Handler.backend = backend
    Handler.max_body_bytes = max_body_bytes
    Handler.request_path = request_path
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
