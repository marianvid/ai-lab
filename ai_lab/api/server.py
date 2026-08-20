"""The HTTP server: routing, JSON, static files, and the event stream.

Threaded, because the event stream holds a connection open while other
requests still have to be answered.

There are no decisions about models or engines in this file. Its whole job is
turning a request into a call and a return value into a response — which is
the rule that keeps everything below it testable without HTTP.
"""

from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..events import EventBus
from ..operations import Operations
from . import sse
from .passthrough import Passthrough
from .router import Router
from .routes import register_all

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"

# Which exception means which status. Everything else is a bad request, since
# the alternative — a 500 with a stack trace — tells the user nothing.
STATUS = {
    KeyError: HTTPStatus.NOT_FOUND,
    FileNotFoundError: HTTPStatus.NOT_FOUND,
    NotImplementedError: HTTPStatus.NOT_IMPLEMENTED,
}


class Handler(BaseHTTPRequestHandler):
    router: Router
    bus: EventBus
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/events":
            return self._events()
        if not self._dispatch("GET", parsed):
            self._static(parsed.path)

    def do_POST(self):
        self._dispatch("POST", urlparse(self.path), required=True)

    def do_PATCH(self):
        self._dispatch("PATCH", urlparse(self.path), required=True)

    def do_DELETE(self):
        self._dispatch("DELETE", urlparse(self.path), required=True)

    # -- dispatch ----------------------------------------------------------

    def _dispatch(self, method: str, parsed, required: bool = False) -> bool:
        matched = self.router.match(method, parsed.path)
        if matched is None:
            if required:
                self._error(FileNotFoundError(parsed.path))
            return False
        handler, captured = matched
        try:
            query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
            result = handler(query=query, body=self._body(), **captured)
            if isinstance(result, Passthrough):
                self._passthrough(result)
            else:
                self._json(result if result is not None else {"ok": True})
        except Exception as error:
            self._error(error)
        return True

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            raise ValueError("Request body is not valid JSON") from None

    # -- responses ---------------------------------------------------------

    def _json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload).encode()
        self._head(status, "application/json", len(data))
        self.wfile.write(data)

    def _error(self, error: Exception):
        status = STATUS.get(type(error), HTTPStatus.BAD_REQUEST)
        message = str(error) or error.__class__.__name__
        self._json({"error": message}, status)

    def _head(self, status, content_type: str, length: int, cache: bool = False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        if not cache:
            # Nothing here is worth caching, and caching it is actively
            # harmful: with no headers at all a browser caches by its own
            # guesswork, so a deployment lands on the server while the page in
            # front of you stays as it was, and refreshing does not help.
            self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()

    def _events(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        sse.stream(self.bus, self._write_chunk, lambda: not self.wfile.closed)

    def _passthrough(self, response: Passthrough):
        """Write an engine's answer straight through, as it arrives.

        Chunked rather than buffered: a streamed answer has to reach the client
        while it is being produced, which is the whole reason a client asks for
        one.
        """
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        for name, value in response.headers.items():
            if name.lower() != "content-type":
                self.send_header(name, value)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for chunk in response.chunks:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The client hung up mid-answer. The generator's cleanup still runs
            # and still releases the lease, which is what matters.
            pass

    def _write_chunk(self, data: bytes):
        self.wfile.write(data)
        self.wfile.flush()

    # -- static files ------------------------------------------------------

    def _static(self, path: str):
        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in target.parents or not target.is_file():
            return self._error(FileNotFoundError(path))
        data = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._head(HTTPStatus.OK, content_type, len(data))
        self.wfile.write(data)

    def log_message(self, _format, *_args):
        pass


def build_router(operations: Operations, model_gateway=None) -> Router:
    router = Router()
    register_all(router, operations, model_gateway)
    return router


def serve(operations: Operations, bus: EventBus, host: str, port: int,
          model_gateway=None) -> None:
    handler = type("ConfiguredHandler", (Handler,),
                   {"router": build_router(operations, model_gateway), "bus": bus})
    ThreadingHTTPServer((host, port), handler).serve_forever()
