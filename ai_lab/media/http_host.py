"""The small HTTP hosts every isolated engine server is built on.

Each engine (music, speech, transcription, OCR) runs in a Python environment of
its own and answers one HTTP contract on one port. Two shapes of contract:

- **JSON in, JSON out** (`GenerationHost`): music and speech. The request body
  is one JSON object handed to `backend.generate`.
- **An upload in, JSON out** (`UploadHost`): transcription, diarization, voice
  activity, alignment and OCR. The request is a `multipart/form-data` form
  carrying a file and a few fields; the subclass's `answer` turns it into JSON.

The handler class is built for one backend when the server starts
(`serve`, `serve_uploads`): a copy of the class with the backend set on it.
The backend is therefore never missing while a request is answered, and the
type says so — there is no `backend = None` patched in from outside.
"""

from __future__ import annotations

import json
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol, cast

from ..network import ALL_INTERFACES

MUSIC_PATH = "/v1/audio/music/generations"
# The largest JSON request a generation host reads, unless told otherwise.
MAX_JSON_BYTES = 65536

# A form as read from a multipart upload: field name -> (file name, bytes).
Form = dict[str, tuple[str | None, bytes]]


class Generator(Protocol):
    """A backend answering one JSON request with one JSON answer."""

    def generate(self, body: dict) -> dict:
        """The answer to one request."""
        ...


class UploadBackend(Protocol):
    """A backend answering uploads: who made it, and the path it answers."""

    owner: str
    path: str


class JsonReplies(BaseHTTPRequestHandler):
    """Answers in JSON: the health check, the listing, errors and results."""

    def do_GET(self) -> None:
        """`/health`, or what `_listing` answers for this path, or 404."""
        if self.path == "/health":
            self._json(200, {"status": "ok"})
            return
        listing = self._listing()
        if listing is None:
            self._not_found()
        else:
            self._json(200, listing)

    def _listing(self) -> dict | None:
        """The answer to a GET other than the health check; None for 404."""
        return None

    def _not_found(self) -> None:
        self._json(404, {"error": {"message": "not found"}})

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class GenerationHost(JsonReplies):
    """One JSON request to `request_path`, answered by `backend.generate`.

    A request the backend refuses (`ValueError`, `TypeError`) is a 400 with
    its words; anything else that fails is a 500.
    """

    backend: Generator
    max_body_bytes: int = MAX_JSON_BYTES
    request_path: str = MUSIC_PATH
    # What a request is called in its error messages ("music", "speech").
    noun: str = "music"

    def do_POST(self) -> None:
        """Read the JSON request, generate, answer."""
        if self.path != self.request_path:
            self._not_found()
            return
        try:
            self._json(200, self.backend.generate(self._body()))
        except (ValueError, TypeError) as error:
            self._json(400, {"error": {"message": str(error)}})
        except Exception as error:
            self._json(500, {"error": {"message": str(error)}})

    def _body(self) -> dict:
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= self.max_body_bytes:
            raise ValueError(f"{self.noun} request is empty or too large")
        body = json.loads(self.rfile.read(size))
        if not isinstance(body, dict):
            raise ValueError(f"{self.noun} request must be an object")
        return body


class UploadHost(JsonReplies):
    """One upload to `backend.path`, answered by the subclass's `answer`.

    `GET /v1/models` lists the one model this process serves, under the name
    the manager started it with. Every failure is a 400 naming its type.
    """

    backend: UploadBackend
    model_name: str = ""

    def do_POST(self) -> None:
        """Read the form, answer it."""
        if self.path != self.backend.path:
            self._not_found()
            return
        try:
            self._json(200, self.answer(read_form(self)))
        except Exception as error:
            self._json(400, {"error": {"message": str(error),
                                       "type": error.__class__.__name__}})

    def answer(self, fields: Form) -> dict:
        """The answer to one upload. Each kind of engine says how."""
        raise NotImplementedError

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the base class's name
        """One line per request on standard output, where the manager reads the log."""
        print(f"{self.address_string()} {format % args}", flush=True)

    def _listing(self) -> dict | None:
        if self.path != "/v1/models":
            return None
        return {"object": "list", "data": [{
            "id": self.model_name, "object": "model", "owned_by": self.backend.owner}]}


def read_form(handler: BaseHTTPRequestHandler) -> Form:
    """The fields of a `multipart/form-data` request, each as (file name, bytes)."""
    length = int(handler.headers.get("Content-Length", "0"))
    content_type = handler.headers.get("Content-Type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("expected a multipart/form-data request")
    envelope = (f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
                + handler.rfile.read(length))
    message = cast(EmailMessage, BytesParser(policy=policy.default).parsebytes(envelope))
    fields: Form = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not isinstance(name, str) or not name:
            continue
        payload = part.get_payload(decode=True) or b""
        if not isinstance(payload, bytes):
            raise ValueError(f"form field {name} could not be read")
        fields[name] = (part.get_filename(), payload)
    return fields


def uploaded_file(fields: Form, kind: str) -> tuple[str | None, bytes]:
    """The form's `file`: (its name, its bytes). `kind` names it in the error."""
    uploaded = fields.get("file")
    if uploaded is None:
        raise ValueError(f"the request must contain an {kind} file")
    return uploaded


def form_text(fields: Form, name: str, default: str | None) -> str | None:
    """A text field of the form, or `default` when it is not there."""
    part = fields.get(name)
    return part[1].decode("utf-8") if part is not None else default


def serve(backend: Generator, port: int, max_body_bytes: int = MAX_JSON_BYTES,
          request_path: str = MUSIC_PATH, noun: str = "music") -> None:
    """Answer JSON requests for `backend` on every interface until stopped."""
    handler = type("BoundGenerationHost", (GenerationHost,), {
        "backend": backend, "max_body_bytes": max_body_bytes,
        "request_path": request_path, "noun": noun})
    ThreadingHTTPServer((ALL_INTERFACES, port), handler).serve_forever()


def serve_uploads(host: type[UploadHost], backend: UploadBackend, model_name: str,
                  address: tuple[str, int]) -> None:
    """Answer uploads with `host`, bound to `backend` and its name, until stopped."""
    handler = type(f"Bound{host.__name__}", (host,), {
        "backend": backend, "model_name": model_name})
    print(f"{model_name} ready on {address[0]}:{address[1]}", flush=True)
    ThreadingHTTPServer(address, handler).serve_forever()
