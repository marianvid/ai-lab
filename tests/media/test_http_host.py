"""The shared engine hosts, answering over a real socket."""

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from ai_lab.media.http_host import (
    Form,
    GenerationHost,
    UploadHost,
    form_text,
    uploaded_file,
)


class Echo:
    """Answers with what it was asked; refuses an empty prompt."""

    def generate(self, body):
        if not body.get("prompt"):
            raise ValueError("prompt is required")
        if body["prompt"] == "boom":
            raise RuntimeError("the engine fell over")
        return {"echo": body["prompt"]}


class Recorder:
    owner = "test"
    path = "/v1/images/ocr"


class EchoUpload(UploadHost):
    def answer(self, fields: Form) -> dict:
        name, data = uploaded_file(fields, "image")
        return {"name": name, "size": len(data), "language": form_text(fields, "language", "-")}


def started(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def post(server, path, data, content_type):
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}{path}", data=data, method="POST",
        headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def get(server, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}{path}",
                                    timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def form(fields, file_bytes):
    boundary = "xyz"
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
             .encode() for name, value in fields.items()]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                 f'filename="page.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
                 + file_bytes + b"\r\n")
    return b"".join(parts) + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


class GenerationHostTests(unittest.TestCase):
    def setUp(self):
        handler = type("Bound", (GenerationHost,), {"backend": Echo(), "noun": "speech",
                                                    "request_path": "/v1/speak"})
        self.server = started(handler)
        self.addCleanup(self.server.shutdown)

    def test_health_and_unknown_paths(self):
        self.assertEqual(get(self.server, "/health"), (200, {"status": "ok"}))
        self.assertEqual(get(self.server, "/elsewhere")[0], 404)

    def test_a_request_is_answered_by_the_backend(self):
        self.assertEqual(post(self.server, "/v1/speak", b'{"prompt": "hi"}', "application/json"),
                         (200, {"echo": "hi"}))

    def test_a_refusal_is_400_and_a_failure_500(self):
        status, body = post(self.server, "/v1/speak", b'{"prompt": ""}', "application/json")
        self.assertEqual((status, body["error"]["message"]), (400, "prompt is required"))
        self.assertEqual(post(self.server, "/v1/speak", b'{"prompt": "boom"}',
                              "application/json")[0], 500)

    def test_the_body_must_be_one_object_named_in_the_errors(self):
        status, body = post(self.server, "/v1/speak", b"[1]", "application/json")
        self.assertEqual((status, body["error"]["message"]),
                         (400, "speech request must be an object"))


class UploadHostTests(unittest.TestCase):
    def setUp(self):
        handler = type("Bound", (EchoUpload,), {"backend": Recorder(), "model_name": "ocr-1"})
        self.server = started(handler)
        self.addCleanup(self.server.shutdown)

    def test_the_listing_names_the_one_model(self):
        status, body = get(self.server, "/v1/models")
        self.assertEqual((status, body["data"][0]["id"], body["data"][0]["owned_by"]),
                         (200, "ocr-1", "test"))

    def test_an_upload_is_read_into_its_fields(self):
        data, kind = form({"language": "ro"}, b"\x89PNG....")
        self.assertEqual(post(self.server, "/v1/images/ocr", data, kind),
                         (200, {"name": "page.png", "size": 8, "language": "ro"}))

    def test_a_request_that_is_not_a_form_is_refused(self):
        status, body = post(self.server, "/v1/images/ocr", b"{}", "application/json")
        self.assertEqual((status, body["error"]["type"]), (400, "ValueError"))
