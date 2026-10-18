"""Asking an engine whether it is ready, over a real socket."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ai_lab.engines.probe import http_json, http_ok


class Engine(BaseHTTPRequestHandler):
    """Answers each path the way some engine does."""

    answers = {
        "/health": (200, b'{"status": "ok"}'),
        "/loading": (200, b'{"status": "loading"}'),
        "/plain": (200, b"OK"),
        "/down": (503, b"{}"),
        "/list": (200, b"[1, 2]"),
    }

    def do_GET(self):
        status, body = self.answers.get(self.path, (404, b""))
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class ProbeTests(unittest.TestCase):
    server: ThreadingHTTPServer
    port: int

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Engine)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.port = cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_ok_means_answering_and_not_still_loading(self):
        self.assertTrue(http_ok(self.port))
        self.assertTrue(http_ok(self.port, "/plain"))
        self.assertFalse(http_ok(self.port, "/loading"))
        self.assertFalse(http_ok(self.port, "/down"))

    def test_nobody_listening_is_not_yet_rather_than_an_error(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Engine)
        port = server.server_port
        server.server_close()
        self.assertFalse(http_ok(port))
        self.assertEqual(http_json(port, "/health"), {})

    def test_json_is_the_object_answered_or_nothing(self):
        self.assertEqual(http_json(self.port, "/health"), json.loads(b'{"status": "ok"}'))
        self.assertEqual(http_json(self.port, "/list"), {})
        self.assertEqual(http_json(self.port, "/down"), {})
