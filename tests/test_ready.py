"""The deploy's last check: the manager answers, and still answers a moment later (m14)."""

import json
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ai_lab.hosts import ready
from ai_lab.network import LOOPBACK


class Session(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/api/session" else 404)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *_args):
        pass


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind((LOOPBACK, 0))
        return probe.getsockname()[1]


class ReadyTests(unittest.TestCase):
    def serve(self) -> ThreadingHTTPServer:
        server = ThreadingHTTPServer((LOOPBACK, 0), Session)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        return server

    def test_a_manager_that_never_answers_fails(self):
        why = ready.wait(free_port(), within=0.3, settle=0, pause=0.05)
        self.assertIn("did not answer", why)

    def test_a_manager_that_answers_and_stays_passes(self):
        server = self.serve()
        self.addCleanup(server.shutdown)
        self.assertEqual(ready.wait(server.server_port, within=2, settle=0.1), "")

    def test_a_manager_that_dies_after_answering_fails(self):
        server = self.serve()
        port = server.server_port
        def stop():
            server.shutdown()
            server.server_close()

        stopper = threading.Timer(0.1, stop)
        stopper.start()
        self.addCleanup(stopper.cancel)
        self.assertIn("stopped answering", ready.wait(port, within=2, settle=0.6))

    def test_main_reads_the_port_from_the_configuration(self):
        server = self.serve()
        self.addCleanup(server.shutdown)
        config = Path(tempfile.mkdtemp()) / "config.json"
        config.write_text(json.dumps({"port": server.server_port}))
        self.assertEqual(ready.main(["--config", str(config), "--within", "2", "--settle", "0"]), 0)
        config.write_text(json.dumps({"port": free_port()}))
        self.assertEqual(ready.main(["--config", str(config), "--within", "0.2", "--settle", "0"]), 1)


if __name__ == "__main__":
    unittest.main()
