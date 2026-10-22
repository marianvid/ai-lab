"""The connection to an engine, read on its own thread."""

import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ai_lab.api.upstream import END, FAILED, HEAD, PIECE, Upstream


class Engine(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    pieces = (b"one", b"two")
    pause_s = 0.0

    def log_message(self, *_):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        time.sleep(self.pause_s)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for piece in self.pieces:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(piece), piece))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


class UpstreamTests(unittest.TestCase):
    def serve(self, **behaviour):
        handler = type("Handler", (Engine,), behaviour)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}/v1/completions?x=1"

    def events(self, upstream, until=END):
        seen = []
        while True:
            kind, value = upstream.events.get(timeout=5)
            seen.append((kind, value))
            if kind in (until, FAILED):
                return seen

    def test_the_head_then_every_piece_then_the_end(self):
        upstream = Upstream(self.serve(), b"{}", "application/json", 5.0)
        seen = self.events(upstream)
        self.assertEqual(seen[0][0], HEAD)
        self.assertEqual(seen[0][1].status, 200)
        body = b"".join(value for kind, value in seen if kind == PIECE)
        self.assertEqual(body, b"onetwo")
        self.assertEqual(seen[-1], (END, None))
        upstream.stop()

    def test_stopping_before_the_answer_ends_the_reading_thread(self):
        upstream = Upstream(self.serve(pause_s=5.0), b"{}", "application/json", 10.0)
        upstream.stop()
        self.assertTrue(upstream.ended(5.0))
        self.assertTrue(upstream.events.empty(), "nothing is handed over once stopped")

    def test_an_engine_that_is_not_there_fails_at_once(self):
        with self.assertRaises(OSError):
            Upstream("http://127.0.0.1:9/v1/completions", b"{}", "application/json", 1.0)

    def test_a_reading_thread_nobody_reads_from_stops_when_told(self):
        many = tuple(b"x" * 10 for _ in range(400))
        upstream = Upstream(self.serve(pieces=many), b"{}", "application/json", 5.0)
        time.sleep(0.3)
        self.assertTrue(upstream.events.full(), "the queue should fill and pause the reader")
        upstream.stop()
        self.assertTrue(upstream.ended(5.0), "a paused reader must notice it was stopped")


if __name__ == "__main__":
    unittest.main()
