"""The two waits of `forward`, against a real HTTP engine stand-in.

Each fake engine sends its headers at once, like llama.cpp and mlx-lm do for a
streamed answer, and then behaves in one particular way.
"""

import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ai_lab.api.passthrough import forward

# Scaled-down limits: the idle limit is shorter than the prompt-reading pause,
# the first-byte limit is longer.
FIRST_BYTE_S = 3.0
BETWEEN_BYTES_S = 0.4
PROMPT_PAUSE_S = 1.0


class Engine(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    behaviour = "slow-prompt"

    def log_message(self, *_):
        pass

    def send_chunk(self, data: bytes) -> None:
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.wfile.flush()
        if self.behaviour == "slow-prompt":
            # Silent while "reading the prompt": longer than the idle limit,
            # shorter than the first-byte limit.
            time.sleep(PROMPT_PAUSE_S)
            for word in (b"data: one\n\n", b"data: two\n\n"):
                self.send_chunk(word)
        elif self.behaviour == "keep-alives-while-reading":
            # What llama.cpp does: an SSE comment every so often while the
            # prompt is read. Each gap is longer than the idle limit.
            for _ in range(2):
                time.sleep(BETWEEN_BYTES_S * 1.5)
                self.send_chunk(b": \n\n")
            time.sleep(BETWEEN_BYTES_S * 1.5)
            self.send_chunk(b"data: one\n\n")
        elif self.behaviour == "stalls-mid-answer":
            self.send_chunk(b"data: one\n\n")
            time.sleep(BETWEEN_BYTES_S * 4)
            self.send_chunk(b"data: two\n\n")
        elif self.behaviour == "never-answers":
            time.sleep(FIRST_BYTE_S + 1)
            self.send_chunk(b"data: late\n\n")
        elif self.behaviour == "one-word-then-think":
            self.send_chunk(b"data: one\n\n")
            time.sleep(BETWEEN_BYTES_S * 0.75)
            self.send_chunk(b"data: two\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


class ForwardTests(unittest.TestCase):
    def serve(self, behaviour):
        handler = type("Handler", (Engine,), {"behaviour": behaviour})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"

    def forward(self, behaviour, **kwargs):
        return forward(self.serve(behaviour), {"stream": True},
                       first_byte_s=FIRST_BYTE_S,
                       between_bytes_s=BETWEEN_BYTES_S, **kwargs)

    def test_a_long_silent_prompt_is_covered_by_the_first_byte_limit(self):
        # The fault seen live: a 34k-token prompt on llama.cpp came back empty
        # because the idle limit started at the headers.
        times = []
        answer = self.forward("slow-prompt",
                              on_first_chunk=lambda seconds: times.append(seconds))
        body = b"".join(answer.chunks)
        self.assertEqual(body, b"data: one\n\ndata: two\n\n")
        self.assertGreaterEqual(times[0], PROMPT_PAUSE_S * 0.9,
                                "the time to the first byte counted the headers")

    def test_keep_alives_are_not_the_first_byte(self):
        # llama.cpp's "still here" line during a long prompt used to start the
        # idle limit; the next gap then ended the stream before any answer.
        times = []
        answer = self.forward("keep-alives-while-reading",
                              on_first_chunk=lambda seconds: times.append(seconds))
        body = b"".join(answer.chunks)
        self.assertTrue(body.endswith(b"data: one\n\n"))
        self.assertEqual(body.count(b": \n\n"), 2, "keep-alives should still reach the client")
        self.assertGreaterEqual(times[0], BETWEEN_BYTES_S * 4,
                                "a keep-alive was timed as the first word")

    def test_silence_in_the_middle_of_an_answer_still_ends_it(self):
        answer = self.forward("stalls-mid-answer")
        with self.assertRaises(OSError):
            b"".join(answer.chunks)

    def test_an_engine_that_never_starts_is_still_cut_off(self):
        started = time.perf_counter()
        answer = self.forward("never-answers")
        with self.assertRaises(OSError):
            b"".join(answer.chunks)
        self.assertLess(time.perf_counter() - started, FIRST_BYTE_S + 0.9)

    def test_each_piece_is_passed_on_as_it_arrives(self):
        # Before, the stream was collected into 8 KB pieces, so a small first
        # word waited for dozens more before the client saw it.
        answer = self.forward("one-word-then-think")
        chunks = iter(answer.chunks)
        started = time.perf_counter()
        first = next(chunks)
        waited = time.perf_counter() - started
        self.assertEqual(first, b"data: one\n\n")
        self.assertLess(waited, BETWEEN_BYTES_S * 0.5)
        self.assertEqual(b"".join(chunks), b"data: two\n\n")


if __name__ == "__main__":
    unittest.main()
