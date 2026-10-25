"""The two waits of `forward`, against a real HTTP engine stand-in.

Each fake engine sends its headers at once, like llama.cpp and mlx-lm do for a
streamed answer, and then behaves in one particular way.
"""

import select
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ai_lab.clients.errors import AiLabOffError, ModelReplacedError
from ai_lab.api.passthrough import (
    ClientGoneError,
    EngineStalledError,
    Hooks,
    Waits,
    forward,
)

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

    def forward(self, behaviour, on_first_chunk=None):
        return forward(self.serve(behaviour), {"stream": True},
                       Waits(FIRST_BYTE_S, BETWEEN_BYTES_S),
                       Hooks(on_first_chunk=on_first_chunk))

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

    def test_a_cut_ends_the_answer_with_its_own_refusal(self):
        # A7: AI-Lab Off, or the model replaced, ends the call with that reason.
        reasons: list[Exception] = []
        answer = forward(self.serve("stalls-mid-answer"), {"stream": True},
                         Waits(FIRST_BYTE_S, BETWEEN_BYTES_S * 10, cut=lambda: reasons[0]
                               if reasons else None), Hooks())
        chunks = iter(answer.chunks)
        self.assertEqual(next(chunks), b"data: one\n\n")
        reasons.append(ModelReplacedError("qwen was taken off the card for coder"))
        with self.assertRaises(ModelReplacedError):
            next(chunks)

    def test_a_cut_before_the_first_word_ends_the_answer_with_its_refusal(self):
        answer = forward(self.serve("never-answers"), {"stream": True},
                         Waits(FIRST_BYTE_S, BETWEEN_BYTES_S,
                               cut=lambda: AiLabOffError("switched Off")), Hooks())
        started = time.perf_counter()
        with self.assertRaises(AiLabOffError):
            b"".join(answer.chunks)
        self.assertLess(time.perf_counter() - started, FIRST_BYTE_S)

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


class SlowEngine(BaseHTTPRequestHandler):
    """An engine that answers without streaming: nothing until it has finished.

    While it "writes", it watches its connection, the way vLLM does, and
    notes when its client has gone (`left`).
    """

    protocol_version = "HTTP/1.1"
    writing_s = 1.0
    status = 200
    left: threading.Event

    def log_message(self, *_):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        until = time.perf_counter() + self.writing_s
        while time.perf_counter() < until:
            readable, _, _ = select.select([self.connection], [], [], 0.02)
            if readable and not self.connection.recv(1, socket.MSG_PEEK):
                self.left.set()
                self.close_connection = True
                return
        body = b'{"answer": "done"}'
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class WatchTests(unittest.TestCase):
    """What the gateway does while it waits: the client, and the engine's progress."""

    def serve(self, writing_s=1.0, status=200):
        left = threading.Event()
        handler = type("Handler", (SlowEngine,),
                       {"writing_s": writing_s, "status": status, "left": left})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions", left

    def forward(self, url, waits, closed=None):
        return forward(url, {"stream": False}, waits, Hooks(on_close=closed))

    def test_a_client_that_leaves_stops_its_request_in_the_engine(self):
        url, left = self.serve(writing_s=5.0)
        here_until = time.perf_counter() + 0.3
        closed = []
        waits = Waits(0.3, 0.2, client_here=lambda: time.perf_counter() < here_until,
                      idle_for=lambda: 0.0, tick_s=0.05)
        with self.assertRaises(ClientGoneError):
            self.forward(url, waits, closed=lambda: closed.append(True))
        self.assertTrue(left.wait(2.0), "the engine never saw its client leave")
        self.assertEqual(closed, [True], "the place must be given back once")

    def test_an_engine_that_does_no_work_is_cut_with_its_reason(self):
        url, left = self.serve(writing_s=5.0)
        waits = Waits(0.3, 0.2, idle_for=lambda: 99.0, stall_s=60.0, tick_s=0.05)
        told: list[str] = []
        with self.assertRaises(EngineStalledError) as caught:
            forward(url, {"stream": False}, waits, Hooks(on_stall=told.append))
        self.assertIn("engine stalled", caught.exception.explanation)
        self.assertTrue(left.wait(2.0))
        self.assertEqual(len(told), 1, "the gateway must be told, to restart the engine")
        self.assertIn("engine stalled", told[0])

    def test_waiting_while_the_engine_works_is_never_cut(self):
        # Longer than both plain waits (0.3 s and 0.2 s): with progress reported
        # they do not apply.
        url, _ = self.serve(writing_s=1.0)
        waits = Waits(0.3, 0.2, idle_for=lambda: 0.0, tick_s=0.05)
        answer = self.forward(url, waits)
        self.assertEqual(b"".join(answer.chunks), b'{"answer": "done"}')

    def test_without_progress_reports_the_plain_wait_still_applies(self):
        url, left = self.serve(writing_s=5.0)
        with self.assertRaises(TimeoutError):
            self.forward(url, Waits(0.3, 0.2, tick_s=0.05))
        self.assertTrue(left.wait(2.0))

    def test_the_safety_limit_ends_even_a_working_engine(self):
        url, _ = self.serve(writing_s=5.0)
        waits = Waits(0.3, 0.2, idle_for=lambda: 0.0, safety_s=0.4, tick_s=0.05)
        started = time.perf_counter()
        with self.assertRaises(TimeoutError):
            self.forward(url, waits)
        self.assertLess(time.perf_counter() - started, 2.0)

    def test_a_refusal_comes_back_with_the_engines_own_words(self):
        url, _ = self.serve(writing_s=0.0, status=400)
        closed = []
        answer = self.forward(url, Waits(0.3, 0.2, tick_s=0.05),
                              closed=lambda: closed.append(True))
        self.assertEqual(answer.status, 400)
        self.assertEqual(b"".join(answer.chunks), b'{"answer": "done"}')
        answer.close()
        self.assertEqual(closed, [True], "closing twice must tell the caller once")

    def test_closing_an_answer_never_read_gives_its_place_back(self):
        # The fault of 22 October 2026: the client was gone before the first
        # byte, the body was never started, and the place stayed taken.
        url, _ = self.serve(writing_s=0.0)
        closed = []
        answer = self.forward(url, Waits(0.3, 0.2, tick_s=0.05),
                              closed=lambda: closed.append(True))
        answer.close()
        self.assertEqual(closed, [True])

    def test_an_engine_that_is_not_listening_fails_at_once_and_gives_back(self):
        closed = []
        with self.assertRaises(OSError):
            forward("http://127.0.0.1:9/v1/chat/completions", {}, Waits(0.3, 0.2),
                    Hooks(on_close=lambda: closed.append(True)))
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
