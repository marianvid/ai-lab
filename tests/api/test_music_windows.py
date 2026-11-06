"""The front door's side of window calls: their request ids, their cancel, the songs' files."""

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ai_lab.api.passthrough import Passthrough
from ai_lab.api.router import Router
from ai_lab.api.routes import music_windows
from ai_lab.api.routes.music_windows import (
    GenerationFiles,
    WindowAsk,
    WindowCalls,
    ended_with,
    register_files,
    window_call,
    window_request_id,
)
from ai_lab.gateway.errors import (
    ModelNotLoadedError,
    NotConfiguredError,
    WindowCallActiveError,
    WindowRequestError,
)


class EngineAnswering(BaseHTTPRequestHandler):
    """An engine that answers a cancel, or says the call is unknown."""

    def do_DELETE(self):
        status = 404 if self.path.endswith("/gone") else 200
        body = json.dumps({"state": "cancelling"}).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        """Quiet."""


class WindowCallsTests(unittest.TestCase):
    def test_an_id_is_used_once_at_a_time(self):
        calls = WindowCalls()
        calls.begin("run-1")
        with self.assertRaises(WindowCallActiveError):
            calls.begin("run-1")
        self.assertEqual(calls.active(), ["run-1"])
        calls.end("run-1")
        calls.begin("run-1")

    def test_a_waiting_call_is_dropped_when_cancelled(self):
        calls = WindowCalls()
        calls.begin("run-1")
        self.assertTrue(calls.wanted("run-1"))
        self.assertEqual(calls.cancel("run-1")["was"], "waiting")
        self.assertFalse(calls.wanted("run-1"))
        self.assertFalse(calls.wanted("unknown"))
        with self.assertRaises(KeyError):
            calls.cancel("unknown")

    def test_a_running_call_is_cancelled_by_its_engine(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), EngineAnswering)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            calls = WindowCalls()
            for request_id in ("run-1", "gone"):
                calls.begin(request_id)
                calls.running(request_id, server.server_port)
            self.assertEqual(calls.cancel("run-1"), {"state": "cancelling", "was": "running"})
            self.assertEqual(calls.cancel("gone")["state"], "finished")
        finally:
            server.shutdown()
            server.server_close()

    def test_the_request_id_is_required(self):
        self.assertEqual(window_request_id({"request_id": "a.b-c_1"}), "a.b-c_1")
        bad_bodies: list[object] = [{}, {"request_id": "a b"}, {"request_id": "x" * 65}, []]
        for bad in bad_bodies:
            with self.subTest(bad=bad), self.assertRaises(WindowRequestError):
                window_request_id(bad)

    def test_the_call_ends_when_its_answer_is_closed(self):
        ended = []
        answer = ended_with(Passthrough(200, "application/json", iter([b"{}"]), {}),
                            lambda: ended.append(True))
        answer.close()
        self.assertEqual(ended, [True])


class Engine:
    def __init__(self, folder):
        self.folder = folder

    def generations_dir(self):
        return self.folder


class Operations:
    def __init__(self, folder):
        self.engines = {"yue2": Engine(folder), "llama": object()}

    def configured(self):
        return [{"id": "yue2", "engine": "yue2"}, {"id": "qwen", "engine": "llama"}]

    def engine(self, engine_id):
        return self.engines[engine_id]


class Gateway:
    """Resolves entries; `studio_port` is where the YuE2 Studio answers, None when not loaded."""

    def __init__(self, studio_port=None):
        self.studio_port = studio_port

    def resolve(self, wanted, entries):
        for entry in entries:
            if entry["id"] == wanted:
                return entry
        raise NotConfiguredError(f"no {wanted}")

    def where_loaded(self, wanted, shape):
        if self.studio_port is None:
            raise ModelNotLoadedError(wanted, [])
        return self.studio_port - 10000, "yue2-3b"


class StudioDeletes(BaseHTTPRequestHandler):
    """YuE2 Studio's library delete as `routes.delete_song` answers it.

    Both refusals are a 404 with FastAPI's `detail`: `busy` is a folder a job is
    writing (`library.delete_directory`'s ValueError), `gone` an unknown song.
    """

    deleted: list[str] = []  # noqa: RUF012 - a record kept by the test, on purpose

    def do_DELETE(self):  # noqa: N802 - the name http.server calls
        name = self.path.rsplit("/", 1)[1]
        details = {"busy": "an active job is writing to this directory (job #7); cancel it first",
                   "gone": f"no such item: {name}"}
        if name == "boom":
            self.send_response(500)
            self.send_header("Content-Length", "13")
            self.end_headers()
            self.wfile.write(b"Server error.")
            return
        if name in details:
            status, body = 404, {"detail": details[name]}
        else:
            StudioDeletes.deleted.append(name)
            status, body = 200, {"deleted": name}
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        """Quiet."""


class GenerationFilesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        song = self.folder / "song-a-w101"
        song.mkdir()
        (song / "audio.flac").write_bytes(b"F" * 10)
        (song / "window.wav").write_bytes(b"RIFF")
        self.router = Router()
        self.windows = WindowCalls()
        self.gateway = Gateway()
        register_files(self.router, GenerationFiles(self.gateway, Operations(self.folder), self.windows),
                       self.windows)

    def tearDown(self):
        self.temporary.cleanup()

    def route(self, method, path, **query):
        matched = self.router.match(method, path)
        assert matched is not None, path
        handler, captured = matched
        return handler(query=query, **captured)

    def test_the_files_of_a_generation(self):
        audio = self.route("GET", "/v1/audio/music/generations/song-a-w101/audio", model="yue2")
        self.assertEqual((audio.content_type, b"".join(audio.chunks)), ("audio/flac", b"F" * 10))
        clip = self.route("GET", "/v1/audio/music/generations/song-a-w101/window-audio", model="yue2")
        self.assertEqual(b"".join(clip.chunks), b"RIFF")

    def test_with_the_engine_up_the_studio_deletes(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), StudioDeletes)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            self.gateway.studio_port = server.server_port
            for name in ("busy", "gone", "boom"):
                (self.folder / name).mkdir()
            self.assertEqual(self.route("DELETE", "/v1/audio/music/generations/song-a-w101", model="yue2"),
                             {"deleted": "song-a-w101"})
            self.assertEqual(StudioDeletes.deleted, ["song-a-w101"])
            self.assertTrue((self.folder / "song-a-w101").exists())     # the studio's job, not ours
            with self.assertRaisesRegex(WindowCallActiveError, "active job is writing"):
                self.route("DELETE", "/v1/audio/music/generations/busy", model="yue2")
            with self.assertRaisesRegex(FileNotFoundError, "no generation gone"):
                self.route("DELETE", "/v1/audio/music/generations/gone", model="yue2")
            with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
                self.route("DELETE", "/v1/audio/music/generations/boom", model="yue2")
        finally:
            server.shutdown()
            server.server_close()

    def test_the_source_of_a_window_call_is_not_deleted(self):
        self.windows.begin("run-1", "song-a-w101")
        with self.assertRaisesRegex(WindowCallActiveError, "run-1"):
            self.route("DELETE", "/v1/audio/music/generations/song-a-w101", model="yue2")
        self.assertTrue((self.folder / "song-a-w101").exists())

    def test_a_delete_removes_the_folder(self):
        self.assertEqual(self.route("DELETE", "/v1/audio/music/generations/song-a-w101", model="yue2"),
                         {"deleted": "song-a-w101"})
        self.assertFalse((self.folder / "song-a-w101").exists())

    def test_refusals(self):
        path = "/v1/audio/music/generations/{}/audio"
        with self.assertRaisesRegex(ValueError, r"\?model="):
            self.route("GET", path.format("song-a-w101"))
        with self.assertRaisesRegex(ValueError, "keeps no generations"):
            self.route("GET", path.format("song-a-w101"), model="qwen")
        with self.assertRaises(FileNotFoundError):
            self.route("GET", path.format("nobody"), model="yue2")
        with self.assertRaises(ValueError):
            self.route("GET", path.format(".hidden"), model="yue2")
        (self.folder / "full-song").mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "window.wav"):
            self.route("GET", "/v1/audio/music/generations/full-song/window-audio", model="yue2")

    def test_the_cancel_route(self):
        with self.assertRaises(KeyError):
            self.route("DELETE", "/v1/audio/music/window-generations/nobody")


class Lease:
    port = 1234
    model_name = "yue2-3b"
    drop = ()
    text = False
    progress = None
    instance_id = "music-yue2"

    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class CardGateway:
    """The gateway as the forwarder uses it, with the card's answer chosen by the test.

    `meanwhile` runs just before the place is handed over, as a cancel landing there would.
    """

    def __init__(self, fail=None, meanwhile=None):
        self.fail = fail
        self.meanwhile = meanwhile
        self.lease = Lease()
        self.asked: list[tuple] = []

    def acquire(self, wanted, shape=None, settings=None, still_wanted=None):
        self.asked.append((wanted, shape, still_wanted()))
        if self.fail is not None:
            raise self.fail
        if self.meanwhile is not None:
            self.meanwhile()
        return self.lease


class ForwardingTests(unittest.TestCase):
    def setUp(self):
        self.windows = WindowCalls()
        self.sent: list[object] = []

    def send(self, lease):
        self.sent.append(lease)
        return Passthrough(200, "application/json", iter([b"{}"]), {})

    def handle(self, gateway, payload, send=None):
        ask = WindowAsk(payload, "yue2", None, lambda: True)
        return window_call(gateway, music_windows.WINDOW_PATH, ask, self.windows, send or self.send)

    def test_a_window_call_is_known_until_its_answer_closes(self):
        gateway = CardGateway()
        answer = self.handle(gateway, {"model": "yue2", "request_id": "run-1"})
        self.assertEqual(gateway.asked, [("yue2", music_windows.WINDOW_PATH, True)])
        self.assertEqual((self.windows.active(), self.sent), (["run-1"], [gateway.lease]))
        with self.assertRaises(WindowCallActiveError):
            self.handle(gateway, {"model": "yue2", "request_id": "run-1"})
        answer.close()
        self.assertEqual(self.windows.active(), [])

    def test_a_call_that_never_got_its_place_is_forgotten(self):
        with self.assertRaises(RuntimeError):
            self.handle(CardGateway(fail=RuntimeError("queue full")),
                        {"model": "yue2", "request_id": "run-2"})
        self.assertEqual(self.windows.active(), [])

    def test_a_failed_forward_gives_the_place_back(self):
        gateway = CardGateway()

        def broken(lease):
            raise ConnectionError("engine gone")
        with self.assertRaises(ConnectionError):
            self.handle(gateway, {"model": "yue2", "request_id": "run-3"}, send=broken)
        self.assertTrue(gateway.lease.released)
        self.assertEqual(self.windows.active(), [])

    def test_a_cancel_at_the_hand_over_sends_nothing(self):
        gateway = CardGateway(meanwhile=lambda: self.windows.cancel("run-4"))
        answer = self.handle(gateway, {"model": "yue2", "request_id": "run-4"})
        self.assertEqual(json.loads(b"".join(answer.chunks)),
                         {"request_id": "run-4", "candidates": [], "cancelled": True})
        self.assertEqual(self.sent, [])
        self.assertTrue(gateway.lease.released)
        self.assertEqual(self.windows.active(), [])

    def test_the_source_is_recorded_for_the_delete_guard(self):
        self.handle(CardGateway(), {"model": "yue2", "request_id": "run-5", "source": "song-a"})
        self.assertEqual(self.windows.in_use("song-a"), "run-5")
        self.assertIsNone(self.windows.in_use("other"))

    def test_no_request_id_is_a_422_before_the_queue(self):
        gateway = CardGateway()
        with self.assertRaises(WindowRequestError):
            self.handle(gateway, {"model": "yue2"})
        self.assertEqual(gateway.asked, [])


if __name__ == "__main__":
    unittest.main()
