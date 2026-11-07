"""Moving a YuE2 song between AI-Labs: the bundle, its check, and the pull through a fake scheduler door."""

import io
import json
import tarfile
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ai_lab.api.router import Router
from ai_lab.api.routes.music_transfers import (
    MANIFEST,
    manifest,
    receive,
    register_transfers,
    run_id_of,
    tar_pieces,
)
from ai_lab.api.routes.music_windows import GenerationFiles, WindowCalls
from ai_lab.engines.music.transfer import TransferSettings
from ai_lab.engines.music.yue2 import Yue2Engine
from ai_lab.gateway.errors import (
    ModelNotLoadedError,
    NotConfiguredError,
    SchedulerDoorError,
    TransferCheckError,
    WindowCallActiveError,
)

CHUNK = 4096


def song(folder: Path, name="song-a") -> Path:
    path = folder / name
    path.mkdir(parents=True)
    (path / "audio.flac").write_bytes(b"F" * 10000)
    (path / "request.json").write_text('{"seed": 9202}')
    return path


def bundle_bytes(path: Path) -> bytes:
    return b"".join(tar_pieces(path, manifest(path, CHUNK)))


def tar_of(members: list[tuple[str, bytes]]) -> bytes:
    """A bundle written by hand, for the refusals."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


def listing(*files: tuple[str, bytes]) -> bytes:
    import hashlib
    return json.dumps({"files": [{"name": name, "size": len(data),
                                  "sha256": hashlib.sha256(data).hexdigest()}
                                 for name, data in files]}).encode()


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_a_bundle_round_trip_is_checked_file_by_file(self):
        source = song(self.root / "here")
        data = bundle_bytes(source)
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            self.assertEqual(tar.getnames(), [MANIFEST, "audio.flac", "request.json"])
        self.assertEqual(receive(io.BytesIO(data), self.root / "copy", CHUNK), (2, 10014))
        self.assertEqual((self.root / "copy" / "audio.flac").read_bytes(), b"F" * 10000)

    def test_a_folder_that_is_not_flat_is_refused(self):
        source = song(self.root / "here")
        (source / "sub").mkdir()
        with self.assertRaisesRegex(ValueError, "not a plain file"):
            manifest(source, CHUNK)

    def test_refusals_of_a_wrong_bundle(self):
        good = (("a.bin", b"abc"),)
        cases = [
            (tar_of([("a.bin", b"abc")]), "manifest"),
            (tar_of([(MANIFEST, listing(*good)), ("a.bin", b"abd")]), "different"),
            (tar_of([(MANIFEST, listing(*good)), ("a.bin", b"abc"), ("b.bin", b"x")]), "does not list"),
            (tar_of([(MANIFEST, listing(*good, ("b.bin", b"x"))), ("a.bin", b"abc")]), "without: b.bin"),
            (tar_of([(MANIFEST, listing(("../a", b"abc")))]), "is not one"),
            (tar_of([(MANIFEST, b'{"nothing": 1}')]), "is not one"),
        ]
        for number, (data, words) in enumerate(cases):
            with self.subTest(words=words), self.assertRaisesRegex(TransferCheckError, words):
                receive(io.BytesIO(data), self.root / f"copy{number}", CHUNK)


DOOR_PREFIX = "/ai/ai-lab-linux"


class FakeDoor(BaseHTTPRequestHandler):
    """This machine's scheduler door, with the other AI-Lab behind it.

    Answers the bundle of `served`, records each call's `X-Run-Id`, any token it got and any
    delete (a pull sends none), and gives the door's own refusals (a `scheduler` key) for the
    names that ask for them.
    """

    served: dict[str, bytes] = {}  # noqa: RUF012 - set by each test
    deletes: list[str] = []  # noqa: RUF012 - a record kept by the test, on purpose
    runs: list[str] = []  # noqa: RUF012 - a record kept by the test, on purpose
    tokens: list[str] = []  # noqa: RUF012 - a record kept by the test, on purpose

    def _seen(self) -> str | None:
        FakeDoor.runs.append(self.headers.get("X-Run-Id", ""))
        if self.headers.get("Authorization"):
            FakeDoor.tokens.append(self.headers["Authorization"])
        if not self.path.startswith(DOOR_PREFIX + "/"):
            self._door(404, "unknown_provider", "this scheduler serves only 'ai-lab-mac'")
            return None
        return self.path.split("/")[7].split("?")[0]

    def do_GET(self):  # noqa: N802 - the name http.server calls
        name = self._seen()
        if name is None:
            return
        if name == "busy":
            self._answer(409, b'{"error": "in use"}')
        elif name == "lost":
            self._door(502, "partner_lost", "the partner no longer knows this call")
        elif name == "unrouted":
            self._door(404, "unknown_provider", "the partner's calls are not set up here")
        elif name in FakeDoor.served:
            self._answer(200, FakeDoor.served[name])
        else:
            self._answer(404, b'{"error": "no generation"}')

    def do_DELETE(self):  # noqa: N802 - the name http.server calls
        FakeDoor.deletes.append(self.path)                     # a pull must never send one
        self._answer(200, b'{"deleted": true}')

    def _door(self, status, reason, detail):
        self._answer(status, json.dumps({"scheduler": {"reason": reason, "detail": detail}}).encode())

    def _answer(self, status, body):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        """Quiet."""


class Engine:
    def __init__(self, folder, transfer):
        self.folder = folder
        self.transfer = transfer

    def generations_dir(self):
        return self.folder


class Operations:
    def __init__(self, engine):
        self.engines = {"yue2": engine}

    def configured(self):
        return [{"id": "yue2", "engine": "yue2"}]

    def engine(self, engine_id):
        return self.engines[engine_id]


class Gateway:
    """YuE2 not loaded, unless `studio_port` says where its Studio answers."""

    studio_port = None

    def resolve(self, wanted, entries):
        for entry in entries:
            if entry["id"] == wanted:
                return entry
        raise NotConfiguredError(f"no {wanted}")

    def where_loaded(self, wanted, shape):
        if self.studio_port is None:
            raise ModelNotLoadedError(wanted, [])
        return self.studio_port - 10000, "yue2-3b"


class StudioJobs(BaseHTTPRequestHandler):
    """A YuE2 Studio whose running job writes `song-b`."""

    def do_GET(self):  # noqa: N802 - the name http.server calls
        body = json.dumps({"jobs": [{"status": "running", "output_dir": "/x/song-b", "params": {}},
                                    {"status": "done", "output_dir": "/x/song-a", "params": {}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        """Quiet."""


def serving(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class Headers(dict):
    """Request headers as the server hands them over: looked up without regard to case."""

    def get(self, key, default=None):
        for name, value in self.items():
            if name.lower() == key.lower():
                return value
        return default


class PullTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.here = self.root / "mac"
        self.here.mkdir()
        self.door = serving(FakeDoor)
        transfer = TransferSettings.from_mapping({
            "machine": "ai-lab-mac", "timeout_s": 5, "chunk_bytes": CHUNK,
            "scheduler": f"http://127.0.0.1:{self.door.server_port}", "peers": ["ai-lab-linux"]})
        self.windows = WindowCalls()
        self.gateway = Gateway()
        self.files = GenerationFiles(self.gateway, Operations(Engine(self.here, transfer)), self.windows)
        self.router = Router()
        register_transfers(self.router, self.files)
        FakeDoor.served = {"song-a": bundle_bytes(song(self.root / "linux"))}
        FakeDoor.deletes, FakeDoor.runs, FakeDoor.tokens = [], [], []

    def tearDown(self):
        self.door.shutdown()
        self.door.server_close()
        self.temporary.cleanup()

    def route(self, method, path, body=None, run="run-7"):
        matched = self.router.match(method, path)
        assert matched is not None, path
        handler, captured = matched
        headers = Headers({"x-run-id": run} if run is not None else {})
        return handler(query={"model": "yue2"}, body=body, headers=headers, **captured)

    def pull(self, name="song-a", peer="ai-lab-linux", run="run-7", **more):
        return self.route("POST", f"/v1/audio/music/generations/{name}/pull", {"from": peer, **more}, run)

    def test_the_song_is_copied_through_the_door_and_the_source_left_alone(self):
        answer = self.pull()
        self.assertEqual(answer, {"generation_id": "song-a", "machine": "ai-lab-mac", "files": 2,
                                  "bytes": 10014, "checked": True})
        self.assertEqual((self.here / "song-a" / "audio.flac").read_bytes(), b"F" * 10000)
        self.assertEqual(FakeDoor.runs, ["run-7"])                     # the bundle only
        self.assertEqual(FakeDoor.deletes, [])                         # music deletes the other copy
        self.assertEqual(FakeDoor.tokens, [])                         # no token between the AI-Labs
        self.assertEqual([path.name for path in self.here.iterdir()], ["song-a"])

    def test_a_pull_without_its_run_is_refused_before_any_call(self):
        for run in (None, "", "  "):
            with self.subTest(run=run), self.assertRaisesRegex(ValueError, "needs the header X-Run-Id"):
                self.pull(run=run)
        self.assertEqual((FakeDoor.runs, list(self.here.iterdir())), ([], []))

    def test_a_copy_that_fails_its_check_is_removed_and_the_source_kept(self):
        data = bytearray(FakeDoor.served["song-a"])
        data[data.index(b"FFFF")] = ord("G")
        FakeDoor.served["song-a"] = bytes(data)
        with self.assertRaisesRegex(TransferCheckError, "audio.flac arrived different"):
            self.pull()
        self.assertEqual((list(self.here.iterdir()), FakeDoor.deletes), ([], []))   # source untouched

    def test_the_doors_404_is_not_a_missing_song(self):
        with self.assertRaisesRegex(SchedulerDoorError, r"HTTP 404, unknown_provider"):
            self.pull("unrouted")
        self.assertEqual(list(self.here.iterdir()), [])

    def test_the_old_retry_field_is_refused(self):
        with self.assertRaisesRegex(ValueError, "Unknown pull fields: retry_delete"):
            self.pull(retry_delete=True)
        with self.assertRaisesRegex(ValueError, "Unknown pull fields: extra"):
            self.pull(extra=1)
        self.assertEqual(FakeDoor.runs, [])

    def test_refusals(self):
        with self.assertRaisesRegex(FileNotFoundError, "has no generation nobody"):
            self.pull("nobody")
        with self.assertRaisesRegex(WindowCallActiveError, "refused"):
            self.pull("busy")
        with self.assertRaisesRegex(SchedulerDoorError, "partner_lost"):
            self.pull("lost")
        with self.assertRaisesRegex(ValueError, "no peer 'other'"):
            self.pull(peer="other")
        (self.here / "song-a").mkdir()
        with self.assertRaisesRegex(WindowCallActiveError, "already on this machine"):
            self.pull()
        self.assertEqual(len(FakeDoor.runs), 3)
        self.assertEqual(FakeDoor.deletes, [])
        self.assertEqual([path.name for path in self.here.iterdir()], ["song-a"])   # no half-copy left

    def test_a_door_that_does_not_answer(self):
        port = self.door.server_port
        self.door.shutdown()
        self.door.server_close()
        self.door = serving(FakeDoor)                                # tearDown closes this one
        transfer = TransferSettings.from_mapping({"scheduler": f"http://127.0.0.1:{port}",
                                                  "peers": ["ai-lab-linux"], "timeout_s": 2})
        files = GenerationFiles(self.gateway, Operations(Engine(self.here, transfer)), self.windows)
        router = Router()
        register_transfers(router, files)
        matched = router.match("POST", "/v1/audio/music/generations/song-a/pull")
        assert matched is not None
        handler, captured = matched
        with self.assertRaisesRegex(SchedulerDoorError, "did not answer"):
            handler(query={"model": "yue2"}, body={"from": "ai-lab-linux"},
                    headers=Headers({"X-Run-Id": "run-7"}), **captured)
        self.assertEqual(list(self.here.iterdir()), [])

    def test_a_song_being_edited_is_not_pulled_or_bundled(self):
        self.windows.begin("run-1", "song-a")
        with self.assertRaisesRegex(WindowCallActiveError, "window call run-1"):
            self.pull()
        song(self.here, "song-a")
        with self.assertRaisesRegex(WindowCallActiveError, "window call run-1"):
            self.route("GET", "/v1/audio/music/generations/song-a/bundle")

    def test_the_bundle_route_and_a_studio_job_at_work(self):
        song(self.here, "song-a")
        song(self.here, "song-b")
        answer = self.route("GET", "/v1/audio/music/generations/song-a/bundle")
        self.assertEqual(answer.content_type, "application/x-tar")
        self.assertEqual(receive(io.BytesIO(b"".join(answer.chunks)), self.root / "copy", CHUNK), (2, 10014))
        studio = serving(StudioJobs)
        try:
            self.gateway.studio_port = studio.server_port
            with self.assertRaisesRegex(WindowCallActiveError, "Studio job is working on song-b"):
                self.route("GET", "/v1/audio/music/generations/song-b/bundle")
            self.route("GET", "/v1/audio/music/generations/song-a/bundle")
        finally:
            studio.shutdown()
            studio.server_close()


class RunIdTests(unittest.TestCase):
    def test_the_run_id_of_the_headers(self):
        self.assertEqual(run_id_of(Headers({"x-run-id": " run-1 "})), "run-1")
        self.assertIsNone(run_id_of(Headers({"x-run-id": " "})))
        self.assertIsNone(run_id_of(Headers()))
        self.assertIsNone(run_id_of(None))


class TransferSettingsTests(unittest.TestCase):
    def test_defaults_and_a_peer(self):
        settings = TransferSettings.from_mapping(None)
        self.assertEqual((settings.machine, settings.scheduler, settings.peers, settings.timeout_s,
                          settings.chunk_bytes), ("", "", {}, 60.0, 1 << 20))
        peer = TransferSettings.from_mapping({"scheduler": "http://127.0.0.1:8230/",
                                              "peers": ["ai-lab-linux"]}).peer("ai-lab-linux")
        self.assertEqual((peer.name, peer.url), ("ai-lab-linux", "http://127.0.0.1:8230/ai/ai-lab-linux"))
        with self.assertRaisesRegex(ValueError, "configured: none"):
            TransferSettings.from_mapping({"machine": "ai-lab-mac", "peers": []}).peer("ai-lab-linux")

    def test_refusals(self):
        door = "http://127.0.0.1:8230"
        cases: list[tuple[Any, str]] = [
            ([], "must be an object"),
            ({"other": 1}, "Unknown"),
            ({"machine": 1}, "machine must be text"),
            ({"timeout_s": 0}, "timeout_s"),
            ({"chunk_bytes": 10}, "chunk_bytes"),
            ({"scheduler": "h:8230"}, "scheduler must be"),
            ({"scheduler": 8230}, "scheduler must be"),
            ({"peers": ["ai-lab-linux"]}, "need the scheduler"),
            ({"scheduler": door, "peers": {"ai-lab-linux": {}}}, "list of provider names"),
            ({"scheduler": door, "peers": [1]}, "list of provider names"),
            ({"scheduler": door, "peers": ["a/b"]}, "not a provider name"),
            ({"scheduler": door, "peers": ["a", "a"]}, "twice"),
        ]
        for raw, words in cases:
            with self.subTest(words=words), self.assertRaisesRegex(ValueError, words):
                TransferSettings.from_mapping(raw)

    def test_the_engine_reads_its_section(self):
        engine = Yue2Engine.from_settings({"transfer": {"machine": "ai-lab-mac"}, "device": "mps"})
        self.assertEqual((engine.transfer.machine, engine.device), ("ai-lab-mac", "mps"))
        self.assertEqual(Yue2Engine().transfer.peers, {})

    def test_the_registry_accepts_the_section_and_refuses_a_wrong_one(self):
        from ai_lab.engines.registry import Registry
        engine = Registry({"yue2": {"transfer": {"machine": "ai-lab-mac"}}}).get("yue2")
        assert isinstance(engine, Yue2Engine)
        self.assertEqual(engine.transfer.machine, "ai-lab-mac")
        with self.assertRaisesRegex(ValueError, "Unknown yue2 transfer"):
            Registry({"yue2": {"transfer": {"machnie": "typo"}}})


if __name__ == "__main__":
    unittest.main()
