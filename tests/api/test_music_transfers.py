"""Moving a YuE2 song between AI-Labs: the bundle, its check, and the pull against a fake peer."""

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
    tar_pieces,
)
from ai_lab.api.routes.music_windows import GenerationFiles, WindowCalls
from ai_lab.engines.music.transfer import TransferSettings
from ai_lab.engines.music.yue2 import Yue2Engine
from ai_lab.gateway.errors import (
    ModelNotLoadedError,
    NotConfiguredError,
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


class FakePeer(BaseHTTPRequestHandler):
    """Another AI-Lab: answers the bundle of `served` and records deletes and tokens."""

    served: dict[str, bytes] = {}  # noqa: RUF012 - set by each test
    deletes: list[str] = []  # noqa: RUF012 - a record kept by the test, on purpose
    tokens: list[str] = []  # noqa: RUF012 - a record kept by the test, on purpose
    refuse_delete = False

    def do_GET(self):  # noqa: N802 - the name http.server calls
        FakePeer.tokens.append(self.headers.get("Authorization", ""))
        name = self.path.split("/")[5]
        if name == "busy":
            self._answer(409, b'{"error": "in use"}')
        elif name in FakePeer.served:
            self._answer(200, FakePeer.served[name])
        else:
            self._answer(404, b'{"error": "no generation"}')

    def do_DELETE(self):  # noqa: N802 - the name http.server calls
        if "/gone?" in self.path:
            self._answer(404, b'{"error": "no generation"}')
            return
        if FakePeer.refuse_delete:
            self._answer(409, b'{"error": "a job writes there"}')
            return
        FakePeer.deletes.append(self.path)
        self._answer(200, b'{"deleted": true}')

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


class PullTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.here = self.root / "mac"
        self.here.mkdir()
        self.peer = serving(FakePeer)
        tokens = self.root / "tokens.json"
        tokens.write_text('{"server": "s3cret"}')
        transfer = TransferSettings.from_mapping({
            "machine": "ai-lab-mac", "timeout_s": 5, "chunk_bytes": CHUNK,
            "peers": {"ai-lab-linux": {"url": f"http://127.0.0.1:{self.peer.server_port}",
                                       "token_file": str(tokens), "token_key": "server"}}})
        self.windows = WindowCalls()
        self.gateway = Gateway()
        self.files = GenerationFiles(self.gateway, Operations(Engine(self.here, transfer)), self.windows)
        self.router = Router()
        register_transfers(self.router, self.files)
        FakePeer.served = {"song-a": bundle_bytes(song(self.root / "linux"))}
        FakePeer.deletes, FakePeer.tokens, FakePeer.refuse_delete = [], [], False

    def tearDown(self):
        self.peer.shutdown()
        self.peer.server_close()
        self.temporary.cleanup()

    def route(self, method, path, body=None):
        matched = self.router.match(method, path)
        assert matched is not None, path
        handler, captured = matched
        return handler(query={"model": "yue2"}, body=body, **captured)

    def pull(self, name="song-a", peer="ai-lab-linux", **more):
        return self.route("POST", f"/v1/audio/music/generations/{name}/pull", {"from": peer, **more})

    def kept_here(self, name="song-a"):
        """A song pulled before: complete, as the Studio marks it."""
        folder = song(self.here, name)
        (folder / "result.json").write_text('{"status": "complete"}')
        return folder

    def test_the_song_moves_and_the_source_is_deleted(self):
        answer = self.pull()
        self.assertEqual(answer, {"generation_id": "song-a", "machine": "ai-lab-mac", "files": 2,
                                  "bytes": 10014, "source_removed": True})
        self.assertEqual((self.here / "song-a" / "audio.flac").read_bytes(), b"F" * 10000)
        self.assertEqual(FakePeer.deletes, ["/v1/audio/music/generations/song-a?model=yue2"])
        self.assertEqual(set(FakePeer.tokens), {"Bearer s3cret"})
        self.assertEqual([path.name for path in self.here.iterdir()], ["song-a"])

    def test_a_copy_that_fails_its_check_is_removed_and_the_source_kept(self):
        data = bytearray(FakePeer.served["song-a"])
        data[data.index(b"FFFF")] = ord("G")
        FakePeer.served["song-a"] = bytes(data)
        with self.assertRaisesRegex(TransferCheckError, "audio.flac arrived different"):
            self.pull()
        self.assertEqual((list(self.here.iterdir()), FakePeer.deletes), ([], []))

    def test_a_source_that_is_not_deleted_is_said(self):
        FakePeer.refuse_delete = True
        answer = self.pull()
        self.assertFalse(answer["source_removed"])
        self.assertIn("a job writes there", answer["source_error"])
        self.assertTrue((self.here / "song-a").is_dir())

    def test_a_retry_sends_only_the_delete(self):
        FakePeer.refuse_delete = True
        self.assertFalse(self.pull()["source_removed"])
        (self.here / "song-a" / "result.json").write_text('{"status": "complete"}')
        FakePeer.refuse_delete, FakePeer.tokens = False, []
        answer = self.pull(retry_delete=True)
        self.assertEqual(answer, {"generation_id": "song-a", "machine": "ai-lab-mac", "source_removed": True})
        self.assertEqual(FakePeer.deletes, ["/v1/audio/music/generations/song-a?model=yue2"])
        self.assertEqual(FakePeer.tokens, [])                     # nothing fetched

    def test_a_retry_whose_peer_copy_is_gone_counts_as_removed(self):
        self.kept_here("gone")
        self.assertTrue(self.pull("gone", retry_delete=True)["source_removed"])

    def test_a_retry_that_fails_again_says_why(self):
        self.kept_here()
        FakePeer.refuse_delete = True
        answer = self.pull(retry_delete=True)
        self.assertFalse(answer["source_removed"])
        self.assertIn("a job writes there", answer["source_error"])

    def test_retry_refusals(self):
        with self.assertRaisesRegex(FileNotFoundError, "pull it first"):
            self.pull(retry_delete=True)
        song(self.here, "song-a")                                  # no result.json: not complete
        with self.assertRaisesRegex(WindowCallActiveError, "not a complete song"):
            self.pull(retry_delete=True)
        (self.here / "song-a" / "result.json").write_text('{"status": "failed"}')
        with self.assertRaisesRegex(WindowCallActiveError, "not a complete song"):
            self.pull(retry_delete=True)
        self.assertEqual(FakePeer.deletes, [])
        with self.assertRaisesRegex(ValueError, "retry_delete must be"):
            self.pull(retry_delete="yes")
        with self.assertRaisesRegex(ValueError, "Unknown pull fields: extra"):
            self.pull(extra=1)

    def test_refusals(self):
        with self.assertRaisesRegex(FileNotFoundError, "has no generation nobody"):
            self.pull("nobody")
        with self.assertRaisesRegex(WindowCallActiveError, "refused"):
            self.pull("busy")
        with self.assertRaisesRegex(ValueError, "no peer 'other'"):
            self.pull(peer="other")
        (self.here / "song-a").mkdir()
        with self.assertRaisesRegex(WindowCallActiveError, "already on this machine"):
            self.pull()
        self.assertEqual(FakePeer.tokens, ["Bearer s3cret", "Bearer s3cret"])

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


class TransferSettingsTests(unittest.TestCase):
    def test_defaults_and_a_peer(self):
        settings = TransferSettings.from_mapping(None)
        self.assertEqual((settings.machine, settings.peers, settings.timeout_s, settings.chunk_bytes),
                         ("", {}, 60.0, 1 << 20))
        with tempfile.TemporaryDirectory() as folder:
            tokens = Path(folder) / "t.json"
            tokens.write_text('{"a": "x"}')
            peer = TransferSettings.from_mapping({"peers": {"linux": {
                "url": "http://h:8090/", "token_file": str(tokens), "token_key": "a"}}}).peer("linux")
            self.assertEqual((peer.url, peer.token()), ("http://h:8090", "x"))
            tokens.write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "has no 'a'"):
                peer.token()
            tokens.unlink()
            with self.assertRaisesRegex(RuntimeError, "cannot be read"):
                peer.token()

    def test_refusals(self):
        peer = {"url": "http://h:1", "token_file": "/t.json", "token_key": "a"}
        cases: list[tuple[Any, str]] = [
            ([], "must be an object"),
            ({"other": 1}, "Unknown"),
            ({"machine": 1}, "machine must be text"),
            ({"timeout_s": 0}, "timeout_s"),
            ({"chunk_bytes": 10}, "chunk_bytes"),
            ({"peers": []}, "peers must be an object"),
            ({"peers": {"p": {"url": "h"}}}, "needs exactly"),
            ({"peers": {"p": {**peer, "url": "ftp://h"}}}, "url must be"),
            ({"peers": {"p": {**peer, "token_file": "t.json"}}}, "absolute path"),
            ({"peers": {"p": {**peer, "token_key": ""}}}, "token_key"),
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
