"""The YuE2 Studio's generations listed, and a kept bundle put back under its own id."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from ai_lab.api.router import Router
from ai_lab.api.routes.music_transfers import MANIFEST, register_transfers
from ai_lab.api.routes.music_windows import GenerationFiles, WindowCalls
from ai_lab.engines.music.transfer import TransferSettings
from ai_lab.gateway.errors import WindowCallActiveError
from ai_lab.raw_body import RawBody, is_raw
from tests.api.test_music_transfers import (
    CHUNK,
    Engine,
    Gateway,
    Headers,
    Operations,
    bundle_bytes,
    listing,
    song,
    tar_of,
)


def raw(data: bytes, length: int | None = None) -> RawBody:
    return RawBody("application/x-tar", len(data) if length is None else length, io.BytesIO(data))


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.here = self.root / "mac"
        self.here.mkdir()
        self.transfer = TransferSettings.from_mapping(
            {"machine": "ai-lab-mac", "chunk_bytes": CHUNK, "list_most": 2,
             "put_back_most_bytes": 1 << 20})
        self.windows = WindowCalls()
        self.files = GenerationFiles(Gateway(), Operations(Engine(self.here, self.transfer)), self.windows)
        self.router = Router()
        register_transfers(self.router, self.files)

    def tearDown(self):
        self.temporary.cleanup()

    def route(self, method, path, body=None, run="run-7", **query):
        matched = self.router.match(method, path)
        assert matched is not None, path
        handler, captured = matched
        headers = Headers({"x-run-id": run} if run is not None else {})
        return handler(query={"model": "yue2", **query}, body=body, headers=headers, **captured)

    # -- the list ----------------------------------------------------------------

    def made(self, name, seconds, at, request=None, result=None):
        folder = song(self.here, name)
        (folder / "request.json").write_text(json.dumps(request or {"style": "rap", "lyrics": "la", "seed": 1}))
        if result is not False:
            (folder / "result.json").write_text(json.dumps(result or {"status": "complete",
                                                                      "audio_seconds": seconds}))
            os.utime(folder / "result.json", (at, at))
        return folder

    def test_the_list_is_newest_first_paged_and_recognisable(self):
        self.made("old", 10.0, 1_000_000)
        self.made("new", 314.598, 3_000_000, request={"style": "x" * 400, "lyrics": " [Verse]\nhi",
                                                     "seed": 9, "cot": "melody", "cfg_scale": 1.5})
        self.made("mid", 20.0, 2_000_000, result={"status": "complete", "audio_seconds": 20.0,
                                                  "sources": {"song": "old"}})
        (self.here / ".pull-x").mkdir()
        page = self.route("GET", "/v1/audio/music/generations", run=None)
        self.assertEqual((page["machine"], page["total"], page["offset"], page["limit"]), ("ai-lab-mac", 3, 0, 2))
        first, second = page["generations"]
        self.assertEqual([first["generation_id"], second["generation_id"]], ["new", "mid"])
        self.assertEqual((first["seconds"], first["seed"], first["cot"], first["cfg_scale"], first["kind"]),
                         (314.6, 9, "melody", 1.5, "song"))
        self.assertEqual(first["style"], "x" * 300 + "…")
        self.assertEqual(first["lyrics_start"], "[Verse]\nhi")
        self.assertEqual((first["files"], first["bytes"], first["complete"]), (3, 10000 + len(
            (self.here / "new" / "request.json").read_bytes()) + len((self.here / "new" / "result.json").read_bytes()), True))
        self.assertTrue(first["made_at"].startswith("1970-02-04T"))
        self.assertEqual((second["kind"], second["source"]), ("window", "old"))
        rest = self.route("GET", "/v1/audio/music/generations", run=None, offset="2")
        self.assertEqual([item["generation_id"] for item in rest["generations"]], ["old"])

    def test_an_unfinished_folder_is_listed_as_not_complete(self):
        self.made("half", 0, 0, result=False)
        (item,) = self.route("GET", "/v1/audio/music/generations", run=None)["generations"]
        self.assertEqual((item["complete"], item["seconds"]), (False, None))

    def test_list_limits_are_refused_not_bent(self):
        for query, words in (({"limit": "3"}, "limit must be a whole number from 1 to 2"),
                             ({"limit": "0"}, "limit"), ({"limit": "x"}, "limit"),
                             ({"offset": "-1"}, "offset must be a whole number from 0")):
            with self.subTest(query=query), self.assertRaisesRegex(ValueError, words):
                self.route("GET", "/v1/audio/music/generations", run=None, **query)

    # -- the put-back --------------------------------------------------------------

    def kept(self, name="song-a") -> bytes:
        return bundle_bytes(song(self.root / "kept", name))

    def put_back(self, name, body, run="run-7"):
        return self.route("POST", f"/v1/audio/music/generations/{name}/put-back", body, run)

    def test_a_kept_bundle_goes_back_under_its_id(self):
        answer = self.put_back("song-a", raw(self.kept()))
        self.assertEqual(answer, {"generation_id": "song-a", "machine": "ai-lab-mac",
                                  "files": 2, "bytes": 10014, "checked": True})
        self.assertEqual((self.here / "song-a" / "audio.flac").read_bytes(), b"F" * 10000)
        self.assertEqual([path.name for path in self.here.iterdir()], ["song-a"])

    def test_refused_when_the_id_is_here_or_busy(self):
        song(self.here, "song-a")
        with self.assertRaisesRegex(WindowCallActiveError, "already on this machine"):
            self.put_back("song-a", raw(self.kept()))
        self.windows.begin("w-1", "song-b")
        with self.assertRaisesRegex(WindowCallActiveError, "window call w-1"):
            self.put_back("song-b", raw(self.kept("song-b")))

    def test_refusals_of_the_request(self):
        data = self.kept()
        cases = [
            (raw(data), None, "X-Run-Id"),
            ({"bundle": "x"}, "run-7", "application/x-tar"),
            (raw(b"", 0), "run-7", "1 to 1048576 bytes"),
            (raw(data, (1 << 20) + 1), "run-7", "put_back_most_bytes"),
        ]
        for body, run, words in cases:
            with self.subTest(words=words), self.assertRaisesRegex(ValueError, words):
                self.put_back("song-a", body, run)
        with self.assertRaisesRegex(ValueError, "not a generation"):
            self.put_back("..", raw(data))
        self.assertEqual(list(self.here.iterdir()), [])

    def test_a_wrong_bundle_is_refused_and_nothing_is_kept(self):
        good = (("a.bin", b"abc"),)
        named = json.loads(listing(*good)) | {"generation_id": "song-a"}
        cases = [
            (self.kept("song-b"), "is 'song-b', not 'song-a'"),
            (tar_of([("a.bin", b"abc")]), "manifest"),
            (tar_of([(MANIFEST, json.dumps(named).encode()), ("a.bin", b"abd")]), "different"),
            (tar_of([(MANIFEST, json.dumps(named).encode())]), "without: a.bin"),
            (b"not a tar at all" * 100, "refused"),
            (self.kept()[:6000], "refused"),
        ]
        for data, words in cases:
            with self.subTest(words=words), self.assertRaisesRegex(ValueError, words):
                self.put_back("song-a", raw(data))
            self.assertEqual(list(self.here.iterdir()), [])


class RawBodyTests(unittest.TestCase):
    def test_reads_stop_at_the_announced_length(self):
        body = RawBody("application/x-tar", 5, io.BytesIO(b"abcdefgh"))
        self.assertEqual((body.read(3), body.read(), body.read()), (b"abc", b"de", b""))

    def test_only_a_tar_is_kept_as_a_stream(self):
        self.assertTrue(is_raw("application/x-tar"))
        self.assertTrue(is_raw("Application/X-Tar; charset=binary"))
        self.assertFalse(is_raw("application/json"))


class SettingsTests(unittest.TestCase):
    def test_defaults_and_refusals(self):
        settings = TransferSettings.from_mapping({})
        self.assertEqual((settings.list_most, settings.put_back_most_bytes), (200, 1 << 30))
        for raw_settings, words in (({"list_most": 0}, "list_most"), ({"list_most": 2.0}, "list_most"),
                                    ({"put_back_most_bytes": 10}, "put_back_most_bytes")):
            with self.subTest(words=words), self.assertRaisesRegex(ValueError, words):
                TransferSettings.from_mapping(raw_settings)


if __name__ == "__main__":
    unittest.main()
