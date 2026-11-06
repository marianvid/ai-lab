"""Regenerating a window of a YuE2 song: the request's checks, the backend and its engine host."""

import io
import json
import sys
import tempfile
import threading
import types
import unittest
import urllib.error
import urllib.request
from collections.abc import Iterator
from email.message import Message
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from ai_lab.engines.base import validate
from ai_lab.engines.music.yue2 import WINDOW_SETTINGS
from ai_lab.music.backends import yue2_web_backend
from ai_lab.music.backends.yue2_web_backend import (
    StudioRefusedError,
    WindowActiveError,
    WindowRefusedError,
    WindowSettings,
    Yue2WebBackend,
    ask_studio,
    check_generation_id,
    window_request,
)
from ai_lab.music.servers.yue2_server import WINDOW_PATH, Yue2Host

SETTINGS = WindowSettings.from_params(validate(WINDOW_SETTINGS, {}))
LYRICS = "[Verse]\nLa\n[Chorus]\nminut"


def body(**changes):
    request = {"model": "yue2", "request_id": "run-1", "source": "song-a",
               "window": {"start_s": 41.01, "end_s": 56.4}, "lyrics": LYRICS,
               "candidates": [{"seed": 101}, {"seed": 202}]}
    request.update(changes)
    return request


class WindowRequestTests(unittest.TestCase):
    def test_the_settings_fill_what_the_request_leaves_out(self):
        request = window_request(body(), SETTINGS, "yue2")
        self.assertEqual((request["start_s"], request["end_s"]), (41.0, 56.4))
        self.assertEqual(request["seeds"], [101, 202])
        self.assertEqual((request["context_s"], request["edge_s"], request["crossfade_s"],
                          request["swap_margin_s"]), (5.0, 0.4, 0.3, 2.5))
        self.assertEqual(request["sampling"], {"temperature": 1.0, "top_p": 0.95, "top_k": 100,
                                               "repetition_penalty": 1.2, "penalty_window": 50})

    def test_a_request_may_override_part_of_the_sampling(self):
        request = window_request(body(sampling={"top_k": 50}, context_s=3.0), SETTINGS, "yue2")
        self.assertEqual((request["sampling"]["top_k"], request["sampling"]["top_p"]), (50, 0.95))
        self.assertEqual(request["context_s"], 3.0)

    def test_refusals(self):
        cases = [
            (body(extra=1), "Unknown YuE2 window fields"),
            (body(model="other"), "serves yue2"),
            (body(request_id="a b"), "request_id"),
            (body(source="../x"), "source"),
            (body(window={"start_s": 1}), "window must be"),
            (body(window={"start_s": 10, "end_s": 10.5}), "must last"),
            (body(window={"start_s": 0, "end_s": 90}), "must last"),
            (body(lyrics=" "), "lyrics"),
            (body(candidates=[]), "candidates"),
            (body(candidates=[{"seed": 1}] * 6), "candidates"),
            (body(candidates=[{"seed": 1}, {"seed": 1}]), "differ"),
            (body(candidates=[{"seed": -1}]), "each candidate"),
            (body(context_s=1.0), "swap margin"),
            (body(sampling={"top_k": 1.5}), "sampling.top_k"),
            (body(sampling={"other": 1}), "Unknown sampling"),
            (body(sampling=[]), "sampling must be"),
        ]
        for request, words in cases:
            with self.subTest(words=words), self.assertRaisesRegex(WindowRefusedError, words):
                window_request(request, SETTINGS, "yue2")

    def test_generation_ids_cannot_leave_the_folder(self):
        self.assertEqual(check_generation_id("song-a.v1"), "song-a.v1")
        bad_names: list[object] = ["../x", "", ".hidden", "a/b", None]
        for bad in bad_names:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_generation_id(bad)


class FakeStudio:
    """YuE2 Studio's job API, in memory: each job ends at once with the status given.

    `after_job` runs each time a finished job is read, as if something happened meanwhile.
    """

    def __init__(self, status="done", refuse=None, after_job=None):
        self.status = status
        self.refuse = refuse
        self.after_job = after_job
        self.posted: list[dict] = []
        self.deleted: list[str] = []

    def __call__(self, method, path, payload=None):
        if method == "POST":
            if self.refuse:
                raise StudioRefusedError(400, self.refuse)
            self.posted.append(payload)
            return {"job": {"id": len(self.posted)}}
        if method == "DELETE":
            self.deleted.append(path)
            return {}
        job_id = int(path.rsplit("/", 1)[1])
        if self.after_job is not None:
            self.after_job()
        seed = self.posted[job_id - 1]["params"].get("seed")
        return {"status": self.status, "output_dir": f"/out/song-a-w{seed}", "error": "boom",
                "result": {"audio_seconds": 248.9, "window": {
                    "seed": seed, "tokens_changed_pct": 60.0, "short_window": False,
                    "timings": {"total_s": 100.0}, "window_s": [41.0, 56.4],
                    "clip_s": [36.0, 61.4], "noise_seed": 9202}}}


def backend(studio):
    return Yue2WebBackend("http://studio", "yue2", "full", window=SETTINGS, studio=studio)


class WindowBackendTests(unittest.TestCase):
    def test_one_studio_job_per_candidate_and_the_answer(self):
        studio = FakeStudio()
        answer = backend(studio).generate_window(body())
        self.assertEqual([job["params"]["seed"] for job in studio.posted], [101, 202])
        self.assertEqual(studio.posted[0]["kind"], "window")
        self.assertEqual(studio.posted[0]["params"]["swap_margin_s"], 2.5)
        self.assertEqual(answer["clip_s"], [36.0, 61.4])
        self.assertEqual(answer["noise_seed"], 9202)
        self.assertEqual([item["generation_id"] for item in answer["candidates"]],
                         ["song-a-w101", "song-a-w202"])
        self.assertFalse(answer["cancelled"])
        self.assertNotIn("clip_s", answer["candidates"][0])

    def test_a_studio_refusal_is_a_422(self):
        studio = FakeStudio(refuse="the lyrics must keep the source's section tags")
        with self.assertRaisesRegex(WindowRefusedError, "section tags"):
            backend(studio).generate_window(body())

    def test_a_failed_job_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "boom"):
            backend(FakeStudio(status="failed")).generate_window(body())

    def test_a_cancelled_job_ends_the_call_and_leaves_no_folder(self):
        studio = FakeStudio(status="cancelled")
        answer = backend(studio).generate_window(body())
        self.assertEqual(answer["candidates"], [])
        self.assertEqual(studio.deleted, ["/api/library/songs/song-a-w101"])

    def test_a_folder_already_gone_is_no_error(self):
        studio = FakeStudio(status="cancelled")

        def refuse_deletes(method, path, payload=None):
            if method == "DELETE":
                raise StudioRefusedError(404, "no such item")
            return studio(method, path, payload)
        answer = backend(refuse_deletes).generate_window(body())
        self.assertTrue(answer["cancelled"] is False and answer["candidates"] == [])

    def test_without_settings_windows_are_refused(self):
        made = Yue2WebBackend("http://studio", "yue2", "full", studio=FakeStudio())
        with self.assertRaisesRegex(WindowRefusedError, "without window settings"):
            made.generate_window(body())

    def test_one_request_id_at_a_time_and_its_cancel(self):
        studio = FakeStudio()
        made = backend(studio)
        replies: list[dict] = []
        studio.after_job = lambda: replies.append(made.cancel_window("run-1"))
        with self.assertRaises(KeyError):
            made.cancel_window("nobody")
        answer = made.generate_window(body())
        self.assertEqual((replies[0]["state"], replies[0]["job"]), ("cancelling", 1))
        self.assertEqual(studio.deleted, ["/api/jobs/1"])
        self.assertEqual((len(answer["candidates"]), answer["cancelled"]), (1, True))

    def test_a_second_call_with_the_same_id_is_refused_while_the_first_runs(self):
        studio = FakeStudio()
        made = backend(studio)
        refused: list[Exception] = []

        def second_call():
            try:
                made.generate_window(body())
            except WindowActiveError as error:
                refused.append(error)
        studio.after_job = second_call
        made.generate_window(body(candidates=[{"seed": 1}]))
        self.assertEqual(len(refused), 1)

    def test_the_studio_error_carries_its_status_and_detail(self):
        error = urllib.error.HTTPError("http://studio/api/jobs", 400, "bad", Message(),
                                       io.BytesIO(b'{"detail": "window ends after the song"}'))
        with patch.object(urllib.request, "urlopen", side_effect=error), \
                self.assertRaises(StudioRefusedError) as caught:
            ask_studio("http://studio")("POST", "/api/jobs", {})
        self.assertEqual((caught.exception.status, caught.exception.reason),
                         (400, "window ends after the song"))


class FullSongTests(unittest.TestCase):
    def test_the_answer_carries_the_generation_id(self):
        fake = types.SimpleNamespace(
            read=lambda path, dtype=None: ([0.0] * 480, 48000),
            write=lambda target, samples, rate, **_: target.write(b"RIFF"))
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {"soundfile": fake}):
            output = Path(folder) / "AI-Lab-generation-1"
            output.mkdir()
            (output / "audio.flac").write_bytes(b"fLaC")
            (output / "score.abc").write_text("X:1\n")
            replies: Iterator[dict] = iter([{"job": {"id": 1}}, {"status": "done", "output_dir": str(output),
                                                               "result": {"audio_seconds": 0.01}}])
            made = Yue2WebBackend("http://studio", "yue2", "full",
                                  studio=lambda *args: next(replies))
            answer = made.generate({"prompt": "rap", "lyrics": LYRICS, "instrumental": False,
                                    "seed": 1})
        self.assertEqual(answer["generation_id"], "AI-Lab-generation-1")
        self.assertEqual(answer["score_abc"], "X:1\n")

    def test_a_failed_song_is_an_error(self):
        replies: Iterator[dict] = iter([{"job": {"id": 1}}, {"status": "failed", "error": "out of memory"}])
        made = Yue2WebBackend("http://studio", "yue2", "full", studio=lambda *args: next(replies))
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            made.generate({"prompt": "rap", "lyrics": LYRICS, "instrumental": False})

    def test_a_song_that_never_ends_times_out(self):
        made = Yue2WebBackend("http://studio", "yue2", "full",
                              studio=lambda *args: {"status": "running", "job": {"id": 1}})
        with patch.object(yue2_web_backend, "SONG_LIMIT_S", 0), self.assertRaises(TimeoutError):
            made.generate({"prompt": "rap", "lyrics": LYRICS, "instrumental": False})


class Answering:
    """A window backend whose answers the test chooses."""

    def __init__(self, failure=None):
        self.failure = failure

    def generate(self, body):
        return {"song": True}

    def generate_window(self, body):
        if self.failure is not None:
            raise self.failure
        return {"window": body["request_id"]}

    def cancel_window(self, request_id):
        if request_id == "nobody":
            raise KeyError("no window call nobody is running")
        return {"state": "cancelling"}


def call(backend_object, method, path, payload=None):
    handler = type("Bound", (Yue2Host,), {"backend": backend_object})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}",
                                     data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)
    finally:
        server.shutdown()
        server.server_close()


class Yue2HostTests(unittest.TestCase):
    def test_the_window_call_and_the_song_call(self):
        self.assertEqual(call(Answering(), "POST", WINDOW_PATH, {"request_id": "r"}),
                         (200, {"window": "r"}))
        self.assertEqual(call(Answering(), "POST", "/v1/audio/music/generations", {"x": 1}),
                         (200, {"song": True}))

    def test_each_refusal_has_its_status(self):
        failures = ((WindowRefusedError("no"), 422), (WindowActiveError("busy"), 409),
                    (ValueError("bad"), 400), (RuntimeError("down"), 500))
        for failure, status in failures:
            with self.subTest(status=status):
                answer = call(Answering(failure), "POST", WINDOW_PATH, {"request_id": "r"})
                self.assertEqual(answer[0], status)

    def test_the_cancel(self):
        self.assertEqual(call(Answering(), "DELETE", WINDOW_PATH + "/run-1"),
                         (200, {"state": "cancelling"}))
        self.assertEqual(call(Answering(), "DELETE", WINDOW_PATH + "/nobody"),
                         (404, {"error": {"message": "no window call nobody is running"}}))
        self.assertEqual(call(Answering(), "DELETE", "/v1/elsewhere/x")[0], 404)
        self.assertEqual(call(Answering(), "DELETE", WINDOW_PATH + "/a/b")[0], 404)


if __name__ == "__main__":
    unittest.main()
