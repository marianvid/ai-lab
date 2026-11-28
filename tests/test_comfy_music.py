"""ComfyUI music built by its real constructor, with ComfyUI and ffmpeg replaced.

The private ComfyUI is a stand-in that answers the three calls a song makes
(`/prompt`, `/history/<id>`, `/view`); ffmpeg is a small script that writes a
WAV. What the workflow itself makes of a caption cannot be tested without
ComfyUI and its models.
"""

import base64
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest import mock

from ai_lab.music.backends import comfy_backend
from ai_lab.music.backends.comfy_backend import ComfyMusicBackend, ComfyRuntime


WORKFLOW = {"prompt": {"1": {"inputs": {
    "caption": "__AI_LAB_CAPTION__", "lyrics": "__AI_LAB_LYRICS__",
    "seconds": "__AI_LAB_DURATION__", "seed": "__AI_LAB_SEED__", "prefix": "__AI_LAB_OUTPUT__"}}}}
FFMPEG = """import sys, wave
sys.stdin.buffer.read()
with wave.open(sys.argv[-1], "wb") as audio:
    audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(8000)
    audio.writeframes(b"\\0\\0" * 12000)
"""


class Comfy:
    """The private ComfyUI: a prompt is queued, finished on the second look, and its FLAC served."""

    def __init__(self, launch):
        self.launch = launch
        launch.state.mkdir(parents=True, exist_ok=True)     # as the real bridge does
        self.asked: list[tuple] = []
        self.status = "success"
        self.outputs = [{"filename": "song.flac", "subfolder": "ai_lab_music", "type": "output"}]
        self.looks = 0
        self.interrupted = False

    def request(self, method, path, payload=None):
        self.asked.append((method, path, payload))
        if path == "/prompt":
            return {"prompt_id": "p1"}
        if path.startswith("/history/"):
            self.looks += 1
            if self.looks < 2 or self.status == "never":
                return {}
            return {"p1": {"status": {"status_str": self.status},
                           "outputs": {"9": {"audio": self.outputs}}}}
        if path.startswith("/view?"):
            return b"fLaC stand-in"
        raise AssertionError(path)

    def interrupt(self):
        self.interrupted = True

    def close(self):
        pass


def comfy(backend: ComfyMusicBackend) -> Comfy:
    """The stand-in ComfyUI the backend was built with."""
    return cast(Comfy, backend.comfy)


class ComfyMusicTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "models").mkdir()
        self.workflow = self.root / "minimax.json"
        self.workflow.write_text(json.dumps(WORKFLOW))
        ffmpeg = self.root / "ffmpeg"
        ffmpeg.write_text(f"#!{sys.executable}\n{FFMPEG}")
        ffmpeg.chmod(0o755)
        self.ffmpeg = str(ffmpeg)
        patcher = mock.patch.object(comfy_backend, "ComfyBackend", Comfy)
        patcher.start()
        self.addCleanup(patcher.stop)
        timing = mock.patch.multiple(comfy_backend, POLL_S=0.001)
        timing.start()
        self.addCleanup(timing.stop)

    def backend(self, workflow=None, models=None) -> ComfyMusicBackend:
        runtime = ComfyRuntime(python="python", comfyui="main.py",
                               model_paths=[str(models or self.root / "models")],
                               state=self.root / "state", port=8188)
        return ComfyMusicBackend(runtime, workflow or self.workflow, "minimax-music3", self.ffmpeg)

    def test_a_song_fills_every_marker_and_comes_back_as_a_wav(self):
        backend = self.backend()
        saved = self.root / "state" / "output" / "ai_lab_music" / "song.flac"
        saved.parent.mkdir(parents=True)
        saved.write_bytes(b"ComfyUI's own copy")
        answer = backend.generate({"model": "minimax-music3", "prompt": "soul piano",
                                   "lyrics": "[Verse] Hello", "instrumental": False,
                                   "duration": 20, "seed": 42})
        _, _, queued = comfy(backend).asked[0]
        inputs = queued["prompt"]["1"]["inputs"]
        self.assertEqual((inputs["caption"], inputs["lyrics"], inputs["seconds"], inputs["seed"]),
                         ("soul piano", "[Verse] Hello", 20, 42))
        self.assertTrue(inputs["prefix"].startswith("ai_lab_music/"))
        self.assertEqual(answer["seed"], 42)
        self.assertEqual(answer["duration"], 1.5)
        self.assertTrue(base64.b64decode(answer["data"][0]["b64_wav"]).startswith(b"RIFF"))
        self.assertFalse(saved.exists(), "ComfyUI's copy of the song was left behind")
        self.assertEqual(comfy(backend).launch.port, 8188)

    def test_an_instrumental_song_is_sent_as_such(self):
        backend = self.backend()
        backend.generate({"prompt": "ambient pulse", "instrumental": True, "duration": 12, "seed": 43})
        self.assertEqual(comfy(backend).asked[0][2]["prompt"]["1"]["inputs"]["lyrics"], "[Instrumental]")

    def test_a_workflow_without_its_markers_or_models_is_refused_at_start(self):
        broken = self.root / "broken.json"
        broken.write_text(json.dumps({"prompt": {"1": {"inputs": {"caption": "x"}}}}))
        with self.assertRaisesRegex(ValueError, "missing input markers"):
            self.backend(workflow=broken)
        broken.write_text(json.dumps({"no": "prompt"}))
        with self.assertRaisesRegex(ValueError, "no prompt graph"):
            self.backend(workflow=broken)
        with self.assertRaisesRegex(ValueError, "absent"):
            self.backend(models=self.root / "none")

    def test_a_failed_workflow_or_two_outputs_is_an_error(self):
        backend = self.backend()
        song = {"prompt": "soul", "instrumental": True, "seed": 1}
        comfy(backend).status = "error"
        with self.assertRaisesRegex(RuntimeError, "workflow failed"):
            backend.generate(song)
        comfy(backend).status, comfy(backend).looks = "success", 0
        comfy(backend).outputs = comfy(backend).outputs * 2
        with self.assertRaisesRegex(RuntimeError, "one audio file"):
            backend.generate(song)

    def test_a_song_past_its_limit_is_interrupted(self):
        backend = self.backend()
        comfy(backend).status = "never"
        with mock.patch.object(comfy_backend, "SONG_LIMIT_S", 0.01), \
                self.assertRaisesRegex(TimeoutError, "timed out"):
            backend.generate({"prompt": "soul", "instrumental": True, "seed": 1})
        self.assertTrue(comfy(backend).interrupted)

    def test_a_bad_request_never_reaches_comfyui(self):
        backend = self.backend()
        for body, refusal in (({"model": "other", "prompt": "soul", "instrumental": True}, "serves"),
                              ({"prompt": "soul", "instrumental": False}, "lyrics are required"),
                              ({"prompt": "soul", "instrumental": "yes"}, "true or false")):
            with self.subTest(refusal), self.assertRaisesRegex(ValueError, refusal):
                backend.generate(body)
        self.assertEqual(comfy(backend).asked, [])


if __name__ == "__main__":
    unittest.main()
