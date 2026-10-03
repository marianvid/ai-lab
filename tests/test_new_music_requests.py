import unittest
from pathlib import Path

from ai_lab.music import levo2_backend, stableaudio3_backend


class Levo2RequestTests(unittest.TestCase):
    def test_instrumental_needs_no_lyrics(self):
        request = levo2_backend.validate(
            {"prompt": "ambient piano", "instrumental": True, "duration": 30,
             "seed": 7}, "levo2-gguf")
        self.assertTrue(request["instrumental"])
        self.assertEqual(request["duration"], 30)

    def test_a_vocal_song_needs_lyrics(self):
        with self.assertRaisesRegex(ValueError, "needs lyrics"):
            levo2_backend.validate({"prompt": "pop", "instrumental": False},
                                   "levo2-gguf")

    def test_unknown_fields_are_named(self):
        with self.assertRaisesRegex(ValueError, "abc"):
            levo2_backend.validate({"prompt": "pop", "abc": "X:1"}, "levo2-gguf")


class StableAudio3RequestTests(unittest.TestCase):
    def test_defaults_to_an_instrumental(self):
        request = stableaudio3_backend.validate(
            {"prompt": "house 124 BPM", "duration": 60, "seed": 3}, "m")
        self.assertEqual(request["duration"], 60)

    def test_refuses_vocals(self):
        with self.assertRaisesRegex(ValueError, "instrumentals only"):
            stableaudio3_backend.validate({"prompt": "pop", "instrumental": False}, "m")
        with self.assertRaisesRegex(ValueError, "does not sing"):
            stableaudio3_backend.validate({"prompt": "pop", "lyrics": "la la"}, "m")

    def test_text_encoder_is_read_from_the_model_folder(self):
        config = {"model": {"conditioning": {"configs": [
            {"id": "prompt", "config": {"repo_id": "x/y", "subfolder": "enc"}},
            {"id": "seconds_total", "config": {}}]}}}
        local = stableaudio3_backend.local_config(config, Path("/models/sa3"))
        prompt = local["model"]["conditioning"]["configs"][0]["config"]
        self.assertEqual(prompt, {"model_path": "/models/sa3", "subfolder": "enc"})
