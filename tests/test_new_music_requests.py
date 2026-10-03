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


class StableAudio3SourceAudioTests(unittest.TestCase):
    def wav(self):
        import base64, io, struct, wave
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out:
            out.setnchannels(1); out.setsampwidth(2); out.setframerate(8000)
            out.writeframes(struct.pack("<4h", 0, 1, 2, 3))
        return base64.b64encode(buffer.getvalue()).decode()

    def test_init_audio_with_noise_level(self):
        request = stableaudio3_backend.validate(
            {"prompt": "jazz", "init_audio": self.wav(), "init_noise_level": 0.3}, "m")
        self.assertTrue(request["init_audio"].startswith(b"RIFF"))
        self.assertEqual(request["init_noise_level"], 0.3)

    def test_noise_level_needs_audio_and_range(self):
        with self.assertRaisesRegex(ValueError, "needs init_audio"):
            stableaudio3_backend.validate({"prompt": "jazz", "init_noise_level": 0.3}, "m")
        with self.assertRaisesRegex(ValueError, "0 to 1"):
            stableaudio3_backend.validate(
                {"prompt": "jazz", "init_audio": self.wav(), "init_noise_level": 2}, "m")

    def test_inpaint_regions(self):
        request = stableaudio3_backend.validate(
            {"prompt": "jazz", "duration": 30, "inpaint_audio": self.wav(),
             "inpaint_mask_start_seconds": [4, 20], "inpaint_mask_end_seconds": [8, 25]}, "m")
        self.assertEqual(request["inpaint_starts"], [4.0, 20.0])
        with self.assertRaisesRegex(ValueError, "same length"):
            stableaudio3_backend.validate(
                {"prompt": "jazz", "duration": 30, "inpaint_audio": self.wav(),
                 "inpaint_mask_start_seconds": [4, 20], "inpaint_mask_end_seconds": [8]}, "m")

    def test_not_a_wav_is_refused(self):
        import base64
        with self.assertRaisesRegex(ValueError, "WAV"):
            stableaudio3_backend.validate(
                {"prompt": "jazz", "init_audio": base64.b64encode(b"MP3!").decode()}, "m")
