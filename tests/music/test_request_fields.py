"""The request checks every music engine shares."""

import base64
import io
import tempfile
import unittest
import wave
from pathlib import Path

from ai_lab.music.bundles import Checkpoint, Sampling
from ai_lab.music.request_fields import (
    TextRule,
    WavRule,
    check_fields,
    checked_wav,
    decoded_wav,
    flag,
    seconds,
    seed,
    text,
    wav_answer,
    wav_length_s,
)

RULE = WavRule(64, "missing or large", "not base64", "not a wav")


def wav_bytes(frames=8000, rate=8000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\0\0" * frames)
    return out.getvalue()


class FieldTests(unittest.TestCase):
    def test_unknown_fields_and_another_model_are_refused(self):
        check_fields({"prompt": "x", "model": "m"}, {"prompt", "model"}, "Thing", "m")
        with self.assertRaisesRegex(ValueError, "Unknown Thing fields: extra"):
            check_fields({"extra": 1}, {"prompt"}, "Thing", "m")
        with self.assertRaisesRegex(ValueError, "this engine serves m"):
            check_fields({"model": "other"}, {"model"}, "Thing", "m")

    def test_text_is_trimmed_and_bounded(self):
        rule = TextRule("bad", 5, shortest=1)
        self.assertEqual(text({"prompt": "  ab "}, "prompt", rule), "ab")
        for body in ({"prompt": ""}, {"prompt": "toolong"}, {"prompt": 3}):
            with self.assertRaisesRegex(ValueError, "bad"):
                text(body, "prompt", rule)
        self.assertEqual(text({"prompt": 42}, "prompt", TextRule("bad", 5, coerce=True)), "42")

    def test_flag_seconds_and_seed(self):
        self.assertTrue(flag({}, "instrumental", True))
        with self.assertRaisesRegex(ValueError, "true or false"):
            flag({"instrumental": "yes"}, "instrumental", False)
        self.assertEqual(seconds({"duration": 10}, 30, 5, 120), 10)
        with self.assertRaisesRegex(ValueError, "between 5 and 120 seconds"):
            seconds({"duration": 200}, 30, 5, 120)
        self.assertEqual(seed({"seed": 7}), 7)
        self.assertLess(seed({}, 10), 10)
        with self.assertRaisesRegex(ValueError, "32-bit"):
            seed({"seed": -1})


class WavTests(unittest.TestCase):
    def test_a_source_recording_is_checked(self):
        good = base64.b64encode(b"RIFF" + b"0" * 20).decode()
        self.assertTrue(decoded_wav(good, RULE).startswith(b"RIFF"))
        cases = (("", "missing or large"), ("@@@@", "not base64"),
                 (base64.b64encode(b"OggS1234").decode(), "not a wav"))
        for encoded, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                decoded_wav(encoded, RULE)

    def test_the_answer_and_the_length(self):
        audio = wav_bytes()
        answer = wav_answer(audio, seed=3)
        self.assertEqual(answer["seed"], 3)
        self.assertEqual(base64.b64decode(answer["data"][0]["b64_wav"]), audio)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.wav"
            path.write_bytes(audio)
            self.assertAlmostEqual(wav_length_s(path), 1.0)
        self.assertEqual(checked_wav(audio, "X"), audio)
        with self.assertRaisesRegex(RuntimeError, "X did not return WAV"):
            checked_wav(b"nope", "X")


class BundleTests(unittest.TestCase):
    def test_a_checkpoint_must_be_inside_its_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle = Path(folder)
            (bundle / "ckpt").mkdir()
            Checkpoint(bundle, bundle / "ckpt", "ckpt").check("E")
            with self.assertRaisesRegex(ValueError, "E bundle does not point"):
                Checkpoint(bundle, bundle / "ckpt", "../ckpt").check("E")
            with self.assertRaisesRegex(ValueError, "E bundle or checkpoint is absent"):
                Checkpoint(bundle, bundle / "none", "none").check("E")

    def test_sampling_settings_are_bounded(self):
        Sampling(50, 1.0, 1.5).check("E")
        for wrong in (Sampling(0, 1.0, 1.5), Sampling(50, 9.0, 1.5), Sampling(50, 1.0, 0)):
            with self.assertRaisesRegex(ValueError, "E inference settings are invalid"):
                wrong.check("E")
