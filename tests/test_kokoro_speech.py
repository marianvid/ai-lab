"""Kokoro built by its real constructor, with the model replaced (tests/support/stand_ins.py)."""

import shutil
import tempfile
import unittest
from pathlib import Path

from ai_lab.speech.kokoro_backend import SAMPLE_RATE, KokoroBackend

from .support import stand_ins


class KModel:
    def __init__(self, **built):
        self.built = built

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


class KPipeline:
    """Says each line in chunks of samples, recording what it was asked."""

    def __init__(self, **built):
        self.built = built
        self.asked = []
        self.chunks = [stand_ins.Tensor([0.0, 0.5]), stand_ins.Tensor([-0.5])]

    def __call__(self, text, voice, speed):
        self.asked.append((text, voice, speed))
        return [type("Result", (), {"audio": chunk}) for chunk in self.chunks]


class KokoroTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        folder = self.root / "kokoro-82m"
        (folder / "voices").mkdir(parents=True)
        (folder / "kokoro.pth").write_bytes(b"weights")
        (folder / "voices" / "bm_george.pt").write_bytes(b"voice")
        self.checkpoint = folder / "kokoro.pth"
        modules = stand_ins.installed(
            torch=stand_ins.torch(), soundfile=stand_ins.soundfile(),
            kokoro=stand_ins.package("kokoro", KModel=KModel, KPipeline=KPipeline))
        modules.start()
        self.addCleanup(modules.stop)
        self.backend = KokoroBackend(self.checkpoint, "b", "bm_george", "hexgrad/Kokoro-82M")

    def test_it_is_built_from_the_checkpoint_beside_its_config(self):
        self.assertEqual(self.backend.model_name, "kokoro-82m")
        self.assertEqual(self.backend.device, "cpu")
        self.assertEqual(self.backend.model.built["config"], str(self.checkpoint.parent / "config.json"))
        self.assertEqual(self.backend.pipeline.built["lang_code"], "b")

    def test_an_absent_checkpoint_is_refused(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            KokoroBackend(self.root / "none.pth", "b", "bm_george", "hexgrad/Kokoro-82M")

    def test_a_line_is_said_in_the_default_voice_and_answered_as_one_wav(self):
        answer = self.backend.generate({"text": "Hello"})
        self.assertEqual(self.backend.pipeline.asked,
                         [("Hello", str(self.checkpoint.parent / "voices" / "bm_george.pt"), 1.0)])
        self.assertEqual((answer["model"], answer["mode"], answer["sample_rate"]),
                         ("kokoro-82m", "kokoro", SAMPLE_RATE))

    def test_what_kokoro_cannot_do_is_refused_before_the_model(self):
        for body, refusal in (({"text": "Hello", "instruction": "Whisper"}, "does not support"),
                              ({"text": "Hello", "speaker": "../other"}, "voice must be a name"),
                              ({"text": "Hello", "speaker": "missing"}, "Unknown voice"),
                              ({"text": "Hello", "language": "German"}, "configured for English"),
                              ({"text": "Hello", "model": "another"}, "serves kokoro-82m")):
            with self.subTest(refusal), self.assertRaisesRegex(ValueError, refusal):
                self.backend.generate(body)
        self.assertEqual(self.backend.pipeline.asked, [])

    def test_no_audio_at_all_is_an_error(self):
        self.backend.pipeline.chunks = []
        with self.assertRaisesRegex(RuntimeError, "no audio"):
            self.backend.generate({"text": "Hello"})


if __name__ == "__main__":
    unittest.main()
