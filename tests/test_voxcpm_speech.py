"""VoxCPM built by its real constructor, with the model replaced (tests/support/stand_ins.py)."""

import base64
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai_lab.speech.voxcpm_backend import VoxCpmBackend

from .support import stand_ins


class VoxCPM:
    """The model: records each generation and the file it was given to clone from."""

    built: dict = {}
    tts_model = SimpleNamespace(sample_rate=48000)

    def __init__(self, **built):
        self.built = built
        self.asked: list[dict] = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls(path=path, **kwargs)

    def generate(self, **kwargs):
        reference = kwargs.get("reference_wav_path")
        self.asked.append({**kwargs, "reference_bytes": Path(reference).read_bytes() if reference else None})
        return [0.0, 0.1, -0.1]


class VoxCpmSpeechTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "voxcpm-2").mkdir()
        self.torch = stand_ins.torch()
        modules = stand_ins.installed(torch=self.torch, soundfile=stand_ins.soundfile(),
                                      voxcpm=stand_ins.package("voxcpm", VoxCPM=VoxCPM))
        modules.start()
        self.addCleanup(modules.stop)
        self.backend = VoxCpmBackend(self.root / "voxcpm-2", 2.4, 12)

    def test_it_is_built_offline_from_the_folder(self):
        built = self.backend.model.built
        self.assertEqual(built["path"], str(self.root / "voxcpm-2"))
        self.assertEqual((built["local_files_only"], built["load_denoiser"], built["optimize"]),
                         (True, False, False))

    def test_an_absent_folder_or_settings_out_of_range_are_refused(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            VoxCpmBackend(self.root / "none", 2.4, 12)
        for cfg, steps in ((0, 12), (11, 12), (2.4, 0), (2.4, 101)):
            with self.subTest(cfg=cfg, steps=steps), self.assertRaisesRegex(ValueError, "out of range"):
                VoxCpmBackend(self.root / "voxcpm-2", cfg, steps)

    def test_voice_design_puts_the_instruction_before_the_text(self):
        answer = self.backend.generate({"model": "voxcpm-2", "text": "Hello",
                                        "instruction": "Warm, relaxed delivery", "seed": 7})
        asked, = self.backend.model.asked
        self.assertEqual(asked["text"], "(Warm, relaxed delivery)Hello")
        self.assertEqual((asked["cfg_value"], asked["inference_timesteps"]), (2.4, 12))
        self.assertIsNone(asked["reference_bytes"])
        self.assertEqual(self.torch.seeds, [7])
        self.assertEqual((answer["mode"], answer["sample_rate"]), ("voxcpm", 48000))

    def test_a_reference_clip_is_cloned_from_and_its_file_removed(self):
        clip = stand_ins.wav_bytes(0.5)
        self.backend.generate({"text": "Hello", "reference_audio": base64.b64encode(clip).decode(),
                               "reference_text": "Hi there"})
        asked, = self.backend.model.asked
        self.assertEqual(asked["reference_bytes"], clip)
        self.assertEqual((asked["prompt_wav_path"], asked["prompt_text"]),
                         (asked["reference_wav_path"], "Hi there"))
        self.assertFalse(Path(asked["reference_wav_path"]).exists(), "the clip was left on disk")

    def test_a_named_voice_or_another_model_is_refused_before_the_model(self):
        for body, refusal in (({"text": "Hello", "speaker": "unknown"}, "reference audio"),
                              ({"text": "Hello", "model": "other"}, "serves voxcpm-2")):
            with self.subTest(refusal), self.assertRaisesRegex(ValueError, refusal):
                self.backend.generate(body)
        self.assertEqual(self.backend.model.asked, [])


if __name__ == "__main__":
    unittest.main()
