"""MuLaCover built by its real constructor, with the model replaced (tests/support/stand_ins.py)."""

import base64
import shutil
import tempfile
import unittest
from pathlib import Path

from ai_lab.music.backends.mulacover_backend import MAX_SOURCE_BYTES, MulaCoverBackend
from ai_lab.music.bundles import Checkpoint, Sampling

from .support import stand_ins


class MuLaCover(stand_ins.Pipeline):
    """MuLaCoverGenPipeline: a cover of a reference recording."""


SAMPLING = Sampling(topk=50, temperature=1.0, cfg_scale=1.5)


class MulaCoverTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "bundle" / "cover").mkdir(parents=True)
        self.torch = stand_ins.torch(cuda=True)
        modules = stand_ins.installed(torch=self.torch,
                                      mulacover=stand_ins.package("mulacover", MuLaCoverGenPipeline=MuLaCover))
        modules.start()
        self.addCleanup(modules.stop)
        self.reference = stand_ins.wav_bytes(1.0)
        self.song = {"model": "cover", "prompt": "genre:[soul]", "lyrics": "[Verse]\nHello",
                     "instrumental": False, "seed": 42,
                     "reference_audio_base64": base64.b64encode(self.reference).decode()}

    def backend(self, sampling=SAMPLING) -> MulaCoverBackend:
        checkpoint = Checkpoint(self.root / "bundle", self.root / "bundle" / "cover", "cover")
        return MulaCoverBackend(checkpoint, self.root / "out", sampling)

    def test_it_is_built_on_the_card_with_its_four_parts(self):
        built = self.backend().pipeline.built
        self.assertEqual(built["device"].name, "cuda:0")
        self.assertEqual(built["dtype"], {"mulacover": "bfloat16", "codec": "float32",
                                          "qwen": "float32", "transcriptor": "float32"})

    def test_a_cover_is_made_from_the_reference_with_the_seed_on_both_generators(self):
        backend = self.backend()
        answer = backend.generate(self.song)
        (inputs, asked), = backend.pipeline.calls
        self.assertEqual(inputs["ref_audio"], self.reference)
        self.assertEqual((inputs["lyrics"], inputs["tags"]), ("[Verse]\nHello\n", "genre:[soul]\n"))
        self.assertEqual(self.torch.seeds, [42, 42])
        self.assertEqual((answer["seed"], answer["duration"]), (42, 2.0))
        self.assertEqual(asked["topk"], 50)

    def test_the_reference_must_be_a_wav_of_bounded_size(self):
        backend = self.backend()
        cases = (({k: v for k, v in self.song.items() if k != "reference_audio_base64"}, "reference WAV"),
                 ({**self.song, "reference_audio_base64": "!!"}, "not valid base64"),
                 ({**self.song, "reference_audio_base64": base64.b64encode(b"not wav").decode()}, "must be a WAV"),
                 ({**self.song, "reference_audio_base64": "A" * (MAX_SOURCE_BYTES * 2)}, "too large"),
                 ({**self.song, "instrumental": True}, "vocal arrangement"))
        for body, refusal in cases:
            with self.subTest(refusal), self.assertRaisesRegex(ValueError, refusal):
                backend.generate(body)
        self.assertEqual(backend.pipeline.calls, [])

    def test_sampling_outside_its_range_is_refused(self):
        with self.assertRaisesRegex(ValueError, "settings are invalid"):
            self.backend(Sampling(topk=50, temperature=9.0, cfg_scale=1.5))


if __name__ == "__main__":
    unittest.main()
