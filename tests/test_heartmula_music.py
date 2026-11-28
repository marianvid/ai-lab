"""HeartMuLa built by its real constructor, with the model replaced (tests/support/stand_ins.py)."""

import base64
import shutil
import tempfile
import unittest
from pathlib import Path

from ai_lab.music.backends.heartmula_backend import HeartMulaBackend
from ai_lab.music.bundles import Checkpoint, Sampling

from .support import stand_ins


class HeartMula(stand_ins.Pipeline):
    """HeartMuLaGenPipeline: one song from lyrics and tags."""


def bundle(root: Path) -> Checkpoint:
    (root / "bundle" / "heartmula-3b").mkdir(parents=True, exist_ok=True)
    return Checkpoint(root / "bundle", root / "bundle" / "heartmula-3b", "heartmula-3b")


SAMPLING = Sampling(topk=50, temperature=1.0, cfg_scale=1.5)
SONG = {"model": "heartmula-3b", "prompt": "jazz,piano", "lyrics": "Hello from AI Lab",
        "instrumental": False, "duration": 30, "seed": 42}


class HeartMulaTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.torch = stand_ins.torch()
        modules = stand_ins.installed(torch=self.torch,
                                      heartlib=stand_ins.package("heartlib", HeartMuLaGenPipeline=HeartMula))
        modules.start()
        self.addCleanup(modules.stop)

    def backend(self, **changes) -> HeartMulaBackend:
        made = {"checkpoint": bundle(self.root), "output_root": self.root / "out",
                "version": "3B", "sampling": SAMPLING, **changes}
        return HeartMulaBackend(**made)

    def test_it_is_built_on_the_cpu_when_there_is_no_card(self):
        backend = self.backend()
        self.assertEqual(backend.model_name, "heartmula-3b")
        self.assertTrue((self.root / "out").is_dir())
        built = backend.pipeline.built
        self.assertEqual(built["root"], str(self.root / "bundle"))
        self.assertEqual(built["device"]["mula"].type, "cpu")
        self.assertEqual(built["dtype"], {"mula": "float32", "codec": "float32"})
        self.assertEqual((built["version"], built["lazy_load"]), ("3B", True))

    def test_a_bad_bundle_version_or_sampling_is_refused_before_loading(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            self.backend(checkpoint=Checkpoint(self.root / "no", self.root / "no" / "x", "x"))
        with self.assertRaisesRegex(ValueError, "settings are invalid"):
            self.backend(version="")
        with self.assertRaisesRegex(ValueError, "settings are invalid"):
            self.backend(sampling=Sampling(topk=0, temperature=1.0, cfg_scale=1.5))

    def test_a_song_is_made_from_the_lyrics_and_tags_with_the_seed(self):
        backend = self.backend()
        answer = backend.generate(SONG)
        (inputs, asked), = backend.pipeline.calls
        self.assertEqual(inputs, {"lyrics": "Hello from AI Lab\n", "tags": "jazz,piano\n"})
        self.assertEqual(asked["max_audio_length_ms"], 30000)
        self.assertEqual((asked["topk"], asked["temperature"], asked["cfg_scale"]), (50, 1.0, 1.5))
        self.assertEqual(self.torch.seeds, [42])
        self.assertEqual((answer["model"], answer["seed"], answer["duration"]), ("heartmula-3b", 42, 2.0))
        self.assertTrue(base64.b64decode(answer["data"][0]["b64_wav"]).startswith(b"RIFF"))
        self.assertEqual(list((self.root / "out").iterdir()), [], "its working folder was left behind")

    def test_lyrics_and_style_are_required_and_it_is_never_instrumental(self):
        backend = self.backend()
        for body, refusal in (({**SONG, "lyrics": ""}, "style tags and lyrics"),
                              ({**SONG, "prompt": ""}, "style tags and lyrics"),
                              ({**SONG, "instrumental": True}, "lyric-conditioned"),
                              ({**SONG, "speed": 2}, "speed")):
            with self.subTest(refusal), self.assertRaisesRegex(ValueError, refusal):
                backend.generate(body)
        self.assertEqual(backend.pipeline.calls, [], "a refused request reached the model")

    def test_an_answer_that_is_not_a_wav_is_refused(self):
        backend = self.backend()

        def writes_text(inputs, **kwargs):
            Path(kwargs["save_path"]).write_bytes(b"not audio")
        backend.pipeline = writes_text
        with self.assertRaisesRegex(RuntimeError, "did not return WAV"):
            backend.generate(SONG)


if __name__ == "__main__":
    unittest.main()
