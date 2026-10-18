"""MLX Whisper: the audio server's mlx-whisper backend, for MLX checkpoints."""

import unittest

from ai_lab.engines.mlxwhisper import MlxWhisperEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


def model(fmt=Format.SAFETENSORS, task=Task.TRANSCRIPTION):
    return ModelSet(id="audio/whisper", name="whisper", format=fmt, task=task,
                    entrypoint="/models/whisper",
                    files=(ModelFile("weights.safetensors", 3 * 1024 * 1024),))


class MlxWhisperTests(unittest.TestCase):
    def setUp(self):
        self.engine = MlxWhisperEngine(binary="/env/python", server="/ai_lab/audio/server.py")

    def test_the_plan_starts_the_audio_server_with_this_backend(self):
        plan = self.engine.plan(model(), 8301, {})
        self.assertEqual(plan.argv[:4], ["/env/python", "/ai_lab/audio/server.py",
                                         "--backend", "mlx-whisper"])
        self.assertEqual(plan.argv[plan.argv.index("--port") + 1], "8301")

    def test_the_wrong_kind_of_model_is_refused(self):
        with self.assertRaisesRegex(ValueError, "cannot load gguf"):
            self.engine.plan(model(fmt=Format.GGUF), 8301, {})
        with self.assertRaisesRegex(ValueError, "cannot perform"):
            self.engine.plan(model(task=Task.OCR), 8301, {})
        with self.assertRaisesRegex(ValueError, "Unknown settings"):
            self.engine.plan(model(), 8301, {"beam": 5})

    def test_it_answers_transcription_one_at_a_time(self):
        self.assertEqual(self.engine.concurrency({}), 1)
        self.assertEqual(self.engine.needs_mb(model(), {}, 0.0), 3.0)
        self.assertIn("/v1/audio/transcriptions", self.engine.api_paths(Task.TRANSCRIPTION))
        self.assertEqual(self.engine.api_paths(Task.OCR), ())
