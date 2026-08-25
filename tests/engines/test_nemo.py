import unittest

from ai_lab.engines.nemo import NemoEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


def model(format=Format.NEMO, task=Task.TRANSCRIPTION):
    path = "/models/audio/asr/nemo/parakeet"
    return ModelSet(id="audio/parakeet", name="parakeet", format=format,
                    entrypoint=path,
                    files=(ModelFile(path + "/parakeet.nemo", 100),), task=task)


class NemoEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = NemoEngine(binary="/opt/ai/nemo/.venv/bin/python",
                                 server="/opt/ai-lab/ai_lab/audio/server.py")

    def test_it_is_an_audio_engine_for_nemo_checkpoints(self):
        self.assertEqual(self.engine.formats(), frozenset({Format.NEMO}))
        self.assertEqual(self.engine.tasks(),
                         frozenset({Task.TRANSCRIPTION, Task.DIARIZATION}))
        self.assertIn("/v1/audio/transcriptions", self.engine.api_paths())
        self.assertIn("/v1/audio/diarizations",
                      self.engine.api_paths(Task.DIARIZATION))

    def test_the_isolated_python_and_adapter_are_started(self):
        argv = self.engine.plan(model(), 8096, {}).argv
        self.assertEqual(argv[:2], ["/opt/ai/nemo/.venv/bin/python",
                                    "/opt/ai-lab/ai_lab/audio/server.py"])
        self.assertEqual(argv[argv.index("--model") + 1], model().entrypoint)
        self.assertEqual(argv[argv.index("--precision") + 1], "bf16")

    def test_wrong_format_and_task_are_refused(self):
        with self.assertRaises(ValueError):
            self.engine.plan(model(format=Format.SAFETENSORS), 8096, {})
        with self.assertRaises(ValueError):
            self.engine.plan(model(task=Task.VAD), 8096, {})

    def test_sortformer_uses_the_diarization_backend(self):
        argv = self.engine.plan(model(task=Task.DIARIZATION), 8100, {}).argv
        self.assertEqual(argv[argv.index("--backend") + 1], "sortformer")

    def test_fp32_accounts_for_twice_the_weight_memory(self):
        normal = self.engine.needs_mb(model(), {"precision": "bf16"}, 32000)
        fp32 = self.engine.needs_mb(model(), {"precision": "fp32"}, 32000)
        self.assertEqual(fp32, normal * 2)


if __name__ == "__main__":
    unittest.main()
