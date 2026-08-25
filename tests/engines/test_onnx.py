import unittest

from ai_lab.engines.onnx import OnnxEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


def model(format=Format.ONNX, task=Task.VAD):
    path = "/models/audio/vad/silero-vad"
    return ModelSet(id="vad/silero", name="silero-vad", format=format,
                    entrypoint=path,
                    files=(ModelFile(path + "/silero_vad.onnx", 100),), task=task)


class OnnxEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = OnnxEngine(binary="/opt/ai/silero/.venv/bin/python",
                                 server="/opt/ai-lab/ai_lab/audio/server.py")

    def test_it_serves_vad_without_reserving_the_gpu(self):
        self.assertEqual(self.engine.formats(), frozenset({Format.ONNX}))
        self.assertEqual(self.engine.tasks(), frozenset({Task.VAD}))
        self.assertEqual(self.engine.needs_mb(model(), {}, 32000), 0)
        self.assertIn("/v1/audio/speech-segments", self.engine.api_paths())

    def test_it_starts_the_silero_adapter(self):
        argv = self.engine.plan(model(), 8099, {}).argv
        self.assertEqual(argv[argv.index("--backend") + 1], "silero")
        self.assertEqual(argv[argv.index("--model") + 1], model().entrypoint)

    def test_wrong_format_task_and_settings_are_refused(self):
        with self.assertRaises(ValueError):
            self.engine.plan(model(format=Format.SAFETENSORS), 8099, {})
        with self.assertRaises(ValueError):
            self.engine.plan(model(task=Task.TRANSCRIPTION), 8099, {})
        with self.assertRaises(ValueError):
            self.engine.plan(model(), 8099, {"mystery": True})


if __name__ == "__main__":
    unittest.main()
