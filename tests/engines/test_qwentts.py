import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.engines.qwentts import QwenTtsEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


class QwenTtsEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = QwenTtsEngine(binary='/tts/python', model_modes={
            'qwen3-tts-1.7b-voicedesign': 'voice-design'})
        self.model = ModelSet(
            id='tts/qwen3-tts-1.7b-voicedesign',
            name='qwen3-tts-1.7b-voicedesign',
            format=Format.SAFETENSORS, task=Task.SPEECH_SYNTHESIS,
            entrypoint='/models/tts/qwen3-tts-1.7b-voicedesign',
            files=(ModelFile('model.safetensors', 1024),))

    def test_plan_uses_isolated_runtime_and_model(self):
        plan = self.engine.plan(self.model, 8113, {})
        self.assertEqual(plan.argv[0], '/tts/python')
        self.assertEqual(plan.argv[plan.argv.index('--mode') + 1], 'voice-design')
        self.assertEqual(plan.argv[plan.argv.index('--model-path') + 1],
                         self.model.entrypoint)
        self.assertEqual(self.engine.api_paths(),
                         ('/v1/audio/speech/generations',))
        self.assertEqual(self.engine.concurrency({}), 1)

    def test_unmapped_models_are_rejected(self):
        self.engine.model_modes.clear()
        self.assertFalse(self.engine.supports(self.model))
        with self.assertRaises(ValueError):
            self.engine.plan(self.model, 8113, {})

    def test_memory_estimate_includes_runtime_headroom(self):
        model = replace(self.model, files=(ModelFile(
            'model.safetensors', 4096 * 1024 * 1024),))
        self.assertEqual(self.engine.needs_mb(model, {}, 32768), 6144)

    def test_manifest_entrypoint_uses_checkpoint_directory(self):
        with TemporaryDirectory() as root:
            checkpoint = Path(root) / 'model.safetensors'
            checkpoint.touch()
            model = replace(self.model, entrypoint=str(checkpoint))
            plan = self.engine.plan(model, 8113, {})
            self.assertEqual(plan.argv[plan.argv.index('--model-path') + 1],
                             root)


if __name__ == '__main__':
    unittest.main()
