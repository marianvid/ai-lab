import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.engines.voxcpm import VoxCpmEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


class VoxCpmEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = VoxCpmEngine(binary='/runtime/python', model_options={
            'future-checkpoint': {'cfg_value': 2.4, 'inference_timesteps': 12}})
        self.model = ModelSet(
            id='tts/future-checkpoint', name='future-checkpoint',
            format=Format.SAFETENSORS, task=Task.SPEECH_SYNTHESIS,
            entrypoint='/models/future-checkpoint/model.safetensors',
            files=(ModelFile('model.safetensors', 4096 * 1024 * 1024),))

    def test_plan_uses_configured_checkpoint_and_settings(self):
        with TemporaryDirectory() as root:
            checkpoint = Path(root) / 'model.safetensors'
            checkpoint.touch()
            model = replace(self.model, entrypoint=str(checkpoint))
            plan = self.engine.plan(model, 8118, {})
            self.assertEqual(plan.argv[plan.argv.index('--model-path') + 1], root)
        self.assertEqual(plan.argv[plan.argv.index('--cfg-value') + 1], '2.4')
        self.assertEqual(plan.argv[plan.argv.index('--inference-timesteps') + 1],
                         '12')
        self.assertEqual(self.engine.api_paths(),
                         ('/v1/audio/speech/generations',))
        self.assertEqual(self.engine.needs_mb(self.model, {}, 32768), 6144)

    def test_unmapped_checkpoint_is_rejected(self):
        self.engine.model_options.clear()
        self.assertFalse(self.engine.supports(self.model))
        with self.assertRaises(ValueError):
            self.engine.plan(self.model, 8118, {})


if __name__ == '__main__':
    unittest.main()
