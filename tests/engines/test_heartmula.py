import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.engines.heartmula import HeartMulaEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


class HeartMulaEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = HeartMulaEngine(
            binary='/heartlib/python', bundle_root='/runtime/bundle',
            output_root='/tmp/music', heartmuse_root='/opt/heartmuse',
            model_options={
                'future-checkpoint': {'version': '3B',
                                      'checkpoint_subdir': 'HeartMuLa-oss-3B', 'topk': 50,
                                      'temperature': 1.0, 'cfg_scale': 1.5,
                                      'memory_reservation_mb': 18000}})
        self.model = ModelSet(
            id='music/future-checkpoint', name='future-checkpoint',
            format=Format.SAFETENSORS, task=Task.MUSIC_GENERATION,
            entrypoint='/models/future-checkpoint',
            files=(ModelFile('model.safetensors', 1024),))

    def test_plan_uses_configured_checkpoint_and_generation_values(self):
        with TemporaryDirectory() as root:
            checkpoint = Path(root) / 'model.safetensors'
            checkpoint.touch()
            plan = self.engine.plan(replace(self.model,
                                            entrypoint=str(checkpoint)), 8123, {})
            self.assertEqual(plan.argv[plan.argv.index('--checkpoint-path') + 1], root)
        self.assertEqual(plan.argv[plan.argv.index('--bundle-root') + 1],
                         '/runtime/bundle')
        self.assertEqual(plan.argv[plan.argv.index('--version') + 1], '3B')
        self.assertEqual(plan.argv[plan.argv.index('--ui-port') + 1], '18123')
        self.assertEqual(plan.argv[plan.argv.index('--heartmuse-root') + 1],
                         '/opt/heartmuse')
        self.assertEqual(self.engine.needs_mb(self.model, {}, 32623), 18000)
        self.assertEqual(self.engine.music_form('future-checkpoint'),
                         {'lyrics_required': True})

    def test_wrong_task_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.plan(replace(self.model, task=Task.TRANSCRIPTION), 8123, {})


if __name__ == '__main__':
    unittest.main()
