import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.engines.yue2 import Yue2Engine
from ai_lab.types import Format, ModelFile, ModelSet, Task


class Yue2EngineTests(unittest.TestCase):
    def test_plan_uses_configured_paths_and_generation_mode(self):
        engine = Yue2Engine(binary='/yue2/python', vae_path='/models/vae',
                            output_root='/tmp/music', model_options={
            'next-checkpoint': {'cot': 'full', 'memory_budget_gib': 24,
                                'memory_reservation_mb': 28000}})
        with TemporaryDirectory() as root:
            checkpoint = Path(root) / 'model.safetensors'
            checkpoint.touch()
            model = ModelSet(id='music/next-checkpoint', name='next-checkpoint',
                             format=Format.SAFETENSORS,
                             task=Task.MUSIC_GENERATION,
                             entrypoint=str(checkpoint),
                             files=(ModelFile('model.safetensors', 1024),))
            plan = engine.plan(model, 8124, {})
            self.assertEqual(plan.argv[plan.argv.index('--model-path') + 1], root)
            self.assertEqual(plan.argv[plan.argv.index('--vae-path') + 1], '/models/vae')
            self.assertEqual(plan.argv[plan.argv.index('--cot') + 1], 'full')
            self.assertEqual(engine.needs_mb(model, {}, 32623), 28000)
            self.assertTrue(engine.music_form(model.name)['editable_score'])


if __name__ == '__main__':
    unittest.main()
