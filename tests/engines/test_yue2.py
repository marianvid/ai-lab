import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.engines.yue2 import Yue2Engine
from ai_lab.types import Format, ModelFile, ModelSet, Task


class Yue2EngineTests(unittest.TestCase):
    def test_plan_uses_configured_paths_and_generation_mode(self):
        engine = Yue2Engine(binary='/yue2/python', vae_path='/models/vae',
                            output_root='/tmp/music', webui_root='/opt/yue-studio',
                            model_options={
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
            self.assertEqual(plan.argv[plan.argv.index('--ui-port') + 1], '18124')
            self.assertEqual(plan.argv[plan.argv.index('--webui-root') + 1],
                             '/opt/yue-studio')
            self.assertEqual(engine.needs_mb(model, {}, 32623), 28000)
            self.assertTrue(engine.music_form(model.name)['editable_score'])
            self.assertEqual(plan.argv[plan.argv.index('--device') + 1], 'cuda')
            metal = Yue2Engine(binary='/yue2/python', vae_path='/models/vae',
                               output_root='/tmp/music', webui_root='/opt/yue-studio',
                               model_options=engine.model_options, device='mps')
            plan = metal.plan(model, 8124, {})
            self.assertEqual(plan.argv[plan.argv.index('--device') + 1], 'mps')

    def test_unknown_accelerator_is_refused(self):
        for bad in ('cpu', 'metal', ''):
            with self.assertRaises(ValueError):
                Yue2Engine(device=bad)

    def test_studio_uses_metal_fallback_only_on_mac(self):
        import os
        from unittest.mock import patch
        from ai_lab.music.yue2_server import studio_env
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(studio_env('mps'), {
                'HF_HUB_OFFLINE': '1', 'PYTORCH_ENABLE_MPS_FALLBACK': '1'})
            self.assertEqual(studio_env('cuda'), {'HF_HUB_OFFLINE': '1'})


if __name__ == '__main__':
    unittest.main()
