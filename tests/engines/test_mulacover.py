import unittest

from ai_lab.engines.mulacover import MulaCoverEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


class MulaCoverEngineTests(unittest.TestCase):
    def test_configured_bundle_and_reference_form(self):
        engine = MulaCoverEngine(binary='/cover/python', bundle_root='/cover/ckpt',
                                 output_root='/tmp/music', model_options={
            'future-cover': {'checkpoint_subdir': 'MuLaCover', 'topk': 250,
                             'temperature': 1.0, 'cfg_scale': 1.5,
                             'memory_reservation_mb': 18000}})
        model = ModelSet(id='music/future-cover', name='future-cover',
                         format=Format.SAFETENSORS,
                         task=Task.MUSIC_GENERATION,
                         entrypoint='/models/future-cover',
                         files=(ModelFile('model.safetensors', 1024),))
        plan = engine.plan(model, 8126, {})
        self.assertEqual(plan.argv[plan.argv.index('--bundle-root') + 1],
                         '/cover/ckpt')
        self.assertEqual(plan.argv[plan.argv.index('--checkpoint-path') + 1],
                         '/models/future-cover')
        self.assertEqual(engine.needs_mb(model, {}, 32623), 18000)
        self.assertTrue(engine.music_form(model.name)['reference_audio_required'])


if __name__ == '__main__':
    unittest.main()
