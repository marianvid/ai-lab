import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.engines.khala import KhalaEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


class KhalaEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = KhalaEngine(
            binary='/khala/python', generator_script='/runtime/generate.py',
            output_root='/tmp/khala', model_options={
                'future-checkpoint': {'default_bucket': 0,
                                      'maximum_bucket': 4}})
        self.model = ModelSet(
            id='music/future-checkpoint', name='future-checkpoint',
            format=Format.SAFETENSORS, task=Task.MUSIC_GENERATION,
            entrypoint='/models/future-checkpoint',
            files=(ModelFile('model.safetensors', 1024),))

    def test_plan_uses_configured_paths_and_bucket_range(self):
        with TemporaryDirectory() as root:
            checkpoint = Path(root) / 'model.safetensors'
            checkpoint.touch()
            plan = self.engine.plan(replace(self.model,
                                            entrypoint=str(checkpoint)), 8120, {})
            self.assertEqual(plan.argv[plan.argv.index('--model-path') + 1], root)
        self.assertEqual(plan.argv[plan.argv.index('--generator-script') + 1],
                         '/runtime/generate.py')
        self.assertEqual(self.engine.music_form('future-checkpoint'), {
            'duration_kind': 'bucket', 'default_bucket': 0,
            'maximum_bucket': 4})
        self.assertEqual(self.engine.api_paths(),
                         ('/v1/audio/music/generations',))

    def test_unmapped_checkpoint_is_rejected(self):
        self.engine.model_options.clear()
        self.assertFalse(self.engine.supports(self.model))
        with self.assertRaises(ValueError):
            self.engine.plan(self.model, 8120, {})


if __name__ == '__main__':
    unittest.main()
