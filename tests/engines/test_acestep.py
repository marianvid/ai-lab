import unittest

from ai_lab.engines.acestep import AceStepEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


class AceStepEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = AceStepEngine(
            binary='/ace/python', project_root='/ace/source',
            output_root='/state/music',
            model_configs={'ace-step-1.5-xl-turbo': 'acestep-v15-xl-turbo'})
        self.model = ModelSet(
            id='music/ace-step-1.5-xl-turbo', name='ace-step-1.5-xl-turbo',
            format=Format.SAFETENSORS, task=Task.MUSIC_GENERATION,
            entrypoint='/models/music/ace-step-1.5-xl-turbo',
            files=(ModelFile('weights.safetensors', 1024),))

    def test_plan_uses_explicit_runtime_and_checkpoint(self):
        plan = self.engine.plan(self.model, 8105, {})
        self.assertEqual(plan.argv[0], '/ace/python')
        self.assertEqual(plan.argv[plan.argv.index('--config-name') + 1],
                         'acestep-v15-xl-turbo')
        self.assertEqual(plan.argv[plan.argv.index('--model-path') + 1],
                         self.model.entrypoint)
        self.assertEqual(plan.argv[-1], '8105')
        self.assertEqual(self.engine.concurrency({}), 1)
        self.assertEqual(self.engine.api_paths(),
                         ('/v1/audio/music/generations',))

    def test_unmapped_model_is_refused(self):
        self.engine.model_configs.clear()
        with self.assertRaisesRegex(ValueError, 'not configured'):
            self.engine.plan(self.model, 8105, {})

    def test_only_music_safetensors_are_supported(self):
        other = ModelSet(id='x', name='x', format=Format.SAFETENSORS,
                         entrypoint='/x', files=(), task=Task.SPEECH_SYNTHESIS)
        with self.assertRaises(ValueError):
            self.engine.plan(other, 8105, {})


if __name__ == '__main__':
    unittest.main()
