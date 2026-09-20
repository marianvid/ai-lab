import unittest

from ai_lab.engines.kokoro import KokoroEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


class KokoroEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = KokoroEngine(binary='/runtime/python', model_options={
            'future-checkpoint': {'language_code': 'b',
                                  'default_voice': 'bm_george',
                                  'repo_id': 'publisher/model'}})
        self.model = ModelSet(
            id='tts/future-checkpoint', name='future-checkpoint',
            format=Format.SAFETENSORS, task=Task.SPEECH_SYNTHESIS,
            entrypoint='/models/future-checkpoint/weights.pth',
            files=(ModelFile('weights.pth', 1024),))

    def test_checkpoint_and_voice_come_from_configuration(self):
        plan = self.engine.plan(self.model, 8117, {})
        self.assertTrue(self.engine.supports(self.model))
        self.assertEqual(plan.argv[plan.argv.index('--model-path') + 1],
                         self.model.entrypoint)
        self.assertEqual(plan.argv[plan.argv.index('--default-voice') + 1],
                         'bm_george')
        self.assertEqual(plan.argv[plan.argv.index('--repo-id') + 1],
                         'publisher/model')
        self.assertEqual(self.engine.api_paths(),
                         ('/v1/audio/speech/generations',))

    def test_unmapped_checkpoint_is_not_offered(self):
        self.engine.model_options.clear()
        self.assertFalse(self.engine.supports(self.model))
        with self.assertRaises(ValueError):
            self.engine.plan(self.model, 8117, {})


if __name__ == '__main__':
    unittest.main()
