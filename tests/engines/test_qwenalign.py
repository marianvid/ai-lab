import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.engines.qwenalign import QwenAlignEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


class QwenAlignEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = QwenAlignEngine(binary='/runtime/python')
        self.model = ModelSet(
            id='alignment/new-checkpoint', name='new-checkpoint',
            format=Format.SAFETENSORS, task=Task.ALIGNMENT,
            entrypoint='/models/new-checkpoint',
            files=(ModelFile('model.safetensors', 1024),))

    def test_plan_uses_model_path_and_alignment_contract(self):
        with TemporaryDirectory() as root:
            checkpoint = Path(root) / 'model.safetensors'
            checkpoint.touch()
            plan = self.engine.plan(replace(self.model,
                                            entrypoint=str(checkpoint)), 8119, {})
            self.assertEqual(plan.argv[plan.argv.index('--model') + 1], root)
        self.assertEqual(self.engine.api_paths(), ('/v1/audio/alignments',))

    def test_wrong_task_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.plan(replace(self.model, task=Task.TRANSCRIPTION), 8119, {})


if __name__ == '__main__':
    unittest.main()
