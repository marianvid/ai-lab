import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.engines.higgs import HiggsEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


class HiggsEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = HiggsEngine(
            binary='/runtime/python', worker_binary='/runtime/sgl-omni',
            model_options={'future-checkpoint': {
                'worker_port': 9000, 'mem_fraction_static': 0.8,
                'memory_reservation_mb': 30000}})
        self.model = ModelSet(
            id='tts/future-checkpoint', name='future-checkpoint',
            format=Format.SAFETENSORS, task=Task.SPEECH_SYNTHESIS,
            entrypoint='/models/future-checkpoint',
            files=(ModelFile('model.safetensors', 1024),))

    def test_plan_uses_configured_worker_and_reservation(self):
        with TemporaryDirectory() as root:
            checkpoint = Path(root) / 'model.safetensors'
            checkpoint.touch()
            plan = self.engine.plan(replace(self.model,
                                            entrypoint=str(checkpoint)), 8121, {})
            self.assertEqual(plan.argv[plan.argv.index('--model-path') + 1], root)
        self.assertEqual(plan.argv[plan.argv.index('--worker-port') + 1], '9000')
        self.assertEqual(plan.argv[plan.argv.index('--ui-port') + 1], '18121')
        self.assertEqual(self.engine.needs_mb(self.model, {}, 32623), 30000)
        self.assertEqual(self.engine.api_paths(),
                         ('/v1/audio/speech/generations',))

    def test_parallel_limit_reaches_gateway_and_worker_host(self):
        self.assertEqual(self.engine.concurrency({}), 1)
        engine = HiggsEngine(worker_binary='/runtime/sgl-omni',
                             model_options=self.engine.model_options,
                             max_parallel=8)
        self.assertEqual(engine.concurrency({}), 8)
        plan = engine.plan(self.model, 8121, {})
        self.assertEqual(plan.argv[plan.argv.index('--max-parallel') + 1], '8')

    def test_parallel_limit_is_bounded(self):
        for bad in (0, 65, 2.0, '8'):
            with self.assertRaises(ValueError):
                HiggsEngine(max_parallel=bad)

    def test_unmapped_checkpoint_is_rejected(self):
        self.engine.model_options.clear()
        self.assertFalse(self.engine.supports(self.model))
        with self.assertRaises(ValueError):
            self.engine.plan(self.model, 8121, {})


if __name__ == '__main__':
    unittest.main()
