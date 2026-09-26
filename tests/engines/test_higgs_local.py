import unittest

from ai_lab.engines.higgs_local import HiggsLocalEngine
from ai_lab.speech.higgs_local_backend import frame_limit
from ai_lab.types import Format, ModelFile, ModelSet, Task


class HiggsLocalEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = HiggsLocalEngine(
            binary='/runtime/python',
            model_options={'higgs-port': {'memory_reservation_mb': 12000}})
        self.model = ModelSet(
            id='audio-tts/higgs-port', name='higgs-port',
            format=Format.SAFETENSORS, task=Task.SPEECH_SYNTHESIS,
            entrypoint='/models/higgs-port',
            files=(ModelFile('model.safetensors', 1024),))

    def test_plan_runs_the_shared_speech_host_offline(self):
        plan = self.engine.plan(self.model, 8125, {})
        self.assertEqual(plan.argv[plan.argv.index('--backend') + 1], 'higgs')
        self.assertEqual(plan.argv[plan.argv.index('--model-path') + 1],
                         '/models/higgs-port')
        self.assertEqual(plan.env['HF_HUB_OFFLINE'], '1')
        self.assertEqual(self.engine.needs_mb(self.model, {}, 0), 12000)
        self.assertEqual(self.engine.concurrency({}), 1)
        self.assertEqual(plan.argv[plan.argv.index('--max-batch') + 1], '1')
        # The browser playground sits on the instance port plus 10000.
        self.assertEqual(plan.argv[plan.argv.index('--ui-port') + 1], '18125')

    def test_batch_size_sets_gateway_concurrency(self):
        engine = HiggsLocalEngine(
            model_options={'higgs-port': {'memory_reservation_mb': 12000}},
            max_batch=8, batch_window_ms=50)
        plan = engine.plan(self.model, 8125, {})
        self.assertEqual(engine.concurrency({}), 8)
        self.assertEqual(plan.argv[plan.argv.index('--max-batch') + 1], '8')
        self.assertEqual(
            plan.argv[plan.argv.index('--batch-window-ms') + 1], '50')
        for bad in ({'max_batch': 0}, {'max_batch': 17}, {'max_batch': '8'},
                    {'batch_window_ms': -1}):
            with self.assertRaises(ValueError):
                HiggsLocalEngine(**bad)

    def test_form_offers_cloning_and_seed(self):
        form = self.engine.speech_form('higgs-port')
        self.assertTrue(form['reference_supported'])
        self.assertTrue(form['seed_supported'])

    def test_unmapped_checkpoint_and_settings_are_refused(self):
        with self.assertRaises(ValueError):
            self.engine.plan(self.model, 8125, {'context_size': 1})
        self.engine.model_options.clear()
        self.assertFalse(self.engine.supports(self.model))

    def test_runaway_limit_follows_the_text(self):
        short, long = frame_limit('Salut'), frame_limit('x' * 400)
        self.assertLess(short, long)
        self.assertLessEqual(long, 2048)
        self.assertGreaterEqual(frame_limit(''), 64)


if __name__ == '__main__':
    unittest.main()
