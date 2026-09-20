import unittest
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

from ai_lab.speech.voxcpm_backend import VoxCpmBackend


class FakeModel:
    tts_model = SimpleNamespace(sample_rate=48000)

    def generate(self, **kwargs):
        self.last_request = kwargs
        return [0.0, 0.1, -0.1]


class VoxCpmSpeechTests(unittest.TestCase):
    def setUp(self):
        self.backend = VoxCpmBackend.__new__(VoxCpmBackend)
        self.backend.model = FakeModel()
        self.backend.model_name = 'future-checkpoint'
        self.backend.cfg_value = 2.4
        self.backend.inference_timesteps = 12
        self.backend.lock = Lock()

    def test_voice_design_uses_instruction_without_checkpoint_assumptions(self):
        with patch('ai_lab.speech.voxcpm_backend.wav_result',
                   return_value={'sample_rate': 48000, 'mode': 'voxcpm'}) as wav:
            result = self.backend.generate({
                'model': 'future-checkpoint', 'text': 'Hello',
                'instruction': 'Warm, relaxed delivery'})
        wav.assert_called_once()
        self.assertEqual(self.backend.model.last_request['text'],
                         '(Warm, relaxed delivery)Hello')
        self.assertEqual(result['sample_rate'], 48000)
        self.assertEqual(result['mode'], 'voxcpm')

    def test_speaker_without_reference_audio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'reference audio'):
            self.backend.generate({'text': 'Hello', 'speaker': 'unknown'})


if __name__ == '__main__':
    unittest.main()
