import base64
import os
import unittest
from threading import BoundedSemaphore, Lock
from types import SimpleNamespace
from unittest.mock import patch

from ai_lab.speech.contract import ReferenceFile, validate_payload
from ai_lab.speech.higgs_backend import HiggsBackend
from ai_lab.speech.voxcpm_backend import VoxCpmBackend

WAV = b'RIFF' + b'\0' * 40
WAV_B64 = base64.b64encode(WAV).decode()


class SpeechContractTests(unittest.TestCase):
    def test_seed_and_reference_refused_unless_the_engine_uses_them(self):
        with self.assertRaisesRegex(ValueError, 'seed'):
            validate_payload({'text': 'Salut', 'seed': 3})
        with self.assertRaisesRegex(ValueError, 'reference_audio'):
            validate_payload({'text': 'Salut', 'reference_audio': WAV_B64})

    def test_reference_is_decoded_and_checked(self):
        request = validate_payload({'text': 'Salut', 'seed': 3,
                                    'reference_audio': WAV_B64,
                                    'reference_text': 'Bună ziua.'},
                                   seed=True, reference=True)
        self.assertEqual(request['seed'], 3)
        self.assertEqual(request['reference_audio'], WAV)
        self.assertEqual(request['reference_text'], 'Bună ziua.')
        for bad in ({'reference_audio': 'not base64!'},
                    {'reference_audio': base64.b64encode(b'ID3mp3').decode()},
                    {'reference_text': 'orphan transcript'},
                    {'seed': -1}, {'seed': 1.5}):
            with self.assertRaises(ValueError):
                validate_payload({'text': 'Salut', **bad},
                                 seed=True, reference=True)

    def test_reference_file_exists_only_during_use(self):
        with ReferenceFile(WAV) as path:
            with open(path, 'rb') as handle:
                self.assertEqual(handle.read(), WAV)
        self.assertFalse(os.path.exists(path))
        with ReferenceFile(None) as path:
            self.assertIsNone(path)


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self):
        return b'RIFFfake'


class HiggsReferenceTests(unittest.TestCase):
    def test_reference_and_seed_reach_the_worker(self):
        backend = HiggsBackend.__new__(HiggsBackend)
        backend.model_name = 'm'
        backend.model_path = '/models/m'
        backend.worker_port = 8122
        backend.slots = BoundedSemaphore(1)
        seen = {}

        def fake_open(call, timeout):
            import json
            seen.update(json.loads(call.data))
            seen['file_existed'] = os.path.exists(seen['references'][0]['audio_path'])
            return FakeResponse()

        with patch('urllib.request.urlopen', side_effect=fake_open):
            backend.generate({'text': 'Salut', 'seed': 7,
                              'reference_audio': WAV_B64,
                              'reference_text': 'Bună ziua.'})
        self.assertEqual(seen['seed'], 7)
        self.assertEqual(seen['references'][0]['text'], 'Bună ziua.')
        self.assertTrue(seen['file_existed'])
        self.assertFalse(os.path.exists(seen['references'][0]['audio_path']))


class VoxCpmReferenceTests(unittest.TestCase):
    def test_reference_clones_and_transcript_continues(self):
        backend = VoxCpmBackend.__new__(VoxCpmBackend)
        calls = {}
        backend.model = SimpleNamespace(
            tts_model=SimpleNamespace(sample_rate=48000),
            generate=lambda **kw: calls.update(kw) or [0.0])
        backend.model_name = 'm'
        backend.cfg_value = 2.0
        backend.inference_timesteps = 10
        backend.lock = Lock()
        with patch('ai_lab.speech.voxcpm_backend.wav_result', return_value={}):
            backend.generate({'text': 'Salut', 'reference_audio': WAV_B64,
                              'reference_text': 'Bună ziua.'})
        self.assertEqual(calls['reference_wav_path'], calls['prompt_wav_path'])
        self.assertEqual(calls['prompt_text'], 'Bună ziua.')


if __name__ == '__main__':
    unittest.main()
