import base64
import json
import unittest
from pathlib import Path
from threading import BoundedSemaphore
from unittest.mock import patch

import socket

from ai_lab.speech.higgs_backend import HiggsBackend, wait_for_free_port


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self):
        return b'RIFFfake-wav'


class HiggsRequestTests(unittest.TestCase):
    def setUp(self):
        self.backend = HiggsBackend.__new__(HiggsBackend)
        self.backend.model_name = 'future-checkpoint'
        self.backend.model_path = Path('/models/future-checkpoint')
        self.backend.worker_port = 8122
        self.backend.slots = BoundedSemaphore(8)

    def test_converts_binary_worker_audio_to_speech_contract(self):
        with patch('urllib.request.urlopen', return_value=FakeResponse()) as call:
            result = self.backend.generate({'model': 'future-checkpoint',
                                            'text': 'Hello', 'speaker': 'default'})
        payload = json.loads(call.call_args.args[0].data)
        self.assertEqual(payload['model'], '/models/future-checkpoint')
        self.assertEqual(payload['input'], 'Hello')
        self.assertEqual(base64.b64decode(result['data'][0]['b64_wav']),
                         b'RIFFfake-wav')

    def test_instruction_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, 'inline'):
            self.backend.generate({'text': 'Hello', 'instruction': 'calm'})


if __name__ == '__main__':
    unittest.main()


class WorkerPortTests(unittest.TestCase):
    def test_a_free_port_is_accepted_at_once(self):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        wait_for_free_port(port, timeout_s=1)

    def test_a_held_port_is_reported_instead_of_silently_moved(self):
        with socket.socket() as holder:
            holder.bind(('127.0.0.1', 0))
            holder.listen()
            with self.assertRaisesRegex(RuntimeError, 'still in use'):
                wait_for_free_port(holder.getsockname()[1], timeout_s=0)
