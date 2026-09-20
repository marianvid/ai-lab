import unittest
from threading import Lock
from types import SimpleNamespace

from ai_lab.audio.aligner import QwenAlignBackend


class FakeModel:
    def align(self, **kwargs):
        self.last_request = kwargs
        return [[SimpleNamespace(text='Hello', start_time=0.1, end_time=0.5)]]


class QwenAlignRequestTests(unittest.TestCase):
    def setUp(self):
        self.backend = QwenAlignBackend.__new__(QwenAlignBackend)
        self.backend.model = FakeModel()
        self.backend.torch = SimpleNamespace(inference_mode=lambda: Lock())
        self.backend.lock = Lock()

    def test_alignment_returns_word_timestamps(self):
        words = self.backend.align('/tmp/example.wav', 'Hello', 'English')
        self.assertEqual(words, [{'text': 'Hello', 'start': 0.1, 'end': 0.5}])
        self.assertEqual(self.backend.model.last_request['text'], 'Hello')

    def test_language_is_checked_before_running_model(self):
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            self.backend.align('/tmp/example.wav', 'Hello', 'Romanian')


if __name__ == '__main__':
    unittest.main()
