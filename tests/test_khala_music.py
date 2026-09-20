import unittest
from pathlib import Path

from ai_lab.music.khala_backend import KhalaBackend


class KhalaRequestTests(unittest.TestCase):
    def setUp(self):
        self.backend = KhalaBackend.__new__(KhalaBackend)
        self.backend.model_name = 'future-checkpoint'
        self.backend.default_bucket = 0
        self.backend.maximum_bucket = 4

    def test_instrumental_request_uses_configured_bucket(self):
        request = self.backend._validate({
            'model': 'future-checkpoint', 'prompt': 'Warm jazz piano',
            'instrumental': True, 'seed': 42})
        self.assertEqual(request['length_bucket'], 0)
        self.assertTrue(request['instrumental'])

    def test_seconds_are_not_silently_treated_as_buckets(self):
        with self.assertRaisesRegex(ValueError, 'Unknown Khala fields'):
            self.backend._validate({'prompt': 'Piano', 'duration': 30})


if __name__ == '__main__':
    unittest.main()
