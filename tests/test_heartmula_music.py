import unittest

from ai_lab.music.heartmula_backend import HeartMulaBackend


class HeartMulaRequestTests(unittest.TestCase):
    def setUp(self):
        self.backend = HeartMulaBackend.__new__(HeartMulaBackend)
        self.backend.model_name = 'future-checkpoint'

    def test_lyrics_and_style_are_required(self):
        with self.assertRaisesRegex(ValueError, 'style tags and lyrics'):
            self.backend._validate({'prompt': 'jazz', 'lyrics': ''})
        with self.assertRaisesRegex(ValueError, 'style tags and lyrics'):
            self.backend._validate({'prompt': '', 'lyrics': 'hello'})

    def test_valid_song_request_preserves_duration_and_seed(self):
        request = self.backend._validate({
            'model': 'future-checkpoint', 'prompt': 'jazz,piano',
            'lyrics': 'Hello from AI Lab', 'instrumental': False,
            'duration': 30, 'seed': 42})
        self.assertEqual(request['duration'], 30)
        self.assertEqual(request['seed'], 42)


if __name__ == '__main__':
    unittest.main()
