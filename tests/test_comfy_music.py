import unittest

from ai_lab.music.comfy_backend import ComfyMusicBackend


class ComfyMusicRequestTests(unittest.TestCase):
    def setUp(self):
        self.backend = ComfyMusicBackend.__new__(ComfyMusicBackend)
        self.backend.model_name = 'future-music'

    def test_vocal_and_instrumental_requests(self):
        vocal = self.backend._validate({
            'model': 'future-music', 'prompt': 'soul piano',
            'lyrics': '[Verse] Hello', 'instrumental': False,
            'duration': 20, 'seed': 42})
        self.assertEqual(vocal['lyrics'], '[Verse] Hello')
        instrumental = self.backend._validate({
            'prompt': 'ambient pulse', 'instrumental': True,
            'duration': 12, 'seed': 43})
        self.assertEqual(instrumental['lyrics'], '[Instrumental]')

    def test_rejects_wrong_model_and_missing_vocal_lyrics(self):
        with self.assertRaisesRegex(ValueError, 'serves future-music'):
            self.backend._validate({'model': 'other', 'prompt': 'soul',
                                    'instrumental': True})
        with self.assertRaisesRegex(ValueError, 'lyrics are required'):
            self.backend._validate({'prompt': 'soul', 'instrumental': False})


if __name__ == '__main__':
    unittest.main()
