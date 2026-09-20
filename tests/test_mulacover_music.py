import base64
import unittest

from ai_lab.music.mulacover_backend import MulaCoverBackend


class MulaCoverRequestTests(unittest.TestCase):
    def setUp(self):
        self.backend = MulaCoverBackend.__new__(MulaCoverBackend)
        self.backend.model_name = 'future-cover'
        self.reference = base64.b64encode(b'RIFF' + b'0' * 64).decode()

    def test_source_audio_and_lyrics(self):
        result = self.backend._validate({
            'model': 'future-cover', 'prompt': 'genre:[soul]',
            'lyrics': '[Verse]\nHello', 'instrumental': False,
            'reference_audio_base64': self.reference, 'seed': 42})
        self.assertEqual(result['audio'][:4], b'RIFF')
        self.assertEqual(result['seed'], 42)

    def test_rejects_missing_or_invalid_audio(self):
        body = {'prompt': 'genre:[soul]', 'lyrics': '[Verse]\nHello',
                'instrumental': False}
        with self.assertRaisesRegex(ValueError, 'reference WAV'):
            self.backend._validate(body)
        body['reference_audio_base64'] = base64.b64encode(b'not wav').decode()
        with self.assertRaisesRegex(ValueError, 'must be a WAV'):
            self.backend._validate(body)


if __name__ == '__main__':
    unittest.main()
