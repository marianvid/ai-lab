import unittest

from ai_lab.music.yue2_backend import Yue2Backend


class Yue2RequestTests(unittest.TestCase):
    def setUp(self):
        self.backend = Yue2Backend.__new__(Yue2Backend)
        self.backend.model_name = 'next-checkpoint'

    def test_editable_score_and_lyrics_reach_pipeline(self):
        request = self.backend._validate({
            'model': 'next-checkpoint', 'prompt': 'soul piano',
            'lyrics': '[Verse]\nHello AI Lab', 'abc': 'X:1\nT:Test\n',
            'instrumental': False, 'seed': 42})
        self.assertEqual(request['abc'], 'X:1\nT:Test\n')
        self.assertEqual(request['seed'], 42)

    def test_rejects_duration_and_instrumental_requests(self):
        with self.assertRaisesRegex(ValueError, 'Unknown YuE2 fields'):
            self.backend._validate({'prompt': 'soul', 'lyrics': 'hello',
                                    'duration': 30})
        with self.assertRaisesRegex(ValueError, 'lyric-conditioned'):
            self.backend._validate({'prompt': 'soul', 'lyrics': 'hello',
                                    'instrumental': True})


if __name__ == '__main__':
    unittest.main()
