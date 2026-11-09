import unittest

from ai_lab.music.backends.yue2_web_backend import SongDefaults, Yue2WebBackend


class Yue2RequestTests(unittest.TestCase):
    def setUp(self):
        self.backend = Yue2WebBackend.__new__(Yue2WebBackend)
        self.backend.model_name = 'next-checkpoint'
        self.backend.song = SongDefaults('full', 1.0)

    def _song(self, **fields):
        return self.backend._validate({'prompt': 'soul', 'lyrics': 'hello',
                                       'instrumental': False, **fields})

    def test_cot_and_cfg_fall_back_on_the_instance(self):
        request = self._song()
        self.assertEqual((request['cot'], request['cfg_scale']), ('full', 1.0))

    def test_cot_and_cfg_per_request(self):
        request = self._song(cot='melody', cfg_scale=1.5)
        self.assertEqual((request['cot'], request['cfg_scale']), ('melody', 1.5))
        self.assertEqual(self._song(cfg_scale=0)['cfg_scale'], 0.0)
        self.assertEqual(self._song(cfg_scale=20)['cfg_scale'], 20.0)

    def test_cot_and_cfg_out_of_range_are_refused(self):
        for cot in ('off', 'FULL', '', None, 1):
            with self.assertRaisesRegex(ValueError, 'cot must be full or melody'):
                self._song(cot=cot)
        for cfg in (-0.1, 20.01, True, '1.5', None, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'cfg_scale must be a number from 0 to 20'):
                self._song(cfg_scale=cfg)

    def test_the_studio_receives_cot_and_cfg(self):
        sent = []

        def studio(method, path, body):
            sent.append(body)
            raise RuntimeError('stop after the submit')
        backend = Yue2WebBackend('http://studio', 'yue2', SongDefaults('full'), studio=studio)
        with self.assertRaisesRegex(RuntimeError, 'stop after the submit'):
            backend.generate({'prompt': 'soul', 'lyrics': 'hello', 'instrumental': False,
                                   'cot': 'melody', 'cfg_scale': 1.3, 'seed': 7})
        params = sent[0]['params']
        self.assertEqual((params['cot'], params['cfg_scale'], params['seed']), ('melody', 1.3, 7))

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

    def test_truncation_is_read_per_stage(self):
        from ai_lab.music.backends.yue2_web_backend import was_truncated
        self.assertFalse(was_truncated({'abc': False, 'semantic': False}))
        self.assertTrue(was_truncated({'abc': False, 'semantic': True}))
        self.assertFalse(was_truncated(None))
        self.assertTrue(was_truncated(True))


if __name__ == '__main__':
    unittest.main()
