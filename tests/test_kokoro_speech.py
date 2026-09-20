import unittest
from pathlib import Path

from ai_lab.speech.kokoro_backend import KokoroBackend


class KokoroRequestTests(unittest.TestCase):
    def setUp(self):
        self.backend = KokoroBackend.__new__(KokoroBackend)
        self.backend.model_name = 'future-checkpoint'
        self.backend.model_path = Path('/models/future-checkpoint')
        self.backend.default_voice = 'bm_george'

    def test_voice_instruction_is_rejected_instead_of_ignored(self):
        with self.assertRaisesRegex(ValueError, 'does not support'):
            self.backend.generate({'text': 'Hello', 'instruction': 'Whisper'})

    def test_voice_name_cannot_escape_checkpoint(self):
        with self.assertRaisesRegex(ValueError, 'voice must be a name'):
            self.backend.generate({'text': 'Hello', 'speaker': '../other'})

    def test_unknown_voice_is_named(self):
        with self.assertRaisesRegex(ValueError, 'Unknown voice'):
            self.backend.generate({'text': 'Hello', 'speaker': 'missing'})


if __name__ == '__main__':
    unittest.main()
