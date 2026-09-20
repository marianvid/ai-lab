import unittest

from ai_lab.music.server import validate_payload


class MusicRequestTests(unittest.TestCase):
    def test_defaults_are_valid_and_seed_is_in_range(self):
        result = validate_payload({'prompt': 'Ambient piano'})
        self.assertEqual(result['duration'], 30)
        self.assertTrue(0 <= result['seed'] <= 2**32 - 1)

    def test_rejects_duration_and_unexpected_fields(self):
        with self.assertRaisesRegex(ValueError, 'between 5 and 180'):
            validate_payload({'prompt': 'Music', 'duration': 300})
        with self.assertRaisesRegex(ValueError, 'Unknown music fields'):
            validate_payload({'prompt': 'Music', 'command': 'run'})

    def test_rejects_empty_prompt(self):
        with self.assertRaisesRegex(ValueError, 'prompt must contain'):
            validate_payload({'prompt': ' '})


if __name__ == '__main__':
    unittest.main()
