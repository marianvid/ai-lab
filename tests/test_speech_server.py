import unittest

from ai_lab.speech.server import validate_payload


class SpeechRequestTests(unittest.TestCase):
    def test_valid_text_and_defaults(self):
        result = validate_payload({'text': '  Hello  '})
        self.assertEqual(result['text'], 'Hello')
        self.assertEqual(result['language'], 'Auto')

    def test_invalid_text_and_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'text must contain'):
            validate_payload({'text': ' '})
        with self.assertRaisesRegex(ValueError, 'Unknown speech fields'):
            validate_payload({'text': 'Hello', 'command': 'run'})


if __name__ == '__main__':
    unittest.main()
