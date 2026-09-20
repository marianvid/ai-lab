import base64
import unittest

from ai_lab.video.comfy_backend import ComfyVideoBackend, PNG_SIGNATURE


class ComfyVideoRequestTests(unittest.TestCase):
    def setUp(self):
        self.backend = ComfyVideoBackend.__new__(ComfyVideoBackend)
        self.backend.model_name = 'future-video'
        self.image = base64.b64encode(PNG_SIGNATURE + b'0' * 32).decode()

    def test_prompt_and_reference_image(self):
        result = self.backend._validate({
            'model': 'future-video', 'prompt': 'A gentle camera move',
            'image_base64': self.image, 'seed': 42})
        self.assertEqual(result['image'][:8], PNG_SIGNATURE)
        self.assertEqual(result['seed'], 42)

    def test_rejects_non_png_and_wrong_model(self):
        with self.assertRaisesRegex(ValueError, 'must be a PNG'):
            self.backend._validate({'prompt': 'Move',
                                    'image_base64': base64.b64encode(b'jpg').decode()})
        with self.assertRaisesRegex(ValueError, 'serves future-video'):
            self.backend._validate({'model': 'other', 'prompt': 'Move',
                                    'image_base64': self.image})


if __name__ == '__main__':
    unittest.main()
