import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.images.server import PaddleOcrBackend


class PaddleOcrModelPairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_resolves_a_bundled_model(self):
        bundle = self.root / "pp-ocr"
        (bundle / "det").mkdir(parents=True)
        (bundle / "rec").mkdir()
        self.assertEqual(PaddleOcrBackend.model_directories(bundle),
                         (bundle / "det", bundle / "rec", None))

    def test_resolves_official_managed_sibling_packages(self):
        detection = self.root / "PP-OCRv5_mobile_det"
        recognition = self.root / "PP-OCRv5_mobile_rec"
        detection.mkdir()
        recognition.mkdir()
        self.assertEqual(PaddleOcrBackend.model_directories(detection),
                         (detection, recognition, None))

    def test_refuses_an_incomplete_pair(self):
        detection = self.root / "PP-OCRv5_mobile_det"
        detection.mkdir()
        with self.assertRaisesRegex(ValueError, "requires det/ and rec/"):
            PaddleOcrBackend.model_directories(detection)


if __name__ == "__main__":
    unittest.main()
