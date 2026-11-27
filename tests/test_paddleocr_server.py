import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import json
from threading import Lock

from ai_lab.images.server import OcrHost, PaddleOcrBackend


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



class Page:
    """One page as PaddleOCR answers it: an object whose `json` is a property or a method."""

    def __init__(self, payload, as_method=False):
        self._payload = payload
        if as_method:
            self.json = lambda: payload
        else:
            self.json = payload


class RecognitionTests(unittest.TestCase):
    """What the OCR host makes of PaddleOCR's pages, with no model loaded."""

    def backend(self, pages):
        backend = PaddleOcrBackend.__new__(PaddleOcrBackend)
        backend.lock = Lock()
        backend.pipeline = type("Pipeline", (), {"predict": lambda _self, _path: pages})()
        return backend

    def test_every_form_of_page_becomes_lines_with_confidence_and_outline(self):
        first = {"res": {"rec_texts": ["Hello"], "rec_scores": [0.98],
                         "rec_polys": [[[0, 0], [10, 0], [10, 5], [0, 5]]],
                         "doc_preprocessor_res": {"angle": 90}}}
        second = json.dumps({"rec_texts": ["world", "no score"], "rec_scores": [0.5],
                             "dt_polys": [[[1, 1], [2, 2]]]})
        answer = self.backend([Page(first), Page(second, as_method=True), Page(42)]).recognize(
            "page.png", "en")
        self.assertEqual(answer["text"], "Hello\nworld\nno score")
        self.assertEqual(answer["orientation_degrees"], 90)
        self.assertEqual(answer["lines"][0]["polygon"], [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0],
                                                         [0.0, 5.0]])
        self.assertEqual([line["confidence"] for line in answer["lines"]], [0.98, 0.5, 0.0])
        self.assertEqual((answer["language"], answer["warnings"]), ("en", []))

    def test_an_image_with_no_text_says_so(self):
        answer = self.backend([Page({"rec_texts": []})]).recognize("blank.png")
        self.assertEqual((answer["text"], answer["warnings"]), ("", ["no text recognized"]))

    def test_the_host_reads_the_uploaded_image_and_its_language(self):
        seen = []

        class Backend:
            def recognize(self, path, language):
                seen.append((path.endswith(".jpg"), language))
                return {"text": "ok"}
        host = OcrHost.__new__(OcrHost)
        host.backend = Backend()  # type: ignore[assignment]
        answer = host.answer({"file": ("scan.jpg", b"jpeg bytes"), "language": (None, b"ro")})
        self.assertEqual((answer, seen), ({"text": "ok"}, [(True, "ro")]))
        with self.assertRaisesRegex(ValueError, "must contain an image file"):
            host.answer({})


if __name__ == "__main__":
    unittest.main()
