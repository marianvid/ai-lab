import unittest

from ai_lab.engines.paddleocr import PaddleOcrEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


class PaddleOcrEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = PaddleOcrEngine(binary="/opt/ai/paddleocr/.venv/bin/python",
                                      server="/opt/ai-lab/ai_lab/images/server.py")
        self.model = ModelSet(id="images-paddleocr/pp-ocrv5", name="pp-ocrv5",
                              format=Format.PADDLEOCR,
                              entrypoint="/models/images/ocr/pp-ocrv5",
                              files=(ModelFile("det/inference.pdmodel", 10),),
                              task=Task.OCR)

    def test_it_serves_ocr(self):
        self.assertEqual(self.engine.tasks(), frozenset({Task.OCR}))
        self.assertIn("/v1/images/ocr", self.engine.api_paths())

    def test_it_starts_the_isolated_adapter(self):
        argv = self.engine.plan(self.model, 8092, {}).argv
        self.assertEqual(argv[argv.index("--backend") + 1], "paddleocr")
        self.assertEqual(argv[argv.index("--model") + 1], self.model.entrypoint)

    def test_it_refuses_a_model_in_the_wrong_format(self):
        wrong = ModelSet(id="gguf/qwen", name="qwen", format=Format.GGUF,
                         entrypoint="/models/gguf/qwen", files=(),
                         task=Task.OCR)
        with self.assertRaises(ValueError):
            self.engine.plan(wrong, 8092, {})

    def test_it_refuses_a_model_with_the_wrong_task(self):
        wrong_task = ModelSet(id="images-paddleocr/pp-ocrv5", name="pp-ocrv5",
                              format=Format.PADDLEOCR,
                              entrypoint="/models/images/ocr/pp-ocrv5",
                              files=(), task=Task.TEXT_GENERATION)
        with self.assertRaises(ValueError):
            self.engine.plan(wrong_task, 8092, {})

    def test_it_reports_no_shape_for_a_task_it_does_not_serve(self):
        self.assertEqual(self.engine.api_paths(Task.TEXT_GENERATION), ())


if __name__ == "__main__":
    unittest.main()
