import unittest

from ai_lab.engines.comfyui import ComfyUiEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


class ComfyUiEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = ComfyUiEngine(binary="/opt/comfy/python",
                                    comfyui="/opt/ComfyUI/main.py",
                                    model_paths=["/models", "/test_models"])
        self.model = ModelSet(id="images-comfyui-generation/flux", name="flux",
                              format=Format.COMFYUI, entrypoint="/models/flux",
                              files=(ModelFile("model.safetensors", 10),),
                              task=Task.IMAGE_GENERATION)

    def test_it_is_serial_and_serves_generation(self):
        self.assertEqual(self.engine.concurrency({}), 1)
        self.assertIn("/v1/images/generations",
                      self.engine.api_paths(Task.IMAGE_GENERATION))

    def test_one_image_instance_serves_generation_and_editing(self):
        expected = ("/v1/images/generations", "/v1/images/edits")
        self.assertEqual(self.engine.api_paths(Task.IMAGE_GENERATION), expected)
        self.assertEqual(self.engine.api_paths(Task.IMAGE_EDIT), expected)

    def test_plan_uses_private_bridge_and_model_root(self):
        argv = self.engine.plan(self.model, 8093, {}).argv
        self.assertEqual(argv[argv.index("--comfyui") + 1], "/opt/ComfyUI/main.py")
        self.assertEqual(argv[argv.index("--model-root") + 1], "/models/flux")
        self.assertEqual(argv.count("--extra-model-root"), 2)

    def test_low_vram_is_an_explicit_cpu_split(self):
        plan = self.engine.plan(self.model, 8093, {"vram_mode": "low"})
        self.assertIn("--lowvram", plan.argv)
        self.assertTrue(plan.splits_across_cpu)

    def test_normal_vram_keeps_the_admission_check(self):
        plan = self.engine.plan(self.model, 8093, {})
        self.assertNotIn("--lowvram", plan.argv)
        self.assertFalse(plan.splits_across_cpu)

    def test_wrong_format_is_rejected(self):
        wrong = ModelSet(id="x", name="x", format=Format.GGUF,
                         entrypoint="/x", files=(), task=Task.IMAGE_GENERATION)
        with self.assertRaises(ValueError):
            self.engine.plan(wrong, 8093, {})


if __name__ == "__main__":
    unittest.main()
