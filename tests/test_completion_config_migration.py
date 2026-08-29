import importlib.util
import unittest
from pathlib import Path


path = Path(__file__).parents[1] / "scripts" / "migrate-completion-config.py"
spec = importlib.util.spec_from_file_location("completion_config", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CompletionConfigMigrationTests(unittest.TestCase):
    def test_preserves_instances_and_adds_each_item_once(self):
        config = {"models_root": "/models",
                  "instances": [{"id": "existing"}],
                  "repositories": [{"id": "gguf", "format": "gguf"}]}
        first = module.merge(config)
        second = module.merge(first)
        self.assertEqual(second["instances"], [{"id": "existing"}])
        self.assertEqual([item["id"] for item in second["model_roots"]].count(
            "benchmark"), 1)
        self.assertEqual([item["id"] for item in second["repositories"]].count(
            "images-paddleocr"), 1)
        self.assertIn("comfyui", second["engines"])
        profile = second["images"]["profiles"]["sd15-smoke"]
        self.assertEqual(profile["model"], "image-smoke")
        self.assertEqual(profile["task"], "generation")
        edit = second["images"]["profiles"]["sd15-edit-smoke"]
        self.assertEqual(edit["task"], "edit")
        self.assertEqual(edit["inputs"]["image"], ["4", "image"])
        self.assertEqual(second["images"]["profiles"]["qwen-image-benchmark"]
                         ["model"], "image-qwen")
        self.assertEqual(second["images"]["profiles"]["flux2-benchmark"]
                         ["task"], "generation")
        qwen_edit = second["images"]["profiles"]["qwen-edit-benchmark"]
        self.assertEqual(qwen_edit["task"], "edit")
        self.assertEqual(qwen_edit["inputs"]["image"], ["1", "image"])
        bundles = second["downloads"]["bundles"]
        self.assertEqual(len({item["name"] for item in bundles}), 3)
        self.assertEqual(len(bundles), 3)
        self.assertEqual(len(next(item for item in bundles if item["name"] ==
                                  "qwen-image-edit-2511-fp8mixed")["components"]), 3)
        source = second["engines"]["paddleocr"]["source"]
        self.assertIn("paddlepaddle_gpu-3.3.0-cp312", source["requirements"][0])
        self.assertEqual(source["minimum_versions"],
                         {"paddlepaddle-gpu": "3.3.0"})
        self.assertIn("cu130", source["requirements"][0])
        self.assertNotIn("pip_args", source)


if __name__ == "__main__":
    unittest.main()
