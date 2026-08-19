import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.config import ConfigStore
from ai_lab.engines.registry import Registry
from ai_lab.settings import Settings

from tests.support import FakeHost


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        (self.root / "gguf").mkdir()
        path = self.root / "config.json"
        path.write_text(json.dumps({
            "repositories": [
                {"id": "gguf", "name": "GGUF", "path": str(self.root / "gguf"),
                 "format": "gguf"},
                {"id": "gone", "name": "Missing", "path": "/does/not/exist",
                 "format": "safetensors"},
            ],
        }))
        self.settings = Settings(ConfigStore(path), FakeHost(), Registry())

    def test_free_space_is_reported_for_a_real_repository(self):
        rows = {item["id"]: item for item in self.settings.view()["repositories"]}
        self.assertTrue(rows["gguf"]["exists"])
        self.assertGreater(rows["gguf"]["free_bytes"], 0)

    def test_a_missing_directory_is_flagged_rather_than_hidden(self):
        rows = {item["id"]: item for item in self.settings.view()["repositories"]}
        self.assertFalse(rows["gone"]["exists"])
        self.assertEqual(rows["gone"]["free_bytes"], 0)

    def test_the_accelerator_is_included(self):
        self.assertEqual(self.settings.view()["accelerator"]["name"], "Fake GPU")

    def test_engines_are_listed_with_availability(self):
        engines = {item["id"]: item for item in self.settings.view()["engines"]}
        self.assertTrue(engines["llamacpp"]["available"])
        self.assertFalse(engines["vllm"]["available"])

    def test_the_accelerator_is_declared_read_only(self):
        self.assertFalse(self.settings.view()["host"]["can_configure_accelerator"])

    def test_the_operating_system_is_reported_explicitly(self):
        self.assertEqual(self.settings.view()["host"]["operating_system"], "Test OS")


if __name__ == "__main__":
    unittest.main()
