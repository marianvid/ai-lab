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
        # One root, and only one of its two format folders was ever created.
        # A machine set up on another disk looks exactly like this.
        (self.root / "gguf").mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "models_root": str(self.root),
            "repositories": [
                {"id": "gguf", "name": "GGUF", "format": "gguf"},
                {"id": "gone", "name": "Missing", "format": "safetensors"},
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


class WhenTheRootIsNotSet(unittest.TestCase):
    """A configuration whose repositories never shared a directory.

    Nothing can be derived from it, so every path comes out empty — which must
    read as "not configured" rather than as a folder that happens to be there.
    `Path("")` is the current working directory and exists, which is how this
    was found.
    """

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        path = self.root / "config.json"
        path.write_text(json.dumps({
            "repositories": [
                {"id": "a", "name": "A", "path": "/one/place/gguf", "format": "gguf"},
                {"id": "b", "name": "B", "path": "/another/nvfp4", "format": "nvfp4"},
            ],
        }))
        self.settings = Settings(ConfigStore(path), FakeHost(), Registry())

    def test_every_repository_reports_no_directory(self):
        rows = self.settings.view()["repositories"]
        self.assertEqual([item["path"] for item in rows], ["", ""])
        self.assertEqual([item["exists"] for item in rows], [False, False])
        self.assertEqual([item["writable"] for item in rows], [False, False])

    def test_the_root_is_reported_empty_rather_than_guessed(self):
        self.assertEqual(self.settings.view()["models_root"], "")
