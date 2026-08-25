from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ai_lab.events import EventBus
from ai_lab.storage import Storage


class ReclaimableStorage(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "one" / "cache"
        self.cache.mkdir(parents=True)
        (self.cache / "wheel.bin").write_bytes(b"x" * 123)
        self.storage = Storage({"reclaimable": [{
            "id": "packages", "name": "Package downloads",
            "path": str(self.cache), "kind": "cache",
        }]}, EventBus())

    def test_it_reports_only_configured_space(self):
        view = self.storage.view()
        self.assertEqual(view["recoverable_bytes"], 123)
        self.assertEqual(view["items"][0]["id"], "packages")

    def test_an_id_clears_its_fixed_path(self):
        self.storage.clear("packages")
        self.assertFalse(self.cache.exists())

    def test_a_path_cannot_be_smuggled_in_as_an_id(self):
        with self.assertRaises(KeyError):
            self.storage.clear("../../models")
        self.assertTrue(self.cache.exists())

    def test_broad_paths_are_refused_at_configuration_time(self):
        with self.assertRaises(ValueError):
            Storage({"reclaimable": [{"id": "bad", "path": "/var"}]})


if __name__ == "__main__":
    unittest.main()
