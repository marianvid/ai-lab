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
        view = self.storage.clear("packages")
        self.assertFalse(self.cache.exists())
        self.assertEqual(view["items"], [],
                         "a deleted location left an empty row behind")

    def test_a_configured_location_that_does_not_exist_is_not_inventory(self):
        absent = self.root / "two" / "old-copy"
        storage = Storage({"reclaimable": [{
            "id": "old-copy", "path": str(absent), "kind": "leftover",
        }]})
        self.assertEqual(storage.view(), {"items": [], "recoverable_bytes": 0})

    def test_an_empty_cache_directory_is_not_offered_as_space(self):
        empty = self.root / "two" / "empty-cache"
        empty.mkdir(parents=True)
        storage = Storage({"reclaimable": [{
            "id": "empty", "path": str(empty), "kind": "cache",
        }]})
        self.assertEqual(storage.view()["items"], [])

    def test_a_path_cannot_be_smuggled_in_as_an_id(self):
        with self.assertRaises(KeyError):
            self.storage.clear("../../models")
        self.assertTrue(self.cache.exists())

    def test_broad_paths_are_refused_at_configuration_time(self):
        with self.assertRaises(ValueError):
            Storage({"reclaimable": [{"id": "bad", "path": "/var"}]})


if __name__ == "__main__":
    unittest.main()
