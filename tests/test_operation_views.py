"""`Operations` is the union of narrow views, one per group of routes."""

import inspect
import unittest

from ai_lab import operations
from ai_lab.operations import (CardView, DownloadsView, EngineUpdatesView, EntriesView,
                               LibraryView, MachineSettingsView, Operations, StorageView)

VIEWS = (CardView, EntriesView, LibraryView, DownloadsView, EngineUpdatesView,
         MachineSettingsView, StorageView)


def own(view: type) -> set[str]:
    """The public sentences a view itself defines (not those it inherits)."""
    return {name for name in vars(view) if not name.startswith("_")}


class OperationViewTests(unittest.TestCase):
    def test_every_sentence_belongs_to_exactly_one_view(self):
        names = [name for view in VIEWS for name in own(view)]
        self.assertEqual(len(names), len(set(names)), "a sentence in two views")
        public = {name for name, _ in inspect.getmembers(Operations)
                  if not name.startswith("_")}
        self.assertEqual(public - {"assemble"}, set(names))
        self.assertEqual(own(Operations), {"assemble"})

    def test_operations_is_every_view(self):
        for view in VIEWS:
            with self.subTest(view.__name__):
                self.assertTrue(issubclass(Operations, view))

    def test_each_route_module_is_typed_with_its_view(self):
        from ai_lab.api.routes import catalog, downloads, huggingface, runtime, storage
        expected = {runtime: "EntriesView", catalog: "LibraryView",
                    downloads: "DownloadsView", huggingface: "DownloadsView",
                    storage: "StorageView"}
        for module, view in expected.items():
            with self.subTest(module.__name__):
                self.assertEqual(module.register.__annotations__["operations"], view)
                self.assertTrue(hasattr(operations, view))


if __name__ == "__main__":
    unittest.main()
