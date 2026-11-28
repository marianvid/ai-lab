import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
from types import SimpleNamespace

from ai_lab.application.move_files import (
    prune_empty,
    reject_unfit_destination,
    reject_unremovable_sources,
)


class MoveFilesTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)

    def plan(self, destinations, size_bytes=1):
        return SimpleNamespace(destinations=destinations, target_path=self.root,
                               model=SimpleNamespace(size_bytes=size_bytes),
                               target_root=SimpleNamespace(name="Fast"))

    def test_an_existing_destination_is_refused(self):
        taken = self.root / "model.gguf"
        taken.write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "already contains"):
            reject_unfit_destination(self.plan([taken]))

    def test_a_model_larger_than_the_free_space_is_refused(self):
        with self.assertRaisesRegex(ValueError, "Fast does not have enough free space"):
            reject_unfit_destination(self.plan([self.root / "new"], size_bytes=10**30))

    def test_a_free_destination_with_room_passes(self):
        destination = self.root / "new"
        reject_unfit_destination(self.plan([destination]))
        self.assertFalse(destination.exists())                # checking writes nothing

    def test_a_source_in_a_read_only_directory_is_refused(self):
        folder = self.root / "locked"
        folder.mkdir()
        (folder / "model.gguf").write_bytes(b"x")
        folder.chmod(0o500)
        self.addCleanup(folder.chmod, 0o700)
        def as_the_manager(path, mode):
            # Root may write anywhere (CT 102 runs the tests as root); the manager
            # does not, so the folder's mode is answered as it would answer it.
            return not (mode & os.W_OK and Path(path) == folder)
        access = as_the_manager if os.geteuid() == 0 else os.access
        with mock.patch("ai_lab.application.move_files.os.access", access), \
                self.assertRaisesRegex(ValueError, "cannot remove files from"):
            reject_unremovable_sources([folder / "model.gguf"])

    def test_only_empty_directories_under_a_root_are_removed(self):
        empty = self.root / "a" / "empty"
        full = self.root / "a" / "full"
        empty.mkdir(parents=True)
        full.mkdir()
        (full / "keep").write_bytes(b"x")
        prune_empty([empty / "gone", full / "gone", self.root / "top"], [self.root])
        self.assertFalse(empty.exists())
        self.assertTrue(full.exists())
        self.assertTrue(self.root.exists())


if __name__ == "__main__":
    unittest.main()
