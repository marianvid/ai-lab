"""The services `Operations` is built from: listing, paths and settings changes."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from ai_lab.application import paths
from ai_lab.application.model_listing import ModelListing, _note
from ai_lab.application.settings_changes import SettingsChanges
from ai_lab.config import ConfigStore


def store_at(directory: Path) -> ConfigStore:
    path = directory / "config.json"
    path.write_text(json.dumps({"models_root": str(directory), "engines": {"llamacpp": {}}}))
    return ConfigStore(path)


class PathsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def test_folders_always_programs_only_when_asked_libraries_never(self):
        (self.root / "models").mkdir()
        (self.root / ".hidden").mkdir()
        for name in ("llama-server", "libggml.so.1"):
            program = self.root / name
            program.write_text("")
            os.chmod(program, 0o755)
        kinds = {item["name"]: item["kind"] for item in paths.browse(self.root)["entries"]}
        self.assertEqual(kinds, {"models": "folder"})
        listed = paths.browse(self.root, programs=True)["entries"]
        self.assertEqual({item["name"] for item in listed}, {"models", "llama-server"})

    def test_a_file_is_not_browsable(self):
        (self.root / "file").write_text("")
        with self.assertRaises(ValueError):
            paths.browse(self.root / "file")

    def test_it_starts_beside_the_first_repository_that_exists(self):
        repository = self.root / "gguf"
        repository.mkdir()
        self.assertEqual(paths.browse_root(["/nowhere", str(repository)]), self.root)
        self.assertEqual(paths.browse_root(["/nowhere"]), Path.home())

    def test_making_a_directory_says_whether_it_was_new(self):
        target = self.root / "a" / "b"
        self.assertTrue(paths.create_directory(str(target))["created"])
        self.assertFalse(paths.create_directory(str(target))["created"])
        (self.root / "file").write_text("")
        with self.assertRaises(ValueError):
            paths.create_directory(str(self.root / "file"))


class SettingsChangesTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.store = store_at(self.root)
        self.topics: list[str] = []
        self.changes = SettingsChanges(self.store, cast(Any, SimpleNamespace()), cast(Any, SimpleNamespace()),
                                       self.topics.append)

    def test_gateway_limits_are_checked_then_saved(self):
        self.assertEqual(self.changes.update_gateway({"max_waiting": "20"})["max_waiting"], 20)
        self.assertEqual(self.topics, ["settings"])
        for bad in ({"max_waiting": 0}, {"first_byte_s": "soon"}, {"colour": 1}):
            with self.assertRaises(ValueError):
                self.changes.update_gateway(bad)

    def test_an_engine_program_must_exist_and_run(self):
        with self.assertRaises(KeyError):
            self.changes.update_engine_binary("nope", "/bin/sh")
        with self.assertRaisesRegex(ValueError, "is not there"):
            self.changes.update_engine_binary("llamacpp", str(self.root / "missing"))
        with self.assertRaisesRegex(ValueError, "directory"):
            self.changes.update_engine_binary("llamacpp", str(self.root))
        program = self.root / "server"
        program.write_text("")
        os.chmod(program, 0o755)
        saved = self.changes.update_engine_binary("llamacpp", str(program))
        self.assertEqual(saved["binary"], str(program))

    def test_a_disabled_root_cannot_take_downloads(self):
        self.changes.update_model_root("core", {"enabled": False})
        with self.assertRaisesRegex(ValueError, "disabled"):
            self.changes.update_model_root("core", {"download_default": True})


class ModelListingTests(unittest.TestCase):
    def test_a_note_may_be_one_sentence_or_a_short_and_a_long_form(self):
        self.assertEqual(_note("Fast."), {"summary": "Fast.", "description": "Fast."})
        self.assertEqual(_note({"short": "Fast", "detail": "Fast and small"}),
                         {"summary": "Fast", "description": "Fast and small"})
        self.assertEqual(_note(None), {})

    def test_an_engine_sees_only_what_it_reads(self):
        model = SimpleNamespace(format="gguf", task="text")
        other = SimpleNamespace(format="mlx", task="text")
        engine = SimpleNamespace(formats=lambda: {"gguf"}, tasks=lambda: {"text"})
        listing = ModelListing(cast(Any, None), cast(Any, None),
                               SimpleNamespace(get=lambda _id: engine))
        self.assertEqual(listing._loadable([model, other], "llamacpp"), [model])
