"""The small per-vendor state file the logins and the limits are kept in."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.providers import state_file


class StateFileTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.path = Path(self._temporary.name) / "sub" / "states.json"

    def test_written_states_are_read_back(self):
        state_file.write(self.path, {"openai": {"state": "ok"}})
        self.assertEqual(state_file.read(self.path), {"openai": {"state": "ok"}})
        self.assertFalse(self.path.with_suffix(".tmp").exists())

    def test_a_missing_or_damaged_file_starts_empty(self):
        self.assertEqual(state_file.read(self.path), {})
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{not json")
        self.assertEqual(state_file.read(self.path), {})
        self.path.write_text('["a list"]')
        self.assertEqual(state_file.read(self.path), {})
        self.path.write_text('{"openai": "not a state", "anthropic": {"state": "ok"}}')
        self.assertEqual(state_file.read(self.path), {"anthropic": {"state": "ok"}})
