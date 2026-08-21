"""Putting back what was on the card.

One model at a time is the rule, and until now what came up after a reboot was
whatever systemd had been told by hand months earlier — two units enabled in
August and nothing in the application able to say otherwise. That flag is gone.
What is on the card is remembered as it changes and put back on startup.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.lastloaded import LastLoaded


class RememberingTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.memory = LastLoaded(Path(self._temporary.name))

    def test_nothing_is_remembered_to_begin_with(self):
        self.assertIsNone(self.memory.read())

    def test_a_model_and_how_it_was_started(self):
        # Both halves matter. A request can ask for a model with a bigger
        # context than the entry is configured for, and bringing back the same
        # model set up differently would be a poor restore.
        self.memory.remember("coder-fast", {"context_size": 98304})
        self.assertEqual(self.memory.read(),
                         {"instance_id": "coder-fast",
                          "settings": {"context_size": 98304}})

    def test_an_empty_card_is_remembered_as_firmly_as_a_model(self):
        # Somebody who empties the card and then reboots does not want the
        # model back.
        self.memory.remember("coder-fast")
        self.memory.forget("coder-fast")
        self.assertIsNone(self.memory.read())

    def test_forgetting_something_else_leaves_it_alone(self):
        # A stray engine is unloaded from beside the model that stays, and that
        # must not read as the card having been emptied.
        self.memory.remember("coder-fast")
        self.memory.forget("gemma-general")
        self.assertEqual(self.memory.read()["instance_id"], "coder-fast")

    def test_forgetting_without_naming_anything_clears_it(self):
        self.memory.remember("coder-fast")
        self.memory.forget()
        self.assertIsNone(self.memory.read())

    def test_the_newest_answer_wins(self):
        self.memory.remember("a", {"context_size": 1024})
        self.memory.remember("b", {"context_size": 2048})
        self.assertEqual(self.memory.read()["instance_id"], "b")


class BrokenFileTests(unittest.TestCase):
    """A memory that cannot be read costs a model not coming back by itself.

    Refusing to start over it would cost far more, so every one of these is a
    quiet None rather than an exception.
    """

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.directory = Path(self._temporary.name)
        self.memory = LastLoaded(self.directory)

    def write(self, text):
        (self.directory / "last-loaded.json").write_text(text)

    def test_nonsense_is_no_memory(self):
        self.write("{not json at all")
        self.assertIsNone(self.memory.read())

    def test_a_file_with_no_model_in_it_is_no_memory(self):
        self.write(json.dumps({"settings": {"context_size": 1}}))
        self.assertIsNone(self.memory.read())

    def test_settings_of_the_wrong_shape_are_dropped_not_obeyed(self):
        self.write(json.dumps({"instance_id": "a", "settings": "lots"}))
        self.assertEqual(self.memory.read(), {"instance_id": "a", "settings": {}})

    def test_a_directory_that_cannot_be_written_does_not_raise(self):
        memory = LastLoaded(Path("/proc/nowhere-at-all"))
        memory.remember("a")            # must not raise
        self.assertIsNone(memory.read())


if __name__ == "__main__":
    unittest.main()
