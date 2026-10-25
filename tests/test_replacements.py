"""What was taken off the card for what, and when (A5, A7), and the log lines (A9)."""

import unittest
from types import SimpleNamespace

from ai_lab.gateway.admission.replaced import Replacements


class ReplacementsTest(unittest.TestCase):
    def setUp(self):
        self.now = [100.0]
        self.lines: list[str] = []
        self.replacements = Replacements(clock=lambda: self.now[0])
        self.replacements.note = self.lines.append

    def test_a_victim_taken_off_after_a_call_started_names_what_took_its_place(self):
        self.replacements.switched("coder", ["qwen"])
        self.assertEqual(self.replacements.since("qwen", 99.0), "coder")
        self.assertIsNone(self.replacements.since("qwen", 101.0), "a call started after")
        self.assertIsNone(self.replacements.since("coder", 0.0))
        self.assertEqual(self.lines, ["gateway: putting coder on the card, taking off qwen"])

    def test_a_switch_that_takes_nothing_off_says_so(self):
        self.replacements.switched("coder", [])
        self.assertEqual(self.lines, ["gateway: putting coder on the card, taking off nothing"])

    def test_everything_stopped_marks_every_loaded_model(self):
        loaded = [{"shape": SimpleNamespace(instance_id="qwen")},
                  {"shape": SimpleNamespace(instance_id="bge")}]
        self.replacements.stopped("qwen was stopped by hand", loaded)
        self.assertEqual(self.replacements.since("bge", 50.0), "qwen was stopped by hand")
        self.assertIn("everything stopped", self.lines[0])


if __name__ == "__main__":
    unittest.main()
