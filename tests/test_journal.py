"""The operation log (A9): newest lines on screen, every line in a rotating file."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai_lab.events import EventBus
from ai_lab.runtime.journal import LOG_NAME, OperationJournal, PhaseFollower
from ai_lab.types import Phase


class JournalTest(unittest.TestCase):
    def test_lines_since_an_id_and_on_disk(self):
        folder = Path(tempfile.mkdtemp())
        journal = OperationJournal(folder, clock=lambda: 0.0)
        journal.write("first")
        journal.write("second")
        self.assertEqual([line["text"] for line in journal.since(1)["lines"]], ["second"])
        self.assertEqual(journal.since()["last_id"], 2)
        text = (folder / LOG_NAME).read_text()
        self.assertIn("1970-01-01 00:00:00 UTC first", text)

    def test_the_screen_keeps_only_the_newest(self):
        journal = OperationJournal(None, keep=2)
        for number in range(5):
            journal.write(str(number))
        self.assertEqual([line["text"] for line in journal.since()["lines"]], ["3", "4"])

    def test_one_line_per_step_not_per_progress_event(self):
        journal = OperationJournal(None)
        follower = PhaseFollower(journal)
        for phase in (Phase.STARTING, Phase.STARTING, Phase.WEIGHTS_LOADING):
            follower.take(SimpleNamespace(instance_id="qwen", phase=phase, message=""))
        follower.take(SimpleNamespace(instance_id="qwen", phase=Phase.STOPPING,
                                      message="by hand"))
        follower.take(SimpleNamespace(topic="instances"))
        texts = [line["text"] for line in journal.since()["lines"]]
        self.assertEqual(texts, ["qwen: starting", "qwen: weights loading",
                                 "qwen: stopping — by hand"])

    def test_it_follows_the_bus(self):
        journal = OperationJournal(None)
        bus = EventBus()
        PhaseFollower(journal).follow(bus)
        bus.publish(SimpleNamespace(instance_id="bge", phase=Phase.STARTING, message=""))
        for _ in range(100):
            if journal.since()["lines"]:
                break
            import time
            time.sleep(0.02)
        self.assertEqual(journal.since()["lines"][0]["text"], "bge: starting")


if __name__ == "__main__":
    unittest.main()
