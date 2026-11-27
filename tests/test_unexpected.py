"""An expected failure gives its default quietly; any other gives it too, and is said once."""

import contextlib
import io
import time
import unittest
from typing import Any, cast

from ai_lab import budget
from ai_lab.builds.update_watch import UpdateWatcher
from ai_lab.engines.base import live_state, progress_reader, withheld
from ai_lab.events import EventBus
from ai_lab.gateway.loading import CardLoader
from ai_lab.gateway.resources import ask_engine
from ai_lab.gateway.state import Shape
from ai_lab.gateway.stats import GatewayCounters
from ai_lab.shared.unexpected import NOTES, Notes


class Said:
    """What the service log was told while the block ran."""

    def __enter__(self):
        NOTES.forget()
        self._err = io.StringIO()
        self._redirect = contextlib.redirect_stderr(self._err)
        self._redirect.__enter__()
        return self

    def __exit__(self, *exc):
        self._redirect.__exit__(*exc)
        self.lines = [line for line in self._err.getvalue().splitlines() if line]
        return False


class Raises:
    """An engine, a host or a probe that fails with `error` whenever asked."""

    def __init__(self, error: BaseException):
        self.error = error

    def __call__(self, *_args, **_kwargs):
        raise self.error

    def live(self, _port):
        raise self.error

    def progress(self, _port):
        raise self.error

    def withholds(self, _params):
        raise self.error

    def accelerator(self):
        raise self.error

    def system_memory(self):
        raise self.error

    def read_pools(self):
        raise self.error


class NotesTests(unittest.TestCase):
    def test_a_place_says_each_kind_of_failure_once(self):
        notes = Notes()
        with Said() as said:
            for _ in range(3):
                notes.note("a reading", KeyError("model"))
            notes.note("a reading", TypeError("x"))
            notes.note("another reading", KeyError("model"))
        self.assertEqual(len(said.lines), 3)
        self.assertIn("unexpected KeyError in a reading: 'model'", said.lines[0])

    def test_forgetting_says_it_again(self):
        notes = Notes()
        with Said() as said:
            notes.note("a reading", KeyError("model"))
            notes.forget()
            notes.note("a reading", KeyError("model"))
        self.assertEqual(len(said.lines), 2)


class EngineQuestionsTests(unittest.TestCase):
    def test_a_failed_call_is_an_empty_answer_and_nothing_is_said(self):
        engine = Raises(OSError("refused"))
        with Said() as said:
            self.assertEqual(live_state(engine, 8000), {})
            read = progress_reader(engine, 8000)
            assert read is not None
            self.assertEqual(read(), {})
            self.assertEqual(withheld(Raises(ValueError("no")), {}), frozenset())
        self.assertEqual(said.lines, [])

    def test_a_bug_in_the_engine_is_an_empty_answer_and_is_said(self):
        engine = Raises(KeyError("num_requests_running"))
        with Said() as said:
            self.assertEqual(live_state(engine, 8000), {})
            read = progress_reader(engine, 8000)
            assert read is not None
            self.assertEqual(read(), {})
            self.assertEqual(withheld(engine, {}), frozenset())
        self.assertEqual(len(said.lines), 3)
        self.assertTrue(all("KeyError" in line for line in said.lines))


class MachineReadingsTests(unittest.TestCase):
    def test_a_probe_that_fails_reads_as_no_answer_quietly(self):
        with Said() as said:
            figures = budget.of(Raises(OSError("nvidia-smi")))
        self.assertEqual(said.lines, [])
        self.assertEqual(figures.available_mb, 0)

    def test_a_bug_in_the_probe_reads_as_no_answer_and_is_said(self):
        with Said() as said:
            figures = budget.of(Raises(AttributeError("memory_total_mb")))
        self.assertEqual(figures.available_mb, 0)
        self.assertEqual(len(said.lines), 2)            # the card, and the machine

    def test_the_card_keeps_its_last_reading_and_a_bug_is_said(self):
        loader = CardLoader(cast(Any, None), cast(Any, Raises(TypeError("pools"))),
                            GatewayCounters(), lambda: {"loaded": []}, lambda _shape: "")
        with Said() as said:
            loader.read_the_card()
            loader.read_the_card()
        self.assertEqual(len(said.lines), 1)
        self.assertIn("TypeError in the card's memory reading", said.lines[0])

    def test_an_entry_gone_meanwhile_gives_the_default_quietly(self):
        class Gone:
            def instance(self, instance_id):
                raise KeyError(instance_id)
        with Said() as said:
            self.assertEqual(ask_engine(Gone(), Shape("old"), lambda *_: 1.0, 0.0), 0.0)
        self.assertEqual(said.lines, [])


class UpdateCheckTests(unittest.TestCase):
    def test_a_bug_in_a_check_is_said_and_the_timer_goes_on(self):
        asked = []

        class Item:
            def check(self):
                asked.append(1)
                raise TypeError("a renamed field")
        watcher = UpdateWatcher(lambda: [Item()], "test-watch")
        with Said() as said:
            watcher.start(interval_s=0.01, startup_delay_s=0.0)
            deadline = time.monotonic() + 3
            while len(asked) < 3 and time.monotonic() < deadline:
                time.sleep(0.01)
            watcher.stop()
            time.sleep(0.05)
        self.assertGreaterEqual(len(asked), 3)
        self.assertEqual(len(said.lines), 1)
        self.assertIn("update check (test-watch)", said.lines[0])


class DroppedEventsTests(unittest.TestCase):
    def test_what_a_slow_page_lost_is_counted_and_said_when_it_leaves(self):
        bus = EventBus(queue_size=2)
        slow = bus.subscribe()
        for number in range(5):
            bus.publish(number)
        self.assertEqual(slow.dropped, 3)
        with Said() as said:
            slow.close()
        self.assertEqual(len(said.lines), 1)
        self.assertIn("lost 3 events", said.lines[0])


if __name__ == "__main__":
    unittest.main()
