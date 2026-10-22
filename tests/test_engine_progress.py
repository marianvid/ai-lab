"""Telling a request parked in the engine from an engine that is stuck."""

import unittest

from ai_lab.engines.base import progress_reader
from ai_lab.gateway.progress import EngineProgress, ProgressBook


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Reports:
    """The engine's `live` reports, one per reading; the last one repeats."""

    def __init__(self, *reports):
        self.reports = list(reports)
        self.reads = 0

    def __call__(self):
        self.reads += 1
        return self.reports.pop(0) if len(self.reports) > 1 else self.reports[0]


def busy(work):
    return {"tokens_processed": work, "requests_running": 2, "requests_waiting": 5}


class EngineProgressTests(unittest.TestCase):
    def watch(self, *reports):
        clock = Clock()
        progress = EngineProgress(Reports(*reports), clock, read_every_s=2.0)
        return progress, clock

    def test_an_engine_that_keeps_working_is_never_idle(self):
        progress, clock = self.watch(busy(100), busy(200), busy(300), busy(400))
        for _ in range(4):
            self.assertLess(progress.idle_for(), 1.0)
            clock.now += 2.0

    def test_an_engine_with_work_and_no_progress_grows_idle(self):
        progress, clock = self.watch(busy(100))
        progress.idle_for()
        for _ in range(31):
            clock.now += 2.0
            idle = progress.idle_for()
        self.assertGreaterEqual(idle, 60.0)

    def test_an_engine_with_nothing_to_do_is_not_stuck(self):
        quiet = {"tokens_processed": 100, "requests_running": 0, "requests_waiting": 0}
        progress, clock = self.watch(quiet)
        for _ in range(40):
            clock.now += 2.0
            self.assertLess(progress.idle_for(), 1.0)

    def test_a_report_that_cannot_be_read_is_no_progress(self):
        progress, clock = self.watch(busy(100), {})
        progress.idle_for()
        for _ in range(31):
            clock.now += 2.0
            idle = progress.idle_for()
        self.assertGreaterEqual(idle, 60.0)

    def test_the_engine_is_read_at_most_every_few_seconds(self):
        reports = Reports(busy(100))
        clock = Clock()
        progress = EngineProgress(reports, clock, read_every_s=2.0)
        for _ in range(10):
            progress.idle_for()
            clock.now += 0.1
        self.assertEqual(reports.reads, 1)

    def test_a_watch_nobody_asked_for_a_while_starts_its_clock_again(self):
        # Idle for an hour with nobody asking, then a request arrives: the
        # engine has not processed it yet, which is not an hour of being stuck.
        progress, clock = self.watch(busy(100))
        progress.idle_for()
        clock.now += 3600.0
        self.assertLess(progress.idle_for(), 1.0)


class ProgressBookTests(unittest.TestCase):
    def test_one_watch_per_engine_port_shared_by_every_request(self):
        book = ProgressBook(Clock())
        first = book.of("qwen", 8141, dict)
        self.assertIs(book.of("qwen", 8141, dict), first)
        self.assertIsNot(book.of("qwen", 8142, dict), first)


class ProgressReaderTests(unittest.TestCase):
    def test_an_engine_without_counters_has_no_reader(self):
        self.assertIsNone(progress_reader(object(), 8141))

    def test_the_reader_never_raises(self):
        class Broken:
            def progress(self, port):
                raise OSError("refused")

        read = progress_reader(Broken(), 8141)
        assert read is not None
        self.assertEqual(read(), {})

    def test_the_reader_passes_the_engines_report_on(self):
        class Engine:
            def progress(self, port):
                return {"tokens_processed": port}

        read = progress_reader(Engine(), 8141)
        assert read is not None
        self.assertEqual(read(), {"tokens_processed": 8141})


if __name__ == "__main__":
    unittest.main()
