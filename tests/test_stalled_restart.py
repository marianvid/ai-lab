"""A stalled engine is restarted once, however many requests notice it."""

import unittest

from ai_lab.gateway.core import Gateway
from ai_lab.gateway.limits import GatewayLimits
from ai_lab.gateway.progress import ProgressBook
from ai_lab.gateway.restart import StalledEngines


class StalledEnginesTests(unittest.TestCase):
    def test_the_first_report_restarts_and_the_rest_wait_for_it(self):
        restarted = []
        pending: list = []
        engines = StalledEngines(lambda instance, reason: restarted.append((instance, reason)),
                                 start=pending.append)
        self.assertTrue(engines.report("qwen", "engine stalled"))
        self.assertFalse(engines.report("qwen", "engine stalled"),
                         "a second request must not restart it again")
        self.assertTrue(engines.report("gemma", "engine stalled"), "another engine is its own")
        for work in pending:
            work()
        self.assertEqual(restarted, [("qwen", "engine stalled"), ("gemma", "engine stalled")])
        self.assertTrue(engines.report("qwen", "engine stalled"),
                        "once restarted, a later stall restarts it again")

    def test_a_restart_that_fails_does_not_block_the_next(self):
        def broken(instance, reason):
            raise RuntimeError("could not stop")

        pending: list = []
        engines = StalledEngines(broken, start=pending.append)
        engines.report("qwen", "engine stalled")
        with self.assertRaises(RuntimeError):
            pending.pop()()
        self.assertTrue(engines.report("qwen", "engine stalled"))

    def test_by_default_the_restart_runs_on_its_own_thread(self):
        import threading
        done = threading.Event()
        engines = StalledEngines(lambda instance, reason: done.set())
        engines.report("qwen", "engine stalled")
        self.assertTrue(done.wait(2.0))


class Operations:
    """Just enough of `CardOperations` for a restart."""

    def __init__(self, fails=False):
        self.unloaded = []
        self.fails = fails

    def unload(self, instance_id):
        if self.fails:
            raise RuntimeError("systemd said no")
        self.unloaded.append(instance_id)


class GatewayRestartTests(unittest.TestCase):
    def gateway(self, operations):
        return Gateway(operations, GatewayLimits(), memory=object())  # type: ignore[arg-type]

    def test_the_stalled_engine_is_stopped_and_its_watch_forgotten(self):
        operations = Operations()
        gateway = self.gateway(operations)
        book: ProgressBook = gateway._watches.book  # noqa: SLF001
        first = book.of("qwen", 8141, dict)
        gateway._restart_stalled("qwen", "engine stalled: it did no work for 61 s")  # noqa: SLF001
        self.assertEqual(operations.unloaded, ["qwen"])
        self.assertIsNot(book.of("qwen", 8141, dict), first, "the counters start again")
        self.assertIn("qwen restarted", gateway._counters.last_error)  # noqa: SLF001

    def test_a_stop_that_fails_is_said_and_the_queue_still_cleared(self):
        gateway = self.gateway(Operations(fails=True))
        gateway._restart_stalled("qwen", "engine stalled")  # noqa: SLF001
        self.assertIn("stopping qwen failed", gateway._counters.last_error)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
