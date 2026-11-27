import sys
import threading
import time
import unittest
from unittest import mock

from ai_lab.hosts.darwin import DarwinHost
from ai_lab.types import ProcessSpec


@unittest.skipUnless(sys.platform == "darwin", "macOS only")
class DarwinHostTests(unittest.TestCase):
    def setUp(self):
        self.host = DarwinHost()

    def test_capabilities_describe_unified_memory_and_no_boot_supervision(self):
        capabilities = self.host.capabilities()
        self.assertEqual(capabilities.supervisor, "subprocess")
        self.assertEqual(capabilities.accelerator_kind, "metal")
        self.assertNotIn("vllm", capabilities.engines)

    def test_the_accelerator_reports_real_system_memory(self):
        snapshot = self.host.accelerator()
        self.assertTrue(snapshot.available)
        self.assertEqual(snapshot.memory_kind, "unified")
        self.assertGreater(snapshot.memory_total_mb, 1000)
        self.assertIn("Apple", snapshot.name)

    def test_a_process_can_be_started_watched_and_stopped(self):
        self.host.start(ProcessSpec("sleeper", ["/bin/sleep", "30"], {}))
        self.addCleanup(self.host.stop, "sleeper")
        status = self.host.status("sleeper")
        self.assertTrue(status.running)
        self.assertIsNotNone(status.pid)
        self.host.stop("sleeper")
        self.assertFalse(self.host.status("sleeper").running)

    def test_resident_memory_is_reported_while_running(self):
        self.host.start(ProcessSpec("sleeper", ["/bin/sleep", "30"], {}))
        self.addCleanup(self.host.stop, "sleeper")
        self.assertGreater(self.host.accelerator().memory_used_mb, 0)
        self.host.stop("sleeper")
        self.assertEqual(self.host.accelerator().memory_used_mb, 0)

    def test_starting_twice_is_refused(self):
        self.host.start(ProcessSpec("sleeper", ["/bin/sleep", "30"], {}))
        self.addCleanup(self.host.stop, "sleeper")
        with self.assertRaises(RuntimeError):
            self.host.start(ProcessSpec("sleeper", ["/bin/sleep", "30"], {}))

    @mock.patch("ai_lab.hosts.darwin.TERMINATE_GRACE_SECONDS", 2.0)
    def test_a_slow_stop_does_not_hold_up_the_other_engines(self):
        # Audit m6: the wait for an engine to end happens outside the lock.
        deaf = ["/bin/sh", "-c", 'trap "" TERM; exec /bin/sleep 30']
        self.host.start(ProcessSpec("deaf", deaf, {}))
        self.addCleanup(self.host.stop, "deaf")
        time.sleep(0.3)
        stopping = threading.Thread(target=self.host.stop, args=("deaf",))
        stopping.start()
        time.sleep(0.3)
        began = time.monotonic()
        self.host.start(ProcessSpec("other", ["/bin/sleep", "30"], {}))
        self.addCleanup(self.host.stop, "other")
        self.assertLess(time.monotonic() - began, 1.0)
        with self.assertRaises(RuntimeError):
            self.host.start(ProcessSpec("deaf", deaf, {}))
        stopping.join(10)
        self.assertFalse(self.host.status("deaf").running)

    def test_a_start_that_cannot_run_leaves_no_open_log(self):
        # Audit L16: the opened log handle leaked when Popen raised.
        with self.assertRaises(OSError):
            self.host.start(ProcessSpec("missing", ["/no/such/program"], {}))
        self.assertFalse(self.host.status("missing").running)
        self.host.start(ProcessSpec("missing", ["/bin/sleep", "30"], {}))
        self.addCleanup(self.host.stop, "missing")
        self.assertTrue(self.host.status("missing").running)

    def test_stopping_something_that_is_not_running_is_harmless(self):
        self.host.stop("never-started")
        self.assertFalse(self.host.status("never-started").running)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(sys.platform == "darwin", "macOS only")
class ShutdownTests(unittest.TestCase):
    """Engines must not outlive the manager.

    An orphan keeps its port, and the next manager finds a stranger answering
    its health probe.
    """

    def test_stop_all_ends_every_engine(self):
        host = DarwinHost()
        host.start(ProcessSpec("a", ["/bin/sleep", "30"], {}))
        host.start(ProcessSpec("b", ["/bin/sleep", "30"], {}))
        self.assertTrue(host.status("a").running)
        host.stop_all()
        self.assertFalse(host.status("a").running)
        self.assertFalse(host.status("b").running)

    def test_engines_share_the_managers_process_group(self):
        """So they die with it rather than being adopted by init."""
        import os
        host = DarwinHost()
        host.start(ProcessSpec("grouped", ["/bin/sleep", "30"], {}))
        self.addCleanup(host.stop, "grouped")
        pid = host.status("grouped").pid
        assert pid is not None
        self.assertEqual(os.getpgid(pid), os.getpgid(0))
