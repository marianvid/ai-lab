import sys
import unittest

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

    def test_stopping_something_that_is_not_running_is_harmless(self):
        self.host.stop("never-started")


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
        self.assertEqual(os.getpgid(pid), os.getpgid(0))
