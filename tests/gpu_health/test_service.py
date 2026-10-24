"""Starting the monitor where there is a card, and saying why not where there is none."""

import tempfile
from pathlib import Path
import unittest
from unittest import mock

from ai_lab.gpu_health.nvml import Nvml
from ai_lab.gpu_health.service import GpuHealth, log_channel

from .fakes import FakeLibrary


def no_driver():
    raise OSError("no library")


class Service(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def test_without_a_driver_it_only_says_so(self):
        health = GpuHealth.start({}, self.folder, lambda: ("", False), load=no_driver)
        self.assertEqual(health.status(), {"available": False, "reason": health.reason})
        self.assertIn("No NVIDIA driver", health.reason)
        with self.assertRaises(ValueError):
            health.report(7)
        health.stop()

    def test_switched_off(self):
        health = GpuHealth.start({"enabled": False}, self.folder, lambda: ("", False))
        self.assertIn("Switched off", health.status()["reason"])

    def test_a_bad_section_is_refused(self):
        with self.assertRaises(ValueError):
            GpuHealth.start({"sample": 1}, self.folder, lambda: ("", False))

    def test_with_a_card_every_call_answers(self):
        library = FakeLibrary()
        health = GpuHealth.start({"sample_s": 0.01}, self.folder, lambda: ("", False),
                                 load=lambda: Nvml(library))
        self.assertTrue(health.status()["available"])
        self.assertIn("points", health.history(1))
        self.assertIn("text", health.report(7))
        self.assertEqual(health.test_alarm()["events"][0]["message"],
                         "Test alarm from the GPU health page")
        with self.assertRaises(ValueError):
            health.reset_baseline("")
        self.assertIsNotNone(health.reset_baseline("new paste")["baseline_reset"])
        health.stop()
        assert health.monitor is not None
        self.assertFalse(health.monitor.running)

    def test_the_log_channel_writes_one_line(self):
        with mock.patch("builtins.print") as printed:
            log_channel({"level": "critical", "kind": "event", "message": "Xid 79"})
        printed.assert_called_once_with("GPU health CRITICAL [event] Xid 79", flush=True)
