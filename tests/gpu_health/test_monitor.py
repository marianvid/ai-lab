"""The schedule: each step when it is due, failures counted, the thread never dies."""

import time
import unittest
from unittest import mock

from ai_lab.gpu_health.alarms import AlarmBook
from ai_lab.gpu_health.monitor import GpuHealthMonitor
from ai_lab.gpu_health.nvml import Nvml
from ai_lab.gpu_health.steps import Steps

from .fakes import FakeLibrary, settings, temporary_store


class Schedule(unittest.TestCase):
    def setUp(self):
        self.store = temporary_store(self)
        self.library = FakeLibrary()
        self.sent: list[dict] = []
        book = AlarmBook(self.store, settings(), self.sent.append)
        steps = Steps(settings(sample_s=0.01), self.store, Nvml(self.library),
                      lambda: ("", False), book)
        self.monitor = GpuHealthMonitor(steps, self.sent.append)

    def test_fast_every_turn_slow_every_hour(self):
        for ts in range(0, 3600 + 1, 15):
            self.monitor.tick(1000.0 + ts)
        self.assertEqual(self.store.count("samples"), 241)
        self.assertEqual(len(self.store.slow_readings(0)), 2)

    def test_a_failure_is_counted_and_the_next_turn_still_runs(self):
        self.library.codes["nvmlDeviceGetTemperature"] = 15
        self.monitor.tick(0.0)
        self.monitor.tick(15.0)
        self.assertEqual(self.monitor.failures["fast"]["count"], 2)
        self.assertIn("NvmlError", self.monitor.failures["fast"]["error"])
        del self.library.codes["nvmlDeviceGetTemperature"]
        self.monitor.tick(30.0)
        self.assertEqual(self.monitor.failures["fast"]["recovered"], 30.0)

    def test_the_weekly_report_goes_out_after_a_week(self):
        self.monitor.tick(0.0)
        self.monitor.tick(7 * 86400.0)
        reports = [event for event in self.sent if event["kind"] == "report"]
        self.assertEqual(len(reports), 1)
        self.assertIn("GPU health, last 7 days", reports[0]["message"])
        self.assertEqual(self.store.events_after(0, 10)[-1]["kind"], "report")

    def test_a_failing_database_in_the_daily_step_is_counted(self):
        with mock.patch.object(self.store, "set_state", side_effect=OSError("disk full")):
            self.monitor.tick(0.0)
        self.assertIn("daily", self.monitor.failures)
        self.monitor.tick(15.0)
        self.assertEqual(self.store.count("samples"), 2)

    def test_the_thread_outlives_a_failure_outside_the_steps(self):
        with mock.patch.object(self.monitor, "tick", side_effect=RuntimeError("boom")):
            self.monitor.start()
            deadline = time.time() + 5
            while "tick" not in self.monitor.failures and time.time() < deadline:
                time.sleep(0.01)
            self.assertTrue(self.monitor.running)
            self.monitor.stop()
        self.assertGreaterEqual(self.monitor.failures["tick"]["count"], 1)

    def test_the_thread_runs_and_stops(self):
        self.monitor.start()
        deadline = time.time() + 5
        while self.store.count("samples") < 2 and time.time() < deadline:
            time.sleep(0.01)
        self.assertTrue(self.monitor.running)
        self.monitor.stop()
        self.assertFalse(self.monitor.running)
        self.assertGreaterEqual(self.store.count("samples"), 2)
        self.assertIn("nvmlShutdown", self.library.calls)
