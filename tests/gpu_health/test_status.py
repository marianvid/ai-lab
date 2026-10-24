"""The status object and the chart's history."""

import unittest
from unittest import mock
from pathlib import Path

from ai_lab.gpu_health.alarms import AlarmBook, Condition
from ai_lab.gpu_health.monitor import GpuHealthMonitor
from ai_lab.gpu_health.nvml import Nvml
from ai_lab.gpu_health.readings import Sample
from ai_lab.gpu_health.output.status import CHART_POINTS, alerts, history, status, summary
from ai_lab.gpu_health.steps import Steps

from .fakes import FakeLibrary, settings, temporary_store


class Status(unittest.TestCase):
    def test_everything_the_page_shows(self):
        store = temporary_store(self)
        book = AlarmBook(store, settings(), lambda event: None)
        steps = Steps(settings(), store, Nvml(FakeLibrary()), lambda: ("", False), book)
        monitor = GpuHealthMonitor(steps, lambda event: None)
        monitor.tick(1000.0)
        state = status(monitor, 1001.0, Path("/x/gpu-health.sqlite"))
        self.assertTrue(state["available"])
        self.assertFalse(state["running"])
        self.assertEqual(state["current"]["temp_gpu"], 28)
        self.assertEqual(state["not_available"], ["temp_hotspot", "temp_mem"])
        self.assertEqual(state["slow"]["driver"], "610.57.04")
        self.assertFalse(state["helper"]["installed"])
        self.assertEqual(state["counts"]["samples"], 1)
        self.assertEqual(state["database"], "/x/gpu-health.sqlite")


class Summary(unittest.TestCase):
    def setUp(self):
        self.store = temporary_store(self)
        book = AlarmBook(self.store, settings(), lambda event: None)
        self.steps = Steps(settings(), self.store, Nvml(FakeLibrary()), lambda: ("", False), book)
        self.monitor = GpuHealthMonitor(self.steps, lambda event: None)
        self.monitor.tick(1000.0)

    def test_a_stopped_monitor_is_an_alarm_by_itself(self):
        short = summary(self.monitor, 1010.0)
        self.assertEqual((short["health"], short["monitor"], short["stale"]),
                         ("alarm", "stopped", False))
        self.assertEqual(short["reading_age_s"], 10.0)

    def test_the_worst_alarm_in_force_and_info_does_not_count(self):
        with mock.patch.object(GpuHealthMonitor, "running", new=True):
            self.assertEqual(summary(self.monitor, 1010.0)["health"], "ok")
            self.steps.book.conditions("trend", 1001.0, [Condition("idle_rise", "info", "x")])
            self.assertEqual(summary(self.monitor, 1010.0)["health"], "ok")
            self.steps.book.conditions("fast", 1002.0, [Condition("temp_gpu", "alarm", "hot"),
                                                        Condition("pcie_replays", "warning", "r")])
            short = summary(self.monitor, 1010.0)
            self.assertEqual(short["health"], "alarm")
            self.assertEqual([item["key"] for item in short["alarms"]], ["temp_gpu", "pcie_replays"])
            self.assertEqual(short["last_alert_id"], 3)  # the info line (1) was not sent
            self.assertEqual(summary(self.monitor, 1400.0)["stale"], True)

    def test_alerts_by_cursor(self):
        self.steps.book.event(1.0, Condition("xid:79", "critical", "Xid 79"))
        self.steps.book.event(2.0, Condition("xid:79", "critical", "Xid 79 again"))
        first = alerts(self.store, 0, 1, False)
        self.assertEqual(([item["message"] for item in first["alerts"]], first["more"]),
                         (["Xid 79"], True))
        rest = alerts(self.store, first["last_id"], 100, False)
        self.assertEqual([item["message"] for item in rest["alerts"]], ["Xid 79 again"])
        self.assertEqual(alerts(self.store, rest["last_id"], 100, False),
                         {"alerts": [], "last_id": rest["last_id"], "more": False})
        for after, limit in ((-1, 10), (0, 0), (0, 501)):
            with self.subTest(after=after, limit=limit), self.assertRaises(ValueError):
                alerts(self.store, after, limit, False)


class History(unittest.TestCase):
    def test_raw_and_folded_readings_averaged_into_at_most_360_points(self):
        store = temporary_store(self)
        for index in range(2000):
            store.add_sample(Sample(ts=index * 15.0, temp_gpu=30.0, power_w=None))
        store.fold_old_samples(15000.0)
        result = history(store, 2000 * 15.0, 24)
        self.assertLessEqual(len(result["points"]), CHART_POINTS)
        self.assertEqual({point["temp_gpu"] for point in result["points"]}, {30.0})
        self.assertIsNone(result["points"][0]["power_w"])

    def test_hours_must_be_positive(self):
        with self.assertRaises(ValueError):
            history(temporary_store(self), 0.0, 0)
