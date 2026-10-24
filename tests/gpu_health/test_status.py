"""The status object and the chart's history."""

import unittest
from pathlib import Path

from ai_lab.gpu_health.alarms import AlarmBook
from ai_lab.gpu_health.monitor import GpuHealthMonitor
from ai_lab.gpu_health.nvml import Nvml
from ai_lab.gpu_health.readings import Sample
from ai_lab.gpu_health.output.status import CHART_POINTS, history, status
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
