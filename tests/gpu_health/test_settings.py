"""The gpu_health section: defaults are the requirement's values; mistakes are refused."""

import unittest
from typing import Any

from dataclasses import fields

from ai_lab.gpu_health.settings import BOUNDS, CRITICAL_XIDS, GpuHealthSettings


class Settings(unittest.TestCase):
    def test_the_defaults_are_the_requirement(self):
        settings = GpuHealthSettings.from_mapping(None)
        self.assertEqual((settings.sample_s, settings.slow_s), (15.0, 3600.0))
        self.assertEqual((settings.stable_min_s, settings.stable_band, settings.stable_min_w),
                         (300.0, 0.10, 100.0))
        self.assertEqual((settings.idle_max_w, settings.idle_max_util, settings.idle_min_s),
                         (25.0, 5.0, 600.0))
        self.assertEqual((settings.baseline_days, settings.baseline_min_windows,
                          settings.trend_days), (21, 10, 14))
        self.assertEqual((settings.r_warn, settings.r_alarm), (0.15, 0.25))
        self.assertEqual((settings.temp_gpu_c, settings.temp_mem_c), (87.0, 95.0))
        self.assertEqual(settings.repeat_s, 6 * 3600)
        self.assertEqual(settings.xid_critical, CRITICAL_XIDS)
        self.assertEqual(settings.channel, "log")

    def test_a_change_is_kept_with_its_type(self):
        settings = GpuHealthSettings.from_mapping({"sample_s": 30, "baseline_days": 28.0,
                                                   "enabled": False, "xid_critical": [79],
                                                   "helper_events_path": "/x"})
        self.assertEqual(settings.sample_s, 30.0)
        self.assertIsInstance(settings.baseline_days, int)
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.xid_critical, (79,))
        self.assertEqual(settings.json()["xid_critical"], [79])

    def test_mistakes_are_refused(self):
        mistakes: list[Any] = [{"sample": 15}, {"sample_s": -1}, {"sample_s": "15"}, {"sample_s": True},
                    {"enabled": 1}, {"xid_critical": [1.5]}, {"xid_critical": 79},
                    {"helper_events_path": 3}, {"channel": "telegram"}, ["sample_s"]]
        for raw in mistakes:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                GpuHealthSettings.from_mapping(raw)

    def test_a_value_that_breaks_the_monitor_is_refused_at_start(self):
        breaking = {"sample_s": 0, "repeat_s": 0, "critical_repeat_s": 59, "raw_keep_days": 0,
                    "r_alarm": 1.5, "temp_gpu_c": 200}
        for name, value in breaking.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, f"gpu_health {name}"):
                GpuHealthSettings.from_mapping({name: value})

    def test_every_number_has_a_range_that_its_default_is_in(self):
        defaults = GpuHealthSettings()
        numbers = {item.name for item in fields(GpuHealthSettings)
                   if type(getattr(defaults, item.name)) in (int, float)}
        self.assertEqual(set(BOUNDS), numbers)
        for name, (low, high) in BOUNDS.items():
            with self.subTest(name):
                self.assertGreaterEqual(getattr(defaults, name), low)
                if high is not None:
                    self.assertLessEqual(getattr(defaults, name), high)

    def test_a_pair_out_of_order_is_refused(self):
        for raw, words in (({"stable_min_s": 900, "stable_max_s": 10}, "stable_min_s .900."),
                           ({"idle_max_s": 300}, "idle_min_s .600. must be at most"),
                           ({"r_warn": 0.25}, "r_warn .0.25. must be below r_alarm")):
            with self.subTest(raw), self.assertRaisesRegex(ValueError, words):
                GpuHealthSettings.from_mapping(raw)
