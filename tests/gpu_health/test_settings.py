"""The gpu_health section: defaults are the requirement's values; mistakes are refused."""

import unittest
from typing import Any

from ai_lab.gpu_health.settings import CRITICAL_XIDS, GpuHealthSettings


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
