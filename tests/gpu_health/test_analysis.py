"""The analysis over a stored month: references formed, trend alarms, a reset by hand."""

import time
import unittest

from ai_lab.gpu_health.analysis import trend
from ai_lab.gpu_health.analysis.references import Analysis

from .fakes import settings, temporary_store


def noon(day: str) -> float:
    return time.mktime(time.strptime(day + " 12:00", "%Y-%m-%d %H:%M"))


class Month(unittest.TestCase):
    """22 days of one window a day for model "a", resting at 30 °C, R 0.2 °C/W."""

    def setUp(self):
        self.store = temporary_store(self)
        self.analysis = Analysis(self.store, settings())
        for index in range(22):
            self.add_day(trend.shift("2026-10-01", index))

    def add_day(self, day, temp=60.0):
        self.store.add_idle_window(noon(day) - 7200, noon(day) - 6000, day, 30.0)
        self.store.add_window({"start": noon(day), "end": noon(day) + 300, "day": day,
                               "model": "a", "temp_avg": temp, "power_avg": 150.0,
                               "fan_avg": 50.0, "fan_std": 1.0})

    def test_the_references_form_once_and_stay(self):
        self.assertEqual(self.analysis.run(noon("2026-10-22")), [])
        baselines = self.store.baselines()
        self.assertEqual(set(baselines), {"global", "a"})
        self.assertAlmostEqual(baselines["global"]["r"], 0.2)
        self.assertEqual(baselines["global"]["period_end"], "2026-10-21")
        self.assertEqual(self.store.events()[0]["kind"], "baseline")
        self.analysis.run(noon("2026-10-22") + 300)
        self.assertEqual(len([e for e in self.store.events() if e["kind"] == "baseline"]), 2)

    def test_a_warmer_fortnight_raises_r(self):
        self.analysis.run(noon("2026-10-22"))
        for index in range(14):
            self.add_day(trend.shift("2026-10-23", index), temp=70.0)
        found = {item.key: item.level for item in self.analysis.run(noon("2026-11-05"))}
        self.assertEqual(found, {"r_rise:global": "alarm", "r_rise:a": "alarm"})

    def test_the_overview(self):
        self.analysis.run(noon("2026-10-22"))
        overview = self.analysis.overview(noon("2026-10-22"))
        self.assertEqual(overview["idle_today"], {"temp": 30.0, "from_day": "2026-10-22"})
        self.assertAlmostEqual(overview["trend"]["a"]["r"], 0.2)
        self.assertEqual(len(overview["daily_idle"]), 22)

    def test_a_reset_needs_a_note_and_counts_only_what_follows(self):
        self.analysis.run(noon("2026-10-22"))
        with self.assertRaises(ValueError):
            self.analysis.reset_baselines(noon("2026-10-22"), " ")
        self.analysis.reset_baselines(noon("2026-10-22"), "new thermal paste")
        self.assertEqual(self.store.baselines(), {})
        self.analysis.run(noon("2026-10-22") + 60)
        self.assertEqual(self.store.baselines(), {})
        for index in range(22):
            self.add_day(trend.shift("2026-10-23", index), temp=55.0)
        self.analysis.run(noon("2026-11-14"))
        base = self.store.baselines()["global"]
        self.assertAlmostEqual(base["r"], 25 / 150)
        self.assertEqual(base["note"], "after a reset: new thermal paste")
