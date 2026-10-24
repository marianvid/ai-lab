"""Resting temperature, R, the reference and the trend alarms, on synthetic windows."""

import unittest

from ai_lab.gpu_health.analysis import trend

from .fakes import settings


def window(day, **values):
    """A stable window: 60 °C at 150 W, fan 50 % unless told otherwise."""
    names = {"temp": "temp_avg", "power": "power_avg", "fan": "fan_avg", "hotspot": "hotspot_delta"}
    found = {"day": day, "temp_avg": 60.0, "power_avg": 150.0, "fan_avg": 50.0, "fan_std": 1.0,
             "model": "", "hotspot_delta": None, "start": 0}
    found.update({names.get(key, key): value for key, value in values.items()})
    return found


def with_r(windows, idle=30.0):
    """Windows with R, as if each day rested at `idle` °C."""
    return [{**item, "idle": idle, "r": (item["temp_avg"] - idle) / item["power_avg"]}
            for item in windows]


def days(first, count):
    return [trend.shift(first, index) for index in range(count)]


class IdleReference(unittest.TestCase):
    IDLE = [{"day": "2026-10-01", "temp_median": 30.0}, {"day": "2026-10-01", "temp_median": 32.0},
            {"day": "2026-10-03", "temp_median": 29.0}]

    def test_the_days_median(self):
        self.assertEqual(trend.idle_reference(self.IDLE, "2026-10-01", 3), (31.0, "2026-10-01"))

    def test_falls_back_to_the_latest_day_within_three(self):
        self.assertEqual(trend.idle_reference(self.IDLE, "2026-10-06", 3), (29.0, "2026-10-03"))

    def test_no_reference_after_that(self):
        self.assertIsNone(trend.idle_reference(self.IDLE, "2026-10-07", 3))


class Resistance(unittest.TestCase):
    def test_r_is_temperature_above_rest_per_watt(self):
        idle = [{"day": "2026-10-01", "temp_median": 30.0}]
        found = trend.with_r([window("2026-10-01"), window("2026-10-09"),
                              {**window("2026-10-01"), "power_avg": 0}], idle, settings())
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0]["r"], 0.2)

    def test_fan_bands(self):
        bands = trend.fan_bands([window("d", power=140, fan=40), window("d", power=160, fan=50),
                                 window("d", power=170, fan=60), window("d", fan=None)], 25)
        self.assertEqual(bands, {"125": 40, "150": 55.0})


class Baseline(unittest.TestCase):
    def test_formed_after_twenty_one_days_with_ten_windows(self):
        windows = with_r([window(day) for day in days("2026-10-01", 30)])
        self.assertIsNone(trend.form_baseline("global", windows, "2026-10-21", 1.0, settings()))
        base = trend.form_baseline("global", windows, "2026-10-22", 1.0, settings())
        assert base is not None
        self.assertEqual((base["windows"], base["period_start"], base["period_end"]),
                         (21, "2026-10-01", "2026-10-21"))
        self.assertAlmostEqual(base["r"], 0.2)

    def test_stretched_until_it_has_ten_windows(self):
        windows = with_r([window(day) for day in days("2026-10-01", 30)[::3]])
        base = trend.form_baseline("global", windows, "2026-11-30", 1.0, settings())
        assert base is not None
        self.assertEqual((base["windows"], base["period_end"]), (10, "2026-10-28"))

    def test_not_formed_with_too_few(self):
        windows = with_r([window("2026-10-01")] * 3)
        self.assertIsNone(trend.form_baseline("global", windows, "2026-12-01", 1.0, settings()))
        self.assertIsNone(trend.form_baseline("global", [], "2026-12-01", 1.0, settings()))


class Compare(unittest.TestCase):
    BASE = {"r": 0.2, "fan_std": 1.0, "fan_bands": {"150": 50.0}}

    def keyed(self, recent, base=None):
        found = trend.compare("global", base or self.BASE, with_r(recent), settings())
        return {item.key: item.level for item in found}

    def test_no_change_no_alarm(self):
        self.assertEqual(self.keyed([window("d")]), {})

    def test_r_fifteen_percent_up_is_a_warning_twenty_five_an_alarm(self):
        self.assertEqual(self.keyed([window("d", temp=65.0)]), {"r_rise:global": "warning"})
        self.assertEqual(self.keyed([window("d", temp=68.0)]), {"r_rise:global": "alarm"})

    def test_fan_at_the_same_power_and_its_spread(self):
        found = self.keyed([window("d", fan=66.0, fan_std=3.5)])
        self.assertEqual(found, {"fan_rise:global:150": "warning", "fan_unsteady:global": "warning"})

    def test_nothing_to_compare(self):
        self.assertEqual(trend.compare("m", {"r": None}, with_r([window("d")]), settings()), [])
        self.assertEqual(trend.compare("m", self.BASE, [], settings()), [])

    def test_a_models_alarm_names_it(self):
        found = trend.compare("qwen", self.BASE, with_r([window("d", temp=70.0)]), settings())
        self.assertIn("(qwen)", found[0].message)


class Other(unittest.TestCase):
    def test_hotspot_far_above_the_chip(self):
        self.assertEqual(trend.hotspot([window("d", hotspot=25.0)], settings())[0].key,
                         "hotspot_delta")
        self.assertEqual(trend.hotspot([window("d", hotspot=10.0), window("d")], settings()), [])

    def test_corrected_memory_errors_in_a_week(self):
        readings: list[dict] = [{"ecc": {"corrected_aggregate": 3}}, {"ecc": {}},
                    {"ecc": {"corrected_aggregate": 14}}]
        self.assertEqual(trend.corrected_ecc(readings, settings())[0].detail["rise"], 11)
        self.assertEqual(trend.corrected_ecc(readings[:1], settings()), [])

    def test_resting_temperature_rising_over_thirty_days(self):
        rising = [(day, 28.0 + index * 0.1) for index, day in enumerate(days("2026-09-01", 30))]
        found = trend.idle_rise(rising, settings())
        self.assertEqual((found[0].key, found[0].level), ("idle_rise", "info"))
        flat = [(day, 28.0) for day, _ in rising]
        self.assertEqual(trend.idle_rise(flat, settings()), [])
        self.assertEqual(trend.idle_rise(rising[:5], settings()), [])
