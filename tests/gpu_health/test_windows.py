"""Stable and idle windows found in synthetic runs of readings."""

import unittest

from ai_lab.gpu_health.analysis.windows import IdleFinder, StableFinder, day_of

from .fakes import sample, settings


def feed(finder, rows):
    return [window for window in map(finder.feed, rows) if window is not None]


def loaded(start, count, power=150.0, step=15.0, **values):
    return [sample(start + index * step, power_w=power, temp_gpu=60.0, fan_pct=50.0,
                   util_pct=90.0, **values) for index in range(count)]


class Stable(unittest.TestCase):
    def test_ten_minutes_at_even_power_is_one_window_capped_at_ten_minutes(self):
        windows = feed(StableFinder(settings()), loaded(0, 60))
        self.assertEqual(len(windows), 1)
        window = windows[0]
        self.assertEqual((window["start"], window["end"]), (0, 600))
        self.assertEqual((window["temp_avg"], window["power_avg"], window["fan_avg"]),
                         (60.0, 150.0, 50.0))
        self.assertEqual(window["fan_std"], 0.0)
        self.assertIsNone(window["hotspot_delta"])

    def test_under_five_minutes_is_nothing(self):
        rows = loaded(0, 19) + [sample(19 * 15.0)]
        self.assertEqual(feed(StableFinder(settings()), rows), [])

    def test_power_leaving_the_band_ends_the_window(self):
        rows = loaded(0, 25) + loaded(375, 5, power=200.0) + [sample(450.0)]
        windows = feed(StableFinder(settings()), rows)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["end"], 360)

    def test_a_gap_ends_the_window(self):
        rows = loaded(0, 21) + loaded(21 * 15 + 120, 5)
        windows = feed(StableFinder(settings()), rows)
        self.assertEqual([window["end"] for window in windows], [300])

    def test_another_model_ends_the_window(self):
        rows = loaded(0, 21, model="a") + loaded(315, 5, model="b")
        windows = feed(StableFinder(settings()), rows)
        self.assertEqual(windows[0]["model"], "a")

    def test_below_100_watts_is_not_load(self):
        self.assertEqual(feed(StableFinder(settings()), loaded(0, 60, power=90.0)), [])

    def test_hotspot_above_the_chip_when_known(self):
        rows = loaded(0, 41, temp_hotspot=75.0)
        self.assertEqual(feed(StableFinder(settings()), rows)[0]["hotspot_delta"], 15.0)


class Idle(unittest.TestCase):
    def test_ten_quiet_minutes_give_their_median(self):
        rows = [sample(index * 15.0, temp_gpu=28.0 + index % 3) for index in range(41)]
        rows.append(sample(41 * 15.0, power_w=150.0))
        windows = feed(IdleFinder(settings()), rows)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["temp_median"], 29.0)
        self.assertEqual(windows[0]["day"], day_of(0))

    def test_a_long_quiet_stretch_is_cut_every_hour(self):
        rows = [sample(index * 15.0) for index in range(4 * 60 * 3 + 10)]
        self.assertEqual(len(feed(IdleFinder(settings()), rows)), 3)

    def test_use_or_a_gap_breaks_it(self):
        rows = [sample(index * 15.0) for index in range(30)]
        rows += [sample(30 * 15.0, util_pct=40.0)]
        rows += [sample(1000.0 + index * 15.0) for index in range(30)]
        rows += [sample(5000.0)]
        self.assertEqual(feed(IdleFinder(settings()), rows), [])

    def test_without_a_temperature_it_is_not_idle(self):
        rows = [sample(index * 15.0, temp_gpu=None) for index in range(60)]
        self.assertEqual(feed(IdleFinder(settings()), rows), [])

    def test_use_unknown_counts_as_quiet(self):
        rows = [sample(index * 15.0, util_pct=None) for index in range(41)]
        rows.append(sample(41 * 15.0, power_w=150.0))
        self.assertEqual(len(feed(IdleFinder(settings()), rows)), 1)
