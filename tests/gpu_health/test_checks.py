"""The immediate rules on each reading and each hourly reading."""

import unittest

from ai_lab.gpu_health.analysis.checks import FastChecks, slow_conditions, slow_events

from .fakes import sample, settings


def keys(found):
    return [condition.key for condition in found]


def run(checks, rows):
    """The conditions after the last reading, and every event on the way."""
    conditions, events = [], []
    for row in rows:
        conditions, new = checks.check(row)
        events += new
    return conditions, events


class Temperatures(unittest.TestCase):
    def test_hot_for_more_than_a_minute(self):
        checks = FastChecks(settings())
        rows = [sample(ts, temp_gpu=88.0) for ts in range(0, 76, 15)]
        self.assertEqual(keys(run(checks, rows[:-1])[0]), [])
        self.assertEqual(keys(run(checks, rows[-1:])[0]), ["temp_gpu"])

    def test_cooling_down_starts_the_count_again(self):
        checks = FastChecks(settings())
        rows = [sample(0, temp_gpu=88.0), sample(30), sample(45, temp_gpu=96.0, temp_mem=96.0),
                sample(110, temp_gpu=88.0, temp_mem=96.0)]
        self.assertEqual(keys(run(checks, rows)[0]), ["temp_gpu", "temp_mem"])


class Throttle(unittest.TestCase):
    def test_thermal_slowdown_for_more_than_thirty_seconds(self):
        rows = [sample(ts, throttle=0x40) for ts in (0, 15, 31)]
        conditions = run(FastChecks(settings()), rows)[0]
        self.assertEqual(keys(conditions), ["throttle"])
        self.assertIn("hw_thermal_slowdown", conditions[0].message)

    def test_power_capping_is_normal(self):
        rows = [sample(ts, throttle=0x4 | 0x1) for ts in range(0, 300, 15)]
        self.assertEqual(run(FastChecks(settings()), rows)[0], [])


class Link(unittest.TestCase):
    def busy(self, ts, gen, width=4):
        return sample(ts, power_w=150.0, util_pct=80.0, pcie_gen=gen, pcie_width=width)

    def test_slower_than_the_best_seen_under_load_for_a_minute(self):
        checks = FastChecks(settings())
        rows = [self.busy(0, 4)] + [self.busy(ts, 2) for ts in range(15, 91, 15)]
        conditions = run(checks, rows)[0]
        self.assertEqual(keys(conditions), ["pcie_link"])
        self.assertEqual(conditions[0].detail["normal"], [4, 4])

    def test_gen_1_at_rest_is_normal(self):
        checks = FastChecks(settings())
        rows = [self.busy(0, 4)] + [sample(ts) for ts in range(15, 300, 15)]
        self.assertEqual(run(checks, rows)[0], [])

    def test_the_best_link_survives_a_restart(self):
        checks = FastChecks(settings(), {"best_link": [4, 4]})
        rows = [self.busy(ts, 4, 2) for ts in range(0, 91, 15)]
        self.assertEqual(keys(run(checks, rows)[0]), ["pcie_link"])


class Replays(unittest.TestCase):
    def test_more_than_a_hundred_in_an_hour(self):
        checks = FastChecks(settings())
        rows = [sample(0, pcie_replays=5), sample(1800, pcie_replays=100),
                sample(3000, pcie_replays=106)]
        self.assertEqual(keys(run(checks, rows)[0]), ["pcie_replays"])
        later = run(checks, [sample(3000 + 3700, pcie_replays=110)])[0]
        self.assertEqual(later, [])

    def test_no_counter_no_rule(self):
        self.assertEqual(run(FastChecks(settings()), [sample(0, pcie_replays=None)])[0], [])


class PowerLimit(unittest.TestCase):
    def test_a_change_is_an_event(self):
        rows = [sample(0), sample(15, power_limit_w=180.0), sample(30, power_limit_w=None),
                sample(45, power_limit_w=180.2)]
        events = run(FastChecks(settings()), rows)[1]
        self.assertEqual(keys(events), ["power_limit"])
        self.assertIn("200 W to 180 W", events[0].message)


class Slow(unittest.TestCase):
    BEFORE = {"ecc": {"uncorrected_aggregate": 0, "uncorrected_volatile": 0},
              "rows": {"correctable": 0, "uncorrectable": 0, "pending": 0, "failure": 0},
              "driver": "610.57.04"}

    def test_the_first_reading_has_nothing_to_compare(self):
        self.assertEqual(slow_events(self.BEFORE, None), [])

    def test_new_errors_new_rows_and_a_new_driver(self):
        now = {"ecc": {"uncorrected_aggregate": 1, "uncorrected_volatile": 0},
               "rows": {"correctable": 1, "uncorrectable": 1, "pending": 0, "failure": 0},
               "driver": "611.0"}
        found = slow_events(now, self.BEFORE)
        self.assertEqual(keys(found), ["ecc_uncorrected", "remap_uncorrectable",
                                       "remap_correctable", "driver"])
        self.assertEqual([item.level for item in found], ["critical", "critical", "alarm", "info"])

    def test_pending_and_failure_are_conditions(self):
        reading = {"rows": {"pending": 1, "failure": 1}}
        self.assertEqual(keys(slow_conditions(reading)), ["remap_pending", "remap_failure"])
        self.assertEqual(slow_conditions({"rows": None}), [])
