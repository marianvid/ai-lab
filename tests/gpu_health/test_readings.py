"""One reading: the fast values into a sample, the slow ones into a slow reading."""

import unittest

from ai_lab.gpu_health import readings
from ai_lab.gpu_health.nvml import Nvml

from .fakes import FakeLibrary


def opened(values=None) -> Nvml:
    nvml = Nvml(FakeLibrary(values))
    nvml.open()
    return nvml


class Fast(unittest.TestCase):
    def test_a_sample_and_what_is_missing(self):
        missing: set[str] = set()
        sample = readings.fast(opened(), 100.0, missing)
        self.assertEqual((sample.ts, sample.temp_gpu, sample.power_w), (100.0, 28, 9.6))
        self.assertIsNone(sample.temp_mem)
        self.assertEqual(missing, {"temp_mem", "temp_hotspot"})
        self.assertEqual(sample.row()["running"], 0)

    def test_running_is_stored_as_a_number(self):
        sample = readings.Sample(ts=1.0, running=True)
        self.assertEqual(sample.row()["running"], 1)


class Slow(unittest.TestCase):
    def test_every_counter_and_the_rows(self):
        reading = readings.slow(opened())
        assert reading.rows is not None
        self.assertEqual(set(reading.ecc), {"corrected_volatile", "corrected_aggregate",
                                            "uncorrected_volatile", "uncorrected_aggregate"})
        self.assertEqual(reading.rows["pending"], 0)
        self.assertEqual(reading.driver, "610.57.04")

    def test_a_card_without_ecc_or_rows(self):
        reading = readings.slow(opened({"ecc": None, "rows": None}))
        self.assertEqual(reading.ecc, {})
        self.assertIsNone(reading.rows)

    def test_one_counter_missing(self):
        reading = readings.slow(opened({"ecc": {(0, 0): 1, (0, 1): 2, (1, 1): 0}}))
        self.assertEqual(reading.ecc, {"corrected_volatile": 1, "corrected_aggregate": 2,
                                       "uncorrected_aggregate": 0})
