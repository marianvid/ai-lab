"""The monitor's steps over a fake card: readings stored, windows found, alarms raised."""

import unittest

from ai_lab.gpu_health.alarms import AlarmBook
from ai_lab.gpu_health.nvml import Nvml
from ai_lab.gpu_health.steps import Steps

from .fakes import FakeLibrary, settings, temporary_store


class Card(unittest.TestCase):
    def setUp(self):
        self.store = temporary_store(self)
        self.library = FakeLibrary()
        self.sent: list[dict] = []
        self.activity = ("qwen", True)
        book = AlarmBook(self.store, settings(), self.sent.append)
        self.steps = Steps(settings(), self.store, Nvml(self.library), lambda: self.activity, book)

    def test_a_reading_is_stored_with_what_ai_lab_was_doing(self):
        self.steps.fast(100.0)
        row = self.store.last_sample()
        assert row is not None
        self.assertEqual((row["temp_gpu"], row["model"], row["running"]), (28, "qwen", 1))
        self.assertEqual(self.steps.missing, {"temp_mem", "temp_hotspot"})
        self.assertEqual(self.steps.last_ok, 100.0)

    def test_ten_minutes_of_load_make_a_window_and_rest_an_idle_window(self):
        self.library.values.update({"nvmlDeviceGetPowerUsage": 150000, "util": 90,
                                    "nvmlDeviceGetTemperature": 60})
        for index in range(41):
            self.steps.fast(index * 15.0)
        self.library.values.update({"nvmlDeviceGetPowerUsage": 9000, "util": 0,
                                    "nvmlDeviceGetTemperature": 30})
        for index in range(41, 41 + 42):
            self.steps.fast(index * 15.0)
        self.steps.fast(83 * 15.0 + 1000)
        self.assertEqual(self.store.count("windows"), 1)
        self.assertEqual(self.store.count("idle_windows"), 1)

    def test_a_hot_card_raises_an_alarm(self):
        self.library.values["nvmlDeviceGetTemperature"] = 90
        for ts in range(0, 76, 15):
            self.steps.fast(float(ts))
        self.assertEqual([event["key"] for event in self.sent], ["temp_gpu"])

    def test_a_failing_driver_is_reopened_and_silence_noticed(self):
        self.steps.started = 0.0
        self.library.codes["nvmlDeviceGetTemperature"] = 15
        with self.assertRaises(RuntimeError):
            self.steps.fast(0.0)
        self.steps.silence(400.0)
        self.assertEqual(self.sent[-1]["key"], "nvml_silent")
        del self.library.codes["nvmlDeviceGetTemperature"]
        self.steps.fast(410.0)
        self.steps.silence(411.0)
        self.assertEqual(self.sent[-1]["kind"], "resolved")
        self.assertEqual(self.library.calls.count("nvmlInit_v2"), 2)

    def test_the_hourly_reading_compares_with_the_last(self):
        self.steps.slow(0.0)
        self.library.values["ecc"] = {(1, 1): 1}
        self.steps.slow(3600.0)
        self.assertEqual(self.sent[-1]["key"], "ecc_uncorrected")
        self.assertEqual(len(self.store.slow_readings(0)), 2)

    def test_trend_fold_kernel_and_close(self):
        self.steps.trend(0.0)
        self.steps.kernel_log(0.0)
        self.assertEqual(self.steps.fold(1e9), 0)
        self.steps.fast(1.0)
        self.steps.close()
        self.steps.close()
        self.assertEqual(self.library.calls.count("nvmlShutdown"), 1)
