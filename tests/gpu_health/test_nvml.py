"""The NVML binding: values in the right units, "not supported" as NA, other failures raised."""

import unittest
from unittest import mock

from ai_lab.gpu_health.nvml import (
    ECC_AGGREGATE,
    ECC_UNCORRECTED,
    NA,
    Nvml,
    NvmlError,
    throttle_names,
)

from .fakes import FakeLibrary


def opened(values=None, codes=None) -> Nvml:
    nvml = Nvml(FakeLibrary(values, codes))
    nvml.open()
    return nvml


class Readings(unittest.TestCase):
    def test_the_card_as_measured_on_ct102(self):
        nvml = opened()
        self.assertEqual(nvml.temperature(), 28)
        self.assertEqual(nvml.power_draw_w(), 9.6)
        self.assertEqual(nvml.power_limit_w(), 200.0)
        self.assertEqual(nvml.fan_percent(), 30)
        self.assertEqual((nvml.clock_sm_mhz(), nvml.clock_mem_mhz()), (180, 405))
        self.assertEqual(nvml.utilization_percent(), 0)
        self.assertEqual(nvml.throttle_mask(), 0)
        self.assertEqual((nvml.pcie_generation(), nvml.pcie_width()), (1, 4))
        self.assertEqual(nvml.pcie_replays(), 0)
        self.assertEqual(nvml.ecc_errors(ECC_UNCORRECTED, ECC_AGGREGATE), 0)
        self.assertEqual(nvml.remapped_rows(), {"correctable": 0, "uncorrectable": 0,
                                                "pending": 0, "failure": 0})
        self.assertEqual(nvml.driver_version(), "610.57.04")

    def test_what_the_card_does_not_have_is_na(self):
        nvml = opened({"nvmlDeviceGetFanSpeed": None, "util": None, "rows": None})
        self.assertIs(nvml.memory_temperature(), NA)
        self.assertIs(nvml.fan_percent(), NA)
        self.assertIs(nvml.utilization_percent(), NA)
        self.assertIs(nvml.remapped_rows(), NA)
        self.assertEqual(repr(NA), "NA")

    def test_a_memory_temperature_when_there_is_one(self):
        self.assertEqual(opened({"memory_temp": 61}).memory_temperature(), 61)

    def test_power_na_stays_na(self):
        self.assertIs(opened({"nvmlDeviceGetPowerUsage": None}).power_draw_w(), NA)

    def test_another_failure_is_raised_with_its_code(self):
        nvml = opened(codes={"nvmlDeviceGetTemperature": 15})
        with self.assertRaises(NvmlError) as caught:
            nvml.temperature()
        self.assertEqual(caught.exception.code, 15)

    def test_a_driver_that_does_not_start(self):
        with self.assertRaises(NvmlError):
            Nvml(FakeLibrary(codes={"nvmlInit_v2": 9})).open()

    def test_close_only_after_open(self):
        library = FakeLibrary()
        nvml = Nvml(library)
        nvml.close()
        self.assertNotIn("nvmlShutdown", library.calls)
        nvml.open()
        nvml.close()
        self.assertIn("nvmlShutdown", library.calls)

    def test_older_drivers_name_it_throttle_reasons(self):
        class Old:
            def __init__(self):
                self.inner = FakeLibrary()

            def __getattr__(self, name):
                if name == "nvmlDeviceGetCurrentClocksEventReasons":
                    raise AttributeError(name)
                return getattr(self.inner, name)

            def nvmlDeviceGetCurrentClocksThrottleReasons(self, device, out):  # noqa: N802 - NVML's own name
                out._obj.value = 0x4
                return 0

        nvml = Nvml(Old())
        nvml.open()
        self.assertEqual(nvml.throttle_mask(), 0x4)

    def test_load_fails_without_a_driver(self):
        with mock.patch("ctypes.CDLL", side_effect=OSError("no library")), \
                self.assertRaises(OSError):
            Nvml.load()

    def test_load_opens_the_library(self):
        with mock.patch("ctypes.CDLL", return_value=FakeLibrary()) as cdll:
            Nvml.load().open()
        cdll.assert_called_once_with("libnvidia-ml.so.1")

    def test_throttle_names(self):
        self.assertEqual(throttle_names(0x4 | 0x40), ["sw_power_cap", "hw_thermal_slowdown"])
