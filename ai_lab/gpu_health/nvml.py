"""Reading the card through NVML, NVIDIA's management library, and nothing else.

NVML is the library `nvidia-smi` itself reads. It comes with the driver
(`libnvidia-ml.so.1`), so this needs no package: a few functions are called
through `ctypes`. Only reading functions are bound here — nothing that sets a
power limit, a clock, a fan or a mode, and nothing that resets the card — so
the monitor cannot change the card even by mistake. NVML does not create a
context on the card either: the monitor never appears among the card's
processes.

Each reading returns its value, or `NA` when this card does not have it
(NVML's "not supported"). Any other failure raises `NvmlError`: the driver is
gone or not answering, and the monitor counts how long that lasts.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import Any

LIBRARY = "libnvidia-ml.so.1"
SUCCESS = 0
NOT_SUPPORTED = 3
# NVML's names for what a reading is about.
TEMPERATURE_GPU = 0
CLOCK_SM = 1
CLOCK_MEM = 2
ECC_CORRECTED, ECC_UNCORRECTED = 0, 1
ECC_VOLATILE, ECC_AGGREGATE = 0, 1
FIELD_MEMORY_TEMP = 82
DRIVER_TEXT = 96
LOW_32_BITS = 0xFFFFFFFF


class _NotAvailable:
    """A reading this card does not have."""

    def __repr__(self) -> str:
        return "NA"


NA: Any = _NotAvailable()

# What each bit of "clocks event reasons" (formerly "throttle reasons") means.
THROTTLE_BITS = {
    0x1: "gpu_idle",
    0x2: "applications_clocks",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}


class NvmlError(RuntimeError):
    """NVML failed for a reason other than "this card does not have it"."""

    def __init__(self, function: str, code: int) -> None:
        super().__init__(f"{function} failed with NVML code {code}")
        self.code = code


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _FieldValue(ctypes.Structure):
    _fields_ = [("fieldId", ctypes.c_uint), ("scopeId", ctypes.c_uint),
                ("timestamp", ctypes.c_longlong), ("latencyUsec", ctypes.c_longlong),
                ("valueType", ctypes.c_uint), ("nvmlReturn", ctypes.c_uint),
                ("value", ctypes.c_ulonglong)]


def throttle_names(mask: int) -> list[str]:
    """The names of the reasons set in a clocks-event mask."""
    return [name for bit, name in THROTTLE_BITS.items() if mask & bit]


class Nvml:
    """One card through NVML. `load` opens the library; tests pass a stand-in."""

    def __init__(self, library: Any, index: int = 0) -> None:
        self._lib = library
        self._index = index
        self._device = ctypes.c_void_p()
        self._open = False

    @classmethod
    def load(cls, index: int = 0) -> Nvml:
        """The real library; OSError when the driver is not installed."""
        return cls(ctypes.CDLL(LIBRARY), index)

    def open(self) -> None:
        """Start NVML and find the card; raises NvmlError when the driver does not answer."""
        self._check("nvmlInit_v2", self._lib.nvmlInit_v2())
        self._check("nvmlDeviceGetHandleByIndex_v2", self._lib.nvmlDeviceGetHandleByIndex_v2(
            ctypes.c_uint(self._index), ctypes.byref(self._device)))
        self._open = True

    def close(self) -> None:
        """Let NVML go; harmless when it was never opened."""
        if self._open:
            self._open = False
            self._lib.nvmlShutdown()

    # -- the readings ---------------------------------------------------------

    def temperature(self) -> Any:
        """The chip's temperature, °C."""
        return self._uint("nvmlDeviceGetTemperature", ctypes.c_uint(TEMPERATURE_GPU))

    def memory_temperature(self) -> Any:
        """The memory's temperature, °C, through NVML's field values; NA on most cards."""
        field = _FieldValue(fieldId=FIELD_MEMORY_TEMP)
        code = self._lib.nvmlDeviceGetFieldValues(self._device, 1, ctypes.byref(field))
        if code == NOT_SUPPORTED or field.nvmlReturn == NOT_SUPPORTED:
            return NA
        self._check("nvmlDeviceGetFieldValues", code)
        if field.nvmlReturn != SUCCESS:
            return NA
        return int(field.value & LOW_32_BITS)

    def power_draw_w(self) -> Any:
        """What the card draws now, in watts."""
        return _watts(self._uint("nvmlDeviceGetPowerUsage"))

    def power_limit_w(self) -> Any:
        """The power limit in force, in watts."""
        return _watts(self._uint("nvmlDeviceGetEnforcedPowerLimit"))

    def fan_percent(self) -> Any:
        """The fan's speed, as a share of its maximum."""
        return self._uint("nvmlDeviceGetFanSpeed")

    def clock_sm_mhz(self) -> Any:
        """The chip's clock."""
        return self._uint("nvmlDeviceGetClockInfo", ctypes.c_uint(CLOCK_SM))

    def clock_mem_mhz(self) -> Any:
        """The memory's clock."""
        return self._uint("nvmlDeviceGetClockInfo", ctypes.c_uint(CLOCK_MEM))

    def utilization_percent(self) -> Any:
        """How busy the chip was over the last sample period."""
        rates = _Utilization()
        code = self._lib.nvmlDeviceGetUtilizationRates(self._device, ctypes.byref(rates))
        return self._value("nvmlDeviceGetUtilizationRates", code, lambda: int(rates.gpu))

    def throttle_mask(self) -> Any:
        """Why the clocks are held down now, as NVML's bit mask (see `THROTTLE_BITS`)."""
        mask = ctypes.c_ulonglong()
        function = ("nvmlDeviceGetCurrentClocksEventReasons"
                    if hasattr(self._lib, "nvmlDeviceGetCurrentClocksEventReasons")
                    else "nvmlDeviceGetCurrentClocksThrottleReasons")
        code = getattr(self._lib, function)(self._device, ctypes.byref(mask))
        return self._value(function, code, lambda: int(mask.value))

    def pcie_generation(self) -> Any:
        """The PCIe generation the link runs at now (it drops when the card idles)."""
        return self._uint("nvmlDeviceGetCurrPcieLinkGeneration")

    def pcie_width(self) -> Any:
        """The PCIe lanes in use now."""
        return self._uint("nvmlDeviceGetCurrPcieLinkWidth")

    def pcie_replays(self) -> Any:
        """How many PCIe transfers had to be sent again since the driver started."""
        return self._uint("nvmlDeviceGetPcieReplayCounter")

    def ecc_errors(self, kind: int, counter: int) -> Any:
        """Memory errors of one kind (corrected/uncorrected), since boot or ever."""
        count = ctypes.c_ulonglong()
        code = self._lib.nvmlDeviceGetTotalEccErrors(
            self._device, ctypes.c_uint(kind), ctypes.c_uint(counter), ctypes.byref(count))
        return self._value("nvmlDeviceGetTotalEccErrors", code, lambda: int(count.value))

    def remapped_rows(self) -> Any:
        """Memory rows replaced: {correctable, uncorrectable, pending, failure}."""
        values = [ctypes.c_uint() for _ in range(4)]
        code = self._lib.nvmlDeviceGetRemappedRows(self._device, *map(ctypes.byref, values))
        return self._value("nvmlDeviceGetRemappedRows", code, lambda: dict(zip(
            ("correctable", "uncorrectable", "pending", "failure"),
            (int(value.value) for value in values), strict=True)))

    def driver_version(self) -> str:
        """The driver's version, for lining up changes in behaviour with updates."""
        text = ctypes.create_string_buffer(DRIVER_TEXT)
        self._check("nvmlSystemGetDriverVersion",
                    self._lib.nvmlSystemGetDriverVersion(text, ctypes.c_uint(DRIVER_TEXT)))
        return text.value.decode()

    # -- inside ---------------------------------------------------------------

    def _uint(self, function: str, *before: Any) -> Any:
        value = ctypes.c_uint()
        code = getattr(self._lib, function)(self._device, *before, ctypes.byref(value))
        return self._value(function, code, lambda: int(value.value))

    def _value(self, function: str, code: int, read: Callable[[], Any]) -> Any:
        if code == NOT_SUPPORTED:
            return NA
        self._check(function, code)
        return read()

    @staticmethod
    def _check(function: str, code: int) -> None:
        if code != SUCCESS:
            raise NvmlError(function, code)


def _watts(milliwatts: Any) -> Any:
    return NA if milliwatts is NA else round(milliwatts / 1000.0, 2)
