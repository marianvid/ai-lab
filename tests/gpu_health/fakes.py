"""Stand-ins for the tests: an NVML library, a card's readings, a store in a temporary folder."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ai_lab.gpu_health.nvml import NOT_SUPPORTED, Nvml
from ai_lab.gpu_health.settings import GpuHealthSettings
from ai_lab.gpu_health.store import Store

# What the real card answered on CT 102 (2026-10-24), in NVML's own units.
CARD = {
    "nvmlDeviceGetTemperature": 28, "nvmlDeviceGetPowerUsage": 9600,
    "nvmlDeviceGetEnforcedPowerLimit": 200000, "nvmlDeviceGetFanSpeed": 30,
    "nvmlDeviceGetCurrPcieLinkGeneration": 1, "nvmlDeviceGetCurrPcieLinkWidth": 4,
    "nvmlDeviceGetPcieReplayCounter": 0,
}


class FakeLibrary:
    """Answers the NVML calls `Nvml` makes, from `values`; a value of None means "not supported"."""

    def __init__(self, values: dict | None = None, codes: dict | None = None) -> None:
        self.values = {**CARD, "clock_1": 180, "clock_2": 405, "util": 0, "mask": 0, "ecc": 0,
                       "rows": (0, 0, 0, 0), "driver": b"610.57.04", "memory_temp": None,
                       **(values or {})}
        self.codes = codes or {}
        self.calls: list[str] = []

    def _answer(self, name: str, out, value) -> int:
        self.calls.append(name)
        if name in self.codes:
            return self.codes[name]
        if value is None:
            return NOT_SUPPORTED
        out._obj.value = value
        return 0

    def nvmlInit_v2(self):  # noqa: N802 - NVML's own name
        self.calls.append("nvmlInit_v2")
        return self.codes.get("nvmlInit_v2", 0)

    def nvmlShutdown(self):  # noqa: N802 - NVML's own name
        self.calls.append("nvmlShutdown")
        return 0

    def nvmlDeviceGetHandleByIndex_v2(self, index, out):  # noqa: N802 - NVML's own name
        return self.codes.get("nvmlDeviceGetHandleByIndex_v2", 0)

    def __getattr__(self, name):
        if name in CARD:
            return lambda device, *rest: self._answer(name, rest[-1], self.values[name])
        raise AttributeError(name)

    def nvmlDeviceGetClockInfo(self, device, kind, out):  # noqa: N802 - NVML's own name
        return self._answer("clock", out, self.values[f"clock_{kind.value}"])

    def nvmlDeviceGetUtilizationRates(self, device, out):  # noqa: N802 - NVML's own name
        if self.values["util"] is None:
            return NOT_SUPPORTED
        out._obj.gpu = self.values["util"]
        return 0

    def nvmlDeviceGetCurrentClocksEventReasons(self, device, out):  # noqa: N802 - NVML's own name
        return self._answer("mask", out, self.values["mask"])

    def nvmlDeviceGetFieldValues(self, device, count, out):  # noqa: N802 - NVML's own name
        field = out._obj
        if self.values["memory_temp"] is None:
            field.nvmlReturn = NOT_SUPPORTED
        else:
            field.nvmlReturn, field.value = 0, self.values["memory_temp"]
        return 0

    def nvmlDeviceGetTotalEccErrors(self, device, kind, counter, out):  # noqa: N802 - NVML's own name
        value = self.values["ecc"]
        if isinstance(value, dict):
            value = value.get((kind.value, counter.value))
        return self._answer("ecc", out, value)

    def nvmlDeviceGetRemappedRows(self, device, *outs):  # noqa: N802 - NVML's own name
        if self.values["rows"] is None:
            return NOT_SUPPORTED
        for out, value in zip(outs, self.values["rows"], strict=True):
            out._obj.value = value
        return 0

    def nvmlSystemGetDriverVersion(self, text, length):  # noqa: N802 - NVML's own name
        text.value = self.values["driver"]
        return 0


def card(**values) -> Nvml:
    """An `Nvml` over a fake library with these values changed."""
    return Nvml(FakeLibrary(values))


def settings(**changes) -> GpuHealthSettings:
    """The default settings with some changed."""
    return GpuHealthSettings.from_mapping(changes)


def temporary_store(test) -> Store:
    """A store in a folder removed after the test."""
    folder = tempfile.TemporaryDirectory()
    test.addCleanup(folder.cleanup)
    store = Store(Path(folder.name) / "gpu-health.sqlite")
    test.addCleanup(store.close)
    return store


def sample(ts: float, **values) -> dict:
    """A reading as the database keeps it: an idle card unless told otherwise."""
    row = {"ts": ts, "temp_gpu": 30.0, "temp_mem": None, "temp_hotspot": None,
           "power_w": 10.0, "power_limit_w": 200.0, "fan_pct": 30.0, "clock_sm": 180,
           "clock_mem": 405, "util_pct": 0.0, "throttle": 0, "pcie_gen": 1, "pcie_width": 4,
           "pcie_replays": 0, "model": "", "running": 0}
    row.update(values)
    return row
