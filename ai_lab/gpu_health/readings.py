"""One reading of the card: the fast values every 15 seconds, the slow ones every hour.

A value this card does not have is `None` here (NVML's "not supported"), and
the names of such values are collected once, so the page can say which ones
are N/A instead of showing an empty column for ever.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .nvml import (
    ECC_AGGREGATE,
    ECC_CORRECTED,
    ECC_UNCORRECTED,
    ECC_VOLATILE,
    NA,
    Nvml,
)

# The fast readings: the sample's field, and the NVML call that gives it.
FAST = (
    ("temp_gpu", "temperature"),
    ("temp_mem", "memory_temperature"),
    ("power_w", "power_draw_w"),
    ("power_limit_w", "power_limit_w"),
    ("fan_pct", "fan_percent"),
    ("clock_sm", "clock_sm_mhz"),
    ("clock_mem", "clock_mem_mhz"),
    ("util_pct", "utilization_percent"),
    ("throttle", "throttle_mask"),
    ("pcie_gen", "pcie_generation"),
    ("pcie_width", "pcie_width"),
    ("pcie_replays", "pcie_replays"),
)
# This card has no hotspot reading in NVML; kept as a column so a card that
# has one needs no new table.
NOT_IN_NVML = ("temp_hotspot",)


@dataclass
class Sample:
    """One fast reading, plus what AI-Lab was doing with the card at that moment."""

    ts: float
    temp_gpu: float | None = None
    temp_mem: float | None = None
    temp_hotspot: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    fan_pct: float | None = None
    clock_sm: float | None = None
    clock_mem: float | None = None
    util_pct: float | None = None
    throttle: int | None = None
    pcie_gen: int | None = None
    pcie_width: int | None = None
    pcie_replays: int | None = None
    model: str = ""
    running: bool = False

    def row(self) -> dict:
        """The sample as the database stores it."""
        values = asdict(self)
        values["running"] = int(self.running)
        return values


@dataclass
class SlowReading:
    """The hourly values: memory errors, replaced memory rows, the driver."""

    ecc: dict
    rows: dict | None
    driver: str


def fast(nvml: Nvml, ts: float, missing: set[str]) -> Sample:
    """Read the fast values; adds the names of N/A ones to `missing`."""
    sample = Sample(ts=ts)
    for name, call in FAST:
        value = getattr(nvml, call)()
        if value is NA:
            missing.add(name)
        else:
            setattr(sample, name, value)
    missing.update(NOT_IN_NVML)
    return sample


def slow(nvml: Nvml) -> SlowReading:
    """Read the hourly values; a counter the card does not keep is left out."""
    ecc = {}
    for kind, kind_name in ((ECC_CORRECTED, "corrected"), (ECC_UNCORRECTED, "uncorrected")):
        for counter, counter_name in ((ECC_VOLATILE, "volatile"), (ECC_AGGREGATE, "aggregate")):
            value = nvml.ecc_errors(kind, counter)
            if value is not NA:
                ecc[f"{kind_name}_{counter_name}"] = value
    rows = nvml.remapped_rows()
    return SlowReading(ecc=ecc, rows=None if rows is NA else rows,
                       driver=nvml.driver_version())
