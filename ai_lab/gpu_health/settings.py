"""The `gpu_health` section of the configuration: every interval and threshold, checked.

Nothing here is in the code as a decision: each number has a default that is
the requirement's own value, and the section may change any of them. An
unknown key is refused, so a typo does not silently leave a default in force.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields

# Xid codes that mean the card or its memory may be failing (NVIDIA's list of
# serious Xids): reported as critical. Every other Xid is a warning.
CRITICAL_XIDS = (48, 63, 64, 74, 79, 92, 94, 95)
# Where alarms can go. Only the log exists: no other channel was set up in
# AI-Lab or home-lab, and the requirement says not to invent one.
CHANNELS = ("log",)


@dataclass(frozen=True)
class GpuHealthSettings:
    """How often the monitor reads, what counts as stable or idle, and when it alarms."""

    enabled: bool = True
    # How often: the fast readings, and the slow ones (memory errors, driver).
    sample_s: float = 15.0
    slow_s: float = 3600.0
    # A stable load window: at least this long, power within ±band of its mean, above min_w.
    stable_min_s: float = 300.0
    stable_max_s: float = 600.0
    stable_band: float = 0.10
    stable_min_w: float = 100.0
    # An idle window, for the day's resting temperature.
    idle_max_w: float = 25.0
    idle_max_util: float = 5.0
    idle_min_s: float = 600.0
    # A long quiet stretch is cut into windows of at most this, so a day that
    # is quiet from morning to night still gives its temperature that day.
    idle_max_s: float = 3600.0
    idle_fallback_days: int = 3
    # The reference (baseline) and the trend.
    baseline_days: int = 21
    baseline_min_windows: int = 10
    trend_days: int = 14
    r_warn: float = 0.15
    r_alarm: float = 0.25
    fan_warn_points: float = 15.0
    # The fan is compared at the same power: windows are grouped in bands this wide.
    fan_band_w: float = 25.0
    fan_std_factor: float = 3.0
    hotspot_delta_c: float = 20.0
    ecc_corrected_week: int = 10
    idle_rise_days: int = 30
    # "Rising": the straight line through those days climbs at least this much.
    idle_rise_c: float = 1.5
    # Immediate alarms.
    temp_gpu_c: float = 87.0
    temp_mem_c: float = 95.0
    temp_hold_s: float = 60.0
    throttle_hold_s: float = 30.0
    link_hold_s: float = 60.0
    link_load_w: float = 50.0
    replays_per_hour: int = 100
    nvml_silent_s: float = 300.0
    xid_critical: tuple[int, ...] = CRITICAL_XIDS
    # The helper on the Proxmox host (kernel log); empty path: no helper.
    helper_events_path: str = ""
    helper_heartbeat_s: float = 7200.0
    # The card's PCI address: AER lines are counted only when they name it.
    pci_address: str = "0000:01:00.0"
    # Repeats of an alarm whose condition persists.
    repeat_s: float = 21600.0
    critical_repeat_s: float = 900.0
    # Raw 15-second samples older than this are folded into 5-minute rows.
    raw_keep_days: int = 90
    # Where alarms go: "log" (the event list, the page and the service's log).
    channel: str = "log"

    @classmethod
    def from_mapping(cls, raw: Mapping | None) -> GpuHealthSettings:
        """Read and check the section; ValueError names what is wrong."""
        raw = {} if raw is None else raw
        if not isinstance(raw, Mapping):
            raise ValueError("gpu_health must be an object")
        known = {item.name for item in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError("Unknown gpu_health settings: " + ", ".join(sorted(unknown)))
        defaults = cls()
        values = {name: _checked(name, raw[name], getattr(defaults, name)) for name in raw}
        if values.get("channel", defaults.channel) not in CHANNELS:
            raise ValueError("gpu_health channel must be one of: " + ", ".join(CHANNELS))
        return cls(**values)

    def json(self) -> dict:
        """The settings as plain values, for the page."""
        values = asdict(self)
        values["xid_critical"] = list(self.xid_critical)
        return values


def _checked(name: str, value, default):
    """`value` of the same kind as the default; for numbers, not negative."""
    if isinstance(default, bool):
        return _flag(name, value)
    if isinstance(default, tuple):
        return _whole_numbers(name, value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ValueError(f"gpu_health {name} must be text")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"gpu_health {name} must be a number, not negative")
    return type(default)(value)


def _flag(name: str, value) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"gpu_health {name} must be true or false")
    return value


def _whole_numbers(name: str, value) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"gpu_health {name} must be a list of whole numbers")
    return tuple(value)
