"""The immediate checks: each 15-second reading, and each hourly one.

A rule with a duration ("above 87 °C for more than a minute") becomes a
condition only once it has held that long; a single hot reading is not an
alarm. Power capping (the card holding itself at its power limit) is normal
work, not an alarm, and is never reported.
"""

from __future__ import annotations

from ..alarms import Condition
from ..nvml import throttle_names
from ..settings import GpuHealthSettings

# Thermal slowdown and hardware slowdown (including the power brake): the
# card slowed itself because something is wrong, not because it was asked to.
BAD_THROTTLE = 0x8 | 0x20 | 0x40 | 0x80
HOUR = 3600.0
# A limit read as 199.8 W one time and 200 W the next has not changed.
LIMIT_CHANGE_W = 0.5


class Holds:
    """How long each rule has been true, to require it for a while before alarming."""

    def __init__(self) -> None:
        self._since: dict[str, float] = {}

    def held(self, key: str, true: bool, now: float, hold_s: float) -> bool:
        """Whether `key` has been true for more than `hold_s` (forgotten when false)."""
        if not true:
            self._since.pop(key, None)
            return False
        since = self._since.setdefault(key, now)
        return now - since > hold_s


class FastChecks:
    """The rules on every reading. `state` survives restarts (best link, replays, limit)."""

    def __init__(self, settings: GpuHealthSettings, state: dict | None = None) -> None:
        self._settings = settings
        self._holds = Holds()
        self.state = {"best_link": None, "replays": [], "power_limit": None, **(state or {})}

    def check(self, sample: dict) -> tuple[list[Condition], list[Condition]]:
        """(conditions true now, events that happened) for one reading."""
        now = sample["ts"]
        conditions = [*self._temperatures(sample, now), *self._throttle(sample, now),
                      *self._link(sample, now), *self._replays(sample, now)]
        return conditions, self._power_limit(sample)

    def _temperatures(self, sample: dict, now: float) -> list[Condition]:
        found = []
        for key, name, limit in (("temp_gpu", "Chip", self._settings.temp_gpu_c),
                                 ("temp_mem", "Memory", self._settings.temp_mem_c)):
            value = sample.get(key)
            hot = value is not None and value > limit
            if self._holds.held(key, hot, now, self._settings.temp_hold_s):
                found.append(Condition(key, "alarm",
                                       f"{name} at {value:.0f} °C, above {limit:.0f} °C "
                                       f"for more than {self._settings.temp_hold_s:.0f} s",
                                       {"value": value, "limit": limit}))
        return found

    def _throttle(self, sample: dict, now: float) -> list[Condition]:
        mask = sample.get("throttle") or 0
        bad = mask & BAD_THROTTLE
        if not self._holds.held("throttle", bool(bad), now, self._settings.throttle_hold_s):
            return []
        names = ", ".join(throttle_names(bad))
        return [Condition("throttle", "alarm",
                          f"The card slows itself down ({names}) for more than "
                          f"{self._settings.throttle_hold_s:.0f} s", {"mask": mask})]

    def _link(self, sample: dict, now: float) -> list[Condition]:
        """PCIe slower than the best seen under load, for more than a minute.

        Only under load: at rest the card drops its own link to Gen 1 to save
        power, which is normal. "Normal" is the best generation and width seen
        under load since the monitor started (x4 on this OcuLink cable).
        """
        gen, width = sample.get("pcie_gen"), sample.get("pcie_width")
        power, use = sample.get("power_w") or 0.0, sample.get("util_pct") or 0.0
        if gen is None or width is None or power <= self._settings.link_load_w or use <= 0:
            self._holds.held("pcie_link", False, now, 0)
            return []
        best = self.state["best_link"] or [gen, width]
        best = [max(best[0], gen), max(best[1], width)]
        self.state["best_link"] = best
        slow = gen < best[0] or width < best[1]
        if not self._holds.held("pcie_link", slow, now, self._settings.link_hold_s):
            return []
        return [Condition("pcie_link", "alarm",
                          f"PCIe link at Gen {gen} x{width} under load; normal is "
                          f"Gen {best[0]} x{best[1]}", {"gen": gen, "width": width,
                                                        "normal": best})]

    def _replays(self, sample: dict, now: float) -> list[Condition]:
        """More than `replays_per_hour` PCIe replays (resent packets) within the last hour."""
        count = sample.get("pcie_replays")
        if count is None:
            return []
        history = [item for item in self.state["replays"] if now - item[0] <= HOUR]
        history.append([now, count])
        self.state["replays"] = history
        rise = count - min(item[1] for item in history)
        if rise <= self._settings.replays_per_hour:
            return []
        return [Condition("pcie_replays", "warning",
                          f"{rise} PCIe replays in the last hour "
                          f"(more than {self._settings.replays_per_hour})", {"rise": rise})]

    def _power_limit(self, sample: dict) -> list[Condition]:
        limit, before = sample.get("power_limit_w"), self.state["power_limit"]
        if limit is None:
            return []
        self.state["power_limit"] = limit
        if before is None or abs(limit - before) < LIMIT_CHANGE_W:
            return []
        return [Condition("power_limit", "warning",
                          f"The power limit changed from {before:.0f} W to {limit:.0f} W",
                          {"before": before, "after": limit})]


def slow_events(reading: dict, before: dict | None) -> list[Condition]:
    """What changed since the previous hourly reading: new memory errors, new replaced rows."""
    if before is None:
        return []
    found = _new_errors(reading, before) + _new_rows(reading, before)
    if before.get("driver") and reading["driver"] != before["driver"]:
        found.append(Condition("driver", "info",
                               f"Driver changed from {before['driver']} to {reading['driver']}"))
    return found


def _new_errors(reading: dict, before: dict) -> list[Condition]:
    found = []
    ecc, old = reading["ecc"], before.get("ecc", {})
    for name in ("uncorrected_volatile", "uncorrected_aggregate"):
        if name in ecc and ecc[name] > old.get(name, ecc[name]):
            found.append(Condition("ecc_uncorrected", "critical",
                                   f"New uncorrectable memory error ({name}: "
                                   f"{old.get(name)} → {ecc[name]})", {"counter": name}))
    return found


def _new_rows(reading: dict, before: dict) -> list[Condition]:
    found = []
    rows, old_rows = reading.get("rows") or {}, before.get("rows") or {}
    for name, level in (("uncorrectable", "critical"), ("correctable", "alarm")):
        if rows.get(name, 0) > old_rows.get(name, rows.get(name, 0)):
            found.append(Condition(f"remap_{name}", level,
                                   f"The card replaced a memory row ({name}: "
                                   f"{old_rows.get(name)} → {rows[name]})", {"rows": rows}))
    return found


def slow_conditions(reading: dict) -> list[Condition]:
    """A memory row waiting to be replaced (pending), or a replacement that failed."""
    rows = reading.get("rows") or {}
    found = []
    if rows.get("pending"):
        found.append(Condition("remap_pending", "alarm",
                               "A memory row is waiting to be replaced (needs a card reset)",
                               {"rows": rows}))
    if rows.get("failure"):
        found.append(Condition("remap_failure", "critical",
                               "Replacing a memory row failed: the card is out of spare rows",
                               {"rows": rows}))
    return found
