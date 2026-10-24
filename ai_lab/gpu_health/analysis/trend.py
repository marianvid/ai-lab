"""The slow picture: resting temperature, thermal resistance, its reference, its trend.

Thermal resistance R (°C per watt) says how well heat leaves the chip:

    R = (temperature under load − resting temperature that day) / power

It is computed for every stable load window. Dust in the cooler, dried
thermal paste or a tired fan make R rise slowly over months; the room's
temperature mostly cancels out, because the resting temperature is measured
in the same room on the same day.

- The reference (baseline) is the median R of the first 21 days, formed once
  there are at least 10 windows, and kept with its dates. It changes only when
  somebody resets it by hand, with a note.
- The trend is the median R of the last 14 days.
- The fan is compared at the same power: windows are grouped in 25 W bands.
"""

from __future__ import annotations

import datetime
import statistics

from ..alarms import Condition
from ..settings import GpuHealthSettings

# A straight line needs two points.
MIN_POINTS = 2


def shift(day: str, days: int) -> str:
    """`day` (YYYY-MM-DD) moved by `days`."""
    return (datetime.date.fromisoformat(day) + datetime.timedelta(days=days)).isoformat()


def idle_reference(idle: list[dict], day: str, fallback_days: int) -> tuple[float, str] | None:
    """The day's resting temperature and the day it comes from; None: no reference.

    The median of that day's idle windows; on a day without one, the latest
    day with one within `fallback_days` before it.
    """
    by_day: dict[str, list[float]] = {}
    for window in idle:
        by_day.setdefault(window["day"], []).append(window["temp_median"])
    for back in range(fallback_days + 1):
        wanted = shift(day, -back)
        if wanted in by_day:
            return statistics.median(by_day[wanted]), wanted
    return None


def with_r(windows: list[dict], idle: list[dict], settings: GpuHealthSettings) -> list[dict]:
    """The windows that have a resting temperature, each with its R."""
    found = []
    for window in windows:
        reference = idle_reference(idle, window["day"], settings.idle_fallback_days)
        if reference is None or window.get("temp_avg") is None or not window.get("power_avg"):
            continue
        found.append({**window, "idle": reference[0],
                      "r": (window["temp_avg"] - reference[0]) / window["power_avg"]})
    return found


def fan_bands(windows: list[dict], width_w: float) -> dict[str, float]:
    """The median fan per power band, keyed by the band's lower edge in watts."""
    bands: dict[str, list[float]] = {}
    for window in windows:
        if window.get("fan_avg") is not None:
            edge = int(window["power_avg"] // width_w * width_w)
            bands.setdefault(str(edge), []).append(window["fan_avg"])
    return {edge: statistics.median(values) for edge, values in bands.items()}


def summary(windows: list[dict], settings: GpuHealthSettings) -> dict:
    """Median R, fan spread and fan per band of a set of windows that have R."""
    spreads = [window["fan_std"] for window in windows if window.get("fan_std") is not None]
    return {"r": statistics.median(window["r"] for window in windows),
            "fan_std": statistics.median(spreads) if spreads else None,
            "fan_bands": fan_bands(windows, settings.fan_band_w),
            "windows": len(windows)}


def form_baseline(scope: str, windows: list[dict], today: str, now: float,
                  settings: GpuHealthSettings) -> dict | None:
    """The reference of one scope, once its first period is over; None: not yet.

    The period is the first `baseline_days` days from the scope's first
    window. If it ends with fewer than `baseline_min_windows` windows, it
    stretches until it has that many, so a card used rarely still gets one.
    """
    if not windows:
        return None
    first = windows[0]["day"]
    end = shift(first, settings.baseline_days - 1)
    if today <= end:
        return None
    chosen = [window for window in windows if window["day"] <= end]
    if len(chosen) < settings.baseline_min_windows:
        chosen = windows[:settings.baseline_min_windows]
        if len(chosen) < settings.baseline_min_windows:
            return None
        end = chosen[-1]["day"]
    return {"scope": scope, **summary(chosen, settings), "period_start": first,
            "period_end": end, "formed_at": now, "note": ""}


def compare(scope: str, base: dict, recent: list[dict],
            settings: GpuHealthSettings) -> list[Condition]:
    """The trend of one scope against its reference: R, fan at the same power, fan spread."""
    if not recent or not base.get("r"):
        return []
    now = summary(recent, settings)
    label = "" if scope == "global" else f" ({scope})"
    return [*_r_rise(scope, label, base, now, settings),
            *_fan_rise(scope, label, base, now, settings),
            *_fan_spread(scope, label, base, now, settings)]


def _r_rise(scope: str, label: str, base: dict, now: dict,
            settings: GpuHealthSettings) -> list[Condition]:
    rise = (now["r"] - base["r"]) / base["r"]
    for limit, level in ((settings.r_alarm, "alarm"), (settings.r_warn, "warning")):
        if rise >= limit:
            return [Condition(f"r_rise:{scope}", level,
                              f"Thermal resistance{label} {rise:+.0%} above its reference "
                              f"({now['r']:.3f} vs {base['r']:.3f} °C/W)",
                              {"rise": rise, "r": now["r"], "baseline": base["r"]})]
    return []


def _fan_rise(scope: str, label: str, base: dict, now: dict,
              settings: GpuHealthSettings) -> list[Condition]:
    found = []
    for edge, fan in now["fan_bands"].items():
        before = (base.get("fan_bands") or {}).get(edge)
        if before is not None and fan - before >= settings.fan_warn_points:
            found.append(Condition(f"fan_rise:{scope}:{edge}", "warning",
                                   f"Fan{label} at {edge}–{int(edge) + int(settings.fan_band_w)} W: "
                                   f"{fan:.0f} % vs {before:.0f} % in the reference",
                                   {"band_w": edge, "fan": fan, "baseline": before}))
    return found


def _fan_spread(scope: str, label: str, base: dict, now: dict,
                settings: GpuHealthSettings) -> list[Condition]:
    spread, before = now["fan_std"], base.get("fan_std")
    if spread is None or not before or spread <= settings.fan_std_factor * before:
        return []
    return [Condition(f"fan_unsteady:{scope}", "warning",
                      f"Fan{label} unsteady: spread {spread:.1f} vs {before:.1f} "
                      f"in the reference", {"fan_std": spread, "baseline": before})]


def hotspot(recent: list[dict], settings: GpuHealthSettings) -> list[Condition]:
    """Hotspot far above the chip's own temperature (paste or contact going bad)."""
    deltas = [window["hotspot_delta"] for window in recent if window.get("hotspot_delta") is not None]
    if not deltas or statistics.median(deltas) <= settings.hotspot_delta_c:
        return []
    value = statistics.median(deltas)
    return [Condition("hotspot_delta", "warning",
                      f"Hotspot {value:.0f} °C above the chip (more than "
                      f"{settings.hotspot_delta_c:.0f} °C)", {"delta": value})]


def corrected_ecc(readings: list[dict], settings: GpuHealthSettings) -> list[Condition]:
    """More than `ecc_corrected_week` corrected memory errors in the readings given (a week)."""
    counts = [reading["ecc"].get("corrected_aggregate") for reading in readings]
    counts = [count for count in counts if count is not None]
    if len(counts) < MIN_POINTS or counts[-1] - counts[0] <= settings.ecc_corrected_week:
        return []
    rise = counts[-1] - counts[0]
    return [Condition("ecc_corrected_week", "warning",
                      f"{rise} corrected memory errors in a week "
                      f"(more than {settings.ecc_corrected_week})", {"rise": rise})]


def idle_rise(daily: list[tuple[str, float]], settings: GpuHealthSettings) -> list[Condition]:
    """The resting temperature climbing over `idle_rise_days` (information only).

    `daily` is (day, resting temperature) for the period. The straight line
    through them (least squares) must climb at least `idle_rise_c` over the
    period, with at least two thirds of its days measured.
    """
    if len(daily) < max(MIN_POINTS, settings.idle_rise_days * 2 // 3):
        return []
    first = datetime.date.fromisoformat(daily[0][0])
    xs = [(datetime.date.fromisoformat(day) - first).days for day, _ in daily]
    ys = [value for _, value in daily]
    slope = statistics.linear_regression(xs, ys).slope
    climb = slope * settings.idle_rise_days
    if climb < settings.idle_rise_c:
        return []
    return [Condition("idle_rise", "info",
                      f"Resting temperature rising: about {climb:+.1f} °C over "
                      f"{settings.idle_rise_days} days (the room, or the cooling)",
                      {"climb": climb})]
