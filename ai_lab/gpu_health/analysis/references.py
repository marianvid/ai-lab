"""The slow analysis, run every few minutes: references, trend, and their alarms.

It reads what the store has (windows, idle windows, hourly readings), forms a
reference for any scope whose first period has ended, and compares the last
14 days with it. Scopes: "global" (every window), and each model that has
windows of its own.
"""

from __future__ import annotations

import statistics
import threading

from . import trend
from ..alarms import Condition
from ..settings import GpuHealthSettings
from ..store import Store
from .windows import day_of

WEEK = 7 * 86400.0


def scopes(windows: list[dict]) -> dict[str, list[dict]]:
    """The windows of each scope: "global" (all of them), and each model by name."""
    found = {"global": windows}
    for window in windows:
        if window["model"]:
            found.setdefault(window["model"], []).append(window)
    return found


class Analysis:
    """Forms the references and finds the trend alarms."""

    def __init__(self, store: Store, settings: GpuHealthSettings) -> None:
        self._store = store
        self._settings = settings
        # The monitor's thread forms references; the page may reset them.
        self._lock = threading.Lock()

    def run(self, now: float) -> list[Condition]:
        """Every trend condition true now (references formed on the way)."""
        with self._lock:
            return self._run(now)

    def _run(self, now: float) -> list[Condition]:
        today = day_of(now)
        idle = self._store.idle_windows(trend.shift(today, -self._history_days()))
        recent_since = trend.shift(today, -(self._settings.trend_days - 1))
        recent = trend.with_r(self._store.windows(recent_since), idle, self._settings)
        found = self._compare_scopes(recent, today, now)
        found += trend.hotspot([w for w in recent if w["day"] == today], self._settings)
        found += trend.corrected_ecc(self._store.slow_readings(now - WEEK), self._settings)
        found += trend.idle_rise(self.daily_idle(today, idle), self._settings)
        return found

    def _compare_scopes(self, recent: list[dict], today: str, now: float) -> list[Condition]:
        """Each scope with a reference and enough recent windows, against its reference."""
        found: list[Condition] = []
        for scope, mine in scopes(recent).items():
            base = self._baseline(scope, today, now)
            if base is not None and len(mine) >= self._settings.baseline_min_windows:
                found += trend.compare(scope, base, mine, self._settings)
        return found

    def _history_days(self) -> int:
        return max(self._settings.idle_rise_days, self._settings.trend_days) \
            + self._settings.idle_fallback_days

    def _baseline(self, scope: str, today: str, now: float) -> dict | None:
        """The scope's reference; formed now if its first period just ended."""
        known = self._store.baselines().get(scope)
        if known is not None:
            return known
        # After a reset by hand, only what came after it counts.
        reset = self._store.state("baseline_reset") or {}
        windows = self._store.windows(reset.get("day", ""), None if scope == "global" else scope)
        idle = self._store.idle_windows(
            trend.shift(windows[0]["day"], -self._settings.idle_fallback_days)) if windows else []
        formed = trend.form_baseline(scope, trend.with_r(windows, idle, self._settings),
                                     today, now, self._settings)
        if formed is not None:
            formed["note"] = f"after a reset: {reset['note']}" if reset else ""
            self._store.save_baseline(formed)
            self._store.add_event({"ts": now, "kind": "baseline", "level": "info", "key": scope,
                                   "message": f"Reference formed for {scope}: R "
                                              f"{formed['r']:.3f} °C/W from {formed['windows']} "
                                              f"windows ({formed['period_start']} – "
                                              f"{formed['period_end']})", "detail": formed})
        return formed

    def daily_idle(self, today: str, idle: list[dict]) -> list[tuple[str, float]]:
        """(day, resting temperature) of each measured day of the last `idle_rise_days`."""
        by_day: dict[str, list[float]] = {}
        start = trend.shift(today, -(self._settings.idle_rise_days - 1))
        for window in idle:
            if window["day"] >= start:
                by_day.setdefault(window["day"], []).append(window["temp_median"])
        return [(day, statistics.median(values)) for day, values in sorted(by_day.items())]

    def overview(self, now: float) -> dict:
        """References, the last 14 days and today's resting temperature, for the page."""
        today = day_of(now)
        idle = self._store.idle_windows(trend.shift(today, -self._history_days()))
        reference = trend.idle_reference(idle, today, self._settings.idle_fallback_days)
        recent_since = trend.shift(today, -(self._settings.trend_days - 1))
        recent = trend.with_r(self._store.windows(recent_since), idle, self._settings)
        summaries = {scope: trend.summary(mine, self._settings) if mine else None
                     for scope, mine in scopes(recent).items()}
        return {"idle_today": None if reference is None else
                {"temp": reference[0], "from_day": reference[1]},
                "baselines": self._store.baselines(), "trend": summaries,
                "daily_idle": self.daily_idle(today, idle)}

    def reset_baselines(self, now: float, note: str) -> None:
        """Forget the references by hand; they form again from the windows that follow."""
        if not note.strip():
            raise ValueError("A reset needs a note saying why")
        with self._lock:
            self._reset(now, note)

    def _reset(self, now: float, note: str) -> None:
        old = self._store.baselines()
        self._store.clear_baselines()
        self._store.set_state("baseline_reset", {"ts": now, "note": note, "day": day_of(now)})
        self._store.add_event({"ts": now, "kind": "baseline", "level": "info", "key": "reset",
                               "message": f"References reset by hand: {note}",
                               "detail": {"old": old}})
