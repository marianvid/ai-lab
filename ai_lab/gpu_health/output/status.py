"""The monitor's state as one plain object, and the history behind the chart."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from ..readings import FAST, NOT_IN_NVML
from ..store import Store

CHART_POINTS = 360
CHART_FIELDS = (("temp_gpu", "temp_avg"), ("power_w", "power_avg"),
                ("fan_pct", "fan_avg"), ("util_pct", "util_avg"))
COUNTED = ("samples", "samples_5m", "windows", "idle_windows", "events", "baselines")


def status(monitor: Any, now: float, path: Path | None) -> dict:
    """Everything the page, the endpoint and the command show."""
    steps = monitor.steps
    store = steps.store
    return {
        "available": True,
        "running": monitor.running,
        "now": now,
        "started": steps.started,
        "last_ok": steps.last_ok,
        "failures": monitor.failures,
        "not_available": sorted(steps.missing | set(NOT_IN_NVML)),
        "readings": [name for name, _ in FAST],
        "current": store.last_sample(),
        "slow": store.state("slow_last"),
        "helper": {"installed": steps.kernel.installed,
                   "path": steps.settings.helper_events_path,
                   "last_heartbeat": steps.kernel.position.get("heartbeat")},
        "alarms": steps.book.active(),
        "analysis": steps.analysis.overview(now),
        "events": store.events(limit=40),
        "counts": {table: store.count(table) for table in COUNTED},
        "baseline_reset": store.state("baseline_reset"),
        "settings": steps.settings.json(),
        "database": str(path) if path else "",
    }


def history(store: Store, now: float, hours: float) -> dict:
    """Up to 360 points over the last `hours`: each the mean of its stretch of readings."""
    if hours <= 0:
        raise ValueError("hours must be more than 0")
    since = now - hours * 3600.0
    rows = [{"ts": row["ts"], **{name: row[old] for name, old in CHART_FIELDS}}
            for row in store.samples_5m_since(since)]
    rows += [{"ts": row["ts"], **{name: row[name] for name, _ in CHART_FIELDS}}
             for row in store.samples_since(since)]
    step = hours * 3600.0 / CHART_POINTS
    buckets: dict[int, list[dict]] = {}
    for row in rows:
        buckets.setdefault(int((row["ts"] - since) // step), []).append(row)
    points = [_point(buckets[index]) for index in sorted(buckets)]
    return {"hours": hours, "from": since, "to": now, "points": points}


def _point(group: list[dict]) -> dict:
    """The mean of a stretch of readings; a value never read stays None."""
    point: dict = {"ts": statistics.fmean(row["ts"] for row in group)}
    for name, _ in CHART_FIELDS:
        values = [row[name] for row in group if row[name] is not None]
        point[name] = statistics.fmean(values) if values else None
    return point
