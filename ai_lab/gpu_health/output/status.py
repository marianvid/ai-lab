"""The monitor's state: in full for the page, in short for other programs; its
messages after a given one; and the history behind the chart."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from ..alarms import LEVELS
from ..readings import FAST, NOT_IN_NVML
from ..store import Store

CHART_POINTS = 360
MAX_ALERTS = 500
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


def summary(monitor: Any, now: float) -> dict:
    """The state in a few fields, for a program that asks every minute.

    `health` is the worst level among the alarms in force ("ok" when there is
    none; "info" does not count). A monitor whose thread stopped, or that has
    not read the card for longer than `nvml_silent_s`, makes it at least
    "alarm": then nobody is watching the card, which is itself the problem.
    """
    steps = monitor.steps
    alarms = [item for item in steps.book.active() if item["level"] != "info"]
    worst = max((LEVELS.index(item["level"]) for item in alarms), default=-1)
    age = None if steps.last_ok is None else now - steps.last_ok
    since = steps.last_ok if steps.last_ok is not None else steps.started
    stale = since is not None and now - since > steps.settings.nvml_silent_s
    if not monitor.running or stale:
        worst = max(worst, LEVELS.index("alarm"))
    return {
        "available": True,
        "health": "ok" if worst < 0 else LEVELS[worst],
        "monitor": "running" if monitor.running else "stopped",
        "last_reading": steps.last_ok,
        "reading_age_s": None if age is None else round(age, 1),
        "stale": stale,
        "alarms": [{key: item[key] for key in ("key", "level", "message", "since")}
                   for item in alarms],
        "failures": {name: item["count"] for name, item in monitor.failures.items()},
        "last_alert_id": steps.store.last_event_id(),
        "now": now,
    }


def alerts(store: Store, after: int, limit: int, everything: bool) -> dict:
    """The messages written after the one numbered `after`, oldest first.

    By default only what the channel sent (the same lines as the log, with the
    anti-spam applied); `everything` adds what was held back or only recorded.
    A client keeps `last_id` and asks again with `after=last_id`, so it never
    misses a message and never gets one twice.
    """
    if after < 0 or not 1 <= limit <= MAX_ALERTS:
        raise ValueError(f"after must be 0 or more, limit between 1 and {MAX_ALERTS}")
    found = store.events_after(after, limit, sent_only=not everything)
    return {"alerts": found, "last_id": found[-1]["id"] if found else after,
            "more": len(found) == limit}


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
