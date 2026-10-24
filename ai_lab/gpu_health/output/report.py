"""The weekly report, and the status everybody reads (the page, the endpoint, the command).

The report sums up a period (a week by default): temperatures and power, how
many stable and idle windows there were, R against its reference, today's
resting temperature, and the alarms. It is written as plain sentences as well
as numbers, so it can be read in the service's log.
"""

from __future__ import annotations

import statistics

from ..analysis.references import Analysis
from ..store import Store
from ..analysis.windows import day_of

DAY = 86400.0


def weekly(store: Store, analysis: Analysis, now: float, days: int = 7) -> dict:
    """The report of the last `days` days."""
    since = now - days * DAY
    samples = store.samples_since(since) or store.samples_5m_since(since)
    alarms = [event for event in store.events(limit=1000, since=since)
              if event["kind"] in {"alarm", "event"} and event["level"] != "info"]
    report = {
        "days": days, "from": since, "to": now, "samples": len(samples),
        "temp": _spread(_values(samples, "temp_gpu", "temp_avg")),
        "power": _spread(_values(samples, "power_w", "power_avg")),
        "windows": _since(store.windows(day_of(since)), since),
        "idle_windows": _since(store.idle_windows(day_of(since)), since),
        "alarms": len(alarms),
        "critical": len([event for event in alarms if event["level"] == "critical"]),
        "alarm_list": [{"ts": event["ts"], "level": event["level"], "message": event["message"]}
                       for event in alarms[:20]],
        **analysis.overview(now),
    }
    report["text"] = text(report)
    return report


def _values(rows: list[dict], raw: str, folded: str) -> list[float]:
    """One value from each row: the raw reading's name, or the folded row's."""
    found = [row.get(raw, row.get(folded)) for row in rows]
    return [value for value in found if value is not None]


def _since(windows: list[dict], since: float) -> int:
    return len([window for window in windows if window["start"] >= since])


def _spread(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"min": min(values), "max": max(values), "mean": statistics.fmean(values)}


def text(report: dict) -> str:
    """The report as a few plain sentences."""
    lines = [f"GPU health, last {report['days']} days: {report['samples']} readings, "
             f"{report['windows']} stable load windows, {report['idle_windows']} idle windows."]
    if report["temp"]:
        lines.append(f"Chip {report['temp']['min']:.0f}–{report['temp']['max']:.0f} °C "
                     f"(mean {report['temp']['mean']:.0f}); power mean "
                     f"{report['power']['mean']:.0f} W, peak {report['power']['max']:.0f} W.")
    idle = report.get("idle_today")
    lines.append("Resting temperature today: " + (
        f"{idle['temp']:.0f} °C (from {idle['from_day']})." if idle else "no reference."))
    base, recent = report["baselines"].get("global"), report["trend"].get("global")
    if base and recent:
        rise = (recent["r"] - base["r"]) / base["r"]
        lines.append(f"Thermal resistance {recent['r']:.3f} °C/W, {rise:+.0%} against the "
                     f"reference {base['r']:.3f} ({base['period_start']} – {base['period_end']}).")
    elif recent:
        lines.append(f"Thermal resistance {recent['r']:.3f} °C/W; no reference yet "
                     "(it forms after the first 21 days).")
    else:
        lines.append("No thermal resistance: no stable load window with a resting temperature.")
    lines.append(f"Alarms: {report['alarms']} ({report['critical']} critical)."
                 if report["alarms"] else "No alarms.")
    return " ".join(lines)
