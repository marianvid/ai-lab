"""The GPU health routes: the page's status, chart, report and two buttons, and
the short state and the messages for other programs (docs/gpu-health.md)."""

from __future__ import annotations

from typing import Any


def _number(query: dict | None, name: str, default: float) -> float:
    value = (query or {}).get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None


def _flag(query: dict | None, name: str) -> bool:
    return str((query or {}).get(name, "")).lower() in {"1", "true", "yes"}


def register(router, health: Any) -> None:
    """The routes, each one call into `GpuHealth`."""
    router.add("GET", "/api/gpu-health", lambda **_: health.status())
    # For other programs (a relay to Telegram, a dashboard): the state in
    # short, and the messages after the last one they saw.
    router.add("GET", "/api/gpu-health/summary", lambda **_: health.summary())
    router.add("GET", "/api/gpu-health/alerts",
               lambda query=None, **_: health.alerts(int(_number(query, "after", 0)),
                                                     int(_number(query, "limit", 100)),
                                                     _flag(query, "all")))
    router.add("GET", "/api/gpu-health/history",
               lambda query=None, **_: health.history(_number(query, "hours", 24)))
    router.add("GET", "/api/gpu-health/report",
               lambda query=None, **_: health.report(int(_number(query, "days", 7))))
    router.add("POST", "/api/gpu-health/baseline/reset",
               lambda body=None, **_: health.reset_baseline(str((body or {}).get("note", ""))))
    router.add("POST", "/api/gpu-health/test-alarm", lambda **_: health.test_alarm())
