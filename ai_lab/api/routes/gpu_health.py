"""The GPU health page's routes: the status, the chart's history, the report, two buttons."""

from __future__ import annotations

from typing import Any


def _number(query: dict | None, name: str, default: float) -> float:
    value = (query or {}).get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None


def register(router, health: Any) -> None:
    """The routes, each one call into `GpuHealth`."""
    router.add("GET", "/api/gpu-health", lambda **_: health.status())
    router.add("GET", "/api/gpu-health/history",
               lambda query=None, **_: health.history(_number(query, "hours", 24)))
    router.add("GET", "/api/gpu-health/report",
               lambda query=None, **_: health.report(int(_number(query, "days", 7))))
    router.add("POST", "/api/gpu-health/baseline/reset",
               lambda body=None, **_: health.reset_baseline(str((body or {}).get("note", ""))))
    router.add("POST", "/api/gpu-health/test-alarm", lambda **_: health.test_alarm())
