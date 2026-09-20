"""Typed runtime policies read from the versioned JSON configuration.

The JSON remains a portable, editable mapping. These immutable values are the
boundary used by application services so defaults and limits have one owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .types import Task


def _number(raw: Mapping, name: str, default: float, *, minimum: float,
            maximum: float) -> float:
    value = raw.get(name, default)
    if type(value) not in (int, float) or not minimum <= value <= maximum:
        raise ValueError(f"Gateway {name} must be between {minimum:g} and {maximum:g}")
    return float(value)


def _integer(raw: Mapping, name: str, default: int, *, minimum: int,
             maximum: int, section: str) -> int:
    value = raw.get(name, default)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{section} {name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class TaskTimeout:
    first_byte_s: float
    between_bytes_s: float


@dataclass(frozen=True, slots=True)
class GatewayPolicy:
    first_byte_s: float = 120.0
    between_bytes_s: float = 30.0
    max_waiting: int = 150
    task_timeouts: dict[str, TaskTimeout] = field(default_factory=dict)
    max_upload_bytes: int = 0
    max_upload_pixels: int = 0
    max_upload_dimension: int = 0

    @classmethod
    def from_mapping(cls, raw: Mapping | None) -> GatewayPolicy:
        raw = {} if raw is None else raw
        if not isinstance(raw, Mapping):
            raise ValueError("Gateway settings must be an object")
        first = _number(raw, "first_byte_s", 120, minimum=1, maximum=86400)
        between = _number(raw, "between_bytes_s", 30, minimum=1, maximum=86400)
        timeouts = raw.get("task_timeouts", {})
        if not isinstance(timeouts, dict):
            raise ValueError("Gateway task_timeouts must be an object")
        tasks = {task.value for task in Task}
        by_task = {}
        for task, values in timeouts.items():
            if task not in tasks or not isinstance(values, dict):
                raise ValueError(f"Gateway task_timeouts has invalid task {task!r}")
            by_task[task] = TaskTimeout(
                first_byte_s=_number(values, "first_byte_s", first,
                                     minimum=1, maximum=86400),
                between_bytes_s=_number(values, "between_bytes_s", between,
                                        minimum=1, maximum=86400),
            )
        return cls(
            first_byte_s=first,
            between_bytes_s=between,
            max_waiting=_integer(raw, "max_waiting", 150, minimum=1,
                                 maximum=10000, section="Gateway"),
            task_timeouts=by_task,
            max_upload_bytes=_integer(raw, "max_upload_bytes", 0, minimum=0,
                                      maximum=1024 ** 3, section="Gateway"),
            max_upload_pixels=_integer(raw, "max_upload_pixels", 0, minimum=0,
                                       maximum=1024 ** 3, section="Gateway"),
            max_upload_dimension=_integer(raw, "max_upload_dimension", 0,
                                          minimum=0, maximum=100000,
                                          section="Gateway"),
        )

    def timeout_mapping(self) -> dict[str, dict[str, float]]:
        return {task: {"first_byte_s": value.first_byte_s,
                       "between_bytes_s": value.between_bytes_s}
                for task, value in self.task_timeouts.items()}


@dataclass(frozen=True, slots=True)
class MediaPolicy:
    result_ttl_s: int = 86400
    max_queue: int = 32
    max_input_bytes: int = 40 * 1024 * 1024
    max_result_bytes: int = 256 * 1024 * 1024

    @classmethod
    def from_mapping(cls, raw: Mapping | None) -> MediaPolicy:
        raw = {} if raw is None else raw
        if not isinstance(raw, Mapping):
            raise ValueError("Media settings must be an object")
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("Unknown media settings: " + ", ".join(sorted(unknown)))
        limits = {
            "result_ttl_s": (86400, 60, 30 * 86400),
            "max_queue": (32, 1, 1000),
            "max_input_bytes": (40 * 1024 * 1024, 1024, 512 * 1024 * 1024),
            "max_result_bytes": (256 * 1024 * 1024, 1024, 1024 * 1024 * 1024),
        }
        return cls(**{name: _integer(raw, name, default, minimum=minimum,
                                     maximum=maximum, section="Media")
                      for name, (default, minimum, maximum) in limits.items()})
