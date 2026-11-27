"""Every setting kept in the configuration file, as a field on the Settings page.

The owner's rule (2026-11-27): no setting lives only in a file. Each one here
carries what the page needs to show it and explain it — its label, unit,
default, the range it may take, a plain sentence on what it means and what too
small or too large does, and whether it takes effect only after a restart —
and the place it is kept (`key`, dotted: `limits.runtime.load_s` is
`config.json`'s `limits` → `runtime` → `load_s`).

A change is written into a copy of the configuration and checked by the parsers
that own its sections (`config.policy` for `limits` and `gateway`,
`gpu_health.settings` for `gpu_health`) before it is saved, so the page and the
start of AI-Lab refuse the same values with the same words.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any, cast

from ..config import policy
from ..config.store import Config, ConfigStore
from ..config.validation import editor_port_problems
from ..gpu_health.settings import ORDERED as MONITOR_ORDER
from ..gpu_health.settings import GpuHealthSettings
from .field_list import BY_KEY, FIELDS, FIXED, GROUPS, RESTART, Field

__all__ = ["BY_KEY", "FIELDS", "GROUPS", "Field", "SettingFields"]


def _read(config: Config, key: str):
    """The value saved at `key`, or None when the file does not set it.

    A key without a dot is the top of the file (`title`): it always has a value.
    """
    section, *path = key.split(".")
    value = getattr(config, section)
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _write(config: Config, key: str, value) -> None:
    """Set `key` in the configuration, creating the sections on the way."""
    section, *path = key.split(".")
    if not path:
        setattr(config, section, value)
        return
    holder = getattr(config, section)
    for part in path[:-1]:
        holder = holder.setdefault(part, {})
    holder[path[-1]] = value


def _text(field: Field, value):
    if not isinstance(value, str) or not (value.strip() or field.empty):
        raise ValueError(f"{field.label} must be text")
    return value.strip()


def _on_off(field: Field, value):
    if not isinstance(value, bool):
        raise ValueError(f"{field.label} must be on or off")
    return value


def _whole_numbers(field: Field, value):
    if not isinstance(value, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise ValueError(f"{field.label} must be whole numbers")
    return value


def _fixed(field: Field, _value):
    raise ValueError(FIXED.format(label=field.label))


def _checked(field: Field, value):
    """The value as the field takes it; ValueError naming the field otherwise."""
    other = {"text": _text, "boolean": _on_off, "list": _whole_numbers,
             "fixed": _fixed}.get(field.kind)
    if other is not None:
        return other(field, value)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field.label} must be a number")
    if field.kind == "integer":
        if value != int(value):
            raise ValueError(f"{field.label} must be a whole number")
        value = int(value)
    return _within(field, value)


def _within(field: Field, value: float) -> float:
    """`value`, refused outside the field's own bounds, naming them."""
    if field.low is not None and value < field.low:
        raise ValueError(f"{field.label} must be at least {field.low:g}")
    if field.high is not None and value > field.high:
        raise ValueError(f"{field.label} must be at most {field.high:g}")
    return value


# Pairs whose first value must stay under the second (`True`: strictly), checked on
# the whole saved section, so two separate saves cannot put them out of order.
# The monitor's own pairs (`gpu_health.settings.ORDERED`), named by the page's keys.
ORDERED = tuple((f"gpu_health.{first}", f"gpu_health.{second}", strict)
                for first, second, strict in MONITOR_ORDER)


def _number_now(config: Config, field: Field) -> float:
    """A number field's saved value, or its default when the file sets none."""
    saved = _read(config, field.key)
    return cast(float, field.default if saved is None else saved)


def _check_order(config: Config) -> None:
    """ValueError naming both fields when a pair of `ORDERED` is out of order."""
    for first_key, second_key, strict in ORDERED:
        first, second = BY_KEY[first_key], BY_KEY[second_key]
        low, high = _number_now(config, first), _number_now(config, second)
        if low > high or (strict and low == high):
            word = "below" if strict else "at most"
            raise ValueError(f"{first.label} ({low:g}) must be {word} {second.label} ({high:g})")


class SettingFields:
    """The fields, with their saved values, and the change of some of them."""

    def __init__(self, store: ConfigStore, changed: Callable[[str], None],
                 pages: tuple[frozenset[str], Any] = (frozenset(), None)) -> None:
        """`pages`: the engines that serve a page of their own, and the other
        packages' sections (`wiring.CONFIG_SECTIONS`) whose ports those pages
        may not take, for a new `limits.editor_port_offset`."""
        self.store = store
        self._changed = changed
        self._page_engines, self._sections = pages

    def view(self) -> dict:
        """Every field with its value now (its default when the file sets none), by group."""
        config = self.store.load()
        fields = []
        for field in FIELDS:
            saved = _read(config, field.key)
            fields.append({**asdict(field), "value": field.default if saved is None else saved,
                           "saved": saved is not None,
                           "restart_note": RESTART if field.restart else ""})
        return {"groups": GROUPS, "fields": fields}

    def update(self, changes: dict) -> dict:
        """Save some fields, each checked by the parser that owns its section."""
        if not isinstance(changes, dict) or not changes:
            raise ValueError("Send the fields to change, by key")
        unknown = sorted(set(changes) - set(BY_KEY))
        if unknown:
            raise ValueError("Unknown settings: " + ", ".join(unknown))
        with self.store.mutate() as config:
            for key, value in changes.items():
                _write(config, key, _checked(BY_KEY[key], value))
            # The sections' own parsers, as at start: ValueError leaves the file as it was.
            policy.MachineLimits.from_mapping(config.limits)
            policy.GatewayPolicy.from_mapping(config.gateway)
            # The pairs first: the page names them by their labels.
            _check_order(config)
            GpuHealthSettings.from_mapping(config.gpu_health)
            problems = editor_port_problems(config, self._page_engines, self._sections)
            if problems:
                raise ValueError("; ".join(problems))
        self._changed("settings")
        return self.view()
