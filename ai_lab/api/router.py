"""Matching a request to a handler.

Patterns use `{name}` for a segment, so `/api/instances/{id}/load` matches
`/api/instances/qwen/load` and hands over `{"id": "qwen"}`. Segments are URL
decoded, which matters because a model id contains slashes when it comes from
a nested directory — the browser sends it encoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from urllib.parse import unquote


@dataclass(frozen=True, slots=True)
class Route:
    method: str
    parts: tuple[str, ...]
    handler: Callable


class Router:
    def __init__(self) -> None:
        self._routes: list[Route] = []

    def add(self, method: str, pattern: str, handler: Callable) -> None:
        self._routes.append(Route(method.upper(), _split(pattern), handler))

    def match(self, method: str, path: str) -> tuple[Callable, dict] | None:
        parts = _split(path)
        for route in self._routes:
            if route.method != method.upper() or len(route.parts) != len(parts):
                continue
            captured = _capture(route.parts, parts)
            if captured is not None:
                return route.handler, captured
        return None


def _split(path: str) -> tuple[str, ...]:
    return tuple(item for item in path.strip("/").split("/") if item)


def _capture(pattern: tuple[str, ...], parts: tuple[str, ...]) -> dict | None:
    captured: dict[str, str] = {}
    for expected, actual in zip(pattern, parts):
        if expected.startswith("{") and expected.endswith("}"):
            captured[expected[1:-1]] = unquote(actual)
        elif expected != actual:
            return None
    return captured
