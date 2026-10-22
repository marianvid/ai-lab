"""The calls without a reservation that wait at the front door while the card is held.

They are not reservations, so the queue does not list them, and they have not
reached the card's own queue either, so the gateway's `waiting` does not count
them. Without this record a page showed "Nobody is waiting" while eleven calls
were waiting (news' benchmark, 22 October 2026). This keeps how many wait, for
which model, and since when.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import Counter
from collections.abc import Callable

Clock = Callable[[], float]


class DoorWaiters:
    """Who waits at the door now: one entry per call, removed when it goes in or gives up."""

    def __init__(self, clock: Clock = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._numbers = itertools.count(1)
        self._waiting: dict[int, tuple[str, float]] = {}

    def arrive(self, model: str) -> int:
        """A call starts waiting; the number it gets is what `leave` takes."""
        with self._lock:
            number = next(self._numbers)
            self._waiting[number] = (model or "", self._clock())
            return number

    def leave(self, number: int) -> None:
        """The call went in, was refused, or its client left."""
        with self._lock:
            self._waiting.pop(number, None)

    def view(self) -> dict:
        """How many wait, per model, and when the oldest began (None when nobody waits)."""
        with self._lock:
            entries = list(self._waiting.values())
        return {"calls": len(entries),
                "models": dict(Counter(model for model, _ in entries)),
                "oldest_since": min((since for _, since in entries), default=None)}
