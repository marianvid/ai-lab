"""Which models were taken off the card for another, and when (the approved list, A5, A7).

A call whose model is taken off while it runs ends with `409 model_replaced`
and the name of the model that took its place, instead of whatever a stopped
engine leaves on the connection. The gateway marks a model here when it takes
it off, or when Load or Unload is forced by hand; the front door asks.

Each switch is also one line in the operation log (A9), through `note`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


def _silent(_line: str) -> None:
    """The operation log before the application connects one."""


class Replacements:
    """The last time each model was taken off the card, and for what."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self.note: Callable[[str], None] = _silent
        self._lock = threading.Lock()
        self._taken_off: dict[str, tuple[float, str]] = {}

    def switched(self, wanted: str, victims: list[str]) -> None:
        """`wanted` goes on the card; `victims` come off for it."""
        gone = ", ".join(victims) or "nothing"
        self.note(f"gateway: putting {wanted} on the card, taking off {gone}")
        self.mark(victims, wanted)

    def stopped(self, reason: str, loaded: list[dict]) -> None:
        """Everything was stopped (a forced Load or Unload, a restart): its calls were cut."""
        self.note(f"gateway: everything stopped and the queue turned away: {reason}")
        self.mark([item["shape"].instance_id for item in loaded], reason)

    def mark(self, victims: list[str], by: str) -> None:
        """These models were taken off the card, now, for `by`."""
        now = self.clock()
        with self._lock:
            for victim in victims:
                self._taken_off[victim] = (now, by)

    def since(self, model: str, started: float) -> str | None:
        """What took `model`'s place after `started`, or None if it was not taken off."""
        with self._lock:
            taken = self._taken_off.get(model)
        if taken is None or taken[0] < started:
            return None
        return taken[1]
