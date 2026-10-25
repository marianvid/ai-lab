"""When each client was last heard from, and the last map each scheduler sent.

Kept in memory only (A8): after a restart a scheduler's map is back with its
next heartbeat, within a minute. Only a client that has sent a heartbeat can
be shown as down; a client that only calls now and then is never "down".
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

DOWN_AFTER_S = 180.0


class Presence:
    """Last call, last heartbeat and last map, per client name."""

    def __init__(self, clock: Callable[[], float] = time.time,
                 down_after_s: float = DOWN_AFTER_S) -> None:
        self.clock = clock
        self.down_after_s = down_after_s
        self._lock = threading.Lock()
        self._last_call: dict[str, float] = {}
        self._beats: dict[str, tuple[float, dict]] = {}

    def seen(self, name: str) -> None:
        """A call from `name` arrived just now."""
        with self._lock:
            self._last_call[name] = self.clock()

    def heartbeat(self, name: str, reservation_map: object) -> float:
        """Keep `name`'s map; the time it was received is returned."""
        if not isinstance(reservation_map, dict):
            raise ValueError("The heartbeat needs a 'map' object")
        now = self.clock()
        with self._lock:
            self._last_call[name] = now
            self._beats[name] = (now, reservation_map)
        return now

    def view(self, names: list[str]) -> list[dict]:
        """One row per registered client, in the order given."""
        now = self.clock()
        with self._lock:
            return [self._row(name, now) for name in names]

    def _row(self, name: str, now: float) -> dict:
        beat = self._beats.get(name)
        row: dict[str, object] = {"name": name, "last_call_at": self._last_call.get(name),
                                  "heartbeat_at": None, "down": False, "map": None}
        if beat is not None:
            row["heartbeat_at"], row["map"] = beat
            row["down"] = now - beat[0] > self.down_after_s
        return row
