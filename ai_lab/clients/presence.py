"""When each client was last heard from: its last call and its last heartbeat.

Kept in memory only. A scheduler beats only while it runs something, to find
out early that AI-Lab is unusable; silence therefore means nothing, and no
client is ever marked "down". AI-Lab keeps no reservation map: it knows only
calls.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class Presence:
    """Last call and last heartbeat, per client name."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self._lock = threading.Lock()
        self._last_call: dict[str, float] = {}
        self._last_beat: dict[str, float] = {}

    def seen(self, name: str) -> None:
        """A call from `name` arrived just now."""
        with self._lock:
            self._last_call[name] = self.clock()

    def heartbeat(self, name: str) -> float:
        """`name` beat just now; the time it was received is returned."""
        now = self.clock()
        with self._lock:
            self._last_call[name] = now
            self._last_beat[name] = now
        return now

    def view(self, names: list[str]) -> list[dict]:
        """One row per registered client, in the order given."""
        with self._lock:
            return [{"name": name, "last_call_at": self._last_call.get(name),
                     "heartbeat_at": self._last_beat.get(name)} for name in names]
