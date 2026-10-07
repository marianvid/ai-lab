"""One gate per vendor: how many calls at once, how far apart, and the pause.

The allowance belongs to the vendor, not to a model, so the gate does too:
Sonnet and Opus pass through the same one. Three rules:

- at most `concurrency` calls in flight;
- at least `launch_spacing_s` between two launches, because CLI start-ups that
  overlap tread on each other;
- after a rate limit, the whole vendor pauses — 30 s, then 120 s, then 300 s
  for every further one — so the first caller to meet the limit saves the
  others from spending calls on it. A success clears the count.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from .settings import VendorLimits


class VendorGate:
    """Admission for one vendor's calls (thread-safe)."""

    def __init__(self, limits: VendorLimits,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.limits = limits
        self._clock = clock
        self._sleep = sleep
        self._slots = threading.BoundedSemaphore(limits.concurrency)
        self._lock = threading.Lock()
        self._last_launch = float("-inf")
        self._paused_until = 0.0
        self._strikes = 0
        self._in_flight = 0

    @contextmanager
    def slot(self) -> Iterator[None]:
        """Hold one of the vendor's places for the length of a call."""
        with self._slots:
            self._wait_for_turn()
            with self._lock:
                self._in_flight += 1
            try:
                yield
            finally:
                with self._lock:
                    self._in_flight -= 1

    def _wait_for_turn(self) -> None:
        """Sleep through a pause and the launch spacing, then claim the launch."""
        while True:
            with self._lock:
                now = self._clock()
                ready_at = max(self._paused_until,
                               self._last_launch + self.limits.launch_spacing_s)
                if ready_at <= now:
                    self._last_launch = now
                    return
            self._sleep(ready_at - now)

    def rate_limited(self) -> float:
        """Pause the whole vendor after a rate limit; returns the pause in seconds."""
        waits = self.limits.rate_limit_waits_s
        with self._lock:
            pause = waits[min(self._strikes, len(waits) - 1)]
            self._strikes += 1
            self._paused_until = max(self._paused_until, self._clock() + pause)
        return pause

    def succeeded(self) -> None:
        """A call went through: the allowance is back, the count starts again."""
        with self._lock:
            self._strikes = 0

    def state(self) -> dict:
        """What the page shows: places in use and any pause still running."""
        with self._lock:
            return {"in_flight": self._in_flight,
                    "concurrency": self.limits.concurrency,
                    "paused_for_s": round(max(0.0, self._paused_until - self._clock()), 1),
                    "rate_limit_strikes": self._strikes}
