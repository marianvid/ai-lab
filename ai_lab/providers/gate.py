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

from .errors import ProviderCallCancelledError
from .settings import VendorLimits


def _check(still_wanted: Callable[[], bool] | None) -> None:
    """Leave the queue when the client has gone."""
    if still_wanted is not None and not still_wanted():
        raise ProviderCallCancelledError("the client left while its call waited")


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
    def slot(self, still_wanted: Callable[[], bool] | None = None) -> Iterator[None]:
        """Hold one of the vendor's places for the length of a call.

        A call that waits (for a place, a pause or the launch spacing) looks
        every `client_check_s` whether it is still wanted; one that is not
        leaves the queue without starting (`ProviderCallCancelledError`).
        """
        self._take_place(still_wanted)
        try:
            self._wait_for_turn(still_wanted)
            with self._lock:
                self._in_flight += 1
            try:
                yield
            finally:
                with self._lock:
                    self._in_flight -= 1
        finally:
            self._slots.release()

    def _take_place(self, still_wanted: Callable[[], bool] | None) -> None:
        if still_wanted is None:
            self._slots.acquire()
            return
        while not self._slots.acquire(timeout=self.limits.client_check_s):
            _check(still_wanted)

    def _wait_for_turn(self, still_wanted: Callable[[], bool] | None = None) -> None:
        """Sleep through a pause and the launch spacing, then claim the launch."""
        while True:
            _check(still_wanted)
            with self._lock:
                now = self._clock()
                ready_at = max(self._paused_until,
                               self._last_launch + self.limits.launch_spacing_s)
                if ready_at <= now:
                    self._last_launch = now
                    return
            wait_s = ready_at - now
            self._sleep(wait_s if still_wanted is None
                        else min(wait_s, self.limits.client_check_s))

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
